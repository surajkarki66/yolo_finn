import copy
import math

import torch
from torch import nn
from torch.nn import Sequential

from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU, QuantIdentity
from brevitas.quant import IntBias

from brevitas_examples.imagenet_classification.models.common import CommonIntActQuant, CommonUintActQuant
from brevitas_examples.imagenet_classification.models.common import CommonIntWeightPerChannelQuant

from utils.tal import dist2bbox, make_anchors
from utils.loss import v8DetectionLoss
from utils.general import V8_non_max_suppression
from models.common import DFL


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class ConvBNReLU(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=None,
                 groups=1,
                 norm_layer=None):
        self.padding = (kernel_size - 1) // 2 if padding == None else padding
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, self.padding, groups=groups, bias=False)
        self.bn = norm_layer(out_channels)
        self.activation = nn.ReLU6(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x
    

class DwsConv(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            stride=1,
            padding=None):
        super(DwsConv, self).__init__()
        self.dw = ConvBNReLU(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            groups=in_channels
        )
        self.pw = ConvBNReLU(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=None
        )

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MobileNetV2(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None,
                 norm_layer=None):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)
    


# -----------------QUANTIZED -----------

class QuantConvBNReLU(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=None,
            weight_bit_width=8,
            act_bit_width=8,
            groups=1,
            bn_eps=1e-5,
            activation_scaling_per_channel=True):
        super(QuantConvBNReLU, self).__init__()
        self.padding = (kernel_size - 1) // 2 if padding == None else padding
        self.conv = QuantConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            groups=groups,
            bias=False,
            weight_quant=CommonIntWeightPerChannelQuant,
            weight_bit_width=weight_bit_width)
        self.bn = nn.BatchNorm2d(num_features=out_channels, eps=bn_eps)
        self.activation = QuantReLU(
            act_quant=CommonUintActQuant,
            bit_width=act_bit_width,
            per_channel_broadcastable_shape=(1, out_channels, 1, 1),
            scaling_per_channel=activation_scaling_per_channel,
            return_quant_tensor=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x
    

class QuantDwsConv(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            stride=1,
            padding=None,
            weight_bit_width=8,
            act_bit_width=8,
            bn_eps=1e-5):
        super(QuantDwsConv, self).__init__()
        self.dw = QuantConvBNReLU(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            weight_bit_width=weight_bit_width,
            act_bit_width=act_bit_width,
            groups=in_channels,
            bn_eps=bn_eps
        )
        self.pw = QuantConvBNReLU(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=None,
            weight_bit_width=weight_bit_width,
            act_bit_width=act_bit_width,
            bn_eps=bn_eps
        )

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return x


class QuantInvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio=1, weight_bit_width=8, act_bit_width=8, skip=True):
        super(QuantInvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup and skip
        # self.activation = QuantReLU(
        #                             act_quant=CommonUintActQuant,
        #                             bit_width=act_bit_width,
        #                             per_channel_broadcastable_shape=(1, hidden_dim, 1, 1),
        #                             scaling_per_channel=False,
        #                             return_quant_tensor=True)
        self.identity = QuantIdentity(
                                      act_quant=CommonIntActQuant,
                                      bit_width=act_bit_width,
                                      per_channel_broadcastable_shape=(1, hidden_dim, 1, 1),
                                      scaling_per_channel=False,
                                      return_quant_tensor=False)
        
        layers = []
        # pw
        if expand_ratio != 1:
            layers.append(QuantConvBNReLU(inp, hidden_dim, kernel_size=1, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width))
        
        layers.extend([
                        # dw
                        QuantConvBNReLU(hidden_dim, hidden_dim, kernel_size=3, weight_bit_width=weight_bit_width, 
                                   act_bit_width=act_bit_width, stride=stride, groups=hidden_dim),
                        
                        # pw-linear
                        QuantConv2d(
                        in_channels=hidden_dim,
                        # in_channels=inp,
                        out_channels=oup,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True,
                        weight_quant=CommonIntWeightPerChannelQuant,
                        weight_bit_width=weight_bit_width),
                        
                        # bn
                        nn.BatchNorm2d(num_features=oup),
                        # self.activation
                        self.identity
                    ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return self.identity(x) + self.conv(x)
            # return x + self.conv(x)
        else:
            return self.conv(x)



class QuantDetect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export
    end2end = False
    include_nms = False
    concat = False
    finn_export = False

    def __init__(self, nc=80, anchors=(), weight_bit_width=8, ch=()):  # detection layer
        super(QuantDetect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        # self.m = nn.ModuleList(QuantConv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.m = nn.ModuleList(QuantConv2d(
                                        in_channels=x,
                                        out_channels=self.no * self.na,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        bias=True,
                                        weight_quant=CommonIntWeightPerChannelQuant,
                                        weight_bit_width=weight_bit_width)
                                for x in ch)

    def forward(self, x):
        # print('anchors forward:', self.anchors)
        # print('detect shape:', [xi.shape for xi in x])
        # x = x.copy()  # for profiling
        # print('quantdect input:', len(x), [el.shape for el in x])
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            if not self.finn_export:
                x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

                if not self.training:  # inference
                    if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                        self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                    y = x[i].sigmoid()
                    if not torch.onnx.is_in_onnx_export():
                        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    else:
                        xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                        xy = xy * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))  # new xy
                        wh = wh ** 2 * (4 * self.anchor_grid[i].data)  # new wh
                        y = torch.cat((xy, wh, conf), 4)
                    z.append(y.view(bs, -1, self.no))

        if self.training or self.finn_export:
            out = x
        elif self.end2end:
            out = torch.cat(z, 1)
        elif self.include_nms:
            z = self.convert(z)
            out = (z, )
        elif self.concat:
            out = torch.cat(z, 1)
        else:
            out = (torch.cat(z, 1), x)

        return out

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def convert(self, z):
        z = torch.cat(z, 1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                           dtype=torch.float32,
                                           device=z.device)
        box @= convert_matrix                          
        return (box, score)
    


# --------------------- quantyolov8


class QuantConv(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            weight_bit_width=8,
            act_bit_width=8,
            padding=None,
            groups=1,
            dilation=1,
            act=True):
        super(QuantConv, self).__init__()
        self.conv = QuantConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=autopad(kernel_size, padding, dilation),
            groups=groups,
            bias=False,
            weight_quant=CommonIntWeightPerChannelQuant,
            weight_bit_width=weight_bit_width)
        self.bn = nn.BatchNorm2d(num_features=out_channels, eps=1e-5)
        self.act = act if isinstance(act, nn.Module) else QuantReLU(
                                                                    act_quant=CommonUintActQuant,
                                                                    bit_width=act_bit_width,
                                                                    per_channel_broadcastable_shape=(1, out_channels, 1, 1),
                                                                    scaling_per_channel=True,
                                                                    return_quant_tensor=False)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class QuantC2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    use_common_act = False

    def __init__(self, c1, c2, n=1, shortcut=False, weight_bit_width=8, act_bit_width=8, g=1, e=0.5, act=True):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        if self.use_common_act:
            self.common_act = QuantReLU(
                                        act_quant=CommonUintActQuant,
                                        bit_width=act_bit_width,
                                        scaling_per_channel=True,
                                        return_quant_tensor=False)
        else:
            self.common_act = True
            print("INDEPENDENT ACTS IN QuantC2f")
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = QuantConv(c1, 2 * self.c, 1, 1, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width, act=self.common_act)
        self.cv2 = QuantConv((2 + n) * self.c, c2, 1, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width, act=act)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(QuantBottleneck(self.c, self.c, shortcut=shortcut, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width, cv2_act=self.common_act, g=g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class QuantBottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, weight_bit_width=8, act_bit_width=8, g=1, k=(3, 3), e=0.5, cv2_act=True):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = QuantConv(c1, c_, k[0], 1, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width)
        self.cv2 = QuantConv(c_, c2, k[1], 1, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width, act=cv2_act, groups=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class QuantSPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5, weight_bit_width=8, act_bit_width=8, act=True):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = QuantConv(c1, c_, 1, 1, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width)
        self.cv2 = QuantConv(c_ * 4, c2, 1, 1, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width, act=act)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))
    

class QuantV8Detect(nn.Module):
    """YOLOv8 Detect head for detection models."""

    finn_export = False
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    end2end = False  # end2end
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    dedicated_loss = v8DetectionLoss
    dedicated_nms = V8_non_max_suppression

    def __init__(self, nc=80, weight_bit_width=8, act_bit_width=8, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(QuantConv(x, c2, 3, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width),
                          QuantConv(c2, c2, 3, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width),
                          QuantConv2d(c2, 4 * self.reg_max, 1, weight_quant=CommonIntWeightPerChannelQuant, weight_bit_width=weight_bit_width)) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(QuantConv(x, c3, 3, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width),
                          QuantConv(c3, c3, 3, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width),
                          QuantConv2d(c3, self.nc, 1, weight_quant=CommonIntWeightPerChannelQuant, weight_bit_width=weight_bit_width)) for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if self.end2end:
            return self.forward_end2end(x)
        # print('detect input:', [xi.shape for xi in x])

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training or self.finn_export:  # Training path
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def forward_end2end(self, x):
        """
        Performs forward pass of the v10Detect module.

        Args:
            x (tensor): Input tensor.

        Returns:
            (dict, tensor): If not in training mode, returns a dictionary containing the outputs of both one2many and one2one detections.
                           If in training mode, returns a dictionary containing the outputs of one2many and one2one detections separately.
        """
        x_detach = [xi.detach() for xi in x]
        one2one = [
            torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)
        ]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return {"one2many": x, "one2one": one2one}

        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, {"one2many": x, "one2one": one2one})

    def _inference(self, x):
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
        # Inference path
        # print('inference', [xi.shape for xi in x])
        shape = x[0].shape  # BCHW
        # print('x precat:', [xi.view(shape[0], self.no, -1).shape for xi in x])
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        # print('x cat:', x_cat.shape)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
            # print('anchors, strides:', self.anchors.shape, self.strides.shape)
            # # torch.set_printoptions(profile="full")
            # print(self.anchors)
            # # torch.set_printoptions(profile="default")
            # print(self.strides)

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes, anchors):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=not self.end2end, dim=1)

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
        """
        Post-processes the predictions obtained from a YOLOv10 model.

        Args:
            preds (torch.Tensor): The predictions obtained from the model. It should have a shape of (batch_size, num_boxes, 4 + num_classes).
            max_det (int): The maximum number of detections to keep.
            nc (int, optional): The number of classes. Defaults to 80.

        Returns:
            (torch.Tensor): The post-processed predictions with shape (batch_size, max_det, 6),
                including bounding boxes, scores and cls.
        """
        assert 4 + nc == preds.shape[-1]
        boxes, scores = preds.split([4, nc], dim=-1)
        max_scores = scores.amax(dim=-1)
        max_scores, index = torch.topk(max_scores, min(max_det, max_scores.shape[1]), axis=-1)
        index = index.unsqueeze(-1)
        boxes = torch.gather(boxes, dim=1, index=index.repeat(1, 1, boxes.shape[-1]))
        scores = torch.gather(scores, dim=1, index=index.repeat(1, 1, scores.shape[-1]))

        # NOTE: simplify but result slightly lower mAP
        # scores, labels = scores.max(dim=-1)
        # return torch.cat([boxes, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=-1)

        scores, index = torch.topk(scores.flatten(1), max_det, axis=-1)
        labels = index % nc
        index = index // nc
        boxes = boxes.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, boxes.shape[-1]))

        return torch.cat([boxes, scores.unsqueeze(-1), labels.unsqueeze(-1).to(boxes.dtype)], dim=-1)