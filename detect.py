import argparse
import time
from pathlib import Path
import os
import sys

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import yaml

# from models.experimental import attempt_load
from models.yolo import get_model
from models.finn_models import QuantC2f
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box, plot_images, output_to_target
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


def detect(save_img=False):
    source, weights, cfg, view_img, save_txt, imgsz, trace, mot_format, classes = opt.source, opt.weights, opt.cfg, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace, opt.mot_format, opt.classes
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    # half = device.type != 'cpu'  # half precision only supported on CUDA
    half = False

    # Load model
    # model = attempt_load(weights, map_location=device)  # load FP32 model
    model, ckpt, _ = get_model(cfg, weights, num_classes=data_dict['nc'], device=device, load_ema=True)
    model.eval()
    for m in model.modules():
    # print(type(m))
        if isinstance(m, QuantC2f):
            m.forward = m.forward_split
    stride = int(model.stride.max())  # model stride
    # imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
   
    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        

    # Get names and colors
    # names = model.module.names if hasattr(model, 'module') else model.names    
    names = data_dict['names']
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    if len(imgsz) == 2:
        old_img_h, old_img_w = imgsz
    elif len(imgsz) == 1:
        imgsz = imgsz[0]
        old_img_h = old_img_w = imgsz
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, old_img_h, old_img_w).to(device).type_as(next(model.parameters())))  # run once
    old_img_b = 1

    
    if mot_format:
        seqnames = os.listdir(source)
        # seqnames = ["MVI_40701", "MVI_40771", "MVI_40863"]
    for seqname in seqnames:
        # if 'FRCNN' not in seqname:
        #     continue
        seq_save_dir = save_dir / seqname
        mot_txt_path = seq_save_dir / 'det' / 'det.txt'
        savedimg_path = seq_save_dir / 'img'
        os.makedirs(seq_save_dir / 'det')
        if save_img:
            os.makedirs(seq_save_dir / 'img')
        dataset = LoadImages(os.path.join(source, seqname, 'img1'), img_size=imgsz, stride=stride)
        t0 = time.time()
        nms = non_max_suppression
        last_layer = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]
        if hasattr(last_layer, 'dedicated_nms'):
            nms = last_layer.dedicated_nms
        key = None
        for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
            print('img:', img.shape)
            exp_dir = '/home/vision/danilowi/serious_mot/finn/notebooks/experiments/yolov8'
            test_input_np = img.copy()
            # cv2.imshow('img', img.transpose(1, 2, 0))
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=opt.augment)[0]

            # Inference
            t1 = time_synchronized()
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred, x = model(img, augment=opt.augment)
            t2 = time_synchronized()
            print('PRED:', pred.shape)
            print(pred)

            # Apply NMS
            pred = nms(pred, opt.conf_thres, opt.iou_thres, agnostic=opt.agnostic_nms)
            t3 = time_synchronized()

            # plot_images(img, output_to_target(pred), [path], save_dir / "result.jpg", names)
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(savedimg_path / p.name)  # img.jpg
                # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if mot_format:
                            xywh = [xyxy[0], xyxy[1], xyxy[2]-xyxy[0], xyxy[3]-xyxy[1]]
                            line = (frame_idx+1, -1, *xywh, conf, -1, -1, -1)
                            if not len(classes) or cls in classes:
                                with open(mot_txt_path, 'a') as f:
                                    f.write(('%g,' * len(line)).rstrip(',') % line + '\n')

                        if save_img or view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                # Print time (inference + NMS)
                # print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

                # Stream results
                if view_img:
                    # plt.imshow(im0)
                    # plt.show()
                    cv2.imshow('demo', im0)
                    key = cv2.waitKey(0)
                    if key == ord('v'):
                        np.save(exp_dir + '/test_input_{}x{}.npy'.format(test_input_np.shape[1], test_input_np.shape[2]), np.expand_dims(test_input_np, 0))
                        print("IMAGE SAVED")
                        if model.saved_features:
                            for i, f in enumerate(model.saved_features):
                                np.save(exp_dir + '/{}_f{}.npy'.format(opt.name, i), f.detach().cpu())
                        for out_idx, out in enumerate(x):
                            np.save(exp_dir + '/{}_trainout{}.npy'.format(opt.name, out_idx), out.detach().cpu())
                        print('FEATURES SAVED')
                        key = cv2.waitKey(0)



                # Save results (image with detections)
                if save_img:
                    cv2.imwrite(save_path, im0)
                    print(save_path, 'saved')
                    # if dataset.mode == 'image':
                    #     cv2.imwrite(save_path, im0)
                    #     # print(f" The image with the result is saved in: {save_path}")
                    # else:  # 'video' or 'stream'
                    #     if vid_path != save_path:  # new video
                    #         vid_path = save_path
                    #         if isinstance(vid_writer, cv2.VideoWriter):
                    #             vid_writer.release()  # release previous video writer
                    #         if vid_cap:  # video
                    #             fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    #             w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    #             h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    #         else:  # stream
                    #             fps, w, h = 30, im0.shape[1], im0.shape[0]
                    #             save_path += '.mp4'
                    #         vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    #     vid_writer.write(im0)
            if key == ord('s') or key == ord('q'):
                break
        if key == ord('q'):
            break

        # if save_txt or save_img:
        #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #     #print(f"Results saved to {save_dir}{s}")

        print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='data.yaml path (for classnames)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', nargs='+', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, default=[], help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--mot-format', action='store_true', help='output detections to txt file in mot format')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
