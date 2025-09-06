#python3 export.py --weights runs/quantyolov8n_8w8a_640/best.pt --cfg runs/quantyolov8n_8w8a_640/cfg.yaml --data data/data.yaml --load_ema --input_shape 640 640
python3 export.py --weights runs/train/quantyolov8_8w8a4/weights/best.pt --cfg runs/train/quantyolov8_8w8a4/cfg.yaml --data data/data.yaml --load_ema --input_shape 416 416
#python3 export.py --weights runs/quantyolov8n_8w8a_320/best.pt --cfg runs/quantyolov8n_8w8a_320/cfg.yaml --data data/data.yaml --load_ema --input_shape 320 320
