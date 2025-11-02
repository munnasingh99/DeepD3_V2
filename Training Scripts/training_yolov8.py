"""
yolov8_optimized_training.py
Rectangular YOLOv8 training tuned for tiny, dense spines (366x1444).
"""

from ultralytics import YOLO


config = {
    # Model
    "model": "yolov8l.pt",   

    # Image geometry (CRITICAL)
    "imgsz": 1024,             # near-native, stride-32 aligned
    "rect": False,                     # keep aspect ratio

    # Training
    "epochs": 60,
    "batch": 8,                  # maximize GPU
    "patience": 15,                   # early stop if flat

    # Optimizer / LR
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "weight_decay": 0.0005,
    "warmup_epochs": 5.0,

    # Augmentations for small, dense objects
    "mosaic": 0.8,
    "mixup": 0.0,                     
    "scale": 0.15,                    
    "translate": 0.05,
    "degrees": 10.0,
    "flipud": 0.0,                    # set 0.5 if orientation truly doesn’t matter
    "fliplr": 0.5,
    "hsv_h": 0.015,
    "hsv_s": 0.4,                     
    "hsv_v": 0.2,                     
       
   # Loss weights (single class)
    "single_cls": True,
    "box": 10.0,
    "cls": 0.3,
    "dfl": 1.5,
    "label_smoothing": 0.0,

    # System
    "device": 0,
    "workers": 8,
    "project": "yolov8_training",
    "save_period": 10,
    "amp": True,
    "cache": False,                       
    "seed": 42,
}


def train_yolov8_optimized(data_yaml: str, cfg: dict = None):
    cfg = cfg or config
    print("YOLOv8 — Rectangular training for tiny, dense spines")
    for k, v in cfg.items():
        print(f"{k:16s}: {v}")



    results = model.train(
        data=data_yaml,
        imgsz=cfg["imgsz"],
        rect=cfg["rect"],
        epochs=cfg["epochs"],
        batch=cfg["batch"],
        patience=cfg["patience"],
        optimizer=cfg["optimizer"],
        lr0=cfg["lr0"],
        lrf=cfg["lrf"],
        weight_decay=cfg["weight_decay"],
        warmup_epochs=cfg["warmup_epochs"],
        # augs
        mosaic=cfg["mosaic"],
        mixup=cfg["mixup"],
        scale=cfg["scale"],
        translate=cfg["translate"],
        degrees=cfg["degrees"],
        flipud=cfg["flipud"],
        fliplr=cfg["fliplr"],
        hsv_h=cfg["hsv_h"],
        hsv_s=cfg["hsv_s"],
        hsv_v=cfg["hsv_v"],
        # loss / task
        single_cls=cfg["single_cls"],
        box=cfg["box"],
        cls=cfg["cls"],
        dfl=cfg["dfl"],
        label_smoothing=cfg["label_smoothing"],
        # sys
        device=cfg["device"],
        workers=cfg["workers"],
        project=cfg["project"],
        name=cfg["name"],
        save_period=cfg["save_period"],
        amp=cfg["amp"],
        cache=cfg["cache"],
        seed=cfg["seed"],
        exist_ok=True,
        verbose=True,
    )

    # Post-train: validate with same rectangular geometry (and save COCO JSON/plots)
    model.val(data=data_yaml, imgsz=cfg["imgsz"], rect=True, save_json=True, plots=True)
    return results


def create_p2_model_yaml():
    """
    YOLOv8l with an extra P2 head (stride-4) for tiny-object detection.
    Indices follow the official YOLOv8 topology; channels match v8l.
    """
    yaml_content = """
# YOLOv8l with P2 head (P2,P3,P4,P5 detection)

nc: 1      # number of classes
ch: 3      # input channels

# Backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]       # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]      # 1-P2/4
  - [-1, 3, C2f,  [128]]            # 2
  - [-1, 1, Conv, [256, 3, 2]]      # 3-P3/8
  - [-1, 6, C2f,  [256]]            # 4
  - [-1, 1, Conv, [512, 3, 2]]      # 5-P4/16
  - [-1, 6, C2f,  [512]]            # 6
  - [-1, 1, Conv, [1024, 3, 2]]     # 7-P5/32
  - [-1, 3, C2f,  [1024]]           # 8
  - [-1, 1, SPPF, [1024, 5]]        # 9

# Head (PAFPN + extra P2 branch)
head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 6], 1, Concat, [1]]       # cat P4
  - [-1, 3, C2f, [512]]             # 12

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 4], 1, Concat, [1]]       # cat P3
  - [-1, 3, C2f, [256]]             # 15  (P3/8)

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 2], 1, Concat, [1]]       # cat P2
  - [-1, 3, C2f, [128]]             # 18  (P2/4) ← extra branch

  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 15], 1, Concat, [1]]      # cat P3
  - [-1, 3, C2f, [256]]             # 21  (P3/8)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]      # cat P4
  - [-1, 3, C2f, [512]]             # 24  (P4/16)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 9],  1, Concat, [1]]      # cat P5
  - [-1, 3, C2f, [1024]]            # 27  (P5/32)

  - [[18, 21, 24, 27], 1, Detect, [nc]]   # Detect at P2,P3,P4,P5
"""
    with open('yolov8l_p2.yaml', 'w') as f:
        f.write(yaml_content)
    return 'yolov8l_p2.yaml'


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='dataset.yaml path')
    parser.add_argument('--mode', type=str, default='train_p2', 
                       choices=['train', 'train_p2'])
    args = parser.parse_args()
    
    if args.mode == 'train':
        # Standard YOLOv8l training with optimizations
        print("\nTraining YOLOv8l optimized model...")
        results = train_yolov8_optimized(args.data, config)
        
        print("\nTraining complete!")
        print(f"Best weights: {results.save_dir}/weights/best.pt")
    
    elif args.mode == 'train_p2':
        # Train with P2 head
        print("\nCreating and training YOLOv8l-P2 model...")
        p2_yaml = create_p2_model_yaml()
        model = YOLO('yolov8l.pt')
    
    # Then override with custom P2 architecture
        model = YOLO(p2_yaml)
        # Modify config for P2
        #config['name'] = 'optimized_v8l_p2_1024'
        
        results = train_yolov8_optimized(args.data, config)
        
        print("\nP2 training complete")

# Usage example:
"""
python3 training_yolov8.py --data spine_detection.yaml 
"""