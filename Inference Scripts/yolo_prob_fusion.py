#!/usr/bin/env python3
"""
YOLO + Prob-UNet
Strategy: Use YOLO for detection, Prob-UNet ONLY to validate/boost scores.
"""

import argparse
from pathlib import Path
import numpy as np
import cv2
import tifffile as tiff
import torch
from tqdm import tqdm
import json
import pandas as pd

from ultralytics import YOLO
from models.prob_unet_with_tversky import ProbabilisticUnetDualLatent
from yolo_inference import Metrics, dedup_3d_predictions

def minmax01(im: np.ndarray) -> np.ndarray:
    vmin, vmax = float(im.min()), float(im.max())
    if vmax > vmin:
        im = (im - vmin) / (vmax - vmin)
    else:
        im = np.zeros_like(im, dtype=np.float32)
    return im.astype(np.float32)

def get_box_mean_prob(prob_map, box_xyxy):
    """Get mean probability in a box region."""
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    h, w = prob_map.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    patch = prob_map[y1:y2, x1:x2]
    return float(patch.mean())


def main():
    ap = argparse.ArgumentParser()
    
    # Model weights
    ap.add_argument("--yolo_weights", required=True)
    ap.add_argument("--punet_weights", required=True)
    
    # Data
    ap.add_argument("--tif", required=True)
    ap.add_argument("--annotations", default="dataset/Annotations_and_Clusters.csv")
    ap.add_argument("--out", required=True)
    
    # YOLO parameters
    ap.add_argument("--img_h", type=int, default=352)
    ap.add_argument("--img_w", type=int, default=1472)
    ap.add_argument("--yolo_conf", type=float, default=0.02)
    ap.add_argument("--yolo_iou", type=float, default=0.40)
    
    # Tiling
    ap.add_argument("--tiling", action="store_true")
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.35)
    
    # Prob-UNet
    ap.add_argument("--probunet_samples", type=int, default=24)
    
    # Validation strategy
    ap.add_argument("--yolo_high_conf", type=float, default=0.50,
                   help="Auto-keep if YOLO conf >= this")
    ap.add_argument("--yolo_low_conf", type=float, default=0.08,
                   help="Reject if YOLO conf < this")
    ap.add_argument("--punet_validation", type=float, default=0.35,
                   help="Prob-UNet prob threshold for validation")
    ap.add_argument("--weight_yolo", type=float, default=0.45)
    ap.add_argument("--weight_punet", type=float, default=0.55)
    
    # 3D dedup
    ap.add_argument("--dedup_nm", type=float, default=1000.0)
    ap.add_argument("--px_x", type=float, default=94.0)
    ap.add_argument("--px_y", type=float, default=94.0)
    ap.add_argument("--px_z", type=float, default=200.0)
    
    # Visualization
    ap.add_argument("--save_vis", action="store_true")
    ap.add_argument("--vis_interval", type=int, default=10)
    
    args = ap.parse_args()
    
    # Setup
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    visdir = outdir / "vis"
    if args.save_vis:
        visdir.mkdir(exist_ok=True, parents=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load models
    print("Loading YOLO...")
    yolo = YOLO(args.yolo_weights)
    
    print("Loading Prob-UNet...")
    punet = ProbabilisticUnetDualLatent(
        input_channels=1, num_classes=1,
        num_filters=[32, 64, 128, 192],
        latent_dim_dendrite=12, latent_dim_spine=12,
        no_convs_fcomb=4,
        recon_loss="tversky", tversky_alpha=0.3, tversky_beta=0.7, tversky_gamma=1.0
    ).to(device).eval()
    
    ckpt = torch.load(args.punet_weights, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    punet.load_state_dict(state, strict=True)
    
    # Load volume
    print(f"Loading {args.tif}...")
    vol = tiff.imread(args.tif)
    if vol.ndim == 2:
        vol = vol[None, ...]
    Z, H, W = vol.shape
    print(f"Volume: Z={Z}, H={H}, W={W}\n")
    
    # Stats
    stats = {
        'yolo_total': 0,
        'kept_high_conf': 0,
        'kept_validated': 0,
        'rejected_low_yolo': 0,
        'rejected_no_punet': 0
    }
    
    all_detections = []
    
    # Process slices
    for z in tqdm(range(Z), desc="Processing"):
        img = minmax01(vol[z].astype(np.float32))
        img8 = (img * 255).astype(np.uint8)
        
        if img8.ndim == 2:
            img_rgb = cv2.cvtColor(img8, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = img8
        
        # === YOLO Detection ===
        if not args.tiling:
            res = yolo.predict(
                source=[img_rgb],
                imgsz=[args.img_h, args.img_w],
                rect=True,
                conf=args.yolo_conf,
                iou=args.yolo_iou,
                verbose=False
            )[0]
            
            yolo_boxes = np.zeros((0, 4), dtype=np.float32)
            yolo_scores = np.zeros(0, dtype=np.float32)
            
            if res.boxes is not None and len(res.boxes) > 0:
                yolo_boxes = res.boxes.xyxy.cpu().numpy().astype(np.float32)
                yolo_scores = res.boxes.conf.cpu().numpy().astype(np.float32)
        else:
            from yolo_inference import infer_slice_tiled
            yolo_boxes, yolo_scores = infer_slice_tiled(
                yolo, img_rgb,
                tile=args.tile,
                overlap=args.overlap,
                conf=args.yolo_conf,
                iou=args.yolo_iou,
                wbf_iou=0.55
            )
        
        stats['yolo_total'] += len(yolo_boxes)
        
        if len(yolo_boxes) == 0:
            continue
        
        # === Prob-UNet Probability Map ===
        with torch.no_grad():
            img_tensor = torch.from_numpy(img)[None, None].float().to(device)
            punet.forward(img_tensor, training=False)
            
            # Average over samples
            probs = []
            for _ in range(args.probunet_samples):
                _, spine_logits = punet.sample(testing=True, use_posterior=False)
                prob = torch.sigmoid(spine_logits).squeeze().cpu().numpy()
                probs.append(prob)
            
            spine_prob = np.mean(probs, axis=0)
        
        # === Validate Each YOLO Detection ===
        kept_boxes = []
        kept_scores = []
        kept_status = []
        
        for box, yolo_conf in zip(yolo_boxes, yolo_scores):
            # Get Prob-UNet confidence in this box
            punet_prob = get_box_mean_prob(spine_prob, box)
            
            # Decision tree
            if yolo_conf >= args.yolo_high_conf:
                # High YOLO confidence → always keep
                final_score = yolo_conf
                status = 'high'
                keep = True
                stats['kept_high_conf'] += 1
                
            elif yolo_conf < args.yolo_low_conf:
                # Very low YOLO → reject
                keep = False
                status = 'rejected_low_yolo'
                stats['rejected_low_yolo'] += 1
                
            elif punet_prob >= args.punet_validation:
                # Medium YOLO + Prob-UNet confirms → keep with fused score
                final_score = (
                    args.weight_yolo * yolo_conf + 
                    args.weight_punet * punet_prob
                ) / (args.weight_yolo + args.weight_punet)
                status = 'validated'
                keep = True
                stats['kept_validated'] += 1
                
            else:
                # Medium YOLO but Prob-UNet doesn't confirm → reject
                keep = False
                status = 'rejected_no_punet'
                stats['rejected_no_punet'] += 1
            
            if keep:
                kept_boxes.append(box)
                kept_scores.append(final_score)
                kept_status.append(status)
        
        # Save
        for box, score, status in zip(kept_boxes, kept_scores, kept_status):
            all_detections.append({
                'z': z,
                'x1': float(box[0]),
                'y1': float(box[1]),
                'x2': float(box[2]),
                'y2': float(box[3]),
                'score': float(score),
                'status': status
            })
        
        # Visualization
        if args.save_vis and z % args.vis_interval == 0:
            vis = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
            
            for box, score, status in zip(kept_boxes, kept_scores, kept_status):
                x1, y1, x2, y2 = [int(v) for v in box]
                
                if status == 'high':
                    color = (0, 255, 0)  # Green
                else:
                    color = (0, 165, 255)  # Orange
                
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                label = f"{score:.2f}"
                cv2.putText(vis, label, (x1, max(10, y1-4)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
                cv2.putText(vis, label, (x1, max(10, y1-4)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            cv2.imwrite(str(visdir / f"z{z:03d}.png"), vis)
    
    # === Summary ===
    print("\n" + "="*70)
    print("STATISTICS")
    print("="*70)
    print(f"YOLO detections:        {stats['yolo_total']}")
    print(f"  Kept (high conf):     {stats['kept_high_conf']}")
    print(f"  Kept (validated):     {stats['kept_validated']}")
    print(f"  Rejected (low YOLO):  {stats['rejected_low_yolo']}")
    print(f"  Rejected (no PU):     {stats['rejected_no_punet']}")
    print(f"Total kept:             {len(all_detections)}")
    
    # Save
    df = pd.DataFrame(all_detections)
    csv_raw = outdir / "predictions_raw.csv"
    df.to_csv(csv_raw, index=False)
    print(f"\nSaved: {csv_raw}")
    
    # 3D dedup
    print("\nApplying 3D deduplication...")
    df_dedup = dedup_3d_predictions(
        df[['z', 'x1', 'y1', 'x2', 'y2', 'score']],
        d=np.array([args.px_x, args.px_y, args.px_z], dtype=np.float32),
        thr=args.dedup_nm
    )
    
    csv_dedup = outdir / "predictions_dedup.csv"
    df_dedup.to_csv(csv_dedup, index=False)
    print(f"Saved: {csv_dedup}")
    print(f"  Before: {len(df)}")
    print(f"  After:  {len(df_dedup)}")
    
    # Evaluate
    print("\nEvaluating...")
    metrics = Metrics(
        prediction=df_dedup,
        tif_path=args.tif,
        csv_path=args.annotations
    )
    
    P, R, F1 = metrics.get_precision_recall(
        threshold=args.dedup_nm,
        d=np.array([args.px_x, args.px_y, args.px_z], dtype=np.float32)
    )
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Precision: {P:.4f} ({P*100:.2f}%)")
    print(f"Recall:    {R:.4f} ({R*100:.2f}%)")
    print(f"F1 Score:  {F1:.4f} ({F1*100:.2f}%)")
    print("="*70)
    
    # Save results
    results = {
        'precision': float(P),
        'recall': float(R),
        'f1_score': float(F1),
        'detections': len(df_dedup),
        'statistics': stats,
        'parameters': vars(args)
    }
    
    with open(outdir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved: {outdir / 'results.json'}")


if __name__ == "__main__":
    main()

# Usage example:
"""
python3 yolo_prob_fusion.py \
--yolo_weights yolo_weights/best.pt \
--punet_weights punet_weights/best_model_f1_0.7119.pth \
--tif dataset/DeepD3_Benchmark.tif \
--out fusion
"""