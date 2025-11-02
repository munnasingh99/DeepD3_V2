#!/usr/bin/env python3
"""
Visualize predictions by overlaying red dots on benchmark.tif
"""

import argparse
from pathlib import Path
import numpy as np
import cv2
import tifffile as tiff
import pandas as pd
from tqdm import tqdm


def minmax01(im: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1]"""
    vmin, vmax = float(im.min()), float(im.max())
    if vmax > vmin:
        im = (im - vmin) / (vmax - vmin)
    else:
        im = np.zeros_like(im, dtype=np.float32)
    return im.astype(np.float32)


def draw_dot(img, center_x, center_y, color=(0, 0, 255), radius=5, thickness=-1):
    """Draw a filled dot at the center position"""
    cv2.circle(img, (int(center_x), int(center_y)), radius, color, thickness)


def draw_box_with_dot(img, x1, y1, x2, y2, score=None, 
                      box_color=(0, 255, 0), dot_color=(0, 0, 255),
                      box_thickness=1, dot_radius=3):
    """Draw bounding box and center dot"""
    # Draw box
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), 
                  box_color, box_thickness)
    
    # Draw center dot
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    draw_dot(img, cx, cy, dot_color, dot_radius)
    
    # Optional: Add score label
    if score is not None:
        label = f"{score:.2f}"
        cv2.putText(img, label, (int(x1), max(10, int(y1)-4)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
        cv2.putText(img, label, (int(x1), max(10, int(y1)-4)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)


def main():
    ap = argparse.ArgumentParser(description='Overlay predictions on benchmark.tif')
    
    # Required
    ap.add_argument("--tif", required=True, help="Path to benchmark.tif")
    ap.add_argument("--predictions", required=True, 
                   help="Path to predictions CSV (raw or dedup)")
    ap.add_argument("--out", required=True, 
                   help="Output directory for visualizations")
    
    # Visualization options
    ap.add_argument("--mode", choices=['dots', 'boxes', 'both'], default='dots',
                   help="Visualization mode: dots only, boxes only, or both")
    ap.add_argument("--dot_radius", type=int, default=4,
                   help="Radius of dots in pixels")
    ap.add_argument("--box_thickness", type=int, default=2,
                   help="Thickness of bounding boxes")
    ap.add_argument("--show_score", action="store_true",
                   help="Show confidence scores on boxes")
    
    # Color options
    ap.add_argument("--dot_color", type=str, default="255,0,0",
                   help="Dot color in BGR format (default: 255,0,0 = red)")
    ap.add_argument("--box_color", type=str, default="0,255,0",
                   help="Box color in BGR format (default: 0,255,0 = green)")
    
    # Processing options
    ap.add_argument("--slice_start", type=int, default=None,
                   help="Start slice (default: all)")
    ap.add_argument("--slice_end", type=int, default=None,
                   help="End slice (default: all)")
    ap.add_argument("--slice_step", type=int, default=1,
                   help="Save every Nth slice (default: 1 = all)")
    
    # Output format
    ap.add_argument("--save_stacked", action="store_true",
                   help="Save as multi-page TIFF in addition to PNGs")
    ap.add_argument("--enhance_contrast", action="store_true",
                   help="Apply CLAHE for better visibility")
    
    args = ap.parse_args()
    
    # Parse colors
    dot_color = tuple(map(int, args.dot_color.split(',')))
    box_color = tuple(map(int, args.box_color.split(',')))
    
    # Setup output
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading {args.tif}...")
    vol = tiff.imread(args.tif)
    if vol.ndim == 2:
        vol = vol[None, ...]
    Z, H, W = vol.shape
    print(f"Volume shape: Z={Z}, H={H}, W={W}")
    
    print(f"\nLoading predictions from {args.predictions}...")
    df = pd.read_csv(args.predictions)
    print(f"Found {len(df)} predictions")
    
    # Determine slice range
    z_start = args.slice_start if args.slice_start is not None else 0
    z_end = args.slice_end if args.slice_end is not None else Z
    z_start = max(0, z_start)
    z_end = min(Z, z_end)
    
    print(f"\nProcessing slices {z_start} to {z_end-1} (step={args.slice_step})")
    print(f"Visualization mode: {args.mode}")
    print(f"Dot color (BGR): {dot_color}")
    if args.mode in ['boxes', 'both']:
        print(f"Box color (BGR): {box_color}")
    
    # Group predictions by z-slice
    predictions_by_z = {}
    for _, row in df.iterrows():
        z = int(row['z'])
        if z not in predictions_by_z:
            predictions_by_z[z] = []
        predictions_by_z[z].append(row)
    
    print(f"Predictions spread across {len(predictions_by_z)} slices")
    
    # Process slices
    output_slices = []
    saved_count = 0
    
    for z in tqdm(range(z_start, z_end), desc="Creating overlays"):
        # Get image
        img = vol[z].astype(np.float32)
        img_norm = minmax01(img)
        img8 = (img_norm * 255).astype(np.uint8)
        
        # Apply CLAHE for better contrast if requested
        if args.enhance_contrast:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img8 = clahe.apply(img8)
        
        # Convert to BGR for color overlay
        img_bgr = cv2.cvtColor(img8, cv2.COLOR_GRAY2RGB)
        
        # Draw predictions for this slice
        if z in predictions_by_z:
            for pred in predictions_by_z[z]:
                x1, y1 = pred['x1'], pred['y1']
                x2, y2 = pred['x2'], pred['y2']
                score = pred.get('score', None)
                
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                if args.mode == 'dots':
                    draw_dot(img_bgr, cx, cy, dot_color, args.dot_radius)
                    
                elif args.mode == 'boxes':
                    cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)),
                                box_color, args.box_thickness)
                    if args.show_score and score is not None:
                        label = f"{score:.2f}"
                        cv2.putText(img_bgr, label, (int(x1), max(10, int(y1)-4)),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)
                
                elif args.mode == 'both':
                    draw_box_with_dot(img_bgr, x1, y1, x2, y2, 
                                     score if args.show_score else None,
                                     box_color, dot_color,
                                     args.box_thickness, args.dot_radius)
        
        # Save individual PNG if at step interval
        if z % args.slice_step == 0:
            png_path = outdir / f"slice_{z:04d}.png"
            cv2.imwrite(str(png_path), img_bgr)
            saved_count += 1
        
        # Store for stacked TIFF
        if args.save_stacked:
            output_slices.append(img_bgr)
    
    print(f"\nSaved {saved_count} PNG files to {outdir}")
    
    # Save as stacked TIFF if requested
    if args.save_stacked:
        print("\nSaving stacked TIFF...")
        stacked = np.stack(output_slices, axis=0)
        tiff_path = outdir / "overlay_stack.tif"
        tiff.imwrite(tiff_path, stacked, compression='zlib')
        print(f"Saved: {tiff_path}")
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total predictions: {len(df)}")
    print(f"Slices with predictions: {len(predictions_by_z)}")
    print(f"Slices processed: {z_end - z_start}")
    print(f"PNG files saved: {saved_count}")
    
    if len(predictions_by_z) > 0:
        preds_per_slice = [len(predictions_by_z[z]) for z in predictions_by_z]
        print(f"Predictions per slice: min={min(preds_per_slice)}, "
              f"max={max(preds_per_slice)}, "
              f"mean={np.mean(preds_per_slice):.1f}")
    
    print("="*70)
    print("\nDone!")


if __name__ == "__main__":
    main()


"""
# Prediction Visualization Guide

This script overlays your predictions on the benchmark.tif file with red dots (or boxes) for easy visualization.

## Basic Usage

### Simple red dots (recommended):
```bash
python visualize_predictions.py \
    --tif benchmark.tif \
    --predictions output/predictions_dedup.csv \
    --out visualizations/
```

### With boxes and dots:
```bash
python visualize_predictions.py \
    --tif benchmark.tif \
    --predictions output/predictions_dedup.csv \
    --out visualizations/ \
    --mode both
```

### Just boxes (no dots):
```bash
python visualize_predictions.py \
    --tif benchmark.tif \
    --predictions output/predictions_dedup.csv \
    --out visualizations/ \
    --mode boxes \
    --show_score
```
"""