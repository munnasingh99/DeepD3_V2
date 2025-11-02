"""
yolo_inference.py

End-to-end benchmark evaluator:
1) YOLO inference on DeepD3_Benchmark.tif (rectangular or tiled)
2) Save per-slice overlay PNGs + detections CSV
3) 3D de-duplication of detections

"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import imageio.v3 as iio
from ultralytics import YOLO
from numba import njit
import imageio as io
import glob
import re
import os
from ensemble_boxes import weighted_boxes_fusion

@njit
def distance(A, B, d=np.array([94, 94, 200], dtype=np.float32)):
    diff = (A-B) * d
    return (diff**2).sum()**0.5
    
@njit
def distanceMatrix(pA, pB, d=np.array([94, 94, 200], dtype=np.float32)):
    """
    Compute distance matrix between two point sets pA and pB.
    """
    M = np.zeros((pA.shape[0], pB.shape[0]), dtype=np.float32)

    for i in range(pA.shape[0]):
        for j in range(pB.shape[0]):
            M[i, j] = distance(pA[i], pB[j], d)

    return M


class Metrics():
    """
    Compute precision and recall based on 3D distance thresholding.
    """

    def __init__(self, prediction=None, tif_path='dataset/DeepD3_Benchmark.tif', 
                 csv_path='dataset/Annotations_and_Clusters.csv'):
        """
        Args:
            prediction: DataFrame with columns [z, x1, y1, x2, y2, score] OR numpy array [N, 3] (x, y, z)
            tif_path: Path to benchmark TIFF
            csv_path: Path to annotations CSV
        """
        self.stack = np.asarray(io.mimread(tif_path))
        self.df = pd.read_csv(csv_path, index_col=0)
        self.pred = prediction
        
    def transform(self):
        """
        Transform predictions to center coordinates format.
        """
        if isinstance(self.pred, pd.DataFrame):
            # Convert DataFrame to center coordinates
            pred = np.c_[
                0.5 * (self.pred.x1 + self.pred.x2), 
                0.5 * (self.pred.y1 + self.pred.y2), 
                self.pred.z
            ]
        elif isinstance(self.pred, np.ndarray):
            # Already in correct format
            pred = self.pred
        else:
            raise TypeError(f"prediction must be DataFrame or ndarray, got {type(self.pred)}")
        
        return pred.astype(np.float32)
        
    def distance_metrics(self, d=np.array([94, 94, 200], dtype=np.float32)):
        """
        Compute distance matrix between ground truth and predictions.
        """
        labels = self.df.groupby('label').sum()
        r = labels.Rater.apply(len)
        self.labels_avg = labels.values[:, 1:].astype(float) / r.values[..., None]
        
        pred = self.transform()
        
        self.distance_matrix = distanceMatrix(self.labels_avg, pred, d=d)
        return self.distance_matrix
    
    def get_precision_recall(self, threshold=1000.0, d=np.array([94, 94, 200], dtype=np.float32)):
        """
        Compute precision, recall, and F1 score based on distance threshold.
        """
        self.distance_matrix = self.distance_metrics(d=d)
        self.threshold = threshold

        # Initial guess: match each GT to closest prediction
        Mfound = np.zeros_like(self.distance_matrix, dtype=bool)
        initial_guesses = np.argmin(self.distance_matrix, 1)

        for i in range(self.distance_matrix.shape[0]):
            Mfound[i, initial_guesses[i]] = self.distance_matrix[i, initial_guesses[i]] <= self.threshold

        # Resolve ambiguities: if multiple GTs claim same prediction, keep only closest
        for j in range(Mfound.shape[1]):
            ambiguous = Mfound[:, j].sum()

            if ambiguous > 1:
                ix = np.where(Mfound[:, j])[0]
                ix_smallest = np.argmin(self.distance_matrix[ix, j])

                for k in range(ix.shape[0]):
                    if k != ix_smallest:
                        Mfound[ix[k], j] = False

        # Metrics
        TP = Mfound.sum()
        P = Mfound.shape[0]  # Total GT
        TP_FP = Mfound.shape[1]  # Total predictions

        self.recall = TP / P if P > 0 else 0.0
        self.precision = TP / TP_FP if TP_FP > 0 else 0.0
        

        if self.precision + self.recall > 0:
            self.f1_score = 2 * self.precision * self.recall / (self.precision + self.recall)
        else:
            self.f1_score = 0.0

        return (self.precision, self.recall, self.f1_score)



def sliding_windows(h, w, size=640, overlap=0.5):
    """
    Generate sliding window coordinates for image of size (h, w).
    """
    step = int(size * (1 - overlap))
    step = max(1, step)  # Ensure step is at least 1
    
    ys = list(range(0, max(1, h - size + 1), step))
    xs = list(range(0, max(1, w - size + 1), step))
    
    # Handle edge cases
    if not ys or h > size:
        ys = ys if ys else [0]
        if ys[-1] + size < h:
            ys.append(h - size)
    
    if not xs or w > size:
        xs = xs if xs else [0]
        if xs[-1] + size < w:
            xs.append(w - size)
    
    return [(y, x, min(y + size, h), min(x + size, w)) for y in ys for x in xs]


def boxes_to_yoloformat(xyxy, scores, w, h):
    """Convert boxes to normalized [x1, y1, x2, y2] format."""
    if len(xyxy) == 0:
        return np.zeros((0, 4), dtype=np.float32), [], []
    
    x1, y1, x2, y2 = [xyxy[:, i] for i in range(4)]
    boxes_norm = np.stack([x1 / w, y1 / h, x2 / w, y2 / h], axis=1)
    return boxes_norm, scores.tolist(), [0] * len(scores)


def yoloformat_to_xyxy(boxes_norm, w, h):
    """Convert normalized boxes back to pixel coordinates."""
    if len(boxes_norm) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    
    return np.stack([
        boxes_norm[:, 0] * w,
        boxes_norm[:, 1] * h,
        boxes_norm[:, 2] * w,
        boxes_norm[:, 3] * h
    ], axis=1)


def fuse_wbf(list_boxes, list_scores, list_labels, iou_thr=0.55, skip_box_thr=0.0):
    """
    Apply Weighted Boxes Fusion to combine overlapping boxes.
    """
    b, s, l = weighted_boxes_fusion(
        list_boxes, list_scores, list_labels,
        iou_thr=iou_thr, skip_box_thr=skip_box_thr
    )
    return b, s, l



def draw_detections(gray_img, det_xyxy, det_scores, color=(0, 255, 0)):
    """
    Draw detections on grayscale image.
    """
    if len(gray_img.shape) == 2:
        vis = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    elif gray_img.shape[2] == 1:
        vis = cv2.cvtColor(gray_img[:, :, 0], cv2.COLOR_GRAY2BGR)
    else:
        vis = gray_img.copy()
    
    for (x1, y1, x2, y2), sc in zip(det_xyxy, det_scores):
        x1i, y1i, x2i, y2i = [int(round(v)) for v in (x1, y1, x2, y2)]
        cx, cy = int((x1i + x2i) * 0.5), int((y1i + y2i) * 0.5)
        
        # Draw center point
        cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)
        # Draw bounding box
        cv2.rectangle(vis, (x1i, y1i), (x2i, y2i), color, 2)
        
        # Draw score label
        label = f"{sc:.2f}"
        cv2.putText(vis, label, (x1i, max(10, y1i-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
        cv2.putText(vis, label, (x1i, max(10, y1i-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    return vis


def normalize_to_uint8(img):
    """Normalize image to uint8 range."""
    if img.dtype == np.uint8:
        return img
    
    vmin, vmax = float(img.min()), float(img.max())
    if vmax - vmin < 1e-8:
        return np.zeros_like(img, dtype=np.uint8)
    
    normalized = (img - vmin) / (vmax - vmin) * 255.0
    return np.clip(normalized, 0, 255).astype(np.uint8)



def infer_slice_single(model, img, imgsz_hw, conf, iou):
    """
    Single-pass inference on full image with rectangular aspect ratio.
    """
    res = model.predict(
        source=[img], 
        imgsz=list(imgsz_hw), 
        rect=True,
        conf=conf, 
        iou=iou, 
        verbose=False
    )[0]
    
    if res.boxes is None or len(res.boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    
    xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float32)
    scores = res.boxes.conf.cpu().numpy().astype(np.float32)
    
    return xyxy, scores


def infer_slice_tiled(model, img, tile=640, overlap=0.5, conf=0.05, iou=0.6, wbf_iou=0.55):
    """
    Apply tiled inference with overlapping windows.
    """
    h, w = img.shape[:2]
    boxes_all, scores_all, labels_all = [], [], []
    
    for (y1, x1, y2, x2) in sliding_windows(h, w, size=tile, overlap=overlap):
        crop = img[y1:y2, x1:x2]
        
        # Skip very small crops
        if crop.shape[0] < 32 or crop.shape[1] < 32:
            continue
        
        res = model.predict(
            source=[crop], 
            imgsz=tile, 
            conf=conf, 
            iou=iou, 
            verbose=False
        )[0]
        
        if res.boxes is None or len(res.boxes) == 0:
            continue
        
        xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float32)
        scores = res.boxes.conf.cpu().numpy().astype(np.float32)
        
        # Transform to global coordinates
        xyxy[:, [0, 2]] += x1
        xyxy[:, [1, 3]] += y1
        
        # Convert to normalized format for WBF
        b_norm, s_list, l_list = boxes_to_yoloformat(xyxy, scores, w, h)
        boxes_all.append(b_norm)
        scores_all.append(s_list)
        labels_all.append(l_list)

    if len(boxes_all) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    # Apply WBF
    b_norm_f, s_f, _ = fuse_wbf(
        boxes_all, scores_all, labels_all, 
        iou_thr=wbf_iou, skip_box_thr=0.0
    )
    
    # Convert back to pixel coordinates
    xyxy_f = yoloformat_to_xyxy(np.array(b_norm_f, dtype=np.float32), w, h)
    scores_f = np.array(s_f, dtype=np.float32)
    
    return xyxy_f, scores_f



def run_once_and_save(weights, tif_path, out_dir,
                      conf, iou,
                      imgsz_hw=(352, 1472),
                      tiling=False, tile=640, overlap=0.5, wbf_iou=0.55):
    """
    Run inference on a TIFF stack and save results.
    """
    out_dir = Path(out_dir)
    png_dir = out_dir / f"slices_conf{conf:.2f}_iou{iou:.2f}"
    png_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {weights}...")
    model = YOLO(weights)
    
    print(f"Loading stack from {tif_path}...")
    stack = iio.imread(tif_path)
    
    if stack.ndim == 2:
        stack = stack[None, ...]
    
    Z = stack.shape[0]
    print(f"Processing {Z} slices...")
    
    rows = []

    for z in range(Z):
        if z % 10 == 0:
            print(f"  Slice {z}/{Z}...")
        
        img = stack[z]
        
        # Handle different image formats
        if img.ndim == 3:
            if img.shape[2] == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img[:, :, 0]
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        gray8 = normalize_to_uint8(gray)

        # Run inference
        if tiling:
            xyxy, scores = infer_slice_tiled(
                model, gray8, tile=tile, overlap=overlap,
                conf=conf, iou=iou, wbf_iou=wbf_iou
            )
        else:
            xyxy, scores = infer_slice_single(
                model, gray8, imgsz_hw=imgsz_hw,
                conf=conf, iou=iou
            )

        # Draw and save visualization
        vis = draw_detections(gray8, xyxy, scores, color=(0, 255, 0))
        #cv2.imwrite(str(png_dir / f"stack_z{z:03d}.png"), vis)

        # Store detections
        for (x1, y1, x2, y2), sc in zip(xyxy, scores):
            rows.append({
                "z": z, 
                "x1": float(x1), 
                "y1": float(y1),
                "x2": float(x2), 
                "y2": float(y2), 
                "score": float(sc)
            })

    # Save CSV
    df = pd.DataFrame(rows, columns=["z", "x1", "y1", "x2", "y2", "score"])
    csv_path = out_dir / f"predictions_conf{conf:.2f}_iou{iou:.2f}.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"  Saved {len(df)} detections to {csv_path}")
    #print(f"  Saved visualizations to {png_dir}")
    
    return df


def dedup_3d_predictions(csv_path_or_df, d=np.array([94,94,200], dtype=np.float32), thr=1000.0):
    """
    Apply 3D non-maximum suppression to deduplicate predictions.
    """
    if isinstance(csv_path_or_df, (str, Path)):
        df = pd.read_csv(csv_path_or_df)
    elif isinstance(csv_path_or_df, pd.DataFrame):
        df = csv_path_or_df.copy()
    else:
        raise TypeError("Input must be file path or DataFrame")
    
    if len(df) == 0:
        return df
    
    # Centers in pixels + z index
    xc = 0.5 * (df.x1 + df.x2).to_numpy(np.float32)
    yc = 0.5 * (df.y1 + df.y2).to_numpy(np.float32)
    zc = df.z.to_numpy(np.float32)
    s  = df.score.to_numpy(np.float32)

    idx = np.argsort(-s)  # Highest score first
    keep = []
    taken = np.zeros(len(df), dtype=bool)

    for i in idx:
        if taken[i]: 
            continue
        keep.append(i)
        
        # Compute scaled distances to remaining
        dx = (xc - xc[i]) * d[0]
        dy = (yc - yc[i]) * d[1]
        dz = (zc - zc[i]) * d[2]
        dist = np.sqrt(dx*dx + dy*dy + dz*dz)
        close = dist <= thr
        taken |= close

    return df.iloc[keep].reset_index(drop=True)


def read_prediction_csv_files_simple(folder_path):
    """
    Read and process prediction CSV files from a folder.
    """
    csv_files = glob.glob(os.path.join(folder_path, "predictions_*.csv"))
    
    if len(csv_files) == 0:
        print(f"WARNING: No CSV files found in {folder_path}")
        return {}
    
    dataframes = {}
    
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        filename_without_ext = os.path.splitext(filename)[0]
        
        # Extract conf and iou values
        pattern = r'predictions_conf([\d.]+)_iou([\d.]+)'
        match = re.search(pattern, filename_without_ext)
        
        if match:
            conf_value = match.group(1).replace('.', '')
            iou_value = match.group(2).replace('.', '')
            df_name = f"conf{conf_value}_iou{iou_value}"
        else:
            df_name = filename_without_ext.replace('.', '').replace('-', '_').replace(' ', '_')
        
        try:
            dataframes[df_name] = dedup_3d_predictions(file_path)
            print(f"  Loaded: {filename} as '{df_name}' ({len(dataframes[df_name])} detections)")
        except Exception as e:
            print(f"  ERROR loading {filename}: {str(e)}")
    
    return dataframes


def evaluate_grid(weights, tif_path, out_dir,
                  conf=0.10,
                  iou=0.55,
                  imgsz_hw=(352, 1472),
                  tiling=False, tile=640, overlap=0.5, wbf_iou=0.55,
                  d=np.array([94,94,200], dtype=np.float32), thr=1000.0,
                  annotations_csv='dataset/Annotations_and_Clusters.csv'):
    """
    Evaluate precision/recall over a grid of (conf, iou) settings
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = []

    # Step 1: Run inference for given configurations
    print("Running inference for given configurations")
    
   
    print(f"\n--- Inference conf={conf:.2f}, iou={iou:.2f} ---")
    run_once_and_save(
        weights, tif_path, out_dir,
        conf, iou, imgsz_hw,
        tiling, tile, overlap, wbf_iou
    )

    # Step 2: Load all prediction CSV
    print("Loading and deduplicating predictions")
    
    dfs = read_prediction_csv_files_simple(out_dir)
    
    if len(dfs) == 0:
        print("ERROR: No predictions found!")
        return

    # Step 3: Evaluate each configuration
    print("Evaluating metrics")
    
    for config_name, pred_df in dfs.items():
        print(f"\nEvaluating {config_name}...")
        print(f"  Detections after 3D NMS: {len(pred_df)}")
        
        try:
            # Create Metrics instance and evaluate
            metrics = Metrics(
                prediction=pred_df, 
                tif_path=tif_path,
                csv_path=annotations_csv
            )
            p, r, f1 = metrics.get_precision_recall(threshold=thr, d=d)
            
            # Parse config name
            conf_match = re.search(r'conf(\d+)', config_name)
            iou_match = re.search(r'iou(\d+)', config_name)
            
            conf_val = float(conf_match.group(1)) / 100 if conf_match else 0.0
            iou_val = float(iou_match.group(1)) / 100 if iou_match else 0.0
            
            results.append({
                'configuration': config_name,
                'conf_threshold': conf_val,
                'iou_threshold': iou_val,
                'num_detections': len(pred_df),
                'precision': p,
                'recall': r,
                'f1_score': f1
            })
            
            print(f"  Precision: {p:.4f} ({p*100:.2f}%)")
            print(f"  Recall:    {r:.4f} ({r*100:.2f}%)")
            print(f"  F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
            
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            
            results.append({
                'configuration': config_name,
                'conf_threshold': 0.0,
                'iou_threshold': 0.0,
                'num_detections': len(pred_df),
                'precision': np.nan,
                'recall': np.nan,
                'f1_score': np.nan
            })

    # Step 4: Create results summary
    print("Results Summary")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('f1_score', ascending=False)
    
    print("\n" + results_df.to_string(index=False))
    
    # Find best configuration
    if not results_df['f1_score'].isna().all():
        best_idx = results_df['f1_score'].idxmax()
        best = results_df.iloc[best_idx]
        
        print("CONFIGURATION")
        print(f"  Config:      {best['configuration']}")
        print(f"  Conf:        {best['conf_threshold']:.2f}")
        print(f"  IoU:         {best['iou_threshold']:.2f}")
        print(f"  Detections:  {best['num_detections']:.0f}")
        print(f"  Precision:   {best['precision']:.4f} ({best['precision']*100:.2f}%)")
        print(f"  Recall:      {best['recall']:.4f} ({best['recall']*100:.2f}%)")
        print(f"  F1 Score:    {best['f1_score']:.4f} ({best['f1_score']*100:.2f}%)")
    
    # Save results
    results_path = out_dir / 'precision_recall_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")
    
    return results_df



def parse_args():
    ap = argparse.ArgumentParser(
        description="Benchmark evaluation pipeline for YOLO spine detection"
    )
    
    # Required
    ap.add_argument("--weights", type=str, required=True, 
                    help="Path to YOLO weights (.pt)")
    
    # Paths
    ap.add_argument("--tif", type=str, default="dataset/DeepD3_Benchmark.tif", 
                    help="Path to benchmark TIFF")
    ap.add_argument("--annotations", type=str, default="dataset/Annotations_and_Clusters.csv",
                    help="Path to annotations CSV")
    ap.add_argument("--out", type=str, default="pred_eval_out", 
                    help="Output directory")

    # Inference settings
    ap.add_argument("--rect_h", type=int, default=352, 
                    help="Rectangular inference height")
    ap.add_argument("--rect_w", type=int, default=1472, 
                    help="Rectangular inference width")

    # Tiling
    ap.add_argument("--tiling", action="store_true", 
                    help="Use tiled inference + WBF")
    ap.add_argument("--tile", type=int, default=640, 
                    help="Tile size (if tiling)")
    ap.add_argument("--overlap", type=float, default=0.5, 
                    help="Tile overlap (0.0-1.0)")
    ap.add_argument("--wbf_iou", type=float, default=0.55, 
                    help="WBF IoU threshold for fusion")

    # Parameter sweep
    ap.add_argument("--conf", type=float, 
                    default=0.10,
                    help="Confidence value")
    ap.add_argument("--iou", type=float, 
                    default=0.55,
                    help="IoU NMS value")

    # 3D deduplication
    ap.add_argument("--thr", type=float, default=1000.0, 
                    help="3D dedup radius threshold")

    return ap.parse_args()


def main():
    args = parse_args()
    

    print("YOLO BENCHMARK EVALUATION PIPELINE\n")
    print(f"Weights:      {args.weights}")
    print(f"Benchmark:    {args.tif}")
    print(f"Annotations:  {args.annotations}")
    print(f"Output:       {args.out}")
    print(f"Tiling:       {args.tiling}")
    if args.tiling:
        print(f"  Tile size:  {args.tile}")
        print(f"  Overlap:    {args.overlap}")
        print(f"  WBF IoU:    {args.wbf_iou}")
    else:
        print(f"  Rect size:  {args.rect_h}x{args.rect_w}")
    print(f"Conf:        {args.conf}")
    print(f"IoU:        {args.iou}")
    print(f"3D dedup:     {args.thr}")
    print("="*70)
    
    evaluate_grid(
        weights=args.weights,
        tif_path=args.tif,
        out_dir=args.out,
        conf=tuple(args.conf),
        iou=tuple(args.iou),
        imgsz_hw=(args.rect_h, args.rect_w),
        tiling=args.tiling, 
        tile=args.tile, 
        overlap=args.overlap, 
        wbf_iou=args.wbf_iou,
        thr=args.thr,
        annotations_csv=args.annotations
    )


if __name__ == "__main__":
    main()

# Usage example:
"""python3 benchmark_eval_pipeline.py \
--weights yolov_weights/best.pt \
"""