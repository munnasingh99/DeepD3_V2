"""
Run inference with a trained Probabilistic U-Net model for dendrite and spine segmentation.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import tifffile as tiff
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import label as cc_label, center_of_mass


from prob_unet_with_tversky import ProbabilisticUnetDualLatent


def filter_small_objects_3d(binary_mask, min_size_voxels=40):
    """
    Remove small connected components (common false positives).
    
    Args:
        binary_mask: 3D binary array (Z, H, W)
        min_size_voxels: Minimum object size to keep
    
    Returns:
        Filtered binary mask (same dtype as input)
    """
    labeled, num_features = cc_label(binary_mask)
    
    if num_features == 0:
        return binary_mask
    
    # Count voxels per object
    sizes = np.bincount(labeled.ravel())[1:]  # Skip background (0)
    
    # Keep only objects >= min_size
    keep_labels = np.where(sizes >= min_size_voxels)[0] + 1
    
    mask_filtered = np.isin(labeled, keep_labels)
    return mask_filtered.astype(binary_mask.dtype)


def parse_float_list(x: str):
    """Parse comma-separated float string."""
    return [float(t.strip()) for t in x.split(",") if t.strip()]


def ensure_outdir(p: Path):
    """Create output directory if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)


def minmax01(im: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] range."""
    vmin, vmax = float(im.min()), float(im.max())
    if vmax > vmin:
        im = (im - vmin) / (vmax - vmin)
    else:
        im = np.zeros_like(im, dtype=np.float32)
    return im.astype(np.float32)


def save_stack(path: Path, arr: np.ndarray, dtype):
    """Save numpy array as TIFF stack."""
    tiff.imwrite(path.as_posix(), arr.astype(dtype))


@torch.no_grad()
def infer_logits_mean(model, device, img_2d: np.ndarray, mc_samples: int, 
                      use_posterior: bool, temperature: float = 1.0):
    """
    Return mean logits over multiple samples for a single 2D slice.
    Applies temperature scaling before sigmoid for better calibration.
    
    Args:
        model: Trained model
        device: torch device
        img_2d: 2D image slice
        mc_samples: Number of Monte Carlo samples
        use_posterior: Whether to use posterior sampling
        temperature: Temperature scaling factor (T>1 = softer predictions)
    
    Returns:
        (d_mean, s_mean): Mean logits after temperature scaling
    """
    H, W = img_2d.shape
    pad_h = (32 - (H % 32)) % 32
    pad_w = (32 - (W % 32)) % 32
    
    if pad_h or pad_w:
        x_np = np.pad(img_2d, ((0, pad_h), (0, pad_w)), mode="reflect")
    else:
        x_np = img_2d

    x = torch.from_numpy(x_np).unsqueeze(0).unsqueeze(0).to(device)
    model.forward(x, training=False)

    d_logits, s_logits = [], []
    S = max(1, mc_samples)
    for _ in range(S):
        ld, ls = model.sample(testing=True, use_posterior=use_posterior)
        d_logits.append(ld)
        s_logits.append(ls)

    d_stack = torch.stack(d_logits, 0)
    s_stack = torch.stack(s_logits, 0)
    
    d_mean = (d_stack.mean(0) / temperature).squeeze(0).squeeze(0)
    s_mean = (s_stack.mean(0) / temperature).squeeze(0).squeeze(0)

    if pad_h or pad_w:
        d_mean = d_mean[:H, :W]
        s_mean = s_mean[:H, :W]
    
    return d_mean, s_mean



def main():
    ap = argparse.ArgumentParser(description="Run inference and evaluate precision/recall")
    
    ap.add_argument("--weights", required=True, help="Model checkpoint path")
    ap.add_argument("--tif", required=True, help="Input TIF (Z,H,W) or (H,W)")
    ap.add_argument("--out", default="inference_results_punet/", help="Output directory")
    
    ap.add_argument("--posterior", action="store_true", help="Use posterior sampling")
    ap.add_argument("--samples", type=int, default=24, help="MC samples (default: 24)")
    ap.add_argument("--temp", type=float, default=1.4, help="Temperature scaling (default: 1.4)")

    ap.add_argument("--thr", type=float, default=0.50, help="Prediction threshold")

    ap.add_argument("--match_nm", type=float, default=1000.0, help="Centroid match distance (nm)")
    ap.add_argument("--px_x", type=float, default=94.0, help="Pixel size X (nm)")
    ap.add_argument("--px_y", type=float, default=94.0, help="Pixel size Y (nm)")
    ap.add_argument("--px_z", type=float, default=200.0, help="Pixel size Z (nm)")
    
    ap.add_argument("--min_size", type=int, default=40, help="Minimum object size (voxels)")
    
    args = ap.parse_args()

    outdir = Path(args.out)
    ensure_outdir(outdir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"INFERENCE PIPELINE")
    print(f"Device: {device}")
    print(f"Posterior sampling: {args.posterior}")
    print(f"MC samples: {args.samples}")
    print(f"Temperature: {args.temp}")

    model = ProbabilisticUnetDualLatent(
        input_channels=1,
        num_classes=1,
        num_filters=[32, 64, 128, 192],
        latent_dim_dendrite=12,
        latent_dim_spine=12,
        no_convs_fcomb=4,
        recon_loss="tversky",
        tversky_alpha=0.3,
        tversky_beta=0.7,
        tversky_gamma=1.0,
        beta_dendrite=1.0,
        beta_spine=1.0,
        loss_weight_dendrite=1.0,
        loss_weight_spine=1.0,
    ).to(device)
    model.eval()

    ckpt = torch.load(args.weights, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)

    vol = tiff.imread(args.tif)
    if vol.ndim == 2:
        vol = vol[np.newaxis, ...]
    Z, H, W = vol.shape

    thr = args.thr

    prob_d = np.zeros((Z, H, W), dtype=np.float32)
    prob_s = np.zeros((Z, H, W), dtype=np.float32)
    raw_masks = np.zeros((Z, H, W), dtype=np.uint8)

    print("\nRunning inference...")
    for z in tqdm(range(Z), desc="Inferring"):
        img = minmax01(vol[z].astype(np.float32))
        d_logits, s_logits = infer_logits_mean(
            model, device, img, args.samples, args.posterior, temperature=args.temp
        )

        d_prob = torch.sigmoid(d_logits).float().cpu().numpy()
        s_prob = torch.sigmoid(s_logits).float().cpu().numpy()
        prob_d[z] = d_prob
        prob_s[z] = s_prob


        raw_masks[z] = (s_prob >= thr).astype(np.uint8)


    raw_masks = filter_small_objects_3d(raw_masks, min_size_voxels=args.min_size)
    raw_masks = (raw_masks * 255).astype(np.uint8)

    print("\nSaving raw masks.")
    save_stack(outdir / "prob_dendrite.tif", prob_d, np.float32)
    save_stack(outdir / "prob_spine.tif", prob_s, np.float32)
    
    print("\nSaving filtered masks.")
    tag = f"{'post' if args.posterior else 'prior'}_S{args.samples}_T{args.temp:.2f}_size{args.min_size}"
    save_stack(outdir / f"mask_spine_{tag}_thr{thr:.2f}.tif", raw_masks, np.uint8)

    d_mask_binary = (prob_d >= args.thr).astype(np.uint8)
    d_mask_filtered = filter_small_objects_3d(d_mask_binary, min_size_voxels=args.min_size)
    save_stack(outdir / f"mask_dendrite_{tag}_thr{args.thr:.2f}.tif",
               (d_mask_filtered * 255).astype(np.uint8), np.uint8)
    

if __name__ == "__main__":
    main()

# Usage example:
'''
python punet_inference.py \
    --weights output/models/best_model_f1_*.pth \
    --tif dataset/DeepD3_Benchmark.tif \
    --out inference_results/ \
'''