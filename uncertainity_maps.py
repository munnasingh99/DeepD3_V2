#!/usr/bin/env python3
"""


Generates and visualizes epistemic and aleatoric uncertainty for spine/dendrite predictions.
Combines uncertainty computation and visualization in one streamlined workflow.

"""

import argparse
from pathlib import Path
import numpy as np
import torch
import tifffile as tiff
import imageio as io
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from prob_unet_with_tversky import ProbabilisticUnetDualLatent



def minmax01(im):
    """Min-max normalization to [0,1]"""
    vmin, vmax = float(im.min()), float(im.max())
    if vmax > vmin:
        im = (im - vmin) / (vmax - vmin)
    else:
        im = np.zeros_like(im, dtype=np.float32)
    return im.astype(np.float32)


@torch.no_grad()
def multisample_logits(model, device, img2d, S=16, use_posterior=False, temperature=1.0, target="spine"):
    """
    Generate S samples of logits for uncertainty quantification.
    
    Returns:
        logits: torch.Tensor of shape [S, H, W]
    """
    H, W = img2d.shape
    pad_h = (32 - (H % 32)) % 32
    pad_w = (32 - (W % 32)) % 32
    
    x_np = np.pad(img2d, ((0, pad_h), (0, pad_w)), mode="reflect") if (pad_h or pad_w) else img2d
    x = torch.from_numpy(x_np)[None, None].to(device)

    model.forward(x, training=False)

    d_logits, s_logits = [], []
    for _ in range(max(1, S)):
        ld, ls = model.sample(testing=True, use_posterior=use_posterior)
        d_logits.append(ld)
        s_logits.append(ls)

    d = torch.stack(d_logits, 0).squeeze(1).squeeze(1)  # [S, H', W']
    s = torch.stack(s_logits, 0).squeeze(1).squeeze(1)

    # Apply temperature scaling
    d = d / temperature
    s = s / temperature

    # Remove padding
    if pad_h or pad_w:
        d = d[..., :H, :W]
        s = s[..., :H, :W]

    return s if target == "spine" else d


def entropy(p, eps=1e-7):
    """Binary entropy: -[p*log(p) + (1-p)*log(1-p)]"""
    p = torch.clamp(p, eps, 1 - eps)
    return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))


def compute_uncertainty_maps(model, device, volume, target="spine", S=16, use_posterior=False, temperature=1.0):
    """
    Compute full uncertainty decomposition for a 3D volume.
    
    Returns:
        dict with keys: 'mean', 'variance', 'entropy', 'mi', 'aleatoric'
    """
    Z, H, W = volume.shape
    
    meanP = np.zeros((Z, H, W), np.float32)
    varP = np.zeros((Z, H, W), np.float32)
    Hbar = np.zeros((Z, H, W), np.float32)  # predictive entropy (total)
    MI = np.zeros((Z, H, W), np.float32)    # mutual information (epistemic)
    
    for z in tqdm(range(Z), desc=f"Computing {target} uncertainty"):
        img = minmax01(volume[z].astype(np.float32))
        
        logits = multisample_logits(model, device, img, S=S,
                                    use_posterior=use_posterior,
                                    temperature=temperature,
                                    target=target)
        p = torch.sigmoid(logits)  # [S, H, W]
        
        pbar = p.mean(0)                    # E[p] - predictive mean
        v = p.var(0, unbiased=False)        # Var[p] - total variance
        H_pbar = entropy(pbar)              # H[E[p]] - entropy of mean
        H_p = entropy(p).mean(0)            # E[H[p]] - expected entropy
        mi = H_pbar - H_p                   # MI ≈ epistemic uncertainty
        
        meanP[z] = pbar.cpu().numpy()
        varP[z] = v.cpu().numpy()
        Hbar[z] = H_pbar.cpu().numpy()
        MI[z] = mi.cpu().numpy()
    
    aleatoric = np.clip(Hbar - MI, 0.0, None)  # approximation
    
    return {
        'mean': meanP,
        'variance': varP,
        'entropy': Hbar,
        'mi': MI,
        'aleatoric': aleatoric
    }



def add_cbar_left(fig, ax, im, label):
    """Add colorbar on the left side of axis"""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("left", size="3.5%", pad=0.04)
    cb = fig.colorbar(im, cax=cax, ticklocation='left')
    cb.ax.set_ylabel(label, rotation=90, labelpad=10)
    cb.ax.tick_params(labelsize=9, length=2)
    return cb


def add_cbar_right(fig, ax, im, label):
    """Add colorbar on the right side of axis"""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.04)
    cb = fig.colorbar(im, cax=cax)
    cb.ax.set_ylabel(label, rotation=270, labelpad=10)
    cb.ax.tick_params(labelsize=9, length=2)
    return cb


def integral_image(img):
    """Compute integral image for fast box filtering"""
    return np.cumsum(np.cumsum(img, axis=0), axis=1)


def box_sum(ii, y0, x0, y1, x1):
    """Fast box sum using integral image"""
    A = ii[y1-1, x1-1]
    B = ii[y0-1, x1-1] if y0 > 0 else 0.0
    C = ii[y1-1, x0-1] if x0 > 0 else 0.0
    D = ii[y0-1, x0-1] if (y0 > 0 and x0 > 0) else 0.0
    return A - B - C + D


def find_best_window(uncertainty_map, win_h, win_w, stride=16):
    """Find window with maximum mean uncertainty"""
    H, W = uncertainty_map.shape
    ii = integral_image(uncertainty_map)
    best_v, best_xy = -1.0, (0, 0)
    
    for y in range(0, max(1, H - win_h + 1), stride):
        for x in range(0, max(1, W - win_w + 1), stride):
            y1, x1 = y + win_h, x + win_w
            if y1 > H or x1 > W:
                continue
            mean_val = box_sum(ii, y, x, y1, x1) / (win_h * win_w)
            if mean_val > best_v:
                best_v, best_xy = mean_val, (y, x)
    
    return best_xy


def visualize_uncertainty(raw_volume, uncertainty_maps, target, tag, outdir, 
                         inter_rater_path=None, win_size=(180, 360), stride=16):
    """
    Create comprehensive uncertainty visualization panel.
    
    Args:
        raw_volume: Raw image volume (Z, H, W)
        uncertainty_maps: Dict with 'mean', 'variance', 'entropy', 'mi', 'aleatoric'
        target: 'spine' or 'dendrite'
        tag: Descriptive tag for filename
        outdir: Output directory
        inter_rater_path: Optional path to inter-rater disagreement TIF
        win_size: (height, width) for zoomed view
        stride: Stride for window search
    """
    mean = uncertainty_maps['mean']
    var = uncertainty_maps['variance']
    ent = uncertainty_maps['entropy']
    mi = uncertainty_maps['mi']
    ale = uncertainty_maps['aleatoric']
    inter = np.asarray(io.mimread(inter_rater_path)) if inter_rater_path else None
    flag = "inter"  # select different criterion for slice selection example : "mi" for mutual information, "inter" for inter-rater disagreement
    
    # Load inter-rater disagreement if provided
    if flag == "inter": 
        z = int(np.argmax(inter.mean(axis=(1, 2))))
        print(f"Selected slice z={z} (max inter-rater disagreement)")
    else:
        # Find slice with max MI (epistemic uncertainty)
        z = int(np.argmax(mi.mean(axis=(1, 2))))
        inter = None
        print(f"Selected slice z={z} (max epistemic uncertainty)")
    
    # Find best window
    win_h, win_w = win_size
    reference = inter[z] if inter is not None else mi[z]
    y0, x0 = find_best_window(reference, win_h, win_w, stride=stride)
    y1, x1 = y0 + win_h, x0 + win_w
    print(f"Window: y={y0}:{y1}, x={x0}:{x1}")
    
    # Crop all maps
    raw_c = raw_volume[z, y0:y1, x0:x1]
    mean_c = mean[z, y0:y1, x0:x1]
    var_c = var[z, y0:y1, x0:x1]
    ent_c = ent[z, y0:y1, x0:x1]
    mi_c = mi[z, y0:y1, x0:x1]
    ale_c = ale[z, y0:y1, x0:x1]
    inter_c = inter[z, y0:y1, x0:x1] if inter is not None else None
    
    raw_disp = minmax01(raw_c)
    mask_c = (mean_c >= 0.5).astype(np.float32)
    
    # Fixed scales for consistency
    H_max = np.log(2.0)  # max binary entropy ≈ 0.693 nats
    scales = {
        'mean': (0.0, 1.0),
        'var': (0.0, 0.25),
        'ent': (0.0, H_max),
        'mi': (0.0, H_max),
        'ale': (0.0, H_max),
        'inter': (0.0, 2.0/9.0)  # 3 raters: max var at 2-vs-1 split
    }
    
    # Setup plot
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.linewidth": 0.8,
    })
    
    num_rows = 4 if inter_c is not None else 4
    fig, axes = plt.subplots(num_rows, 2, figsize=(10, 10))
    axs = axes.ravel()
    
    idx = 0
    
    # 0) Raw image
    axs[idx].imshow(raw_disp, cmap="gray", interpolation="nearest")
    axs[idx].set_title(f"Stack (z={z}) - {target}")
    axs[idx].axis("off")
    idx += 1
    
    # 1) Prediction mask
    axs[idx].imshow(raw_disp, cmap="gray", interpolation="nearest")
    axs[idx].imshow(mask_c, vmin=0, vmax=1, cmap="inferno", alpha=0.55, interpolation="nearest")
    axs[idx].set_title("Prediction mask (thr=0.50)")
    axs[idx].axis("off")
    idx += 1
    
    # 2) Predictive mean
    axs[idx].imshow(raw_disp, cmap="gray", interpolation="nearest")
    im2 = axs[idx].imshow(mean_c, vmin=scales['mean'][0], vmax=scales['mean'][1], 
                          cmap="inferno", alpha=0.55, interpolation="nearest")
    axs[idx].set_title("Predictive mean")
    axs[idx].axis("off")
    add_cbar_left(fig, axs[idx], im2, "mean")
    idx += 1
    
    # 3) Predictive variance
    axs[idx].imshow(raw_disp, cmap="gray", interpolation="nearest")
    im3 = axs[idx].imshow(var_c, vmin=scales['var'][0], vmax=scales['var'][1],
                          cmap="inferno", alpha=0.55, interpolation="nearest")
    axs[idx].set_title("Predictive variance (total)")
    axs[idx].axis("off")
    add_cbar_right(fig, axs[idx], im3, "var")
    idx += 1
    
    # 4) Predictive entropy
    axs[idx].imshow(raw_disp, cmap="gray", interpolation="nearest")
    im4 = axs[idx].imshow(ent_c, vmin=scales['ent'][0], vmax=scales['ent'][1],
                          cmap="inferno", alpha=0.55, interpolation="nearest")
    axs[idx].set_title("Predictive entropy (total)")
    axs[idx].axis("off")
    add_cbar_left(fig, axs[idx], im4, "H (nats)")
    idx += 1
    
    # 5) Inter-rater disagreement (if available)
    if inter_c is not None:
        axs[idx].imshow(raw_disp, cmap="gray", interpolation="nearest")
        im5 = axs[idx].imshow(inter_c, vmin=scales['inter'][0], vmax=scales['inter'][1],
                              cmap="inferno", alpha=0.55, interpolation="nearest")
        axs[idx].set_title("Inter-rater disagreement")
        axs[idx].axis("off")
        add_cbar_right(fig, axs[idx], im5, "var (raters)")
    else:
        axs[idx].axis("off")
    idx += 1
    
    # 6) Mutual information (epistemic)
    axs[idx].imshow(raw_disp, cmap="gray", interpolation="nearest")
    im6 = axs[idx].imshow(mi_c, vmin=scales['mi'][0], vmax=scales['mi'][1],
                          cmap="inferno", alpha=0.55, interpolation="nearest")
    axs[idx].set_title("Mutual information (epistemic)")
    axs[idx].axis("off")
    add_cbar_left(fig, axs[idx], im6, "MI (nats)")
    idx += 1
    
    # 7) Aleatoric uncertainty
    axs[idx].imshow(raw_disp, cmap="gray", interpolation="nearest")
    im7 = axs[idx].imshow(ale_c, vmin=scales['ale'][0], vmax=scales['ale'][1],
                          cmap="inferno", alpha=0.55, interpolation="nearest")
    axs[idx].set_title("Aleatoric uncertainty")
    axs[idx].axis("off")
    add_cbar_right(fig, axs[idx], im7, "H−MI (nats)")
    
    plt.subplots_adjust(wspace=0.06, hspace=0.20, left=0.03, right=0.98, top=0.95, bottom=0.04)
    
    # Save
    vis_dir = Path(outdir) / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    out_png = vis_dir / f"uncertainty_panel_{target}_{tag}.png"
    out_pdf = vis_dir / f"uncertainty_panel_{target}_{tag}.pdf"
    
    plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    print(f"Saved visualization: {out_png}")
    print(f"Saved visualization: {out_pdf}")
    plt.close()



def main():
    parser = argparse.ArgumentParser(
        description="Generate and visualize uncertainty maps for spine or dendrite predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--target", type=str, choices=["spine", "dendrite"], required=True,
                       help="Target structure: 'spine' or 'dendrite'")
    parser.add_argument("--weights", type=str, required=True,
                       help="Path to model checkpoint (.pth)")
    parser.add_argument("--tif", type=str, required=True,
                       help="Path to input TIFF volume")
    
    # Optional arguments
    parser.add_argument("--outdir", type=str, default="uncertainty_outputs",
                       help="Output directory (default: uncertainty_outputs)")
    parser.add_argument("--posterior", action="store_true",
                       help="Use posterior sampling (requires training mode)")
    parser.add_argument("--samples", type=int, default=24,
                       help="Number of Monte Carlo samples (default: 24)")
    parser.add_argument("--temp", type=float, default=1.4,
                       help="Temperature for sampling (default: 1.0)")
    parser.add_argument("--visualize", action="store_true",
                       help="Generate visualization panel")
    parser.add_argument("--inter-rater", type=str, default=None,
                       help="Path to inter-rater disagreement TIF for comparison")
    parser.add_argument("--window-size", type=int, nargs=2, default=[180, 360],
                       help="Window size (H W) for visualization (default: 180 360)")
    
    args = parser.parse_args()
    
    # Setup
    outdir = Path(args.outdir) / f"{args.target}_uncertainty"
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 80)
    print(f"Uncertainty Map Generation")
    print("=" * 80)
    print(f"Target:      {args.target}")
    print(f"Device:      {device}")
    print(f"Samples:     {args.samples}")
    print(f"Temperature: {args.temp}")
    print(f"Posterior:   {args.posterior}")
    print(f"Output:      {outdir}")
    print("=" * 80)
    
    # Load model
    print("\nLoading model...")
    model = ProbabilisticUnetDualLatent(
        input_channels=1, num_classes=1,
        num_filters=[32, 64, 128, 192],
        latent_dim_dendrite=12, latent_dim_spine=12,
        no_convs_fcomb=4,
        recon_loss="tversky", 
        tversky_alpha=0.5, tversky_beta=0.5, tversky_gamma=1.0,
        beta_dendrite=1.0, beta_spine=1.0
    ).to(device).eval()
    
    ckpt = torch.load(args.weights, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    print("Model loaded successfully")
    
    # Load volume
    print(f"\nLoading volume: {args.tif}")
    vol = tiff.imread(args.tif)
    if vol.ndim == 2:
        vol = vol[None, ...]
    Z, H, W = vol.shape
    print(f"Volume shape: Z={Z}, H={H}, W={W}")
    
    # Compute uncertainty maps
    print(f"\nComputing uncertainty maps...")
    uncertainty_maps = compute_uncertainty_maps(
        model, device, vol,
        target=args.target,
        S=args.samples,
        use_posterior=args.posterior,
        temperature=args.temp
    )
    
    # Save uncertainty maps
    tag = f"{args.target}_{'post' if args.posterior else 'prior'}_S{args.samples}_T{args.temp:.2f}"
    print(f"\nSaving uncertainty maps with tag: {tag}")
    
    tiff.imwrite((outdir / f"mean_{tag}.tif").as_posix(), 
                 uncertainty_maps['mean'].astype(np.float32))
    tiff.imwrite((outdir / f"variance_{tag}.tif").as_posix(), 
                 uncertainty_maps['variance'].astype(np.float32))
    tiff.imwrite((outdir / f"entropy_{tag}.tif").as_posix(), 
                 uncertainty_maps['entropy'].astype(np.float32))
    tiff.imwrite((outdir / f"mi_epistemic_{tag}.tif").as_posix(), 
                 uncertainty_maps['mi'].astype(np.float32))
    tiff.imwrite((outdir / f"aleatoric_{tag}.tif").as_posix(), 
                 uncertainty_maps['aleatoric'].astype(np.float32))
    
    print(f"\nSaved maps:")
    print(f"Mean:       {outdir / f'mean_{tag}.tif'}")
    print(f"Variance:   {outdir / f'variance_{tag}.tif'}")
    print(f"Entropy:    {outdir / f'entropy_{tag}.tif'}")
    print(f"MI:         {outdir / f'mi_epistemic_{tag}.tif'}")
    print(f"Aleatoric:  {outdir / f'aleatoric_{tag}.tif'}")

    # Generate visualization
    if args.visualize:
        print(f"\nGenerating visualization...")
        visualize_uncertainty(
            vol, uncertainty_maps, args.target, tag, outdir,
            inter_rater_path=args.inter_rater,
            win_size=tuple(args.window_size),
            stride=16
        )

    print("\nComplete!")



if __name__ == "__main__":
    main()


# Usage:
"""
python3 uncertainty_maps.py \
    --target spine \
    --weights best_model.pth \
    --tif input.tif \
    --visualize \
    --inter-rater inter_rater_disagreement_spine.tif \
    --samples 32
"""