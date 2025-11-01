
"""
compute_inter_rater_disagreement.py
Computes per-pixel variance across multiple rater annotations.
"""

import numpy as np
import imageio as io
from pathlib import Path

def load_stack(path):
    arr = np.asarray(io.mimread(path))
    if arr.ndim == 2:
        arr = arr[None, ...]
    return arr.astype(np.float32)

def compute_disagreement(rater_paths):
    """
    Compute per-pixel variance across raters.
    
    Parameters:
    -----------
    rater_paths : list of str
        Paths to binary annotation stacks (one per rater)
    
    Returns:
    --------
    disagreement : np.ndarray
        Per-pixel variance (0 = unanimous, 0.33 = 2-vs-1 split)
    """
    # Load all raters
    raters = [load_stack(path) for path in rater_paths]
    
    # Stack along new axis: (num_raters, Z, H, W)
    rater_stack = np.stack(raters, axis=0)
    
    # Binarize (in case not already binary)
    rater_stack = (rater_stack > 0.5).astype(np.float32)
    
    # Compute variance along rater axis
    # Variance of Bernoulli: p(1-p), max at p=0.5 → variance = 0.25
    # For 3 raters: var = 0.33 when 2-vs-1, var = 0 when unanimous
    disagreement = np.var(rater_stack, axis=0)
    
    return disagreement

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--rater-u-spine', default="../dataset/Spine_U.tif", help='Rater U spine annotations')
    parser.add_argument('--rater-v-spine', default="../dataset/Spine_V.tif", help='Rater V spine annotations')
    parser.add_argument('--rater-w-spine', default="../dataset/Spine_W.tif", help='Rater W spine annotations')
    parser.add_argument('--rater-u-dendrite', default="../dataset/Dendrite_U_dendrite.tif", help='Rater U dendrite annotations')
    parser.add_argument('--rater-y-dendrite', default="../dataset/Dendrite_V_dendrite.tif", help='Rater Y dendrite annotations')
    parser.add_argument('--rater-z-dendrite', default="../dataset/Dendrite_W_dendrite.tif", help='Rater Z dendrite annotations')

    parser.add_argument('--output-dendrite', default='../dataset/inter_rater_disagreement_dendrite.tif')
    parser.add_argument('--output-spine', default='../dataset/inter_rater_disagreement_spine.tif')
    args = parser.parse_args()
    
    print("Computing inter-rater disagreement...")
    disagreement_spine = compute_disagreement([args.rater_u_spine, args.rater_v_spine, args.rater_w_spine])
    disagreement_dendrite = compute_disagreement([args.rater_u_dendrite, args.rater_y_dendrite, args.rater_z_dendrite])

    print(f"Disagreement Spine statistics:")
    print(f"  Mean: {disagreement_spine.mean():.4f}")
    print(f"  Max:  {disagreement_spine.max():.4f}")
    print(f"  % Unanimous (var=0): {100*(disagreement_spine==0).sum()/disagreement_spine.size:.2f}%")
    print(f"  % Contested (var>0.2): {100*(disagreement_spine>0.2).sum()/disagreement_spine.size:.2f}%")
    print(f"Disagreement Dendrite statistics:")
    print(f"  Mean: {disagreement_dendrite.mean():.4f}")
    print(f"  Max:  {disagreement_dendrite.max():.4f}")
    print(f"  % Unanimous (var=0): {100*(disagreement_dendrite==0).sum()/disagreement_dendrite.size:.2f}%")
    print(f"  % Contested (var>0.2): {100*(disagreement_dendrite>0.2).sum()/disagreement_dendrite.size:.2f}%")
    
    # Save as 32-bit float TIFF
    output_path_dendrite = Path(args.output_dendrite)
    output_path_dendrite.parent.mkdir(parents=True, exist_ok=True)
    output_path_spine = Path(args.output_spine)
    output_path_spine.parent.mkdir(parents=True, exist_ok=True)

    # Convert to list of 2D arrays for mimwrite
    slices = [disagreement_spine[z] for z in range(disagreement_spine.shape[0])]
    io.mimwrite(args.output_spine, slices)
    print(f"Saved to {args.output_spine}")

    slices = [disagreement_dendrite[z] for z in range(disagreement_dendrite.shape[0])]
    io.mimwrite(args.output_dendrite, slices)
    print(f"Saved to {args.output_dendrite}")