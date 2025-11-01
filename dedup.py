import numpy as np
import pandas as pd

def dedup_3d_predictions(csv_path, d=np.array([94,94,200], dtype=np.float32), thr=1000.0):
    """
    Greedy 3D NMS over detection centers using the same metric as your evaluator.
    Keeps highest-score detection and removes others within distance 'thr'.
    """
    df = pd.read_csv(csv_path)
    # centers in pixels + z index
    xc = 0.5*(df.x1 + df.x2).to_numpy(np.float32)
    yc = 0.5*(df.y1 + df.y2).to_numpy(np.float32)
    zc = df.z.to_numpy(np.float32)
    s  = df.score.to_numpy(np.float32)

    idx = np.argsort(-s)  # highest score first
    keep = []
    taken = np.zeros(len(df), dtype=bool)

    for i in idx:
        if taken[i]: 
            continue
        keep.append(i)
        # compute scaled distances to remaining
        dx = (xc - xc[i]) * d[0]
        dy = (yc - yc[i]) * d[1]
        dz = (zc - zc[i]) * d[2]
        dist = np.sqrt(dx*dx + dy*dy + dz*dz)
        close = dist <= thr
        taken |= close  # suppress all within radius

    return df.iloc[keep].reset_index(drop=True)

# Example usage:
# dedup = dedup_3d_predictions("predictions_benchmark_boxes.csv", thr=1000.0)
# dedup.to_csv("predictions_benchmark_boxes_dedup.csv", index=False)
# Then pass dedup centers (x,y,z) to Metrics(prediction=...)
