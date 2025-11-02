# DeepD3.V2: Uncertainity Aware Instance Segmentation of Dendritic Spines

**3D microscopy spine detection using ensemble deep learning with uncertainty quantification**

## Overview

This project tackles the challenging problem of **automated dendritic spine detection in 3D microscopy images**. The solution combines multiple state-of-the-art deep learning approaches through an intelligent ensemble strategy:

### Key Features

-   **Multi-Model Ensemble**: YOLOv8 detection + Hierarchical Probabilistic U-Net segmentation
- **3D Analysis**: Native support for 3D TIFF stacks with spatial deduplication
- **Uncertainty Quantification**: Epistemic and aleatoric uncertainty maps
-  **Inter-rater Analysis**: Compare model predictions with human annotator disagreement

### Ensemble Method Explanation

The ensemble uses a **complementary fusion strategy** where:
1. **YOLOv8** provides fast, accurate bounding box detections
2. **Probabilistic U-Net** validates and refines detection confidence using segmentation probability maps
3. **Hierarchical Prob-UNet** (optional) provides dual-task learning for dendrite+spine segmentation

**Why This Works**: Object detection (YOLO) excels at localization while probabilistic segmentation (Prob-UNet) excels at providing segmentation with uncertainity quantification. Combining both achieves superior precision-recall tradeoff.

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Install Dependencies

```bash
# Clone repository
git clone https://github.com/munnasingh99/DeepD3.V2
cd DeepD3.V2

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Conda environment
conda create -n <env-name> python=3.10

# Install Pytorch with CUDA=11.8
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install requirements
pip install -r requirements.txt
```

##  Complete Project Structure

```
spine-detection/
├──  dataset/                      # Data files
│   ├── DeepD3_Training.d3set
│   ├── DeepD3_Validation.d3set
│   ├── DeepD3_Benchmark.tif
│   ├── Annotations_and_Clusters.csv
│   └── *.tif                        # Rater annotations
├──  yolo_dataset/                 # YOLO-formatted data
│   ├── images/
│   ├── labels/
│   └── spine_detection.yaml
│
├──  Training Scripts
│   ├── convert_d3set_to_yolo.py     # Dataset conversion
│   ├── yolov8_optimized_training.py # YOLO training
│   ├── training_punet.py            # Prob-UNet training
│   └── hpunet_main.py               # Hierarchical Prob-UNet
│
├──  Inference Scripts
│   ├── benchmark_eval_pipeline.py   # YOLO inference + eval
│   ├── punet_infer.py             # PUNet inference
│   ├── hpunet_infer.py             # HPUNet inference
│   └── yolo_prob_fusion.py         # Ensemble fusion
│
├──  Analysis Scripts
│   ├── uncertainity_maps.py        # Uncertainty quantification
│   ├── inter-rater.py              # Inter-rater agreement
│   ├── visualize_ensemble.py       # Visualization
│   └── dedup.py                    # 3D NMS utilities
│
├──  Model Architectures
│   ├── prob_unet_with_tversky.py   # Prob-UNet model
│   ├── hpunet_model.py             # Hierarchical model
│   └── unet_blocks.py              # U-Net components
│
├──  Utilities
│   ├── hpunet_datagen.py           # Data generators
│   ├── hpunet_train.py             # Training loops
│   └── utils.py                    # Helper functions
│
├──  Configuration
│   ├── requirements.txt            # Dependencies
│   ├── hpunet_config.json          # HPUNet config
│
└──  README.md                     # 
```

---
---

## Dataset Setup

### Required Data Files

The project uses the **DeepD3 dataset** format:

```
dataset/
├── DeepD3_Training.d3set           # Training data (images + masks)
├── DeepD3_Validation.d3set         # Validation data
├── DeepD3_Benchmark.tif            # Test volume (3D TIFF)
├── Annotations_and_Clusters.csv    # Ground truth annotations
├── Spine_U.tif                     # Rater U spine annotations
├── Spine_V.tif                     # Rater V spine annotations
├── Spine_W.tif                     # Rater W spine annotations
├── Dendrite_U_dendrite.tif         # Rater U dendrite annotations
├── Dendrite_V_dendrite.tif         # Rater V dendrite annotations
└── Dendrite_W_dendrite.tif         # Rater Z dendrite annotations
```

### Download Dataset

1. **Download DeepD3 dataset** from the official source
2. Place files in the `dataset/` directory as shown above
3. Verify file integrity:

```bash
python -c "import flammkuchen as fl; print('Training:', fl.load('dataset/DeepD3_Training.d3set').keys())"
```

### Data Format

- **.d3set files**: HDF5-based format containing image patches and spine masks
- **.tif files**: 3D TIFF stacks (Z, H, W) with 16-bit grayscale images
- **.csv files**: Annotations with columns `[x, y, z, cluster_id]`

---

## Model Training

### Step 1: Convert Dataset to YOLO Format

```bash
python convert_d3set_to_yolo.py \
    --train_d3set dataset/DeepD3_Training.d3set \
    --val_d3set dataset/DeepD3_Validation.d3set \
    --output_dir yolo_dataset 
```

**What this does**: Converts 3D spine masks to 2D YOLO bounding boxes with size filtering

**Expected output**:
```
yolo_dataset/
├── images/
│   ├── train/           # Training images
│   └── val/             # Validation images
├── labels/
│   ├── train/           # YOLO format labels
│   └── val/
└── spine_detection.yaml # Dataset configuration
```

---

### Step 2: Train YOLOv8 Detection Model

#### Training

```bash
python training_yolov8.py \
    --data yolo_dataset/spine_detection.yaml \
    --mode train
```

**Configuration highlights**:
- Image size: 1024×1024 (optimized for tiny objects)
- Batch size: 8
- Epochs: 60 with early stopping (patience=15)
- Optimizer: AdamW with cosine annealing
- Augmentations: Tailored for dense, small spines
- Adds an extra detection head at P2 (stride-4) for detecting smaller spines


**Model saves to**: `yolov8_training_output/optimized_v8l_p2_1024/weights/best.pt`


### Step 3: Train Probabilistic U-Net

#### Standard Prob-UNet with Tversky Loss

```bash
python training_punet.py \
    --train_data dataset/DeepD3_Training.d3set \
    --val_data dataset/DeepD3_Validation.d3set \
    --batch_size 16 \
    --epochs 60 \
    --lr 0.0003
```

**What this model does**: Produce spine and dendrite segmentation with uncertainity modelled.

**Model saves to**: `punet_training_output/models/best_model_f1_*.pth`

#### (Option): Hierarchical Prob-UNet (Dual Task: Dendrite + Spine)

```bash
python hpunet_main.py \
    --config hpunet_config.json \
    --epochs 120 \
    --bs 64 \
    --use_wandb
```

**Note**: Training of Hierarchical Prob-UNet is very unstable.
**What this model does**: Produces dendrite and spine segmentation with hierarchical latent variables

**Model saves to**: `hpunet_training_output/<timestamp>/dendrite_model_final.pth` and `spine_model_final.pth`

---

## Inference

### Individual Model Inference

#### YOLOv8 Inference with Parameter Sweep

```bash
python yolo_inference.py \
    --weights yolo_weights \        # Trained YOLOv8 weights
    --tif dataset/DeepD3_Benchmark.tif \
    --annotations dataset/Annotations_and_Clusters.csv \
    --out yolo_results/ \
    --thr 1000.0
```

**What this does**: 
- Runs inference at confidence/IoU thresholds
- Applies 3D spatial deduplication (1000nm radius)
- Evaluates precision/recall/F1
- Saves CSV prediction for given parameter combination

**Output structure**:
```
yolo_results/
├── predictions_conf0.10_iou0.55.csv.
└── metrics_summary.csv
```

#### YOLOv8 with Tiled Inference (Better for Large Images)

```bash
python yolo_inference.py \
    --weights yolo_weights.pt \     # Your trained YOLOv8 weights
    --tif dataset/DeepD3_Benchmark.tif \
    --out yolo_tiled_results/ \
    --tiling \
    --tile 640 \
    --overlap 0.5 \
    --wbf_iou 0.55
```

**What this does**: Processes image in overlapping tiles and fuses predictions using Weighted Box Fusion

---

#### Prob-UNet Inference


```bash
python punet_inference.py \
    --weights punet_weights \       # Your trained Prob-UNet weights
    --tif dataset/DeepD3_Benchmark.tif \
    --out inference_results/ \

```

**Output**: Probability maps and Prediction Maks for dendrites and spines
```
inference_results/
├── prob_dendrite.tif    # Dendrite probability map
└── prob_spine.tif   # Spine probability map
├── mask_dendrite_prior_S24_T1.40_size40_thr0.50.tif   # Dendrite prediction mask
└── mask_spine_prior_S24_T1.40_size40_thr0.50.tif   # Spine prediction mask

```
#### HProb-UNet Inference 

```bash
python hpunet_infer.py \
    --tif_path dataset/DeepD3_Benchmark.tif \
    --output_dir hpunet_results/ \
    --config hpunet_config.json \
    --dendrite_model hpunet_dendrite_weights \ # Your trained HProb-UNet spine weights
    --spine_model hpunet_spine_weights   # Your trained HProb-UNet spine weights
```
### Output: Probability maps for dendrite and spines
```
hpunet_results/
├── H_Prob_UNet_Dend.tiff    # Dendrite probability map
└── H_Prob_UNet_Spine.tiff   # Spine probability map
```
---

## Ensemble Fusion

### YOLO + Prob-UNet Intelligent Fusion

This is the **recommended** approach that combines the strengths of both models:

```bash
python yolo_prob_fusion.py \
    --yolo_weights yolo_weights.pt \     # Your trained YOLOv8 weights
    --punet_weights punet_weights \      # Your trained Prob-UNet weights 
    --tif dataset/DeepD3_Benchmark.tif \
    --annotations dataset/Annotations_and_Clusters.csv \
    --out fusion_results/ \
    --save_vis
```

### Fusion Strategy Explained

The fusion uses a **decision tree approach**:

1. **High Confidence (YOLO ≥ 0.50)** →  Auto-keep (trusted detection)
2. **Low Confidence (YOLO < 0.08)** →  Auto-reject (likely false positive)
3. **Medium Confidence (0.08 ≤ YOLO < 0.50)**:
   - Get Prob-UNet probability in bounding box region
   - If Prob-UNet ≥ 0.35 →  Keep with fused score
   - If Prob-UNet < 0.35 →  Reject

**Fused Score** = `(0.45 × YOLO_score + 0.55 × ProbUNet_prob) / 1.0`

**Why this works**: Prob-UNet acts as a **validator**, reducing false positives while maintaining high recall

**Output structure**:
```
fusion_results/
├── predictions_raw.csv       # All kept detections before 3D dedup
├── predictions_dedup.csv     # Final detections after 3D NMS
├── results.json              # Precision, Recall, F1 metrics
└── vis/                      # Visualization overlays (if --save_vis)
    ├── z000.png
    ├── z010.png
    └── ...
```

---

## Results

### Expected Output Format

**predictions_dedup.csv**:
```csv
z,x1,y1,x2,y2,score,status
0,145.2,302.1,158.7,315.6,0.7234,high
0,487.3,122.9,501.2,136.8,0.6123,validated
1,201.5,445.2,215.3,459.1,0.5892,validated
...
```

**Columns**:
- `z`: Slice index (0-based)
- `x1, y1, x2, y2`: Bounding box coordinates
- `score`: Final detection confidence
- `status`: `high` (YOLO-only) or `validated` (fusion)

### Interpreting Results

**results.json** contains:
```json
{
  "precision": 0.8542,
  "recall": 0.7891,
  "f1_score": 0.8203,
  "detections": 1247,
  "statistics": {
    "yolo_total": 1589,
    "kept_high_conf": 723,
    "kept_validated": 524,
    "rejected_low_yolo": 189,
    "rejected_no_punet": 153
  }
}
```

**Metrics**:
- **Precision**: % of predictions that match ground truth (within 1000nm)
- **Recall**: % of ground truth spines detected
- **F1 Score**: Harmonic mean of precision and recall

---

##  Uncertainty Quantification

### Generate Uncertainty Maps

```bash
python uncertainity_maps.py \
    --target spine \
    --weights punet_weights \      # Your trained Prob-UNet weights \
    --tif dataset/DeepD3_Benchmark.tif \
    --outdir uncertainty_outputs/ \
    --samples 24 \
    --temp 1.4 \
    --visualize \
    --inter-rater dataset/inter_rater_disagreement_spine.tif
```

**What this generates**:
- **Predictive Mean**: Average prediction across samples
- **Predictive Variance**: Total uncertainty
- **Epistemic Uncertainty**: Model uncertainty (reducible with more data)
- **Aleatoric Uncertainty**: Data uncertainty (irreducible noise)
- **Mutual Information**: Information gained per sample
- **Inter-rater Comparison**: Compare with human annotator disagreement

**Output**:
```
uncertainty_outputs/spine_uncertainty/
├── mean_spine_prior_S24_T1.40.tif
├── variance_spine_prior_S24_T1.40.tif
├── entropy_spine_prior_S24_T1.40.tif
├── mi_epistemic_spine_prior_S24_T1.40.tif
├── aleatoric_spine_prior_S24_T1.40.tif
└── visualizations/
    ├── uncertainty_panel_spine_*.png
    └── uncertainty_panel_spine_*.pdf
```

### Compute Inter-rater Disagreement

```bash
python inter-rater.py \
    --rater-u-spine dataset/Spine_U.tif \
    --rater-v-spine dataset/Spine_V.tif \
    --rater-w-spine dataset/Spine_W.tif \
    --rater-u-dendrite dataset/Dendrite_U_dendrite.tif \
    --rater-y-dendrite dataset/Dendrite_V_dendrite.tif \
    --rater-z-dendrite dataset/Dendrite_W_dendrite.tif \
    --output-spine dataset/inter_rater_disagreement_spine.tif \
    --output-dendrite dataset/inter_rater_disagreement_dendrite.tif
```

**What this does**: Computes per-pixel variance across human raters (0 = unanimous, 0.22 = 2-vs-1 split)

---

##  Visualization

### Visualize Predictions on TIFF Volume

```bash
python visualize_ensemble.py \
    --tif dataset/DeepD3_Benchmark.tif \
    --predictions fusion_results/predictions_dedup.csv \
    --out visualizations/ \
    --mode dots \
    --dot_radius 4 \
    --dot_color 255,0,0 \
    --slice_step 1 \
    --save_stacked
```

**Visualization modes**:
- `dots`: Red dots at detection centers (recommended)
- `boxes`: Green bounding boxes
- `both`: Dots + boxes

**Options**:
- `--enhance_contrast`: Apply CLAHE for better visibility
- `--show_score`: Display confidence scores
- `--slice_start/--slice_end`: Process specific Z range
- `--save_stacked`: Save as multi-page TIFF

**Output**:
```
visualizations/
├── slice_0000.png
├── slice_0001.png
├── ...
└── overlay_stack.tif  # Stacked TIFF (if --save_stacked)
```

---


##  Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Solution**: Reduce batch size
```bash
# For YOLO
python yolov8_optimized_training.py --data ... --batch 4  # Default: 8

# For Prob-UNet
python training_punet.py --batch_size 8  # Default: 16
```

#### 2. Dataset Files Not Found

**Check**:
```python
import os
print(os.path.exists('dataset/DeepD3_Training.d3set'))
```

**Solution**: Verify dataset placement and paths in config files

#### 3. Model Loading Errors

**Common cause**: Model architecture mismatch

**Solution**: Ensure inference script uses same architecture as training
```python
# Check model parameters match training config
model = ProbabilisticUnetDualLatent(
    num_filters=[32, 64, 128, 192],  # Must match training!
    latent_dim_spine=12,
    ...
)
```

#### 4. Low Recall in Results

**Possible causes**:
- Confidence threshold too high
- 3D deduplication radius too small
- Insufficient training

**Solution**: Adjust parameters:
```bash
python yolo_prob_fusion.py \
    --yolo_conf 0.01 \          # Lower confidence threshold
    --yolo_low_conf 0.05 \      # More permissive
    --dedup_nm 800.0            # Smaller dedup radius
```

#### 5. High False Positive Rate

**Solution**: Increase validation value
```bash
python yolo_prob_fusion.py \
    --yolo_high_conf 0.60 \     # Stricter high conf threshold
    --punet_validation 0.45 \   # Higher Prob-UNet requirement
    --weight_punet 0.65         # Trust Prob-UNet more
```

---

##  Advanced Usage

### Hyperparameter Tuning

Create a sweep configuration:

```python
# sweep_config.yaml
program: yolo_prob_fusion.py
method: bayes
metric:
  name: f1_score
  goal: maximize
parameters:
  yolo_conf: {min: 0.01, max: 0.10}
  punet_validation: {min: 0.20, max: 0.50}
  weight_yolo: {min: 0.30, max: 0.60}
```

Run sweep:
```bash
wandb sweep sweep_config.yaml
wandb agent <sweep_id>
```

### Tiled Inference for Memory Efficiency

For very large volumes:
```bash
python yolo_prob_fusion.py \
    ... \
    --tiling \
    --tile 512 \
    --overlap 0.40
```

### Batch Processing Multiple Volumes

```bash
#!/bin/bash
for tif in data/*.tif; do
    python yolo_prob_fusion.py \
        --tif "$tif" \
        --out "results/$(basename "$tif" .tif)/" \
        ...
done
```



---

## Tips for Best Results

1. **Start with YOLOv8 alone** to establish baseline performance
2. **Train Prob-UNet separately** and validate uncertainty maps
3. **Tune fusion parameters** on a validation set before final evaluation
4. **Visualize predictions** frequently to catch issues early
5. **Compare with inter-rater disagreement** to understand model limitations
6. **Use uncertainty maps** to identify regions needing manual review

---

## Support

## Contact Information

Munna Prithvinath Singh \
Friedrich Alexander University, Erlangen-Nürnberg \
<mailto:singhmunna0786@gmail.com>\
\
Prof. Dr. Andreas Kist \
Friedrich Alexander University, Erlangen-Nürnberg

---

## Acknowledgments

- DeepD3 creators
- Ultralytics (YOLOv8)
- Probabilistic U-Net original authors
- Hierarchical Probabilistic U-Net original authors

---

**Happy spine detecting! 🧠**