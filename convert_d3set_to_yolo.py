"""
convert_d3set_to_yolo.py

Converts DeepD3 .d3set files to YOLO detection format.
Handles 3D spine masks → 2D bounding boxes with proper filtering and validation.
"""

import flammkuchen as fl
import cv2
import numpy as np
from pathlib import Path
from scipy import ndimage
from tqdm import tqdm
import argparse
from typing import List, Tuple, Dict
import json


class D3setToYOLOConverter:
    def __init__(self, min_spine_pixels: int = 10, max_spine_pixels: int = 5000):
        """
        Args:
            min_spine_pixels: Minimum pixels for valid spine (filter noise)
            max_spine_pixels: Maximum pixels for valid spine (filter artifacts)
        """
        self.min_spine_pixels = min_spine_pixels
        self.max_spine_pixels = max_spine_pixels
        self.stats = {
            'total_stacks': 0,
            'total_planes': 0,
            'total_spines': 0,
            'filtered_too_small': 0,
            'filtered_too_large': 0,
            'empty_planes': 0
        }
    
    def extract_bboxes_from_mask(self, spine_mask_2d: np.ndarray, 
                                  img_height: int, img_width: int) -> List[List[float]]:
        """
        Extract YOLO format bounding boxes from 2D spine mask.
        
        Args:
            spine_mask_2d: Binary mask (H, W) with spine annotations
            img_height: Image height for normalization
            img_width: Image width for normalization
            
        Returns:
            List of boxes in YOLO format: [class_id, x_center, y_center, width, height]
            All coordinates normalized to [0, 1]
        """
        if spine_mask_2d.sum() == 0:
            return []
        
        # Find connected components (individual spines)
        labeled_mask, num_spines = ndimage.label(spine_mask_2d)
        
        boxes = []
        for spine_id in range(1, num_spines + 1):
            spine_pixels = (labeled_mask == spine_id)
            pixel_count = spine_pixels.sum()
            
            # Filter by size
            if pixel_count < self.min_spine_pixels:
                self.stats['filtered_too_small'] += 1
                continue
            if pixel_count > self.max_spine_pixels:
                self.stats['filtered_too_large'] += 1
                continue
            
            # Get bounding box coordinates
            coords = np.where(spine_pixels)
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            
            # Convert to YOLO format (normalized)
            x_center = ((x_min + x_max) / 2) / img_width
            y_center = ((y_min + y_max) / 2) / img_height
            width = (x_max - x_min + 1) / img_width
            height = (y_max - y_min + 1) / img_height
            
            # Clamp to valid range [0, 1]
            x_center = np.clip(x_center, 0, 1)
            y_center = np.clip(y_center, 0, 1)
            width = np.clip(width, 0, 1)
            height = np.clip(height, 0, 1)
            
            boxes.append([0, x_center, y_center, width, height])  # class_id=0 for spine
            self.stats['total_spines'] += 1
        
        return boxes
    
    def normalize_image(self, img: np.ndarray) -> np.ndarray:
        """Normalize image to 0-255 range."""
        img_min, img_max = img.min(), img.max()
        if img_max - img_min < 1e-5:
            return np.zeros_like(img, dtype=np.uint8)
        normalized = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        return normalized
    
    def convert_d3set(self, d3set_path: str, output_dir: str, split: str = 'train',
                      save_visualization: bool = False) -> Dict:
        """
        Convert .d3set file to YOLO format dataset.
        
        Args:
            d3set_path: Path to .d3set file
            output_dir: Output directory for YOLO dataset
            split: 'train' or 'val'
            save_visualization: If True, save bbox visualization images
            
        Returns:
            Dictionary with conversion statistics
        """
        print(f"\n{'='*60}")
        print(f"Converting {d3set_path} to YOLO format ({split} split)")
        print(f"{'='*60}\n")
        
        # Load data
        print("Loading .d3set file...")
        try:
            data = fl.load(d3set_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load {d3set_path}: {e}")
        
        stacks = data['data']['stacks']
        spines = data['data']['spines']
        meta = data['meta']
        
        print(f"Found {len(meta)} stacks in dataset")
        
        # Create output directories
        output_dir = Path(output_dir)
        img_dir = output_dir / 'images' / split
        label_dir = output_dir / 'labels' / split
        img_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        
        if save_visualization:
            vis_dir = output_dir / 'visualizations' / split
            vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each stack
        self.stats['total_stacks'] = len(meta)
        
        for stack_idx in tqdm(range(len(meta)), desc="Processing stacks"):
            stack_key = f'x{stack_idx}'
            
            # Get stack and spine mask
            stack_3d = stacks[stack_key]  # Shape: (Z, H, W)
            spine_mask_3d = spines[stack_key]  # Shape: (Z, H, W)
            
            # Get metadata
            stack_meta = meta.iloc[stack_idx]
            resolution_xy = stack_meta.Resolution_XY if hasattr(stack_meta, 'Resolution_XY') else None
            
            # Process each z-plane
            for z in range(stack_3d.shape[0]):
                plane = stack_3d[z]
                spine_mask = spine_mask_3d[z]
                
                self.stats['total_planes'] += 1
                
                # Normalize image
                normalized_plane = self.normalize_image(plane)
                
                # Extract bounding boxes
                img_h, img_w = plane.shape
                boxes = self.extract_bboxes_from_mask(spine_mask, img_h, img_w)
                
                if len(boxes) == 0:
                    self.stats['empty_planes'] += 1
                
                # Generate filenames
                img_name = f'stack{stack_idx:04d}_z{z:03d}.png'
                label_name = f'stack{stack_idx:04d}_z{z:03d}.txt'
                
                # Save image
                cv2.imwrite(str(img_dir / img_name), normalized_plane)
                
                # Save labels
                with open(label_dir / label_name, 'w') as f:
                    for box in boxes:
                        f.write(' '.join(map(str, box)) + '\n')
                
                # Save visualization if requested
                if save_visualization and len(boxes) > 0:
                    vis_img = self._visualize_boxes(normalized_plane, boxes)
                    cv2.imwrite(str(vis_dir / img_name), vis_img)
        
        # Save conversion statistics
        stats_path = output_dir / f'{split}_conversion_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        self._print_statistics()
        
        return self.stats
    
    def _visualize_boxes(self, img: np.ndarray, boxes: List[List[float]]) -> np.ndarray:
        """Draw bounding boxes on image for visualization."""
        vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img_h, img_w = img.shape
        
        for box in boxes:
            _, x_center, y_center, width, height = box
            
            # Convert normalized coords to pixel coords
            x_center_px = int(x_center * img_w)
            y_center_px = int(y_center * img_h)
            width_px = int(width * img_w)
            height_px = int(height * img_h)
            
            # Calculate corner points
            x1 = int(x_center_px - width_px / 2)
            y1 = int(y_center_px - height_px / 2)
            x2 = int(x_center_px + width_px / 2)
            y2 = int(y_center_px + height_px / 2)
            
            # Draw rectangle
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw center point
            cv2.circle(vis_img, (x_center_px, y_center_px), 3, (0, 0, 255), -1)
        
        return vis_img
    
    def _print_statistics(self):
        """Print conversion statistics."""
        print("\n" + "="*60)
        print("CONVERSION STATISTICS")
        print("="*60)
        print(f"Total stacks processed:    {self.stats['total_stacks']}")
        print(f"Total planes processed:    {self.stats['total_planes']}")
        print(f"Total spines detected:     {self.stats['total_spines']}")
        print(f"Empty planes (no spines):  {self.stats['empty_planes']} ({self.stats['empty_planes']/self.stats['total_planes']*100:.1f}%)")
        print(f"Filtered (too small):      {self.stats['filtered_too_small']}")
        print(f"Filtered (too large):      {self.stats['filtered_too_large']}")
        print(f"Avg spines per plane:      {self.stats['total_spines']/self.stats['total_planes']:.2f}")
        print("="*60 + "\n")


def create_dataset_yaml(output_dir: str, dataset_name: str = 'spine_detection'):
    """Create YOLO dataset configuration file."""
    yaml_content = f"""# Spine Detection Dataset Configuration
# Generated automatically by convert_d3set_to_yolo.py

path: {Path(output_dir).absolute()}
train: images/train
val: images/val

# Classes
nc: 1
names: ['spine']

# Optional: Dataset info
# Source: DeepD3 dataset
# Task: Dendritic spine detection
"""
    
    yaml_path = Path(output_dir) / f'{dataset_name}.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\nCreated dataset configuration: {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description='Convert DeepD3 .d3set to YOLO format')
    parser.add_argument('--train_d3set', type=str, required=True,
                       help='Path to training .d3set file')
    parser.add_argument('--val_d3set', type=str, required=True,
                       help='Path to validation .d3set file')
    parser.add_argument('--output_dir', type=str, default='yolo_dataset',
                       help='Output directory for YOLO dataset')
    parser.add_argument('--min_pixels', type=int, default=10,
                       help='Minimum pixels for valid spine')
    parser.add_argument('--max_pixels', type=int, default=5000,
                       help='Maximum pixels for valid spine')
    parser.add_argument('--visualize', action='store_true',
                       help='Save visualization images with bounding boxes')
    parser.add_argument('--dataset_name', type=str, default='spine_detection',
                       help='Name for dataset YAML file')
    
    args = parser.parse_args()
    
    # Create converter
    converter = D3setToYOLOConverter(
        min_spine_pixels=args.min_pixels,
        max_spine_pixels=args.max_pixels
    )
    
    # Convert training set
    train_stats = converter.convert_d3set(
        args.train_d3set,
        args.output_dir,
        split='train',
        save_visualization=args.visualize
    )
    
    # Reset stats for validation set
    converter.stats = {k: 0 for k in converter.stats.keys()}
    
    # Convert validation set
    val_stats = converter.convert_d3set(
        args.val_d3set,
        args.output_dir,
        split='val',
        save_visualization=args.visualize
    )
    
    # Create dataset YAML
    yaml_path = create_dataset_yaml(args.output_dir, args.dataset_name)
    
    print("\n" + "="*60)
    print("CONVERSION COMPLETE!")
    print("="*60)
    print(f"Dataset location: {Path(args.output_dir).absolute()}")
    print(f"Configuration file: {yaml_path}")
    print(f"\nNext step: Train YOLO model using:")
    print(f"  python yolov8_optimized_training.py --data {yaml_path}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()