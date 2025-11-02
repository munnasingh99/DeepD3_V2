import os
import json
import argparse
from types import SimpleNamespace

import torch
import numpy as np
import tifffile as tiff
import imageio as io
from tqdm import tqdm
import torchvision.transforms as transforms

from models.hpunet_model import HPUNet

class InferencePipeline:
    def __init__(self, config_path, dendrite_model_path, spine_model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.load_config(config_path)
        self.load_models(dendrite_model_path, spine_model_path)
        self.transform = transforms.ToTensor()
    
    def load_config(self, config_path):
        """Load configuration from JSON file"""
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        self.args = SimpleNamespace(**config_dict)
    
    def load_models(self, dendrite_path, spine_path):
        """Load dendrite and spine models"""
        self.model_dendrite = HPUNet(
            in_ch=self.args.in_ch,
            chs=self.args.intermediate_ch,
            latent_num=self.args.latent_num,
            out_ch=self.args.out_ch,
            scale_depth=self.args.scale_depth,
            kernel_size=self.args.kernel_size,
            dilation=self.args.dilation,
            padding_mode=self.args.padding_mode,
            latent_channels=self.args.latent_chs,
            latent_locks=self.args.latent_locks
        )
        
        self.model_spine = HPUNet(
            in_ch=self.args.in_ch,
            chs=self.args.intermediate_ch,
            latent_num=self.args.latent_num,
            out_ch=self.args.out_ch,
            scale_depth=self.args.scale_depth,
            kernel_size=self.args.kernel_size,
            dilation=self.args.dilation,
            padding_mode=self.args.padding_mode,
            latent_channels=self.args.latent_chs,
            latent_locks=self.args.latent_locks
        )
        
        self.model_dendrite.load_state_dict(torch.load(dendrite_path))
        self.model_spine.load_state_dict(torch.load(spine_path))
        
        self.model_dendrite.to(self.device)
        self.model_spine.to(self.device)
        
        self.model_dendrite.eval()
        self.model_spine.eval()
    
    def preprocess_image(self, image):
        """Normalize and pad image for model inference"""
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        h, w = image.shape
        
        # Pad to multiple of 32
        pad_h = (32 - h % 32) if h % 32 else 0
        pad_w = (32 - w % 32) if w % 32 else 0
        
        if pad_h or pad_w:
            image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
        
        image = self.transform(image.astype(np.float32)).unsqueeze(0).to(self.device)
        return image, (h, w)
    
    def predict_single_slice(self, image_slice, model):
        """Run inference on a single image slice"""
        processed_slice, (orig_h, orig_w) = self.preprocess_image(image_slice)
        
        with torch.no_grad():
            pred = torch.sigmoid(model(processed_slice)[0])
            pred = pred.squeeze().squeeze().cpu().numpy()[:orig_h, :orig_w]
        
        return pred
    
    def process_stack(self, tif_path):
        """Process entire TIFF stack"""
        # Load TIFF stack
        tif_stack = np.asarray(io.mimread(tif_path, memtest=False))
        
        if len(tif_stack.shape) == 2:
            tif_stack = tif_stack[np.newaxis, ...]
        
        # Initialize prediction arrays
        predictions_dendrite = np.zeros(tif_stack.shape, dtype=np.float32)
        predictions_spine = np.zeros(tif_stack.shape, dtype=np.float32)
        
        # Process each slice
        print("Processing dendrite predictions...")
        for i in tqdm(range(len(tif_stack))):
            predictions_dendrite[i] = self.predict_single_slice(tif_stack[i], self.model_dendrite)
        
        print("Processing spine predictions...")
        for i in tqdm(range(len(tif_stack))):
            predictions_spine[i] = self.predict_single_slice(tif_stack[i], self.model_spine)
        
        return predictions_dendrite, predictions_spine
    
    def save_predictions(self, dendrite_pred, spine_pred, output_dir):
        """Save prediction results"""
        os.makedirs(output_dir, exist_ok=True)
        
        dendrite_path = os.path.join(output_dir, 'H_Prob_UNet_Dend.tiff')
        spine_path = os.path.join(output_dir, 'H_Prob_UNet_Spine.tiff')
        
        tiff.imwrite(dendrite_path, dendrite_pred)
        tiff.imwrite(spine_path, spine_pred)
        
        print(f"Predictions saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Run inference on TIFF stack')
    parser.add_argument('--tif_path', type=str, required=True, help='Path to input TIFF file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for predictions')
    parser.add_argument('--config', type=str, default='test_config.json', help='Path to config file')
    parser.add_argument('--dendrite_model', type=str, required=True, help='Path to dendrite model weights')
    parser.add_argument('--spine_model', type=str, required=True, help='Path to spine model weights')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = InferencePipeline(
        config_path=args.config,
        dendrite_model_path=args.dendrite_model,
        spine_model_path=args.spine_model
    )
    
    # Run inference
    dendrite_pred, spine_pred = pipeline.process_stack(args.tif_path)
    
    # Save results
    pipeline.save_predictions(dendrite_pred, spine_pred, args.output_dir)
    
    print("Inference completed successfully!")

if __name__ == "__main__":
    main()
