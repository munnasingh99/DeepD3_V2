import argparse
import datetime
import json
import time
import socket
import os
import math

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.hpunet_model import *
from models.hpunet_train import train_model
from models.hpunet_datagen import DataGeneratorDataset


try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Metrics will only be saved to CSV/JSON")

parser = argparse.ArgumentParser(description="Hierarchical Probabilistic U-Net Training")

# General
parser.add_argument("--config", type=str, help="Path to JSON config file")
parser.add_argument("--random_seed", type=int, default=0, help="Random seed")
parser.add_argument("--output_dir", default="hpunet_training_output", help="Output directory")
parser.add_argument("--comment", default="", help="Comment for run")

# Data
parser.add_argument("--train_data_path", default="dataset/DeepD3_Training.d3set")
parser.add_argument("--val_data_path", default="dataset/DeepD3_Validation.d3set")
parser.add_argument("--image_size", type=int, default=128)
parser.add_argument("--samples_per_epoch", type=int, default=16384)
parser.add_argument("--val_samples", type=int, default=2048)
parser.add_argument("--val_period", type=int, default=500)
parser.add_argument("--val_bs", type=int, default=32)
parser.add_argument("--num_workers", type=int, default=4)

# Model
parser.add_argument("--in_ch", type=int, default=1)
parser.add_argument("--out_ch", type=int, default=1)
parser.add_argument("--intermediate_ch", type=int, nargs='+', default=[32, 64, 128, 256])
parser.add_argument("--kernel_size", type=int, nargs='+', default=[3, 3, 3, 3])
parser.add_argument("--scale_depth", type=int, nargs='+', default=[2, 2, 2, 2])
parser.add_argument("--dilation", type=int, nargs='+', default=[1, 1, 1, 1])
parser.add_argument("--padding_mode", default='zeros')
parser.add_argument("--latent_num", type=int, default=4)
parser.add_argument("--latent_chs", type=int, nargs='+', default=[2, 4, 8, 16])
parser.add_argument("--latent_locks", type=int, nargs='+', default=[0, 0, 0, 0])

# Loss
parser.add_argument("--rec_type", default="mse")
parser.add_argument("--loss_type", default="enhanced_elbo")
parser.add_argument("--beta", type=float, default=0.1)
parser.add_argument("--beta_asc_steps", type=int)
parser.add_argument("--beta_cons_steps", type=int, default=1)
parser.add_argument("--beta_saturation_step", type=int)

# Training
parser.add_argument("--epochs", type=int, default=120)
parser.add_argument("--bs", type=int, default=64)
parser.add_argument("--optimizer", default="adam")
parser.add_argument("--wd", type=float, default=1e-4)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--scheduler_type", default='cosine')
parser.add_argument("--scheduler_warmup_epochs", type=int, default=5)
parser.add_argument("--save_period", type=int, default=10)


# Logging
parser.add_argument("--use_wandb", action='store_true')
parser.add_argument("--wandb_project", default="hpunet")
parser.add_argument("--wandb_entity", default=None)

args = parser.parse_args()

# Load config if provided
if args.config:
    with open(args.config, "r") as f:
        config_dict = json.load(f)
    for key, value in config_dict.items():
        if not key.startswith('_'):  # Skip comment fields
            setattr(args, key, value)

# Set random seeds
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Generate stamp
timestamp = datetime.datetime.now().strftime('%m%d-%H%M')
compute_node = socket.gethostname()
suffix = datetime.datetime.now().strftime('%f')
stamp = f"{timestamp}_{compute_node[:2]}_{suffix}"
if args.comment:
    stamp += f"_{args.comment}"
args.stamp = stamp
print(f'Run ID: {stamp}')

# Setup data
train_dataset = DataGeneratorDataset(
    fn=args.train_data_path,
    samples_per_epoch=args.samples_per_epoch,
    size=(1, args.image_size, args.image_size),
    augment=True,
    shuffle=True,
    seed=args.random_seed
)

val_dataset = DataGeneratorDataset(
    fn=args.val_data_path,
    samples_per_epoch=args.val_samples,
    size=(1, args.image_size, args.image_size),
    augment=False,
    shuffle=True,
    seed=args.random_seed
)

print(f'Train dataset: {len(train_dataset)} samples')
print(f'Val dataset: {len(val_dataset)} samples')

train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, 
                          num_workers=args.num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=args.val_bs, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

# Expand and validate args
args.size = args.image_size
args.pixels = args.image_size * args.image_size

if args.latent_locks is None:
    args.latent_locks = [0] * args.latent_num
args.latent_locks = [bool(l) for l in args.latent_locks]

# Expand lists if needed
for attr in ['kernel_size', 'dilation', 'scale_depth']:
    val = getattr(args, attr)
    if len(val) < len(args.intermediate_ch):
        if len(val) == 1:
            setattr(args, attr, val * len(args.intermediate_ch))

# Save config
save_dir = os.path.join(args.output_dir, stamp)
os.makedirs(save_dir, exist_ok=True)

with open(os.path.join(save_dir, 'config.json'), 'w') as f:
    json.dump(vars(args), f, indent=2)

# Initialize wandb if requested
if args.use_wandb and WANDB_AVAILABLE:
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, 
               name=stamp, config=vars(args))
    print("W&B initialized")
elif args.use_wandb:
    print("Warning: W&B requested but not available")

# Train both dendrite and spine
start_time = time.time()

for task in ['dendrite', 'spine']:
    print(f'\n{"="*60}')
    print(f'Training {task.upper()} model')
    print(f'{"="*60}\n')
    
    # Initialize model
    model = HPUNet(
        in_ch=args.in_ch, out_ch=args.out_ch, chs=args.intermediate_ch,
        latent_num=args.latent_num, latent_channels=args.latent_chs,
        latent_locks=args.latent_locks, scale_depth=args.scale_depth,
        kernel_size=args.kernel_size, dilation=args.dilation,
        padding_mode=args.padding_mode
    ).to(device)
    
    # Setup loss
    if args.rec_type.lower() == 'mse':
        reconstruction_loss = MSELossWrapper()
    else:
        raise ValueError(f'Invalid reconstruction loss: {args.rec_type}')
    
    if args.loss_type.lower() == 'enhanced_elbo':
        beta_scheduler = BetaConstant(args.beta)
        criterion = EnhancedELBOLoss(
            reconstruction_loss=reconstruction_loss,
            n_latents=args.latent_num,
            beta=beta_scheduler,
            conv_dim=2,
            kl_balancing=True
        ).to(device)
    elif args.loss_type.lower() == 'elbo':
        beta_scheduler = BetaConstant(args.beta)
        criterion = ELBOLoss(
            reconstruction_loss=reconstruction_loss,
            beta=beta_scheduler,
            conv_dim=2
        ).to(device)
    else:
        raise ValueError(f'Invalid loss type: {args.loss_type}')
    
    # Setup optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        raise ValueError(f'Unknown optimizer: {args.optimizer}')
    
    # Setup scheduler
    if args.scheduler_type == 'cosine':
        warmup_epochs = args.scheduler_warmup_epochs
        
        if warmup_epochs > 0:
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return float(epoch) / float(max(1, warmup_epochs))
                else:
                    progress = (epoch - warmup_epochs) / (args.epochs - warmup_epochs)
                    return 0.5 * (1. + math.cos(math.pi * progress))
            lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs)
    
    # Train model
    history = train_model(
        args=args,
        model=model,
        dataloader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        val_dataloader=val_loader,
        start_time=start_time,
        task=task,
        use_wandb=args.use_wandb and WANDB_AVAILABLE
    )
    
    # Save final model
    torch.save(model.state_dict(), f'{save_dir}/{task}_model_final.pth')
    torch.save(model, f'{save_dir}/{task}_model_complete.pth')
    
    # Save history to JSON
    with open(f'{save_dir}/{task}_history.json', 'w') as f:
        # Convert tensors to floats for JSON serialization
        history_json = history.copy()
        for metrics_list in [history_json['train_metrics'], history_json['val_metrics']]:
            for metrics in metrics_list:
                for key, val in metrics.items():
                    if isinstance(val, torch.Tensor):
                        metrics[key] = val.item()
        json.dump(history_json, f, indent=2)
    
    print(f'\n{task.capitalize()} model saved to {save_dir}/{task}_model_final.pth')
    print(f'Metrics saved to {save_dir}/{task}_train_metrics.csv')
    print(f'Metrics saved to {save_dir}/{task}_val_metrics.csv')
    print(f'History saved to {save_dir}/{task}_history.json\n')

# Finish
end_time = time.time()
total_time = (end_time - start_time) / 3600
print(f'\n{"="*60}')
print(f'Training completed in {total_time:.2f} hours')
print(f'{"="*60}')

if args.use_wandb and WANDB_AVAILABLE:
    wandb.finish()

# Usage example:
"""python3 hpunet_main.py --config hpunet_config.json --use_wandb """