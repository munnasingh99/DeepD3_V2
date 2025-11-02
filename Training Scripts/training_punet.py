"""
Training script for Probabilistic U-Net with KL annealing and balanced loss.

Features:
- Cyclical and monotonic KL annealing
- Free bits technique to prevent KL collapse
- Balanced Tversky loss
- Cosine annealing with warmup
- Mixed precision training
- WandB logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import os
import numpy as np
import cv2
from torch.utils.data import DataLoader
import wandb
from pathlib import Path
from models.datagen import DataGeneratorDataset
from models.prob_unet_with_tversky import ProbabilisticUnetDualLatent
from torch import autocast


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


class BalancedTverskyLoss(nn.Module):
    """
    Balanced Tversky loss with equal weighting for precision and recall.
    
    Args:
        alpha: False positive weight
        beta: False negative weight
        gamma: Focal exponent
        eps: Numerical stability constant
    """
    def __init__(self, alpha=0.5, beta=0.5, gamma=1.0, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps
        
    def forward(self, logits, target):
        p = torch.sigmoid(logits)
        tp = (p * target).sum()
        fp = (p * (1.0 - target)).sum()
        fn = ((1.0 - p) * target).sum()
        
        tversky = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        loss = (1.0 - tversky) ** self.gamma
        return loss


def cyclical_beta_schedule(epoch, cycle_length=10, min_beta=0.0, max_beta=1.0):
    """
    Cyclical annealing schedule to prevent posterior collapse.
    
    Alternates between increasing and decreasing beta values.
    """
    position = epoch % cycle_length
    if position < cycle_length // 2:
        beta = min_beta + (max_beta - min_beta) * (position / (cycle_length // 2))
    else:
        beta = max_beta - (max_beta - min_beta) * ((position - cycle_length // 2) / (cycle_length // 2))
    return beta


def monotonic_beta_schedule(epoch, total_epochs, min_beta=0.0, max_beta=1.0, warmup_frac=0.3):
    """
    Monotonic warmup schedule for stable KL annealing.
    """
    warmup_epochs = int(total_epochs * warmup_frac)
    if epoch < warmup_epochs:
        return min_beta + (max_beta - min_beta) * (epoch / warmup_epochs)
    return max_beta


def free_bits_kl(kl_per_sample, free_bits_lambda=0.5):
    """
    Free bits technique to prevent KL collapse.
    Only penalizes KL above a threshold.
    
    Args:
        kl_per_sample: KL divergence per sample
        free_bits_lambda: Minimum KL target (nats)
    """
    return torch.clamp(kl_per_sample - free_bits_lambda, min=0.0).mean()


def iou_score_from_logits(pred_logits, target, threshold=0.5, eps=1e-6):
    """Calculate IoU score from logits."""
    with torch.no_grad():
        probs = torch.sigmoid(pred_logits)
        pred = (probs > threshold).float()
        inter = (pred * target).sum()
        union = pred.sum() + target.sum() - inter
        return ((inter + eps) / (union + eps)).item()


def prf1_from_logits(pred_logits, target, threshold=0.5, eps=1e-6):
    """Calculate precision, recall, and F1 score from logits."""
    with torch.no_grad():
        probs = torch.sigmoid(pred_logits)
        pred = (probs > threshold).float()
        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        fn = ((1 - pred) * target).sum()
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)
        return f1.item(), precision.item(), recall.item()


class CosineAnnealingWarmup(optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with warmup and cosine annealing.
    
    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of warmup epochs
        max_epochs: Total training epochs
        eta_min: Minimum learning rate
    """
    def __init__(self, optimizer, warmup_epochs, max_epochs, eta_min=0.0, last_epoch=-1):
        self.warmup_epochs = max(1, int(warmup_epochs))
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        
        progress = (self.last_epoch - self.warmup_epochs) / max(1, (self.max_epochs - self.warmup_epochs))
        cosine = 0.5 * (1 + np.cos(np.pi * np.clip(progress, 0.0, 1.0)))
        return [self.eta_min + (base_lr - self.eta_min) * cosine for base_lr in self.base_lrs]


def save_images(image, mask, pred_logits, epoch, sample_idx, outdir, mode="train"):
    """Save training/validation images for visualization."""
    os.makedirs(outdir, exist_ok=True)
    image_np = image.detach().float().cpu().squeeze().numpy()
    mask_np = mask.detach().float().cpu().squeeze().numpy()
    prob = torch.sigmoid(pred_logits.detach()).float().cpu().squeeze().numpy()
    pred = (prob > 0.5).astype(np.uint8) * 255

    cv2.imwrite(
        os.path.join(outdir, f"{mode}_e{epoch:03d}_i{sample_idx}_img.png"),
        (np.clip(image_np, 0, 1) * 255).astype(np.uint8)
    )
    cv2.imwrite(
        os.path.join(outdir, f"{mode}_e{epoch:03d}_i{sample_idx}_gt.png"),
        (mask_np > 0.5).astype(np.uint8) * 255
    )
    cv2.imwrite(
        os.path.join(outdir, f"{mode}_e{epoch:03d}_i{sample_idx}_pred.png"), 
        pred
    )
    cv2.imwrite(
        os.path.join(outdir, f"{mode}_e{epoch:03d}_i{sample_idx}_prob.png"),
        (np.clip(prob, 0, 1) * 255).astype(np.uint8)
    )


def train_epoch(model, loader, optimizer, scaler, device, epoch, config):
    """
    Execute one training epoch.
    
    Returns:
        Dictionary of training metrics
    """
    model.train()
    
    meters = {
        'loss': 0.0, 'recon': 0.0, 'kl_d': 0.0, 'kl_s': 0.0,
        'iou_d': 0.0, 'iou_s': 0.0, 'f1_d': 0.0, 'f1_s': 0.0,
        'p_d': 0.0, 'r_d': 0.0, 'p_s': 0.0, 'r_s': 0.0
    }
    
    beta_strategy = config.get('beta_strategy', 'monotonic')
    
    if beta_strategy == 'cyclical':
        beta_d = cyclical_beta_schedule(
            epoch, 
            cycle_length=config.get('beta_cycle_length', 10),
            min_beta=config.get('beta_min', 0.0),
            max_beta=config.get('beta_dendrite', 1.0)
        )
        beta_s = cyclical_beta_schedule(
            epoch,
            cycle_length=config.get('beta_cycle_length', 10),
            min_beta=config.get('beta_min', 0.0),
            max_beta=config.get('beta_spine', 1.0)
        )
    else:
        beta_d = monotonic_beta_schedule(
            epoch,
            total_epochs=config.get('epochs', 50),
            min_beta=config.get('beta_min', 0.0),
            max_beta=config.get('beta_dendrite', 1.0),
            warmup_frac=config.get('beta_warmup_frac', 0.5)
        )
        beta_s = monotonic_beta_schedule(
            epoch,
            total_epochs=config.get('epochs', 50),
            min_beta=config.get('beta_min', 0.0),
            max_beta=config.get('beta_spine', 1.0),
            warmup_frac=config.get('beta_warmup_frac', 0.5)
        )
    
    use_free_bits = config.get('use_free_bits', True)
    free_bits_lambda = config.get('free_bits_lambda', 0.5)
    
    loop = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1} [Train]")
    
    for b, (image, (d_mask, s_mask)) in loop:
        image = image.to(device, non_blocking=True).float()
        d_mask = d_mask.to(device, non_blocking=True).float()
        s_mask = s_mask.to(device, non_blocking=True).float()
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast(device_type='cuda'):
            model.forward(image, segm_dendrite=d_mask, segm_spine=s_mask, training=True)
            
            d_logit, s_logit = model.reconstruct(use_posterior_mean=False)
            
            recon_d = model._recon_loss(d_logit, d_mask)
            recon_s = model._recon_loss(s_logit, s_mask)
            recon_loss = (
                config.get('loss_weight_dendrite', 1.0) * recon_d +
                config.get('loss_weight_spine', 1.0) * recon_s
            )
            
            kl_d_per_sample, kl_s_per_sample = model.kl_divergence()
            
            if use_free_bits:
                kl_d_loss = free_bits_kl(kl_d_per_sample, free_bits_lambda)
                kl_s_loss = free_bits_kl(kl_s_per_sample, free_bits_lambda)
            else:
                kl_d_loss = kl_d_per_sample.mean()
                kl_s_loss = kl_s_per_sample.mean()
            
            total_loss = recon_loss + beta_d * kl_d_loss + beta_s * kl_s_loss
        
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        with torch.no_grad():
            meters['loss'] += total_loss.item()
            meters['recon'] += recon_loss.item()
            meters['kl_d'] += kl_d_per_sample.mean().item()
            meters['kl_s'] += kl_s_per_sample.mean().item()
            
            meters['iou_d'] += iou_score_from_logits(d_logit, d_mask)
            meters['iou_s'] += iou_score_from_logits(s_logit, s_mask)
            
            f1d, pd, rd = prf1_from_logits(d_logit, d_mask)
            f1s, ps, rs = prf1_from_logits(s_logit, s_mask)
            meters['f1_d'] += f1d
            meters['f1_s'] += f1s
            meters['p_d'] += pd
            meters['r_d'] += rd
            meters['p_s'] += ps
            meters['r_s'] += rs
        
        loop.set_postfix({
            'loss': f'{total_loss.item():.4f}',
            'kl_d': f'{kl_d_per_sample.mean().item():.4f}',
            'kl_s': f'{kl_s_per_sample.mean().item():.4f}',
            'f1_d': f'{f1d:.3f}',
            'f1_s': f'{f1s:.3f}'
        })
    
    n = len(loader)
    return {
        'train_loss': meters['loss'] / n,
        'train_recon': meters['recon'] / n,
        'train_kl_dend': meters['kl_d'] / n,
        'train_kl_spine': meters['kl_s'] / n,
        'train_iou_dend': meters['iou_d'] / n,
        'train_iou_spine': meters['iou_s'] / n,
        'train_f1_dend': meters['f1_d'] / n,
        'train_f1_spine': meters['f1_s'] / n,
        'train_prec_dend': meters['p_d'] / n,
        'train_rec_dend': meters['r_d'] / n,
        'train_prec_spine': meters['p_s'] / n,
        'train_rec_spine': meters['r_s'] / n,
        'beta_d_current': beta_d,
        'beta_s_current': beta_s,
    }


@torch.no_grad()
def validate_epoch(model, loader, device, epoch, config, outdir=None):
    """
    Execute one validation epoch with multi-sample averaging.
    
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    
    meters = {
        'loss': 0.0, 'iou_d': 0.0, 'iou_s': 0.0,
        'f1_d': 0.0, 'f1_s': 0.0, 'p_d': 0.0, 'r_d': 0.0, 'p_s': 0.0, 'r_s': 0.0
    }
    
    n_mc = config.get('val_mc_samples', 16)
    
    loop = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1} [Val]")
    
    for b, (image, (d_mask, s_mask)) in loop:
        image = image.to(device, non_blocking=True).float()
        d_mask = d_mask.to(device, non_blocking=True).float()
        s_mask = s_mask.to(device, non_blocking=True).float()
        
        model.forward(image, training=False)
        
        pd_list, ps_list = [], []
        for _ in range(n_mc):
            ld, ls = model.sample(testing=True, use_posterior=False)
            pd_list.append(torch.sigmoid(ld))
            ps_list.append(torch.sigmoid(ls))
        
        pd_mean = torch.stack(pd_list).mean(0)
        ps_mean = torch.stack(ps_list).mean(0)
        
        eps = 1e-6
        ld_mean = torch.log((pd_mean + eps) / (1 - pd_mean + eps))
        ls_mean = torch.log((ps_mean + eps) / (1 - ps_mean + eps))
        
        bce = nn.BCEWithLogitsLoss(reduction='mean')
        val_loss = bce(ld_mean, d_mask) + bce(ls_mean, s_mask)
        meters['loss'] += val_loss.item()
        
        meters['iou_d'] += iou_score_from_logits(ld_mean, d_mask)
        meters['iou_s'] += iou_score_from_logits(ls_mean, s_mask)
        
        f1d, pd, rd = prf1_from_logits(ld_mean, d_mask)
        f1s, ps, rs = prf1_from_logits(ls_mean, s_mask)
        meters['f1_d'] += f1d
        meters['f1_s'] += f1s
        meters['p_d'] += pd
        meters['r_d'] += rd
        meters['p_s'] += ps
        meters['r_s'] += rs
        
        loop.set_postfix({
            'loss': f'{val_loss.item():.4f}',
            'f1_d': f'{f1d:.3f}',
            'f1_s': f'{f1s:.3f}',
            'p_d': f'{pd:.3f}',
            'r_d': f'{rd:.3f}'
        })
        
        if outdir is not None and b == 0 and (epoch % 3 == 0):
            save_images(image[0], d_mask[0], ld_mean[0], epoch, 0, outdir, mode="val_dend")
            save_images(image[0], s_mask[0], ls_mean[0], epoch, 0, outdir, mode="val_spine")
    
    n = len(loader)
    return {
        'val_loss': meters['loss'] / n,
        'val_iou_dend': meters['iou_d'] / n,
        'val_iou_spine': meters['iou_s'] / n,
        'val_f1_dend': meters['f1_d'] / n,
        'val_f1_spine': meters['f1_s'] / n,
        'val_prec_dend': meters['p_d'] / n,
        'val_rec_dend': meters['r_d'] / n,
        'val_prec_spine': meters['p_s'] / n,
        'val_rec_spine': meters['r_s'] / n,
    }


def train_model(config=None):
    """
    Main training function.
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        Tuple of (trained_model, best_f1_score)
    """
    wandb.init(project="prob-unet-improved", config=config)
    config = wandb.config if config is None else config
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    out_root = Path("punet_training_output")
    model_dir = out_root / "models"
    img_dir = out_root / "images"
    model_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    
    train_dataset = DataGeneratorDataset(
        r"../dataset/DeepD3_Training.d3set",
        samples_per_epoch=config.get('samples_per_epoch', 256 * 128),
        size=(1, 128, 128),
        augment=True,
        shuffle=True,
    )
    val_dataset = DataGeneratorDataset(
        r"../dataset/DeepD3_Validation.d3set",
        samples_per_epoch=config.get('val_samples', 64 * 8),
        size=(1, 128, 128),
        augment=False,
        shuffle=False,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 16),
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 16),
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    model = ProbabilisticUnetDualLatent(
        input_channels=1,
        num_classes=1,
        num_filters=config.get('num_filters', [32, 64, 128, 192]),
        latent_dim_dendrite=config.get('latent_dim_dendrite', 12),
        latent_dim_spine=config.get('latent_dim_spine', 12),
        no_convs_fcomb=4,
        beta_dendrite=config.get('beta_dendrite', 1.0),
        beta_spine=config.get('beta_spine', 1.0),
        loss_weight_dendrite=1.0,
        loss_weight_spine=1.0,
        recon_loss='tversky',
        tversky_alpha=0.5,
        tversky_beta=0.5,
        tversky_gamma=1.5,
    ).to(device)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get('learning_rate', 1e-3),
        weight_decay=config.get('weight_decay', 1e-5)
    )
    scheduler = CosineAnnealingWarmup(
        optimizer,
        warmup_epochs=config.get('warmup_epochs', 5),
        max_epochs=config.get('epochs', 50),
        eta_min=1e-6
    )
    scaler = GradScaler()
    
    best_f1 = 0.0
    patience_cnt = 0
    patience = config.get('patience', 15)
    
    print(f"\nTraining for {config.get('epochs', 50)} epochs")
    print("=" * 80)
    
    for epoch in range(config.get('epochs', 50)):
        train_metrics = train_epoch(model, train_loader, optimizer, scaler, device, epoch, config)
        val_metrics = validate_epoch(
            model, val_loader, device, epoch, config, 
            outdir=img_dir if epoch % 3 == 0 else None
        )
        
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        
        all_metrics = {**train_metrics, **val_metrics, 'learning_rate': lr, 'epoch': epoch + 1}
        wandb.log(all_metrics)
        
        cur_f1 = 0.5 * (val_metrics['val_f1_dend'] + val_metrics['val_f1_spine'])
        
        print(f"\nEpoch {epoch+1} | LR {lr:.2e}")
        print(f"Train: Loss {train_metrics['train_loss']:.4f} | KL_d {train_metrics['train_kl_dend']:.4f} | KL_s {train_metrics['train_kl_spine']:.4f}")
        print(f"       Dend F1 {train_metrics['train_f1_dend']:.3f} (P {train_metrics['train_prec_dend']:.3f} / R {train_metrics['train_rec_dend']:.3f})")
        print(f"       Spine F1 {train_metrics['train_f1_spine']:.3f} (P {train_metrics['train_prec_spine']:.3f} / R {train_metrics['train_rec_spine']:.3f})")
        print(f"Val:   Loss {val_metrics['val_loss']:.4f}")
        print(f"       Dend F1 {val_metrics['val_f1_dend']:.3f} (P {val_metrics['val_prec_dend']:.3f} / R {val_metrics['val_rec_dend']:.3f})")
        print(f"       Spine F1 {val_metrics['val_f1_spine']:.3f} (P {val_metrics['val_prec_spine']:.3f} / R {val_metrics['val_rec_spine']:.3f})")
        print(f"Combined F1: {cur_f1:.3f} (Best: {best_f1:.3f})")
        print("=" * 80)
        
        if cur_f1 > best_f1:
            best_f1 = cur_f1
            patience_cnt = 0
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'config': dict(config),
            }, model_dir / f"best_model_f1_{best_f1:.4f}.pth")
            print(f"Saved best model")
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    print(f"\nTraining complete. Best F1: {best_f1:.4f}")
    wandb.finish()
    return model, best_f1


if __name__ == "__main__":
    config = {
        'num_filters': [32, 64, 128, 192],
        'latent_dim_dendrite': 12,
        'latent_dim_spine': 12,
        'batch_size': 16,
        'epochs': 50,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'warmup_epochs': 5,
        'beta_strategy': 'monotonic',
        'beta_dendrite': 1.0,
        'beta_spine': 1.0,
        'beta_min': 0.0,
        'beta_warmup_frac': 0.5,
        'beta_cycle_length': 10,
        'use_free_bits': True,
        'free_bits_lambda': 0.5,
        'samples_per_epoch': 256 * 128,
        'val_samples': 64 * 8,
        'val_mc_samples': 24,
        'patience': 15,
    }
    
    print("Training configuration:")
    for k, v in config.items():
        print(f"{k}: {v}")
    print("=" * 80)
    
    model, best_f1 = train_model(config)