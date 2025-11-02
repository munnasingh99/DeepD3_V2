import time
import os
import json
import csv
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.utils import save_image
from hpunet_model import *

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Metrics will only be saved to CSV/JSON")


def calculate_iou(pred, target, threshold=0.5):
    """
    Calculate Intersection over Union (IoU) for binary segmentation.
    
    Args:
        pred: Model predictions (logits or probabilities)
        target: Ground truth labels
        threshold: Threshold for binarizing predictions
    
    Returns:
        IoU score
    """
    # Apply sigmoid if needed and threshold
    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    # Calculate intersection and union
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    
    # Avoid division by zero
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    iou = intersection / union
    return iou.item()


def train_model(args, model, dataloader, criterion, optimizer, lr_scheduler, device='cpu', val_dataloader=None, start_time=None, task='dendrite', use_wandb=False): 
    """
    Train model for dendrite or spine segmentation.
    
    Args:
        task: 'dendrite' or 'spine' - determines which label to use
        use_wandb: Whether to log to Weights & Biases
    """
    history = {
        'training_time(min)': None,
        'train_metrics': [],
        'val_metrics': []
    }

    if val_dataloader is not None:
        val_minibatches = len(val_dataloader)

    # Determine which label index to use (0=dendrite, 1=spine)
    label_idx = 0 if task == 'dendrite' else 1

    # Setup CSV logging
    save_dir = os.path.join(args.output_dir, args.stamp)
    os.makedirs(save_dir, exist_ok=True)
    
    csv_train_path = os.path.join(save_dir, f'{task}_train_metrics.csv')
    csv_val_path = os.path.join(save_dir, f'{task}_val_metrics.csv')
    
    # Initialize CSV files with IoU column
    with open(csv_train_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss', 'reconstruction', 'kl_term', 'iou'] + 
                       [f'kl_scale_{i}' for i in range(args.latent_num)] + 
                       ['beta', 'lr'])
    
    with open(csv_val_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss', 'reconstruction', 'kl_term', 'iou'] + 
                       [f'kl_scale_{i}' for i in range(args.latent_num)])

    # Track epoch-level metrics for CSV logging
    epoch_metrics = {
        'loss': 0.0,
        'reconstruction': 0.0,
        'kl_term': 0.0,
        'iou': 0.0,
        'kl_scales': [0.0] * args.latent_num,
        'count': 0
    }

    def record_history(idx, loss_dict, iou_score, type='train', epoch=0, lr=None):
        """Record metrics to wandb and in-memory history"""
        loss_per_pixel = loss_dict['loss'].item() / args.pixels
        reconstruction_per_pixel = loss_dict['reconstruction_term'].item() / args.pixels
        kl_term_per_pixel = loss_dict['kl_term'].item() / args.pixels
        kl_per_pixel = [loss_dict['kls'][v].item() / args.pixels for v in range(args.latent_num)]

        # Prepare metrics dict
        metrics = {
            'step': idx,
            'epoch': epoch,
            'loss': loss_per_pixel,
            'reconstruction': reconstruction_per_pixel,
            'kl_term': kl_term_per_pixel,
            'iou': iou_score,
        }
        
        # Add individual KL terms
        for i, kl_val in enumerate(kl_per_pixel):
            metrics[f'kl_scale_{i}'] = kl_val
        
        # Add beta if available
        if type == 'train':
            if hasattr(criterion, 'beta_scheduler'):
                metrics['beta'] = criterion.beta_scheduler.beta
            if lr is not None:
                metrics['lr'] = lr

        # Save to history
        if type == 'train':
            history['train_metrics'].append(metrics)
        else:
            history['val_metrics'].append(metrics)

        # Log to wandb
        if use_wandb and WANDB_AVAILABLE:
            wandb_metrics = {f'{task}/{type}_{k}': v for k, v in metrics.items() 
                           if k not in ['step', 'epoch']}
            wandb_metrics['step'] = idx
            wandb_metrics['epoch'] = epoch
            wandb.log(wandb_metrics)

    def save_epoch_metrics_to_csv(epoch, epoch_metrics, type='train', lr=None):
        """Save epoch-averaged metrics to CSV"""
        count = epoch_metrics['count']
        if count == 0:
            return
        
        # Average the accumulated metrics
        avg_loss = epoch_metrics['loss'] / count
        avg_reconstruction = epoch_metrics['reconstruction'] / count
        avg_kl_term = epoch_metrics['kl_term'] / count
        avg_iou = epoch_metrics['iou'] / count
        avg_kl_scales = [kl / count for kl in epoch_metrics['kl_scales']]
        
        csv_path = csv_train_path if type == 'train' else csv_val_path
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [epoch, avg_loss, avg_reconstruction, avg_kl_term, avg_iou]
            row.extend(avg_kl_scales)
            if type == 'train':
                beta = criterion.beta_scheduler.beta if hasattr(criterion, 'beta_scheduler') else 0
                row.extend([beta, lr if lr is not None else 0])
            writer.writerow(row)

    # Prepare validation images
    val_images, val_truths = next(iter(val_dataloader))
    val_images, val_truths = val_images[:16], val_truths[label_idx][:16]
    truth_grid = make_grid(val_truths, nrow=4, pad_value=val_truths.min().item())
    image_grid = make_grid(val_images, nrow=4, pad_value=val_images.min().item())
    
    save_image(truth_grid, os.path.join(save_dir, f"{task}_val_truths_grid.png"))
    save_image(image_grid, os.path.join(save_dir, f"{task}_val_images_grid.png"))
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({
            f'{task}/val_ground_truth': wandb.Image(truth_grid),
            f'{task}/val_input_images': wandb.Image(image_grid)
        })
    
    val_images_selection = val_images.to(device)
    val_truths_selection = val_truths.to(device)
    
    last_time_checkpoint = start_time
    
    # Training loop
    for e in range(args.epochs):
        # Reset epoch metrics
        epoch_metrics = {
            'loss': 0.0,
            'reconstruction': 0.0,
            'kl_term': 0.0,
            'iou': 0.0,
            'kl_scales': [0.0] * args.latent_num,
            'count': 0
        }
        
        for mb, (images, truths) in enumerate(tqdm(dataloader, desc=f'{task.capitalize()} Epoch {e+1}/{args.epochs}')):
            idx = e*len(dataloader) + mb+1

            # Initialization
            criterion.train()
            model.train()
            model.zero_grad()
            images, truths = images.to(device), truths[label_idx].to(device)

            # Get Predictions
            if args.rec_type.lower() == 'mse':
                preds, infodicts = model(images, truths)
                preds, infodict = preds[:,0], infodicts[0]

            truths = truths.squeeze(dim=1)
            print(preds.max(), preds.min())

            # Calculate Loss
            loss = criterion(preds, truths, kls=infodict['kls'], lr=lr_scheduler.get_last_lr()[0])

            # Calculate IoU
            iou_score = calculate_iou(preds, truths)

            # Backpropagate
            loss.backward()
            optimizer.step()

            # Step Beta Scheduler
            if args.loss_type.lower() == 'elbo':
                criterion.beta_scheduler.step()

            # Accumulate metrics for epoch-level CSV
            loss_dict = criterion.last_loss.copy()
            loss_dict.update({'kls': infodict['kls']})
            
            epoch_metrics['loss'] += loss_dict['loss'].item() / args.pixels
            epoch_metrics['reconstruction'] += loss_dict['reconstruction_term'].item() / args.pixels
            epoch_metrics['kl_term'] += loss_dict['kl_term'].item() / args.pixels
            epoch_metrics['iou'] += iou_score
            for i in range(args.latent_num):
                epoch_metrics['kl_scales'][i] += loss_dict['kls'][i].item() / args.pixels
            epoch_metrics['count'] += 1

            # Record to wandb and history (per step)
            record_history(idx, loss_dict, iou_score, type='train', epoch=e+1, 
                          lr=lr_scheduler.get_last_lr()[0])
            
            # Validation
            if idx % args.val_period == 0 and val_dataloader is not None:
                criterion.eval()
                model.eval()

                # Show Sample Validation Images
                with torch.no_grad():
                    val_preds = model(val_images_selection)[0]
                    
                    out_grid = make_grid(val_preds, nrow=4, pad_value=val_preds.min().item())
                    save_image(out_grid, os.path.join(save_dir, f"{task}_val_preds_grid_{idx}.png"))
                    
                    if use_wandb and WANDB_AVAILABLE:
                        wandb.log({f'{task}/val_predictions': wandb.Image(out_grid), 'step': idx})

                # Calculate Validation Loss and IoU
                mean_val_loss = torch.zeros(1, device=device)
                mean_val_reconstruction_term = torch.zeros(1, device=device)
                mean_val_kl_term = torch.zeros(1, device=device)
                mean_val_kl = torch.zeros(args.latent_num, device=device)
                mean_val_iou = 0.0

                with torch.no_grad():
                    for _, (val_images, val_truths) in enumerate(val_dataloader):
                        val_images, val_truths = val_images.to(device), val_truths[label_idx].to(device)
                        
                        if args.rec_type.lower() == 'mse':
                            preds, infodicts = model(val_images, val_truths)
                            preds, infodict = preds[:,0], infodicts[0]

                        val_truths = val_truths.squeeze(dim=1)

                        loss = criterion(preds, val_truths, kls=infodict['kls'])
                        val_iou = calculate_iou(preds, val_truths)

                        mean_val_loss += loss
                        mean_val_reconstruction_term += criterion.last_loss['reconstruction_term']
                        mean_val_kl_term += criterion.last_loss['kl_term']
                        mean_val_kl += infodict['kls']
                        mean_val_iou += val_iou

                    mean_val_loss /= val_minibatches
                    mean_val_reconstruction_term /= val_minibatches
                    mean_val_kl_term /= val_minibatches
                    mean_val_kl /= val_minibatches
                    mean_val_iou /= val_minibatches

                # Record Validation History
                loss_dict = {
                    'loss': mean_val_loss,
                    'reconstruction_term': mean_val_reconstruction_term,
                    'kl_term': mean_val_kl_term,
                    'kls': mean_val_kl
                }
                record_history(idx, loss_dict, mean_val_iou, type='val', epoch=e+1)
                
                print(f'\n{task.capitalize()} Validation @ step {idx}: '
                      f'Loss={mean_val_loss.item()/args.pixels:.6f}, '
                      f'Rec={mean_val_reconstruction_term.item()/args.pixels:.6f}, '
                      f'KL={mean_val_kl_term.item()/args.pixels:.6f}, '
                      f'IoU={mean_val_iou:.4f}')
        
        # Save epoch metrics to CSV (once per epoch)
        save_epoch_metrics_to_csv(e+1, epoch_metrics, type='train', 
                                  lr=lr_scheduler.get_last_lr()[0])
        
        # Calculate and save validation metrics for this epoch
        if val_dataloader is not None:
            criterion.eval()
            model.eval()
            
            val_epoch_metrics = {
                'loss': 0.0,
                'reconstruction': 0.0,
                'kl_term': 0.0,
                'iou': 0.0,
                'kl_scales': [0.0] * args.latent_num,
                'count': 0
            }
            
            with torch.no_grad():
                for _, (val_images, val_truths) in enumerate(val_dataloader):
                    val_images, val_truths = val_images.to(device), val_truths[label_idx].to(device)
                    
                    if args.rec_type.lower() == 'mse':
                        preds, infodicts = model(val_images, val_truths)
                        preds, infodict = preds[:,0], infodicts[0]

                    val_truths = val_truths.squeeze(dim=1)

                    loss = criterion(preds, val_truths, kls=infodict['kls'])
                    val_iou = calculate_iou(preds, val_truths)

                    val_epoch_metrics['loss'] += criterion.last_loss['loss'].item() / args.pixels
                    val_epoch_metrics['reconstruction'] += criterion.last_loss['reconstruction_term'].item() / args.pixels
                    val_epoch_metrics['kl_term'] += criterion.last_loss['kl_term'].item() / args.pixels
                    val_epoch_metrics['iou'] += val_iou
                    for i in range(args.latent_num):
                        val_epoch_metrics['kl_scales'][i] += infodict['kls'][i].item() / args.pixels
                    val_epoch_metrics['count'] += 1
            
            save_epoch_metrics_to_csv(e+1, val_epoch_metrics, type='val')
        
        # Report Epoch Completion
        time_checkpoint = time.time()
        epoch_time = (time_checkpoint - last_time_checkpoint) / 60
        total_time = (time_checkpoint - start_time) / 60
        
        avg_epoch_iou = epoch_metrics['iou'] / epoch_metrics['count']
        print(f'{task.capitalize()} Epoch {e+1}/{args.epochs} done in {epoch_time:.1f} min. '
              f'Total: {total_time:.1f} min - Avg IoU: {avg_epoch_iou:.4f}')
        last_time_checkpoint = time_checkpoint
        
        # Save Model
        if (e+1) % args.save_period == 0 and (e+1) != args.epochs:
            torch.save(model.state_dict(), f'{args.output_dir}/{args.stamp}/{task}_model_epoch{e+1}.pth')
        
        # Step Learning Rate
        lr_scheduler.step()

    return history