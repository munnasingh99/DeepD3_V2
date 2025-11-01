# File: utils.py (Updated for Hierarchical Probabilistic U-Net)

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        truncated_normal_(m.bias, mean=0, std=0.001)

def init_weights_orthogonal_normal(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.orthogonal_(m.weight)
        truncated_normal_(m.bias, mean=0, std=0.001)

def l2_regularisation(m):
    l2_reg = None
    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg

# Latent Visualization for Experimentation
def visualize_latent_distribution(latent_list, title_prefix=""):
    for idx, z in enumerate(latent_list):
        mean = z.base_dist.loc.mean().item()
        std = z.base_dist.scale.mean().item()
        print(f"{title_prefix} Latent Level {idx+1} -> Mean: {mean:.4f}, Std: {std:.4f}")

# Save Prediction and Mask Images for Debugging
def save_mask_prediction_example(mask, pred, iter_id, save_path='images/'):
    plt.imshow(pred, cmap='Greys')
    plt.axis('off')
    plt.savefig(f'{save_path}{iter_id}_prediction.png', bbox_inches='tight')
    plt.close()

    plt.imshow(mask, cmap='Greys')
    plt.axis('off')
    plt.savefig(f'{save_path}{iter_id}_mask.png', bbox_inches='tight')
    plt.close()
