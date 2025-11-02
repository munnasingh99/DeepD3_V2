"""
Probabilistic U-Net components integrated with DeepD3 model.

Contains encoder, Gaussian latent spaces, and feature combination modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl
import numpy as np

from deepd3_model import DeepD3Model
from unet_blocks import *
from utils import init_weights, init_weights_orthogonal_normal, l2_regularisation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):
    """
    Convolutional encoder with downsampling blocks.
    
    Args:
        input_channels: Number of input channels
        num_filters: List of filter counts per block
        no_convs_per_block: Number of convolutions per block
        initializers: Weight initialization config
        segm_channels: Number of segmentation channels to concatenate
        padding: Whether to use padding
        posterior: Whether this is a posterior encoder (concatenates segmentation)
    """
    def __init__(
        self, 
        input_channels, 
        num_filters, 
        no_convs_per_block, 
        initializers, 
        segm_channels,
        padding=True, 
        posterior=False
    ):
        super(Encoder, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.num_filters = num_filters

        if posterior:
            self.input_channels += segm_channels

        layers = []
        for i in range(len(self.num_filters)):
            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = num_filters[i]
            
            if i != 0:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))
            
            layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=int(padding)))
            layers.append(nn.BatchNorm2d(output_dim)) 
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_per_block - 1):
                layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=int(padding)))
                layers.append(nn.BatchNorm2d(output_dim))
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)
        self.layers.apply(init_weights)

    def forward(self, input):
        return self.layers(input)


class AxisAlignedConvGaussian(nn.Module):
    """
    Convolutional network that parametrizes a Gaussian distribution 
    with diagonal covariance matrix.
    
    Args:
        input_channels: Number of input channels
        num_filters: List of filter counts per block
        no_convs_per_block: Number of convolutions per block
        latent_dim: Dimensionality of latent space
        initializers: Weight initialization config
        segm_channels: Number of segmentation channels
        posterior: Whether this is posterior (uses segmentation) or prior
    """
    def __init__(
        self, 
        input_channels, 
        num_filters, 
        no_convs_per_block, 
        latent_dim, 
        initializers, 
        segm_channels,
        posterior=False
    ):
        super(AxisAlignedConvGaussian, self).__init__()
        self.input_channels = input_channels
        self.channel_axis = 1
        self.num_filters = num_filters
        self.no_convs_per_block = no_convs_per_block
        self.latent_dim = latent_dim
        self.segm_channels = segm_channels
        self.posterior = posterior
        self.name = 'Posterior' if self.posterior else 'Prior'
        
        self.encoder = Encoder(
            self.input_channels, 
            self.num_filters, 
            self.no_convs_per_block, 
            initializers, 
            self.segm_channels,
            posterior=self.posterior
        )
        self.conv_layer = nn.Conv2d(num_filters[-1], 2 * self.latent_dim, (1, 1), stride=1)
        self.show_img = 0
        self.show_seg = 0
        self.show_concat = 0
        self.show_enc = 0
        self.sum_input = 0

        nn.init.kaiming_normal_(self.conv_layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.normal_(self.conv_layer.bias)

    def forward(self, input, segm=None):
        """
        Forward pass through encoder to latent distribution.
        
        Args:
            input: Input image
            segm: Segmentation mask (for posterior only)
            
        Returns:
            Multivariate normal distribution with diagonal covariance
        """
        if segm is not None:
            self.show_img = input
            self.show_seg = segm
            input = torch.cat((input, segm), dim=1)
            self.show_concat = input
            self.sum_input = torch.sum(input)

        encoding = self.encoder(input)
        self.show_enc = encoding

        encoding = torch.mean(encoding, dim=2, keepdim=True)
        encoding = torch.mean(encoding, dim=3, keepdim=True)

        mu_log_sigma = self.conv_layer(encoding)

        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)

        mu = mu_log_sigma[:, :self.latent_dim]
        log_sigma = mu_log_sigma[:, self.latent_dim:]
        log_sigma = torch.clamp(log_sigma, min=-10, max=10)

        dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma + 1e-6)), 1)
        return dist


class Fcomb(nn.Module):
    """
    Feature combination module.
    
    Combines latent sample with U-Net features via 1x1 convolutions.
    
    Args:
        num_filters: Filter configuration
        latent_dim: Latent space dimensionality
        num_output_channels: Number of output channels
        num_classes: Number of output classes
        no_convs_fcomb: Number of 1x1 convolutions
        initializers: Weight initialization config
        use_tile: Whether to tile latent sample to match spatial dimensions
    """
    def __init__(
        self, 
        num_filters, 
        latent_dim, 
        num_output_channels, 
        num_classes, 
        no_convs_fcomb, 
        initializers, 
        use_tile=True
    ):
        super(Fcomb, self).__init__()
        self.num_channels = num_output_channels
        self.num_classes = num_classes
        self.channel_axis = 1
        self.spatial_axes = [2, 3]
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.use_tile = use_tile
        self.no_convs_fcomb = no_convs_fcomb
        self.name = 'Fcomb'

        if self.use_tile:
            layers = []

            layers.append(nn.Conv2d(
                self.num_filters[0] + self.latent_dim, 
                self.num_filters[0], 
                kernel_size=1
            ))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_fcomb - 2):
                layers.append(nn.Conv2d(
                    self.num_filters[0], 
                    self.num_filters[0], 
                    kernel_size=1
                ))
                layers.append(nn.ReLU(inplace=True))

            self.layers = nn.Sequential(*layers)
            self.last_layer = nn.Conv2d(self.num_filters[0], self.num_classes, kernel_size=1)

            if initializers['w'] == 'orthogonal':
                self.layers.apply(init_weights_orthogonal_normal)
                self.last_layer.apply(init_weights_orthogonal_normal)
            else:
                self.layers.apply(init_weights)
                self.last_layer.apply(init_weights)

    def tile(self, a, dim, n_tile):
        """
        Tile tensor along specified dimension.
        Mimics TensorFlow's tf.tile behavior.
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(
            np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])
        ).to(device)
        return torch.index_select(a, dim, order_index)

    def forward(self, feature_map, z):
        """
        Combine feature map with latent sample.
        
        Args:
            feature_map: Feature map from U-Net [B, C, H, W]
            z: Latent sample [B, latent_dim]
            
        Returns:
            Combined output logits
        """
        if self.use_tile:
            z = torch.unsqueeze(z, 2)
            z = self.tile(z, 2, feature_map.shape[self.spatial_axes[0]])
            z = torch.unsqueeze(z, 3)
            z = self.tile(z, 3, feature_map.shape[self.spatial_axes[1]])

            feature_map = torch.cat((feature_map, z), dim=self.channel_axis)
            output = self.layers(feature_map)
            return self.last_layer(output)


class ProbabilisticUnet(nn.Module):
    """
    Probabilistic U-Net integrated with custom DeepD3 dual-decoder architecture.
    
    This is a basic version with single latent space. For dual latent spaces,
    see ProbabilisticUnetDualLatent in prob_unet_with_tversky.py.
    """
    def __init__(
        self, 
        input_channels=1, 
        num_classes=1, 
        num_filters=[32, 64, 128, 192], 
        latent_dim=6, 
        no_convs_fcomb=4, 
        beta=1
    ):
        super(ProbabilisticUnet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.no_convs_per_block = 3
        self.no_convs_fcomb = no_convs_fcomb
        self.initializers = {'w': 'he_normal', 'b': 'normal'}
        self.beta = beta
        self.z_prior_sample = 0

        self.unet = DeepD3Model(
            in_channels=self.input_channels,
            base_filters=self.num_filters[0],
            num_layers=len(self.num_filters),
            activation="swish",
            use_batchnorm=True,
            apply_last_layer=False
        ).to(device)

        self.fcomb_dendrites = Fcomb(
            self.num_filters, 
            self.latent_dim, 
            self.input_channels, 
            self.num_classes, 
            self.no_convs_fcomb, 
            {'w': 'orthogonal', 'b': 'normal'}, 
            use_tile=True
        ).to(device)
        
        self.fcomb_spines = Fcomb(
            self.num_filters, 
            self.latent_dim, 
            self.input_channels, 
            self.num_classes, 
            self.no_convs_fcomb, 
            {'w': 'orthogonal', 'b': 'normal'}, 
            use_tile=True
        ).to(device)

        self.prior = AxisAlignedConvGaussian(
            self.input_channels, 
            self.num_filters, 
            self.no_convs_per_block, 
            self.latent_dim, 
            self.initializers, 
            posterior=False,
            segm_channels=1
        ).to(device)
        
        self.posterior = AxisAlignedConvGaussian(
            self.input_channels, 
            self.num_filters, 
            self.no_convs_per_block, 
            self.latent_dim, 
            self.initializers, 
            posterior=True,
            segm_channels=2
        ).to(device)

    def forward(self, patch, segm, training=True):
        """
        Forward pass through prior/posterior and U-Net.
        """
        if training:
            self.posterior_latent_space = self.posterior.forward(patch, segm)
        
        self.prior_latent_space = self.prior.forward(patch)

        dendrite_features, spine_features = self.unet(patch)
        self.dendrite_features = dendrite_features
        self.spine_features = spine_features

    def sample(self, testing=False):
        """
        Sample segmentation by fusing latent with U-Net features.
        """
        if not testing:
            z_prior = self.prior_latent_space.rsample()
        else:
            z_prior = self.prior_latent_space.sample()
        
        self.z_prior_sample = z_prior

        dendrites = self.fcomb_dendrites(self.dendrite_features, z_prior)
        spines = self.fcomb_spines(self.spine_features, z_prior)
        
        return dendrites, spines

    def reconstruct(self, use_posterior_mean=False, calculate_posterior=False, z_posterior=None, training=True):
        """
        Reconstruct segmentation from latent space.
        """
        if self.posterior_latent_space is not None:
            if use_posterior_mean:
                z_posterior = self.posterior_latent_space.loc
            elif calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
        else:
            z_posterior = self.prior_latent_space.rsample()

        dendrites = self.fcomb_dendrites(self.dendrite_features, z_posterior)
        spines = self.fcomb_spines(self.spine_features, z_posterior)
        
        return dendrites, spines

    def kl_divergence(self, analytic=True, calculate_posterior=False, z_posterior=None):
        """
        Calculate KL divergence between posterior and prior.
        """
        if analytic:
            kl_div = kl.kl_divergence(self.posterior_latent_space, self.prior_latent_space)
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
            log_posterior_prob = self.posterior_latent_space.log_prob(z_posterior)
            log_prior_prob = self.prior_latent_space.log_prob(z_posterior)
            kl_div = log_posterior_prob - log_prior_prob
        
        return kl_div

    def elbo(self, segm_d, segm_s, analytic_kl=True, reconstruct_posterior_mean=False):
        """
        Calculate evidence lower bound (negative log-likelihood).
        """
        criterion = nn.BCEWithLogitsLoss(reduction='sum')
        z_posterior = self.posterior_latent_space.rsample()

        self.kl = torch.mean(
            self.kl_divergence(analytic=analytic_kl, calculate_posterior=False, z_posterior=z_posterior)
        )
        
        dendrites_rec, spines_rec = self.reconstruct(
            use_posterior_mean=reconstruct_posterior_mean,
            calculate_posterior=False, 
            z_posterior=z_posterior
        )

        segm_dendrites = segm_d
        segm_spines = segm_s

        loss_dendrites = criterion(dendrites_rec, segm_dendrites)
        loss_spines = criterion(spines_rec, segm_spines)
        reconstruction_loss = loss_dendrites + loss_spines

        epsilon = 1e-7
        self.reconstruction_loss = torch.sum(reconstruction_loss + epsilon)
        self.mean_reconstruction_loss = torch.mean(reconstruction_loss + epsilon)

        return -(self.reconstruction_loss + self.beta * self.kl)