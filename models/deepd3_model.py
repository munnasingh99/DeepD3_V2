"""
DeepD3 U-Net model with dual decoders for dendrites and spines.

Architecture:
- Single encoder with residual connections
- Dual decoders (one for dendrites, one for spines)
- Skip connections from encoder to both decoders
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    """
    Encoder block with residual connection.
    
    Structure:
    - 1x1 conv for identity mapping
    - Two 3x3 convs with normalization and activation
    - Residual addition
    - Max pooling
    """
    def __init__(self, in_channels, out_channels, activation, use_batchnorm=True):
        super().__init__()
        self.identity_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(8, out_channels) if use_batchnorm else nn.Identity()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(8, out_channels) if use_batchnorm else nn.Identity()
        
        self.activation = activation
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        identity = self.identity_conv(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x + identity)
        
        pooled = self.pool(x)
        return x, pooled


class DecoderBlock(nn.Module):
    """
    Decoder block with two 3x3 convolutions.
    """
    def __init__(self, in_channels, out_channels, activation, use_batchnorm=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(8, out_channels) if use_batchnorm else nn.Identity()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(8, out_channels) if use_batchnorm else nn.Identity()
        
        self.activation = activation

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        
        return x


class Decoder(nn.Module):
    """
    Decoder module with skip connections from encoder.
    
    For each level:
    1. Upsample latent features
    2. Concatenate with encoder features
    3. Apply decoder block
    """
    def __init__(self, num_layers, base_filters, activation, use_batchnorm=True):
        super().__init__()
        self.num_layers = num_layers
        self.blocks = nn.ModuleList()
        
        for i in range(num_layers):
            k = num_layers - 1 - i
            if i == 0:
                in_ch = base_filters * (2 ** num_layers) + base_filters * (2 ** (num_layers - 1))
            else:
                in_ch = base_filters * (2 ** (k + 1)) + base_filters * (2 ** k)
            out_ch = base_filters * (2 ** k)
            self.blocks.append(DecoderBlock(in_ch, out_ch, activation, use_batchnorm))
        
        self.last_layer_features = None
        self.final_conv = nn.Conv2d(base_filters, 1, kernel_size=1)
        self.final_activation = nn.Sigmoid()
        
    def forward(self, x, encoder_features):
        """
        Args:
            x: Latent representation
            encoder_features: List of encoder skip connections (deepest first)
        """
        for block in self.blocks:
            enc_feat = encoder_features.pop()
            
            x = F.interpolate(
                x, 
                size=enc_feat.shape[2:], 
                mode='bilinear', 
                align_corners=True
            )
            
            x = torch.cat([x, enc_feat], dim=1)
            x = block(x)
        
        self.last_layer_features = x
        x = self.final_conv(x)
        x = self.final_activation(x)
        
        return x


class DeepD3Model(nn.Module):
    """
    U-Net with one encoder and dual decoders for dendrites and spines.
    
    Args:
        in_channels: Number of input channels (default: 1 for grayscale)
        base_filters: Base number of filters (doubled at each level)
        num_layers: Network depth (number of encoder/decoder blocks)
        activation: Activation function ('swish' or 'relu')
        use_batchnorm: Whether to use batch normalization
        apply_last_layer: Whether to apply final 1x1 conv and sigmoid
    """
    def __init__(
        self, 
        in_channels=1, 
        base_filters=32, 
        num_layers=4, 
        activation="swish", 
        use_batchnorm=True,
        apply_last_layer=True
    ):
        super().__init__()
        
        self.apply_last_layer = apply_last_layer
        
        if activation == "swish":
            act = nn.SiLU(inplace=True)
        else:
            act = nn.ReLU(inplace=True)
        
        self.activation = act
        self.num_layers = num_layers
        self.base_filters = base_filters
        self.use_batchnorm = use_batchnorm
        
        self.encoder_blocks = nn.ModuleList()
        current_in = in_channels
        for i in range(num_layers):
            out_channels = base_filters * (2 ** i)
            self.encoder_blocks.append(
                EncoderBlock(current_in, out_channels, self.activation, use_batchnorm)
            )
            current_in = out_channels
            
        latent_in = base_filters * (2 ** (num_layers - 1))
        latent_out = base_filters * (2 ** num_layers)
        self.latent_conv = nn.Sequential(
            nn.Conv2d(latent_in, latent_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(latent_out) if use_batchnorm else nn.Identity(),
            self.activation
        )
        
        self.decoder_dendrites = Decoder(num_layers, base_filters, self.activation, use_batchnorm)
        self.decoder_spines = Decoder(num_layers, base_filters, self.activation, use_batchnorm)
        
    def forward(self, x):
        """
        Forward pass through encoder and dual decoders.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Tuple of (dendrite_output, spine_output)
        """
        encoder_features = []
        
        for block in self.encoder_blocks:
            feat, x = block(x)
            encoder_features.append(feat)
        
        enc_feats_d = encoder_features.copy()
        enc_feats_s = encoder_features.copy()
        
        x_latent = self.latent_conv(x)
        
        dendrites_features = self.decoder_dendrites.forward(x_latent, enc_feats_d)
        spines_features = self.decoder_spines.forward(x_latent, enc_feats_s)
        
        if self.apply_last_layer:
            return dendrites_features, spines_features
        else:
            return (
                self.decoder_dendrites.last_layer_features, 
                self.decoder_spines.last_layer_features
            )


if __name__ == '__main__':
    x = torch.randn(1, 1, 48, 48)
    model = DeepD3Model(
        in_channels=1, 
        base_filters=32, 
        num_layers=4, 
        activation="swish",
        apply_last_layer=False
    )
    dendrites, spines = model(x)
    print("Dendrites output shape:", dendrites.shape)
    print("Spines output shape:", spines.shape)