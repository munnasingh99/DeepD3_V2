"""
Enhanced Probabilistic U-Net with dual latent spaces and Tversky loss.

Features:
- Separate latent spaces for dendrites and spines
- Tversky/Focal-Tversky loss for handling class imbalance
- Temperature scaling support
- KL divergence with optional annealing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl
from deepd3_model import DeepD3Model
from prob_unet_deepd3 import AxisAlignedConvGaussian, Fcomb


class TverskyLoss(nn.Module):
    """
    Tversky / Focal-Tversky loss on logits.
    
    Args:
        alpha: False positive weight (higher = penalize FP more)
        beta: False negative weight (higher = penalize FN more)
        gamma: Focal exponent (>1 for focal behavior)
        eps: Numerical stability constant
    """
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.0, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        tp = (p * target).sum()
        fp = (p * (1.0 - target)).sum()
        fn = ((1.0 - p) * target).sum()
        
        tversky_index = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        loss = (1.0 - tversky_index) ** self.gamma
        
        return loss


class ProbabilisticUnetDualLatent(nn.Module):
    """
    Probabilistic U-Net with separate latent spaces for dendrites and spines.
    
    Key features:
    - Dual latent spaces allow independent uncertainty modeling
    - Flexible reconstruction loss (BCE or Tversky)
    - KL annealing support via beta parameters
    - Temperature scaling for calibration
    """

    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 1,
        num_filters = (32, 64, 128, 192),
        latent_dim_dendrite: int = 8,
        latent_dim_spine: int = 8,
        no_convs_fcomb: int = 4,
        beta_dendrite: float = 1.0,
        beta_spine: float = 1.0,
        loss_weight_dendrite: float = 1.0,
        loss_weight_spine: float = 1.0,
        recon_loss: str = "tversky",
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        tversky_gamma: float = 1.0,
        bce_reduction: str = "mean",
        activation: str = "swish",
        use_batchnorm: bool = True,
        apply_last_layer: bool = False,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = list(num_filters)
        self.latent_dim_dendrite = latent_dim_dendrite
        self.latent_dim_spine = latent_dim_spine
        self.no_convs_per_block = 3
        self.no_convs_fcomb = no_convs_fcomb
        self.initializers = {"w": "he_normal", "b": "normal"}

        self.beta_dendrite = float(beta_dendrite)
        self.beta_spine = float(beta_spine)
        self.loss_weight_dendrite = float(loss_weight_dendrite)
        self.loss_weight_spine = float(loss_weight_spine)

        self.recon_loss_kind = recon_loss.lower()
        if self.recon_loss_kind == "bce":
            self.recon_criterion = nn.BCEWithLogitsLoss(reduction=bce_reduction)
        elif self.recon_loss_kind == "tversky":
            self.recon_criterion = TverskyLoss(
                alpha=tversky_alpha, 
                beta=tversky_beta, 
                gamma=tversky_gamma
            )
        else:
            raise ValueError("recon_loss must be 'bce' or 'tversky'")

        self.unet = DeepD3Model(
            in_channels=self.input_channels,
            base_filters=self.num_filters[0],
            num_layers=len(self.num_filters),
            activation=activation,
            use_batchnorm=use_batchnorm,
            apply_last_layer=apply_last_layer,
        )

        self.prior_dendrite = self._create_prior_posterior(
            self.latent_dim_dendrite, posterior=False, segm_channels=1
        )
        self.posterior_dendrite = self._create_prior_posterior(
            self.latent_dim_dendrite, posterior=True, segm_channels=1
        )
        self.prior_spine = self._create_prior_posterior(
            self.latent_dim_spine, posterior=False, segm_channels=1
        )
        self.posterior_spine = self._create_prior_posterior(
            self.latent_dim_spine, posterior=True, segm_channels=1
        )

        feat_c = self.num_filters[0]
        self.fcomb_dendrites = self._create_fcomb(
            latent_dim=self.latent_dim_dendrite,
            feature_channels=feat_c
        )
        self.fcomb_spines = self._create_fcomb(
            latent_dim=self.latent_dim_spine,
            feature_channels=feat_c
        )

        self.dendrite_features = None
        self.spine_features = None
        self.kl_dendrite = torch.tensor(0.0)
        self.kl_spine = torch.tensor(0.0)
        self.reconstruction_loss = torch.tensor(0.0)
        self.mean_reconstruction_loss = torch.tensor(0.0)

    def _create_prior_posterior(self, latent_dim, posterior=False, segm_channels=1):
        """Create prior or posterior network."""
        return AxisAlignedConvGaussian(
            self.input_channels,
            self.num_filters,
            self.no_convs_per_block,
            latent_dim,
            self.initializers,
            posterior=posterior,
            segm_channels=segm_channels,
        )

    def _create_fcomb(self, latent_dim, feature_channels: int):
        """Create feature combination network."""
        return Fcomb(
            self.num_filters,
            latent_dim,
            feature_channels,
            self.num_classes,
            self.no_convs_fcomb,
            {"w": "orthogonal", "b": "normal"},
            use_tile=True,
        )

    def set_beta(self, beta_d: float = None, beta_s: float = None):
        """Update KL weights for annealing."""
        if beta_d is not None:
            self.beta_dendrite = float(beta_d)
        if beta_s is not None:
            self.beta_spine = float(beta_s)

    def forward(self, patch, segm_dendrite=None, segm_spine=None, training=True):
        """
        Forward pass through the network.
        
        Args:
            patch: Input image tensor
            segm_dendrite: Dendrite ground truth (training only)
            segm_spine: Spine ground truth (training only)
            training: Whether in training mode
            
        Returns:
            Tuple of (dendrite_features, spine_features)
        """
        self.dendrite_features, self.spine_features = self.unet(patch)

        if training:
            if segm_dendrite is None or segm_spine is None:
                raise ValueError("Ground truth required in training mode")
            self.posterior_latent_dendrite = self.posterior_dendrite.forward(
                patch, segm_dendrite
            )
            self.posterior_latent_spine = self.posterior_spine.forward(
                patch, segm_spine
            )

        self.prior_latent_dendrite = self.prior_dendrite.forward(patch)
        self.prior_latent_spine = self.prior_spine.forward(patch)

        return self.dendrite_features, self.spine_features

    @torch.no_grad()
    def predict_proba(self, patch, n_samples: int = 1, use_posterior: bool = False):
        """
        Predict probabilities by averaging over multiple samples.
        
        Args:
            patch: Input image tensor
            n_samples: Number of samples to average
            use_posterior: Whether to use posterior (if available)
            
        Returns:
            Tuple of (dendrite_probs, spine_probs)
        """
        self.forward(patch, training=False)
        pd_list, ps_list = [], []
        
        for _ in range(max(1, n_samples)):
            d_logit, s_logit = self.sample(
                testing=not use_posterior, 
                use_posterior=use_posterior
            )
            pd_list.append(torch.sigmoid(d_logit))
            ps_list.append(torch.sigmoid(s_logit))
        
        pd_mean = torch.stack(pd_list, 0).mean(0)
        ps_mean = torch.stack(ps_list, 0).mean(0)
        
        return pd_mean, ps_mean

    def sample(self, testing: bool = False, use_posterior: bool = False):
        """
        Sample logits from the model.
        
        Args:
            testing: Use sample() instead of rsample() (no gradient)
            use_posterior: Use posterior if available, else prior
            
        Returns:
            Tuple of (dendrite_logits, spine_logits)
        """
        if use_posterior and hasattr(self, "posterior_latent_dendrite"):
            dist_d = self.posterior_latent_dendrite
            dist_s = self.posterior_latent_spine
        else:
            dist_d = self.prior_latent_dendrite
            dist_s = self.prior_latent_spine

        if testing:
            z_d = dist_d.sample()
            z_s = dist_s.sample()
        else:
            z_d = dist_d.rsample()
            z_s = dist_s.rsample()

        dendrites = self.fcomb_dendrites(self.dendrite_features, z_d)
        spines = self.fcomb_spines(self.spine_features, z_s)
        
        return dendrites, spines

    def reconstruct(self, use_posterior_mean: bool = False):
        """
        Reconstruct logits from posterior latent spaces.
        
        Args:
            use_posterior_mean: Use mean of posterior instead of sampling
            
        Returns:
            Tuple of (dendrite_logits, spine_logits)
        """
        if not hasattr(self, "posterior_latent_dendrite"):
            raise RuntimeError("Posterior not available. Call forward() with training=True first")

        if use_posterior_mean:
            z_d = self.posterior_latent_dendrite.loc
            z_s = self.posterior_latent_spine.loc
        else:
            z_d = self.posterior_latent_dendrite.rsample()
            z_s = self.posterior_latent_spine.rsample()

        dendrites = self.fcomb_dendrites(self.dendrite_features, z_d)
        spines = self.fcomb_spines(self.spine_features, z_s)
        
        return dendrites, spines

    def _recon_loss(self, pred_logits, target):
        """Compute reconstruction loss using configured criterion."""
        return self.recon_criterion(pred_logits, target)

    def kl_divergence(self):
        """
        Compute KL divergence per sample for both latent spaces.
        
        Returns:
            Tuple of (kl_dendrite, kl_spine) tensors
        """
        if not hasattr(self, "posterior_latent_dendrite"):
            raise RuntimeError("KL requires posteriors. Call forward() with training=True first")

        kl_d = kl.kl_divergence(
            self.posterior_latent_dendrite, 
            self.prior_latent_dendrite
        )
        kl_s = kl.kl_divergence(
            self.posterior_latent_spine, 
            self.prior_latent_spine
        )
        
        return kl_d, kl_s

    def elbo(self, segm_d: torch.Tensor, segm_s: torch.Tensor):
        """
        Compute negative ELBO (the loss to minimize).
        
        ELBO = Reconstruction Loss + beta * KL Divergence
        
        Args:
            segm_d: Dendrite ground truth
            segm_s: Spine ground truth
            
        Returns:
            Total loss (negative ELBO)
        """
        dendrites_rec, spines_rec = self.reconstruct(use_posterior_mean=False)

        loss_d = self._recon_loss(dendrites_rec, segm_d)
        loss_s = self._recon_loss(spines_rec, segm_s)
        weighted_recon = (
            self.loss_weight_dendrite * loss_d + 
            self.loss_weight_spine * loss_s
        )

        kl_d, kl_s = self.kl_divergence()
        self.kl_dendrite = kl_d.mean()
        self.kl_spine = kl_s.mean()

        self.reconstruction_loss = weighted_recon
        self.mean_reconstruction_loss = (
            weighted_recon if weighted_recon.ndim == 0 else weighted_recon.mean()
        )

        total = (
            weighted_recon + 
            self.beta_dendrite * self.kl_dendrite + 
            self.beta_spine * self.kl_spine
        )
        
        return total

    @torch.no_grad()
    def multisample(self, n: int = 8, use_posterior: bool = False):
        """
        Average probabilities over multiple samples.
        Assumes forward() has been called.
        
        Args:
            n: Number of samples
            use_posterior: Whether to use posterior
            
        Returns:
            Tuple of averaged (dendrite_probs, spine_probs)
        """
        pd, ps = [], []
        for _ in range(max(1, n)):
            ld, ls = self.sample(testing=True, use_posterior=use_posterior)
            pd.append(torch.sigmoid(ld))
            ps.append(torch.sigmoid(ls))
        
        return torch.stack(pd).mean(0), torch.stack(ps).mean(0)