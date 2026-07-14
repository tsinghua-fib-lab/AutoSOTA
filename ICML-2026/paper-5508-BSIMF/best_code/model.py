import math
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from utils import LOG_VAR_MIN, LOG_VAR_MAX, MLP, ViTBackbone, CrossAttention


PROB_EPS = 1e-6
COV_FACTOR_CLIP = 50.0

class MaskDecoder(nn.Module):
    """
    g_theta(y,z) = sigmoid( f_upsampler( [ViT(y) || CA(z, ViT(y))] ) )

    - ViT(y) produces patch tokens of shape (B, N_patches, D).
    - CA(z, ViT(y)) produces another (B, N_patches, D).
    - We fuse them along the feature dimension -> (B, N_patches, 2D).
    - Reshape to (B, 2D, H_p, W_p).
    - A learnable upsampler f_upsampler : (B, 2D, H_p, W_p) -> (B, 1, H, W)
      via Conv2d + ConvTranspose2d (stride=patch_size).
    - Finally, apply sigmoid to get per-pixel Bernoulli probabilities.
    """

    def __init__(
        self,
        embed_dim: int,
        latent_dim: int,
        num_heads: int,
        image_size: int,
        patch_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.H_p = image_size // patch_size
        self.W_p = image_size // patch_size

        self.cross_attn = CrossAttention(
            embed_dim=embed_dim,
            latent_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # f_upsampler: Conv2d -> ReLU -> ConvTranspose2d
        up_in_channels = 2 * embed_dim
        up_hidden_channels = 2 * embed_dim

        self.upsampler = nn.Sequential(
            nn.Conv2d(
                in_channels=up_in_channels,
                out_channels=up_hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=up_hidden_channels,
                out_channels=1,
                kernel_size=patch_size,
                stride=patch_size,
                padding=0,
                output_padding=0,
            ),
        )

    def forward(self, vit_tokens: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vit_tokens: (B, N_patches, D) from ViT(y)
            z:         (B, d_Z)

        Returns:
            g_pixel_flat: (B, H*W) per-pixel Bernoulli probabilities.
        """
        B, N_patches, D = vit_tokens.shape

        # Cross-attention: CA(z, ViT(y))
        ca_tokens = self.cross_attn(vit_tokens, z)  # (B, N_patches, D)

        # Fuse along feature dimension: (B, N_patches, 2D)
        fused = torch.cat([vit_tokens, ca_tokens], dim=-1)

        # Reshape to (B, 2D, H_p, W_p)
        fused_grid = fused.view(B, self.H_p, self.W_p, 2 * D).permute(0, 3, 1, 2)
        # (B, 2D, H_p, W_p)

        # Upsample to pixel grid: (B, 1, H, W)
        logits_pixel = self.upsampler(fused_grid)  # (B, 1, H, W)

        logits_flat = logits_pixel.view(B, -1)  # (B, H*W)
        g_probs = torch.sigmoid(logits_flat).clamp(PROB_EPS, 1.0 - PROB_EPS)
        return g_probs  # (B, H*W)


class ImageDecoder(nn.Module):
    """
    Parameter-efficient image decoder using a slim seed MLP + convolutional upsampling.

    Architecture:
        z  →  seed MLP (z_dim → seed_hidden_dims → conv_ch × H₀ × W₀)
           →  ConvTranspose2d × n_stages (stride-2 each)
           →  Conv2d 1×1  →  (1+rank_r, H, W)

    Channel 0 of the output is the mean shift δμ; channels 1..rank_r are the
    columns of the low-rank covariance factor G_omega.
    """

    def __init__(
        self,
        z_dim: int,
        y_dim_flat: int,
        rank_r: int,
        hidden_dims: Sequence[int],
        # --- base distribution capacity controls ---
        base_mean_mode: str = "learnable_per_pixel",
        base_var_mode: str = "learnable_per_pixel",
        base_init_mean: float = 0.0,
        base_init_var: float = 1.0,
        # --- convolutional decoder width ---
        conv_channels: int = 32,
        # --- seed MLP hidden dims (separate from general mlp_hidden_dims) ---
        seed_hidden_dims: Sequence[int] = (24,),
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.y_dim_flat = y_dim_flat
        self.rank_r = rank_r

        image_size = int(math.isqrt(y_dim_flat))
        self._image_size = image_size

        self.base_mean_mode = str(base_mean_mode)
        self.base_var_mode = str(base_var_mode)
        base_init_log_var = float(math.log(base_init_var))

        # Base mean parameterization
        if self.base_mean_mode == "learnable_per_pixel":
            self.mu_base = nn.Parameter(torch.full((y_dim_flat,), float(base_init_mean)))
        elif self.base_mean_mode == "learnable_scalar":
            self.mu_base = nn.Parameter(torch.tensor([float(base_init_mean)], dtype=torch.float32))
        elif self.base_mean_mode == "fixed_zero":
            self.register_buffer("mu_base", torch.zeros(1, dtype=torch.float32))
        elif self.base_mean_mode == "fixed_scalar":
            self.register_buffer("mu_base", torch.tensor([float(base_init_mean)], dtype=torch.float32))

        # Base variance parameterization
        if self.base_var_mode == "learnable_per_pixel":
            self.log_var_base = nn.Parameter(torch.full((y_dim_flat,), base_init_log_var))
        elif self.base_var_mode == "learnable_scalar":
            self.log_var_base = nn.Parameter(torch.tensor([base_init_log_var], dtype=torch.float32))
        elif self.base_var_mode == "fixed_scalar":
            self.register_buffer("log_var_base", torch.tensor([base_init_log_var], dtype=torch.float32))
        elif self.base_var_mode == "fixed_per_pixel":
            self.register_buffer("log_var_base", torch.full((y_dim_flat,), base_init_log_var, dtype=torch.float32))

        # Convolutional decoder
        n_stages = 0
        while (image_size % (2 ** (n_stages + 1)) == 0 and
               image_size // (2 ** (n_stages + 1)) >= 4):
            n_stages += 1
        init_size = image_size // (2 ** n_stages)
        self._init_size = init_size
        self._conv_channels = conv_channels

        # Slim seed MLP: z → seed_hidden_dims → (conv_ch × init_size²)
        seed_dim = conv_channels * init_size * init_size
        self.seed_mlp = MLP(z_dim, seed_hidden_dims, seed_dim)

        # Upsampler: n_stages × (ConvTranspose2d + GELU) then a 1×1 Conv2d.
        # Final output has (1 + rank_r) channels: channel 0 = δμ, rest = G columns.
        out_ch = 1 + rank_r
        ups_layers: list = []
        for _ in range(n_stages):
            ups_layers += [
                nn.ConvTranspose2d(conv_channels, conv_channels,
                                   kernel_size=4, stride=2, padding=1),
                nn.GELU(),
            ]
        ups_layers.append(nn.Conv2d(conv_channels, out_ch, kernel_size=1))
        self.conv_upsampler = nn.Sequential(*ups_layers)

    def _broadcast_base_param(self, p: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Broadcast a base parameter tensor to shape (B, d_Y)."""
        if p.numel() == 1:
            return p.view(1, 1).expand(batch_size, self.y_dim_flat)
        return p.view(1, self.y_dim_flat).expand(batch_size, self.y_dim_flat)

    def base_params(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns mu_base and log_var_base broadcast to (B, d_Y)."""
        mu = self._broadcast_base_param(self.mu_base, batch_size)
        log_var = self._broadcast_base_param(self.log_var_base, batch_size)
        log_var = log_var.clamp(LOG_VAR_MIN, LOG_VAR_MAX)
        return mu, log_var

    def content_factor(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (B, d_Z)

        Returns:
            mu_cont:    (B, d_Y)
            cov_factor: (B, d_Y, r)  -- G_omega(z) in Sigma_cont = diag(var_base) + G G^T
        """
        B = z.size(0)

        # Project z to spatial seed and upsample
        seed = self.seed_mlp(z)                                        # (B, conv_ch × H₀²)
        x = seed.view(B, self._conv_channels, self._init_size, self._init_size)
        out = self.conv_upsampler(x)  # (B, 1+rank_r, image_size, image_size)

        # Flatten spatial dims and rearrange: (B, 1+rank_r, H*W) → (B, H*W, 1+rank_r)
        out_flat = out.reshape(B, 1 + self.rank_r, self.y_dim_flat).permute(0, 2, 1)

        delta_mu = out_flat[:, :, 0]                                           # (B, d_Y)
        cov_factor = out_flat[:, :, 1:].clamp(-COV_FACTOR_CLIP, COV_FACTOR_CLIP)  # (B, d_Y, r)

        mu_base, _ = self.base_params(batch_size=B)
        mu_cont = mu_base + delta_mu
        return mu_cont, cov_factor

    def content_params(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (B, d_Z)

        Returns:
            mu_cont:      (B, d_Y)
            log_var_cont: (B, d_Y)  -- log diagonal of Sigma_cont(z)
                where Sigma_cont(z) = diag(var_base) + G_omega(z) G_omega(z)^T.
        """
        B = z.size(0)
        mu_cont, cov_factor = self.content_factor(z)

        _, log_var_base = self.base_params(batch_size=B)      # (B, d_Y)
        var_base = log_var_base.exp()

        var_extra = cov_factor.pow(2).sum(dim=-1)             # (B, d_Y)
        var_cont = var_base + var_extra
        log_var_cont = torch.log(var_cont + 1e-8).clamp(LOG_VAR_MIN, LOG_VAR_MAX)
        return mu_cont, log_var_cont

class ContinuousLabelDecoder(nn.Module):
    """
    Continuous IRT-style decoder (simplified):

        p_phi(x | z, u) = N(A z + u, diag(sigma_X^2))

    """

    def __init__(self, z_dim: int, u_dim: int, x_dim: int) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.u_dim = u_dim
        self.x_dim = x_dim

        self.A = nn.Parameter(torch.randn(x_dim, z_dim) * 0.01)
        self.log_var = nn.Parameter(torch.zeros(x_dim))

    def forward(self, z: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (B, d_Z)
            u: (B, d_X)

        Returns:
            mean_x: (B, d_X)
            log_var_x: (B, d_X)  -- shared across batch
        """
        mean_x = z.matmul(self.A.t()) + u  # (B, d_X)
        log_var_x = self.log_var.clamp(LOG_VAR_MIN, LOG_VAR_MAX).unsqueeze(0).expand_as(mean_x)  # (B, d_X)
        return mean_x, log_var_x


class CategoricalLabelDecoder(nn.Module):
    """
    Categorical IRT-style decoder:

        p_phi(x | z, u) = prod_i Softmax_i(A_i z + B_i u)

    We assume the same number of classes for each dimension i.
    """

    def __init__(self, z_dim: int, u_dim: int, x_dim: int, num_classes: int) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.u_dim = u_dim
        self.x_dim = x_dim
        self.num_classes = num_classes

        # A: (d_X, C, d_Z), B: (d_X, C, d_U)
        self.A = nn.Parameter(torch.randn(x_dim, num_classes, z_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(x_dim, num_classes, u_dim) * 0.01)

    def logits(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, d_Z)
            u: (B, d_U)

        Returns:
            logits: (B, d_X, num_classes)
        """
        # (B, d_X, C) from z
        Az = torch.einsum("bd,kcd->bkc", z, self.A)
        Bu = torch.einsum("bd,kcd->bkc", u, self.B)
        logits = Az + Bu
        return logits


class UConditionalPrior(nn.Module):
    """
    Conditional prior p(u | y) = N(mu_psi, Sigma_psi(y)) with diagonal covariance.
    Mean mu_psi does NOT depend on y; only the covariance does.
    """

    def __init__(
        self,
        y_feat_dim: int,
        u_dim: int,
        hidden_dims: Sequence[int],
    ) -> None:
        super().__init__()
        self.register_buffer("mu_psi", torch.zeros(u_dim))
        # Only covariance depends on y
        self.log_var_net = MLP(y_feat_dim, hidden_dims, u_dim)

    def forward(self, y_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            y_feat: (B, D_y_feat) global image feature (e.g., ViT CLS token)

        Returns:
            mu_psi:      (B, d_U)  -- same for all y in the batch
            log_var_psi: (B, d_U)  -- depends on y
        """
        B = y_feat.size(0)
        # Broadcast the global mean to the batch
        mu = self.mu_psi.unsqueeze(0).expand(B, -1)      # (B, d_U)
        log_var = self.log_var_net(y_feat)               # (B, d_U)
        return mu, log_var


# -----------------------
# Full DAG model
# -----------------------


class ContentUncertaintyDAG(nn.Module):
    """
    Full DAG model described in the specification.

    Generative model:
        C ~ Categorical(pi)
        Z | C=k ~ N(mu_k, Sigma_k)     (diagonal covariance, vector per component)
        U | Y ~ N(mu_psi, Sigma_psi(Y))
        M_i ~ Ber(rho_i)
        Y | Z, M ~ p_omega(y | z, m)
        X | Z, U ~
            continuous: N(A z + U, diag(sigma_X^2))
            categorical: prod_i Softmax_i(A_i z + B_i u)

    Variational distributions (amortized mean-field):
        q(z | x, y) = N(h_z(x,y), diag(s_z^2(x,y)))
        q(u | x, y) = N(h_u(residual(x, z), y), diag(s_u^2(residual(x, z), y)))
        q(m | y, z) = prod_i Bern(g_theta(y,z)_i)

    """

    def __init__(
        self,
        x_dim: int,
        image_size: int,
        z_dim: int,
        u_dim: int,
        num_components: int,
        x_distribution: str = "continuous",
        x_num_classes: Optional[int] = None,
        vit_embed_dim: int = 128,
        vit_depth: int = 4,
        vit_num_heads: int = 4,
        vit_mlp_ratio: float = 4.0,
        vit_dropout: float = 0.0,
        mlp_hidden_dims: Sequence[int] = (128, 128),
        rank_r: int = 8,
        vit_patch_size: int = 1,   # patch size for ViT & g_theta
        mask_prior_rho: float = 0.01,
        learn_mask_prior: bool = False,
        # ---- encoder behavior under missing X ----
        use_mask_x_in_encoder: bool = False,
        u_missing_fallback_to_prior: bool = True,
        # ---- base distribution capacity (image background / noise) ----
        base_mean_mode: str = "learnable_per_pixel",
        base_var_mode: str = "learnable_per_pixel",
        base_init_mean: float = 0.0,
        base_init_var: float = 1.0,
        # ---- convolutional image decoder ----
        image_decoder_conv_channels: int = 32,
        image_decoder_seed_hidden_dims: Sequence[int] = (24,),
    ) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.image_size = image_size
        self.y_dim_flat = image_size * image_size  # d_Y^2
        self.z_dim = z_dim
        self.u_dim = u_dim
        self.num_components = num_components
        self.x_distribution = x_distribution
        self.x_num_classes = x_num_classes
        self.vit_patch_size = vit_patch_size

        # Encoder options for handling missing X
        self.use_mask_x_in_encoder = bool(use_mask_x_in_encoder)
        self.u_missing_fallback_to_prior = bool(u_missing_fallback_to_prior)

        # Shared backbones
        self.x_backbone = MLP(
            in_dim=x_dim,
            hidden_dims=mlp_hidden_dims,
            out_dim=mlp_hidden_dims[-1],
        )
        self.vit = ViTBackbone(
            image_size=image_size,
            patch_size=vit_patch_size,
            in_channels=1,
            embed_dim=vit_embed_dim,
            depth=vit_depth,
            num_heads=vit_num_heads,
            mlp_ratio=vit_mlp_ratio,
            dropout=vit_dropout,
        )

        feat_x_dim = mlp_hidden_dims[-1]
        feat_y_dim = vit_embed_dim

        # Variational q(z | x, y): mean and log-variance heads
        mask_feat_dim = x_dim if self.use_mask_x_in_encoder else 0
        z_feat_dim = feat_x_dim + feat_y_dim + mask_feat_dim
        self.h_z_net = MLP(
            in_dim=z_feat_dim,
            hidden_dims=mlp_hidden_dims,
            out_dim=z_dim,
        )
        self.log_var_z_net = MLP(
            in_dim=z_feat_dim,
            hidden_dims=mlp_hidden_dims,
            out_dim=z_dim,
        )

        # U residual backbone:
        # continuous: residual has shape (d_X,)
        # categorical: residual is one-hot - probs, shape (d_X * num_classes,)
        if x_distribution == "continuous":
            u_resid_in_dim = x_dim
        else:
            u_resid_in_dim = x_dim * x_num_classes

        self.u_resid_backbone = MLP(
            in_dim=u_resid_in_dim,
            hidden_dims=mlp_hidden_dims,
            out_dim=mlp_hidden_dims[-1],
        )

        # q(u | x, y): mean and variance heads take residual features + y features
        u_feat_dim = mlp_hidden_dims[-1] + feat_y_dim + mask_feat_dim
        self.h_u_net = MLP(
            in_dim=u_feat_dim,
            hidden_dims=mlp_hidden_dims,
            out_dim=u_dim,
        )
        self.log_var_u_net = MLP(
            in_dim=u_feat_dim,
            hidden_dims=mlp_hidden_dims,
            out_dim=u_dim,
        )

        # q(m | y, z) with shared ViT backbone, cross-attention, and learnable upsampler
        self.mask_net = MaskDecoder(
            embed_dim=feat_y_dim,
            latent_dim=z_dim,
            num_heads=vit_num_heads,
            image_size=image_size,
            patch_size=vit_patch_size,
            dropout=vit_dropout,
        )

        # Image decoder p_omega(y | z, m)
        self.image_decoder = ImageDecoder(
            z_dim=z_dim,
            y_dim_flat=self.y_dim_flat,
            rank_r=rank_r,
            hidden_dims=mlp_hidden_dims,
            base_mean_mode=base_mean_mode,
            base_var_mode=base_var_mode,
            base_init_mean=base_init_mean,
            base_init_var=base_init_var,
            conv_channels=image_decoder_conv_channels,
            seed_hidden_dims=image_decoder_seed_hidden_dims,
        )

        # Label decoder p_phi(x | z, u)
        if x_distribution == "continuous":
            self.x_decoder = ContinuousLabelDecoder(
                z_dim=z_dim, u_dim=u_dim, x_dim=x_dim
            )
        else:
            self.x_decoder = CategoricalLabelDecoder(
                z_dim=z_dim,
                u_dim=u_dim,
                x_dim=x_dim,
                num_classes=x_num_classes,
            )

        # Conditional prior p(u | y)
        self.u_prior_net = UConditionalPrior(
            y_feat_dim=feat_y_dim,
            u_dim=u_dim,
            hidden_dims=mlp_hidden_dims,
        )

        # Mixture prior p(z, c)
        self.pi_logits = nn.Parameter(torch.zeros(num_components))
        self.mu_components = nn.Parameter(torch.randn(num_components, z_dim) * 0.01)
        # log variances per component and dimension: (K, d_Z)
        self.log_var_components = nn.Parameter(torch.zeros(num_components, z_dim))

        # Mask prior p(m) = prod_i Ber(rho_i)
        rho = float(mask_prior_rho)
        rho = min(max(rho, PROB_EPS), 1.0 - PROB_EPS)
        init_logit = math.log(rho / (1.0 - rho))
        init_logits = torch.full((self.y_dim_flat,), init_logit, dtype=torch.float32)
        if learn_mask_prior:
            self.mask_logit_prior = nn.Parameter(init_logits)
        else:
            self.register_buffer("mask_logit_prior", init_logits)

    # ----- Convenience accessors -----

    @property
    def mixture_weights(self) -> torch.Tensor:
        """pi in the Categorical prior over C, shape (K,)."""
        return F.softmax(self.pi_logits, dim=0)

    @property
    def mixture_means(self) -> torch.Tensor:
        """Component means mu_k, shape (K, d_Z)."""
        return self.mu_components

    @property
    def mixture_log_vars(self) -> torch.Tensor:
        """Log variances log diag(Sigma_k), shape (K, d_Z)."""
        return self.log_var_components.clamp(LOG_VAR_MIN, LOG_VAR_MAX)

    @property
    def mask_prior_probs(self) -> torch.Tensor:
        """rho_i in p(m_i) = Ber(rho_i), shape (d_Y^2,)."""
        return torch.sigmoid(self.mask_logit_prior)

    # ----- Encoding / inference -----

    def encode(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask_x: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, d_X)
            y: (B, 1, image_size, image_size)
            mask_x: (B, d_X) float mask (1=observed, 0=missing)

        Returns:
            dict with:
                x_feat:    (B, D_x_feat)
                y_tokens:  (B, N_patches, D_y_feat)
                y_global:  (B, D_y_feat)
                mu_z:      (B, d_Z)
                log_var_z: (B, d_Z)
                mu_u:      (B, d_U)
                log_var_u: (B, d_U)
                mask_x:    (B, d_X)
        """
        B = x.size(0)

        # --------------------
        # Handle missingness in X
        # --------------------
        if self.x_distribution == "continuous":
            if mask_x is None:
                mask_x = torch.isfinite(x).to(dtype=x.dtype, device=x.device)
            else:
                mask_x = mask_x.to(dtype=x.dtype, device=x.device)
            x_in = torch.where(mask_x.bool(), x, torch.zeros_like(x))
        else:
            if mask_x is None:
                mask_x = (x >= 0).to(dtype=torch.float32, device=x.device)
            else:
                mask_x = mask_x.to(dtype=torch.float32, device=x.device)
            x_in = torch.where(mask_x.bool(), x, torch.zeros_like(x))

        # Backbone features
        x_feat = self.x_backbone(x_in.float())  # (B, feat_x_dim)
        y_tokens, y_global = self.vit(y)        # (B, N_patches, D_y_feat), (B, D_y_feat)

        # --------------------
        # q(z | x, y)
        # --------------------
        z_parts = [x_feat, y_global]
        if self.use_mask_x_in_encoder:
            z_parts.append(mask_x.to(dtype=x_feat.dtype))
        z_feat = torch.cat(z_parts, dim=-1)
        mu_z = self.h_z_net(z_feat)
        log_var_z = self.log_var_z_net(z_feat).clamp(LOG_VAR_MIN, LOG_VAR_MAX)

        # --------------------
        # q(u | x, y) via residuals
        # --------------------
        u_zero = torch.zeros(B, self.u_dim, device=x.device, dtype=mu_z.dtype)

        if self.x_distribution == "continuous":
            mean_x_z, _ = self.decode_x_continuous(mu_z, u_zero)  # (B, d_X)
            residual = (x_in - mean_x_z.detach()) * mask_x         # (B, d_X)
            residual_flat = residual.view(B, -1)
        else:
            logits_z = self.decode_x_categorical(mu_z, u_zero)     # (B, d_X, C)
            probs_z = F.softmax(logits_z, dim=-1)                  # (B, d_X, C)
            x_onehot = F.one_hot(x_in.long(), num_classes=self.x_num_classes).float()
            residual = (x_onehot - probs_z.detach()) * mask_x.unsqueeze(-1)  # (B, d_X, C)
            residual_flat = residual.view(B, -1)                   # (B, d_X*C)

        resid_feat = self.u_resid_backbone(residual_flat)          # (B, feat_x_dim)
        u_parts = [resid_feat, y_global]
        if self.use_mask_x_in_encoder:
            u_parts.append(mask_x.to(dtype=resid_feat.dtype))
        u_feat = torch.cat(u_parts, dim=-1)

        mu_u = self.h_u_net(u_feat)
        log_var_u = self.log_var_u_net(u_feat).clamp(LOG_VAR_MIN, LOG_VAR_MAX)

        if self.u_missing_fallback_to_prior and (self.x_distribution == "continuous"):
            if mask_x is not None and mask_x.ndim == 2 and mask_x.shape[1] == self.u_dim:
                mu_psi, log_var_psi = self.u_prior_net(y_global)
                log_var_psi = log_var_psi.clamp(LOG_VAR_MIN, LOG_VAR_MAX)
                m_bool = mask_x.bool()
                mu_u = torch.where(m_bool, mu_u, mu_psi)
                log_var_u = torch.where(m_bool, log_var_u, log_var_psi)

        return {
            "x_feat": x_feat,
            "y_tokens": y_tokens,
            "y_global": y_global,
            "mu_z": mu_z,
            "log_var_z": log_var_z,
            "mu_u": mu_u,
            "log_var_u": log_var_u,
            "mask_x": mask_x,
        }

    def prior_u(
        self,
        y: torch.Tensor,
        y_global: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Conditional prior parameters for p(u | y).

        Args:
            y: (B, 1, image_size, image_size)
            y_global: optional (B, D_y_feat)

        Returns:
            mu_psi: (B, d_U)
            log_var_psi: (B, d_U)
        """
        if y_global is None:
            _, y_global = self.vit(y)
        mu_psi, log_var_psi = self.u_prior_net(y_global)
        return mu_psi, log_var_psi

    def mask_probs(self, y_tokens: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        q(m | y, z) Bernoulli probabilities g_theta(y,z), at PIXEL level.

        Args:
            y_tokens: (B, N_patches, D_y_feat)
            z:        (B, d_Z)

        Returns:
            g_pixel_flat: (B, image_size * image_size)
        """
        return self.mask_net(y_tokens, z)

    # ----- Decoders -----

    def image_base_params(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience wrapper for mu_base, log_var_base.
        """
        return self.image_decoder.base_params(batch_size)

    def image_content_params(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        mu_cont(z), log_var_cont(z).
        """
        return self.image_decoder.content_params(z)

    def image_content_factor(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """\
        mu_cont(z) and the low-rank covariance factor G_omega(z).

        Returns:
            mu_cont:    (B, d_Y^2)
            cov_factor: (B, d_Y^2, r)
        """
        return self.image_decoder.content_factor(z)

    def decode_x_continuous(
        self, z: torch.Tensor, u: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mean and log-variance for continuous labels.
        """
        return self.x_decoder(z, u)

    def decode_x_categorical(
        self, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """
        Logits for categorical labels.
        """
        return self.x_decoder.logits(z, u)

    # ----- Forward -----

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, mask_x: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        return self.encode(x, y, mask_x=mask_x)
