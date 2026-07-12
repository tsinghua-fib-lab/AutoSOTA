# -*- coding: utf-8 -*-
"""
scChord Model Definitions

This module contains:
1. ProteinVAE - Protein Variational Autoencoder
2. RNAEncoder - RNA Condition Encoder
3. FlowNet - Conditional Flow Matching Network (DiT-style AdaLN)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


# ===========================================================================
# Distribution Utility Functions (referenced from scvi-tools and scVAEIT)
# ===========================================================================

def log_nb_positive(
    x: torch.Tensor, 
    mu: torch.Tensor, 
    theta: torch.Tensor, 
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute log probability of Negative Binomial distribution (referenced from scvi-tools).
    
    NB(x | mu, theta) parameterization:
    - mu: Mean (must be positive)
    - theta: Inverse dispersion parameter (must be positive, larger values approach Poisson)
    
    Args:
        x: Observed data (can be float) [B, M]
        mu: Mean of NB distribution (must be positive) [B, M]
        theta: Inverse dispersion parameter (must be positive) [M] or [B, M]
        eps: Numerical stability constant
        
    Returns:
        log_prob: Log probability [B, M]
    """
    # If theta is 1D, expand to 2D
    if theta.ndimension() == 1:
        theta = theta.unsqueeze(0)  # [1, M]
    
    log_theta_mu_eps = torch.log(theta + mu + eps)
    res = (
        theta * (torch.log(theta + eps) - log_theta_mu_eps)
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )
    return res


def log_zinb_positive(
    x: torch.Tensor, 
    mu: torch.Tensor, 
    theta: torch.Tensor, 
    pi: torch.Tensor, 
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute log probability of Zero-Inflated Negative Binomial distribution 
    (referenced from scvi-tools).
    
    Args:
        x: Observed data [B, M]
        mu: Mean of NB distribution [B, M]
        theta: Inverse dispersion parameter [M] or [B, M]
        pi: Logit of dropout parameter (logit of zero-inflation probability) [B, M]
        eps: Numerical stability constant
        
    Returns:
        log_prob: Log probability [B, M]
    """
    # If theta is 1D, expand to 2D
    if theta.ndimension() == 1:
        theta = theta.unsqueeze(0)  # [1, M]

    # Uses log(sigmoid(x)) = -softplus(-x)
    softplus_pi = F.softplus(-pi)
    log_theta_eps = torch.log(theta + eps)
    log_theta_mu_eps = torch.log(theta + mu + eps)
    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

    case_zero = F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

    case_non_zero = (
        -softplus_pi
        + pi_theta_log
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )
    mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

    res = mul_case_zero + mul_case_non_zero
    return res


# ===========================================================================
# Helper Modules
# ===========================================================================

def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Sinusoidal positional encoding (for time embedding).
    
    Args:
        t: Time tensor [B], range [0, 1]
        dim: Embedding dimension
        
    Returns:
        emb: Time embedding [B, dim]
    """
    half_dim = dim // 2
    emb_scale = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb_scale)
    emb = t.unsqueeze(-1) * emb.unsqueeze(0)  # [B, half_dim]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # [B, dim]
    return emb


class AdaLNBlock(nn.Module):
    """
    Adaptive Layer Normalization Residual Block (DiT-style).
    
    Uses condition vector to modulate LayerNorm's scale and shift.
    """
    
    def __init__(self, hidden_dim: int, cond_dim: int, dropout: float = 0.1):
        """
        Args:
            hidden_dim: Hidden layer dimension
            cond_dim: Condition vector dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        
        # Condition projection: generate scale, shift for norm1 and norm2
        # Total 6 * hidden_dim: (scale1, shift1, scale2, shift2, gate1, gate2)
        self.cond_proj = nn.Linear(cond_dim, 6 * hidden_dim)
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, hidden_dim]
            cond: Condition vector [B, cond_dim]
            
        Returns:
            out: Output tensor [B, hidden_dim]
        """
        # Generate modulation parameters
        params = self.cond_proj(cond)  # [B, 6 * hidden_dim]
        scale1, shift1, scale2, shift2, gate1, gate2 = params.chunk(6, dim=-1)
        
        # First sub-layer (with residual)
        h = self.norm1(x)
        h = h * (1 + scale1) + shift1
        h = self.fc1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.dropout(h)
        x = x + gate1.tanh() * h
        
        # Second sub-layer (with residual)
        h = self.norm2(x)
        h = h * (1 + scale2) + shift2
        h = self.fc1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.dropout(h)
        x = x + gate2.tanh() * h
        
        return x


# ===========================================================================
# ProteinVAE
# ===========================================================================

class ProteinVAE(nn.Module):
    """
    Protein Variational Autoencoder.
    
    Input: Preprocessed protein expression + batch_id
    Output: Reconstructed protein expression
    Loss: Gaussian NLL or NB/ZINB NLL + KL divergence
    
    Supported distribution types:
    - 'Gaussian': Gaussian distribution, suitable for log-normalized continuous values
    - 'NB': Negative Binomial distribution, suitable for raw count data (ref: scvi-tools/scVAEIT)
    - 'ZINB': Zero-Inflated Negative Binomial, suitable for highly sparse count data
    """
    
    def __init__(
        self,
        n_proteins: int,
        dz: int = 32,
        hidden_dims: list = [256, 256],
        batch_emb_dim: int = 8,
        n_batches: int = 2,
        beta_kl: float = 1.0,
        learnable_dispersion: bool = True,
        dist_type: str = 'Gaussian'
    ):
        """
        Args:
            n_proteins: Number of proteins (M)
            dz: Latent dimension
            hidden_dims: Encoder/decoder hidden layer dimensions
            batch_emb_dim: Batch embedding dimension
            n_batches: Number of batches
            beta_kl: KL loss weight
            learnable_dispersion: Whether to learn per-protein std/dispersion parameter
            dist_type: Distribution type: 'Gaussian', 'NB', or 'ZINB'
        """
        super().__init__()
        
        assert dist_type in ['Gaussian', 'NB', 'ZINB'], f"dist_type must be 'Gaussian', 'NB' or 'ZINB', got {dist_type}"
        
        self.n_proteins = n_proteins
        self.dz = dz
        self.beta_kl = beta_kl
        self.learnable_dispersion = learnable_dispersion
        self.dist_type = dist_type
        
        # Batch embedding
        self.batch_emb = nn.Embedding(n_batches, batch_emb_dim)
        
        # Encoder
        enc_dims = [n_proteins + batch_emb_dim] + hidden_dims
        enc_layers = []
        for i in range(len(enc_dims) - 1):
            enc_layers.extend([
                nn.Linear(enc_dims[i], enc_dims[i + 1]),
                nn.LayerNorm(enc_dims[i + 1]),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], dz)
        self.fc_logvar = nn.Linear(hidden_dims[-1], dz)
        
        # Decoder
        dec_dims = [dz + batch_emb_dim] + hidden_dims[::-1] + [n_proteins]
        dec_layers = []
        for i in range(len(dec_dims) - 2):
            dec_layers.extend([
                nn.Linear(dec_dims[i], dec_dims[i + 1]),
                nn.LayerNorm(dec_dims[i + 1]),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
        dec_layers.append(nn.Linear(dec_dims[-2], dec_dims[-1]))
        self.decoder = nn.Sequential(*dec_layers)
        
        # Distribution parameters
        if dist_type == 'Gaussian':
            # Learnable standard deviation (one per protein)
            if learnable_dispersion:
                self.sigma_param = nn.Parameter(torch.zeros(n_proteins))
            else:
                self.register_buffer('sigma', torch.ones(1))
        elif dist_type in ['NB', 'ZINB']:
            # Learnable dispersion parameter (one per protein)
            # theta = 1 / dispersion is the inverse dispersion
            # Larger dispersion means stronger overdispersion
            if learnable_dispersion:
                # Initialize to 0, softplus gives ~0.69
                self.disp_param = nn.Parameter(torch.zeros(n_proteins))
            else:
                self.register_buffer('dispersion', torch.ones(n_proteins))
            
            # NB/ZINB distribution scale factor (for converting sigmoid output to mean)
            # Consistent with scVAEIT: mu = sigmoid(output) * log(1e4 + 1)
            self.register_buffer('nb_scale', torch.tensor(np.log(1e4 + 1.)))
            
            # ZINB-specific: zero-inflation probability logit prediction head
            if dist_type == 'ZINB':
                # Predict pi_logit from decoder's second-to-last layer input dimension
                self.pi_decoder = nn.Linear(hidden_dims[0], n_proteins)
            
    def encode(
        self, 
        y_prot: torch.Tensor, 
        batch_id: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode protein expression to latent space.
        
        Args:
            y_prot: Protein expression [B, M]
            batch_id: Batch identifiers [B]
            
        Returns:
            mu: Mean [B, dz]
            logvar: Log variance [B, dz]
        """
        batch_emb = self.batch_emb(batch_id)  # [B, batch_emb_dim]
        x = torch.cat([y_prot, batch_emb], dim=-1)  # [B, M + batch_emb_dim]
        h = self.encoder(x)  # [B, hidden]
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(
        self, 
        mu: torch.Tensor, 
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Reparameterization trick.
        
        Args:
            mu: Mean [B, dz]
            logvar: Log variance [B, dz]
            
        Returns:
            z: Sampled latent variable [B, dz]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(
        self, 
        z: torch.Tensor, 
        batch_id: torch.Tensor,
        return_raw: bool = False,
        return_pi: bool = False
    ) -> torch.Tensor:
        """
        Decode latent variable to protein expression.
        
        Args:
            z: Latent variable [B, dz]
            batch_id: Batch identifiers [B]
            return_raw: If True, return raw network output (for NB/ZINB special handling)
            return_pi: If True and ZINB, return (y_hat, pi_logit)
            
        Returns:
            y_hat: Reconstructed protein [B, M]
                - Gaussian: Direct output
                - NB/ZINB: sigmoid output multiplied by scale to get mu
            pi_logit: Zero-inflation probability logit [B, M] (only when return_pi=True and dist_type='ZINB')
        """
        batch_emb = self.batch_emb(batch_id)  # [B, batch_emb_dim]
        x = torch.cat([z, batch_emb], dim=-1)  # [B, dz + batch_emb_dim]
        
        # Forward through decoder, need intermediate features for ZINB
        if self.dist_type == 'ZINB':
            # Manually execute decoder forward pass to get intermediate features
            h = x
            # Number of decoder layers: len(dec_dims) - 1 = len(hidden_dims[::-1]) + 2
            # Each layer has 4 sub-modules (Linear, LayerNorm, GELU, Dropout), last layer only has Linear
            n_layers = len(self.decoder)
            # Last layer is single Linear, preceding layers have 4 sub-modules each
            n_hidden_layers = (n_layers - 1) // 4
            
            # Execute until second-to-last layer to get intermediate features
            for i in range(n_hidden_layers * 4):
                h = self.decoder[i](h)
            h_intermediate = h  # Intermediate features for predicting pi
            
            # Final layer output
            y_hat = self.decoder[-1](h)
            
            # Predict zero-inflation probability logit
            pi_logit = self.pi_decoder(h_intermediate)
            
            if not return_raw:
                # ZINB distribution: use softplus for unbounded positive mean mu
                # (sigmoid * nb_scale is insufficient for CITE-seq protein counts that can exceed 100k)
                y_hat = F.softplus(y_hat)

            if return_pi:
                return y_hat, pi_logit
            return y_hat
        else:
            y_hat = self.decoder(x)

            if self.dist_type == 'NB' and not return_raw:
                # NB distribution: use softplus for unbounded positive mean mu
                y_hat = F.softplus(y_hat)
            
            return y_hat
    
    def get_sigma(self) -> torch.Tensor:
        """Get standard deviation (only for Gaussian distribution)."""
        if self.dist_type != 'Gaussian':
            raise ValueError("get_sigma() is only available for Gaussian distribution")
        if self.learnable_dispersion:
            sigma = F.softplus(self.sigma_param)
            sigma = sigma.clamp(1e-3, 10)
            return sigma
        else:
            return self.sigma
    
    def get_dispersion(self) -> torch.Tensor:
        """
        Get dispersion parameter (for NB/ZINB distribution).
        
        Returns:
            dispersion: Dispersion parameter [M], theta = 1/dispersion
        """
        if self.dist_type not in ['NB', 'ZINB']:
            raise ValueError("get_dispersion() is only available for NB/ZINB distribution")
        if self.learnable_dispersion:
            # Use softplus to ensure dispersion > 0
            # Clamp to reasonable range (ref scVAEIT: [0, 6])
            dispersion = F.softplus(self.disp_param)
            dispersion = dispersion.clamp(1e-4, 6.0)
            return dispersion
        else:
            return self.dispersion
    
    def get_theta(self) -> torch.Tensor:
        """
        Get inverse dispersion parameter theta (for NB/ZINB distribution).
        theta = 1 / dispersion
        """
        dispersion = self.get_dispersion()
        return 1.0 / (dispersion + 1e-8)
    
    def forward(
        self, 
        y_prot: torch.Tensor, 
        batch_id: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Returns:
            y_hat, mu, logvar, z, pi_logit (ZINB) or None (others)
        """
        mu, logvar = self.encode(y_prot, batch_id)
        z = self.reparameterize(mu, logvar)
        
        if self.dist_type == 'ZINB':
            y_hat, pi_logit = self.decode(z, batch_id, return_pi=True)
            return y_hat, mu, logvar, z, pi_logit
        else:
            y_hat = self.decode(z, batch_id)
            return y_hat, mu, logvar, z, None
    
    def loss(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        y_raw: Optional[torch.Tensor] = None,
        pi_logit: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss.
        
        Args:
            y: Ground truth protein [B, M]
                - Gaussian: log-normalized values
                - NB/ZINB: can be log-normalized values (requires y_raw)
            y_hat: Reconstructed protein [B, M]
                - Gaussian: direct prediction
                - NB/ZINB: predicted mean mu
            mu: Latent variable mean [B, dz]
            logvar: Latent variable log variance [B, dz]
            y_raw: Raw protein counts [B, M] (only required for NB/ZINB)
            pi_logit: Zero-inflation probability logit [B, M] (only required for ZINB)
            
        Returns:
            dict: loss_total, nll, kl
        """
        if self.dist_type == 'Gaussian':
            sigma = self.get_sigma()  # [M] or [1]
            
            # Gaussian NLL: 0.5 * ((y - y_hat)/sigma)^2 + log(sigma) + const
            var = sigma ** 2
            nll = 0.5 * ((y - y_hat) ** 2 / var + torch.log(var))  # [B, M]
            nll = nll.sum(dim=-1).mean()
            
        elif self.dist_type == 'NB':
            theta = self.get_theta()  # [M]
            
            # Use raw counts for NB loss
            if y_raw is not None:
                y_counts = y_raw
            else:
                # If raw counts not provided, assume y is raw counts
                y_counts = y
            
            # y_hat is already mu (obtained via sigmoid * scale in decode)
            log_prob = log_nb_positive(y_counts, y_hat, theta)  # [B, M]
            nll = -log_prob.sum(dim=-1).mean()  # Negative log likelihood
            
        else:  # ZINB
            theta = self.get_theta()  # [M]
            
            # Use raw counts for ZINB loss
            if y_raw is not None:
                y_counts = y_raw
            else:
                # If raw counts not provided, assume y is raw counts
                y_counts = y
            
            if pi_logit is None:
                raise ValueError("pi_logit is required for ZINB distribution")
            
            # y_hat is already mu (obtained via sigmoid * scale in decode)
            # pi_logit is the logit of zero-inflation probability
            log_prob = log_zinb_positive(y_counts, y_hat, theta, pi_logit)  # [B, M]
            nll = -log_prob.sum(dim=-1).mean()  # Negative log likelihood
        
        # KL: KL(N(mu, var) || N(0, I)) = 0.5 * (mu^2 + var - 1 - log(var))
        kl = 0.5 * (mu ** 2 + logvar.exp() - 1 - logvar)  # [B, dz]
        kl = kl.sum(dim=-1).mean()
        
        loss_total = nll + self.beta_kl * kl
        
        return {
            'loss_total': loss_total,
            'nll': nll,
            'kl': kl
        }
    
    def get_latent(
        self,
        y_prot: torch.Tensor,
        batch_id: torch.Tensor,
        deterministic: bool = False
    ) -> torch.Tensor:
        """
        Get latent representation.
        
        Args:
            y_prot: Protein expression [B, M]
            batch_id: Batch identifiers [B]
            deterministic: If True, return mu; otherwise sample
            
        Returns:
            z: Latent variable [B, dz]
        """
        mu, logvar = self.encode(y_prot, batch_id)
        if deterministic:
            return mu
        return self.reparameterize(mu, logvar)


# ===========================================================================
# RNAEncoder
# ===========================================================================

class RNAEncoder(nn.Module):
    """
    RNA Condition Encoder.
    
    Encodes each cell's HVG vector into a condition vector c_rna.
    """
    
    def __init__(
        self,
        n_genes: int,
        dc: int = 512,
        hidden_dims: list = [1024, 512],
        batch_emb_dim: int = 8,
        n_batches: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            n_genes: Number of genes (G)
            dc: Condition vector dimension
            hidden_dims: Hidden layer dimensions
            batch_emb_dim: Batch embedding dimension
            n_batches: Number of batches
            dropout: Dropout probability
        """
        super().__init__()
        
        self.n_genes = n_genes
        self.dc = dc
        
        # Batch embedding
        self.batch_emb = nn.Embedding(n_batches, batch_emb_dim)
        
        # Encoder
        dims = [n_genes + batch_emb_dim] + hidden_dims + [dc]
        layers = []
        for i in range(len(dims) - 2):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.encoder = nn.Sequential(*layers)
        
    def forward(
        self, 
        rna_norm: torch.Tensor, 
        batch_id: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            rna_norm: Preprocessed RNA [B, G]
            batch_id: Batch identifiers [B]
            
        Returns:
            c_rna: Condition vector [B, dc]
        """
        batch_emb = self.batch_emb(batch_id)  # [B, batch_emb_dim]
        x = torch.cat([rna_norm, batch_emb], dim=-1)  # [B, G + batch_emb_dim]
        c_rna = self.encoder(x)
        return c_rna


# ===========================================================================
# FlowNet
# ===========================================================================

class FlowNet(nn.Module):
    """
    Conditional Flow Matching Network.
    
    Predicts vector field v(x_t, t, c_rna, batch).
    Uses DiT-style AdaLN residual blocks.
    """
    
    def __init__(
        self,
        dz: int = 32,
        dc: int = 512,
        hidden_dim: int = 256,
        n_blocks: int = 4,
        time_emb_dim: int = 64,
        batch_emb_dim: int = 8,
        n_batches: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            dz: Latent variable dimension
            dc: Condition vector dimension
            hidden_dim: Hidden layer dimension
            n_blocks: Number of residual blocks
            time_emb_dim: Time embedding dimension
            batch_emb_dim: Batch embedding dimension
            n_batches: Number of batches
            dropout: Dropout probability
        """
        super().__init__()
        
        self.dz = dz
        self.dc = dc
        self.time_emb_dim = time_emb_dim
        
        # Batch embedding
        self.batch_emb = nn.Embedding(n_batches, batch_emb_dim)
        
        # Condition dimension: c_rna + time_emb + batch_emb
        cond_dim = dc + time_emb_dim + batch_emb_dim
        
        # Unconditional embedding (for CFG)
        self.cond_null = nn.Parameter(torch.randn(dc) * 0.01)
        
        # Input projection
        self.input_proj = nn.Linear(dz, hidden_dim)
        
        # AdaLN residual blocks
        self.blocks = nn.ModuleList([
            AdaLNBlock(hidden_dim, cond_dim, dropout)
            for _ in range(n_blocks)
        ])
        
        # Output projection
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, dz)
        
        # Initialize output layer to zero
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        c_rna: torch.Tensor,
        batch_id: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict vector field.
        
        Args:
            x_t: Current latent variable [B, dz]
            t: Time [B] (range 0~1)
            c_rna: RNA condition vector [B, dc]
            batch_id: Batch identifiers [B]
            
        Returns:
            v: Predicted vector field [B, dz]
        """
        # Time embedding
        t_emb = sinusoidal_embedding(t, self.time_emb_dim)  # [B, time_emb_dim]
        
        # Batch embedding
        batch_emb = self.batch_emb(batch_id)  # [B, batch_emb_dim]
        
        # Condition vector
        cond = torch.cat([c_rna, t_emb, batch_emb], dim=-1)  # [B, cond_dim]
        
        # Input projection
        h = self.input_proj(x_t)  # [B, hidden_dim]
        
        # Residual blocks
        for block in self.blocks:
            h = block(h, cond)
        
        # Output
        h = self.output_norm(h)
        v = self.output_proj(h)
        
        return v
    
    def get_cond_null(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Get unconditional embedding for CFG."""
        return self.cond_null.unsqueeze(0).expand(batch_size, -1)


# ===========================================================================
# Utility Functions
# ===========================================================================

def apply_gene_mask(
    rna_norm: torch.Tensor,
    mask_ratio_range: Tuple[float, float] = (0.2, 0.5)
) -> torch.Tensor:
    """
    Apply gene mask to RNA features.
    
    Args:
        rna_norm: Preprocessed RNA [B, G]
        mask_ratio_range: Mask ratio range (r_min, r_max)
        
    Returns:
        rna_masked: Masked RNA [B, G]
    """
    B, G = rna_norm.shape
    device = rna_norm.device
    
    # Sample mask ratio for each sample
    r_min, r_max = mask_ratio_range
    r = torch.rand(B, device=device) * (r_max - r_min) + r_min  # [B]
    k = (r * G).long()  # Number of genes to mask per sample
    
    # Generate mask
    mask = torch.ones(B, G, device=device)
    for i in range(B):
        indices = torch.randperm(G, device=device)[:k[i]]
        mask[i, indices] = 0
    
    return rna_norm * mask

