# -*- coding: utf-8 -*-
"""
scBridge-Flow model definitions.
Contains:
1. ProteinVAE - variational autoencoder for protein expression.
2. RNAEncoder - RNA condition encoder.
3. FlowNet - conditional flow matching network (DiT-style AdaLN).
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


# ===========================================================================
# Distribution-related utility functions (inspired by scvi-tools/scVAEIT).
# ===========================================================================

def log_nb_positive(
    x: torch.Tensor, 
    mu: torch.Tensor, 
    theta: torch.Tensor, 
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute log-probability under the Negative Binomial distribution.

    NB(x | mu, theta) parameterization:
    - mu: mean (must be positive)
    - theta: inverse-dispersion (must be positive; larger is closer to Poisson)

    Parameters
    ----------
    x : torch.Tensor
        Observations (can be floating-point) [B, M].
    mu : torch.Tensor
        Negative Binomial mean (must be positive) [B, M].
    theta : torch.Tensor
        Inverse-dispersion parameter (must be positive) [M] or [B, M].
    eps : float
        Numerical stability constant.

    Returns
    ----------
    log_prob : torch.Tensor
        Log-probability [B, M].
    """
    # If theta is 1D, expand to 2D.
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
    Compute log-probability under the Zero-Inflated Negative Binomial distribution.

    Parameters
    ----------
    x : torch.Tensor
        Observations [B, M].
    mu : torch.Tensor
        Negative Binomial mean [B, M].
    theta : torch.Tensor
        Inverse-dispersion parameter [M] or [B, M].
    pi : torch.Tensor
        Dropout logits (logits of zero-inflation probability) [B, M].
    eps : float
        Numerical stability constant.

    Returns
    ----------
    log_prob : torch.Tensor
        Log-probability [B, M].
    """
    # If theta is 1D, expand to 2D.
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
# Helper modules
# ===========================================================================

def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Sinusoidal positional encoding for time embedding.

    Parameters
    ----------
    t : torch.Tensor
        Time [B], range [0, 1].
    dim : int
        Embedding dimension.

    Returns
    ----------
    emb : torch.Tensor
        Time embedding [B, dim].
    """
    half_dim = dim // 2
    emb_scale = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb_scale)
    emb = t.unsqueeze(-1) * emb.unsqueeze(0)  # [B, half_dim]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # [B, dim]
    return emb


class AdaLNBlock(nn.Module):
    """
    Adaptive LayerNorm residual block (DiT-style).

    Uses condition vectors to modulate LayerNorm scale and shift.
    """
    
    def __init__(self, hidden_dim: int, cond_dim: int, dropout: float = 0.1):
        """
        Parameters
        ----------
        hidden_dim : int
            Hidden dimension.
        cond_dim : int
            Condition vector dimension.
        dropout : float
            Dropout probability.
        """
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        
        # Condition projection to generate scale/shift for norm1 and norm2.
        # Total 6 * hidden_dim: (scale1, shift1, scale2, shift2, gate1, gate2)
        self.cond_proj = nn.Linear(cond_dim, 6 * hidden_dim)
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input [B, hidden_dim].
        cond : torch.Tensor
            Condition vector [B, cond_dim].

        Returns
        ----------
        out : torch.Tensor
            Output [B, hidden_dim].
        """
        # Generate modulation parameters.
        params = self.cond_proj(cond)  # [B, 6 * hidden_dim]
        scale1, shift1, scale2, shift2, gate1, gate2 = params.chunk(6, dim=-1)
        
        # First sub-layer (with residual).
        h = self.norm1(x)
        h = h * (1 + scale1) + shift1
        h = self.fc1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.dropout(h)
        x = x + gate1.tanh() * h
        
        # Second sub-layer (with residual).
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
    Protein variational autoencoder.

    Input: preprocessed protein expression + batch_id
    Output: reconstructed protein expression
    Loss: Gaussian NLL or NB/ZINB NLL + KL

    Supported distributions:
    - 'Gaussian': for log-normalized continuous values
    - 'NB': Negative Binomial for count data
    - 'ZINB': Zero-Inflated Negative Binomial for sparse count data
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
        Parameters
        ----------
        n_proteins : int
            Number of proteins (M).
        dz : int
            Latent dimension.
        hidden_dims : list
            Encoder/decoder hidden dimensions.
        batch_emb_dim : int
            Batch embedding dimension.
        n_batches : int
            Number of batches.
        beta_kl : float
            KL weight.
        learnable_dispersion : bool
            Whether to learn per-protein sigma/dispersion parameters.
        dist_type : str
            Distribution type: 'Gaussian', 'NB', or 'ZINB'.
        """
        super().__init__()
        
        assert dist_type in ['Gaussian', 'NB', 'ZINB'], f"dist_type must be 'Gaussian', 'NB' or 'ZINB', got {dist_type}"
        
        self.n_proteins = n_proteins
        self.dz = dz
        self.beta_kl = beta_kl
        self.learnable_dispersion = learnable_dispersion
        self.dist_type = dist_type
        
        # Batch embedding.
        self.batch_emb = nn.Embedding(n_batches, batch_emb_dim)
        
        # Encoder.
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
        
        # Decoder.
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
        
        # Distribution parameters.
        if dist_type == 'Gaussian':
            # Learnable sigma (one per protein).
            if learnable_dispersion:
                self.sigma_param = nn.Parameter(torch.zeros(n_proteins))
            else:
                self.register_buffer('sigma', torch.ones(1))
        elif dist_type in ['NB', 'ZINB']:
            # Learnable dispersion parameter (one per protein).
            # theta = 1 / dispersion is inverse-dispersion.
            # Larger dispersion means stronger overdispersion.
            if learnable_dispersion:
                # Initialize at 0, softplus(value) is about 0.69.
                self.disp_param = nn.Parameter(torch.zeros(n_proteins))
            else:
                self.register_buffer('dispersion', torch.ones(n_proteins))
            
            # Scale factor for NB/ZINB to convert sigmoid output into mean.
            # Same as scVAEIT: mu = sigmoid(output) * log(1e4 + 1)
            self.register_buffer('nb_scale', torch.tensor(np.log(1e4 + 1.)))
            
            # ZINB-specific: logit head for zero-inflation probability.
            if dist_type == 'ZINB':
                # Predict pi_logit from decoder's last hidden representation.
                self.pi_decoder = nn.Linear(hidden_dims[0], n_proteins)
            
    def encode(
        self, 
        y_prot: torch.Tensor, 
        batch_id: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode.

        Parameters
        ----------
        y_prot : torch.Tensor
            Protein expression [B, M].
        batch_id : torch.Tensor
            Batch IDs [B].

        Returns
        ----------
        mu : torch.Tensor
            Mean [B, dz].
        logvar : torch.Tensor
            Log-variance [B, dz].
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
        Reparameterize.

        Parameters
        ----------
        mu : torch.Tensor
            Mean [B, dz].
        logvar : torch.Tensor
            Log-variance [B, dz].

        Returns
        ----------
        z : torch.Tensor
            Sampled latent variable [B, dz].
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
        Decode.

        Parameters
        ----------
        z : torch.Tensor
            Latent variable [B, dz].
        batch_id : torch.Tensor
            Batch IDs [B].
        return_raw : bool
            If True, return raw network output (for NB/ZINB handling).
        return_pi : bool
            If True and distribution is ZINB, return (y_hat, pi_logit).

        Returns
        ----------
        y_hat : torch.Tensor
            Reconstructed protein [B, M].
            - Gaussian: direct output
            - NB/ZINB: sigmoid output scaled to mean mu
        pi_logit : torch.Tensor (only when return_pi=True and dist_type='ZINB')
            Logits of zero-inflation probability [B, M].
        """
        batch_emb = self.batch_emb(batch_id)  # [B, batch_emb_dim]
        x = torch.cat([z, batch_emb], dim=-1)  # [B, dz + batch_emb_dim]
        
        # Forward through decoder while keeping intermediate features for ZINB.
        if self.dist_type == 'ZINB':
            # Run decoder manually to capture intermediate activations.
            h = x
            # Decoder depth: len(dec_dims) - 1 = len(hidden_dims[::-1]) + 2
            # Each hidden layer has 4 submodules (Linear, LayerNorm, GELU, Dropout).
            n_layers = len(self.decoder)
            # Final layer is a standalone Linear; preceding layers use 4 submodules each.
            n_hidden_layers = (n_layers - 1) // 4
            
            # Run up to the penultimate stage to get intermediate features.
            for i in range(n_hidden_layers * 4):
                h = self.decoder[i](h)
            h_intermediate = h  # Intermediate feature used to predict pi.
            
            # Final output layer.
            y_hat = self.decoder[-1](h)
            
            # Predict logits for zero-inflation probability.
            pi_logit = self.pi_decoder(h_intermediate)
            
            if not return_raw:
                # ZINB: constrain output to (0, 1) with sigmoid, then scale to mean mu.
                y_hat = torch.sigmoid(y_hat) * self.nb_scale
            
            if return_pi:
                return y_hat, pi_logit
            return y_hat
        else:
            y_hat = self.decoder(x)
            
            if self.dist_type == 'NB' and not return_raw:
                # NB: constrain output to (0, 1) with sigmoid, then scale to mean mu.
                # Consistent with scVAEIT: mu = sigmoid(output) * log(1e4 + 1)
                y_hat = torch.sigmoid(y_hat) * self.nb_scale
            
            return y_hat
    
    def get_sigma(self) -> torch.Tensor:
        """Get sigma (Gaussian distribution only)."""
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
        Get dispersion parameters for NB/ZINB distributions.

        Returns
        ----------
        dispersion : torch.Tensor
            Dispersion parameters [M], where theta = 1/dispersion.
        """
        if self.dist_type not in ['NB', 'ZINB']:
            raise ValueError("get_dispersion() is only available for NB/ZINB distribution")
        if self.learnable_dispersion:
            # Use softplus to ensure dispersion > 0.
            # Clamp to a stable range (similar to scVAEIT: [0, 6]).
            dispersion = F.softplus(self.disp_param)
            dispersion = dispersion.clamp(1e-4, 6.0)
            return dispersion
        else:
            return self.dispersion
    
    def get_theta(self) -> torch.Tensor:
        """
        Get inverse-dispersion parameter theta for NB/ZINB.
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

        Returns
        ----------
        y_hat, mu, logvar, z, and pi_logit (for ZINB) or None.
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

        Parameters
        ----------
        y : torch.Tensor
            Ground-truth protein [B, M].
            - Gaussian: log-normalized values
            - NB/ZINB: may be log-normalized (then provide y_raw)
        y_hat : torch.Tensor
            Reconstructed protein [B, M].
            - Gaussian: direct prediction
            - NB/ZINB: predicted mean mu
        mu : torch.Tensor
            Latent mean [B, dz].
        logvar : torch.Tensor
            Latent log-variance [B, dz].
        y_raw : torch.Tensor, optional
            Raw protein counts [B, M] (needed for NB/ZINB).
        pi_logit : torch.Tensor, optional
            Zero-inflation logits [B, M] (needed for ZINB).

        Returns
        ----------
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
            
            # Compute NB loss using raw counts.
            if y_raw is not None:
                y_counts = y_raw
            else:
                # If raw counts are missing, assume y is already in raw-count space.
                y_counts = y
            
            # y_hat is already mean mu (from sigmoid * scale in decode).
            log_prob = log_nb_positive(y_counts, y_hat, theta)  # [B, M]
            nll = -log_prob.sum(dim=-1).mean()  # Negative log-likelihood.
            
        else:  # ZINB
            theta = self.get_theta()  # [M]
            
            # Compute ZINB loss using raw counts.
            if y_raw is not None:
                y_counts = y_raw
            else:
                # If raw counts are missing, assume y is already in raw-count space.
                y_counts = y
            
            if pi_logit is None:
                raise ValueError("pi_logit is required for ZINB distribution")
            
            # y_hat is already mean mu (from sigmoid * scale in decode).
            # pi_logit contains zero-inflation logits.
            log_prob = log_zinb_positive(y_counts, y_hat, theta, pi_logit)  # [B, M]
            nll = -log_prob.sum(dim=-1).mean()  # Negative log-likelihood.
        
        # KL divergence: KL(N(mu, var) || N(0, I)).
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
        Get latent variable.

        Parameters
        ----------
        deterministic : bool
            If True, return mu; otherwise sample.
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
    RNA condition encoder.

    Encodes each cell's HVG vector into condition vector c_rna.
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
        Parameters
        ----------
        n_genes : int
            Number of genes (G).
        dc : int
            Condition vector dimension.
        hidden_dims : list
            Hidden dimensions.
        batch_emb_dim : int
            Batch embedding dimension.
        n_batches : int
            Number of batches.
        dropout : float
            Dropout probability.
        """
        super().__init__()
        
        self.n_genes = n_genes
        self.dc = dc
        
        # Batch embedding.
        self.batch_emb = nn.Embedding(n_batches, batch_emb_dim)
        
        # Encoder.
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

        Parameters
        ----------
        rna_norm : torch.Tensor
            Preprocessed RNA [B, G].
        batch_id : torch.Tensor
            Batch IDs [B].

        Returns
        ----------
        c_rna : torch.Tensor
            Condition vector [B, dc].
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
    Conditional flow-matching network.

    Predicts vector field v(x_t, t, c_rna, batch)
    using DiT-style AdaLN residual blocks.
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
        Parameters
        ----------
        dz : int
            Latent dimension.
        dc : int
            Condition vector dimension.
        hidden_dim : int
            Hidden dimension.
        n_blocks : int
            Number of residual blocks.
        time_emb_dim : int
            Time embedding dimension.
        batch_emb_dim : int
            Batch embedding dimension.
        n_batches : int
            Number of batches.
        dropout : float
            Dropout probability.
        """
        super().__init__()
        
        self.dz = dz
        self.dc = dc
        self.time_emb_dim = time_emb_dim
        
        # Batch embedding.
        self.batch_emb = nn.Embedding(n_batches, batch_emb_dim)
        
        # Condition dimension: c_rna + time_emb + batch_emb.
        cond_dim = dc + time_emb_dim + batch_emb_dim
        
        # Unconditional embedding for CFG.
        self.cond_null = nn.Parameter(torch.randn(dc) * 0.01)
        
        # Input projection.
        self.input_proj = nn.Linear(dz, hidden_dim)
        
        # AdaLN residual blocks.
        self.blocks = nn.ModuleList([
            AdaLNBlock(hidden_dim, cond_dim, dropout)
            for _ in range(n_blocks)
        ])
        
        # Output projection.
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, dz)
        
        # Initialize output layer to zero.
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

        Parameters
        ----------
        x_t : torch.Tensor
            Current latent state [B, dz].
        t : torch.Tensor
            Time [B] in [0, 1].
        c_rna : torch.Tensor
            RNA condition vector [B, dc].
        batch_id : torch.Tensor
            Batch IDs [B].

        Returns
        ----------
        v : torch.Tensor
            Predicted vector field [B, dz].
        """
        # Time embedding.
        t_emb = sinusoidal_embedding(t, self.time_emb_dim)  # [B, time_emb_dim]
        
        # Batch embedding.
        batch_emb = self.batch_emb(batch_id)  # [B, batch_emb_dim]
        
        # Condition vector.
        cond = torch.cat([c_rna, t_emb, batch_emb], dim=-1)  # [B, cond_dim]
        
        # Input projection.
        h = self.input_proj(x_t)  # [B, hidden_dim]
        
        # Residual blocks.
        for block in self.blocks:
            h = block(h, cond)
        
        # Output.
        h = self.output_norm(h)
        v = self.output_proj(h)
        
        return v
    
    def get_cond_null(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Get unconditional embedding."""
        return self.cond_null.unsqueeze(0).expand(batch_size, -1)


# ===========================================================================
# Helper function
# ===========================================================================

def apply_gene_mask(
    rna_norm: torch.Tensor,
    mask_ratio_range: Tuple[float, float] = (0.2, 0.5)
) -> torch.Tensor:
    """
    Apply gene mask to RNA features.

    Parameters
    ----------
    rna_norm : torch.Tensor
        Preprocessed RNA [B, G].
    mask_ratio_range : tuple
        Mask ratio range (r_min, r_max).

    Returns
    ----------
    rna_masked : torch.Tensor
        Masked RNA [B, G].
    """
    B, G = rna_norm.shape
    device = rna_norm.device
    
    # Sample mask ratio for each sample.
    r_min, r_max = mask_ratio_range
    r = torch.rand(B, device=device) * (r_max - r_min) + r_min  # [B]
    k = (r * G).long()  # Number of masked genes per sample.
    
    # Build mask.
    mask = torch.ones(B, G, device=device)
    for i in range(B):
        indices = torch.randperm(G, device=device)[:k[i]]
        mask[i, indices] = 0
    
    return rna_norm * mask

