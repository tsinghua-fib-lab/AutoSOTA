# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Neural network training utilities for question embeddings.

This module provides:
- QuestionEncoder: MLP encoder with integrated phi computation
- GP-based loss computation with multiple kernel options (linear, matern, rbf)
- Embedding extraction utilities
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Literal


# Jitter schedule for Cholesky stability (aligned with colabs/nn_utils.py)
_JITTER_SCHEDULE = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]

# Minimum noise variance to prevent GP from becoming overconfident
MIN_NOISE_VAR = 0.3


def _safe_cholesky(matrix: torch.Tensor) -> torch.Tensor:
    """Cholesky decomposition with progressive jitter on failure."""
    try:
        return torch.linalg.cholesky(matrix)
    except RuntimeError:
        pass
    m = matrix.shape[0]
    for jitter in _JITTER_SCHEDULE:
        try:
            return torch.linalg.cholesky(
                matrix + jitter * torch.eye(m, device=matrix.device, dtype=matrix.dtype)
            )
        except RuntimeError:
            continue
    raise RuntimeError(
        f"Cholesky failed even with jitter={_JITTER_SCHEDULE[-1]}. "
        f"Diag range: [{torch.diag(matrix).min():.6f}, {torch.diag(matrix).max():.6f}]"
    )


class QuestionEncoder(nn.Module):
    """Neural network encoder with integrated psi/phi computation and learnable kernel parameters.
    
    Supports multiple kernel types:
    - 'linear': K = φφᵀ + σ²I (original, fast)
    - 'matern': Matérn 2.5 kernel in φ-space with learnable lengthscale
    - 'rbf': RBF/Squared Exponential kernel in φ-space with learnable lengthscale
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 128, 
        init_var: float = 0.3, 
        steepness: float = 30.0,
        kernel_type: Literal['linear', 'matern', 'rbf'] = 'linear',
        init_lengthscale: float = 1.0,
        matern_nu: float = 2.5,
    ):
        super(QuestionEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.steepness = steepness
        self.kernel_type = kernel_type
        self.matern_nu = matern_nu
        
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.log_var = nn.Parameter(torch.tensor(np.log(init_var), dtype=torch.float32))
        self.log_lengthscale = nn.Parameter(torch.tensor(np.log(init_lengthscale), dtype=torch.float32))
    
    @property
    def var(self) -> torch.Tensor:
        """Get variance parameter (always positive, with lower bound)."""
        return torch.clamp(torch.exp(self.log_var), min=MIN_NOISE_VAR)
    
    @property
    def lengthscale(self) -> torch.Tensor:
        """Get lengthscale parameter (always positive)."""
        return torch.exp(self.log_lengthscale)
    
    def forward_psi(self, x: torch.Tensor) -> torch.Tensor:
        """Raw ψ(x): sigmoid(steepness · linear(x))."""
        return torch.sigmoid(self.steepness * self.linear(x))
    
    def forward_phi(self, x: torch.Tensor) -> torch.Tensor:
        """Get phi(x) = normalized psi(x).
        
        Args:
            x: (batch_size, input_dim) tensor
        
        Returns:
            phi(x): (batch_size, hidden_dim) tensor
        """
        psi_x = self.forward_psi(x)
        return self._compute_phi(psi_x)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """Get (psi, phi) tuple.
        
        Args:
            x: (batch_size, input_dim) tensor
        
        Returns:
            (psi(x), phi(x)) tuple
        """
        psi_x = self.forward_psi(x)
        phi_x = self._compute_phi(psi_x)
        return psi_x, phi_x
    
    def _compute_phi(self, psi_x: torch.Tensor) -> torch.Tensor:
        """Compute phi from psi: centered, then L2-normalized.

        ``phi(x) = normalize((psi(x) - mean(psi(x))) / sqrt(d-1))``

        L2 normalization places all phi vectors on the unit sphere, ensuring
        pairwise distances lie in ``[0, √2]``.  This makes the Matérn
        kernel's lengthscale operate in a well-calibrated regime regardless
        of hidden_dim.
        """
        d = psi_x.shape[1]
        mean_psi = torch.mean(psi_x, dim=1, keepdim=True)
        phi = (psi_x - mean_psi) / np.sqrt(d - 1)
        # L2 normalize to unit sphere
        phi = phi / (torch.norm(phi, dim=1, keepdim=True) + 1e-10)
        return phi


def compute_kernel_matrix(
    phi_x: torch.Tensor,
    encoder: nn.Module,
) -> torch.Tensor:
    """
    Compute the kernel (Gram) matrix based on encoder's kernel type.
    
    Args:
        phi_x: (m, d) tensor of phi embeddings
        encoder: QuestionEncoder with kernel_type, var, and lengthscale params
    
    Returns:
        K: (m, m) kernel matrix (without noise term)
    """
    m = phi_x.shape[0]
    kernel_type = getattr(encoder, 'kernel_type', 'linear')
    
    if kernel_type == 'linear':
        # Linear kernel: K = φφᵀ
        return torch.mm(phi_x, phi_x.t())
    
    # For Matérn and RBF, we need pairwise distances in φ-space
    # dist[i,j] = ||phi_i - phi_j||
    # Using: ||a-b||² = ||a||² + ||b||² - 2<a,b>
    phi_sq = torch.sum(phi_x ** 2, dim=1, keepdim=True)  # (m, 1)
    dist_sq = phi_sq + phi_sq.t() - 2 * torch.mm(phi_x, phi_x.t())  # (m, m)
    dist_sq = torch.clamp(dist_sq, min=0)  # Numerical stability
    dist = torch.sqrt(dist_sq + 1e-12)  # (m, m)
    
    # Scale by lengthscale
    lengthscale = encoder.lengthscale
    scaled_dist = dist / lengthscale
    
    if kernel_type == 'rbf':
        # RBF/Squared Exponential: K = exp(-0.5 * (d/l)²)
        return torch.exp(-0.5 * (scaled_dist ** 2))
    
    elif kernel_type == 'matern':
        # Matérn kernel with nu parameter
        nu = getattr(encoder, 'matern_nu', 2.5)
        
        if nu == 0.5:
            # Matérn 1/2 (Exponential): K = exp(-d/l)
            return torch.exp(-scaled_dist)
        
        elif nu == 1.5:
            # Matérn 3/2: K = (1 + √3*d/l) * exp(-√3*d/l)
            sqrt3 = np.sqrt(3)
            return (1 + sqrt3 * scaled_dist) * torch.exp(-sqrt3 * scaled_dist)
        
        else:  # nu == 2.5 (default)
            # Matérn 5/2: K = (1 + √5*d/l + 5/3*(d/l)²) * exp(-√5*d/l)
            sqrt5 = np.sqrt(5)
            return (1 + sqrt5 * scaled_dist + (5/3) * (scaled_dist ** 2)) * torch.exp(-sqrt5 * scaled_dist)
    
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")


def compute_gp_loss(
    encoder: nn.Module,
    question_embeddings: torch.Tensor,
    labels: torch.Tensor
) -> torch.Tensor:
    """
    Compute GP-based loss for a single benchmark.
    
    Loss = sum over models of [y_i^T * Gram^{-1} * y_i + log(det(Gram))]
    
    where:
    - Gram = K(phi, phi) + var * I (kernel depends on encoder.kernel_type)
    - For linear: K = φφᵀ
    - For matern/rbf: K = k(φ_i, φ_j) with learnable lengthscale
    - y_i = labels - mean(psi(x)) for each model
    
    Args:
        encoder: QuestionEncoder model with learnable params and kernel_type
        question_embeddings: (m, d) tensor of question embeddings
        labels: (m, n) tensor of labels for n models
    
    Returns:
        loss: Scalar loss value
    """
    # Ensure float32 for MPS compatibility
    question_embeddings = question_embeddings.float()
    labels = labels.float()
    
    m = question_embeddings.shape[0]  # Number of questions
    n = labels.shape[1]  # Number of models
    
    # 1. Compute psi and phi embeddings using encoder's integrated methods
    psi_x, phi_x = encoder(question_embeddings)  # (m, d), (m, d)
    
    # 2. Compute Gram matrix using encoder's kernel type + noise
    K = compute_kernel_matrix(phi_x, encoder)
    gram_matrix = K + encoder.var * torch.eye(m, device=phi_x.device, dtype=phi_x.dtype)
    
    # 3. Normalize ALL labels at once: y = labels - mean(psi(x))
    mean_psi = torch.mean(psi_x, dim=1, keepdim=True)  # (m, 1)
    Y = labels - mean_psi  # (m, n) - all models at once
    
    # 4. Cholesky decomposition with progressive jitter
    try:
        L = _safe_cholesky(gram_matrix)
    except RuntimeError as e:
        # Return NaN so the training loop can detect and skip this batch.
        # Do NOT return a large finite loss — that creates huge gradients
        # which corrupt parameters and cascade into NaN for all future batches.
        return torch.tensor(float('nan'), device=phi_x.device, dtype=phi_x.dtype)
    
    # 5. Solve for ALL models at once: L L^T alpha = Y
    alpha = torch.cholesky_solve(Y, L)  # (m, n)
    
    # 6. Compute all quadratic forms at once: sum of y_i^T @ Gram^-1 @ y_i
    quad_forms = torch.sum(Y * alpha)  # sum of element-wise product
    
    # 7. Log determinant (same for all models, multiply by n)
    log_det = 2 * torch.sum(torch.log(torch.diag(L)))
    
    total_loss = quad_forms + n * log_det
    
    return total_loss / n  # Average over models


def compute_kl_loss(
    encoder: nn.Module,
    question_embeddings: torch.Tensor,
    sample_mean: torch.Tensor,
    sample_cov: torch.Tensor,
) -> torch.Tensor:
    """
    Compute only KL divergence loss to match learned Gaussian to sample Gaussian.
    
    KL(N_sample || N_learned) = 0.5 * [tr(Σ_learned⁻¹ Σ_sample) + (μ_learned - μ_sample)ᵀ Σ_learned⁻¹ (μ_learned - μ_sample) - m + ln(det(Σ_learned)/det(Σ_sample))]
    
    Args:
        encoder: QuestionEncoder model
        question_embeddings: (m, d) tensor
        sample_mean: (m,) target mean
        sample_cov: (m, m) target covariance
    
    Returns:
        kl_loss: Scalar KL divergence loss
    """
    question_embeddings = question_embeddings.float()
    m = question_embeddings.shape[0]
    
    # Compute psi and phi
    psi_x, phi_x = encoder(question_embeddings)
    
    learned_mean = torch.mean(psi_x, dim=1)  # (m,)
    # Use kernel matrix based on encoder's kernel_type
    learned_cov = compute_kernel_matrix(phi_x, encoder)  # (m, m)
    
    # Add regularization for numerical stability
    # Use encoder's learned variance for learned_cov, small reg for sample_cov
    learned_cov_reg = learned_cov + encoder.var * torch.eye(m, device=phi_x.device, dtype=phi_x.dtype)
    sample_cov_reg = sample_cov + 1e-4 * torch.eye(m, device=phi_x.device, dtype=phi_x.dtype)
    
    # Cholesky decomposition
    try:
        L_learned = _safe_cholesky(learned_cov_reg)
    except RuntimeError:
        return torch.tensor(float('nan'), device=phi_x.device, dtype=phi_x.dtype)
    
    try:
        L_sample = _safe_cholesky(sample_cov_reg)
    except RuntimeError:
        return torch.tensor(float('nan'), device=phi_x.device, dtype=phi_x.dtype)
    
    # tr(Σ_learned⁻¹ Σ_sample)
    inv_learned_times_sample = torch.cholesky_solve(sample_cov_reg, L_learned)
    trace_term = torch.trace(inv_learned_times_sample)
    
    # (μ_learned - μ_sample)ᵀ Σ_learned⁻¹ (μ_learned - μ_sample)
    mean_diff = (learned_mean - sample_mean).unsqueeze(1)
    alpha_mean = torch.cholesky_solve(mean_diff, L_learned)
    quad_term = torch.sum(mean_diff * alpha_mean)
    
    # ln(det(Σ_learned)/det(Σ_sample))
    log_det_learned = 2 * torch.sum(torch.log(torch.diag(L_learned)))
    log_det_sample = 2 * torch.sum(torch.log(torch.diag(L_sample)))
    log_det_ratio = log_det_learned - log_det_sample
    
    # Full KL: 0.5 * [trace + quad - m + log_det_ratio]
    kl_loss = 0.5 * (trace_term + quad_term - m + log_det_ratio)
    
    return kl_loss


def compute_gp_loss_with_reg(
    encoder: nn.Module,
    question_embeddings: torch.Tensor,
    labels: torch.Tensor,
    sample_mean: torch.Tensor = None,
    sample_cov: torch.Tensor = None,
    lambda_mean: float = 1.0,
    lambda_cov: float = 1.0,
) -> torch.Tensor:
    """
    Compute GP loss with mean and covariance regularization.
    
    Args:
        encoder: QuestionEncoder model
        question_embeddings: (m, d) tensor
        labels: (m, n) tensor
        sample_mean: (m,) target mean to match (optional)
        sample_cov: (m, m) target covariance to match (optional)
        lambda_mean: Weight for mean matching penalty
        lambda_cov: Weight for covariance matching penalty
    
    Returns:
        loss: Scalar loss including regularization terms
    """
    question_embeddings = question_embeddings.float()
    labels = labels.float()
    
    m = question_embeddings.shape[0]
    n = labels.shape[1]
    
    # Compute psi and phi
    psi_x, phi_x = encoder(question_embeddings)
    
    # Standard GP loss using encoder's kernel type
    K = compute_kernel_matrix(phi_x, encoder)
    gram_matrix = K + encoder.var * torch.eye(m, device=phi_x.device, dtype=phi_x.dtype)
    mean_psi = torch.mean(psi_x, dim=1, keepdim=True)
    Y = labels - mean_psi
    
    try:
        L = _safe_cholesky(gram_matrix)
    except RuntimeError as e:
        print(f"  WARNING: Cholesky failed in GP loss, skipping: {e}")
        return torch.tensor(float('nan'), device=phi_x.device, dtype=phi_x.dtype)
    
    alpha = torch.cholesky_solve(Y, L)
    quad_forms = torch.sum(Y * alpha)
    log_det = 2 * torch.sum(torch.log(torch.diag(L)))
    gp_loss = (quad_forms + n * log_det) / n
    
    total_loss = gp_loss
    
    # KL divergence between learned N(μ_learned, Σ_learned) and sample N(μ_sample, Σ_sample)
    # KL(N_sample || N_learned) = 0.5 * [tr(Σ_learned⁻¹ Σ_sample) + (μ_learned - μ_sample)ᵀ Σ_learned⁻¹ (μ_learned - μ_sample) - m + ln(det(Σ_learned)/det(Σ_sample))]
    if sample_mean is not None and sample_cov is not None and (lambda_mean > 0 or lambda_cov > 0):
        learned_mean = torch.mean(psi_x, dim=1)  # (m,)
        # Use kernel matrix based on encoder's kernel_type
        learned_cov = compute_kernel_matrix(phi_x, encoder)  # (m, m)
        
        # Add small regularization for numerical stability
        reg = 1e-4
        learned_cov_reg = learned_cov + reg * torch.eye(m, device=phi_x.device, dtype=phi_x.dtype)
        sample_cov_reg = sample_cov + reg * torch.eye(m, device=phi_x.device, dtype=phi_x.dtype)
        
        # Cholesky decomposition of learned covariance
        try:
            L_learned = _safe_cholesky(learned_cov_reg)
        except RuntimeError:
            return gp_loss
        
        try:
            L_sample = _safe_cholesky(sample_cov_reg)
        except RuntimeError:
            return gp_loss
        
        # tr(Σ_learned⁻¹ Σ_sample) using solve
        # Σ_learned⁻¹ Σ_sample = solve(L_learned, solve(L_learned, Σ_sample))
        inv_learned_times_sample = torch.cholesky_solve(sample_cov_reg, L_learned)
        trace_term = torch.trace(inv_learned_times_sample)
        
        # (μ_learned - μ_sample)ᵀ Σ_learned⁻¹ (μ_learned - μ_sample)
        mean_diff = (learned_mean - sample_mean).unsqueeze(1)  # (m, 1)
        alpha_mean = torch.cholesky_solve(mean_diff, L_learned)
        quad_term = torch.sum(mean_diff * alpha_mean)
        
        # ln(det(Σ_learned)/det(Σ_sample)) = 2 * (sum(log(diag(L_learned))) - sum(log(diag(L_sample))))
        log_det_learned = 2 * torch.sum(torch.log(torch.diag(L_learned)))
        log_det_sample = 2 * torch.sum(torch.log(torch.diag(L_sample)))
        log_det_ratio = log_det_learned - log_det_sample
        
        # Full KL: 0.5 * [trace + quad - m + log_det_ratio]
        kl_loss = 0.5 * (trace_term + quad_term - m + log_det_ratio)
        
        total_loss = total_loss + lambda_cov * kl_loss
    
    return total_loss


def get_phi_embeddings(
    encoder: nn.Module,
    question_embeddings: np.ndarray,
    device: Optional[torch.device] = None
) -> np.ndarray:
    """
    Compute phi embeddings for a set of question embeddings.
    
    Args:
        encoder: Trained QuestionEncoder model
        question_embeddings: (m, d) numpy array
        device: Device to use (defaults to encoder's device)
    
    Returns:
        phi_embeddings: (m, d) numpy array
    """
    if device is None:
        device = next(encoder.parameters()).device
    
    encoder.eval()
    
    with torch.no_grad():
        # Convert to tensor
        x = torch.from_numpy(question_embeddings).float().to(device)
        
        # Compute phi using encoder's integrated method
        phi_x = encoder.forward_phi(x)
        
        # Convert back to numpy
        return phi_x.cpu().numpy()


def get_phi_embeddings_batch(
    encoder: nn.Module,
    question_embeddings: np.ndarray,
    batch_size: int = 256,
    device: Optional[torch.device] = None
) -> np.ndarray:
    """
    Compute phi embeddings in batches for large datasets.
    
    Args:
        encoder: Trained QuestionEncoder model
        question_embeddings: (m, d) numpy array
        batch_size: Batch size for processing
        device: Device to use (defaults to encoder's device)
    
    Returns:
        phi_embeddings: (m, d) numpy array
    """
    if device is None:
        device = next(encoder.parameters()).device
    
    encoder.eval()
    m = question_embeddings.shape[0]
    phi_list = []
    
    with torch.no_grad():
        for i in range(0, m, batch_size):
            batch = question_embeddings[i:i+batch_size]
            x = torch.from_numpy(batch).float().to(device)
            
            # Compute phi using encoder's integrated method
            phi_x = encoder.forward_phi(x)
            
            phi_list.append(phi_x.cpu().numpy())
    
    return np.vstack(phi_list)


def save_encoder(
    encoder: nn.Module,
    save_path: str,
    embedding_dim: int,
    var: float,
    loss_history: list,
    **kwargs
):
    """
    Save encoder model and training metadata.
    
    Args:
        encoder: Encoder model to save
        save_path: Path to save the model
        embedding_dim: Dimension of embeddings
        var: Variance hyperparameter used
        loss_history: Training loss history
        **kwargs: Additional metadata to save
    """
    checkpoint = {
        'model_state_dict': encoder.state_dict(),
        'embedding_dim': embedding_dim,
        'hidden_dim': encoder.hidden_dim,
        'var': var,
        'loss_history': loss_history,
        'kernel_type': getattr(encoder, 'kernel_type', 'linear'),
        'matern_nu': getattr(encoder, 'matern_nu', 2.5),
        **kwargs
    }
    torch.save(checkpoint, save_path)


def load_encoder(
    model_path: str,
    device: torch.device,
    hidden_dim: int = None,
    kernel_type: str = None
) -> tuple:
    """
    Load encoder model from checkpoint.
    
    Args:
        model_path: Path to saved model
        device: Device to load model to
        hidden_dim: Hidden dimension (default: read from checkpoint)
        kernel_type: Kernel type override (default: read from checkpoint)
    
    Returns:
        encoder: Loaded encoder model
        checkpoint: Full checkpoint dictionary
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    embedding_dim = checkpoint['embedding_dim']
    if hidden_dim is None:
        hidden_dim = checkpoint.get('hidden_dim', 256)
    if kernel_type is None:
        kernel_type = checkpoint.get('kernel_type', 'linear')
    matern_nu = checkpoint.get('matern_nu', 2.5)
    steepness = checkpoint.get('steepness', 30.0)  # Legacy checkpoints default to 30.0
    
    encoder = QuestionEncoder(
        input_dim=embedding_dim, 
        hidden_dim=hidden_dim,
        steepness=steepness,
        kernel_type=kernel_type,
        matern_nu=matern_nu,
    ).to(device)
    encoder.load_state_dict(checkpoint['model_state_dict'])
    encoder.eval()
    
    return encoder, checkpoint
