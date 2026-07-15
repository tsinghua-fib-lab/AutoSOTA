#!/usr/bin/env python3
"""
Projection modules for transforming activation vectors to soft token embeddings.

These adapters map hidden states into embedding space for self-interpretation via patching.
The paper evaluates: identity, scale-only, scalar affine, scalar affine + low-rank,
low-rank only, and full-rank affine transformations.
"""

from abc import ABC, abstractmethod
from typing import Optional
import torch
import torch.nn as nn


class ProjectionModule(ABC, nn.Module):
    """Abstract base class for vector to soft token projections."""
    
    def __init__(self, dim: int, normalize_input: bool, device: torch.device):
        super().__init__()
        self.dim = dim
        self.normalize_input = normalize_input
        self.device = device
    
    @abstractmethod
    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Project vectors to soft token embeddings.
        
        Args:
            vectors: (batch_size, dim)
        
        Returns:
            soft_tokens: (batch_size, dim)
        """
        pass
    
    @abstractmethod
    def initialize_weights(self, init_scale: float, low_rank_init_factor: Optional[float] = None):
        """Initialize projection parameters."""
        pass
    
    @abstractmethod
    def num_parameters(self) -> int:
        """Return the number of trainable parameters."""
        pass
    
    def _normalize_if_needed(self, vectors: torch.Tensor) -> torch.Tensor:
        """Apply L2 normalization if configured."""
        if self.normalize_input:
            norms = torch.norm(vectors, p=2, dim=-1, keepdim=True)
            return vectors / (norms + 1e-8)
        return vectors


class ScaleOnlyProjection(ProjectionModule):
    """
    Scale-only projection: output = scale * input (with optional normalization).
    
    This is the simplest learnable transformation (1 parameter).
    
    Note: For the identity baseline f(x) = x used in the paper (0 effective parameters),
    initialize with init_scale=1.0 and either freeze the parameter or don't train.
    """
    
    def __init__(self, dim: int, normalize_input: bool, device: torch.device):
        super().__init__(dim, normalize_input, device)
        
        # Learnable scalar scale (parameterized as exp(log_scale) for stability)
        self.log_scale = nn.Parameter(torch.zeros(1, device=device))
    
    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        vectors = self._normalize_if_needed(vectors)
        scale = torch.exp(self.log_scale)
        return scale * vectors
    
    def initialize_weights(self, init_scale: float, low_rank_init_factor: Optional[float] = None):
        with torch.no_grad():
            self.log_scale.fill_(torch.log(torch.tensor(init_scale)).item())
    
    def num_parameters(self) -> int:
        return 1
    
    def get_scale(self) -> float:
        """Get current scale value."""
        return torch.exp(self.log_scale).item()


class ScalarAffineProjection(ProjectionModule):
    """
    Scalar affine projection: output = scale * input + bias (with optional normalization).
    
    The bias vector accounts for ~85% of improvement over untrained baselines.
    This is a strong minimal baseline with only d+1 parameters.
    """
    
    def __init__(self, dim: int, normalize_input: bool, device: torch.device):
        super().__init__(dim, normalize_input, device)
        
        # Learnable scalar scale
        self.log_scale = nn.Parameter(torch.zeros(1, device=device))
        
        # Learnable bias vector
        self.bias = nn.Parameter(torch.zeros(dim, device=device))
    
    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        vectors = self._normalize_if_needed(vectors)
        scale = torch.exp(self.log_scale)
        return scale * vectors + self.bias
    
    def initialize_weights(self, init_scale: float, low_rank_init_factor: Optional[float] = None):
        with torch.no_grad():
            self.log_scale.fill_(torch.log(torch.tensor(init_scale)).item())
            self.bias.zero_()
    
    def num_parameters(self) -> int:
        return 1 + self.dim
    
    def get_scale(self) -> float:
        """Get current scale value."""
        return torch.exp(self.log_scale).item()
    
    def get_bias_norm(self) -> float:
        """Get L2 norm of bias vector."""
        return torch.norm(self.bias, p=2).item()


class FullRankAffineProjection(ProjectionModule):
    """
    Full-rank affine projection: output = W @ input + bias.
    
    WARNING: Full-rank transformations overfit catastrophically in experiments.
    This architecture has d^2 + d parameters, which is excessive for this task.
    """
    
    def __init__(self, dim: int, normalize_input: bool, device: torch.device):
        super().__init__(dim, normalize_input, device)
        
        # Learnable weight matrix
        self.weight = nn.Parameter(torch.zeros(dim, dim, device=device))
        
        # Learnable bias vector
        self.bias = nn.Parameter(torch.zeros(dim, device=device))
    
    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        vectors = self._normalize_if_needed(vectors)
        return torch.matmul(vectors, self.weight.T) + self.bias
    
    def initialize_weights(self, init_scale: float, low_rank_init_factor: Optional[float] = None):
        with torch.no_grad():
            # Xavier initialization scaled by init_scale
            fan_in, fan_out = self.dim, self.dim
            std = init_scale * (2.0 / (fan_in + fan_out)) ** 0.5
            self.weight.normal_(0, std)
            self.bias.zero_()
    
    def num_parameters(self) -> int:
        return self.dim * self.dim + self.dim
    
    def get_bias_norm(self) -> float:
        """Get L2 norm of bias vector."""
        return torch.norm(self.bias, p=2).item()
    
    def get_weight_norm(self) -> float:
        """Get Frobenius norm of weight matrix."""
        return torch.norm(self.weight, p='fro').item()
    
    def get_singular_values(self) -> torch.Tensor:
        """Get singular values of the weight matrix."""
        with torch.no_grad():
            weight_tensor = self.weight
            if weight_tensor.dtype == torch.bfloat16:
                weight_tensor = weight_tensor.float()
            _, s, _ = torch.linalg.svd(weight_tensor, full_matrices=False)
            return s


class LowRankOnlyProjection(ProjectionModule):
    """
    Low-rank only projection: output = input @ U @ V^T + bias
    
    Pure low-rank transformation without any scalar scaling component.
    The low-rank matrix UV^T has shape (dim, dim) with rank at most r.
    
    Note: For row vectors x, this computes x @ (U @ V^T), which is equivalent
    to (V @ U^T) @ x^T when x is treated as a column vector.
    """
    
    def __init__(
        self,
        dim: int,
        normalize_input: bool,
        device: torch.device,
        rank: int,
    ):
        super().__init__(dim, normalize_input, device)
        
        if rank is None or rank <= 0:
            raise ValueError(f"Rank must be a positive integer, got {rank}")
        
        self.rank = rank
        
        # Low-rank components
        self.U = nn.Parameter(torch.zeros(dim, rank, device=device))
        self.V = nn.Parameter(torch.zeros(dim, rank, device=device))
        
        # Learnable bias vector
        self.bias = nn.Parameter(torch.zeros(dim, device=device))
    
    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        vectors = self._normalize_if_needed(vectors)
        
        # Low-rank term: (U @ V^T) @ input
        low_rank_term = torch.matmul(torch.matmul(vectors, self.U), self.V.T)
        
        return low_rank_term + self.bias
    
    def initialize_weights(self, init_scale: float, low_rank_init_factor: Optional[float] = None):
        if low_rank_init_factor is None:
            raise ValueError(
                "low_rank_only requires 'low_rank_init_factor' parameter. "
                "Set config.projection.low_rank_init_factor to a positive float (e.g., 0.01, 0.001)."
            )
        
        with torch.no_grad():
            # Initialize low-rank components with small random values
            std = init_scale * low_rank_init_factor / (self.rank ** 0.5)
            self.U.normal_(0, std)
            self.V.normal_(0, std)
            self.bias.zero_()
    
    def num_parameters(self) -> int:
        return 2 * self.dim * self.rank + self.dim
    
    def get_bias_norm(self) -> float:
        """Get L2 norm of bias vector."""
        return torch.norm(self.bias, p=2).item()
    
    def get_low_rank_norm(self) -> float:
        """Get Frobenius norm of low-rank component U @ V^T."""
        with torch.no_grad():
            return torch.norm(torch.matmul(self.U, self.V.T), p='fro').item()
    
    def get_singular_values(self) -> torch.Tensor:
        """Get singular values of the low-rank component U @ V^T."""
        with torch.no_grad():
            small_matrix = torch.matmul(self.U.T, self.V)
            if small_matrix.dtype == torch.bfloat16:
                small_matrix = small_matrix.float()
            _, s, _ = torch.linalg.svd(small_matrix, full_matrices=False)
            return s


class ScalarAffinePlusLowRankProjection(ProjectionModule):
    """
    Scalar affine + low-rank projection:
    output = scale * input + input @ U @ V^T + bias
    
    This achieves the best performance in the paper - combines simple scaling
    with low-rank adjustment while preserving the identity structure.
    
    The low-rank component has shape (dim, dim) with rank at most r, providing
    subspace-specific adjustments on top of the identity-preserving scalar affine.
    """
    
    def __init__(
        self,
        dim: int,
        normalize_input: bool,
        device: torch.device,
        rank: int,
    ):
        super().__init__(dim, normalize_input, device)
        
        if rank is None or rank <= 0:
            raise ValueError(f"Rank must be a positive integer, got {rank}")
        
        self.rank = rank
        
        # Learnable scalar scale
        self.log_scale = nn.Parameter(torch.zeros(1, device=device))
        
        # Low-rank components
        self.U = nn.Parameter(torch.zeros(dim, rank, device=device))
        self.V = nn.Parameter(torch.zeros(dim, rank, device=device))
        
        # Learnable bias vector
        self.bias = nn.Parameter(torch.zeros(dim, device=device))
    
    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        vectors = self._normalize_if_needed(vectors)
        
        # Scale term
        scale = torch.exp(self.log_scale)
        scale_term = scale * vectors
        
        # Low-rank term
        low_rank_term = torch.matmul(torch.matmul(vectors, self.U), self.V.T)
        
        return scale_term + low_rank_term + self.bias
    
    def initialize_weights(self, init_scale: float, low_rank_init_factor: Optional[float] = None):
        if low_rank_init_factor is None:
            raise ValueError(
                "scalar_affine_plus_low_rank requires 'low_rank_init_factor' parameter. "
                "Set config.projection.low_rank_init_factor to a positive float (e.g., 0.01, 0.001)."
            )
        
        with torch.no_grad():
            self.log_scale.fill_(torch.log(torch.tensor(init_scale)).item())
            
            # Small random initialization for low-rank components
            std = init_scale * low_rank_init_factor / (self.rank ** 0.5)
            self.U.normal_(0, std)
            self.V.normal_(0, std)
            
            self.bias.zero_()
    
    def num_parameters(self) -> int:
        return 1 + 2 * self.dim * self.rank + self.dim
    
    def get_scale(self) -> float:
        """Get current scale value."""
        return torch.exp(self.log_scale).item()
    
    def get_bias_norm(self) -> float:
        """Get L2 norm of bias vector."""
        return torch.norm(self.bias, p=2).item()
    
    def get_low_rank_norm(self) -> float:
        """Get Frobenius norm of low-rank component U @ V^T."""
        with torch.no_grad():
            return torch.norm(torch.matmul(self.U, self.V.T), p='fro').item()
    
    def get_singular_values(self) -> torch.Tensor:
        """Get singular values of the low-rank component U @ V^T."""
        with torch.no_grad():
            small_matrix = torch.matmul(self.U.T, self.V)
            if small_matrix.dtype == torch.bfloat16:
                small_matrix = small_matrix.float()
            _, s, _ = torch.linalg.svd(small_matrix, full_matrices=False)
            return s
    
    def get_low_rank_to_diagonal_ratio(self) -> float:
        """
        Get ratio of low-rank component norm to diagonal component norm.
        
        Returns:
            ||UV^T||_F / ||diag(s)|| ratio
            
        This metric indicates whether the low-rank component is being used:
        - Stays near 0: low-rank component not being used
        - Grows to ~1: model is using low-rank component substantially
        """
        with torch.no_grad():
            low_rank_norm = self.get_low_rank_norm()
            scale = torch.exp(self.log_scale)
            diagonal_norm = (self.dim ** 0.5) * scale.item()
            
            if diagonal_norm < 1e-8:
                return float('inf')
            return low_rank_norm / diagonal_norm


def create_projection_module(
    projection_type: str,
    dim: int,
    normalize_input: bool,
    device: torch.device,
    init_scale: float = 30.0,
    low_rank_rank: Optional[int] = None,
    low_rank_init_factor: Optional[float] = None,
) -> ProjectionModule:
    """
    Factory function to create projection modules.
    
    Args:
        projection_type: One of "scale_only", "scalar_affine", "full_rank",
                        "low_rank_only", "scalar_affine_plus_low_rank"
        dim: Model dimension (hidden size)
        normalize_input: Whether to L2-normalize inputs before projection
        device: Device to place parameters on
        init_scale: Initial scale value
        low_rank_rank: Rank for low-rank projections
        low_rank_init_factor: Factor for low-rank initialization std
    
    Returns:
        ProjectionModule instance
    """
    if projection_type == "scale_only":
        proj = ScaleOnlyProjection(dim, normalize_input, device)
    
    elif projection_type == "scalar_affine":
        proj = ScalarAffineProjection(dim, normalize_input, device)
    
    elif projection_type == "full_rank":
        proj = FullRankAffineProjection(dim, normalize_input, device)
    
    elif projection_type == "low_rank_only":
        if low_rank_rank is None:
            raise ValueError(
                "low_rank_only requires 'low_rank_rank' parameter. "
                "Set config.projection.low_rank_rank to a positive integer (e.g., 64, 128)."
            )
        proj = LowRankOnlyProjection(dim, normalize_input, device, low_rank_rank)
    
    elif projection_type == "scalar_affine_plus_low_rank":
        if low_rank_rank is None:
            raise ValueError(
                "scalar_affine_plus_low_rank requires 'low_rank_rank' parameter. "
                "Set config.projection.low_rank_rank to a positive integer (e.g., 64, 128)."
            )
        proj = ScalarAffinePlusLowRankProjection(dim, normalize_input, device, low_rank_rank)
    
    else:
        raise ValueError(
            f"Unknown projection type: {projection_type}. "
            f"Use one of: scale_only, scalar_affine, full_rank, "
            f"low_rank_only, scalar_affine_plus_low_rank"
        )
    
    # Initialize weights
    proj.initialize_weights(init_scale, low_rank_init_factor)
    
    print(f"Created {projection_type} projection:")
    print(f"  Parameters: {proj.num_parameters():,}")
    print(f"  Normalize input: {normalize_input}")
    print(f"  Initial scale: {init_scale}")
    
    return proj
