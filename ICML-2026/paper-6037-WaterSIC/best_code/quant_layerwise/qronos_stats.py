"""Qronos statistics computation for layerwise quantization.

Qronos minimizes E[(WX - Ŵ X̂)²] where:
- X is the unquantized model's activations
- X̂ is the quantized model's activations (from previously quantized layers)

This requires two statistics:
- Σ_X̂ = E[X̂ X̂^T]  -- Hessian of quantized activations
- Σ_XX̂ = E[X X̂^T]  -- Cross-covariance (order matters!)

The quantization target becomes:
  ŷ = W Σ_XX̂ (L̂^T)^{-1}
where Σ_X̂ = L̂ L̂^T (Cholesky decomposition)
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch


@dataclass
class QronosStats:
    """Statistics needed for Qronos quantization."""

    # Σ_X = E[X X^T] - Hessian of unquantized activations, shape (n, n)
    sigma_x: torch.Tensor

    # Σ_X̂ = E[X̂ X̂^T] - Hessian of quantized activations, shape (n, n)
    sigma_xhat: torch.Tensor

    # Σ_XX̂ = E[X X̂^T] - Cross-covariance, shape (n, n)
    sigma_x_xhat: torch.Tensor

    # Number of sequences/samples used
    nseq: int

    # Number of tokens used
    ntokens: int

    # Optional: Σ_{ΔR,X̂} = E[(R - R̂) X̂^T] for residual compensation (wo/w2 layers only)
    # Shape: (out_features, in_features) where out_features is residual/hidden dim
    sigma_delta_r_xhat: Optional[torch.Tensor] = None

    def save(self, path: str) -> None:
        """Save to pickle file."""
        data = {
            "sigma_x": self.sigma_x.cpu(),
            "sigma_xhat": self.sigma_xhat.cpu(),
            "sigma_x_xhat": self.sigma_x_xhat.cpu(),
            "nseq": self.nseq,
            "ntokens": self.ntokens,
        }
        # Optional: residual compensation stats for wo/w2 layers
        if self.sigma_delta_r_xhat is not None:
            data["sigma_delta_r_xhat"] = self.sigma_delta_r_xhat.cpu()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def broadcast_from_rank0(self) -> "QronosStats":
        """Broadcast all sigma matrices from rank 0 to ensure cross-rank consistency.

        CUDA matmul is non-deterministic across different physical GPUs, so even
        when all ranks compute the same outer products from identical (all-gathered)
        inputs, the accumulated sigma matrices can differ by ~O(1e-13).  In the
        T/Gamma rescaler this causes gamma (per-column, should be global) to diverge
        across ranks, making the optimization incoherent.
        """
        import torch.distributed as _dist
        if not (_dist.is_available() and _dist.is_initialized() and _dist.get_world_size() > 1):
            return self
        for attr in ("sigma_x", "sigma_xhat", "sigma_x_xhat", "sigma_delta_r_xhat"):
            tensor = getattr(self, attr)
            if tensor is None:
                continue
            if tensor.is_cuda:
                _dist.broadcast(tensor, src=0)
            else:
                gpu = tensor.cuda()
                _dist.broadcast(gpu, src=0)
                tensor.copy_(gpu)
        return self

    @classmethod
    def load(cls, path: str, device: torch.device = None) -> "QronosStats":
        """Load from pickle file."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        sigma_x = data.get("sigma_x", data["sigma_xhat"])  # Fallback for old files
        sigma_xhat = data["sigma_xhat"]
        sigma_x_xhat = data["sigma_x_xhat"]

        # Optional: residual compensation stats (may not exist in older files)
        sigma_delta_r_xhat = data.get("sigma_delta_r_xhat", None)

        if device is not None:
            sigma_x = sigma_x.to(device)
            sigma_xhat = sigma_xhat.to(device)
            sigma_x_xhat = sigma_x_xhat.to(device)
            if sigma_delta_r_xhat is not None:
                sigma_delta_r_xhat = sigma_delta_r_xhat.to(device)

        return cls(
            sigma_x=sigma_x,
            sigma_xhat=sigma_xhat,
            sigma_x_xhat=sigma_x_xhat,
            nseq=data["nseq"],
            ntokens=data["ntokens"],
            sigma_delta_r_xhat=sigma_delta_r_xhat,
        )


class QronosStatsAccumulator:
    """Accumulator for Qronos statistics.

    Collects activations from both unquantized (X) and quantized (X̂) models
    and computes:
    - Σ_X = X^T @ X (accumulated) - unquantized activations covariance
    - Σ_X̂ = X̂^T @ X̂ (accumulated) - quantized activations covariance
    - Σ_XX̂ = X^T @ X̂ (accumulated) - cross-covariance
    """

    def __init__(
        self,
        n_features: int,
        device: torch.device,
        dtype: torch.dtype = torch.float64,
        precomputed_sigma_x: Optional[torch.Tensor] = None,
    ):
        self.n = n_features
        self.device = device
        self.dtype = dtype
        self._has_precomputed_sigma_x = precomputed_sigma_x is not None

        # Accumulated statistics
        if precomputed_sigma_x is not None:
            # Use precomputed σ_X (already normalized E[X X^T]) — skip accumulation
            self.sigma_x = precomputed_sigma_x.to(device=device, dtype=dtype)
        else:
            self.sigma_x = torch.zeros((n_features, n_features), device=device, dtype=dtype)
        self.sigma_xhat = torch.zeros((n_features, n_features), device=device, dtype=dtype)
        self.sigma_x_xhat = torch.zeros((n_features, n_features), device=device, dtype=dtype)

        self.nseq = 0
        self.ntokens = 0

    def accumulate(self, X: torch.Tensor, X_hat: torch.Tensor,
                   weights: Optional[torch.Tensor] = None) -> None:
        """Accumulate statistics from a batch.

        Args:
            X: Unquantized activations, shape (batch, seq_len, n_features) or (tokens, n_features)
            X_hat: Quantized activations, same shape as X
            weights: Optional per-token weights, shape (batch, seq_len).
                     When provided, computes weighted covariance: (sqrt(w)*x)^T @ (sqrt(w)*x) = x^T diag(w) x
        """
        # Flatten to (tokens, features) and move to accumulator device
        X_flat = X.detach().reshape(-1, X.shape[-1]).to(device=self.device, dtype=self.dtype)
        X_hat_flat = X_hat.detach().reshape(-1, X_hat.shape[-1]).to(device=self.device, dtype=self.dtype)

        assert X_flat.shape == X_hat_flat.shape, f"Shape mismatch: {X_flat.shape} vs {X_hat_flat.shape}"
        assert X_flat.shape[-1] == self.n, f"Feature dim mismatch: {X_flat.shape[-1]} vs {self.n}"

        if weights is not None:
            # sqrt(p_j) scaling: (sqrt(p)*x)^T @ (sqrt(p)*x) = x^T diag(p) x
            sqrt_w = weights.reshape(-1).to(self.dtype).sqrt().unsqueeze(-1)  # (tokens, 1)
            X_flat = sqrt_w * X_flat
            X_hat_flat = sqrt_w * X_hat_flat

        # Update counts
        self.nseq += 1
        self.ntokens += X_flat.shape[0]

        # Accumulate: Σ_X = X^T @ X (skip if precomputed hessian was provided)
        if not self._has_precomputed_sigma_x:
            self.sigma_x.addmm_(X_flat.T, X_flat)

        # Accumulate: Σ_X̂ = X̂^T @ X̂ (quantized activations covariance)
        self.sigma_xhat.addmm_(X_hat_flat.T, X_hat_flat)

        # Accumulate: Σ_XX̂ = X^T @ X̂ (cross-covariance, order matters!)
        self.sigma_x_xhat.addmm_(X_flat.T, X_hat_flat)

    def get(
        self,
        normalize: bool = True,
        normalize_by: str = "tokens",
        eps: float = 1e-12,
    ) -> QronosStats:
        """Get the accumulated statistics.

        Args:
            normalize: Whether to normalize by count
            normalize_by: "tokens" (default, gives true E[X X^T]) or "seq"
            eps: Small value added to diagonal for numerical stability
        """
        if self.nseq <= 0:
            raise RuntimeError("No samples accumulated")

        # Transfer ownership instead of cloning to avoid OOM on large matrices
        # (w2 accumulators are 3 × 28672² × 8 = 18.4 GiB, clone needs another 6 GiB).
        sigma_x = self.sigma_x
        sigma_xhat = self.sigma_xhat
        sigma_x_xhat = self.sigma_x_xhat
        self.sigma_x = None
        self.sigma_xhat = None
        self.sigma_x_xhat = None

        if normalize:
            if normalize_by in ("seq", "sequences", "batch"):
                denom = max(self.nseq, 1)
            elif normalize_by in ("token", "tokens"):
                denom = max(self.ntokens, 1)
            else:
                raise ValueError(f"normalize_by must be 'seq' or 'tokens', got {normalize_by!r}")

            # sigma_x: skip normalization if precomputed (already E[X X^T])
            if not self._has_precomputed_sigma_x:
                sigma_x.div_(float(denom))
            sigma_xhat.div_(float(denom))
            sigma_x_xhat.div_(float(denom))

        # If accumulated in fp32, upcast to fp64 for downstream numerical stability
        # (compress_w2q, Cholesky, rescaler all require fp64).
        if self.dtype != torch.float64:
            if not self._has_precomputed_sigma_x:
                sigma_x = sigma_x.to(torch.float64)
            sigma_xhat = sigma_xhat.to(torch.float64)
            sigma_x_xhat = sigma_x_xhat.to(torch.float64)

        # Add small ridge to covariance matrices for numerical stability
        diag = torch.arange(self.n, device=sigma_x.device)
        if not self._has_precomputed_sigma_x:
            sigma_x[diag, diag] += eps
        sigma_xhat[diag, diag] += eps

        return QronosStats(
            sigma_x=sigma_x,
            sigma_xhat=sigma_xhat,
            sigma_x_xhat=sigma_x_xhat,
            nseq=self.nseq,
            ntokens=self.ntokens,
        )


@torch.no_grad()
def compute_qronos_target(
    W: torch.Tensor,
    stats: QronosStats,
    eps: float = 1e-6,
    warn_threshold: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Qronos quantization target.

    The target is: ŷ = W @ Σ_XX̂ @ (L̂^T)^{-1}
    where Σ_X̂ = L̂ @ L̂^T (Cholesky)

    Args:
        W: Weight matrix, shape (out_features, in_features)
        stats: QronosStats containing Σ_X̂ and Σ_XX̂
        eps: Regularization for Cholesky
        warn_threshold: Warn if diagonal elements are below this

    Returns:
        (y_hat, L_hat): Target vector and Cholesky factor
    """
    sigma_xhat = stats.sigma_xhat
    sigma_x_xhat = stats.sigma_x_xhat

    n = sigma_xhat.shape[0]

    # Check for small diagonal elements and regularize
    diag = sigma_xhat.diag()
    small_diag_mask = diag.abs() < warn_threshold
    if small_diag_mask.any():
        n_small = small_diag_mask.sum().item()
        print(f"[qronos] Warning: {n_small}/{n} diagonal elements below {warn_threshold}, adding regularization")

        # Add regularization proportional to median diagonal
        median_diag = diag.abs().median().item()
        reg = max(eps, median_diag * 0.01)
        sigma_xhat = sigma_xhat.clone()
        sigma_xhat.diagonal().add_(reg)

    # Cholesky decomposition: Σ_X̂ = L̂ @ L̂^T
    try:
        L_hat = torch.linalg.cholesky(sigma_xhat, upper=False)
    except RuntimeError as e:
        print(f"[qronos] Cholesky failed, adding stronger regularization: {e}")
        # Fallback: add stronger regularization
        sigma_xhat = sigma_xhat.clone()
        diag_mean = sigma_xhat.diag().abs().mean().item()
        sigma_xhat.diagonal().add_(max(eps, diag_mean * 0.1))
        L_hat = torch.linalg.cholesky(sigma_xhat, upper=False)

    # Compute y_hat = W @ Σ_XX̂ @ L̂^{-1}
    # solve_triangular(L^T, B, upper=True) solves L^T @ X = B
    # X = (L^T)^{-1} @ B, then X^T = B^T @ L^{-T} = B^T @ (L^{-1})^T
    # With B = Σ^T, we get X^T = Σ @ L^{-1}

    # Solve L̂^T @ X = Σ_XX̂^T for X
    X = torch.linalg.solve_triangular(L_hat.T, sigma_x_xhat.T, upper=True)
    # temp = X^T = Σ_XX̂ @ L̂^{-1}
    temp = X.T

    # Now y_hat = W @ temp
    y_hat = W @ temp

    return y_hat, L_hat


@torch.no_grad()
def compute_qronos_hessian(stats: QronosStats) -> torch.Tensor:
    """Get the Hessian for Qronos (Σ_X̂).

    This is used in place of the standard Hessian for loss computation.
    """
    return stats.sigma_xhat


# ==============================================================================
# Residual Stream Compensation Statistics
# ==============================================================================

@dataclass
class ResidualStats:
    """Statistics for residual stream compensation in wo/w2 layers.

    For layers that output to the residual stream (wo, w2):
    - Y = WX + R  where R is the residual (skip connection input)
    - ΔR = R - R̂  is the residual error from quantized upstream layers
    - Σ_{ΔR,X̂} = E[ΔR X̂^T]  is the cross-covariance we need

    The corrected quantization target becomes:
      ŷ = (W Σ_{X,X̂} + Σ_{ΔR,X̂}) (L̂^T)^{-1}
    """

    # Σ_{ΔR,X̂} = E[(R - R̂) X̂^T] - cross-covariance of residual error and quantized input
    # Shape: (out_features, in_features) where out_features is residual dim
    sigma_delta_r_xhat: torch.Tensor

    # Number of sequences/samples used
    nseq: int

    # Number of tokens used
    ntokens: int

    def save(self, path: str) -> None:
        """Save to pickle file."""
        data = {
            "sigma_delta_r_xhat": self.sigma_delta_r_xhat.cpu(),
            "nseq": self.nseq,
            "ntokens": self.ntokens,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def broadcast_from_rank0(self) -> "ResidualStats":
        """Broadcast sigma matrix from rank 0 for cross-rank consistency."""
        import torch.distributed as _dist
        if not (_dist.is_available() and _dist.is_initialized() and _dist.get_world_size() > 1):
            return self
        t = self.sigma_delta_r_xhat
        if t.is_cuda:
            _dist.broadcast(t, src=0)
        else:
            gpu = t.cuda()
            _dist.broadcast(gpu, src=0)
            t.copy_(gpu)
        return self

    @classmethod
    def load(cls, path: str, device: torch.device = None) -> "ResidualStats":
        """Load from pickle file."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        sigma_delta_r_xhat = data["sigma_delta_r_xhat"]
        if device is not None:
            sigma_delta_r_xhat = sigma_delta_r_xhat.to(device)

        return cls(
            sigma_delta_r_xhat=sigma_delta_r_xhat,
            nseq=data["nseq"],
            ntokens=data["ntokens"],
        )


class ResidualStatsAccumulator:
    """Accumulator for residual stream compensation statistics.

    Collects:
    - R: residual from unquantized model (skip connection input)
    - R̂: residual from quantized model
    - X̂: quantized input activations to the layer

    Computes: Σ_{ΔR,X̂} = E[(R - R̂) X̂^T]

    For wo layers:
      - R is the hidden state h_in before attention (the skip connection value)
      - X̂ is the input to wo after attention computation

    For w2 layers:
      - R is the hidden state h_mid after attention (before FFN, the skip connection value)
      - X̂ is the input to w2 after FFN activations
    """

    def __init__(
        self,
        out_features: int,  # residual dimension (hidden_dim)
        in_features: int,   # input dimension to the layer
        device: torch.device,
        dtype: torch.dtype = torch.float64,
    ):
        self.out_features = out_features
        self.in_features = in_features
        self.device = device
        self.dtype = dtype

        # Accumulated: Σ_{ΔR,X̂} = (R - R̂)^T @ X̂ accumulated over batches
        # Shape: (out_features, in_features)
        self.sigma_delta_r_xhat = torch.zeros(
            (out_features, in_features), device=device, dtype=dtype
        )

        self.nseq = 0
        self.ntokens = 0

    def accumulate(
        self,
        R: torch.Tensor,       # Residual from unquantized model: (batch, seq, hidden_dim)
        R_hat: torch.Tensor,   # Residual from quantized model: (batch, seq, hidden_dim)
        X_hat: torch.Tensor,   # Quantized input to layer: (batch, seq, in_features)
    ) -> None:
        """Accumulate statistics from a batch.

        Args:
            R: Unquantized residual, shape (batch, seq_len, out_features)
            R_hat: Quantized residual, same shape as R
            X_hat: Quantized input to the layer, shape (batch, seq_len, in_features)
        """
        # Flatten to (tokens, features) and move to accumulator device
        R_flat = R.detach().reshape(-1, R.shape[-1]).to(device=self.device, dtype=self.dtype)
        R_hat_flat = R_hat.detach().reshape(-1, R_hat.shape[-1]).to(device=self.device, dtype=self.dtype)
        X_hat_flat = X_hat.detach().reshape(-1, X_hat.shape[-1]).to(device=self.device, dtype=self.dtype)

        assert R_flat.shape == R_hat_flat.shape
        assert R_flat.shape[-1] == self.out_features
        assert X_hat_flat.shape[-1] == self.in_features
        assert R_flat.shape[0] == X_hat_flat.shape[0], "Token count mismatch"

        # Update counts
        self.nseq += 1
        self.ntokens += R_flat.shape[0]

        # ΔR = R - R̂
        delta_R = R_flat - R_hat_flat

        # Accumulate: Σ_{ΔR,X̂} += ΔR^T @ X̂
        # Result shape: (out_features, in_features)
        self.sigma_delta_r_xhat.addmm_(delta_R.T, X_hat_flat)

    def get(
        self,
        normalize: bool = True,
        normalize_by: str = "tokens",
    ) -> ResidualStats:
        """Get the accumulated statistics.

        Args:
            normalize: Whether to normalize by count
            normalize_by: "tokens" (default) or "seq"
        """
        if self.nseq <= 0:
            raise RuntimeError("No samples accumulated")

        # Transfer ownership instead of cloning to save GPU memory.
        sigma_delta_r_xhat = self.sigma_delta_r_xhat
        self.sigma_delta_r_xhat = None

        if normalize:
            if normalize_by in ("seq", "sequences", "batch"):
                denom = max(self.nseq, 1)
            elif normalize_by in ("token", "tokens"):
                denom = max(self.ntokens, 1)
            else:
                raise ValueError(f"normalize_by must be 'seq' or 'tokens', got {normalize_by!r}")

            sigma_delta_r_xhat.div_(float(denom))

        return ResidualStats(
            sigma_delta_r_xhat=sigma_delta_r_xhat,
            nseq=self.nseq,
            ntokens=self.ntokens,
        )
