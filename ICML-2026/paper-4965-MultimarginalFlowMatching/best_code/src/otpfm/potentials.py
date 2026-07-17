"""
Potentials for OTP-FM. Each potential is based on a distance metric 𝐷(ρ, μ) between the learned marginal ρ and the target μ.

Implements:

- :class:`W2InfPotential`: Random coupling between samples, fastest and default recommendation.
- :class:`W2Potential`: Exact Wasserstein-2 (requires ``pot`` package).
- :class:`MMDRBFPotential`: MMD with RBF kernel.
- :class:`KLPotential`: KL divergence with score estimation.
- :class:`MMDPolyPotential`: MMD with polynomial kernel (not successful in experiments).
- :class:`EntropicW2Potential`: Entropic Wasserstein-2 (requires ``pot`` package, not a significant speed-up over regular :class:`W2Potential`).
"""

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from .lambdas import BoxLambda, DeltaLambda, GaussianLambda, LambdaFunction, TriangleLambda


@dataclass
class Potential(ABC):
    """
    Abstract base class to represent an OTP potential.

    Args:
        tk (float): central time.
        strength (float): strength of the potential.
        lambda_type (str): lambda function type. Choose from {"gaussian", "triangle", "box", "delta"}. Default is "gaussian".
        width (float): width of the potential in time. Set to "auto" to automatically set it to (average Δt to adjacent marginals) / 2. Default is None.
    """

    tk: float
    strength: float
    lambda_type: str = "gaussian"
    width: float | str | None = None

    # Initialized in __post_init__
    lambda_fn: LambdaFunction = field(init=False, repr=False)

    def __post_init__(self):
        if self.width is None and self.lambda_type != "delta":
            warnings.warn(
                "Potential width is not specified and will be determined automatically."
                "Set to 'auto' if this is intended and you don't want this warning in the future.",
                UserWarning,
                stacklevel=2,
            )
            self.width = "auto"
        elif self.lambda_type == "delta":
            self.width = 0.0

        match self.lambda_type:
            case "gaussian":
                self.lambda_fn = GaussianLambda(self.tk, self.width)
            case "triangle":
                self.lambda_fn = TriangleLambda(self.tk, self.width)
            case "box":
                self.lambda_fn = BoxLambda(self.tk, self.width)
            case "delta":
                self.lambda_fn = DeltaLambda(self.tk, self.width)
            case _:
                raise ValueError(f"Unknown lambda function: {self.lambda_type}")

    def set_width(self, width: float | str):
        self.width = width
        self.lambda_fn.set_width(width)

    @abstractmethod
    def grad_gk(self, x_true: Tensor, X_tk: Tensor) -> Tensor:
        """
        ∇g_k(X_tk, tk), where g_k = δD/δρ, see Thm. 3.2 and Table 1 of :cite:t:`kansal2026otpfm`.
        Must be implemented by subclasses.

        Args:
            x_true: samples from true intermediate marginal, shape ``(batch_size, *dim)``
            X_tk: predicted samples from learnt path at time tk, shape ``(batch_size, *dim)``

        Returns:
            ∇g_k(X_tk, tk), shape ``(batch_size, *dim)``
        """
        pass

    def dV_tk(self, x_true: Tensor, X_tk: Tensor) -> Tensor:
        """
        δV/δρ evaluated at central time tk.

        Args:
            x_true: samples from true intermediate marginal, shape ``(batch_size, *dim)``
            X_tk: predicted samples from learnt path at time tk, shape ``(batch_size, *dim)``

        Returns:
            strength * grad_gk, shape ``(batch_size, *dim)``
        """
        return self.strength * self.grad_gk(x_true, X_tk)


@dataclass
class W2InfPotential(Potential):
    """
    W2-infinity potential (independent/random coupling).

    Equivalent to OT map with infinite entropy regularization.
    The gradient is simply (X_tk - x_true).
    """

    def grad_gk(self, x_true: Tensor, X_tk: Tensor) -> Tensor:
        return X_tk - x_true


@dataclass
class W2Potential(Potential):
    """
    Wasserstein-2 distance potential.
    ∇g_k is X - T(X), where T is the OT plan.

    Note: Requires the `pot` package for optimal transport computation, which will be slow.
    We recommend using W2InfPotential with pre-computed OT alignments or random couplings for most applications.
    """

    def __post_init__(self):
        super().__post_init__()
        try:
            import ot  # noqa: F401
        except ImportError:
            raise ImportError(
                "otpfm needs to be installed with otpfm[w2] to enable W2Potentials. "
                "Note that this will slow training; we recommend W2InfPotential with "
                "pre-computed OT alignments or random couplings for most applications."
            )

    def grad_gk(self, x_true: Tensor, X_tk: Tensor) -> Tensor:
        import ot

        T = ot.solve_sample(X_tk.detach().cpu().numpy(), x_true.detach().cpu().numpy()).plan
        _, tgt_indices = T.nonzero()
        aligned_x_true = x_true[tgt_indices]
        return X_tk - aligned_x_true


@dataclass
class MMDRBFPotential(Potential):
    r"""Maximum Mean Discrepancy (MMD) potential with an RBF kernel.

    grad_gk(x_i) = 2 * (E_{y~\rho}[\nabla_x k(x_i,y)] - E_{y~\mu}[\nabla_x k(x_i,y)]).

    For the RBF kernel k(x,y) = exp(-||x-y||^2 / (2*sigma^2)),
        \nabla_x k(x,y) = k(x,y) * (-(x-y) / sigma^2).

    The gradient is computed for each bandwidth in sigma and then averaged,
    providing a multi-scale MMD that is more robust to bandwidth selection.

    Default bandwidth is [3.0], which is reasonable for unit-variance-normalized data, but likely needs to be tuned.

    Args:
        sigma (list[float]): List of RBF bandwidths. The gradient is averaged over all bandwidths.
    """

    sigma: list[float] = field(default_factory=lambda: [3.0])

    def grad_gk(self, x_true: Tensor, X_tk: Tensor) -> Tensor:
        # Flatten to (N, D) for kernel computations, then reshape back.

        sigma = np.atleast_1d(self.sigma)

        x = X_tk
        y_mu = x_true
        orig_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        y_mu = y_mu.reshape(y_mu.shape[0], -1)

        # (N, N) pairwise squared distances
        # d2[i,j] = ||x_i - y_j||^2
        # Use broadcasting for stability and speed.
        diff_xx = x[:, None, :] - x[None, :, :]
        diff_xmu = x[:, None, :] - y_mu[None, :, :]

        # Compute gradient for each sigma and average
        grads = []
        for sigma_val in sigma:
            sigma2 = float(sigma_val) ** 2
            if sigma2 <= 0:
                raise ValueError("sigma must be positive for RBF kernel")

            k_xx = torch.exp(-diff_xx.pow(2).sum(dim=-1) / (2.0 * sigma2))  # (N, N)
            k_xmu = torch.exp(-diff_xmu.pow(2).sum(dim=-1) / (2.0 * sigma2))  # (N, N_mu)

            # grad_x k(x,y) = k(x,y) * (-(x-y)/sigma^2)
            grad_k_xx = k_xx.unsqueeze(-1) * (-diff_xx / sigma2)  # (N, N, D)
            grad_k_xmu = k_xmu.unsqueeze(-1) * (-diff_xmu / sigma2)  # (N, N_mu, D)

            # Expectations over samples: average over the second dimension.
            # Note: including self-interaction in E_{y~rho} is standard in minibatch MMD; if desired,
            # you can remove the diagonal, but we keep it for simplicity and lower variance.
            e_rho = grad_k_xx.mean(dim=1)  # (N, D)
            e_mu = grad_k_xmu.mean(dim=1)  # (N, D)

            grad = 2.0 * (e_rho - e_mu)
            grads.append(grad)

        # Average over all sigmas
        grad_avg = torch.stack(grads).mean(dim=0)
        return grad_avg.reshape(orig_shape)


@dataclass
class KLPotential(Potential):
    r"""
    KL divergence potential.

    ∇g_k = ∇log ρ(x) - ∇log μ(x)

    where ρ is the learned path marginal and μ is the target marginal.

    Supports two main score estimation methods:
        - "gaussian": Closed-form for Gaussian distributions (fastest, but not recommended for highly non-Gaussian or high-dimensional distributions)
        - "kde": Kernel density estimation (default recommendation for most applications)
            Default bandwidth is 3.0, which is reasonable for unit-variance-normalized data, but likely needs to be tuned.

    And some experimental methods that were less successful:
        - "sliced": Sliced score estimation
        - "ssge": Spectral Stein Gradient Estimato

    Args:
        rho_method: Score estimation method for ρ. One of {"gaussian", "kde", "sliced", "ssge"}.
        mu_method: Score estimation method for μ. One of {"gaussian", "kde", "sliced", "ssge"}. If None, uses rho_method.
        bandwidth: KDE/SSGE bandwidth. Either a float or "silverman"/"scott" for automatic selection.
            Automatic bandwidths are computed from x_true only (the fixed target marginal) to ensure
            a stationary potential during fixed-point iteration. Default is 3.0.
        bandwidth_scale: Multiplier for automatic bandwidth (default 2.5). Larger values provide
            smoother, more stable score estimates. Ignored if bandwidth is a float.
        ssge_eta: Regularization for SSGE (default 0.01). Larger values = more stable but biased.
        n_projections: Number of random projections for sliced method (default 64).
        diagonal_cov: If True, use diagonal covariance for Gaussian method (faster).
    """

    rho_method: str = "sliced"
    mu_method: str | None = None
    bandwidth: float | str = 3.0
    bandwidth_scale: float = 2.5
    ssge_eta: float = 0.01  # Regularization for SSGE
    n_projections: int = 64
    diagonal_cov: bool = False
    cache_projections: bool = True  # Cache random projections for FP stability

    # Method callables set in __post_init__
    _rho_score_fn: callable = field(init=False, repr=False)
    _mu_score_fn: callable = field(init=False, repr=False)
    _cached_V: Tensor | None = field(init=False, repr=False, default=None)
    _cached_V_shape: tuple | None = field(init=False, repr=False, default=None)

    def __post_init__(self):
        super().__post_init__()

        valid_methods = {"gaussian", "kde", "sliced", "ssge"}

        if self.rho_method not in valid_methods:
            raise ValueError(f"rho_method must be one of {valid_methods}, got {self.rho_method}")

        if self.mu_method is None:
            self.mu_method = self.rho_method
        elif self.mu_method not in valid_methods:
            raise ValueError(f"mu_method must be one of {valid_methods}, got {self.mu_method}")

        # Map method names to functions
        method_map = {
            "gaussian": self._gaussian_score,
            "kde": self._kde_score,
            "sliced": self._sliced_score,
            "ssge": self._ssge_score,
        }

        self._rho_score_fn = method_map[self.rho_method]
        self._mu_score_fn = method_map[self.mu_method]

        # Initialize projection cache
        self._cached_V = None
        self._cached_V_shape = None

    def _get_projections(self, d: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Get random projections, using cache if enabled and shape matches."""
        P = self.n_projections
        shape = (P, d)

        if self.cache_projections and self._cached_V is not None:
            if self._cached_V_shape == shape and self._cached_V.device == device:
                return self._cached_V

        # Generate new random projections
        V = torch.randn(P, d, device=device, dtype=dtype)
        V = V / V.norm(dim=1, keepdim=True)  # (P, D)

        if self.cache_projections:
            self._cached_V = V
            self._cached_V_shape = shape

        return V

    def reset_projections(self):
        """Reset cached projections. Call between training steps to get new random directions."""
        self._cached_V = None
        self._cached_V_shape = None

    def grad_gk(self, x_true: Tensor, X_tk: Tensor) -> Tensor:
        # For KDE, use optimized joint computation to avoid redundant work
        if self.rho_method == "kde" and self.mu_method == "kde":
            return self._kde_score_diff(X_tk, x_true)

        # For sliced, use optimized joint computation
        if self.rho_method == "sliced" and self.mu_method == "sliced":
            return self._sliced_score_diff(X_tk, x_true)

        # Score of ρ at X_tk (using X_tk as samples from ρ)
        # Use leave-one-out for methods that support it to avoid self-interaction bias
        if self.rho_method == "kde":
            score_rho = self._kde_score(X_tk, X_tk, leave_one_out=True)
        elif self.rho_method == "ssge":
            score_rho = self._ssge_score(X_tk, X_tk, leave_one_out=True)
        else:
            score_rho = self._rho_score_fn(X_tk, X_tk)

        # Score of μ at X_tk (using x_true as samples from μ)
        score_mu = self._mu_score_fn(X_tk, x_true)
        return score_rho - score_mu

    def _kde_score_diff(self, X_tk: Tensor, x_true: Tensor) -> Tensor:
        """
        KDE score difference: ∇log ρ(x) - ∇log μ(x)
        """
        orig_shape = X_tk.shape
        x = X_tk.reshape(X_tk.shape[0], -1)  # (N, D)
        y = x_true.reshape(x_true.shape[0], -1)  # (M, D)
        n, d = x.shape
        m = y.shape[0]

        # Compute bandwidth(s) as Python float
        if isinstance(self.bandwidth, int | float):
            h2_xx = h2_xy = float(self.bandwidth) ** 2
        else:
            # Silverman bandwidth
            std_y = y.std(dim=0).mean().item()
            h = self.bandwidth_scale * 1.06 * std_y * (m ** (-1 / (d + 4)))
            h2_xx = h2_xy = h**2

        # Pairwise differences
        diff_xx = x[:, None, :] - x[None, :, :]  # (N, N, D)
        diff_xy = x[:, None, :] - y[None, :, :]  # (N, M, D)

        # RBF kernels (potentially different bandwidths)
        k_xx = torch.exp(-diff_xx.pow(2).sum(dim=-1) / (2.0 * h2_xx))  # (N, N)
        k_xy = torch.exp(-diff_xy.pow(2).sum(dim=-1) / (2.0 * h2_xy))  # (N, M)

        # Kernel gradients
        grad_k_xx = k_xx.unsqueeze(-1) * (-diff_xx / h2_xx)  # (N, N, D)
        grad_k_xy = k_xy.unsqueeze(-1) * (-diff_xy / h2_xy)  # (N, M, D)

        # LEAVE-ONE-OUT for score_rho to avoid self-interaction bias:
        # k(x_i, x_i) = 1 inflates denominator while ∇k(x_i, x_i) = 0 contributes nothing
        # This biases score toward zero, especially with small bandwidth
        k_xx_loo = k_xx.clone()
        k_xx_loo.fill_diagonal_(0)  # Exclude self-kernel from denominator
        k_xx_sum_loo = k_xx_loo.sum(dim=1, keepdim=True)  # Sum over n-1 neighbors

        # score_rho with leave-one-out (numerator unchanged since self-gradient is 0)
        score_rho = grad_k_xx.sum(dim=1) / k_xx_sum_loo.clamp(min=1e-8)  # (N, D)

        # score_mu doesn't need LOO (x points are different from y samples)
        k_xy_sum = k_xy.sum(dim=1, keepdim=True)  # (N, 1)
        score_mu = grad_k_xy.sum(dim=1) / k_xy_sum.clamp(min=1e-8)  # (N, D)

        return (score_rho - score_mu).reshape(orig_shape)

    def _gaussian_score(self, x: Tensor, p_samples: Tensor) -> Tensor:
        """
        Closed-form score for Gaussian distribution.

        For N(μ, Σ): ∇log p(x) = -Σ⁻¹(x - μ)
        """
        orig_shape = x.shape
        x_flat = x.reshape(x.shape[0], -1)
        s_flat = p_samples.reshape(p_samples.shape[0], -1)
        d = x_flat.shape[-1]
        device = x_flat.device

        # Estimate Gaussian parameters from samples
        mu = s_flat.mean(dim=0)  # (D,)

        if self.diagonal_cov:
            # Diagonal covariance: O(nd)
            var = s_flat.var(dim=0).clamp(min=1e-6)  # (D,)
            score = -(x_flat - mu) / var  # (N, D)
        else:
            # Full covariance: O(nd²)
            cov = torch.cov(s_flat.T) + 1e-4 * torch.eye(d, device=device)  # (D, D)
            score = -torch.linalg.solve(cov, (x_flat - mu).T).T  # (N, D)

        return score.reshape(orig_shape)

    def _kde_score(self, x: Tensor, p_samples: Tensor, leave_one_out: bool = False) -> Tensor:
        """
        KDE-based score estimation.

        ∇log p̂(x) = (Σ_j ∇k(x, x_j)) / (Σ_j k(x, x_j))

        where k is an RBF kernel.

        Args:
            leave_one_out: If True, exclude self-kernel when x == p_samples.
                          Used when computing score of empirical distribution
                          at its own sample points to avoid self-interaction bias.
        """
        orig_shape = x.shape
        x_flat = x.reshape(x.shape[0], -1)
        s_flat = p_samples.reshape(p_samples.shape[0], -1)
        n, d = s_flat.shape

        # Compute bandwidth
        h = self._compute_bandwidth(s_flat)
        h2 = h**2

        # Pairwise differences: (N, M, D)
        diff = x_flat[:, None, :] - s_flat[None, :, :]
        d2 = diff.pow(2).sum(dim=-1)  # (N, M)

        # RBF kernel
        k = torch.exp(-d2 / (2 * h2))  # (N, M)

        # ∇_x k(x,y) = k(x,y) · (-(x-y)/h²)
        grad_k = k.unsqueeze(-1) * (-diff / h2)  # (N, M, D)

        # ∇log p̂ = (Σ_j ∇k) / (Σ_j k)
        if leave_one_out and x_flat.shape[0] == s_flat.shape[0]:
            # Exclude self-kernel from denominator to avoid bias
            k_loo = k.clone()
            k_loo.fill_diagonal_(0)
            score = grad_k.sum(dim=1) / k_loo.sum(dim=1, keepdim=True).clamp(min=1e-8)
        else:
            score = grad_k.sum(dim=1) / k.sum(dim=1, keepdim=True).clamp(min=1e-8)  # (N, D)

        return score.reshape(orig_shape)

    def _sliced_score(self, x: Tensor, p_samples: Tensor) -> Tensor:
        """
        Sliced score estimation.

        Projects to random 1D directions, estimates score there, projects back, to avoid the curse of dimensionality.

        Key identity: ∇log p(x) = d · E_v[(v · ∇log p(x)) · v]
        since E_v[v v^T] = (1/d) I_d for random unit vectors uniformly on sphere.
        """
        orig_shape = x.shape
        device = x.device
        dtype = x.dtype

        x_flat = x.reshape(x.shape[0], -1)  # (N, D)
        s_flat = p_samples.reshape(p_samples.shape[0], -1)  # (M, D)
        n, d = x_flat.shape
        m = s_flat.shape[0]

        P = self.n_projections

        # Get projections (cached for FP stability)
        V = self._get_projections(d, device, dtype)

        # Project samples to 1D: (N, P) and (M, P)
        proj_x = x_flat @ V.T
        proj_samples = s_flat @ V.T

        # FAST 1D score estimation - fully vectorized
        score_1d = self._fast_1d_kde_score(proj_x, proj_samples)  # (N, P)

        # Back-project: ∇log p(x) = d · (1/P) Σ_p score_1d_p · v_p
        grad = (d / P) * (score_1d @ V)  # (N, D)

        return grad.reshape(orig_shape)

    def _fast_1d_kde_score(self, x: Tensor, samples: Tensor) -> Tensor:
        """
        Fast vectorized 1D KDE score estimation using k-nearest neighbors.

        Instead of full O(N*M) KDE, uses O(N*K) computation with K nearest neighbors.
        For 1D, we can find K-NN efficiently using sorting.

        Score approximation: score(x) ≈ -Σ_{k-nn} (x - s_k) / (K * h²)
        where neighbors are weighted by Gaussian kernel.
        """
        N, P = x.shape
        M = samples.shape[0]
        device = x.device
        dtype = x.dtype

        # Bandwidth per projection (Silverman's rule for 1D)
        std = samples.std(dim=0).clamp(min=1e-6)  # (P,)
        h = 1.06 * std * (M**-0.2)  # (P,)
        h2 = h**2  # (P,)

        # For small P, process all at once (most efficient)
        if N * M * P <= 16_000_000:  # ~64MB for float32
            # Direct vectorized computation - very fast for small P
            diff = x[:, None, :] - samples[None, :, :]  # (N, M, P)
            z2 = diff**2 / h2  # (N, M, P)
            k = torch.exp(-0.5 * z2)  # (N, M, P)
            # Fused gradient computation
            scores = (k * (-diff / h2)).sum(dim=1) / k.sum(dim=1).clamp(min=1e-8)
            return scores

        # For larger P, process in chunks but with larger chunk size
        chunk_size = max(1, 16_000_000 // (N * M))
        scores = torch.empty(N, P, device=device, dtype=dtype)

        for i in range(0, P, chunk_size):
            j = min(i + chunk_size, P)
            diff = x[:, None, i:j] - samples[None, :, i:j]
            h2_chunk = h2[i:j]
            z2 = diff**2 / h2_chunk
            k = torch.exp(-0.5 * z2)
            scores[:, i:j] = (k * (-diff / h2_chunk)).sum(dim=1) / k.sum(dim=1).clamp(min=1e-8)

        return scores

    def _sliced_score_diff(self, X_tk: Tensor, x_true: Tensor) -> Tensor:
        """
        Optimized sliced score difference: ∇log ρ(x) - ∇log μ(x)

        Computes both scores with shared projections for efficiency.
        Uses cached projections for stable fixed-point iteration.
        """
        orig_shape = X_tk.shape
        device = X_tk.device
        dtype = X_tk.dtype

        x = X_tk.reshape(X_tk.shape[0], -1)  # (N, D)
        y = x_true.reshape(x_true.shape[0], -1)  # (M, D)
        n, d = x.shape
        m = y.shape[0]
        P = self.n_projections

        # Get projections (cached for FP stability)
        V = self._get_projections(d, device, dtype)

        # Project all samples
        proj_x = x @ V.T  # (N, P)
        proj_y = y @ V.T  # (M, P)

        # Bandwidth per projection (Silverman's rule for 1D, using target samples)
        std = proj_y.std(dim=0).clamp(min=1e-6)  # (P,)
        h = 1.06 * std * (m**-0.2)  # (P,)
        h2 = h**2  # (P,)

        # Compute 1D scores efficiently
        # Use single vectorized computation if memory allows
        if n * max(n, m) * P <= 16_000_000:
            # score_rho: KDE at x using x as samples (leave-one-out)
            diff_xx = proj_x[:, None, :] - proj_x[None, :, :]  # (N, N, P)
            z2_xx = diff_xx**2 / h2
            k_xx = torch.exp(-0.5 * z2_xx)  # (N, N, P)

            # Leave-one-out: zero out diagonal
            diag_mask = torch.eye(n, device=device, dtype=torch.bool).unsqueeze(-1)
            k_xx_loo = k_xx.masked_fill(diag_mask, 0)

            score_rho_1d = (k_xx_loo * (-diff_xx / h2)).sum(dim=1) / k_xx_loo.sum(dim=1).clamp(
                min=1e-8
            )  # (N, P)

            # score_mu: KDE at x using y as samples
            diff_xy = proj_x[:, None, :] - proj_y[None, :, :]  # (N, M, P)
            z2_xy = diff_xy**2 / h2
            k_xy = torch.exp(-0.5 * z2_xy)  # (N, M, P)

            score_mu_1d = (k_xy * (-diff_xy / h2)).sum(dim=1) / k_xy.sum(dim=1).clamp(
                min=1e-8
            )  # (N, P)
        else:
            # Chunked computation for large P
            chunk_size = max(1, 16_000_000 // (n * max(n, m)))
            score_rho_1d = torch.empty(n, P, device=device, dtype=dtype)
            score_mu_1d = torch.empty(n, P, device=device, dtype=dtype)

            for i in range(0, P, chunk_size):
                j = min(i + chunk_size, P)
                h2_c = h2[i:j]

                # score_rho chunk
                diff_xx = proj_x[:, None, i:j] - proj_x[None, :, i:j]
                k_xx = torch.exp(-0.5 * diff_xx**2 / h2_c)
                # LOO: zero diagonal
                diag_idx = torch.arange(n, device=device)
                k_xx[diag_idx, diag_idx, :] = 0
                score_rho_1d[:, i:j] = (k_xx * (-diff_xx / h2_c)).sum(dim=1) / k_xx.sum(
                    dim=1
                ).clamp(min=1e-8)

                # score_mu chunk
                diff_xy = proj_x[:, None, i:j] - proj_y[None, :, i:j]
                k_xy = torch.exp(-0.5 * diff_xy**2 / h2_c)
                score_mu_1d[:, i:j] = (k_xy * (-diff_xy / h2_c)).sum(dim=1) / k_xy.sum(dim=1).clamp(
                    min=1e-8
                )

        # Back-project score difference
        score_diff_1d = score_rho_1d - score_mu_1d  # (N, P)
        grad = (d / P) * (score_diff_1d @ V)  # (N, D)

        return grad.reshape(orig_shape)

    def _ssge_score(self, x: Tensor, p_samples: Tensor, leave_one_out: bool = False) -> Tensor:
        """
        Spectral Stein Gradient Estimator (SSGE) for score estimation :cite:p:`shi2018ssge`.

        Based on the Stein identity: E_p[s(x)k(x,y) + ∇_x k(x,y)] = 0

        This method in theory can capture higher-order moments better than simple KDE because
        it uses the Stein identity to constrain the score estimate, rather than smoothing the empirical distribution.

        Args:
            x: Query points (N, D)
            p_samples: Samples from the distribution (M, D)
            leave_one_out: If True, exclude self when x == p_samples

        Returns:
            Score estimates (N, D)
        """
        orig_shape = x.shape
        x_flat = x.reshape(x.shape[0], -1)  # (N, D)
        s_flat = p_samples.reshape(p_samples.shape[0], -1)  # (M, D)
        n, d = x_flat.shape
        m = s_flat.shape[0]
        device = x_flat.device
        dtype = x_flat.dtype

        # Compute bandwidth (using same logic as KDE)
        h = self._compute_bandwidth(s_flat)
        h2 = h**2

        # Kernel matrix K(samples, samples): (M, M)
        diff_ss = s_flat[:, None, :] - s_flat[None, :, :]  # (M, M, D)
        d2_ss = diff_ss.pow(2).sum(dim=-1)  # (M, M)
        K_ss = torch.exp(-d2_ss / (2.0 * h2))  # (M, M)

        # Gradient of kernel: ∇_x k(x, y) = k(x, y) · (y - x) / h²
        # G_i = (1/M) Σ_j ∇_{x_i} k(x_i, x_j) is the mean gradient (not sum!)
        grad_K_ss = K_ss.unsqueeze(-1) * (-diff_ss / h2)  # (M, M, D)
        G = grad_K_ss.mean(dim=1)  # (M, D) - mean of kernel gradients

        # Regularized inverse: (K + η*I)^{-1}
        eta = self.ssge_eta
        K_reg = K_ss + eta * torch.eye(m, device=device, dtype=dtype)

        # Solve (K + ηI) β = G for each dimension
        # β has shape (M, D)
        try:
            beta = torch.linalg.solve(K_reg, G)  # (M, D)
        except RuntimeError:
            # Fallback to pseudo-inverse if singular
            beta = torch.linalg.lstsq(K_reg, G).solution

        # Score at query points: s(x) = -K(x, samples) @ β
        # K(x, samples): (N, M)
        diff_xs = x_flat[:, None, :] - s_flat[None, :, :]  # (N, M, D)
        d2_xs = diff_xs.pow(2).sum(dim=-1)  # (N, M)
        K_xs = torch.exp(-d2_xs / (2.0 * h2))  # (N, M)

        if leave_one_out and n == m:
            # Zero out diagonal for self-interactions
            K_xs = K_xs.clone()
            K_xs.fill_diagonal_(0)

        # From Stein identity: K @ s = -G, so s = -K^{-1} @ G
        # With regularization: s = -(K+ηI)^{-1} @ G = -beta
        # Score at new point x: s(x) = K(x, samples) @ (-beta)
        # But we solved K_reg @ beta = G, so beta = K_reg^{-1} @ G
        # And s = -K_reg^{-1} @ G = -beta at sample points
        # For interpolation to new points: s(x) = K(x, X) @ alpha where alpha_i = s(x_i) / K_ii
        # Simpler: s(x) ≈ K(x, X) @ (-beta) / K(x,X).sum() for normalized interpolation

        # Actually the correct formula from Stein identity derivation:
        # s(x) = K(x, X) @ (K+ηI)^{-1} @ G  (positive sign, not negative)
        score = K_xs @ beta  # (N, D)

        return score.reshape(orig_shape)

    def _compute_bandwidth(self, samples: Tensor) -> float:
        """Compute KDE bandwidth using specified rule, scaled for score estimation."""
        n, d = samples.shape

        if isinstance(self.bandwidth, int | float):
            return float(self.bandwidth)
        elif self.bandwidth == "silverman":
            # Silverman's rule of thumb, scaled for score estimation
            std = samples.std(dim=0).mean().item()
            return self.bandwidth_scale * 1.06 * std * (n ** (-1 / (d + 4)))
        elif self.bandwidth == "scott":
            # Scott's rule, scaled for score estimation
            std = samples.std(dim=0).mean().item()
            return self.bandwidth_scale * std * (n ** (-1 / (d + 4)))
        else:
            raise ValueError(f"Unknown bandwidth rule: {self.bandwidth}")


@dataclass
class MMDPolyPotential(Potential):
    r"""Maximum Mean Discrepancy (MMD) potential with a polynomial kernel.

    grad_gk(x_i) = 2 * (E_{y~\rho}[\nabla_x k(x_i,y)] - E_{y~\mu}[\nabla_x k(x_i,y)]).

    Polynomial kernel: k(x,y) = (x^T y + c)^p.
        \nabla_x k(x,y) = p * (x^T y + c)^(p-1) * y.

    Args:
        degree (int): polynomial degree p (must be >= 1).
        coef0 (float): additive constant c.
    """

    degree: int = 2
    coef0: float = 1.0

    def grad_gk(self, x_true: Tensor, X_tk: Tensor) -> Tensor:
        if self.degree < 1:
            raise ValueError("degree must be >= 1 for polynomial kernel")

        x = X_tk
        y_mu = x_true
        orig_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        y_mu = y_mu.reshape(y_mu.shape[0], -1)

        # Dot products
        # s_xx[i,j]  = x_i^T x_j + c
        # s_xmu[i,j] = x_i^T y_j + c
        s_xx = x @ x.t() + float(self.coef0)  # (N, N)
        s_xmu = x @ y_mu.t() + float(self.coef0)  # (N, N_mu)

        p = int(self.degree)

        if p == 1:
            # k(x,y) = x^T y + c => grad_x k = y
            # E_rho[y] - E_mu[y]
            e_rho = x.mean(dim=0, keepdim=True).expand_as(x)  # (N, D)
            e_mu = y_mu.mean(dim=0, keepdim=True).expand_as(x)  # (N, D)
            grad = 2.0 * (e_rho - e_mu)
            return grad.reshape(orig_shape)

        # coeff matrices: p * (s)^(p-1)
        coeff_xx = p * s_xx.pow(p - 1)  # (N, N)
        coeff_xmu = p * s_xmu.pow(p - 1)  # (N, N_mu)

        # For each i: E_{y}[ p*(x_i^T y + c)^(p-1) * y ]
        # This is a weighted average of y vectors.
        e_rho = (coeff_xx @ x) / x.shape[0]  # (N, D)
        e_mu = (coeff_xmu @ y_mu) / y_mu.shape[0]  # (N, D)

        grad = 2.0 * (e_rho - e_mu)
        return grad.reshape(orig_shape)


@dataclass
class EntropicW2Potential(Potential):
    """
    Entropic OT potential using Sinkhorn algorithm.
    Much faster than exact OT (W2Potential) and runs on GPU.

    ∇g_k is x - T(x), where T is the entropic OT plan with barycentric projection.

    Args:
        reg (float): Entropy regularization parameter. Lower = closer to exact OT but slower.
        numItermax (int): Maximum number of Sinkhorn iterations.
    """

    reg: float = 0.01
    numItermax: int = 100

    def grad_gk(self, x_true: Tensor, X_tk: Tensor) -> Tensor:
        n = X_tk.shape[0]
        device = X_tk.device
        dtype = X_tk.dtype

        C = torch.cdist(X_tk.detach(), x_true.detach(), p=2).pow(2)

        # Log-domain Sinkhorn (stable for all reg values)
        log_K = -C / self.reg
        f = torch.zeros(n, device=device, dtype=dtype)
        g = torch.zeros(n, device=device, dtype=dtype)

        # Adaptive iterations: larger reg converges faster
        n_iter = min(self.numItermax, max(1, int(self.numItermax * 0.01 / self.reg)))

        for _ in range(n_iter):
            f = -torch.logsumexp(log_K + g.unsqueeze(0), dim=1)
            g = -torch.logsumexp(log_K + f.unsqueeze(1), dim=0)

        # Transport plan: stabilize by subtracting row-wise max before exp
        log_T = f.unsqueeze(1) + log_K + g.unsqueeze(0)
        log_T = log_T - log_T.max(dim=1, keepdim=True).values
        T = torch.exp(log_T)

        # Stochastic assignment: sample target index according to transport plan
        T_normalized = T / T.sum(dim=1, keepdim=True).clamp(min=1e-8)
        target_indices = torch.multinomial(T_normalized, num_samples=1).squeeze(-1)
        aligned_x_true = x_true[target_indices]

        return X_tk - aligned_x_true


__all__ = [
    "Potential",
    "W2InfPotential",
    "W2Potential",
    "MMDRBFPotential",
    "KLPotential",
    "MMDPolyPotential",
    "EntropicW2Potential",
]
