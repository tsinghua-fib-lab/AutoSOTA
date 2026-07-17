"""
Evaluation metrics for multi-marginal flow matching.

Implements:
 - Fréchet Gaussian Distance (FGD)
 - Maximum Mean Discrepancy (MMD)
 - Sliced Wasserstein Distance (SWD)
 - Wasserstein-1 Distance (W1)
 - Wasserstein-2 Distance (W2)

for comparing generated and ground truth distributions.

Author(s): Raghav Kansal
"""

import logging
import warnings

import numpy as np
import pandas as pd
import torch
from scipy import linalg
from torch import Tensor

try:
    import ot
    from ot.sliced import sliced_wasserstein_distance

    HAS_POT = True
except ImportError:
    HAS_POT = False

logger = logging.getLogger(__name__)


def compute_swd(
    generated: Tensor | np.ndarray,
    ground_truth: Tensor | np.ndarray,
    n_projections: int = 50,
) -> float:
    """
    Compute Sliced Wasserstein Distance between two distributions.

    Args:
        generated: Generated samples (n_samples, dim)
        ground_truth: Ground truth samples (m_samples, dim)
        n_projections: Number of random projections (default: 50)

    Returns:
        SWD value (scalar)
    """
    if not HAS_POT:
        raise ImportError(
            "POT library required for SWD. Install with: pip install 'otpfm[experiments]' or (standalone) pip install pot"
        )

    # Convert to numpy if tensor
    if isinstance(generated, Tensor):
        generated = generated.detach().cpu().numpy()
    if isinstance(ground_truth, Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()

    # Ensure float64 for numerical stability
    generated = generated.astype(np.float64)
    ground_truth = ground_truth.astype(np.float64)

    return float(sliced_wasserstein_distance(generated, ground_truth, n_projections=n_projections))


def compute_mmd(
    generated: Tensor | np.ndarray,
    ground_truth: Tensor | np.ndarray,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
) -> float:
    """
    Compute Maximum Mean Discrepancy (MMD) between two distributions.

    Uses multi-scale Gaussian kernel with automatic bandwidth selection:
    - Bandwidth computed from median pairwise distance
    - Multiple scales: bandwidth * kernel_mul^i for i in range(kernel_num)

    Args:
        generated: Generated samples (n_samples, dim)
        ground_truth: Ground truth samples (m_samples, dim)
        kernel_mul: Multiplier for bandwidth scaling (default: 2.0)
        kernel_num: Number of kernel scales (default: 5)

    Returns:
        MMD value (scalar)

    Note:
        Memory complexity is O(N^2) where N = n_source + n_target. The pairwise
        squared L2 distance matrix is computed via the identity
        ``||x - y||^2 = ||x||^2 + ||y||^2 - 2 <x, y>`` instead of materializing
        an (N, N, D) intermediate. For N=8000 and D=100 this drops the peak
        allocation from ~50 GB to ~770 MB.
    """
    # Convert to tensor if numpy
    if isinstance(generated, np.ndarray):
        generated = torch.from_numpy(generated).float()
    if isinstance(ground_truth, np.ndarray):
        ground_truth = torch.from_numpy(ground_truth).float()

    # Ensure on same device
    device = generated.device
    ground_truth = ground_truth.to(device)

    # Flatten to (N, D) if needed
    source = generated.reshape(generated.shape[0], -1)
    target = ground_truth.reshape(ground_truth.shape[0], -1)

    n_source = int(source.size(0))
    n_target = int(target.size(0))
    n_samples = n_source + n_target
    total = torch.cat([source, target], dim=0)  # (N, D)

    # Pairwise squared L2 distance via ||x - y||^2 = ||x||^2 + ||y||^2 - 2 <x, y>.
    # This avoids the (N, N, D) tensor that the naive expand-and-subtract approach
    # allocates and dominates memory for D >> 1.
    sq_norms = (total * total).sum(dim=1, keepdim=True)  # (N, 1)
    L2_distance = sq_norms + sq_norms.t() - 2.0 * (total @ total.t())  # (N, N)
    # Floating-point error can produce tiny negative values; clamp for numerical safety.
    L2_distance = L2_distance.clamp_min_(0)

    # Automatic bandwidth selection (median heuristic)
    bandwidth = torch.sum(L2_distance) / (n_samples**2 - n_samples)
    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))

    # Sum of Gaussian kernels at different scales (compute in-place to avoid
    # materializing kernel_num separate (N, N) tensors)
    kernels = torch.zeros_like(L2_distance)
    for i in range(kernel_num):
        bandwidth_i = bandwidth * (kernel_mul**i)
        kernels.add_(torch.exp(-L2_distance / bandwidth_i))

    # Extract kernel blocks and compute MMD^2 estimate
    XX = kernels[:n_source, :n_source]
    YY = kernels[n_source:, n_source:]
    XY = kernels[:n_source, n_source:]
    YX = kernels[n_source:, :n_source]

    loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)

    return loss.item()


def compute_w1_distance(
    generated: Tensor | np.ndarray,
    ground_truth: Tensor | np.ndarray,
) -> float:
    """
    W1 (Earth Mover's Distance) with Euclidean cost.

    Matches the protocol in Neklyudov et al. (2024)
    which uses ``ot.emd2`` on Euclidean distance with ``numItermax=1e7``.
    """
    if not HAS_POT:
        raise ImportError("POT library required for W1.")

    if isinstance(generated, Tensor):
        generated = generated.detach().cpu().numpy()
    if isinstance(ground_truth, Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()

    num_samples = min(len(generated), len(ground_truth))
    gen_np = generated[:num_samples].astype(np.float64)
    gt_np = ground_truth[:num_samples].astype(np.float64)

    a = np.ones(len(gen_np)) / len(gen_np)
    b = np.ones(len(gt_np)) / len(gt_np)

    M = ot.dist(gen_np, gt_np, metric="euclidean")
    return float(ot.emd2(a, b, M, numItermax=int(1e7)))


def compute_w2_distance(
    generated: Tensor | np.ndarray,
    ground_truth: Tensor | np.ndarray,
    return_plan: bool = False,
) -> float | tuple[float, np.ndarray]:
    """
    Compute W2 (Wasserstein-2) distance using POT library.

    Args:
        generated: Generated samples (n_samples, dim)
        ground_truth: Ground truth samples (m_samples, dim)
        return_plan: If True, also return the transport plan

    Returns:
        W2 distance (scalar), or (distance, plan) if return_plan=True
    """
    if not HAS_POT:
        raise ImportError(
            "POT library required for W2. Install with: pip install 'otpfm[experiments]' or (standalone) pip install pot"
        )

    # Convert to numpy and move to CPU
    if isinstance(generated, Tensor):
        generated = generated.detach().cpu().numpy()
    if isinstance(ground_truth, Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()

    num_samples = min(len(generated), len(ground_truth))

    gen_np = generated[:num_samples].astype(np.float64)
    gt_np = ground_truth[:num_samples].astype(np.float64)

    # Uniform weights
    n = len(gen_np)
    m = len(gt_np)
    a = np.ones(n) / n
    b = np.ones(m) / m

    # Compute cost matrix (squared Euclidean distance)
    M = ot.dist(gen_np, gt_np, metric="sqeuclidean")

    # Solve OT problem
    result = ot.emd2(a, b, M, return_matrix=return_plan)

    if return_plan:
        w2_dist, plan = result
        return np.sqrt(w2_dist), plan
    else:
        w2_dist = result
        return np.sqrt(w2_dist)


def compute_fgd(
    generated: Tensor | np.ndarray,
    ground_truth: Tensor | np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Compute Fréchet Gaussian Distance between two distributions.

    The Fréchet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is:
        d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2))

    Args:
        generated: Generated samples (n_samples, dim)
        ground_truth: Ground truth samples (m_samples, dim)
        eps: Small value added to diagonal for numerical stability (default: 1e-6)

    Returns:
        FGD value (scalar)
    """
    # Convert to numpy and move to CPU
    if isinstance(generated, Tensor):
        generated = generated.detach().cpu().numpy()
    if isinstance(ground_truth, Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()

    gen_np = generated.astype(np.float64)
    gt_np = ground_truth.astype(np.float64)

    # Compute mean and covariance for each distribution
    mu1 = np.mean(gen_np, axis=0)
    mu2 = np.mean(gt_np, axis=0)
    sigma1 = np.cov(gen_np, rowvar=False)
    sigma2 = np.cov(gt_np, rowvar=False)

    # Ensure 2D covariance matrices
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Covariance matrices have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        warnings.warn(
            f"FGD calculation produces singular product; adding {eps} to diagonal of cov estimates",
            RuntimeWarning,
            stacklevel=2,
        )
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not (
            np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3)
            or np.isclose(np.trace(covmean.imag) / np.trace(covmean.real), 0, atol=1e-3)
        ):
            im_trace = np.trace(covmean.imag)
            re_trace = np.trace(covmean.real)
            warnings.warn(
                f"Large imaginary components in covariance matrix while calculating "
                f"Fréchet distance Im: {im_trace:.2f} Re: {re_trace:.2f}",
                RuntimeWarning,
                stacklevel=2,
            )
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    fgd_squared = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return float(np.sqrt(max(fgd_squared, 0)))


def get_metric_columns(
    time_keys: list[str],
    metrics: list[str],
) -> list[str]:
    """
    Generate column names for metrics in standard format.

    Args:
        time_keys: Time keys (e.g., ["t1", "t3", "t2+t4"])
        metrics: Metric names (e.g., ["SWD", "MMD", "FGD", "W2"])

    Returns:
        List of column names like ["t1_SWD", "t1_MMD", ..., "t2+t4_W2"]
    """
    columns = []
    for t in time_keys:
        for m in metrics:
            columns.append(f"{t}_{m}")
    return columns


def create_metrics_dataframe(
    rows: list[dict[str, float]],
    time_keys: list[str],
    metrics: list[str],
    id_column: str = "Method",
    extra_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Create a DataFrame with standardized column ordering for metrics.

    Args:
        rows: List of row dictionaries, each containing id_column and metric values
        time_keys: Time keys to include
        metrics: Metrics to include
        id_column: Name of the identifier column (e.g., "Method" or "subdir_name")
        extra_columns: Additional columns to include at the end

    Returns:
        DataFrame with standardized column ordering
    """
    extra_columns = extra_columns or []

    # Build column order
    metric_columns = get_metric_columns(time_keys, metrics)
    all_columns = [id_column] + metric_columns + extra_columns

    # Create DataFrame and reorder columns
    df = pd.DataFrame(rows)

    # Only include columns that exist in the data
    existing_columns = [c for c in all_columns if c in df.columns]
    # Add any extra columns from data that aren't in our standard order
    for c in df.columns:
        if c not in existing_columns:
            existing_columns.append(c)

    df = df[existing_columns]

    return df
