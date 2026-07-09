# Implement the  Maximum-Mean-Discrepancy (MMD)
# Based on https://github.com/DenisBless/variational_sampling_methods/blob/main/algorithms/common/ipm_eval/mmd_median.py
# and https://pytorch.org/ignite/generated/ignite.metrics.MaximumMeanDiscrepancy.html

# Libraries
import torch


def compute_pairwise_sq_distance_same(A):
    aa = torch.mm(A, A.t())
    ra = aa.diag().unsqueeze(0).expand_as(aa)
    return ra.t() + ra - 2.0 * aa, ra


def compute_pairwise_sq_distance_diff(A, ra, B, rb):
    cc = torch.mm(A, B.t())
    return ra.t() + rb - 2.0 * cc


def gauss_kernel(pairwise_sq_matrix, bandwidth_sq):
    d = pairwise_sq_matrix / bandwidth_sq
    return torch.exp(-d / 2)


def compute_mmd(X, Y):
    """Compute the Maximum-Mean-Discrepancy distance using the median bandwidth heuristic

    Args:
        * X (torch.Tensor (batch_size, dim)): Samples from the first distribution
        * Y (torch.Tensor (batch_size, dim)): Samples from the second distribution

    Returns:
        * mmd (float): MMD distance
    """

    # Assertions
    m = X.shape[0]
    n = Y.shape[0]
    assert n >= 2 and m >= 2
    assert n == m

    # Compute the distances
    d_sq_XX, rx = compute_pairwise_sq_distance_same(X)
    d_sq_YY, ry = compute_pairwise_sq_distance_same(Y)
    d_sq_XY = compute_pairwise_sq_distance_diff(X, rx, Y, ry)

    # Compute the bandwidth
    row_indices, col_indices = torch.triu_indices(n, n, offset=1, device=X.device)
    bandwidth_sq = torch.median(torch.concat([
        d_sq_XX[row_indices, col_indices].flatten(), d_sq_YY[row_indices, col_indices].flatten(), d_sq_XY.flatten()
    ], dim=0))

    # Compute the kernels
    K_XY = gauss_kernel(d_sq_XY, bandwidth_sq)
    K_XX = gauss_kernel(d_sq_XX, bandwidth_sq)
    K_YY = gauss_kernel(d_sq_YY, bandwidth_sq)

    # Compute the MMD
    mmd = (K_XX.sum() - n) / (n * (n - 1))
    mmd += (K_YY.sum() - m) / (m * (m - 1))
    mmd -= 2. * K_XY.mean()
    mmd = torch.sqrt(torch.maximum(torch.tensor(1e-20), mmd))  # Ensure non-negative value
    return mmd.item()
