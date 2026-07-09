import itertools

import torch
import torch.nn.functional as F
from scipy.stats import chi2


def compute_polynomial_terms(latent, degree):
    """
    Computes all unique monomials of a given degree from latent.
    latent: (n, p)
    degree: non-negative int
    Returns: (n, num_terms)
    """
    n, p = latent.shape

    if degree == 0:
        return torch.ones((n, 1), dtype=latent.dtype, device=latent.device)
    elif degree == 1:
        return latent

    # all combinations with replacement of variable indices
    combos = list(itertools.combinations_with_replacement(range(p), degree))

    # compute products for each combo
    out = []
    for combo in combos:
        term = latent[:, combo].prod(dim=1)  # multiply the selected columns
        out.append(term)

    out = torch.stack(out, dim=1)
    return out


def gaussian_entropy(x):
    # x: (batch, dim)
    cov = torch.cov(x.T)  # dim x dim
    entropy = 0.5 * torch.logdet(2 * torch.pi * torch.e * cov)
    return entropy


def knn_entropy(x, k=5, eps=1e-8):
    # x: (batch, dim)
    n, d = x.shape
    # pairwise distances
    dist = torch.cdist(x, x) + torch.eye(n, device=x.device) * 1e9
    # kth nearest neighbor distance
    eps_i, _ = dist.topk(k, largest=False)
    eps_i = eps_i[:, -1]

    # constants
    from math import gamma, pi

    c_d = (pi ** (0.5 * d)) / gamma(0.5 * d + 1)

    psi_n = torch.digamma(torch.tensor(n, dtype=torch.float, device=x.device))
    psi_k = torch.digamma(torch.tensor(k, dtype=torch.float, device=x.device))

    H = (
        psi_n
        - psi_k
        + torch.log(torch.tensor(c_d, device=x.device))
        + (d / n) * torch.sum(torch.log(eps_i + eps))
    )
    return H


## RBF kernel MMD and HSIC


def rbf_kernel(x, y, sigma):
    x_norm = (x**2).sum(dim=1).view(-1, 1)
    y_norm = (y**2).sum(dim=1).view(1, -1)
    dist = x_norm + y_norm - 2 * torch.mm(x, y.T)
    return torch.exp(-dist / (2 * sigma**2))


def compute_median_heuristic(x, y, safe_min=1e-6):
    """
    Computes the median pairwise distance.
    safe_min: 1e-6 (Numerical stability).
    """
    if y is not None:
        z = torch.cat([x, y], dim=0)
    else:
        z = x

    with torch.no_grad():
        dists = torch.pdist(z)
        median_dist = torch.median(dists).item()

    if median_dist < safe_min:
        return torch.tensor(1.0, device=x.device)

    return torch.tensor(median_dist, device=x.device)


def mmd_rbf(x, y, sigma=None, safe_min=1e-6, scales=[0.1, 1.0, 5.0]):
    # [0.1, 0.5, 1.0, 2.0, 5.0]
    """
    Computes MMD with a Multi-Scale RBF Kernel.

    Args:
        x, y: Input tensors
        sigma: Base bandwidth. If None, uses median heuristic.
        scales: List of multipliers. [0.1, 1.0, 5.0] covers
                fine-grained details and global structure.
    """
    if sigma is None:
        sigma = compute_median_heuristic(x, y, safe_min=safe_min)

    # Precompute squared Euclidean distances once
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x, y>
    x_sq = x.pow(2).sum(1, keepdim=True)
    y_sq = y.pow(2).sum(1, keepdim=True)

    xx_dist = x_sq + x_sq.t() - 2 * torch.mm(x, x.t())
    yy_dist = y_sq + y_sq.t() - 2 * torch.mm(y, y.t())
    xy_dist = x_sq + y_sq.t() - 2 * torch.mm(x, y.t())

    total_mmd = 0.0

    for scale in scales:
        current_sigma = sigma * scale
        gamma = 1.0 / (2 * current_sigma**2 + 1e-8)

        # Compute kernels for this scale
        K_xx = torch.exp(-gamma * xx_dist)
        K_yy = torch.exp(-gamma * yy_dist)
        K_xy = torch.exp(-gamma * xy_dist)

        # Unbiased MMD computation
        n = x.size(0)
        m = y.size(0)

        sum_K_xx = (K_xx.sum() - K_xx.diag().sum()) / (n * (n - 1))
        sum_K_yy = (K_yy.sum() - K_yy.diag().sum()) / (m * (m - 1))
        sum_K_xy = K_xy.mean()

        total_mmd += sum_K_xx + sum_K_yy - 2 * sum_K_xy

    return total_mmd, sigma


# def hsic_rbf(x, y, sigma_x=None, sigma_y=None, safe_min=1e-6):

#     n = x.shape[0]
#     H = torch.eye(n, device=x.device) - (1.0 / n) * torch.ones(
#         n, n, device=x.device
#     )

#     if sigma_x is None:
#         with torch.no_grad():
#             sigma_x = torch.median(torch.cdist(x, x)).item()
#             sigma_x = max(sigma_x, safe_min)
#     if sigma_y is None:
#         with torch.no_grad():
#             sigma_y = torch.median(torch.cdist(y, y)).item()
#             sigma_y = max(sigma_y, safe_min)

#     K = rbf_kernel(x, x, sigma_x)
#     L = rbf_kernel(y, y, sigma_y)

#     Kc = H @ K @ H
#     Lc = H @ L @ H

#     hsic = (Kc * Lc).sum() / (n - 1) ** 2  # or n^2 for biased
#     return hsic, sigma_x, sigma_y


def hsic_rbf(
    x,
    y,
    sigma_x=None,
    sigma_y=None,
    safe_min=1e-6,
    scales=[0.1, 1.0, 5.0],  # [0.1, 0.5, 1.0, 2.0, 5.0]
):
    """
    Computes HSIC using a Multi-Scale RBF Kernel.

    Args:
        x (n x dx): First variable
        y (n x dy): Second variable
        scales: List of multipliers for the bandwidth.
    """
    n = x.size(0)

    # 1. Compute Base Sigmas (independently for X and Y)
    # X and Y might have very different scales, so we need separate sigmas.
    if sigma_x is None:
        sigma_x = compute_median_heuristic(x, None, safe_min=safe_min)
    if sigma_y is None:
        sigma_y = compute_median_heuristic(y, None, safe_min=safe_min)

    # 2. Precompute Squared Distances
    # ||x - x'||^2
    x_sq = x.pow(2).sum(1, keepdim=True)
    dist_x = x_sq + x_sq.t() - 2 * torch.mm(x, x.t())

    y_sq = y.pow(2).sum(1, keepdim=True)
    dist_y = y_sq + y_sq.t() - 2 * torch.mm(y, y.t())

    # 3. Centering Matrix H
    # H = I - 1/n
    H = torch.eye(n, device=x.device) - (1.0 / n) * torch.ones(
        n, n, device=x.device
    )

    total_hsic = 0.0

    # 4. Multi-Scale Loop
    for scale in scales:
        # Scale the sigmas
        sx = sigma_x * scale
        sy = sigma_y * scale

        # Compute RBF Kernels
        gamma_x = 1.0 / (2 * sx**2 + 1e-8)
        gamma_y = 1.0 / (2 * sy**2 + 1e-8)

        K = torch.exp(-gamma_x * dist_x)
        L = torch.exp(-gamma_y * dist_y)

        # Center the kernels
        Kc = H @ K @ H
        Lc = H @ L @ H

        # Compute HSIC for this scale
        # Unbiased estimator often divides by (n-1)^2 or n(n-3) etc.
        # Using the V-statistic (biased) n^2 or (n-1)^2 is common in optimization
        # because we only care about minimization, not the exact p-value.
        hsic_scale = (Kc * Lc).sum() / ((n - 1) ** 2)

        total_hsic += hsic_scale

    return total_hsic, sigma_x, sigma_y


## Polynomial kernel MMD and HSIC


def mmd_poly(x, y, degree=2, c=1.0, unbiased=False, normalize=False):
    if normalize:
        x = torch.nn.functional.normalize(x, dim=1)
        y = torch.nn.functional.normalize(y, dim=1)

    K_xx = (x @ x.T + c) ** degree
    K_yy = (y @ y.T + c) ** degree
    K_xy = (x @ y.T + c) ** degree

    if unbiased:
        n, m = x.size(0), y.size(0)
        sum_K_xx = K_xx.sum() - K_xx.diag().sum()
        sum_K_yy = K_yy.sum() - K_yy.diag().sum()
        mmd_sq = (
            sum_K_xx / (n * (n - 1))
            + sum_K_yy / (m * (m - 1))
            - 2 * K_xy.mean()
        )
    else:
        mmd_sq = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()

    return mmd_sq


def hsic_poly(x, y, degree_x=2, degree_y=2, c=1.0, normalize=False):
    n = x.shape[0]
    assert y.shape[0] == n, "x and y must have the same number of samples"

    if normalize:
        x = torch.nn.functional.normalize(x, dim=1)
        y = torch.nn.functional.normalize(y, dim=1)

    K = (x @ x.T + c) ** degree_x
    L = (y @ y.T + c) ** degree_y

    H = torch.eye(n, device=x.device) - torch.ones(n, n, device=x.device) / n
    Kc = H @ K @ H
    Lc = H @ L @ H

    hsic = (Kc * Lc).sum() / (n**2)
    return hsic


def mmd_cosine(x, y):
    """
    Computes MMD using a Cosine Kernel (Linear kernel on normalized data).

    Args:
        x (n x d): Samples from distribution P
        y (m x d): Samples from distribution Q
    """
    # 1. L2 Normalize rows to unit length
    # This projects all points onto the unit sphere
    x_norm = F.normalize(x, p=2, dim=1)
    y_norm = F.normalize(y, p=2, dim=1)

    # 2. Compute Kernel Matrices (Cosine Similarity)
    # K_xx: similarity within x
    # K_yy: similarity within y
    # K_xy: similarity between x and y
    K_xx = torch.mm(x_norm, x_norm.t())
    K_yy = torch.mm(y_norm, y_norm.t())
    K_xy = torch.mm(x_norm, y_norm.t())

    # 3. Compute MMD^2 estimate
    # MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
    # We exclude the diagonal (self-similarity) for an unbiased-style estimate
    n = x.size(0)
    m = y.size(0)

    # Sum of off-diagonal elements divided by n(n-1)
    off_diag_x = K_xx.sum() - torch.diag(K_xx).sum()
    off_diag_y = K_yy.sum() - torch.diag(K_yy).sum()

    expected_kxx = off_diag_x / (n * (n - 1))
    expected_kyy = off_diag_y / (m * (m - 1))
    expected_kxy = K_xy.mean()

    mmd_sq = expected_kxx + expected_kyy - 2 * expected_kxy

    # Return max(0, mmd_sq) to handle tiny negative numerical noise
    return torch.relu(mmd_sq)


def hsic_cosine(x, y):
    n = x.size(0)

    # L2 Normalize rows to project onto the unit sphere
    x_norm = F.normalize(x, p=2, dim=1)
    y_norm = F.normalize(y, p=2, dim=1)

    # Linear kernel on normalized data = Cosine Kernel
    K = torch.mm(x_norm, x_norm.t())
    L = torch.mm(y_norm, y_norm.t())

    # Centering
    H = torch.eye(n, device=x.device) - (1.0 / n)
    Kc = H @ K @ H
    Lc = H @ L @ H

    return (Kc * Lc).sum() / ((n - 1) ** 2)


def wilks_lambda_test_torch(x, y, eps=1e-3):
    """
    Wilks' lambda test for independence between two multivariate normal variables using PyTorch.

    Parameters:
        x: (n x p) torch.Tensor
        y: (n x q) torch.Tensor

    Returns:
        stat: chi-square test statistic
        pval: p-value from chi-square distribution
    """
    n = x.shape[0]
    p = x.shape[1]
    q = y.shape[1]

    # Center the variables
    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)

    # Compute full covariance matrix
    Z = torch.cat([x, y], dim=1)  # shape: (n, p+q)
    # Compute covariance matrices
    Sigma = torch.cov(Z.T)
    Sigma_xx = torch.cov(x.T)
    Sigma_yy = torch.cov(y.T)

    # Log-determinants (more stable)
    log_det_Sigma = torch.logdet(
        Sigma + eps * torch.eye(Z.shape[1], device=Z.device)
    )
    log_det_Sigma_xx = torch.logdet(
        Sigma_xx + eps * torch.eye(p, device=Z.device)
    )
    log_det_Sigma_yy = torch.logdet(
        Sigma_yy + eps * torch.eye(q, device=Z.device)
    )

    # Chi-square statistic
    m = n - 1 - (p + q + 1) / 2
    stat = -m * (log_det_Sigma - log_det_Sigma_xx - log_det_Sigma_yy)

    # Chi-squared approximation
    df = p * q
    pval = 1 - chi2.cdf(stat.item(), df)

    return stat.item(), pval


# def cross_cov_loss(x, y, penalty_strength=1.0):
#     """
#     Penalizes the cross-covariance between x and y to encourage independence.

#     Args:
#         x (torch.Tensor): Shape (batch_size, dim_x)
#         y (torch.Tensor): Shape (batch_size, dim_y)
#         penalty_strength (float): Scaling factor for the penalty.

#     Returns:
#         loss (torch.Tensor): Scalar penalty term.
#     """
#     # Stack x and y into Z = [x, y]
#     Z = torch.cat([x, y], dim=1)  # Shape: (batch_size, dim_x + dim_y)

#     # Compute covariance matrix (normalized by batch_size - 1)
#     cov_Z = torch.cov(Z.T)  # Shape: (dim_x + dim_y, dim_x + dim_y)

#     # Split into blocks
#     dim_x = x.shape[1]

#     # Cross-covariance block (off-diagonal)
#     cross_cov = cov_Z[:dim_x, dim_x:]  # Shape: (dim_x, dim_y)

#     # Penalty: Frobenius norm squared (sum of squared elements)
#     penalty = torch.sum(cross_cov**2) / torch.sum(cov_Z**2)

#     return penalty_strength * penalty


def orthogonality(v, w):
    """
    Computes the squared Frobenius norm of the cross-covariance matrix.
    v: tensor of shape (n, q)
    w: tensor of shape (n, p)
    """
    batch_size = v.shape[0]

    # 1. Center the features (Critical for "Independence")
    # If you don't center, you are forcing them to be orthogonal vectors,
    # not just uncorrelated variables.
    v_centered = v - v.mean(dim=0, keepdim=True)
    w_centered = w - w.mean(dim=0, keepdim=True)

    # 2. Compute the unnormalized covariance matrix (q x p)
    # Transpose v: (q x n) @ (n x p) -> (q x p)
    cov_matrix = torch.matmul(v_centered.t(), w_centered)

    # 3. Compute Loss: Squared Frobenius Norm
    # We square every element and sum them up.
    # Normalizing by batch_size^2 ensures the loss doesn't explode with larger batches.
    loss = torch.sum(cov_matrix**2) / (batch_size * batch_size)

    return loss


def distance_correlation(v, w):
    """
    Computes Distance Correlation (dCor) using torch.cdist for stability.
    Complexity: O(N^2). WARNING: Do not use on large batches (N > 2000).
    """
    # 1. Shape Check
    # Ensure inputs are at least 2D (batch, dim)
    if v.dim() == 1:
        v = v.view(-1, 1)
    if w.dim() == 1:
        w = w.view(-1, 1)

    batch_size = v.size(0)
    if batch_size > 2000:
        print(
            f"Warning: dCor batch size is {batch_size}. This consumes O(N^2) memory!"
        )

    # 2. Compute Pairwise Distance Matrices using cdist (Optimized C++)
    # p=2 is Euclidean distance
    D_v = torch.cdist(v, v, p=2)
    D_w = torch.cdist(w, w, p=2)

    # 3. Double Centering
    # A_ij = D_ij - mean(row_i) - mean(col_j) + mean(grand)

    # We use keepdim=True to allow broadcasting
    dv_mean_row = D_v.mean(dim=1, keepdim=True)
    dv_mean_col = D_v.mean(dim=0, keepdim=True)
    dv_mean_all = D_v.mean()

    dw_mean_row = D_w.mean(dim=1, keepdim=True)
    dw_mean_col = D_w.mean(dim=0, keepdim=True)
    dw_mean_all = D_w.mean()

    A_v = D_v - dv_mean_row - dv_mean_col + dv_mean_all
    A_w = D_w - dw_mean_row - dw_mean_col + dw_mean_all

    # 4. Compute Distance Covariance
    # The mean of the element-wise product of the double-centered matrices
    dcov2 = torch.mean(A_v * A_w)

    # 5. Compute Distance Variances
    dvar2_v = torch.mean(A_v * A_v)
    dvar2_w = torch.mean(A_w * A_w)

    # 6. Compute Distance Correlation
    # Add epsilon to denominators to prevent division by zero / NaN gradients
    dcor = torch.sqrt(dcov2 + 1e-8) / torch.sqrt(
        torch.sqrt(dvar2_v * dvar2_w) + 1e-8
    )

    return dcor
