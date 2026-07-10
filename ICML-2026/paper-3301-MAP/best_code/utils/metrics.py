import unittest

import numpy as np
import torch
from scipy.stats import entropy
from sklearn.metrics import pairwise_distances

import numpy as np

from sklearn.neighbors import NearestNeighbors

def coverage(real, fake, k=5):
    """
    real, fake: (N, D) arrays or tensors
    """
    real = np.asarray(real); fake = np.asarray(fake)
    nn = NearestNeighbors(n_neighbors=k).fit(real)
    dists, _ = nn.kneighbors(real)   # shape (N, k)
    radii = dists[:, -1]             # kth NN distance for each real
    nn_fake = NearestNeighbors(n_neighbors=1).fit(fake)
    d_fake, _ = nn_fake.kneighbors(real)  # dist to nearest fake
    covered = (d_fake[:, 0] <= radii).mean()
    return covered


def filter_valid_samples(tensor, max_abs_value=1e6):
    """
    Remove rows in tensor that contain any NaN, Inf, or abnormally large values.

    Args:
        tensor (torch.Tensor): shape (batch_size, num_features)
        max_abs_value (float): threshold for absolute value considered valid.

    Returns:
        torch.Tensor: shape (num_valid, num_features)
    """
    finite_mask = torch.isfinite(tensor).all(dim=1)
    within_bound_mask = (tensor.abs() < max_abs_value).all(dim=1)
    valid_mask = finite_mask & within_bound_mask
    return tensor[valid_mask]

# Ensure all samples are tensors of shape (num_samples, num_features)
def ensure_tensor_2d(samples, D):
    t = torch.tensor(samples) if not torch.is_tensor(samples) else samples
    return t.view(-1, D)

def gaussian_kernel(x, y, bandwidth=0.1):
    """
    Compute the Gaussian kernel between two points.
    """
    dist = torch.norm(x - y, dim=1)
    return torch.exp(-(dist**2) / (2 * bandwidth**2))


def kde(torch_points, bandwidth=0.1):
    """
    Kernel Density Estimation (KDE) using Gaussian kernels.
    """
    n = torch_points.shape[0]
    kde_values = torch.zeros(n)

    for i in range(n):
        kde_values[i] = torch.mean(
            gaussian_kernel(torch_points[i], torch_points, bandwidth)
        )

    return kde_values / torch.sum(kde_values)

def compute_mmd(X, Y, sigma=1.0):
    """
    Computes Maximum Mean Discrepancy (MMD) between two datasets X and Y using a Gaussian kernel.

    Args:
        X (np.ndarray or torch.Tensor): True dataset, shape (num_samples, num_features)
        Y (np.ndarray or torch.Tensor): Generated dataset, shape (num_samples, num_features)
        sigma (float): Kernel bandwidth for the Gaussian kernel.

    Returns:
        float: The MMD value.
    """
    if torch.is_tensor(X): X = X.cpu().numpy()
    if torch.is_tensor(Y): Y = Y.cpu().numpy()
    XX = pairwise_distances(X, X, metric='euclidean')
    YY = pairwise_distances(Y, Y, metric='euclidean')
    XY = pairwise_distances(X, Y, metric='euclidean')
    Kxx = np.exp(-XX ** 2 / (2 * sigma ** 2))
    Kyy = np.exp(-YY ** 2 / (2 * sigma ** 2))
    Kxy = np.exp(-XY ** 2 / (2 * sigma ** 2))
    mmd = Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()
    return mmd

def _pairwise_sq_dists(x, y=None):
    # compute in double for stability, then cast back
    xd = x.to(dtype=torch.float64)
    if y is None:
        yd = xd
    else:
        yd = y.to(dtype=torch.float64)

    x2 = (xd * xd).sum(dim=1, keepdim=True)            # (n,1)
    y2 = (yd * yd).sum(dim=1, keepdim=True).t()        # (1,m)
    d2 = x2 + y2 - 2.0 * (xd @ yd.t())                 # (n,m)

    d2 = torch.nan_to_num(d2, nan=0.0, posinf=1e38, neginf=0.0)
    d2.clamp_min_(0.0)
    return d2.to(dtype=x.dtype)  # match original dtype

def _kernel_from_d2(d2, kernel="rbf", bandwidths=None):
    if kernel == "multiscale":
        if bandwidths is None: bandwidths = [0.2, 0.5, 0.9, 1.3]
        K = torch.zeros_like(d2)  # <<< important
        for a in bandwidths:
            a2 = a * a
            K = K + a2 * (a2 + d2).reciprocal()
        return torch.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)
    elif kernel == "rbf":
        if bandwidths is None: bandwidths = [10.0, 15.0, 20.0, 50.0]  # σ^2 values
        K = torch.zeros_like(d2)  # <<< important
        for a in bandwidths:
            K = K + torch.exp(-0.5 * d2 / a)
        return torch.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

def MMD(x, y, kernel="rbf", bandwidths=None, unbiased=False):
    # sanitize inputs but *recover* instead of bailing
    x = torch.nan_to_num(x, nan=0.0, posinf=1e19, neginf=-1e19)
    y = torch.nan_to_num(y, nan=0.0, posinf=1e19, neginf=-1e19)

    n = x.shape[0]; m = y.shape[0]
    if n == 0 or m == 0:
        # return 0 (or NaN if you prefer) when one set is empty
        return torch.tensor(0.0, device=(x.device if x.numel() else 'cpu'), dtype=x.dtype)

    dxx = _pairwise_sq_dists(x)        # (n,n)
    dyy = _pairwise_sq_dists(y)        # (m,m)
    dxy = _pairwise_sq_dists(x, y)     # (n,m)

    Kxx = _kernel_from_d2(dxx, kernel, bandwidths)
    Kyy = _kernel_from_d2(dyy, kernel, bandwidths)
    Kxy = _kernel_from_d2(dxy, kernel, bandwidths)

    if unbiased and n > 1 and m > 1:
        # remove diagonals
        Kxx = Kxx - torch.diag(torch.diag(Kxx))
        Kyy = Kyy - torch.diag(torch.diag(Kyy))
        denom_x = float(n * (n - 1))
        denom_y = float(m * (m - 1))
        mmd2 = (Kxx.sum() / denom_x) + (Kyy.sum() / denom_y) - 2.0 * Kxy.mean()
    else:
        mmd2 = Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()

    # last guard
    mmd2 = torch.nan_to_num(mmd2, nan=0.0, posinf=0.0, neginf=0.0)
    return mmd2

def compute_2d_hist(samples, bins=50, range=None):
    H, xedges, yedges = np.histogram2d(samples[:,0], samples[:,1], bins=bins, range=range)
    H = H.astype(np.float64)
    H /= H.sum()  # Normalize to make it a probability distribution
    return H, xedges, yedges

def kl_divergence(P, Q, eps=1e-12):
    P = P.flatten() + eps
    Q = Q.flatten() + eps
    return np.sum(P * np.log(P / Q))

def js_divergence(P, Q, eps=1e-12):
    P = P.flatten() + eps
    Q = Q.flatten() + eps
    M = 0.5 * (P + Q)
    return 0.5 * (np.sum(P * np.log(P / M)) + np.sum(Q * np.log(Q / M)))

def jsd_histogram_2d(samples_a, samples_b, bins=30, range=None, grid_edges=None, eps=1e-12, log_base=np.e):
    """
    Compute the histogram-based Jensen-Shannon Divergence (JSD) between two sets of 2D samples.
    
    Args:
        samples_a: torch.Tensor or np.ndarray of shape (N, 2)
        samples_b: torch.Tensor or np.ndarray of shape (M, 2)
        bins: int, number of bins per dimension (default: 30)
        range: [[xmin,xmax],[ymin,ymax]] or None. If None, inferred from both sets.
        eps: float, small value to avoid log(0)
    
    Returns:
        float: JSD value
    """
    # Convert torch.Tensor to numpy, if needed
    if hasattr(samples_a, 'numpy'):
        samples_a = samples_a.numpy()
    if hasattr(samples_b, 'numpy'):
        samples_b = samples_b.numpy()
    samples_a = np.asarray(samples_a)
    samples_b = np.asarray(samples_b)

    # Handle empty inputs
    if samples_a.size == 0 and samples_b.size == 0:
        return 0.0
    if samples_a.size == 0 or samples_b.size == 0:
        return float('nan')

    try:
        samples_a = samples_a.reshape(-1, 2)
        samples_b = samples_b.reshape(-1, 2)
    except Exception:
        raise ValueError("samples_a and samples_b must be reshaped to (-1,2)")

    # If explicit grid edges are provided, use them (bins should be ignored in that case).
    if grid_edges is not None:
        xedges, yedges = grid_edges
        H_a, xedges_out, yedges_out = np.histogram2d(samples_a[:, 0], samples_a[:, 1], bins=[xedges, yedges])
        H_b, _, _ = np.histogram2d(samples_b[:, 0], samples_b[:, 1], bins=[xedges, yedges])
    else:
        # Infer range if not provided and guard against degenerate spans
        if range is None:
            all_samples = np.vstack([samples_a, samples_b])
            xmin, xmax = all_samples[:, 0].min(), all_samples[:, 0].max()
            ymin, ymax = all_samples[:, 1].min(), all_samples[:, 1].max()

            def _expand(minv, maxv):
                if maxv <= minv:
                    delta = 1e-6 * (abs(minv) + 1.0)
                    return minv - delta, maxv + delta
                return minv, maxv

            xmin, xmax = _expand(xmin, xmax)
            ymin, ymax = _expand(ymin, ymax)
            range = [[xmin, xmax], [ymin, ymax]]

        H_a, _, _ = np.histogram2d(samples_a[:, 0], samples_a[:, 1], bins=bins, range=range)
        H_b, _, _ = np.histogram2d(samples_b[:, 0], samples_b[:, 1], bins=bins, range=range)

    # Add small smoothing to avoid zeros and normalize
    P = H_a.flatten().astype(np.float64) + eps
    Q = H_b.flatten().astype(np.float64) + eps
    P = P / P.sum()
    Q = Q / Q.sum()

    # JSD calculation (allow choice of log base)
    M = 0.5 * (P + Q)
    if log_base == 2:
        logfn = np.log2
    else:
        logfn = np.log
    jsd = 0.5 * (np.sum(P * logfn(P / M)) + np.sum(Q * logfn(Q / M)))
    # If user requested a non-standard base, convert (we only optimized for 2 or e)
    if log_base not in (2, np.e):
        jsd = jsd / np.log(log_base)
    return float(jsd)


def tvd_histogram_2d(samples_a, samples_b, bins=30, range=None, grid_edges=None, eps=1e-12):
    """
    Compute the histogram-based Total Variation Distance (TVD) between two sets of 2D samples.

    TVD = 0.5 * sum |P - Q| where P and Q are histogram probability vectors.
    Arguments mirror `jsd_histogram_2d`.
    """
    # Convert torch.Tensor to numpy, if needed
    if hasattr(samples_a, 'numpy'):
        samples_a = samples_a.numpy()
    if hasattr(samples_b, 'numpy'):
        samples_b = samples_b.numpy()
    samples_a = np.asarray(samples_a)
    samples_b = np.asarray(samples_b)

    # Handle empty inputs
    if samples_a.size == 0 and samples_b.size == 0:
        return 0.0
    if samples_a.size == 0 or samples_b.size == 0:
        return float('nan')

    try:
        samples_a = samples_a.reshape(-1, 2)
        samples_b = samples_b.reshape(-1, 2)
    except Exception:
        raise ValueError("samples_a and samples_b must be reshaped to (-1,2)")

    # Use explicit grid edges if supplied
    if grid_edges is not None:
        xedges, yedges = grid_edges
        H_a, _, _ = np.histogram2d(samples_a[:, 0], samples_a[:, 1], bins=[xedges, yedges])
        H_b, _, _ = np.histogram2d(samples_b[:, 0], samples_b[:, 1], bins=[xedges, yedges])
    else:
        # Infer range if not provided and guard against degenerate spans
        if range is None:
            all_samples = np.vstack([samples_a, samples_b])
            xmin, xmax = all_samples[:, 0].min(), all_samples[:, 0].max()
            ymin, ymax = all_samples[:, 1].min(), all_samples[:, 1].max()

            def _expand(minv, maxv):
                if maxv <= minv:
                    delta = 1e-6 * (abs(minv) + 1.0)
                    return minv - delta, maxv + delta
                return minv, maxv

            xmin, xmax = _expand(xmin, xmax)
            ymin, ymax = _expand(ymin, ymax)
            range = [[xmin, xmax], [ymin, ymax]]

        H_a, _, _ = np.histogram2d(samples_a[:, 0], samples_a[:, 1], bins=bins, range=range)
        H_b, _, _ = np.histogram2d(samples_b[:, 0], samples_b[:, 1], bins=bins, range=range)

    # Add small smoothing to avoid zeros and normalize
    P = H_a.flatten().astype(np.float64) + eps
    Q = H_b.flatten().astype(np.float64) + eps
    P = P / P.sum()
    Q = Q / Q.sum()

    tvd = 0.5 * np.sum(np.abs(P - Q))
    return float(tvd)


def _infer_nd_range(samples_a, samples_b, dims):
    """
    Infer per-dimension ranges for ND histogramming, guarding against degenerate spans.
    Returns list of (min,max) for each dim.
    """
    all_samples = np.vstack([samples_a.reshape(-1, dims), samples_b.reshape(-1, dims)])
    ranges = []
    for d in range(dims):
        mn = all_samples[:, d].min()
        mx = all_samples[:, d].max()
        if mx <= mn:
            delta = 1e-6 * (abs(mn) + 1.0)
            mn, mx = mn - delta, mx + delta
        ranges.append((mn, mx))
    return ranges


def jsd_histogram_nd(samples_a, samples_b, bins=20, range=None, eps=1e-12, log_base=np.e):
    """
    Compute histogram-based Jensen-Shannon Divergence (JSD) between two sets of ND samples.

    Uses `np.histogramdd` to compute joint histograms across dimensions.
    """
    # Convert torch.Tensor to numpy, if needed
    if hasattr(samples_a, 'numpy'):
        samples_a = samples_a.numpy()
    if hasattr(samples_b, 'numpy'):
        samples_b = samples_b.numpy()
    samples_a = np.asarray(samples_a)
    samples_b = np.asarray(samples_b)

    # Handle empty inputs
    if samples_a.size == 0 and samples_b.size == 0:
        return 0.0
    if samples_a.size == 0 or samples_b.size == 0:
        return float('nan')

    # Ensure 2D arrays of shape (N, D)
    try:
        samples_a = samples_a.reshape(-1, samples_a.shape[-1])
        samples_b = samples_b.reshape(-1, samples_b.shape[-1])
    except Exception:
        raise ValueError("samples must be shaped (N, D)")

    D = samples_a.shape[1]

    if range is None:
        ranges = _infer_nd_range(samples_a, samples_b, D)
    else:
        # Expect range as list of [min,max] per dim
        ranges = [(r[0], r[1]) for r in range]

    # bins can be int or list-like; convert to sequence per-dim
    if np.isscalar(bins):
        bins_spec = [bins] * D
    else:
        bins_spec = list(bins)

    H_a, edges = np.histogramdd(samples_a, bins=bins_spec, range=ranges)
    H_b, _ = np.histogramdd(samples_b, bins=bins_spec, range=ranges)

    P = H_a.flatten().astype(np.float64) + eps
    Q = H_b.flatten().astype(np.float64) + eps
    P = P / P.sum()
    Q = Q / Q.sum()

    M = 0.5 * (P + Q)
    if log_base == 2:
        logfn = np.log2
    else:
        logfn = np.log
    jsd = 0.5 * (np.sum(P * logfn(P / M)) + np.sum(Q * logfn(Q / M)))
    if log_base not in (2, np.e):
        jsd = jsd / np.log(log_base)
    return float(jsd)


def tvd_histogram_nd(samples_a, samples_b, bins=20, range=None, eps=1e-12):
    """
    Compute histogram-based Total Variation Distance (TVD) between two sets of ND samples.

    Returns 0.5 * sum |P - Q| where P,Q are flattened histograms.
    """
    if hasattr(samples_a, 'numpy'):
        samples_a = samples_a.numpy()
    if hasattr(samples_b, 'numpy'):
        samples_b = samples_b.numpy()
    samples_a = np.asarray(samples_a)
    samples_b = np.asarray(samples_b)

    # Handle empty inputs
    if samples_a.size == 0 and samples_b.size == 0:
        return 0.0
    if samples_a.size == 0 or samples_b.size == 0:
        return float('nan')

    try:
        samples_a = samples_a.reshape(-1, samples_a.shape[-1])
        samples_b = samples_b.reshape(-1, samples_b.shape[-1])
    except Exception:
        raise ValueError("samples must be shaped (N, D)")

    D = samples_a.shape[1]
    if range is None:
        ranges = _infer_nd_range(samples_a, samples_b, D)
    else:
        ranges = [(r[0], r[1]) for r in range]

    if np.isscalar(bins):
        bins_spec = [bins] * D
    else:
        bins_spec = list(bins)

    H_a, _ = np.histogramdd(samples_a, bins=bins_spec, range=ranges)
    H_b, _ = np.histogramdd(samples_b, bins=bins_spec, range=ranges)

    P = H_a.flatten().astype(np.float64) + eps
    Q = H_b.flatten().astype(np.float64) + eps
    P = P / P.sum()
    Q = Q / Q.sum()

    tvd = 0.5 * np.sum(np.abs(P - Q))
    return float(tvd)


def compute_jsd_3d(X, Y, bins=20, range=None):
    """Wrapper: compute JSD using 3D histogramming. Accepts torch tensors or numpy arrays."""
    return jsd_histogram_nd(X, Y, bins=bins, range=range)


def compute_tvd_3d(X, Y, bins=20, range=None):
    """Wrapper: compute TVD using 3D histogramming. Accepts torch tensors or numpy arrays."""
    return tvd_histogram_nd(X, Y, bins=bins, range=range)


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.pc1 = torch.rand((100, 3))
        self.pc2 = torch.rand((100, 3))

    def test_gaussian_kernel(self):
        x = torch.rand((10, 3))
        y = torch.rand((10, 3))
        result = gaussian_kernel(x, y)
        self.assertEqual(result.shape, (10,))
        self.assertTrue(torch.all(result >= 0))

    def test_kde(self):
        result = kde(self.pc1)
        self.assertEqual(result.shape, (100,))
        self.assertAlmostEqual(torch.sum(result).item(), 1.0, places=5)

def compute_torsion_angles(samples):
    if torch.is_tensor(samples):
        samples = samples.detach().cpu().numpy()
    else:
        samples = np.asarray(samples)

    def torsion(p0, p1, p2, p3):
        b0 = p1 - p0
        b1 = p2 - p1
        b2 = p3 - p2
        b1 = b1 / (np.linalg.norm(b1, axis=-1, keepdims=True) + 1e-12)

        v = b0 - (b0 * b1).sum(axis=-1, keepdims=True) * b1
        w = b2 - (b2 * b1).sum(axis=-1, keepdims=True) * b1

        x = (v * w).sum(axis=-1)
        y = (np.cross(b1, v) * w).sum(axis=-1)
        return np.arctan2(y, x)

    N, L, A, C = samples.shape
    assert A >= 3 and C == 3

    phi = []
    psi = []

    for sample in samples:
        # phi(i): C(i-1)-N(i)-CA(i)-C(i), i = 1..L-1
        for i in range(1, L):
            phi.append(torsion(sample[i-1, 2], sample[i, 0], sample[i, 1], sample[i, 2]))
        # psi(i): N(i)-CA(i)-C(i)-N(i+1), i = 0..L-2
        for i in range(0, L-1):
            psi.append(torsion(sample[i, 0], sample[i, 1], sample[i, 2], sample[i+1, 0]))

    phi = np.asarray(phi).reshape(-1)
    psi = np.asarray(psi).reshape(-1)
    return phi, psi

def torsion_angle_KL(samples_generated, samples_true, bins=40, eps=1e-8):
    phi_g, psi_g = compute_torsion_angles(samples_generated)
    phi_t, psi_t = compute_torsion_angles(samples_true)

    def hist(p):
        h, _ = np.histogram(p, bins=bins, range=(-np.pi, np.pi), density=False)
        h = h.astype(np.float64) + eps
        h /= h.sum()
        return h

    KL_phi = entropy(hist(phi_t), hist(phi_g))
    KL_psi = entropy(hist(psi_t), hist(psi_g))
    return float(KL_phi), float(KL_psi)


# Common van der Waals radii for backbone atoms (angstroms)
VDW_RADII = {
    'N': 1.55,
    'CA': 1.70,
    'C': 1.70,
    'O': 1.52,
    # add others if sidechains included
}

if __name__ == "__main__":
    unittest.main()