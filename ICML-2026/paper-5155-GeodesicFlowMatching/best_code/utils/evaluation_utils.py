import torch
import torch.nn.functional as F
import numpy as np

def compute_cleanup_baseline(
    ssp_space,
    ssp_dim,               # ← included for compatibility
    snr: float | None,
    grid_resolution=64,
    method='sobol',
    num_trials=100,
    device="cpu"
):
    """
    Nearest-grid cleanup baseline.

    ``corrupted = snr * gt_ssp + (1 - snr) * z`` with ``z`` unit noise on the sphere.
    ``snr`` 1 = clean targets, 0 = pure noise. ``None`` is treated as ``0.0``.

    Returns dict with keys:
      mean_cosine, std_cosine, ci95_cosine,
      mean_rmse,   std_rmse,   ci95_rmse
    """
    from sklearn.utils import resample

    # 1) build grid
    grid_ssps, grid_pts = ssp_space.get_sample_pts_and_ssps(
        num_points_per_dim=grid_resolution,
        method=method
    )
    grid_ssps = torch.tensor(grid_ssps, device=device)   # (G, d)
    grid_pts  = grid_pts                                 # (G,2)

    # 2) pick true SSPs & pts
    gt_ssps, gt_pts = ssp_space.get_sample_pts_and_ssps(
        num_points_per_dim=num_trials,
        method='Rd'
    )
    gt_ssps = torch.tensor(gt_ssps, device=device)       # (T, d)
    gt_pts  = gt_pts                                     # (T,2)

    # 3) corrupt  (snr=1 → clean SSP; snr=0 → pure hypersphere noise)
    if snr is None:
        snr = 0.0
    snr = float(snr)
    z = torch.randn_like(gt_ssps)
    z = z / z.norm(dim=1, keepdim=True)
    corrupted = snr * gt_ssps + (1.0 - snr) * z          # (T, d)

    # 4) cleanup by nearest‐grid
    sims = corrupted @ grid_ssps.T                       # (T, G)
    idx = sims.argmax(dim=1)
    cleaned_ssps = grid_ssps[idx]                        # (T, d)
    cleaned_pts  = grid_pts[idx.cpu().numpy()]           # (T,2)

    # 5) cosine stats
    cos = (gt_ssps * cleaned_ssps).sum(dim=1) / (
          gt_ssps.norm(dim=1) * cleaned_ssps.norm(dim=1)
    )
    cos = cos.cpu().numpy()

    # 6) rmse stats
    diffs = cleaned_pts - gt_pts                         # (T,2)
    rmse = np.linalg.norm(diffs, axis=1)

    # helper for mean, std, bootstrap CI
    def stats(arr):
        m = arr.mean()
        std = arr.std(ddof=1)
        boot_means = [resample(arr).mean() for _ in range(100)]
        lower = np.percentile(boot_means, 2.5)
        upper = np.percentile(boot_means, 97.5)
        ci95 = (upper - lower) / 2
        return m, std, ci95

    cos_m, cos_std, cos_ci   = stats(cos)
    rmse_m, rmse_std, rmse_ci = stats(rmse)

    return {
        "mean_cosine":  cos_m,
        "std_cosine":   cos_std,
        "ci95_cosine":  cos_ci,
        "mean_rmse":    rmse_m,
        "std_rmse":     rmse_std,
        "ci95_rmse":    rmse_ci,
    }



def logmap_sphere(p, q):
    """
    Logarithm map on the hypersphere: tangent vector v at p pointing to q
    """
    dot = (p * q).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.acos(dot)
    v = q - dot * p
    v_norm = torch.norm(v, dim=-1, keepdim=True) + 1e-8
    return theta * v / v_norm


def expmap_sphere(p, v):
    """
    Exponential map on the hypersphere: move from p along tangent v
    """
    norm_v = torch.norm(v, dim=-1, keepdim=True) + 1e-8
    return torch.cos(norm_v) * p + torch.sin(norm_v) * (v / norm_v)

def compute_ssp_mean(
    ssp_space,
    num_samples: int = 1000,
    device: str = "cpu"
):
    """
    1) Sample num_samples points & their SSPs.
    2) Normalize each SSP to unit length.
    3) Euclidean mean: normalize the straight average.
    4) Return average COSINE_SIMILARITY of each mean against the SSPs.
    """
    ssps_np, pts_np = ssp_space.get_sample_pts_and_ssps(num_samples)
    ssps = torch.from_numpy(ssps_np).float().to(device)      # (M, D)

    mean_euc = ssps.mean(dim=0, keepdim=True)                # (1, D)
    mean_euc = F.normalize(mean_euc, p=2, dim=1).squeeze(0)  # (D,)

    # 5) average cosine similarities (now in [-1,1])
    sims_euc = F.cosine_similarity(ssps, mean_euc.unsqueeze(0), dim=1).mean().item()

    return sims_euc


def make_unitary(ssp):
    fssp = torch.fft.fft(ssp)
    eps = torch.tensor(1e-8, device=fssp.device, dtype=fssp.real.dtype)
    fssp = fssp / torch.maximum(torch.sqrt(fssp.real**2 + fssp.imag**2), eps)
    return torch.fft.ifft(fssp).real