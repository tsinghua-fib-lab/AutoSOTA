import random

import numpy as np
import torch


def set_random_seed(seed, deterministic=False):
    """Set all RNGs used by the experiment scripts."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def pairwise_sq_dists(x, y):
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
    dists = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
    dists[dists < 0] = 0
    return dists


def split_selected_bandwidths(sigma):
    if isinstance(sigma, dict):
        return sigma["mmd"], sigma["nammd"]
    if isinstance(sigma, (tuple, list)):
        if len(sigma) != 2:
            raise ValueError("Expected two bandwidths: (sigma_mmd, sigma_nammd).")
        return sigma[0], sigma[1]
    return sigma, sigma


def _scalar_tensor(value, device, dtype):
    if torch.is_tensor(value):
        return value.detach().to(device=device, dtype=dtype).reshape(())
    return torch.tensor(float(value), device=device, dtype=dtype)


def _reference_subset(S, N1, max_reference_samples, seed):
    n_ref = int(N1)
    if max_reference_samples is not None:
        n_ref = min(n_ref, int(max_reference_samples))
    rng = np.random.default_rng(int(seed))
    ind_x = rng.choice(int(N1), n_ref, replace=False)
    ind_y = rng.choice(int(S.shape[0] - N1), n_ref, replace=False) + int(N1)
    ind_x = torch.as_tensor(ind_x, dtype=torch.long, device=S.device)
    ind_y = torch.as_tensor(ind_y, dtype=torch.long, device=S.device)
    return S.index_select(0, ind_x), S.index_select(0, ind_y)


def gaussian_bandwidth_grid(X, Y, num_bandwidths=25, multipliers=None):
    if multipliers is None:
        multipliers = np.logspace(-1.0, 1.0, int(num_bandwidths))
    multipliers = torch.as_tensor(multipliers, device=X.device, dtype=X.dtype)

    with torch.no_grad():
        Z = torch.cat((X, Y), dim=0)
        dists = pairwise_sq_dists(Z, Z)
        positive = dists[dists > 0]
        if positive.numel() == 0:
            median_sigma = torch.tensor(1.0, device=X.device, dtype=X.dtype)
        else:
            median_sigma = torch.sqrt(torch.median(positive).clamp_min(1e-12))
        grid = (median_sigma * multipliers).clamp_min(1e-6)
    return grid


def gaussian_mmd_nammd_studentized(X, Y, sigma, eps=1e-12):
    n = X.shape[0]
    if n != Y.shape[0]:
        raise ValueError("X and Y must have the same number of reference samples.")
    if n < 4:
        raise ValueError("At least four samples per side are required.")

    Dxx = pairwise_sq_dists(X, X)
    Dyy = pairwise_sq_dists(Y, Y)
    Dxy = pairwise_sq_dists(X, Y)
    sigma = _scalar_tensor(sigma, X.device, X.dtype).clamp_min(1e-6)

    Kx = torch.exp(-Dxx / sigma**2)
    Ky = torch.exp(-Dyy / sigma**2)
    Kxy = torch.exp(-Dxy / sigma**2)
    mask = ~torch.eye(n, dtype=torch.bool, device=X.device)

    xx = Kx[mask].mean()
    yy = Ky[mask].mean()
    xy = Kxy[mask].mean()
    mmd = xx - 2.0 * xy + yy
    reg = (4.0 - xx - yy).clamp_min(eps)
    nammd = mmd / reg

    h_matrix = Kx + Ky - Kxy - Kxy.t()
    d_matrix = 4.0 - Kx - Ky
    h_rows = h_matrix[mask].reshape(n, n - 1).mean(dim=1)
    d_rows = d_matrix[mask].reshape(n, n - 1).mean(dim=1)

    se_mmd = h_rows.std(unbiased=True).clamp_min(eps) / np.sqrt(n)
    nammd_influence = (h_rows - nammd * d_rows) / reg
    se_nammd = nammd_influence.std(unbiased=True).clamp_min(eps) / np.sqrt(n)

    return {
        "mmd": mmd,
        "nammd": nammd,
        "reg": reg,
        "mmd_score": mmd / se_mmd,
        "nammd_score": nammd / se_nammd,
        "se_mmd": se_mmd,
        "se_nammd": se_nammd,
    }


def select_gaussian_bandwidths_reference(
    S,
    N1,
    seed=1102,
    max_reference_samples=1000,
    num_bandwidths=25,
    multipliers=None,
    verbose=True,
):
    X_ref, Y_ref = _reference_subset(S.detach(), N1, max_reference_samples, seed)
    grid = gaussian_bandwidth_grid(
        X_ref, Y_ref, num_bandwidths=num_bandwidths, multipliers=multipliers
    )

    best_mmd = None
    best_nammd = None
    diagnostics = []
    with torch.no_grad():
        for sigma in grid:
            stats = gaussian_mmd_nammd_studentized(X_ref, Y_ref, sigma)
            row = {
                "sigma": float(sigma.detach().cpu()),
                "mmd": float(stats["mmd"].detach().cpu()),
                "nammd": float(stats["nammd"].detach().cpu()),
                "mmd_score": float(stats["mmd_score"].detach().cpu()),
                "nammd_score": float(stats["nammd_score"].detach().cpu()),
            }
            diagnostics.append(row)
            if best_mmd is None or row["mmd_score"] > best_mmd["mmd_score"]:
                best_mmd = row
            if best_nammd is None or row["nammd_score"] > best_nammd["nammd_score"]:
                best_nammd = row

    sigma_mmd = _scalar_tensor(best_mmd["sigma"], S.device, S.dtype)
    sigma_nammd = _scalar_tensor(best_nammd["sigma"], S.device, S.dtype)

    if verbose:
        print(
            "Selected bandwidths from reference samples: "
            f"MMD sigma={best_mmd['sigma']:.6g} score={best_mmd['mmd_score']:.3f}; "
            f"NAMMD sigma={best_nammd['sigma']:.6g} score={best_nammd['nammd_score']:.3f}"
        )

    return sigma_mmd, sigma_nammd, diagnostics
