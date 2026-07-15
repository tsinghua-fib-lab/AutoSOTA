import math
from typing import Dict, Tuple

import torch


@torch.no_grad()
def compute_nrmse(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """RMSE normalized by the target RMS."""
    pred = pred.to(dtype=torch.float32)
    target = target.to(dtype=torch.float32)
    mse = (pred - target).pow(2).mean()
    rmse = torch.sqrt(mse + eps)
    denom = torch.sqrt(target.pow(2).mean()).clamp_min(eps)
    return (rmse / denom).to(torch.float32)


@torch.no_grad()
def compute_vrmse(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """RMSE normalized by the target standard deviation."""
    pred = pred.to(dtype=torch.float32)
    target = target.to(dtype=torch.float32)
    mse = (pred - target).pow(2).mean()
    var = target.var(unbiased=False).clamp_min(eps)
    return torch.sqrt(mse / var).to(torch.float32)


@torch.no_grad()
def _split_logspace_three_bands(
    kmin: int, kmax: int
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """Split integer frequency bins into three disjoint log-spaced bands."""
    if kmax < kmin:
        raise ValueError(f"kmax ({kmax}) must be >= kmin ({kmin}).")
    if kmax == kmin:
        return (kmin, kmax), (kmin, kmax), (kmin, kmax)

    e = torch.logspace(
        math.log10(float(kmin)),
        math.log10(float(kmax)),
        steps=4,
        dtype=torch.float64,
    ).tolist()

    edges = [int(round(x)) for x in e]
    edges[0] = max(kmin, min(kmax, edges[0]))
    edges[3] = max(kmin, min(kmax, edges[3]))
    for i in range(1, 4):
        edges[i] = max(edges[i - 1], min(kmax, edges[i]))

    b0 = (edges[0], edges[1])
    b1 = (edges[1] + 1, edges[2])
    b2 = (edges[2] + 1, edges[3])
    return b0, b1, b2


@torch.no_grad()
def compute_psrmse_three_bands(
    y_samples: torch.Tensor,
    y_true: torch.Tensor,
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    """Binned spectral RMSE over low/mid/high frequency bands."""
    if y_samples.ndim < 4:
        raise ValueError(f"y_samples must be (K,N,C,...) got {tuple(y_samples.shape)}")
    if y_true.ndim != y_samples.ndim - 1:
        raise ValueError(f"y_true must be (N,C,...) got {tuple(y_true.shape)}")

    K, N = int(y_samples.shape[0]), int(y_samples.shape[1])
    pred = y_samples.reshape(K * N, *y_samples.shape[2:]).to(
        dtype=torch.float32
    )  # (K*N, C, ...)
    true = (
        y_true.unsqueeze(0)
        .expand(K, *y_true.shape)
        .reshape(K * N, *y_true.shape[1:])
        .to(dtype=torch.float32)
    )  # (K*N, C, ...)

    spatial_ndim = pred.ndim - 2  # (B, C, *spatial)
    if spatial_ndim not in (1, 2):
        raise ValueError(
            f"compute_psrmse_three_bands supports 1D/2D fields; got shape={tuple(pred.shape)}"
        )

    if spatial_ndim == 1:
        X = int(pred.shape[-1])
        if X <= 1:
            z = torch.zeros((), dtype=torch.float32)
            return {"psrmse_low": z, "psrmse_mid": z, "psrmse_high": z}
        k = torch.fft.fftfreq(X, d=1.0, device=pred.device) * X
        r = k.abs().round().to(torch.long)  # (X,)
        rmax = int(r.max().item())
        mask_shape = (X,)
        fft_dims = (-1,)
    else:
        H, W = int(pred.shape[-2]), int(pred.shape[-1])
        if H <= 1 or W <= 1:
            z = torch.zeros((), dtype=torch.float32)
            return {"psrmse_low": z, "psrmse_mid": z, "psrmse_high": z}
        ky = torch.fft.fftfreq(H, d=1.0, device=pred.device) * H
        kx = torch.fft.fftfreq(W, d=1.0, device=pred.device) * W
        KY, KX = torch.meshgrid(ky, kx, indexing="ij")
        r = torch.sqrt(KY * KY + KX * KX).floor().to(torch.long)  # (H,W)
        rmax = int(r.max().item())
        mask_shape = (H, W)
        fft_dims = (-2, -1)

    if rmax <= 1:
        z = torch.zeros((), dtype=torch.float32)
        return {"psrmse_low": z, "psrmse_mid": z, "psrmse_high": z}

    kmin, kmax = 1, int(rmax)
    (lo0, hi0), (lo1, hi1), (lo2, hi2) = _split_logspace_three_bands(kmin, kmax)

    F_pred = torch.fft.fftn(pred, dim=fft_dims)
    F_true = torch.fft.fftn(true, dim=fft_dims)

    def band_rmse(lo: int, hi: int) -> torch.Tensor:
        lo = int(max(kmin, lo))
        hi = int(min(kmax, hi))
        if hi < lo:
            return torch.zeros((), dtype=torch.float32)
        mask = (
            ((r >= lo) & (r <= hi))
            .to(dtype=F_pred.dtype, device=F_pred.device)
            .reshape((1, 1, *mask_shape))
        )
        yb_pred = torch.fft.ifftn(F_pred * mask, dim=fft_dims).real
        yb_true = torch.fft.ifftn(F_true * mask, dim=fft_dims).real
        mse = (yb_pred - yb_true).pow(2).mean()
        return torch.sqrt(mse + eps).to(torch.float32)

    return {
        "psrmse_low": band_rmse(lo0, hi0),
        "psrmse_mid": band_rmse(lo1, hi1),
        "psrmse_high": band_rmse(lo2, hi2),
    }


@torch.no_grad()
def _per_example_rmse_nrmse_vrmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-example RMSE, NRMSE, and VRMSE averaged over the batch."""
    if pred.ndim < 3 or target.ndim < 3:
        raise ValueError(
            f"pred/target must be (B,C,...) got {tuple(pred.shape)} / {tuple(target.shape)}"
        )
    if pred.shape != target.shape:
        raise ValueError(
            f"pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}"
        )

    pred = pred.to(dtype=torch.float32)
    target = target.to(dtype=torch.float32)

    reduce_dims = tuple(range(1, pred.ndim))  # C + spatial
    mse_b = (pred - target).pow(2).mean(dim=reduce_dims)  # (B,)
    rmse_b = torch.sqrt(mse_b + eps)

    denom_l2 = torch.sqrt(target.pow(2).mean(dim=reduce_dims)).clamp_min(eps)  # (B,)
    nrmse_b = rmse_b / denom_l2

    var_b = target.var(dim=reduce_dims, unbiased=False).clamp_min(eps)  # (B,)
    vrmse_b = rmse_b / torch.sqrt(var_b)

    return rmse_b.mean(), nrmse_b.mean(), vrmse_b.mean()


@torch.no_grad()
def compute_stochastic_mean_std_metrics(
    pred_samples: torch.Tensor,
    true_samples: torch.Tensor,
    *,
    eps: float = 1e-12,
    ed_k_chunk: int = 4,
    ed_s_chunk: int = 4,
    swd_num_projections: int = 32,
    swd_proj_chunk: int = 16,
) -> Dict[str, torch.Tensor]:
    """Compare predicted and true sample sets with ensemble and mean/std metrics."""
    if pred_samples.ndim < 4 or true_samples.ndim < 4:
        raise ValueError(
            f"pred_samples/true_samples must be (K,B,C,...) got {tuple(pred_samples.shape)} / {tuple(true_samples.shape)}"
        )
    if pred_samples.shape[1:] != true_samples.shape[1:]:
        raise ValueError(
            f"pred_samples and true_samples must match in (B,C,...) got {tuple(pred_samples.shape)} vs {tuple(true_samples.shape)}"
        )

    pred = pred_samples.to(dtype=torch.float32)
    true = true_samples.to(dtype=torch.float32)

    mean_pred = pred.mean(dim=0)
    mean_true = true.mean(dim=0)
    std_pred = pred.std(dim=0, unbiased=False)
    std_true = true.std(dim=0, unbiased=False)

    mean_rmse, mean_nrmse, mean_vrmse = _per_example_rmse_nrmse_vrmse(
        mean_pred, mean_true, eps=eps
    )
    std_rmse, std_nrmse, std_vrmse = _per_example_rmse_nrmse_vrmse(
        std_pred, std_true, eps=eps
    )

    def _energy_distance_l2(
        xs: torch.Tensor,  # (K,B,C,...)
        ys: torch.Tensor,  # (S,B,C,...)
    ) -> torch.Tensor:
        K = int(xs.shape[0])
        S = int(ys.shape[0])
        B = int(xs.shape[1])

        xs_f = xs.reshape(K, B, -1)
        ys_f = ys.reshape(S, B, -1)

        kc = max(1, int(ed_k_chunk))
        sc = max(1, int(ed_s_chunk))

        def cross_mean() -> torch.Tensor:
            total = torch.zeros((), dtype=torch.float64, device=xs_f.device)
            count = 0
            for k0 in range(0, K, kc):
                k1 = min(K, k0 + kc)
                xb = xs_f[k0:k1].unsqueeze(1)  # (k,1,B,D)
                for s0 in range(0, S, sc):
                    s1 = min(S, s0 + sc)
                    yb = ys_f[s0:s1].unsqueeze(0)  # (1,s,B,D)
                    dist = torch.linalg.norm(xb - yb, dim=-1)  # (k,s,B)
                    total = total + dist.sum(dtype=torch.float64)
                    count += int(dist.numel())
            return (
                (total / float(count)).to(torch.float32)
                if count > 0
                else torch.zeros((), dtype=torch.float32)
            )

        def within_mean(zs_f: torch.Tensor, Z: int) -> torch.Tensor:
            if Z <= 1:
                return torch.zeros((), dtype=torch.float32, device=zs_f.device)
            total = torch.zeros((), dtype=torch.float64, device=zs_f.device)
            count = 0
            for i0 in range(0, Z, kc):
                i1 = min(Z, i0 + kc)
                a = zs_f[i0:i1].unsqueeze(1)  # (a,1,B,D)
                for j0 in range(0, Z, kc):
                    j1 = min(Z, j0 + kc)
                    b = zs_f[j0:j1].unsqueeze(0)  # (1,b,B,D)
                    dist = torch.linalg.norm(a - b, dim=-1)  # (a,b,B)
                    total = total + dist.sum(dtype=torch.float64)
                    count += int(dist.numel())
            return (
                (total / float(count)).to(torch.float32)
                if count > 0
                else torch.zeros((), dtype=torch.float32)
            )

        exy = cross_mean()
        exx = within_mean(xs_f, K)
        eyy = within_mean(ys_f, S)
        return (2.0 * exy - exx - eyy).to(torch.float32)

    stochastic_ed = _energy_distance_l2(pred, true)

    mean_true_ref = true.mean(dim=0)
    y_var_pred = pred.var(dim=0, unbiased=False).clamp_min(eps)
    mse_mean = (mean_pred - mean_true_ref).pow(2).mean()
    rmse_mean = (mse_mean + eps).sqrt()
    spread = y_var_pred.mean().sqrt()
    stochastic_ssr = spread / (rmse_mean + eps)

    def _sliced_wasserstein_l1(
        xs: torch.Tensor,  # (K,B,C,...)
        ys: torch.Tensor,  # (S,B,C,...)
        *,
        num_projections: int,
        proj_chunk: int,
        eps: float,
    ) -> torch.Tensor:
        K = int(xs.shape[0])
        S = int(ys.shape[0])
        if K <= 0 or S <= 0:
            return torch.zeros((), dtype=torch.float32, device=xs.device)

        xs_f = xs.reshape(K, int(xs.shape[1]), -1)  # (K,B,D)
        ys_f = ys.reshape(S, int(ys.shape[1]), -1)  # (S,B,D)
        D = int(xs_f.shape[-1])
        if D <= 0:
            return torch.zeros((), dtype=torch.float32, device=xs.device)

        P = max(1, int(num_projections))
        pc = max(1, int(proj_chunk))

        L = max(K, S)
        idx_x = torch.linspace(0, K - 1, steps=L, device=xs.device).to(torch.long)
        idx_y = torch.linspace(0, S - 1, steps=L, device=xs.device).to(torch.long)

        total = torch.zeros((), dtype=torch.float64, device=xs.device)
        count = 0
        for p0 in range(0, P, pc):
            p1 = min(P, p0 + pc)
            pp = int(p1 - p0)

            dirs = torch.randn((pp, D), device=xs.device, dtype=xs_f.dtype)
            dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + eps)

            x_proj = torch.einsum("kbd,pd->pkb", xs_f, dirs)
            y_proj = torch.einsum("sbd,pd->psb", ys_f, dirs)

            x_sorted, _ = x_proj.sort(dim=1)
            y_sorted, _ = y_proj.sort(dim=1)

            x_q = x_sorted[:, idx_x, :]
            y_q = y_sorted[:, idx_y, :]

            wd_pb = (x_q - y_q).abs().mean(dim=1)  # (pp,B)
            wd_p = wd_pb.mean(dim=-1)  # (pp,)
            total = total + wd_p.sum(dtype=torch.float64)
            count += int(wd_p.numel())

        return (
            (total / float(count)).to(torch.float32)
            if count > 0
            else torch.zeros((), dtype=torch.float32, device=xs.device)
        )

    def _sliced_wasserstein_l2(
        xs: torch.Tensor,  # (K,B,C,...)
        ys: torch.Tensor,  # (S,B,C,...)
        *,
        num_projections: int,
        proj_chunk: int,
        eps: float,
    ) -> torch.Tensor:
        """Sliced W_2: sqrt(mean_p W_2^2(1D projections)); W_2^2 via squared quantile differences on a common grid."""
        K = int(xs.shape[0])
        S = int(ys.shape[0])
        if K <= 0 or S <= 0:
            return torch.zeros((), dtype=torch.float32, device=xs.device)

        xs_f = xs.reshape(K, int(xs.shape[1]), -1)
        ys_f = ys.reshape(S, int(ys.shape[1]), -1)
        D = int(xs_f.shape[-1])
        if D <= 0:
            return torch.zeros((), dtype=torch.float32, device=xs.device)

        P = max(1, int(num_projections))
        pc = max(1, int(proj_chunk))

        L = max(K, S)
        idx_x = torch.linspace(0, K - 1, steps=L, device=xs.device).to(torch.long)
        idx_y = torch.linspace(0, S - 1, steps=L, device=xs.device).to(torch.long)

        total = torch.zeros((), dtype=torch.float64, device=xs.device)
        count = 0
        for p0 in range(0, P, pc):
            p1 = min(P, p0 + pc)
            pp = int(p1 - p0)

            dirs = torch.randn((pp, D), device=xs.device, dtype=xs_f.dtype)
            dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + eps)

            x_proj = torch.einsum("kbd,pd->pkb", xs_f, dirs)
            y_proj = torch.einsum("sbd,pd->psb", ys_f, dirs)

            x_sorted, _ = x_proj.sort(dim=1)
            y_sorted, _ = y_proj.sort(dim=1)

            x_q = x_sorted[:, idx_x, :]
            y_q = y_sorted[:, idx_y, :]

            w2_sq_pb = (x_q - y_q).pow(2).mean(dim=1)
            w2_sq_p = w2_sq_pb.mean(dim=-1)
            total = total + w2_sq_p.sum(dtype=torch.float64)
            count += int(w2_sq_p.numel())

        mean_w2_sq = (
            (total / float(count))
            if count > 0
            else torch.zeros((), dtype=torch.float64, device=xs.device)
        )
        return mean_w2_sq.sqrt().to(torch.float32)

    stochastic_swd = _sliced_wasserstein_l1(
        pred,
        true,
        num_projections=swd_num_projections,
        proj_chunk=swd_proj_chunk,
        eps=eps,
    )

    stochastic_swd2 = _sliced_wasserstein_l2(
        pred,
        true,
        num_projections=swd_num_projections,
        proj_chunk=swd_proj_chunk,
        eps=eps,
    )

    return {
        "stochastic_ed": stochastic_ed,
        "stochastic_swd": stochastic_swd,
        "stochastic_swd2": stochastic_swd2,
        "stochastic_ssr": stochastic_ssr.to(torch.float32),
        "stochastic_mean_rmse": mean_rmse.to(torch.float32),
        "stochastic_mean_nrmse": mean_nrmse.to(torch.float32),
        "stochastic_mean_vrmse": mean_vrmse.to(torch.float32),
        "stochastic_std_rmse": std_rmse.to(torch.float32),
        "stochastic_std_nrmse": std_nrmse.to(torch.float32),
        "stochastic_std_vrmse": std_vrmse.to(torch.float32),
    }
