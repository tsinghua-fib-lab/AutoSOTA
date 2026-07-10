# gepc/methods/gepc.py
# -*- coding: utf-8 -*-
"""
GEPC — Group-Equivariant Posterior Consistency (training-free OOD)

At a timestep t, apply a small discrete image group G (flips, 90/180 rotations, optional shifts)
to the noisy sample x_t. For an approximately equivariant denoiser:

    s(g·x_t, t) ≈ g·s(x_t, t)    and    x̂0(g·x_t, t) ≈ g·x̂0(x_t, t)

In-distribution samples satisfy this consistency; OOD samples break it.

This file is a cleaned, Improved-Diffusion-only implementation.
Important: we intentionally keep the same computation path (including RNG consumption)
to preserve historical AUROC values under strict determinism.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple
import math

import numpy as np
import torch
from torch import Tensor


# --------------------- basic helpers ---------------------

def to_minus1_1(x: Tensor) -> Tensor:
    """Convert inputs to [-1,1] if they look like [0,1] or [0,255]."""
    x = x.detach()
    xmin, xmax = float(x.min()), float(x.max())
    if xmin >= -1.01 and xmax <= 1.01:
        return x
    if xmin >= 0.0 and xmax <= 1.01:
        return (x * 2.0 - 1.0).clamp(-1, 1)
    return (x / 127.5 - 1.0).clamp(-1, 1)


@torch.no_grad()
def _pred_raw(adapter, xt: Tensor, t_idx: int, amp: str = "fp16") -> Tensor:
    """Model forward: returns raw UNet output (eps or eps+var)."""
    B, _ = xt.shape[:2]
    t = torch.full((B,), int(t_idx), device=xt.device, dtype=torch.long)
    with torch.cuda.amp.autocast(enabled=(amp == "fp16")):
        y = adapter.model(xt, t)  # [B, C or 2C, H, W]
    return y.float()


def _split_eps_var(y: Tensor, C: int) -> Tuple[Tensor, Optional[Tensor]]:
    """Split eps / var channels if learn_sigma is enabled."""
    if y.shape[1] >= 2 * C:
        return y[:, :C, ...], y[:, C:2 * C, ...]
    return y[:, :C, ...], None


def _score_from_eps(adapter, eps: Tensor, t_idx: int) -> Tensor:
    """score = -eps / sigma_t under DDPM scaling."""
    sigma = adapter.sigma_t(t_idx).to(eps.device).view(1, 1, 1, 1).float()
    return -eps / (sigma + 1e-12)


@torch.no_grad()
def _forward_noisy(adapter, x0: Tensor, t_idx: int, gen: Optional[torch.Generator] = None) -> Tensor:
    """DDPM forward noising: x_t = sqrt(a_bar)*x0 + sqrt(1-a_bar)*noise."""
    ab = adapter.alphas_cumprod[int(t_idx)].to(x0.device).view(1, 1, 1, 1)
    if gen is None:
        noise = torch.randn_like(x0)
    else:
        noise = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=gen)
    return torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * noise


def _predict_x0_from_eps(xt: Tensor, eps: Tensor, ab_t: Tensor) -> Tensor:
    """Analytic x0_hat from (x_t, eps, a_bar_t)."""
    sqrt_ab = torch.sqrt(ab_t).to(xt.device).view(1, 1, 1, 1)
    sqrt_1m = torch.sqrt(1.0 - ab_t).to(xt.device).view(1, 1, 1, 1)
    return (xt - sqrt_1m * eps) / (sqrt_ab + 1e-12)


@torch.no_grad()
def _predict_x0(adapter, xt: Tensor, eps: Tensor, t_idx: int) -> Tensor:
    """Wrapper to compute x0_hat under DDPM scaling."""
    ab_t = adapter.alphas_cumprod[int(t_idx)]
    return _predict_x0_from_eps(xt, eps, ab_t)


def _cosine(a: Tensor, b: Tensor, tau: float = 1e-6) -> Tensor:
    """Cosine similarity per-sample over all dims (flattened)."""
    af = a.flatten(1)
    bf = b.flatten(1)
    na = (af.square().sum(1).sqrt()).clamp_min(tau)
    nb = (bf.square().sum(1).sqrt()).clamp_min(tau)
    dot = (af * bf).sum(1)
    return dot / (na * nb)


def _pool_spatial(v: Tensor, mode: str = "mean", rho: float = 0.10) -> Tensor:
    """
    Spatial pooling of a map to [B].
    - mean: uniform mean
    - topk: mean of top-k% absolute values
    """
    if v.dim() == 4:
        v = v.mean(dim=1)  # [B,H,W] channel-average
    assert v.dim() == 3, f"Expected [B,H,W] or [B,C,H,W], got {v.shape}"
    B, H, W = v.shape
    if mode == "mean":
        return v.view(B, -1).mean(dim=1)
    z = v.abs().view(B, -1)
    k = max(1, int(rho * z.shape[1]))
    topk = torch.topk(z, k=k, dim=1).values
    return topk.mean(dim=1)


def _build_group_ops(H: int, W: int, use_shifts: bool = False, s: int = 1):
    """Discrete group ops (apply, inverse), BCHW->BCHW."""
    def _id(x): return x
    def _idh(x): return x
    def _h(x): return torch.flip(x, dims=[3])
    def _v(x): return torch.flip(x, dims=[2])
    def _rot90(x):  return torch.rot90(x, k=1, dims=(2, 3))
    def _rot270(x): return torch.rot90(x, k=3, dims=(2, 3))
    def _rot180(x): return torch.rot90(x, k=2, dims=(2, 3))

    G = [(_id, _idh), (_h, _h), (_v, _v)]
    if H == W:
        G += [(_rot90, _rot270), (_rot180, _rot180)]

    if use_shifts:
        def rollx(x):   return torch.roll(x, shifts=s, dims=3)
        def unrollx(x): return torch.roll(x, shifts=-s, dims=3)
        def rolly(x):   return torch.roll(x, shifts=s, dims=2)
        def unrolly(x): return torch.roll(x, shifts=-s, dims=2)
        G += [(rollx, unrollx), (rolly, unrolly)]
    return G


# --------------------- tiny 1D KDE (auto bandwidth) ---------------------

class _KDE1D:
    """Minimal 1D Gaussian KDE for ID-only calibration. Stored on CPU."""
    def __init__(self, bandwidth: float = 0.0):
        self.bw = float(bandwidth)
        self.ref: Optional[Tensor] = None

    def fit(self, x: Tensor) -> None:
        x = x.detach().float().view(-1).cpu()
        x = x[torch.isfinite(x)]
        if x.numel() == 0:
            x = torch.zeros(1)

        if self.bw <= 0.0:
            # Silverman, robust via IQR
            N = max(1, x.numel())
            q25, q75 = torch.quantile(x, torch.tensor([0.25, 0.75]))
            iqr = (q75 - q25).item()
            std = float(x.std().clamp_min(1e-8))
            sigma = min(std, iqr / 1.34 if iqr > 0 else std)
            self.bw = 0.9 * sigma * (N ** (-1 / 5))
            if not math.isfinite(self.bw) or self.bw <= 1e-6:
                self.bw = 0.1

        self.ref = x.contiguous()

    @torch.no_grad()
    def logp(self, x: Tensor) -> Tensor:
        z = x.detach().float().view(-1).cpu()
        z = z[torch.isfinite(z)]
        if z.numel() == 0:
            return torch.full((0,), -float("inf"))

        X = self.ref if self.ref is not None else torch.zeros(1)
        N = max(1, int(X.shape[0]))
        var = self.bw * self.bw
        cst = -0.5 * math.log(2.0 * math.pi * var)
        diff2 = (z.unsqueeze(1) - X.unsqueeze(0)) ** 2
        logp = torch.logsumexp(cst - diff2 / (2.0 * var), dim=1) - math.log(N)
        return logp


# --------------------- simple MVN for vector mode ---------------------

class _MVN:
    """Gaussian fit on stacked (t,feature) vectors; returns NLL."""
    def __init__(self):
        self.mu: Optional[Tensor] = None
        self.Sinv: Optional[Tensor] = None

    def fit(self, X: Tensor) -> None:
        X = X.detach().float()
        self.mu = X.mean(0, keepdim=True)
        C = torch.cov((X - self.mu).T) + 1e-4 * torch.eye(X.shape[1])
        self.Sinv = torch.linalg.inv(C)

    @torch.no_grad()
    def nll(self, X: Tensor) -> Tensor:
        Z = X.detach().float() - self.mu
        return 0.5 * (Z @ self.Sinv * Z).sum(-1)


# --------------------- main class ---------------------

class GEPC:
    """
    GEPC score. All returned scores are OOD-high.

    density_mode ∈ {"none","zscore","kde"}
    agg_t ∈ {"mean","max","min","wmean","trimmean"}
    vector_mode ∈ {"none","mvn"}
    """
    name = "GEPC"

    def __init__(
        self,
        # timestep selection
        t_list: Optional[Iterable[int]] = None,
        t_mode: str = "snr",
        snr_levels: Iterable[float] = (0.997, 0.993, 0.985, 0.97),
        keep_k: int = 3,

        # numerics
        tau: float = 1e-8,
        amp: str = "fp16",
        clamp_x: bool = False,

        # features & aggregation
        features: Iterable[str] = ("gepc_s", "gepc_s_cos", "gepc_x0", "cycle"),
        metric_default: str = "gepc_s",
        agg_feat: str = "sum",
        agg_t: str = "wmean",
        trim_alpha: float = 0.10,
        weight_t: str = "inv_cv",

        # calibration
        density_mode: str = "kde",
        bandwidth: float = 0.0,
        fit_batches: int = 12,
        mc_samples: int = 2,

        # vector mode
        vector_mode: str = "none",

        # stability
        internal_bs: int = 64,
        max_fit_batches: Optional[int] = None,
        max_score_batches: Optional[int] = None,
        verbose: bool = True,
        seed: Optional[int] = None,

        # group & pooling
        group_shifts: bool = False,
        shift_px: int = 1,
        spatial_pool: str = "mean",
        topk_rho: float = 0.10,

        **kwargs,
    ):
        self.t_list = list(t_list) if t_list is not None else None
        self.t_mode = str(t_mode)
        self.snr_levels = list(snr_levels)
        self.keep_k = int(max(1, keep_k))

        self.tau = float(tau)
        self.amp = str(amp)
        self.clamp_x = bool(clamp_x)

        self.features = tuple(features)
        self.metric_default = str(metric_default)
        self.agg_feat = str(agg_feat)
        self.agg_t = str(agg_t)
        self.trim_alpha = float(trim_alpha)
        self.weight_t = str(weight_t)

        self.density_mode = str(density_mode)
        self.bandwidth = float(bandwidth)
        self.fit_batches = int(fit_batches)

        self.vector_mode = str(vector_mode)
        self._mvn: Optional[_MVN] = None

        self.bs = int(internal_bs)
        self.mc = int(mc_samples)
        self.max_fit_batches = None if max_fit_batches is None else int(max_fit_batches)
        self.max_score_batches = None if max_score_batches is None else int(max_score_batches)
        self.verbose = bool(verbose)

        self._t_final: List[int] = []
        self._t_kept: Optional[set] = None
        self._mu: Dict[Tuple[int, str], float] = {}
        self._sig: Dict[Tuple[int, str], float] = {}
        self._kde: Dict[Tuple[int, str], _KDE1D] = {}
        self._w_t: Dict[int, float] = {}

        self.seed = None if seed is None else int(seed)
        self._gen: Optional[torch.Generator] = None

        self.group_shifts = bool(group_shifts)
        self.shift_px = int(shift_px)
        self.spatial_pool = str(spatial_pool)
        self.topk_rho = float(topk_rho)

    def _ensure_generator(self, device: torch.device) -> None:
        if self._gen is None:
            self._gen = torch.Generator(device=device)
            base_seed = torch.initial_seed() if self.seed is None else self.seed
            self._gen.manual_seed(base_seed)

    @property
    def return_id_large(self) -> bool:
        return False  # scores are OOD-high

    # --------------- t selection ---------------

    def _build_t_list(self, adapter) -> None:
        if self.t_list:
            T = int(len(adapter.alphas_cumprod))
            if T <= 0:
                self._t_final = [int(t) for t in self.t_list]
            else:
                self._t_final = [max(0, min(int(t), T - 1)) for t in self.t_list]
            return

        if self.t_mode == "fixed":
            T = int(len(adapter.alphas_cumprod))
            self._t_final = [200, 400, 800] if T >= 2000 else [35, 70, 140]
            return

        ab = adapter.alphas_cumprod.detach().float().cpu().numpy()
        snr = np.sqrt(ab)
        tsel = []
        for lvl in self.snr_levels:
            idx = int(np.argmin(np.abs(snr - float(lvl))))
            tsel.append(idx)
        self._t_final = sorted(set(tsel))

    # --------------- per-t weights ---------------

    def _compute_t_weights_from_id(self, id_buf: Dict[int, Tensor]) -> None:
        if self.weight_t == "none":
            self._w_t = {t: 1.0 / max(1, len(self._t_final)) for t in self._t_final}
            return
        base_vals = {}
        for t in self._t_final:
            v = id_buf.get(t, None)
            if v is None or v.numel() == 0:
                base_vals[t] = (1.0, 1.0)
                continue
            mu = float(v.mean())
            sd = float(v.std().clamp_min(1e-6))
            base_vals[t] = (mu, sd)
        w = {}
        for t in self._t_final:
            mu, sd = base_vals[t]
            cv = sd / (abs(mu) + 1e-6)
            w[t] = 1.0 / (cv + 1e-6)
        s = sum(w.values()) + 1e-12
        self._w_t = {t: float(w[t] / s) for t in self._t_final}

        if self.verbose:
            printable = {t: round(self._w_t[t], 3) for t in self._t_final}
            print("[GEPC] per-t weights:", printable)

    def _stable_t_mask(self, base_buf: Dict[int, Tensor], keep_k: int) -> set:
        stats = []
        for t in self._t_final:
            v = base_buf.get(t, None)
            if v is None or v.numel() == 0:
                stats.append((t, float("inf")))
                continue
            mu = float(v.mean())
            sd = float(v.std().clamp_min(1e-8))
            cv = sd / (abs(mu) + 1e-6)
            stats.append((t, cv))
        stats.sort(key=lambda z: z[1])
        kept = [t for t, _ in stats[:min(keep_k, len(stats))]]
        return set(kept)

    # --------------- core GEPC at a single t ---------------

    @torch.no_grad()
    def _gepc_at_t(self, adapter, x0: Tensor, t_idx: int) -> Dict[str, Tensor]:
        """
        Returns a dict of per-sample features at timestep t_idx (all OOD-high).

        NOTE: we keep the full computation path (including x0_hat + cycle forward call)
        to preserve RNG consumption and historical AUROC values.
        """
        xt = _forward_noisy(adapter, x0, t_idx, gen=self._gen).float()
        if self.clamp_x:
            xt = xt.clamp(-1, 1)

        B, C, H, W = xt.shape
        Gops = _build_group_ops(H, W, use_shifts=self.group_shifts, s=self.shift_px)

        # 1) base pass
        y0 = _pred_raw(adapter, xt, t_idx, amp=self.amp)
        eps0, var0 = _split_eps_var(y0, C)
        s0 = _score_from_eps(adapter, eps0, t_idx)
        x0_hat = _predict_x0(adapter, xt, eps0, t_idx)

        # 2) group batched pass
        xs = []
        invs = []
        for (g, ginv) in Gops:
            xs.append(g(xt))
            invs.append(ginv)
        xg = torch.cat(xs, dim=0)

        yg = _pred_raw(adapter, xg, t_idx, amp=self.amp)
        epsg, varg = _split_eps_var(yg, C)
        sg = _score_from_eps(adapter, epsg, t_idx)
        x0g_hat = _predict_x0(adapter, xg, epsg, t_idx)

        # 3) back to canonical frame
        sg_list, x0g_list, var_list = [], [], []
        for i, (_, ginv) in enumerate(Gops):
            sg_i = sg[i * B:(i + 1) * B]
            x0g_i = x0g_hat[i * B:(i + 1) * B]
            sg_list.append(ginv(sg_i))
            x0g_list.append(ginv(x0g_i))
            if varg is not None:
                var_i = varg[i * B:(i + 1) * B]
                var_list.append(ginv(var_i))

        sg_back = torch.stack(sg_list, 0)     # [nG,B,C,H,W]
        x0_back = torch.stack(x0g_list, 0)    # [nG,B,C,H,W]
        var_back = torch.stack(var_list, 0) if len(var_list) else None

        # 4) metrics (OOD-high)
        s_ref = s0.unsqueeze(0)
        x0_ref = x0_hat.unsqueeze(0)

        # (a) L2 score consistency normalized by pooled ||s||^2
        sg_L2_list = []
        for i in range(sg_back.shape[0]):
            diff_map = (sg_back[i] - s_ref[0]).square()
            sg_L2_list.append(_pool_spatial(diff_map, self.spatial_pool, self.topk_rho))
        s_gap = torch.stack(sg_L2_list, 0)  # [nG,B]

        denom_s_map = s0.square()
        denom_s = _pool_spatial(denom_s_map, self.spatial_pool, self.topk_rho).unsqueeze(0) + self.tau
        norm_s_gap = s_gap / denom_s
        gepc_s = norm_s_gap.mean(dim=0)

        # (a') cosine consistency (1 - cos)
        s_cos_list = []
        for i in range(sg_back.shape[0]):
            cos = _cosine(sg_back[i], s_ref[0])
            s_cos_list.append(1.0 - cos)
        gepc_s_cos = torch.stack(s_cos_list, 0).mean(0)

        # (a'') inter-group dispersion + pairwise
        gepc_s_var_g = norm_s_gap.var(dim=0, unbiased=False)

        pairs = []
        nG = sg_back.shape[0]
        for i in range(nG):
            for j in range(i + 1, nG):
                pij = _pool_spatial((sg_back[i] - sg_back[j]).square(), self.spatial_pool, self.topk_rho)
                pij = pij / denom_s[0]
                pairs.append(pij)
        gepc_s_pair = torch.stack(pairs, 0).mean(0) if len(pairs) else norm_s_gap.mean(0)

        # (b) x0 consistency normalized by pooled ||x0_hat||^2
        x0_L2_list = []
        for i in range(x0_back.shape[0]):
            diff_map = (x0_back[i] - x0_ref[0]).square()
            x0_L2_list.append(_pool_spatial(diff_map, self.spatial_pool, self.topk_rho))
        x0_gap = torch.stack(x0_L2_list, 0)

        denom_x0_map = x0_hat.square()
        denom_x0 = _pool_spatial(denom_x0_map, self.spatial_pool, self.topk_rho).unsqueeze(0) + self.tau
        gepc_x0 = (x0_gap / denom_x0).mean(dim=0)

        # (c) one-step cycle (kept for exact RNG consumption)
        xt_rt = _forward_noisy(adapter, x0_hat, t_idx, gen=self._gen)
        cyc_map = (xt_rt - xt).square() / (xt.square() + self.tau)
        cycle = _pool_spatial(cyc_map, self.spatial_pool, self.topk_rho)

        # (d) variance symmetry if learn_sigma
        if var_back is not None:
            vpooled = []
            for i in range(var_back.shape[0]):
                vpooled.append(_pool_spatial(var_back[i], self.spatial_pool, self.topk_rho))
            vpooled = torch.stack(vpooled, 0)
            vmean0 = vpooled[0]
            gepc_var = (vpooled - vmean0.unsqueeze(0)).abs().mean(dim=0)
        else:
            gepc_var = None

        out: Dict[str, Tensor] = {}
        if "gepc_s" in self.features: out["gepc_s"] = gepc_s
        if "gepc_s_cos" in self.features: out["gepc_s_cos"] = gepc_s_cos
        if "gepc_x0" in self.features: out["gepc_x0"] = gepc_x0
        if "cycle" in self.features: out["cycle"] = cycle
        if "gepc_var" in self.features and (gepc_var is not None): out["gepc_var"] = gepc_var
        if "gepc_s_var_g" in self.features: out["gepc_s_var_g"] = gepc_s_var_g
        if "gepc_s_pair" in self.features: out["gepc_s_pair"] = gepc_s_pair
        return out

    @torch.no_grad()
    def _features_per_t(self, adapter, x0: Tensor, t_idx: int) -> Dict[str, Tensor]:
        acc: Dict[str, Tensor] = {}
        for _ in range(max(1, self.mc)):
            f = self._gepc_at_t(adapter, x0, t_idx)
            if not acc:
                acc = {k: v.clone() for k, v in f.items()}
            else:
                for k in acc:
                    acc[k] = acc[k] + f[k]
        for k in acc:
            acc[k] = acc[k] / float(max(1, self.mc))
        return acc

    # --------------- loader iteration ---------------

    @torch.no_grad()
    def _iter_loader(self, loader, limit: Optional[int], desc: str):
        if limit is None:
            total = len(loader)
            it = enumerate(loader)
        else:
            total = min(int(limit), len(loader))
            it = zip(range(total), loader)

        if self.verbose:
            try:
                from tqdm import tqdm
                it = tqdm(it, total=total, desc=desc)
            except Exception:
                pass

        for _, batch in it:
            yield batch

    # --------------- fit (ID) ---------------

    def fit_id_train(self, adapter, loader) -> None:
        self._build_t_list(adapter)

        if self.verbose:
            print(f"[{self.name}] timesteps={self._t_final} | density_mode={self.density_mode} | "
                  f"vector_mode={self.vector_mode} | features={self.features} | "
                  f"spatial_pool={self.spatial_pool} topk_rho={self.topk_rho} | "
                  f"group_shifts={self.group_shifts} shift_px={self.shift_px}")

        dev = next(adapter.model.parameters()).device
        self._ensure_generator(dev)

        nmax = self.fit_batches if self.max_fit_batches is None else min(self.fit_batches, self.max_fit_batches)

        buf: Dict[Tuple[int, str], List[Tensor]] = {(t, f): [] for t in self._t_final for f in self.features}
        base_key = "gepc_s" if "gepc_s" in self.features else ("cycle" if "cycle" in self.features else self.features[0])
        base_buf: Dict[int, Tensor] = {}

        nb = 0
        for x, _ in self._iter_loader(loader, nmax, f"[{self.name}] fit(batches)"):
            x = to_minus1_1(x.to(dev, non_blocking=True))
            for i in range(0, x.shape[0], self.bs):
                xb = x[i:i + self.bs]
                for t in self._t_final:
                    feats = self._features_per_t(adapter, xb, t)
                    for f in self.features:
                        buf[(t, f)].append(feats[f].detach().float().cpu())
            nb += 1
            if nmax is not None and nb >= nmax:
                break

        for t in self._t_final:
            vlist = buf.get((t, base_key), [])
            base_buf[t] = torch.cat(vlist, 0) if len(vlist) else torch.zeros(1)

        self._compute_t_weights_from_id(base_buf)
        self._t_kept = self._stable_t_mask(base_buf, keep_k=self.keep_k)

        if self.verbose:
            print("[GEPC] kept timesteps:", sorted(self._t_kept))

        # Fit calibration
        if self.density_mode == "kde":
            self._kde = {}
            for k, lst in buf.items():
                data = torch.cat(lst, 0) if len(lst) else torch.zeros(1)
                kde = _KDE1D(self.bandwidth)
                kde.fit(data)
                self._kde[k] = kde
        elif self.density_mode == "zscore":
            self._mu, self._sig = {}, {}
            for k, lst in buf.items():
                data = torch.cat(lst, 0) if len(lst) else torch.zeros(1)
                self._mu[k] = float(data.mean())
                self._sig[k] = float(data.std().clamp_min(1e-6))

        # Optional MVN vector mode
        if self.vector_mode == "mvn":
            X = self._stack_vec_from_buf(buf, use_kept=True)
            if X is not None and X.shape[0] >= 100 and X.shape[1] >= 2:
                self._mvn = _MVN()
                self._mvn.fit(X.float())
                if self.verbose:
                    print(f"[GEPC] MVN vector fit: N={X.shape[0]} D={X.shape[1]}")
            elif self.verbose:
                print("[GEPC] MVN skipped (insufficient data)")

    def _stack_vec_from_buf(self, buf: Dict[Tuple[int, str], List[Tensor]], use_kept: bool) -> Optional[Tensor]:
        Z: List[Tensor] = []
        t_iter = [t for t in self._t_final if (not use_kept) or (self._t_kept is None) or (t in self._t_kept)]
        for t in t_iter:
            for f in self.features:
                lst = buf.get((t, f), [])
                if len(lst) == 0:
                    continue
                z = torch.cat(lst, 0).view(-1, 1)
                Z.append(z)
        if not Z:
            return None
        return torch.cat(Z, 1)

    # --------------- aggregation helpers ---------------

    def _agg_features_at_t(self, feats: Dict[str, Tensor], t: int) -> Tensor:
        # density_mode == "none": raw OOD-high features can be combined directly
        if self.density_mode == "none":
            vals = [feats[f] for f in self.features if f in feats]
            if len(vals) == 0:
                raise ValueError(f"No available feature among {self.features}")
            if self.agg_feat == "mean":
                return torch.stack(vals, 0).mean(0)
            if self.agg_feat == "sum":
                return torch.stack(vals, 0).sum(0)
            raw = feats.get(self.metric_default, None)
            if raw is None:
                raise ValueError(f"metric_default={self.metric_default} not in {list(feats.keys())}")
            return raw

        # density_mode != none: produce ID-like log-likelihood and aggregate across features
        vals = []
        if self.density_mode == "kde":
            for f in self.features:
                kde = self._kde.get((t, f), None)
                if kde is None:
                    raise RuntimeError(f"KDE not fit for (t={t}, feature={f})")
                lp = kde.logp(feats[f]).to(next(iter(feats.values())).device)
                vals.append(lp)
            L = torch.stack(vals, 0)  # ID-like
            return L.mean(0) if self.agg_feat == "mean" else L.sum(0)

        if self.density_mode == "zscore":
            for f in self.features:
                mu = self._mu[(t, f)]
                sig = self._sig[(t, f)]
                z = (feats[f] - mu) / sig
                vals.append(-0.5 * (z ** 2))  # ID-like
            L = torch.stack(vals, 0)
            return L.mean(0) if self.agg_feat == "mean" else L.sum(0)

        raise ValueError(self.density_mode)

    def _agg_across_t(self, S: Tensor, t_used: List[int]) -> Tensor:
        T, B = S.shape
        if T == 0:
            return torch.zeros(B, device=S.device)

        if self.agg_t == "max":
            out = S.max(0).values
        elif self.agg_t == "min":
            out = S.min(0).values
        elif self.agg_t == "wmean":
            w = torch.tensor(
                [self._w_t.get(int_t, 1.0 / max(1, T)) for int_t in t_used],
                device=S.device,
                dtype=torch.float32,
            ).view(T, 1)
            w = w / (w.sum() + 1e-12)
            out = (w * S).sum(0)
        elif self.agg_t == "trimmean":
            k = max(0, int(self.trim_alpha * T / 2.0))
            S_sorted, _ = torch.sort(S, dim=0)
            S_trim = S_sorted[k:T - k, :] if k > 0 and (2 * k) < T else S_sorted
            out = S_trim.mean(0)
        else:
            out = S.mean(0)
        return out

    # --------------- scoring ---------------

    @torch.no_grad()
    def score_loader(self, adapter, loader, tag: Optional[str] = None) -> np.ndarray:
        dev = next(adapter.model.parameters()).device
        self._ensure_generator(dev)
        self._build_t_list(adapter)

        scores: List[np.ndarray] = []
        nb = 0
        for x, _ in self._iter_loader(loader, self.max_score_batches, f"[{self.name}] score(batches)"):
            x = to_minus1_1(x.to(dev, non_blocking=True))

            chunk: List[Tensor] = []
            for i in range(0, x.shape[0], self.bs):
                xb = x[i:i + self.bs]

                # Vector MVN path
                if self.vector_mode == "mvn" and (self._mvn is not None):
                    t_used = [t for t in self._t_final if (self._t_kept is None) or (t in self._t_kept)]
                    vec_cols = []
                    for t in t_used:
                        feats = self._features_per_t(adapter, xb, t)
                        for f in self.features:
                            vec_cols.append(feats[f].detach().float().view(-1, 1))
                    if len(vec_cols) >= 2:
                        X = torch.cat(vec_cols, 1)
                        lid = -self._mvn.nll(X.to(self._mvn.mu.device))  # ID-like high
                        sb = (-lid).float().cpu()  # OOD-high
                        chunk.append(sb)
                        continue

                # KDE / zscore / none path
                t_used = [t for t in self._t_final if (self._t_kept is None) or (t in self._t_kept)]
                per_t_vals: List[Tensor] = []
                for t in t_used:
                    feats = self._features_per_t(adapter, xb, t)
                    s_t = self._agg_features_at_t(feats, t)
                    per_t_vals.append(s_t)

                S = torch.stack(per_t_vals, 0) if len(per_t_vals) else torch.zeros(0, xb.shape[0], device=dev)

                if self.density_mode == "none":
                    ood = self._agg_across_t(S, t_used)
                    sb = ood.float().cpu()
                else:
                    lid = self._agg_across_t(S, t_used)   # ID-like high
                    sb = (-lid).float().cpu()             # OOD-high
                chunk.append(sb)

            scores.append(torch.cat(chunk, 0).numpy())
            nb += 1
            if self.max_score_batches is not None and nb >= self.max_score_batches:
                break

        return np.concatenate(scores, 0) if scores else np.zeros(0, dtype=np.float32)

