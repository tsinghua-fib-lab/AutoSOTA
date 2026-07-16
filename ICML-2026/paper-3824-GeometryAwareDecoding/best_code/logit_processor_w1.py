#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
from transformers import LogitsProcessor

__all__ = ["TopW_LogitsProcessor"]


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    ex = np.exp(x)
    s = np.sum(ex)
    if s == 0.0 or not np.isfinite(s):
        ex = np.exp(np.clip(x, -80, 80))
        s = np.sum(ex) + 1e-12
    return (ex / s).astype(np.float64, copy=False)


def _pairwise_dist_embeddings(E: np.ndarray) -> np.ndarray:
    E = np.asarray(E, dtype=np.float32)
    dot = E @ E.T
    np.clip(dot, -1.0, 1.0, out=dot)
    D2 = np.maximum(2.0 - 2.0 * dot, 0.0)
    D = np.sqrt(D2).astype(float, copy=False)
    np.fill_diagonal(D, 0.0)
    return D


def _nearest_set_distance_cosine(
    E: np.ndarray,
    S_local_idx: np.ndarray,
    chunk: int = 4096,
) -> np.ndarray:
    M, _ = E.shape
    if S_local_idx.size == 0:
        return np.zeros(M, dtype=float)

    dmin = np.full(M, np.inf, dtype=float)
    S_emb = E[S_local_idx].astype(np.float32, copy=False)

    for start in range(0, M, chunk):
        stop = min(start + chunk, M)
        block = E[start:stop].astype(np.float32, copy=False)
        dot = block @ S_emb.T
        np.clip(dot, -1.0, 1.0, out=dot)
        cos_max = np.max(dot, axis=1)
        d_block = 1.0 - cos_max
        dmin[start:stop] = np.minimum(dmin[start:stop], d_block)

    return dmin


def _distance_matching_potential(
    embeddings_pool: np.ndarray,
    p_pool: np.ndarray,
    S_local: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    M = p_pool.shape[0]
    if S_local.size == 0 or S_local.size >= M:
        return np.zeros(M, dtype=float)

    p_pool = np.asarray(p_pool, dtype=float)
    gamma = float(np.clip(p_pool[S_local].sum(), eps, 1.0 - eps))

    pS = np.zeros_like(p_pool)
    pSc = np.zeros_like(p_pool)

    pS[S_local] = p_pool[S_local] / gamma
    mask_Sc = np.ones(M, dtype=bool)
    mask_Sc[S_local] = False
    if mask_Sc.any():
        pSc[mask_Sc] = p_pool[mask_Sc] / float(np.clip(1.0 - gamma, eps, 1.0))

    D = _pairwise_dist_embeddings(embeddings_pool)
    dS = D @ pS
    dSc = D @ pSc
    g = dSc - dS

    f_pool = np.min(g[None, :] + D, axis=1)
    return f_pool


def _topw_choose_subset_prefix(
    p_pool: np.ndarray,
    f_pool: np.ndarray,
    lam: float,
    beta: float,
    geom_scale: float,
    eps: float = 1e-12,
) -> np.ndarray:
    p_pool = np.asarray(p_pool, dtype=float)
    f_pool = np.asarray(f_pool, dtype=float)

    log_p = np.log(np.clip(p_pool, eps, 1.0))
    varphi = geom_scale * f_pool + lam * log_p

    order = np.argsort(-varphi)
    p_sorted = p_pool[order]
    varphi_sorted = varphi[order]

    mass_prefix = np.cumsum(p_sorted)
    val_prefix = np.cumsum(p_sorted * varphi_sorted)
    mass_safe = np.maximum(mass_prefix, eps)

    mean_varphi = val_prefix / mass_safe
    score = mean_varphi + (beta - lam) * np.log(mass_safe)

    k_star = int(np.argmax(score)) + 1
    keep = order[:k_star]
    keep = np.sort(keep)
    return keep


def _topw_mask_logits(
    logits: np.ndarray,
    embeddings_full: np.ndarray,
    mean_full: np.ndarray,
    scale_full: np.ndarray,
    temperature: float,
    top_m: int,
    init_top_p: float,
    alt_iters: int,
    geom_chunk: int,
    geom_scale: float,
    lam_fixed: float = 2.2,
    beta_override: Optional[float] = None,
    eps: float = 1e-12,
) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    V = logits.shape[0]

    T = float(max(temperature, eps))
    lam = float(lam_fixed)
    beta = float(beta_override) if beta_override is not None else 2.8

    p = _softmax_np(logits / T)

    M = int(min(max(1, top_m), V))
    I = np.argpartition(-p, M - 1)[:M]
    I = I[np.argsort(-p[I])]

    p_I = p[I].astype(np.float64, copy=False)

    E_I = embeddings_full[I].astype(np.float32, copy=False)
    E_Iw = (E_I - mean_full) * scale_full
    norm = np.linalg.norm(E_Iw, axis=1, keepdims=True).clip(1e-12)
    E_Iw = (E_Iw / norm).astype(np.float32, copy=False)

    mass_total = float(np.sum(p_I))
    target_mass = float(np.clip(init_top_p, 0.0, 1.0)) * mass_total
    csum = np.cumsum(p_I)
    k0 = int(np.searchsorted(csum, target_mass, side="left")) + 1
    k0 = max(1, min(k0, M))
    S_local = np.arange(k0, dtype=int)

    num_iters = max(1, int(alt_iters))
    for _ in range(num_iters):
        d_cos = _nearest_set_distance_cosine(E_Iw, S_local, chunk=geom_chunk)
        f_pool = -d_cos

        new_S = _topw_choose_subset_prefix(
            p_pool=p_I,
            f_pool=f_pool,
            lam=lam,
            beta=beta,
            geom_scale=geom_scale,
            eps=eps,
        )

        if new_S.size == S_local.size and np.array_equal(new_S, S_local):
            S_local = new_S
            break
        S_local = new_S

    keep_idx_global = I[S_local]

    masked = np.full_like(logits, -np.inf, dtype=np.float64)
    masked[keep_idx_global] = logits[keep_idx_global]
    return masked


class TopW_LogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        embeddings,
        selection_temperature: float = 1.0,
        top_m: int = 1200,
        init_top_p: float = 0.999,
        alt_iters: int = 9,
        geom_chunk: int = 4096,
        geom_scale: float = 0.6,
        lam_fixed: float = 2.2,
        beta: float = 2.8,
        print_params: Optional[bool] = None,
        print_kept: Optional[bool] = None,
        debug_steps: Optional[int] = None,
        print_geom_stats: Optional[bool] = None,
        **kwargs,
    ):
        if "temperature" in kwargs:
            selection_temperature = float(kwargs["temperature"])
        if "warm_p" in kwargs:
            init_top_p = float(kwargs["warm_p"])
        if "alt_iters" in kwargs:
            alt_iters = int(kwargs["alt_iters"])
        if "top_m" in kwargs:
            top_m = int(kwargs["top_m"])
        if "geom_chunk" in kwargs:
            geom_chunk = int(kwargs["geom_chunk"])
        if "geom_scale" in kwargs:
            geom_scale = float(kwargs["geom_scale"])
        if "lambda_geom" in kwargs:
            lam_fixed = float(kwargs["lambda_geom"])
        if "beta" in kwargs:
            beta = float(kwargs["beta"])

        self._beta_override = float(beta)

        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().to("cpu").float().numpy()
        if not isinstance(embeddings, np.ndarray):
            raise TypeError(f"embeddings must be torch.Tensor or np.ndarray, got {type(embeddings)}")

        E = embeddings.astype(np.float32, copy=False)
        if E.ndim != 2:
            raise ValueError(f"embeddings must be [V,d], got shape {E.shape}")

        self.embeddings = E

        mean = E.mean(axis=0, keepdims=True).astype(np.float32, copy=False)
        centered = E - mean
        var = np.mean(centered ** 2, axis=0, keepdims=True).astype(np.float32, copy=False)
        scale = (1.0 / np.sqrt(np.clip(var, 1e-6, None))).astype(np.float32, copy=False)

        self._mean_full = mean
        self._scale_full = scale

        if selection_temperature <= 0:
            raise ValueError("selection_temperature must be > 0.")
        self.selection_temperature = float(selection_temperature)

        self.top_m = int(top_m)
        if self.top_m < 1:
            raise ValueError("top_m must be >= 1.")

        if not (0.0 < init_top_p <= 1.0):
            raise ValueError("init_top_p must be in (0,1].")
        self.init_top_p = float(init_top_p)

        self.alt_iters = int(alt_iters)
        if self.alt_iters < 1:
            raise ValueError("alt_iters must be >= 1.")

        self.geom_chunk = int(geom_chunk)
        if self.geom_chunk < 1:
            raise ValueError("geom_chunk must be >= 1.")

        self.geom_scale = float(geom_scale)
        if not np.isfinite(self.geom_scale):
            raise ValueError("geom_scale must be finite.")

        self.lam_fixed = float(lam_fixed)

        if print_params is None:
            print_params = os.environ.get("TOPW_PRINT_PARAMS", "1") == "1"
        if print_kept is None:
            print_kept = os.environ.get("TOPW_PRINT_KEPT", "1") == "1"
        if debug_steps is None:
            debug_steps = int(os.environ.get("TOPW_DEBUG_STEPS", "3"))
        if print_geom_stats is None:
            print_geom_stats = os.environ.get("TOPW_PRINT_GEOM_STATS", "0") == "1"

        self._print_params = bool(print_params)
        self._print_kept = bool(print_kept)
        self._print_geom_stats = bool(print_geom_stats)
        self._debug_steps = int(debug_steps)
        self._dbg_calls = 0

        self.lam = float(self.lam_fixed)
        self.beta = float(self._beta_override)
        self.temperature = float(self.selection_temperature)

        self.lambda_geom = float(self.lam)
        self.warm_p = float(self.init_top_p)
        self.m = int(self.top_m)
        self.geom_mode = os.environ.get("TOPW_GEOM_MODE", "G")

        if self._print_params:
            V, d = self.embeddings.shape
            print(
                f"[TopW:new:init] lam={self.lam_fixed}, beta={self.beta:.6g}, "
                f"T_sel=T={self.selection_temperature}, top_m={self.top_m}, warm_p={self.init_top_p}, "
                f"alt_iters={self.alt_iters}, geom_mode=G(cos), geom_scale={self.geom_scale}, "
                f"E_shape=({V},{d})",
                flush=True,
            )

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size, vocab_size = scores.shape
        if batch_size != 1:
            raise ValueError("TopW_LogitsProcessor currently supports batch_size == 1 only.")

        logits_np = scores.detach().cpu().numpy()[0]

        masked_np = _topw_mask_logits(
            logits=logits_np,
            embeddings_full=self.embeddings,
            mean_full=self._mean_full,
            scale_full=self._scale_full,
            temperature=self.selection_temperature,
            top_m=self.top_m,
            init_top_p=self.init_top_p,
            alt_iters=self.alt_iters,
            geom_chunk=self.geom_chunk,
            geom_scale=self.geom_scale,
            lam_fixed=self.lam_fixed,
            beta_override=self.beta,
        )

        masked_scores = torch.tensor(
            masked_np,
            dtype=scores.dtype,
            device=scores.device,
        ).unsqueeze(0)

        if (self._print_kept or self._print_geom_stats) and (self._dbg_calls < self._debug_steps):
            kept = int(np.isfinite(masked_np).sum())
            V = masked_np.size
            if self._print_kept:
                print(
                    f"[TopW:new:step={self._dbg_calls}] kept={kept}/{V} "
                    f"(top_m={self.top_m}, warm_p={self.init_top_p}, alt_iters={self.alt_iters}, "
                    f"lam={self.lam_fixed}, beta={self.beta:.6g}, geom_scale={self.geom_scale})",
                    flush=True,
                )

            if self._print_geom_stats:
                T = max(self.selection_temperature, 1e-12)
                p = _softmax_np(np.asarray(logits_np, dtype=np.float64) / T)
                M = int(min(max(1, self.top_m), V))
                I = np.argpartition(-p, M - 1)[:M]
                I = I[np.argsort(-p[I])]
                p_I = p[I]
                logp = np.log(np.clip(p_I, 1e-12, 1.0))
                prob_scale = float(np.std(self.lam_fixed * logp))
                print(
                    f"[TopW:new:geom_stats step={self._dbg_calls}] "
                    f"std(lambda*logp)~{prob_scale:.4g} (geom_scale={self.geom_scale})",
                    flush=True,
                )

            self._dbg_calls += 1

        return masked_scores
