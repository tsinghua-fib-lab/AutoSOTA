from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Optional, Tuple, List, Sequence
import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from ..constants import (
    RW_SANITY_SPLIT_FRACS,
    RW_SANITY_M_CORE,
    RW_SANITY_M_SPUR,
    RW_SANITY_P_MATCH_CLEAN,
    RW_SANITY_P_MATCH_SHIFT,
    RW_SANITY_SHIFT_SCALE,
    RW_DEFAULT_LR,
    RW_DEFAULT_EPOCHS,
    RW_DEFAULT_BATCH_SIZE,
    RW_DEFAULT_WEIGHT_DECAY,
)
from .array_dataset import MemmapFeatureDataset
from .text_pipeline import collate_hashing, ensure_text_feature_cache, ensure_text_score_cache, make_vectorizer
from .trainers import evaluate, set_seed, train_head
from .wilds_data import prepare_wilds_splits
from .heads import HeadConfig
from .tuning import ERMTuningConfig, get_or_tune_frozen_erm_hparams
from types import SimpleNamespace

REAL_WORLD_DATASETS = {
    "rw_civilcomments",
    "civilcomments",
}


def _diag_unique(name, arr):
    arr = np.asarray(arr).reshape(-1)
    u, c = np.unique(arr, return_counts=True)
    print(f"[DIAG] {name} unique={u.tolist()} counts={c.tolist()} (num_unique={len(u)} n={arr.size})")
    return u, c

def _diag_group_semantics_from_y_g(y, g, n_groups, prefix=""):
    y = np.asarray(y).astype(int).reshape(-1)
    g = np.asarray(g).astype(int).reshape(-1)
    print(f"[DIAG] {prefix} checking group semantics for n_groups={n_groups}")
    for gid in range(n_groups):
        m = (g == gid)
        n = int(m.sum())
        if n == 0:
            print(f"  gid={gid}: n=0")
            continue
        yu = np.unique(y[m])
        if yu.size != 1:
            print(f"  gid={gid}: n={n} y_not_constant={yu.tolist()}  <-- BUG (groups should be label-conditioned)")
            continue
        yv = int(yu[0])
        implied_idany = int(gid - 2*yv)
        print(f"  gid={gid}: n={n} y={yv} implied_identity_any={implied_idany}")
        if implied_idany not in (0, 1):
            print(f"    <-- BUG: implied_identity_any not in {{0,1}}; g is not encoding (y,id_any) as intended")

def _diag_constant_baselines(y, g, n_groups, prefix=""):
    y = np.asarray(y).astype(int).reshape(-1)
    g = np.asarray(g).astype(int).reshape(-1)

    def _conf(y_true, y_pred, name):
        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred).astype(int)
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        acc = (tn + tp) / max(1, y_true.size)
        pred_pos = float((y_pred == 1).mean()) if y_true.size else float("nan")
        print(f"[DIAG] {prefix} {name}: acc={acc:.6f} pred_pos_rate={pred_pos:.6f} tn={tn} fp={fp} fn={fn} tp={tp}")
        # group-wise
        for gid in range(n_groups):
            m = (g == gid)
            if m.sum() == 0:
                continue
            tn_g = int(((y[m] == 0) & (y_pred[m] == 0)).sum())
            fp_g = int(((y[m] == 0) & (y_pred[m] == 1)).sum())
            fn_g = int(((y[m] == 1) & (y_pred[m] == 0)).sum())
            tp_g = int(((y[m] == 1) & (y_pred[m] == 1)).sum())
            acc_g = (tn_g + tp_g) / max(1, int(m.sum()))
            print(f"    gid={gid}: n={int(m.sum())} acc={acc_g:.6f} (tn={tn_g} fp={fp_g} fn={fn_g} tp={tp_g})")

    _conf(y, np.zeros_like(y), "ALWAYS_PRED_0")
    _conf(y, np.ones_like(y), "ALWAYS_PRED_1")
# ============================================================================

def _diag_feature_matrix(X, name, sample_n=2000, seed=0):
    X = np.asarray(X)  # handles memmap transparently
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(sample_n, n), replace=False)
    Xs = np.asarray(X[idx], dtype=np.float32)
    nnz = np.count_nonzero(Xs, axis=1)
    norms = np.linalg.norm(Xs, axis=1)

    print(f"[DIAG] {name}: shape={X.shape} dtype={X.dtype}")
    print(f"       zero_rows_frac={(nnz==0).mean():.6f}  nnz_mean={nnz.mean():.1f}  nnz_p10={np.percentile(nnz,10):.0f}  nnz_p90={np.percentile(nnz,90):.0f}")
    print(f"       l2norm_mean={norms.mean():.6f}  l2norm_p10={np.percentile(norms,10):.6f}  l2norm_p90={np.percentile(norms,90):.6f}")

def _diag_index_subset(name, y, g, idx, n_groups):
    y = np.asarray(y).astype(int).reshape(-1)
    g = np.asarray(g).astype(int).reshape(-1)
    idx = np.asarray(idx).astype(int).reshape(-1)

    ys = y[idx]
    gs = g[idx]
    print(f"[DIAG] {name}: n={idx.size} frac={idx.size / max(1, y.size):.6f}")
    _diag_unique(f"{name}.y", ys)
    _diag_unique(f"{name}.g", gs)

    # Group-wise within subset
    for gid in range(n_groups):
        m = (gs == gid)
        if m.sum() == 0:
            continue
        print(f"    gid={gid}: n={int(m.sum())}")

@torch.no_grad()
def _diag_model_preds_binary(
    model,
    loader,
    *,
    device: torch.device,
    n_groups: int,
    prefix: str,
    max_batches: int = 50,
) -> None:
    model.eval()

    tn = fp = fn = tp = 0
    total = 0

    g_tn = np.zeros((n_groups,), dtype=np.int64)
    g_fp = np.zeros((n_groups,), dtype=np.int64)
    g_fn = np.zeros((n_groups,), dtype=np.int64)
    g_tp = np.zeros((n_groups,), dtype=np.int64)
    g_n  = np.zeros((n_groups,), dtype=np.int64)

    logit_min = float("inf")
    logit_max = float("-inf")
    logit_sum = 0.0
    logit_count = 0

    for b, (xb, yb, gb, _idx) in enumerate(loader):
        if max_batches is not None and b >= int(max_batches):
            break

        xb = xb.to(device)
        y = yb.to(device).long().view(-1)
        g = gb.to(device).long().view(-1)

        logits = model(xb)

        # ---- convert logits -> predicted label p in {0,1} ----
        if logits.ndim == 1 or logits.shape[-1] == 1:
            # binary logit (LV-BAS-Bin)
            logit = logits.view(-1)
            p = (logit > 0).long()
            # logit stats
            logit_min = min(logit_min, float(logit.min().item()))
            logit_max = max(logit_max, float(logit.max().item()))
            logit_sum += float(logit.sum().item())
            logit_count += int(logit.numel())

        elif logits.ndim == 2 and logits.shape[1] == 2:
            # 2-class CE head
            p = torch.argmax(logits, dim=1).long()
            # logit stats: use margin logit1-logit0 as a scalar diagnostic
            margin = (logits[:, 1] - logits[:, 0]).view(-1)
            logit_min = min(logit_min, float(margin.min().item()))
            logit_max = max(logit_max, float(margin.max().item()))
            logit_sum += float(margin.sum().item())
            logit_count += int(margin.numel())

        else:
            raise RuntimeError(f"[DIAG] {prefix}: unexpected logits shape {tuple(logits.shape)}")

        # ---- overall confusion ----
        y0 = (y == 0); y1 = ~y0
        p0 = (p == 0); p1 = ~p0

        tn_b = int((y0 & p0).sum().item())
        fp_b = int((y0 & p1).sum().item())
        fn_b = int((y1 & p0).sum().item())
        tp_b = int((y1 & p1).sum().item())

        tn += tn_b; fp += fp_b; fn += fn_b; tp += tp_b
        total += int(y.numel())

        # ---- per-group confusion ----
        for gid in range(n_groups):
            m = (g == gid)
            if not bool(m.any()):
                continue
            yg = y[m]; pg = p[m]
            g_n[gid] += int(yg.numel())

            y0g = (yg == 0); y1g = ~y0g
            p0g = (pg == 0); p1g = ~p0g
            g_tn[gid] += int((y0g & p0g).sum().item())
            g_fp[gid] += int((y0g & p1g).sum().item())
            g_fn[gid] += int((y1g & p0g).sum().item())
            g_tp[gid] += int((y1g & p1g).sum().item())

    if total == 0:
        print(f"[DIAG] {prefix}: empty loader?")
        return

    acc = (tn + tp) / total
    pred_pos_rate = (fp + tp) / total
    true_pos_rate = (fn + tp) / total

    mean_logit = (logit_sum / max(1, logit_count)) if logit_count else float("nan")

    print(f"[DIAG] {prefix}: n={total} acc={acc:.6f} pred_pos_rate={pred_pos_rate:.6f} true_pos_rate={true_pos_rate:.6f}")
    print(f"[DIAG] {prefix}: margin_min={logit_min:.4f} margin_mean={mean_logit:.4f} margin_max={logit_max:.4f}")

    for gid in range(n_groups):
        n = int(g_n[gid])
        if n == 0:
            continue
        tn_g = int(g_tn[gid]); fp_g = int(g_fp[gid]); fn_g = int(g_fn[gid]); tp_g = int(g_tp[gid])
        acc_g = (tn_g + tp_g) / n
        tpr_g = tp_g / max(1, (tp_g + fn_g))
        tnr_g = tn_g / max(1, (tn_g + fp_g))
        pred_pos_g = (fp_g + tp_g) / n
        true_pos_g = (fn_g + tp_g) / n
        print(
            f"  gid={gid}: n={n} acc={acc_g:.6f} pred_pos={pred_pos_g:.6f} true_pos={true_pos_g:.6f} "
            f"TPR={tpr_g:.6f} TNR={tnr_g:.6f} (tn={tn_g} fp={fp_g} fn={fn_g} tp={tp_g})"
        )


def is_real_world_dataset(dataset: str) -> bool:
    return dataset.lower() in REAL_WORLD_DATASETS or dataset.startswith("rw_")


def _sync_device(device: torch.device) -> None:
    # Make CUDA timings meaningful (CUDA kernels are async by default).
    if device.type == "cuda":
        torch.cuda.synchronize()


@contextmanager
def _timed(timings: Dict[str, float], key: str, device: torch.device):
    _sync_device(device)
    t0 = perf_counter()
    try:
        yield
    finally:
        _sync_device(device)
        timings[key] = timings.get(key, 0.0) + (perf_counter() - t0)


def _device(dev: str) -> torch.device:
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


def _loader(ds, bs: int, nw: int, shuffle: bool, collate_fn=None) -> DataLoader:
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=nw, collate_fn=collate_fn)


def _quantile_threshold(scores_cal: np.ndarray, gamma: float, delta: float = 0.05) -> float:
    """
    τ_γ = DKW-certified empirical quantile of calibration scores.

    We use the DKW correction:
        q = 1 - gamma + sqrt(log(2/delta) / (2m)),
    where m is the number of calibration samples and delta is fixed to 0.05.

    We use method='higher' so empirical bulk coverage is >= q.
    If q > 1, no DKW certificate exists for the requested (gamma, delta, m):
      - we return max(scores_cal) (equivalently q=1),
      - and raise a warning indicating the smallest certifiable gamma.
    """
    if not (0.0 <= gamma < 1.0):
        raise ValueError(f"gamma must be in [0,1), got {gamma}")

    delta = 0.05
    if not (0.0 < float(delta) < 1.0):
        raise ValueError(f"delta must be in (0,1), got {delta}")

    scores_cal = np.asarray(scores_cal, dtype=float).reshape(-1)
    m = int(scores_cal.size)
    if m <= 0:
        raise ValueError("scores_cal must be non-empty.")

    eps_dkw = float(np.sqrt(np.log(2.0 / float(delta)) / (2.0 * float(m))))
    q = 1.0 - float(gamma) + eps_dkw

    if q > 1.0:
        r = eps_dkw  # smallest certifiable gamma (so that q <= 1)
        warnings.warn(
            f"DKW-certificate doesn't exist for gamma={float(gamma):g}, delta={float(delta):g}, m={m} "
            f"(requested q={q:.6f} > 1). Using tau=max(scores_cal). "
            f"Smallest certifiable gamma is {r:.6f}."
        )
        return float(np.max(scores_cal))

    # numerical safety (should already be <= 1 here)
    q = float(min(max(q, 0.0), 1.0))

    try:
        return float(np.quantile(scores_cal, q=q, method="higher"))
    except TypeError:
        return float(np.quantile(scores_cal, q=q, interpolation="higher"))


@torch.no_grad()
def _collect_binary_margins(
    model,
    loader,
    *,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect binary margins m(x) on a loader:
      - if logits are (N,2): m = logit1 - logit0
      - if logits are (N,) or (N,1): m = logit
    Returns (margins, y, g) as numpy arrays.
    """
    model.eval()
    ms: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    gs: List[np.ndarray] = []

    for xb, yb, gb, _idx in loader:
        xb = xb.to(device)
        logits = model(xb)

        if logits.ndim == 1 or logits.shape[-1] == 1:
            m = logits.view(-1)
        elif logits.ndim == 2 and logits.shape[1] == 2:
            m = (logits[:, 1] - logits[:, 0]).view(-1)
        else:
            raise RuntimeError(f"unexpected logits shape {tuple(logits.shape)} in _collect_binary_margins")

        ms.append(m.detach().cpu().numpy())
        ys.append(yb.numpy())
        gs.append(gb.numpy())

    margins = np.concatenate(ms, axis=0).astype(np.float64, copy=False)
    y = np.concatenate(ys, axis=0).astype(np.int64, copy=False)
    g = np.concatenate(gs, axis=0).astype(np.int64, copy=False)
    return margins, y, g


def _calibrate_threshold_max_worst_group_acc(
    margins: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    *,
    n_groups: int,
    grid: int = 401,
) -> Tuple[float, Dict[str, float]]:
    """
    Pick threshold t maximising worst-group accuracy on validation:
      yhat = 1[margins > t]
    Uses a quantile grid over margins.
    """
    margins = np.asarray(margins, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    g = np.asarray(g, dtype=np.int64).reshape(-1)
    if not (margins.shape[0] == y.shape[0] == g.shape[0]):
        raise ValueError("margins/y/g must have the same length")

    qs = np.linspace(0.0, 1.0, int(grid))
    cand = np.quantile(margins, qs)
    cand = np.unique(cand)

    # include extremes (allow constant predictors if they end up optimal)
    eps = 1e-6
    cand = np.unique(np.concatenate([cand, [float(margins.min() - eps), float(margins.max() + eps)]]))

    best_t = float(cand[0])
    best_worst = -1.0
    best_avg = -1.0

    for t in cand:
        pred = (margins > float(t)).astype(np.int64)

        worst = float("inf")
        acc_sum = 0.0
        used = 0

        for gid in range(int(n_groups)):
            m = (g == gid)
            n = int(m.sum())
            if n == 0:
                continue
            acc = float(np.mean(pred[m] == y[m]))
            worst = min(worst, acc)
            acc_sum += acc
            used += 1

        if used == 0:
            continue
        avg = acc_sum / float(used)

        # primary: maximise worst; tie-break: maximise avg
        if (worst > best_worst + 1e-12) or (abs(worst - best_worst) <= 1e-12 and avg > best_avg + 1e-12):
            best_worst = float(worst)
            best_avg = float(avg)
            best_t = float(t)

    info = {
        "val_worst_group_acc_at_t": float(best_worst),
        "val_avg_group_acc_at_t": float(best_avg),
        "n_threshold_candidates": float(cand.size),
    }
    return best_t, info


class _ShiftedLogitsModel(torch.nn.Module):
    """
    Wrap a trained model and apply a decision-threshold shift at evaluation time.

    For 2-logit models: subtract t from class-1 logit => argmax implements margin > t.
    For single-logit models: subtract t => sign implements logit > t.
    """
    def __init__(self, base, threshold: float):
        super().__init__()
        self.base = base
        self.threshold = float(threshold)

    def forward(self, x):
        logits = self.base(x)
        t = float(self.threshold)

        if logits.ndim == 1 or logits.shape[-1] == 1:
            return logits - t

        if logits.ndim == 2 and logits.shape[1] == 2:
            out = logits.clone()
            out[:, 1] = out[:, 1] - t
            return out

        return logits

def _maybe_threshold_calibrate_and_wrap(
    model,
    val_loader,
    *,
    device: torch.device,
    n_classes: int,
    n_groups: int,
    timings: Optional[Dict[str, float]] = None,
    timings_key: str = "runtime_threshold_calibrate_s",
    grid: int = 401,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    If binary (n_classes==2), calibrate a single decision threshold t on VAL to maximise
    worst-group accuracy, and return a wrapped model that applies the shift at inference.

    Returns (model_for_eval, info_dict). If not binary, returns (model, {}).
    """
    if int(n_classes) != 2:
        return model, {}

    if timings is None:
        margins, yv, gv = _collect_binary_margins(model, val_loader, device=device)
        t_star, t_info = _calibrate_threshold_max_worst_group_acc(
            margins, yv, gv, n_groups=int(n_groups), grid=int(grid)
        )
    else:
        with _timed(timings, str(timings_key), device):
            margins, yv, gv = _collect_binary_margins(model, val_loader, device=device)
            t_star, t_info = _calibrate_threshold_max_worst_group_acc(
                margins, yv, gv, n_groups=int(n_groups), grid=int(grid)
            )

    model_eval = _ShiftedLogitsModel(model, threshold=float(t_star))
    info = {
        "rw_decision_threshold": float(t_star),
        "rw_val_worst_group_acc_at_t": float(t_info["val_worst_group_acc_at_t"]),
        "rw_val_avg_group_acc_at_t": float(t_info["val_avg_group_acc_at_t"]),
        "rw_n_threshold_candidates": int(t_info["n_threshold_candidates"]),
    }
    return model_eval, info


def _scores_binary_perclass_diag_mahalanobis_dense(
    X: np.ndarray,
    mu: np.ndarray,
    std: np.ndarray,
    *,
    ridge: float,
    chunk_size: int = 8192,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Label-free per-class diagonal Mahalanobis distances for binary (C=2):
      d0(x) = || (x - mu0) / sqrt(std0^2 + ridge) ||_2
      d1(x) = || (x - mu1) / sqrt(std1^2 + ridge) ||_2
    Returns (d0, d1) as float32 arrays of shape (N,).
    """
    X = np.asarray(X, dtype=np.float32)
    mu = np.asarray(mu, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)

    if mu.shape[0] != 2 or std.shape[0] != 2:
        raise ValueError(f"binary per-class scores require mu/std with first dim=2; got mu={mu.shape}, std={std.shape}")

    sigma = np.sqrt(std * std + float(ridge)).astype(np.float32)

    n = int(X.shape[0])
    d0 = np.empty((n,), dtype=np.float32)
    d1 = np.empty((n,), dtype=np.float32)

    for start in range(0, n, int(chunk_size)):
        end = min(n, start + int(chunk_size))
        Xb = np.asarray(X[start:end], dtype=np.float32)

        z0 = (Xb - mu[0]) / sigma[0]
        z1 = (Xb - mu[1]) / sigma[1]

        d0[start:end] = np.sqrt(np.sum(z0 * z0, axis=1, dtype=np.float32)).astype(np.float32)
        d1[start:end] = np.sqrt(np.sum(z1 * z1, axis=1, dtype=np.float32)).astype(np.float32)

    return d0, d1

def _class_cond_thresholds(
    scores_cal_true: np.ndarray,
    y_cal: np.ndarray,
    *,
    gamma: float,
    n_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    tau_y := (1-gamma)-quantile of calibration scores within each class.
    Returns (tau, coverage) where coverage_y = P_cal[s_y(x) <= tau_y | y].
    """
    y_cal = np.asarray(y_cal, dtype=np.int64).reshape(-1)
    scores_cal_true = np.asarray(scores_cal_true, dtype=np.float32).reshape(-1)

    tau = np.zeros((int(n_classes),), dtype=np.float32)
    cov = np.zeros((int(n_classes),), dtype=np.float32)
    for c in range(int(n_classes)):
        sc = scores_cal_true[y_cal == c]
        if sc.size == 0:
            tau[c] = np.float32(np.inf)
            cov[c] = np.float32(np.nan)
            continue
        tau[c] = np.float32(_quantile_threshold(sc, float(gamma)))
        cov[c] = np.float32(np.mean(sc <= tau[c]))
    return tau, cov


def _collate_text_only(batch):
    """
    Collate without vectorising: returns raw texts + y/g as tensors.
    Assumes each dataset item is (text, y, g, idx).
    """
    texts = [str(x[0]) for x in batch]
    y = torch.tensor([int(x[1]) for x in batch], dtype=torch.long)
    g = torch.tensor([int(x[2]) for x in batch], dtype=torch.long)
    idx = torch.tensor([int(x[3]) for x in batch], dtype=torch.long)
    return texts, y, g, idx


def _ensure_dense_hash_features(
    *,
    ds,
    vectorizer,
    cache_dir: Path,
    split_name: str,
    batch_size: int,
    num_workers: int,
    verbose: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (once) and load (thereafter) dense hashed features for a split:
      X: float32 memmap (N, D) saved as .npy
      y: int64 (N,)
      g: int64 (N,)

    Returned arrays are in the *split order* (same order as DataLoader shuffle=False).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    X_path = cache_dir / f"{split_name}.X.npy"
    y_path = cache_dir / f"{split_name}.y.npy"
    g_path = cache_dir / f"{split_name}.g.npy"

    if X_path.exists() and y_path.exists() and g_path.exists():
        X = np.load(X_path, mmap_mode="r")
        y = np.load(y_path).astype(np.int64, copy=False)
        g = np.load(g_path).astype(np.int64, copy=False)

        # IndexedWILDSSubset stores full-length _g and a subset index array `indices`.
        if hasattr(ds, "_g") and hasattr(ds, "indices"):
            g_expected = np.asarray(getattr(ds, "_g"))[np.asarray(getattr(ds, "indices"))].astype(np.int64)
            if g.shape != g_expected.shape or (not np.array_equal(g, g_expected)):
                np.save(g_path, g_expected)
                g = g_expected
        return X, y, g

    N = int(len(ds))
    D = int(getattr(vectorizer, "n_features"))

    if verbose:
        print(f"[civilcomments] caching dense hash features: split={split_name} N={N} D={D} -> {X_path}")

    X_mm = np.lib.format.open_memmap(X_path, mode="w+", dtype=np.float32, shape=(N, D))
    y = np.empty((N,), dtype=np.int64)
    g = np.empty((N,), dtype=np.int64)

    loader = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=_collate_text_only,
        pin_memory=False,
    )

    pos = 0
    for texts, yb, gb, _idx in loader:
        X_sp = vectorizer.transform(texts)  # scipy sparse
        Xb = X_sp.toarray().astype(np.float32, copy=False)
        b = int(Xb.shape[0])

        X_mm[pos : pos + b] = Xb
        y[pos : pos + b] = yb.numpy()
        g[pos : pos + b] = gb.numpy()
        pos += b

    if pos != N:
        raise RuntimeError(f"dense feature cache write mismatch for split={split_name}: wrote {pos}, expected {N}")

    # Flush memmap to disk
    del X_mm
    np.save(y_path, y)
    np.save(g_path, g)

    X = np.load(X_path, mmap_mode="r")
    return X, y, g

def _run_civilcomments(
    *,
    algorithm: str,
    epsilon: float,
    gamma: float,
    replication: int,
    dataset_dir: Path,
    cache_root: Path,
    split_seed: int,
    cal_fraction: float,
    device: torch.device,
    n_features: int,
    ngram_range: Tuple[int, int],
    head: str,
    mlp_hidden: int,
    train_bs: int,
    train_nw: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    smoothmax_T: float,
    groupdro_step: float,
    max_grad_norm: Optional[float],
    verbose: bool = False,
    do_threshold_calibration: bool = False,
    checkpoint_selection: str = "val",          # "none" | "val" | "oracle_test"
    checkpoint_metric: str = "loss_mean",  # e.g. "worst_group_acc" or "acc_mean" or "loss_mean"
) -> Dict[str, Any]:
    verbose = verbose and replication == 0
    dataset_tag = "civilcomments"
    timings: Dict[str, float] = {}
    _sync_device(device)
    t_total0 = perf_counter()

    with _timed(timings, "runtime_prepare_splits_s", device):
        splits = prepare_wilds_splits(
            wilds_name="civilcomments",
            root_dir=dataset_dir / "wilds",
            cache_dir=cache_root / "splits",
            split_seed=split_seed,
            cal_fraction=cal_fraction,
            download=True,
            transform=None,
        )
    raw = {"train": splits.train, "cal": splits.cal, "val": splits.val, "test": splits.test}
    raw_splits = raw

    alg = algorithm.lower()
    # ------------------------------------------------------------------
    #   - Non-LV baselines train on FULL official_train without bulk filtering.
    #   - LV methods treat bulk calibration as LV-internal and train on the bulk-filtered official_train.
    #   - Bulk-filtered ablation baselines (ERM-B / CVaR-B / chi2-DRO-B) use the same bulk-filtered official_train as the LV methods.
    #   - "ERM-bulk" is kept only as an alias of ERM for backwards compatibility.
    # ------------------------------------------------------------------
    is_lv_bulk_method = alg in (
        "rw_lv_empirical", "lv_empirical",
        "rw_lv_empirical_fair", "lv_empirical_fair",
    )
    is_bulk_filtered_baseline = alg in (
        "rw_erm_b", "erm_b",
        "rw_cvar_b", "cvar_b",
        "rw_chi2_dro_b", "chi2_dro_b",
    )
    is_erm_bulk_baseline = alg in ("rw_erm_bulk", "erm_bulk")

    # LV methods and the explicit bulk-filtered ablation baselines train on bulk-filtered data.
    train_on_bulk = bool(is_lv_bulk_method or is_bulk_filtered_baseline)

    # Backwards-compatibility alias only.
    train_algorithm = ("rw_erm" if is_erm_bulk_baseline else algorithm)
    with _timed(timings, "runtime_ensure_text_score_cache_s", device):
        cache = ensure_text_score_cache(
            dataset_tag=dataset_tag,
            splits=raw_splits,
            cache_root=cache_root,
            n_features=int(n_features),
            ngram_range=ngram_range,
            split_seed=int(split_seed),
            cal_fraction=float(cal_fraction),
            batch_size=int(train_bs),
            num_workers=int(train_nw),
            verbose=verbose,
            n_classes=int(splits.n_classes),
            ridge=0.0,
            need_trueclass_scores=False,
        )

    with _timed(timings, "runtime_load_scores_s", device):
        scores = cache.load_scores()
        s_tr, s_cal, s_te = scores["train"], scores["cal"], scores["test"]
    
    with _timed(timings, "runtime_ensure_text_feature_cache_s", device):
        vec = make_vectorizer(n_features=int(n_features), ngram_range=ngram_range)
        feat_dir = cache.cache_dir / "dense_features"

        Xtr_dense, ytr_dense, gtr_dense = _ensure_dense_hash_features(
            ds=splits.train,
            vectorizer=vec,
            cache_dir=feat_dir,
            split_name="train",
            batch_size=int(train_bs),
            num_workers=int(train_nw),
            verbose=verbose,
        )
        Xcal_dense, ycal_dense, gcal_dense = _ensure_dense_hash_features(
            ds=splits.cal,
            vectorizer=vec,
            cache_dir=feat_dir,
            split_name="cal",
            batch_size=int(train_bs),
            num_workers=int(train_nw),
            verbose=verbose,
        )
        Xva_dense, yva_dense, gva_dense = _ensure_dense_hash_features(
            ds=splits.val,
            vectorizer=vec,
            cache_dir=feat_dir,
            split_name="val",
            batch_size=int(train_bs),
            num_workers=int(train_nw),
            verbose=verbose,
        )
        Xte_dense, yte_dense, gte_dense = _ensure_dense_hash_features(
            ds=splits.test,
            vectorizer=vec,
            cache_dir=feat_dir,
            split_name="test",
            batch_size=int(train_bs),
            num_workers=int(train_nw),
            verbose=verbose,
        )
    if verbose:
        print(
            f"[civilcomments] split sizes: n_train={int(len(ytr_dense))} n_cal={int(len(ycal_dense))} "
            f"n_official_train={int(len(ytr_dense) + len(ycal_dense))} n_val={int(len(yva_dense))} n_test={int(len(yte_dense))}"
        )

        def _u(x):
            x = np.asarray(x, dtype=np.int64).reshape(-1)
            u, c = np.unique(x, return_counts=True)
            # show up to first 20 groups/classes to avoid huge prints
            k = min(20, u.size)
            return u[:k].tolist(), c[:k].tolist(), int(u.size)

        ug_tr, cg_tr, ng_tr = _u(gtr_dense)
        ug_va, cg_va, ng_va = _u(gva_dense)
        ug_te, cg_te, ng_te = _u(gte_dense)

        print(f"[civilcomments] splits.n_groups={int(splits.n_groups)}")
        print(f"[civilcomments] g_train unique={ug_tr} counts={cg_tr} (num_unique={ng_tr})")
        print(f"[civilcomments] g_val   unique={ug_va} counts={cg_va} (num_unique={ng_va})")
        print(f"[civilcomments] g_test  unique={ug_te} counts={cg_te} (num_unique={ng_te})")

        uy_tr, cy_tr, ny_tr = _u(ytr_dense)
        uy_va, cy_va, ny_va = _u(yva_dense)
        uy_te, cy_te, ny_te = _u(yte_dense)
        print(f"[civilcomments] y_train unique={uy_tr} counts={cy_tr} (num_unique={ny_tr})")
        print(f"[civilcomments] y_val   unique={uy_va} counts={cy_va} (num_unique={ny_va})")
        print(f"[civilcomments] y_test  unique={uy_te} counts={cy_te} (num_unique={ny_te})")
        print(f"[civilcomments] splits.train={splits.train}")

        n_groups = int(splits.n_groups)
        _diag_group_semantics_from_y_g(ytr_dense, gtr_dense, n_groups, prefix="train")
        _diag_group_semantics_from_y_g(yva_dense, gva_dense, n_groups, prefix="val")
        _diag_group_semantics_from_y_g(yte_dense, gte_dense, n_groups, prefix="test")

        # Baselines: these tell you immediately whether any method is just collapsing to a constant classifier
        _diag_constant_baselines(yte_dense, gte_dense, n_groups, prefix="test")
        _diag_feature_matrix(Xtr_dense, "Xtr_dense")
        _diag_feature_matrix(Xva_dense, "Xva_dense")
        _diag_feature_matrix(Xte_dense, "Xte_dense")
    # ------------------------------------------------------------------
    # CivilComments: build split-local metadata arrays so `evaluate()` can
    # compute WILDS-style 16 overlapping identity×label slices.
    #
    # We attach a proxy object `base_subset.metadata_array` to the *memmap*
    # datasets (val/test/test_bulk), because those datasets otherwise have
    # no access to WILDS metadata.
    # ------------------------------------------------------------------

    def _unwrap_ds(ds_obj):
        base = ds_obj
        while isinstance(base, Subset):
            base = base.dataset
        return base

    def _get_meta_source(ds_obj):
        """
        Return an object that has:
          - metadata_array
          - (optional) metadata_fields
        """
        base = _unwrap_ds(ds_obj)
        if hasattr(base, "base_subset") and hasattr(base.base_subset, "metadata_array"):
            return base.base_subset
        if hasattr(base, "metadata_array"):
            return base
        return None

    def _split_local_meta(split_ds, *, split_name: str) -> tuple[np.ndarray, list[str] | None]:
        """
        Build a metadata_array aligned with `split_ds` order (length == len(split_ds)).
        """
        src = _get_meta_source(split_ds)
        if src is None or (not hasattr(src, "metadata_array")):
            raise RuntimeError(
                f"[civilcomments] Could not locate metadata_array for split={split_name}. "
                "Cannot compute 16-slice identity×label worst-group accuracy."
            )

        meta_base = src.metadata_array
        if torch.is_tensor(meta_base):
            meta_base = meta_base.detach().cpu().numpy()
        meta_base = np.asarray(meta_base)

        fields = getattr(src, "metadata_fields", None)
        fields_list = [str(x) for x in list(fields)] if fields is not None else None

        # If metadata is already split-aligned, use as-is.
        if meta_base.ndim == 2 and meta_base.shape[0] == len(split_ds):
            return meta_base, fields_list

        # Otherwise we need an index map from split order -> meta_base row.
        idxs = None
        if isinstance(split_ds, Subset):
            idxs = np.asarray(split_ds.indices, dtype=np.int64)
        elif hasattr(split_ds, "indices"):
            try:
                idxs = np.asarray(getattr(split_ds, "indices"), dtype=np.int64)
            except Exception:
                idxs = None

        if idxs is None:
            # Fall back to reading the 4th field (idx) from dataset items.
            def _collate_idx(batch):
                return torch.tensor([int(x[3]) for x in batch], dtype=torch.long)

            ld = DataLoader(
                split_ds,
                batch_size=min(4096, int(train_bs)),
                shuffle=False,
                num_workers=int(train_nw),
                collate_fn=_collate_idx,
                pin_memory=False,
            )
            idx_list = []
            for idxb in ld:
                idx_list.append(idxb.numpy())
            idxs = np.concatenate(idx_list, axis=0).astype(np.int64)

        if idxs.ndim != 1 or idxs.shape[0] != len(split_ds):
            raise RuntimeError(
                f"[civilcomments] Bad idx map for split={split_name}: idxs.shape={getattr(idxs, 'shape', None)} "
                f"expected=({len(split_ds)},)"
            )

        # Safety check: if idxs are just 0..len(split)-1 but meta_base is larger, this is *not* a valid map.
        if meta_base.shape[0] != len(split_ds) and np.array_equal(idxs, np.arange(len(split_ds), dtype=np.int64)):
            raise RuntimeError(
                f"[civilcomments] split={split_name}: idx field appears to be LOCAL (0..N-1) but metadata_array is GLOBAL "
                f"(meta_rows={int(meta_base.shape[0])}, split_rows={int(len(split_ds))}). "
                "Cannot align identities without an indices map."
            )

        if int(idxs.min()) < 0 or int(idxs.max()) >= int(meta_base.shape[0]):
            raise RuntimeError(
                f"[civilcomments] split={split_name}: idx out of range for metadata_array: "
                f"min={int(idxs.min())} max={int(idxs.max())} meta_rows={int(meta_base.shape[0])}"
            )

        meta_local = meta_base[idxs]
        return meta_local, fields_list

    # Build split-local identity metadata once.
    civil_meta_val, civil_meta_fields = _split_local_meta(splits.val, split_name="val")
    civil_meta_test, _ = _split_local_meta(splits.test, split_name="test")

    if verbose:
        print(
            "[civilcomments] Prepared split-local metadata for 16-slice eval: "
            f"val_meta={tuple(civil_meta_val.shape)} test_meta={tuple(civil_meta_test.shape)}"
        )

    class _DatasetWithBaseSubsetMeta(torch.utils.data.Dataset):
        def __init__(self, inner, meta_array: np.ndarray, meta_fields: list[str] | None):
            self.inner = inner
            # Mimic the structure evaluate() expects: base_subset.metadata_array (+ optional metadata_fields)
            bs = SimpleNamespace(metadata_array=meta_array)
            if meta_fields is not None:
                bs.metadata_fields = meta_fields
            self.base_subset = bs

        def __len__(self):
            return len(self.inner)

        def __getitem__(self, i):
            return self.inner[i]

    def _wrap_for_civilcomments_eval(ds_obj, *, split: str, indices: np.ndarray | None = None):
        """
        Attach split-local metadata to a memmap dataset.

        If `indices` is provided (e.g. te_bulk indices into the *test* split),
        we subset the test metadata accordingly so meta_rows == len(ds_obj),
        which makes evaluation robust even if idx semantics change.
        """
        if split == "val":
            meta = civil_meta_val
        elif split == "test":
            meta = civil_meta_test
        else:
            raise ValueError(f"split must be 'val' or 'test', got {split!r}")

        if indices is not None:
            meta = np.asarray(meta)[np.asarray(indices, dtype=np.int64)]

        if int(np.asarray(meta).shape[0]) != len(ds_obj):
            raise RuntimeError(
                f"[civilcomments] meta length mismatch for split={split}: meta_rows={int(np.asarray(meta).shape[0])} "
                f"len(ds)={int(len(ds_obj))}"
            )

        return _DatasetWithBaseSubsetMeta(ds_obj, np.asarray(meta), civil_meta_fields)

    def _resolve_checkpoint_loader(val_loader, test_loader):
        sel = str(checkpoint_selection).lower()
        if sel in ("none", "off", ""):
            return None
        if sel in ("val", "validation"):
            return val_loader
        if sel in ("oracle_test", "test", "oracle"):
            warnings.warn(
                "[civilcomments] checkpoint_selection='oracle_test' uses TEST for checkpoint selection (data leakage). "
                "Use only for explicitly-labelled oracle diagnostics."
            )
            return test_loader
        raise ValueError(f"Unknown checkpoint_selection={checkpoint_selection!r}. Use 'none', 'val', or 'oracle_test'.")

    with _timed(timings, "runtime_bulk_select_s", device):
        tau = _quantile_threshold(s_cal, gamma=float(gamma))
        tr_bulk = np.where(s_tr <= tau)[0]
        cal_bulk = np.where(s_cal <= tau)[0]
        te_bulk = np.where(s_te <= tau)[0]
        if verbose:
            n_groups = int(splits.n_groups)
            _diag_index_subset("train_bulk", ytr_dense, gtr_dense, tr_bulk, n_groups)
            _diag_index_subset("test_bulk",  yte_dense, gte_dense, te_bulk, n_groups)
            if bool(train_on_bulk):
                print(
                    f"[civilcomments/{algorithm}] bulk sizes (official_train): "
                    f"train_bulk={int(tr_bulk.size)} cal_bulk={int(cal_bulk.size)} total_bulk={int(tr_bulk.size + cal_bulk.size)}"
                )


    with _timed(timings, "runtime_build_dataloaders_s", device):
        # Train/eval on cached dense hashed features.
        # official_train := TRAIN ∪ CAL
        train_ds_full = MemmapFeatureDataset(Xtr_dense, ytr_dense, gtr_dense)
        cal_ds_full = MemmapFeatureDataset(Xcal_dense, ycal_dense, gcal_dense)

        if bool(train_on_bulk):
            # LV methods train on bulk-filtered official_train (bulk calibration is LV-internal).
            train_ds_tr = MemmapFeatureDataset(Xtr_dense, ytr_dense, gtr_dense, indices=tr_bulk)
            train_ds_cal = MemmapFeatureDataset(Xcal_dense, ycal_dense, gcal_dense, indices=cal_bulk)
            train_ds = torch.utils.data.ConcatDataset([train_ds_tr, train_ds_cal])
        else:
            # Non-LV methods train on the full official_train without bulk filtering.
            train_ds = torch.utils.data.ConcatDataset([train_ds_full, cal_ds_full])

        val_ds = MemmapFeatureDataset(Xva_dense, yva_dense, gva_dense)
        test_ds = MemmapFeatureDataset(Xte_dense, yte_dense, gte_dense)
        test_bulk_ds = MemmapFeatureDataset(Xte_dense, yte_dense, gte_dense, indices=te_bulk)

        # Attach identity metadata for 16-slice worst-group eval
        val_ds = _wrap_for_civilcomments_eval(val_ds, split="val")
        test_ds = _wrap_for_civilcomments_eval(test_ds, split="test")
        test_bulk_ds = _wrap_for_civilcomments_eval(test_bulk_ds, split="test", indices=te_bulk)

        train_loader = _loader(train_ds, train_bs, train_nw, shuffle=True)
        val_loader = _loader(val_ds, train_bs, train_nw, shuffle=False)
        test_loader = _loader(test_ds, train_bs, train_nw, shuffle=False)
        test_bulk_loader = _loader(test_bulk_ds, train_bs, train_nw, shuffle=False)

    frozen_weight_decay = float(weight_decay)
    frozen_head_kind = str(head)
    frozen_mlp_hidden = int(mlp_hidden)

    train_stats: Dict[str, Any] = {}
    with _timed(timings, "runtime_train_s", device):
        checkpoint_loader = _resolve_checkpoint_loader(val_loader, test_loader)
        model = train_head(
            train_loader,
            d_in=int(n_features),
            n_classes=int(splits.n_classes),
            algorithm=train_algorithm,
            epsilon=float(epsilon),
            device=device,
            lr=float(lr),
            weight_decay=frozen_weight_decay,
            epochs=int(epochs),
            head=frozen_head_kind,
            mlp_hidden=frozen_mlp_hidden,
            smoothmax_T=float(smoothmax_T),
            groupdro_step=float(groupdro_step),
            max_grad_norm=max_grad_norm,
            train_stats=train_stats,
            known_num_groups=int(splits.n_groups),
            checkpoint_loader=checkpoint_loader,
            checkpoint_metric=str(checkpoint_metric),
            checkpoint_verbose=bool(verbose),
            checkpoint_force_civilcomments_16_slices=True,
        )
    # ---- Decision threshold calibration on VAL (binary-only; shared across all methods) ----
    n_groups = int(splits.n_groups)
    if bool(do_threshold_calibration):
        model_eval, thr_info = _maybe_threshold_calibrate_and_wrap(
            model,
            val_loader,
            device=device,
            n_classes=int(splits.n_classes),
            n_groups=n_groups,
            timings=timings,
            timings_key="runtime_threshold_calibrate_s",
            grid=401,
        )
    else:
        # Skip calibration: evaluate at the default binary threshold (logit > 0.0).
        model_eval = model
        timings["runtime_threshold_calibrate_s"] = 0.0
        thr_info = {
            "rw_decision_threshold": 0.0,
            "rw_val_worst_group_acc_at_t": float("nan"),
            "rw_val_avg_group_acc_at_t": float("nan"),
            "rw_n_threshold_candidates": 0.0,
        }

    if verbose:
        _diag_model_preds_binary(
            model,
            val_loader,
            device=device,
            n_groups=int(splits.n_groups),
            prefix=f"civilcomments/{algorithm}/VAL (post-train)",
            max_batches=50,
        )
        _diag_model_preds_binary(
            model,
            test_loader,
            device=device,
            n_groups=int(splits.n_groups),
            prefix=f"civilcomments/{algorithm}/TEST (post-train)",
            max_batches=50,
        )
        _diag_model_preds_binary(
            model,
            test_bulk_loader,
            device=device,
            n_groups=int(splits.n_groups),
            prefix=f"civilcomments/{algorithm}/TEST_INBULK (post-train)",
            max_batches=50,
        )

    out = {
        "rw_text_n_features": int(n_features),
        "rw_text_ngram_range": f"{ngram_range[0]}-{ngram_range[1]}",
        "tau_gamma": float(tau),
        # official_train := TRAIN ∪ CAL
        "n_train": int(len(s_tr) + len(s_cal)),
        "n_train_bulk": int(tr_bulk.size + cal_bulk.size),
        "test_bulk_coverage": float(te_bulk.size / max(1, len(s_te))),
        "tuned_weight_decay": float(frozen_weight_decay),
        "tuned_head_kind": str(frozen_head_kind),
        "tuned_mlp_hidden": int(frozen_mlp_hidden),
    }
    out.update(train_stats)
    out.update(thr_info)
    with _timed(timings, "runtime_eval_val_s", device):
        val_metrics = evaluate(
            model_eval,
            val_loader,
            device=device,
            n_groups=n_groups,
            force_civilcomments_16_slices=True,
            verbose=bool(verbose),
        )
    out.update({f"val_{k}": v for k, v in val_metrics.items()})

    with _timed(timings, "runtime_eval_test_s", device):
        test_metrics = evaluate(
            model_eval,
            test_loader,
            device=device,
            n_groups=n_groups,
            force_civilcomments_16_slices=True,
            verbose=bool(verbose),
        )
    out.update({f"test_{k}": v for k, v in test_metrics.items()})

    with _timed(timings, "runtime_eval_test_inbulk_s", device):
        test_bulk_metrics = evaluate(
            model_eval,
            test_bulk_loader,
            device=device,
            n_groups=n_groups,
            force_civilcomments_16_slices=True,
            verbose=bool(verbose),
        )
    out.update({f"test_inbulk_{k}": v for k, v in test_bulk_metrics.items()})

    _sync_device(device)
    timings["runtime_total_s"] = float(perf_counter() - t_total0)
    out.update(timings)
    return out

def run_real_world_replication(
    replication: int,
    dataset: str,
    algorithm: str,
    epsilon: float,
    gamma: float,
    dataset_dir: Path,
    *,
    # shared real-world controls
    rw_cal_fraction: float = 0.2,
    rw_split_seed: int = 0,
    rw_cache_dir: Optional[Path] = None,
    rw_device: str = "auto",
    rw_head: str = "linear",
    rw_mlp_hidden: int = 256,
    rw_train_batch_size: int = 256,
    rw_train_num_workers: int = 0,
    rw_epochs: int = 20,
    rw_lr: float = 1e-3,
    rw_weight_decay: float = 1e-4,
    rw_smoothmax_temperature: float = 0.1,
    rw_groupdro_step_size: float = 0.01,
    rw_max_grad_norm: Optional[float] = None,
    rw_text_n_features: int = 2**12,
    rw_text_ngram_min: int = 1,
    rw_text_ngram_max: int = 1,
    verbose: bool = False,
    do_threshold_calibration: bool = False,
) -> Dict[str, Any]:
    """
    Unified real-world runner.
    """
    key = dataset.lower()
    device = _device(rw_device)

    # Deterministic seed per replication
    set_seed(int(replication) + 17)

    cache_root = rw_cache_dir if rw_cache_dir is not None else (dataset_dir / "cache" / "real_world")

    base: Dict[str, Any] = {
        "dataset": dataset,
        "algorithm": algorithm,
        "replication": int(replication),
        "epsilon": float(epsilon),
        "gamma": float(gamma),
        "rw_split_seed": int(rw_split_seed),
        "rw_cal_fraction": float(rw_cal_fraction),
        "device": str(device),
    }
    if key in ("rw_civilcomments", "civilcomments"):
        extra = _run_civilcomments(
            algorithm=algorithm,
            epsilon=epsilon,
            gamma=gamma,
            replication=replication,
            dataset_dir=dataset_dir,
            cache_root=cache_root,
            split_seed=rw_split_seed,
            cal_fraction=rw_cal_fraction,
            device=device,
            n_features=int(rw_text_n_features),
            ngram_range=(int(rw_text_ngram_min), int(rw_text_ngram_max)),
            head=rw_head,
            mlp_hidden=rw_mlp_hidden,
            train_bs=rw_train_batch_size,
            train_nw=rw_train_num_workers,
            epochs=rw_epochs,
            lr=rw_lr,
            weight_decay=rw_weight_decay,
            smoothmax_T=rw_smoothmax_temperature,
            groupdro_step=rw_groupdro_step_size,
            max_grad_norm=rw_max_grad_norm,
            verbose=verbose,
            do_threshold_calibration=do_threshold_calibration,
        )
        base.update(extra)
        return base

    raise ValueError(
        f"Unknown real-world dataset '{dataset}'. "
        "Use: rw_civilcomments."
    )


def prepare_real_world_caches(
    *,
    dataset: str,
    dataset_dir: Path,
    rw_cal_fraction: float = 0.2,
    rw_split_seed: int = 0,
    rw_cache_dir: Optional[Path] = None,
    rw_device: str = "auto",
    # text controls
    rw_text_n_features: int = 2**12,
    rw_text_ngram_min: int = 1,
    rw_text_ngram_max: int = 1,
    rw_text_batch_size: int = 256,
    rw_text_num_workers: int = 4,
    prepare_dense_text_features: bool = False,
) -> Dict[str, Any]:
    """
    Build expensive caches ONCE (splits/embeddings/text scores; optionally dense text features).
    Intended to be run before epsilon sweeps, so subsequent runs do not pay cache build time.

    Returns a small metadata dict you can print/log.
    """
    key = dataset.lower()
    device = _device(rw_device)
    cache_root = rw_cache_dir if rw_cache_dir is not None else (dataset_dir / "cache" / "real_world")

    meta: Dict[str, Any] = {
        "dataset": dataset,
        "rw_cache_dir": str(cache_root),
        "rw_split_seed": int(rw_split_seed),
        "rw_cal_fraction": float(rw_cal_fraction),
        "device": str(device),
    }

    # ------------------- CivilComments -------------------
    if key in ("rw_civilcomments", "civilcomments"):
        splits = prepare_wilds_splits(
            wilds_name="civilcomments",
            root_dir=dataset_dir / "wilds",
            cache_dir=cache_root / "splits",
            split_seed=int(rw_split_seed),
            cal_fraction=float(rw_cal_fraction),
            download=True,
            transform=None,
        )
        raw = {"train": splits.train, "cal": splits.cal, "val": splits.val, "test": splits.test}

        n_features = int(rw_text_n_features)
        ngram_range = (int(rw_text_ngram_min), int(rw_text_ngram_max))

        # Score cache used by all CivilComments methods (ridge=0; label-free scores).
        ensure_text_score_cache(
            dataset_tag="civilcomments",
            splits=raw,
            cache_root=cache_root,
            n_features=n_features,
            ngram_range=ngram_range,
            split_seed=int(rw_split_seed),
            cal_fraction=float(rw_cal_fraction),
            batch_size=int(rw_text_batch_size),
            num_workers=int(rw_text_num_workers),
            verbose=True,
            n_classes=int(splits.n_classes),
            ridge=0.0,
            need_trueclass_scores=False,
        )
        if bool(prepare_dense_text_features):
            ensure_text_feature_cache(
                dataset_tag="civilcomments",
                splits=raw,
                cache_root=cache_root / "text_features",
                n_features=n_features,
                ngram_range=ngram_range,
                split_seed=int(rw_split_seed),
                cal_fraction=float(rw_cal_fraction),
                batch_size=int(rw_text_batch_size),
                num_workers=int(rw_text_num_workers),
                verbose=True,
            )
            pass

        meta.update(
            dict(
                cache_type="civilcomments",
                n_features=n_features,
                ngram_range=f"{ngram_range[0]}-{ngram_range[1]}",
                prepared_ridge0_scores=True,
                prepared_dense_features=bool(prepare_dense_text_features),
            )
        )
        return meta

    raise ValueError(f"Unknown dataset '{dataset}' for cache prep.")
