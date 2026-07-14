"""
Spindle time refinement (small per-event shifts) + similarity-based permutation + QC plots

What this does:
- Loads a previously saved file like "spindles_extracted.npz" (from the earlier pipeline).
- Saves refined arrays and metadata to a new NPZ and produces QC plots.
"""
__date__ = "September 2025"

from typing import Dict, Tuple, List
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list
import matplotlib.pyplot as plt
import os

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def _per_window_zscore(X: np.ndarray) -> np.ndarray:
    X = X - X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return X / sd


def _make_features(
    windows: np.ndarray,
    mode: str = "ref",          # "ref", "all_flat", or "pca"
    ref_channel: int = 0,
    n_components: int = 8,
    per_window_zscore: bool = True,
) -> np.ndarray:
    assert windows.ndim == 3, "Expected windows shape [N, W, C]"
    N, W, C = windows.shape

    if mode == "ref":
        X = windows[:, :, ref_channel]               # [N, W]
    else:
        X = windows.reshape(N, W * C)                # [N, W*C]

    if per_window_zscore:
        X = _per_window_zscore(X)

    if mode == "pca":
        Xc = X - X.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        k = min(n_components, Vt.shape[0])
        X = U[:, :k] * S[:k]
    return X


def _order_spindles_by_similarity(
    windows: np.ndarray,
    mode: str = "ref",
    ref_channel: int = 0,
    metric: str = "cosine",
    method: str = "olo",
    n_components: int = 8,
) -> np.ndarray:
    N = windows.shape[0]
    if N <= 1:
        return np.arange(N, dtype=int)

    X = _make_features(windows, mode=mode, ref_channel=ref_channel, n_components=n_components)

    d = pdist(X, metric=metric)

    if method == "olo":
        Z = linkage(d, method="average")
        Z_olo = optimal_leaf_ordering(Z, d)
        perm = leaves_list(Z_olo)
        return perm
    elif method == "tsp":
        D = squareform(d)
        np.fill_diagonal(D, np.inf)
        visited = np.zeros(N, dtype=bool)
        perm = []
        start = np.argmin(D.mean(axis=1))
        cur = start
        for _ in range(N):
            perm.append(cur)
            visited[cur] = True
            nxt = np.argmin(np.where(visited, np.inf, D[cur]))
            cur = int(nxt) if not np.isinf(D[cur, nxt]) else int(np.where(~visited)[0][0])
        return np.array(perm, dtype=int)
    else:
        raise ValueError("method must be 'olo' or 'tsp'")


def _cosine_at_shift(x: np.ndarray, tpl: np.ndarray, shift: int) -> float:
    """
    Cosine similarity between x rolled by 'shift' and template tpl.
    x, tpl: 1D arrays length W, already zero-mean / scaled (but we compute norms anyway).
    Positive shift means x is moved to the right.
    """
    n = len(x)
    i1, i2 = int(round(0.4 * n)), int(round(0.6 * n))
    tpl = tpl[i1:i2]
    xr = np.roll(x[i1:i2], shift)
    num = float(np.dot(xr, tpl))
    den = float(np.linalg.norm(xr) * np.linalg.norm(tpl))
    if den == 0.0:
        return 0.0
    return num / den


def refine_spindle_times_leave_one_out(
    windows_band: np.ndarray,   # [N, W, C] spindle-band windows (used for alignment)
    ref_channel: int,
    fs: float,
    max_shift_ms: float = 100.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each spindle i, find an integer circular shift within ±max_shift_ms that maximizes
    cosine similarity between the **ref channel** waveform and the **leave-one-out mean template**
    of all other spindles. Returns (shifts, refined_windows_band). Shifts are in samples.
    Only windows_band are refined here; apply the same shifts to raw windows outside this function.
    """
    N, W, C = windows_band.shape
    if N == 0:
        return np.zeros((0,), dtype=int), windows_band.copy()

    # reference channel matrix [N, W], z-score per window for shape matching
    X = windows_band[:, :, ref_channel].astype(np.float64)
    X = _per_window_zscore(X)

    # Precompute sum over all events for fast leave-one-out mean
    sum_all = np.sum(X, axis=0)  # [W]
    shifts = np.zeros(N, dtype=int)
    refined = windows_band.copy()

    max_shift = int(round((max_shift_ms / 1000.0) * fs))
    lags = np.arange(-max_shift, max_shift + 1, dtype=int)

    # Small epsilon to avoid zero template after leave-one-out when N=1
    eps = 1e-9

    for i in range(N):
        # Leave-one-out template
        tpl = (sum_all - X[i]) / max(1, (N - 1))
        tpl_norm = np.linalg.norm(tpl)
        if tpl_norm < eps:
            # Fall back to global mean (rare if N is small)
            tpl = sum_all / max(1, N)
        # Evaluate cosine similarity over allowed shifts
        best_shift = 0
        best_sim = -np.inf
        for s in lags:
            sim = _cosine_at_shift(X[i], tpl, s)
            if sim > best_sim:
                best_sim = sim
                best_shift = s
        shifts[i] = best_shift
        # Apply shift to all channels for this event (spindle-band windows only here)
        refined[i] = np.roll(refined[i], shift=best_shift, axis=0)

    return shifts, refined


def apply_shifts_to_windows(windows: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """
    Apply integer circular shifts (samples) per event to windows [N, W, C].
    """
    N, W, C = windows.shape
    out = windows.copy()
    for i in range(N):
        out[i] = np.roll(out[i], shift=shifts[i], axis=0)
    return out


def refine_shift_and_sort_npz(
    in_npz: str = "spindles_extracted.npz",
    out_npz: str = "spindles_refined_sorted.npz",
    max_shift_ms: float = 100.0,
    mode: str = "ref",
    metric: str = "cosine",
    method: str = "olo",
    n_components: int = 8,
    make_plots: bool = True,
    out_dir: str = '',
):
    """
    Main method:
    - Load NPZ from earlier pipeline
    - Refine times (per-event shifts) on spindle-band windows using ref channel
    - Apply shifts to both spindle-band and raw windows
    - Compute similarity-based permutation on refined spindle-band windows
    - Save refined+sorted results and make QC plots
    """
    D = np.load(in_npz, allow_pickle=True)
    windows_raw = D["windows_raw"]              # [N, W, C]
    windows_sp = D["windows_spindleband"]       # [N, W, C]
    fs = float(D["fs"])
    ref_channel = int(D["ref_channel"])
    centers = D["centers"]                      # [N]
    events = D["events"]                        # [N, 2]
    durations_s = D["durations_s"]              # [N]

    # 1) Time refinement shifts on spindle-band windows
    shifts, windows_sp_refined = refine_spindle_times_leave_one_out(
        windows_sp, ref_channel=ref_channel, fs=fs, max_shift_ms=max_shift_ms
    )
    # 2) Apply shifts to raw windows as well
    windows_raw_refined = apply_shifts_to_windows(windows_raw, shifts)

    # 3) Compute permutation on refined windows
    perm = _order_spindles_by_similarity(
        windows=windows_sp_refined,
        mode=mode,
        ref_channel=ref_channel,
        metric=metric,
        method=method,
        n_components=n_components,
    )

    # 4) Apply permutation to everything
    windows_sp_sorted = windows_sp_refined[perm]
    windows_raw_sorted = windows_raw_refined[perm]
    centers_sorted = centers[perm]
    events_sorted = events[perm]
    durations_sorted = durations_s[perm]
    shifts_sorted = shifts[perm]

    # 5) Save
    np.savez_compressed(
        out_npz,
        windows_raw=windows_raw_sorted,
        windows_spindleband=windows_sp_sorted,
        centers=centers_sorted,
        events=events_sorted,
        durations_s=durations_sorted,
        fs=fs,
        ref_channel=ref_channel,
        shifts=shifts_sorted,
        params={
            "max_shift_ms": max_shift_ms,
            "ordering_mode": mode,
            "ordering_metric": metric,
            "ordering_method": method,
            "n_components": n_components,
            "source_npz": in_npz,
        },
        perm=perm,
    )

    # 6) QC plots
    if make_plots and windows_sp_sorted.shape[0] > 0:
        # A) Sorted raster on ref channel (raw windows after refinement)
        ref_raw = windows_raw_sorted[:, :, ref_channel]  # [N, W]
        t = (np.arange(ref_raw.shape[1]) - ref_raw.shape[1] // 2) / fs
        fig, axarr = plt.subplots(nrows=2)
        plt.sca(axarr[0])
        plt.imshow(
            ref_raw, aspect="auto", origin="lower",
            extent=[t[0], t[-1], 0, ref_raw.shape[0]]
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Spindles (sorted)")
        plt.title("Raw windows (ref channel) — refined & sorted")

        plt.sca(axarr[1])
        ref_filt = windows_sp_sorted[:, :, ref_channel]
        plt.imshow(
            ref_filt, aspect="auto", origin="lower",
            extent=[t[0], t[-1], 0, ref_raw.shape[0]]
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Spindles (sorted)")
        plt.title("Filtered windows (ref channel) — refined & sorted")

        plt.tight_layout()
        plt.savefig("spindle_qc_sorted_raster.png", dpi=150)
        plt.close()

        # B) Mean ± SEM of spindle-band ref channel after refinement
        ref_sp = windows_sp_sorted[:, :, ref_channel]
        mean_ref = np.mean(ref_sp, axis=0)
        sem_ref = np.std(ref_sp, axis=0, ddof=1) / np.sqrt(ref_sp.shape[0])
        plt.figure()
        plt.plot(t, mean_ref)
        plt.fill_between(t, mean_ref - sem_ref, mean_ref + sem_ref, alpha=0.3)
        plt.axvline(0.0, linestyle="--")
        plt.xlabel("Time (s)")
        plt.ylabel("Spindle-band (a.u.)")
        plt.title("Mean ± SEM (ref channel) — refined")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "spindle_qc_mean_ref_refined.png"), dpi=150)
        plt.close()

    return out_npz


if __name__ == '__main__':
    out_dir = os.path.join(ROOT, "spindle_data")

    out_path = refine_shift_and_sort_npz(
        in_npz=os.path.join(out_dir, "spindles_extracted.npz"),
        out_npz=os.path.join(out_dir, "spindles_refined_sorted.npz"),
        max_shift_ms=50.0,
        mode="ref",            # features for ordering: "ref", "all_flat", or "pca"
        metric="cosine",
        method="olo",           # or "tsp"
        n_components=8,
        make_plots=True,
        out_dir=out_dir,
    )
    print("Saved to:", out_path)
