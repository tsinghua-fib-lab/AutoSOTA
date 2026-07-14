"""
Find putative sleep spindles in the LFPs.

Files produced when you run the main() example:
- "spindles_extracted.npz"  : extracted windows, metadata, and parameters
- "spindle_qc_mean_ref.png" : mean ± SEM of spindle-band signal on the reference channel
- "spindle_qc_duration_hist.png" : histogram of event durations
- "spindle_qc_ref_raster.png" : raster (imshow) of raw windows on the reference channel
"""
__date__ = "September 2025"

from typing import Dict, Tuple, List
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list
import matplotlib.pyplot as plt
import os


from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]



def bandpass_filter(x: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    """
    Zero-phase Butterworth bandpass filter via filtfilt.
    x: shape [T] or [T, C]
    returns same shape
    """
    b, a = signal.butter(order, [low / (fs / 2.0), high / (fs / 2.0)], btype="bandpass")
    if x.ndim == 1:
        return signal.filtfilt(b, a, x, axis=0, method="gust")
    else:
        return signal.filtfilt(b, a, x, axis=0, method="gust")


def hilbert_envelope(x: np.ndarray) -> np.ndarray:
    """
    Analytic signal magnitude (envelope) via Hilbert transform.
    x: shape [T] or [T, C]
    """
    analytic = signal.hilbert(x, axis=0)
    env = np.abs(analytic)
    return env


def instantaneous_phase_and_freq(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Instantaneous phase (unwrap(angle(hilbert))) and frequency (Hz) from bandpassed signal.
    x: [T] or [T, C]
    """
    analytic = signal.hilbert(x, axis=0)
    phase = np.unwrap(np.angle(analytic), axis=0)
    dphi = np.diff(phase, axis=0)
    inst_freq = (fs / (2.0 * np.pi)) * dphi  # [T-1] or [T-1, C]
    return phase, inst_freq


def smooth_envelope(env: np.ndarray, fs: float, sigma_ms: float = 75.0) -> np.ndarray:
    """
    Gaussian smoothing of the envelope. sigma given in milliseconds.
    """
    sigma_samples = max(1, int(round((sigma_ms / 1000.0) * fs)))
    if env.ndim == 1:
        return gaussian_filter1d(env, sigma=sigma_samples, axis=0, mode="nearest")
    else:
        return gaussian_filter1d(env, sigma=sigma_samples, axis=0, mode="nearest")


def robust_zscore(x: np.ndarray, mask: np.ndarray = None) -> Tuple[np.ndarray, float, float]:
    """
    Robust z-score using median and MAD (scaled by 1.4826).
    If mask is provided (boolean [T]), compute median/MAD on x[mask].
    Returns z, median, mad_sd
    """
    if mask is not None:
        xm = x[mask]
    else:
        xm = x
    med = np.median(xm, axis=0)
    mad = np.median(np.abs(xm - med), axis=0)
    mad_sd = 1.4826 * mad
    mad_sd = np.where(mad_sd == 0, 1.0, mad_sd)  # avoid divide by zero
    z = (x - med) / mad_sd
    return z, med, mad_sd


def hysteresis_events(
        z: np.ndarray,
        fs: float,
        th_hi: float = 3.0,
        th_lo: float = 1.0,
        min_dur_s: float = 0.4,
        max_dur_s: float = 2.0,
        merge_gap_s: float = 0.25,
        max_n_events=2000,
    ) -> List[Tuple[int, int]]:
    """
    Detect events in a 1D z-scored time series using hysteretic thresholding.
    Returns a list of (start_idx, end_idx) inclusive-exclusive intervals.
    """
    assert z.ndim == 1, "hysteresis_events expects a 1D array."
    T = z.shape[0]
    min_dur = int(round(min_dur_s * fs))
    max_dur = int(round(max_dur_s * fs))
    merge_gap = int(round(merge_gap_s * fs))

    above_hi = z >= th_hi
    above_lo = z >= th_lo

    events = []
    i = 0
    while i < T:
        # Look for upward crossing of high threshold
        if above_hi[i]:
            # back up until we drop below low (start at the first sample >= low before this peak)
            start = i
            while start > 0 and above_lo[start - 1]:
                start -= 1
            # march forward until we drop below low
            j = i + 1
            while j < T and above_lo[j]:
                j += 1
            end = j  # exclusive
            events.append((start, end))
            i = j
            if len(events) == max_n_events:
                break
        else:
            i += 1

    # Merge events with short gaps
    if not events:
        return []
    merged = [events[0]]
    for (s, e) in events[1:]:
        ps, pe = merged[-1]
        if s - pe <= merge_gap:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))

    # Apply duration bounds
    pruned = []
    for (s, e) in merged:
        dur = e - s
        if dur >= min_dur and dur <= max_dur:
            pruned.append((s, e))
    return pruned


def apply_event_vetoes(
        env_hf_z: np.ndarray,
        env_lf_z: np.ndarray,
        events: List[Tuple[int, int]],
        veto_z: float = 3.5,
    ) -> List[Tuple[int, int]]:
    """
    Veto events if high-frequency or low-frequency envelope exceeds veto_z
    anywhere inside the candidate interval.
    env_hf_z, env_lf_z: [T] z-scored envelopes
    """
    kept = []
    for (s, e) in events:
        if np.nanmax(env_hf_z[s:e]) > veto_z:
            continue
        if np.nanmax(env_lf_z[s:e]) > veto_z:
            continue
        kept.append((s, e))
    return kept


def frequency_cycle_check(
        x_spindle: np.ndarray,
        fs: float,
        events: List[Tuple[int, int]],
        fmin: float = 9.0,
        fmax: float = 16.0,
        min_cycles: int = 3,
    ) -> List[Tuple[int, int]]:
    """
    Keep events whose median instantaneous frequency is in [fmin, fmax]
    and contain at least `min_cycles` cycles.
    x_spindle: bandpassed reference-channel signal [T]
    """
    _, inst_freq = instantaneous_phase_and_freq(x_spindle, fs)  # length T-1
    kept = []
    for (s, e) in events:
        # handle indexing since inst_freq is T-1
        s2 = max(0, s - 1)
        e2 = max(s2 + 1, e - 1)
        med_f = np.median(inst_freq[s2:e2])
        dur_s = (e - s) / fs
        cycles = dur_s * med_f
        if (med_f >= fmin) and (med_f <= fmax) and (cycles >= min_cycles):
            kept.append((s, e))
    return kept


def extract_centered_windows(
        data: np.ndarray,
        centers: np.ndarray,
        fs: float,
        window_s: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract windows centered at given centers (sample indices).
    data: [T, C]
    centers: [N] sample indices
    returns (windows [N, W, C], valid_centers [N]) with only those that fit fully.
    """
    T, C = data.shape
    half = int(round((window_s * fs) / 2.0))
    W = 2 * half
    windows = []
    valid_centers = []
    for c in centers:
        s = c - half
        e = c + half
        if s >= 0 and e <= T:
            windows.append(data[s:e, :])
            valid_centers.append(c)
    if len(windows) == 0:
        return np.zeros((0, 0, data.shape[1])), np.array([], dtype=int)
    return np.stack(windows, axis=0), np.array(valid_centers, dtype=int)


def detect_and_extract_spindles(
        data: np.ndarray,
        fs: float,
        ref_channel: int,
        params: Dict = None,
    ) -> Dict:
    """
    Full pipeline:
    - Bandpass 9-16 Hz on ref channel; envelope -> smooth -> robust z
    - Hysteresis detection with duration + gap rules
    - Instantaneous frequency + min cycles checks
    - Artifact veto using 40-100 Hz and 0.5-4 Hz z-scored envelopes
    - Center events at envelope peak within each interval
    - Extract 2 s windows (all channels), both raw and spindle-band-filtered
    Returns a dict with fields and also saves .npz and QC plots if `save=True` in params.
    """
    if params is None:
        params = {}
    # Defaults
    band = params.get("spindle_band", (9.0, 16.0))
    hf_band = params.get("hf_veto_band", (40.0, 100.0))
    lf_band = params.get("lf_veto_band", (0.5, 4.0))
    filt_order = params.get("butter_order", 4)
    sigma_ms = params.get("envelope_sigma_ms", 75.0)
    th_hi = params.get("th_hi", 3.0)
    th_lo = params.get("th_lo", 1.0)
    min_dur_s = params.get("min_dur_s", 0.5)
    max_dur_s = params.get("max_dur_s", 1.5)
    merge_gap_s = params.get("merge_gap_s", 0.25)
    fmin = params.get("inst_freq_min", band[0])
    fmax = params.get("inst_freq_max", band[1])
    min_cycles = params.get("min_cycles", 3)
    veto_z = params.get("veto_z", 3.5)
    window_s = params.get("window_s", 2.0)
    save = params.get("save", True)
    out_npz = params.get("out_npz", "spindles_extracted.npz")
    out_plot_mean = params.get("out_plot_mean", "spindle_qc_mean_ref.png")
    out_plot_hist = params.get("out_plot_hist", "spindle_qc_duration_hist.png")
    out_plot_raster = params.get("out_plot_raster", "spindle_qc_ref_raster.png")

    T, C = data.shape
    # 1) Spindle-band filter (ref channel)
    x_ref = data[:, ref_channel]
    x_sp_ref = bandpass_filter(x_ref, fs, band[0], band[1], order=filt_order)

    # 2) Envelope + smoothing + robust z
    env_ref = hilbert_envelope(x_sp_ref)
    env_ref_s = smooth_envelope(env_ref, fs, sigma_ms=sigma_ms)
    z_ref, z_med, z_sd = robust_zscore(env_ref_s)  # entire session

    # 3) Hysteresis + duration/gap
    events = hysteresis_events(z_ref, fs, th_hi=th_hi, th_lo=th_lo,
                               min_dur_s=min_dur_s, max_dur_s=max_dur_s, merge_gap_s=merge_gap_s,
                               max_n_events=None)

    # 4) Instantaneous freq + cycles checks
    events = frequency_cycle_check(x_sp_ref, fs, events, fmin=fmin, fmax=fmax, min_cycles=min_cycles)

    # 5) Artifact veto envelopes (precompute from ref)
    #    High-frequency and low-frequency envelopes with robust z
    x_hf_ref = bandpass_filter(x_ref, fs, hf_band[0], hf_band[1], order=filt_order)
    env_hf = smooth_envelope(hilbert_envelope(x_hf_ref), fs, sigma_ms=50.0)
    env_hf_z, _, _ = robust_zscore(env_hf)

    x_lf_ref = bandpass_filter(x_ref, fs, lf_band[0], lf_band[1], order=filt_order)
    env_lf = smooth_envelope(hilbert_envelope(x_lf_ref), fs, sigma_ms=100.0)
    env_lf_z, _, _ = robust_zscore(env_lf)

    events = apply_event_vetoes(env_hf_z=env_hf_z, env_lf_z=env_lf_z, events=events, veto_z=veto_z)

    # 6) Center on envelope peak within each event (ref channel)
    centers = []
    final_events = []
    durations_s = []
    for (s, e) in events:
        peak_rel = np.argmax(env_ref_s[s:e])
        center = s + peak_rel
        centers.append(center)
        final_events.append((s, e))
        durations_s.append((e - s) / fs)
    centers = np.array(centers, dtype=int)
    durations_s = np.array(durations_s, dtype=float)

    # 7) Extract windows (all channels, raw)
    windows_raw, valid_centers = extract_centered_windows(data, centers, fs, window_s=window_s)

    # 8) Spindle-band filter all channels, then extract windows for QC mean traces
    data_sp = bandpass_filter(data, fs, band[0], band[1], order=filt_order)
    windows_sp, _ = extract_centered_windows(data_sp, centers, fs, window_s=window_s)

    # Recompute events to only those windows that were valid (fit fully)
    if len(valid_centers) != len(centers):
        mask_valid = np.isin(centers, valid_centers)
        durations_s = durations_s[mask_valid]
        final_events = [ev for ev, keep in zip(final_events, mask_valid) if keep]
        centers = valid_centers

    # 9) Save results
    result = {
        "windows_raw": windows_raw,                 # [N, W, C]
        "windows_spindleband": windows_sp,          # [N, W, C]
        "centers": centers,                         # [N]
        "events": np.array(final_events, dtype=int),# [N,2] (start,end)
        "durations_s": durations_s,                 # [N]
        "fs": fs,
        "ref_channel": ref_channel,
        "params": {
            "spindle_band": band,
            "hf_veto_band": hf_band,
            "lf_veto_band": lf_band,
            "butter_order": filt_order,
            "envelope_sigma_ms": sigma_ms,
            "th_hi": th_hi,
            "th_lo": th_lo,
            "min_dur_s": min_dur_s,
            "max_dur_s": max_dur_s,
            "merge_gap_s": merge_gap_s,
            "inst_freq_min": fmin,
            "inst_freq_max": fmax,
            "min_cycles": min_cycles,
            "veto_z": veto_z,
            "window_s": window_s,
        },
    }

    if save:
        np.savez_compressed(out_npz, **result)

        # QC plots
        # A) Mean ± SEM of spindle-band signal on reference channel
        if windows_sp.shape[0] > 0:
            ref_sp = windows_sp[:, :, ref_channel]  # [N, W]
            mean_ref = np.mean(ref_sp, axis=0)
            sem_ref = np.std(ref_sp, axis=0, ddof=1) / np.sqrt(ref_sp.shape[0])
            t = (np.arange(ref_sp.shape[1]) - ref_sp.shape[1] // 2) / fs
            plt.figure()
            plt.plot(t, mean_ref)
            plt.fill_between(t, mean_ref - sem_ref, mean_ref + sem_ref, alpha=0.3)
            plt.axvline(0.0, linestyle="--")
            plt.xlabel("Time (s)")
            plt.ylabel("Spindle-band (a.u.)")
            plt.title("Mean ± SEM (ref channel spindle-band)")
            plt.tight_layout()
            plt.savefig(out_plot_mean, dpi=150)
            plt.close()

        # B) Histogram of event durations
        if durations_s.size > 0:
            plt.figure()
            plt.hist(durations_s, bins=20)
            plt.xlabel("Duration (s)")
            plt.ylabel("Count")
            plt.title("Spindle duration histogram")
            plt.tight_layout()
            plt.savefig(out_plot_hist, dpi=150)
            plt.close()

        # C) Raster (imshow) of raw reference-channel windows
        if windows_raw.shape[0] > 0:
            ref_raw = windows_raw[:, :, ref_channel]  # [N, W]
            plt.figure()
            plt.imshow(ref_raw, aspect="auto", origin="lower",
                       extent=[-windows_raw.shape[1] / (2 * fs),
                               windows_raw.shape[1] / (2 * fs),
                               0, ref_raw.shape[0]])
            plt.xlabel("Time (s)")
            plt.ylabel("Spindle #")
            plt.title("Raw windows (ref channel)")
            plt.tight_layout()
            plt.savefig(out_plot_raster, dpi=150)
            plt.close()

    return result


def _make_features(
    windows: np.ndarray,
    mode: str = "ref",          # "ref", "all_flat", or "pca"
    ref_channel: int = 0,
    n_components: int = 8,
    per_window_zscore: bool = True,
) -> np.ndarray:
    """
    Build a feature vector per spindle window.

    windows: array [N, W, C] (N spindles, W samples, C channels)
    mode:
      - "ref":       use only ref_channel waveform [W]
      - "all_flat":  flatten all channels [W*C]
      - "pca":       SVD-PCA of flattened [W*C] down to n_components
    per_window_zscore: z-score each spindle's waveform(s) before feature build
    returns: features [N, D]
    """
    assert windows.ndim == 3, "Expected windows shape [N, W, C]"
    N, W, C = windows.shape

    if mode == "ref":
        X = windows[:, :, ref_channel]               # [N, W]
    else:
        X = windows.reshape(N, W * C)                # [N, W*C]

    if per_window_zscore:
        X = X - X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True)
        sd[sd == 0] = 1.0
        X = X / sd

    if mode == "pca":
        # PCA via SVD on centered data across spindles
        Xc = X - X.mean(axis=0, keepdims=True)
        # economy SVD
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        k = min(n_components, Vt.shape[0])
        X = U[:, :k] * S[:k]                          # [N, k]

    return X


def order_spindles_by_similarity(
    windows: np.ndarray,
    mode: str = "ref",            # "ref", "all_flat", "pca"
    ref_channel: int = 0,
    metric: str = "cosine",       # any scipy.spatial.distance pdist metric
    method: str = "olo",          # "olo" (hierarchical optimal leaf ordering) or "tsp"
    n_components: int = 8,
) -> np.ndarray:
    """
    Return a permutation of spindle indices so neighbors are similar.

    windows: [N, W, C] spindle windows (raw or spindle-band; your choice)
    mode: feature construction (see _make_features)
    metric: distance metric for pdist ('cosine', 'euclidean', etc.)
    method:
      - "olo": hierarchical clustering + optimal leaf ordering (recommended)
      - "tsp": greedy nearest-neighbor tour over the distance matrix
    """
    N = windows.shape[0]
    if N == 0:
        return np.array([], dtype=int)
    if N == 1:
        return np.array([0], dtype=int)

    X = _make_features(windows, mode=mode, ref_channel=ref_channel, n_components=n_components)
    print(np.min(X), np.max(X))

    # Pairwise distances
    d = pdist(X, metric=metric)
    print(np.min(d), np.max(d))

    if method == "olo":
        # Hierarchical clustering + optimal leaf ordering
        Z = linkage(d, method="average")  # 'average' is robust; 'ward' needs Euclidean
        Z_olo = optimal_leaf_ordering(Z, d)
        perm = leaves_list(Z_olo)
        return perm

    elif method == "tsp":
        # Simple greedy nearest-neighbor TSP over distances (symmetric)
        D = squareform(d)
        np.fill_diagonal(D, np.inf)
        visited = np.zeros(N, dtype=bool)
        perm = []
        # start at a medoid-ish point (min avg distance)
        start = np.argmin(D.mean(axis=1))
        cur = start
        for _ in range(N):
            perm.append(cur)
            visited[cur] = True
            # pick nearest unvisited
            nxt = np.argmin(np.where(visited, np.inf, D[cur]))
            cur = int(nxt) if not np.isinf(D[cur, nxt]) else int(np.where(~visited)[0][0])
        return np.array(perm, dtype=int)

    else:
        raise ValueError("method must be 'olo' or 'tsp'")


# --- Convenience helpers ---

def apply_permutation(windows: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Reorder windows [N, W, C] by perm."""
    return windows[perm, ...]


def demo_sort_and_plot(
    windows: np.ndarray,
    fs: float,
    ref_channel: int = 0,
    mode: str = "ref",
    metric: str = "cosine",
    method: str = "olo",
    fn="spindle_qc_perm_raster.png",
):
    """
    Quick QC plot to visualize the ordering on the reference channel.
    """
    import matplotlib.pyplot as plt

    perm = order_spindles_by_similarity(
        windows, mode=mode, ref_channel=ref_channel, metric=metric, method=method
    )
    W = windows.shape[1]
    t = (np.arange(W) - W // 2) / fs
    ref = windows[perm, :, ref_channel]

    plt.figure()
    plt.imshow(ref, aspect="auto", origin="lower",
               extent=[t[0], t[-1], 0, ref.shape[0]])
    plt.xlabel("Time (s)")
    plt.ylabel("Spindles (sorted)")
    plt.title(f"Spindles sorted by similarity ({method}, {mode}, {metric})")
    plt.tight_layout()
    plt.savefig(fn)
    plt.close("all")



if __name__ == "__main__":
    # Load data: numpy array of shape [T, C]
    data_fn = os.path.join(ROOT, "data", "lfp_data", "torus_data.npz")
    channels = np.load(data_fn)["channels"].tolist()
    data = np.load(data_fn)["lfps"]

    out_dir = os.path.join(ROOT, "data", "spindle_data")
    
    # [0,28,31,37,49] Amy, MdThal, Nac, PRL, VTa
    target_channels = [
        "MD_Thal_01",
        "MD_Thal_02",
        "Cg_Cx_L_01",
        "Cg_Cx_R_01",
        "IL_Cx_L_01",
        "PrL_Cx_L_01",
        "PrL_Cx_R_01",
        "S1_Cx_01",
        "dHipp_01",
        "vHipp_01"
    ]
    target_indices = [channels.index(i) for i in target_channels]
    i1 = 0
    i2 = 8 * 250 * 60 * 60
    data = data[i1:i2, target_indices] # Only look through the first 8 hours.
    print("data", data.shape)

    fs = 250.0  # samplerate in Hz
    ref_channel = target_channels.index("PrL_Cx_L_01")  # index of the channel to detect spindles on
    
    params = dict(
        spindle_band=(9.0, 16.0),
        hf_veto_band=(40.0, 100.0),
        lf_veto_band=(0.5, 4.0),
        butter_order=4,
        envelope_sigma_ms=75.0,
        th_hi=3.0,
        th_lo=1.0,
        min_dur_s=0.5,
        max_dur_s=1.5,
        merge_gap_s=0.25,
        inst_freq_min=9.0,
        inst_freq_max=16.0,
        min_cycles=3,
        veto_z=3.5,
        window_s=1.5,
        save=True,
        out_npz=os.path.join(out_dir, "spindles_extracted.npz"),
        out_plot_mean=os.path.join(out_dir, "spindle_qc_mean_ref.png"),
        out_plot_hist=os.path.join(out_dir, "spindle_qc_duration_hist.png"),
        out_plot_raster=os.path.join(out_dir, "spindle_qc_ref_raster.png"),
    )
    
    result = detect_and_extract_spindles(data, fs, ref_channel, params)
    print(f"Extracted {result['windows_raw'].shape[0]} spindles; windows saved to {params['out_npz']}")

    # Load saved data
    D = np.load(os.path.join(out_dir, "spindles_extracted.npz"0, allow_pickle=True)
    windows = D["windows_spindleband"] # or D["windows_raw"]
    print(np.max(windows), np.min(windows))
    fs = float(D["fs"])
    ref_ch = int(D["ref_channel"])

    # 1) Get an ordering using ref-channel waveforms + cosine distance
    perm = order_spindles_by_similarity(windows, mode="ref", ref_channel=ref_ch,
                                        metric="cosine", method="olo")

    # 2) Apply permutation to all arrays (windows, centers, etc.)
    windows_sorted = apply_permutation(windows, perm)
    centers_sorted = D["centers"][perm]

    # 3) Visualize the sorted raster on the ref channel
    fn = os.path.join(out_dir, "spindle_qc_perm_raster.png")
    demo_sort_and_plot(windows, fs, ref_channel=ref_ch, mode="ref", metric="cosine", method="olo", fn=fn)
