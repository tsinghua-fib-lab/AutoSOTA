"""
ATC scaling experiment: average regret vs horizon T (fixed number of change points S)

What this script does (modular version):
  1) Implements ATC with pluggable scan strategy and pluggable threshold rule.
  2) Adds GCP (from your attached code): GC-style scan with
       - constant FA-bound threshold  γ_GCP(T,S,α)
       - forced periodic restarts every floor(T/(S+1)) time steps
       - forced restarts are NOT counted as alarms.
  3) Runs Monte-Carlo (MC) estimates of regret for several horizons T, with fixed S (=5 in scenarios).
  4) Plots average regret R_T versus ln(T).
     - If you run multiple algorithm variants, they are plotted on the same figure.

Scan strategies included (for ATC variants):
  - exact scan:           k = r+1, ..., t-1   (quadratic worst-case)
  - uniform max_splits:   M evenly spaced candidates in [r+1, t-1],
                          where M can be an int or "logT" (ceil(ln T))
  - geom_ends (Option A): candidates at geometric offsets from the ends:
                          k = r + d and k = t - d for d = 1, ceil(b), ceil(b^2), ...
                          plus the midpoint. Uses O(log(t-r)) candidates per t.

Threshold rules included (for ATC variants):
  - paper threshold: adaptive gamma_t^r as in ATC
  - constant threshold: gamma = const_gamma * sigma

GCP (added):
  - Uses the same scan statistic over k (max over splits), but with:
      * constant threshold:
          γ = scale * σ * sqrt( 2 log( 2 T^2 / ((S+1) α) ) )
      * forced periodic restarts with period floor(T/(S+1))
    This is the “GCP: GC with forced periodic restarts” described in the attached file.

Key behavior:
  - Runs ONLY the "md" scenario by default (the "easy" scenario remains in code,
    but is not executed unless you choose it via CLI).

Requirements:
  numpy, matplotlib
"""

import os
import math
import argparse
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Union, Dict, Any

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Environment: piecewise-constant mean + Gaussian noise
# -----------------------------
def build_piecewise_instance(T: int, scenario: str = "md", short_len: int = 10):
    """
    Returns (taus, mus) with:
      taus = [tau0=1, tau1, ..., tauS, tau_{S+1}=T+1]  (1-indexed; end exclusive)
      mus  = [mu0, mu1, ..., muS]                       (S+1 means)

    S is fixed (=5) in both scenarios.

    scenario:
      - 'easy': 5 large changes, long segments (should be detected)
      - 'md'  : includes a short "blip" segment of length short_len (a typical MD case)
    """
    if T < 100:
        raise ValueError("Use T >= 100 for the default scenario builders.")

    # Five change points at fixed fractions of T
    cp1 = int(np.floor(0.20 * T)) + 1
    cp2 = int(np.floor(0.40 * T)) + 1
    if scenario == "md":
        cp3 = min(cp2 + short_len, T)  # short-lived segment
    else:
        cp3 = int(np.floor(0.60 * T)) + 1
    cp4 = int(np.floor(0.75 * T)) + 1
    cp5 = int(np.floor(0.90 * T)) + 1
    # Ensure strict monotonicity
    cps = [cp1, cp2, cp3, cp4, cp5]
    if scenario == "adver":
        cps = [int(np.floor(i * T / 6)) for i in range(1, 6)]
    for i in range(1, len(cps)):
        if cps[i] <= cps[i - 1]:
            cps[i] = min(T, cps[i - 1] + 1)

    cp1, cp2, cp3, cp4, cp5 = cps
    taus = [1, cp1, cp2, cp3, cp4, cp5, T + 1]  # tau_{S+1} = T+1

    if scenario == "easy":
        mus = [0.0, 2.0, -1.5, 1.5, -0.5, 2.0]
    elif scenario == "md":
        mus = [0.0, 2.0, 0.5, 2.5, -1.5, 1.5]
    elif scenario == "adver":
        a = T/6
        delta = (math.sqrt(160*math.log(a)/a))
        mus = [0.0, delta, 0, delta, 0, delta]
    else:
        raise ValueError("scenario must be 'easy' or 'md'")

    return taus, mus


def build_equal_spaced_taus(T: int, S: int) -> list[int]:
    """
    Returns taus = [1=tau0, tau1, ..., tauS, tau_{S+1}=T+1]
    with S change points, and (S+1) segments as equal length as possible.
    """
    if S < 0 or S >= T:
        raise ValueError("Require 0 <= S < T.")
    segs = S + 1
    base = T // segs
    rem = T % segs
    lengths = [base + (1 if i < rem else 0) for i in range(segs)]
    taus = [1]
    cur = 1
    for L in lengths:
        cur += L
        taus.append(cur)
    return taus


def build_alternating_mus(S: int, delta: float, start_sign: int = -1) -> list[float]:
    """
    Returns mus = [mu0, ..., muS] alternating between -delta and +delta.
    Jump size is constant (2*delta) at every change.
    """
    mus: list[float] = []
    sign = start_sign
    for _ in range(S + 1):
        mus.append(sign * delta)
        sign *= -1
    return mus


def build_instance(
    T: int,
    *,
    scenario: str,
    short_len: int,
    s_frac: Optional[float],
    delta: float,
):
    """
    If s_frac is provided, use an equal-spaced, alternating-means environment.
    Otherwise, use the fixed-S scenario builder.
    """
    if s_frac is None:
        return build_piecewise_instance(T, scenario=scenario, short_len=short_len)

    if not (0.0 < s_frac < 1.0):
        raise ValueError("--S-frac must be in (0, 1).")
    S = max(1, min(T - 1, int(round(s_frac * T))))
    taus = build_equal_spaced_taus(T, S)
    mus = build_alternating_mus(S, delta=delta, start_sign=-1)
    return taus, mus


def simulate_env(T: int, taus, mus, sigma: float = 1.0, rng=None):
    """
    Generates X_1..X_T with:
      X_t = mu_t + sigma * Z_t,  Z_t ~ N(0,1) i.i.d.
    """
    if rng is None:
        rng = np.random.default_rng()

    mu_t = np.zeros(T + 1)  # indices 1..T used
    for j, mu in enumerate(mus):
        start, end = taus[j], taus[j + 1]
        mu_t[start:end] = mu

    X = mu_t[1:] + sigma * rng.standard_normal(T)
    return X, mu_t[1:]


# -----------------------------
# Threshold rules (ATC)
# -----------------------------
def alpha_r_from_restart(r: int, alpha: float) -> float:
    """Per-restart alpha schedule: alpha_r = (6/pi^2) * alpha / (r+1)^2"""
    return (6.0 / (math.pi**2)) * alpha / ((r + 1) ** 2)


def gamma_paper(t_minus_r: int, sigma: float, alpha_r: float) -> float:
    """
    Paper threshold:
      gamma_t^r = sigma * sqrt( 6 log(t-r) + 2 log(1/alpha_r) + 2 log(pi^2/3) )
    """
    const_term = 2.0 * math.log((math.pi**2) / 3.0)
    return sigma * math.sqrt(
        6.0 * math.log(t_minus_r)
        + 2.0 * math.log(1.0 / alpha_r)
        + const_term
    )


def gamma_constant_factory(gamma0: float, *, scale_by_sigma: bool = True) -> Callable[[int, float, float], float]:
    """
    Constant threshold (for ATC variants):
      gamma_t^r = sigma * gamma0   if scale_by_sigma=True  (default)
      gamma_t^r = gamma0          if scale_by_sigma=False
    """
    gamma0 = float(gamma0)

    if scale_by_sigma:
        def _gamma_const(t_minus_r: int, sigma: float, alpha_r: float) -> float:
            _ = (t_minus_r, alpha_r)
            return float(sigma * gamma0)
        return _gamma_const

    def _gamma_const_abs(t_minus_r: int, sigma: float, alpha_r: float) -> float:
        _ = (t_minus_r, sigma, alpha_r)
        return gamma0
    return _gamma_const_abs


# -----------------------------
# GCP threshold (from attached code)
# -----------------------------
def gcp_gamma_theory(T: int, S: int, sigma: float, alpha: float, scale: float = 1.0) -> float:
    """
    GCP constant threshold ensuring family-wise FA ≤ alpha over the whole horizon:
        γ = scale * σ √{ 2 log( 2 T^2 / ((S+1) α) ) }.

    This is the formula used in your attached code's threshold_gcp_constant.
    """
    T_eff = max(int(T), 1)
    S_plus = max(int(S) + 1, 1)
    a = max(float(alpha), 1e-12)
    val = 3.0 * math.log((2.0 * (T_eff ** 2)) / (S_plus * a))
    return float(scale * sigma * math.sqrt(max(val, 0.0)))


# -----------------------------
# Candidate split selection (scan strategies)
# -----------------------------
def candidate_splits_all(r: int, t: int) -> np.ndarray:
    """Exact scan: all k in {r+1, ..., t-1}."""
    return np.arange(r + 1, t, dtype=int)


def candidate_splits_uniform(r: int, t: int, M: int) -> np.ndarray:
    """Uniform grid: ~M evenly spaced candidate splits in [r+1, t-1]."""
    L = t - r
    if L <= 1:
        return np.array([], dtype=int)
    if M <= 0:
        raise ValueError("M must be positive for uniform candidate splits.")
    M = min(int(M), L - 1)
    k = np.linspace(r + 1, t - 1, num=M, dtype=int)
    return np.unique(k)


def candidate_splits_geom_ends(r: int, t: int, base: float = 2.0, include_midpoint: bool = True) -> np.ndarray:
    """
    Option A: geometric offsets from the ends.

    Candidates include:
      - k = r + d and k = t - d for offsets d = 1, ceil(base), ceil(base^2), ...
      - plus midpoint (optional)

    Uses O(log_{base}(t-r)) candidates.
    """
    if base <= 1.0:
        raise ValueError("geom_ends base must be > 1.0")

    L = t - r
    if L <= 1:
        return np.array([], dtype=int)

    ks = set()

    if include_midpoint:
        ks.add(r + (L // 2))

    d = 1
    while d < L:
        k1 = r + d
        k2 = t - d
        if r < k1 < t:
            ks.add(k1)
        if r < k2 < t:
            ks.add(k2)

        d_next = int(math.ceil(d * base))
        if d_next <= d:
            d_next = d + 1
        d = d_next

    ks.discard(r)
    ks.discard(t)
    return np.array(sorted(ks), dtype=int)


# -----------------------------
# Shared scan statistic: max over k of sqrt(len1*len2/(t-r)) * |mean1-mean2|
# -----------------------------
def scan_statistic_max(
    cs: np.ndarray,  # cumulative sum with cs[0]=0, cs[t]=sum_{i=1..t} X_i
    r: int,
    t: int,
    split_fn: Callable[[int, int], np.ndarray],
) -> float:
    """
    Returns C_t^r = max_{k in candidates} sqrt((k-r)(t-k)/(t-r)) * |mean(r+1:k)-mean(k+1:t)|
    """
    k = split_fn(r, t)
    if k.size == 0:
        return 0.0

    len1 = k - r
    len2 = t - k

    valid = (len1 > 0) & (len2 > 0)
    if not np.any(valid):
        return 0.0
    k = k[valid]
    len1 = len1[valid]
    len2 = len2[valid]

    mean1 = (cs[k] - cs[r]) / len1
    mean2 = (cs[t] - cs[k]) / len2
    vals = np.sqrt(len1 * len2 / (t - r)) * np.abs(mean1 - mean2)
    return float(np.max(vals))


# -----------------------------
# ATC runner
# -----------------------------
def atc_run(
    X: np.ndarray,
    *,
    sigma: float,
    alpha: float,
    split_fn: Callable[[int, int], np.ndarray],
    gamma_fn: Callable[[int, float, float], float],
) -> Dict[str, Any]:
    """
    ATC core:
      alpha_r = (6/pi^2) * alpha / (r+1)^2
      gamma_t^r = gamma_fn(t-r, sigma, alpha_r)
      alarm when C_t^r >= gamma_t^r, then restart r <- t
    """
    T = len(X)
    cs = np.concatenate([[0.0], np.cumsum(X)])

    r = 0
    alpha_r = alpha_r_from_restart(r, alpha)

    hatmu = np.empty(T + 1)
    alarms = []

    for t in range(1, T + 1):
        hatmu[t] = (cs[t] - cs[r]) / (t - r)

        if t >= r + 2:
            C = scan_statistic_max(cs, r, t, split_fn)
            gamma = float(gamma_fn(t - r, sigma, alpha_r))
            if C >= gamma:
                alarms.append(t)
                # Restart at time t: use only X_t for the estimate at this step.
                hatmu[t] = X[t - 1]
                r = t
                alpha_r = alpha_r_from_restart(r, alpha)

    return {"hatmu": hatmu[1:], "alarms": alarms}


# -----------------------------
# GCP runner (from attached code, adapted to our template)
# -----------------------------
def gcp_run(
    X: np.ndarray,
    *,
    sigma: float,
    alpha: float,
    S: int,
    forced_period: int,
    gamma_const: float,
    split_fn: Callable[[int, int], np.ndarray],
) -> Dict[str, Any]:
    """
    GCP:
      - Same scan statistic (max over k) as GC/ATC.
      - Constant threshold gamma_const (computed from theory).
      - Forced periodic restart every 'forced_period' steps since last restart.
      - Forced restarts are NOT alarms.

    Alarm rule:
      if C_t^r >= gamma_const: alarms.append(t); r <- t
      else if (t-r) >= forced_period: r <- t (forced; no alarm)
    """
    _ = S  # S is used outside to define forced_period; kept for clarity
    T = len(X)
    cs = np.concatenate([[0.0], np.cumsum(X)])

    r = 0
    hatmu = np.empty(T + 1)
    alarms = []

    for t in range(1, T + 1):
        hatmu[t] = (cs[t] - cs[r]) / (t - r)

        span = t - r
        if span >= 2:
            C = scan_statistic_max(cs, r, t, split_fn)
            if C >= gamma_const:
                alarms.append(t)
                # Restart at time t: use only X_t for the estimate at this step.
                hatmu[t] = X[t - 1]
                r = t
                continue  # do NOT also force-restart at same time

        # Forced periodic restart (no alarm)
        if span >= forced_period:
            r = t

    return {"hatmu": hatmu[1:], "alarms": alarms}


# -----------------------------
# Sliding window runner
# -----------------------------
def sliding_run(
    X: np.ndarray,
    *,
    window_len: int,
) -> Dict[str, Any]:
    """
    Sliding window estimator:
      hatmu_t = mean of last W samples (or all samples if t < W)
    No alarms are produced.
    """
    T = len(X)
    W = int(window_len)
    if W <= 0:
        raise ValueError("window_len must be positive.")
    cs = np.concatenate([[0.0], np.cumsum(X)])
    hatmu = np.empty(T, dtype=float)

    for t in range(1, T + 1):
        start = max(0, t - W)
        denom = t - start
        hatmu[t - 1] = (cs[t] - cs[start]) / denom

    return {"hatmu": hatmu, "alarms": []}


# -----------------------------
# Discounted mean runner
# -----------------------------
def discount_run(
    X: np.ndarray,
    *,
    discount: float,
) -> Dict[str, Any]:
    """
    Exponentially discounted mean estimator with normalization.
    Weights decay as discount^{t-i} for older samples.
    """
    rho = float(discount)
    if not (0.0 < rho < 1.0):
        raise ValueError("discount must be in (0, 1).")
    T = len(X)
    hatmu = np.empty(T, dtype=float)
    s = 0.0
    w = 0.0
    for t in range(T):
        s = rho * s + X[t]
        w = rho * w + 1.0
        hatmu[t] = s / w
    return {"hatmu": hatmu, "alarms": []}


# -----------------------------
# Metrics
# -----------------------------
def regret_squared(hatmu: np.ndarray, mu_t: np.ndarray) -> float:
    d = hatmu - mu_t
    return float(np.sum(d * d))


def resolve_window_len(*, T: int, S: int, window_len: int, window_formula: Optional[str]) -> int:
    if window_formula in (None, "fixed"):
        return int(window_len)
    if window_formula == "sqrt_TlogT_over_S":
        denom = max(int(S), 1)
        val = (T * math.log(max(T, 2))) / denom
        return max(1, int(0.5*math.ceil(1* math.sqrt(val))))
    raise ValueError(f"Unknown window_formula: {window_formula!r}")


def resolve_discount(*, T: int, S: int, discount: float, discount_formula: Optional[str]) -> float:
    if discount_formula in (None, "fixed"):
        return float(discount)
    if discount_formula == "one_minus_sqrt_S_over_T":
        ratio = S / max(T, 1)
        val = 1.0 - math.sqrt(max(ratio, 0.0))
        return min(max(val, 1e-6), 1.0 - 1e-6)
    raise ValueError(f"Unknown discount_formula: {discount_formula!r}")


def variance_bias_terms(
    hatmu: np.ndarray,
    mu_t: np.ndarray,
    alarms: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Decompose squared error into variance-like and bias-like terms:
      variance-like: (hatmu_t - bar_mu_t^r)^2
      bias-like:     (bar_mu_t^r - mu_t)^2
    where bar_mu_t^r is the running average of true means since last restart r.
    """
    T = len(mu_t)
    mu_cs = np.concatenate([[0.0], np.cumsum(mu_t)])
    alarms_sorted = sorted(int(a) for a in alarms)

    r = 0
    idx_alarm = 0
    var_term = np.empty(T, dtype=float)
    bias_term = np.empty(T, dtype=float)

    for t in range(1, T + 1):
        bar_mu = (mu_cs[t] - mu_cs[r]) / (t - r)
        h = hatmu[t - 1]
        m = mu_t[t - 1]
        var_term[t - 1] = (h - bar_mu) ** 2
        bias_term[t - 1] = (bar_mu - m) ** 2

        if idx_alarm < len(alarms_sorted) and t == alarms_sorted[idx_alarm]:
            r = t
            idx_alarm += 1

    return var_term, bias_term


def count_miss_fa(alarms_list, taus_list):
    """
    Same miss/FA counting logic as in your earlier template.

    Detection windows are the segments after each change:
      windows = [(taus[i], taus[i+1]) for i=1..len(taus)-2]
    miss = number of detection windows with no alarm inside
    fa   = number of alarms before the first change, plus extra alarms
           after the first one within each detection window
    """
    if len(taus_list) < 3:
        return 0, len(alarms_list)

    pre_change = (taus_list[0], taus_list[1])
    windows = [(taus_list[i], taus_list[i + 1]) for i in range(1, len(taus_list) - 1)]
    alarms_sorted = sorted(alarms_list)

    miss = 0
    fa = sum(1 for a in alarms_sorted if pre_change[0] <= a < pre_change[1])
    for w in windows:
        in_window = [a for a in alarms_sorted if w[0] <= a < w[1]]
        if not in_window:
            miss += 1
        else:
            fa += max(0, len(in_window) - 1)
    return miss, fa


def detection_delay_stats(
    alarms: Sequence[int],
    tau_full: Sequence[int],
) -> tuple[float, float, float]:
    """
    Returns:
      - delay_detected_only: average delay over detected changes (NaN if none detected)
      - delay_censored: average delay treating misses as (segment length)
      - detect_rate: fraction of changes that are detected in this run
    """
    alarms = list(map(int, alarms))
    tau_full = list(map(int, tau_full))

    S = len(tau_full) - 2
    if S <= 0:
        return float("nan"), float("nan"), float("nan")

    delays: list[float] = []
    det_delays: list[float] = []
    detected = 0

    for j in range(1, len(tau_full) - 1):
        start = tau_full[j]
        end = tau_full[j + 1]
        a_in = [a for a in alarms if start <= a < end]
        if len(a_in) > 0:
            d = float(a_in[0] - start)
            delays.append(d)
            det_delays.append(d)
            detected += 1
        else:
            d = float(end - start)
            delays.append(d)

    delay_cens = float(np.mean(delays)) if len(delays) else float("nan")
    delay_det = float(np.mean(det_delays)) if len(det_delays) else float("nan")
    detect_rate = float(detected) / float(S)
    return delay_det, delay_cens, detect_rate


# -----------------------------
# Algorithm specification
# -----------------------------
MaxSplitsSpec = Union[None, int, str]  # None | positive int | "logT"


@dataclass(frozen=True)
class AlgoSpec:
    key: str
    label: str
    kind: str  # "atc" or "gcp"
    split_fn_factory: Callable[[int], Callable[[int, int], np.ndarray]]
    gamma_fn: Optional[Callable[[int, float, float], float]] = None  # ATC only
    window_len: Optional[int] = None  # sliding only (fixed window)
    window_formula: Optional[str] = None  # sliding only
    discount: Optional[float] = None  # discount only
    discount_formula: Optional[str] = None  # discount only


def _max_splits_value_from_spec(spec: MaxSplitsSpec, T: int, *, default_if_none: int = 200) -> int:
    if spec is None:
        return int(default_if_none)
    if isinstance(spec, int):
        if spec <= 0:
            raise ValueError("max_splits integer must be positive.")
        return int(spec)
    if isinstance(spec, str) and spec.lower() == "logt":
        if T <= 1:
            return 1
        return max(1, int(math.ceil(math.log(T))))
    raise ValueError(f"Unsupported max_splits spec: {spec!r}")


def make_algo_specs(
    algo_keys: Sequence[str],
    *,
    max_splits_spec: MaxSplitsSpec,
    const_gamma: float,
    geom_base: float,
    gcp_gamma_scale: float,
    window_len: int,
    window_formula: str,
    discount: float,
    discount_formula: str,
) -> list[AlgoSpec]:
    """
    Supported algo_keys:
      - "exact"           : ATC, exact scan + paper threshold
      - "maxsplit"        : ATC, uniform max_splits scan + paper threshold
      - "geom_ends"       : ATC, geom_ends scan + paper threshold
      - "const"           : ATC, exact scan + constant threshold
      - "const_maxsplit"  : ATC, uniform max_splits scan + constant threshold
      - "const_geom_ends" : ATC, geom_ends scan + constant threshold
      - "gcp"             : GCP, exact scan + constant FA-bound threshold + forced periodic restarts
    """
    specs: list[AlgoSpec] = []

    gamma_default = gamma_paper
    gamma_const = gamma_constant_factory(const_gamma, scale_by_sigma=True)

    def exact_split_factory(_T: int) -> Callable[[int, int], np.ndarray]:
        return candidate_splits_all

    def maxsplit_uniform_factory(T: int) -> Callable[[int, int], np.ndarray]:
        M = _max_splits_value_from_spec(max_splits_spec, T, default_if_none=200)

        def _split(r: int, t: int) -> np.ndarray:
            return candidate_splits_uniform(r, t, M)
        return _split

    def geom_ends_factory(_T: int) -> Callable[[int, int], np.ndarray]:
        def _split(r: int, t: int) -> np.ndarray:
            return candidate_splits_geom_ends(r, t, base=geom_base, include_midpoint=True)
        return _split

    def _tag_for_maxsplit() -> str:
        if max_splits_spec is None:
            return "200"
        if isinstance(max_splits_spec, str) and max_splits_spec.lower() == "logt":
            return "ceil(ln T)"
        return str(max_splits_spec)

    for k in algo_keys:
        if k == "exact":
            specs.append(
                AlgoSpec(
                    key="exact",
                    label="ATC algorithm",
                    kind="atc",
                    split_fn_factory=exact_split_factory,
                    gamma_fn=gamma_default,
                )
            )

        elif k == "maxsplit":
            tag = _tag_for_maxsplit()
            key_tag = "logT" if (isinstance(max_splits_spec, str) and max_splits_spec.lower() == "logt") else tag
            specs.append(
                AlgoSpec(
                    key=f"maxsplit_{key_tag}",
                    label=f"ATC (uniform max_splits={tag}, paper γ)",
                    kind="atc",
                    split_fn_factory=maxsplit_uniform_factory,
                    gamma_fn=gamma_default,
                )
            )

        elif k == "geom_ends":
            specs.append(
                AlgoSpec(
                    key=f"geom_ends_b{geom_base:g}",
                    label=f"Efficient ATC",
                    kind="atc",
                    split_fn_factory=geom_ends_factory,
                    gamma_fn=gamma_default,
                )
            )

        elif k == "const":
            specs.append(
                AlgoSpec(
                    key="exact_const",
                    label=f"ATC (exact scan, const γ={const_gamma:g}·σ)",
                    kind="atc",
                    split_fn_factory=exact_split_factory,
                    gamma_fn=gamma_const,
                )
            )

        elif k == "const_maxsplit":
            tag = _tag_for_maxsplit()
            key_tag = "logT" if (isinstance(max_splits_spec, str) and max_splits_spec.lower() == "logt") else tag
            specs.append(
                AlgoSpec(
                    key=f"maxsplit_{key_tag}_const",
                    label=f"ATC (uniform max_splits={tag}, const γ={const_gamma:g}·σ)",
                    kind="atc",
                    split_fn_factory=maxsplit_uniform_factory,
                    gamma_fn=gamma_const,
                )
            )

        elif k == "const_geom_ends":
            specs.append(
                AlgoSpec(
                    key=f"geom_ends_b{geom_base:g}_const",
                    label=f"ATC (geom_ends base={geom_base:g}, const γ={const_gamma:g}·σ)",
                    kind="atc",
                    split_fn_factory=geom_ends_factory,
                    gamma_fn=gamma_const,
                )
            )

        elif k == "gcp":
            # GCP is defined with the GC/ATC scan statistic and full scan over k (exact)
            # + forced periodic restart + theory constant threshold.
            label_scale = "" if abs(gcp_gamma_scale - 1.0) < 1e-12 else f", scale={gcp_gamma_scale:g}"
            specs.append(
                AlgoSpec(
                    key="gcp",
                    label=f"GCP (period=floor(T/(S+1)), γ_th{label_scale})",
                    kind="gcp",
                    split_fn_factory=exact_split_factory,
                    gamma_fn=None,
                )
            )

        elif k == "sliding":
            if window_formula == "fixed":
                if window_len <= 0:
                    raise ValueError("--window must be a positive integer.")
                label = f"Sliding window (W={window_len})"
                key = f"sliding_w{window_len}"
            else:
                label = "Sliding window"
                key = "sliding_auto"
            specs.append(
                AlgoSpec(
                    key=key,
                    label=label,
                    kind="sliding",
                    split_fn_factory=exact_split_factory,
                    gamma_fn=None,
                    window_len=int(window_len),
                    window_formula=window_formula,
                )
            )

        elif k == "discount":
            if discount_formula == "fixed":
                if not (0.0 < discount < 1.0):
                    raise ValueError("--discount must be in (0, 1).")
                label = f"Discounted mean (rho={discount:g})"
                key = f"discount_r{discount:g}"
            else:
                label = "Discounted mean"
                key = "discount_auto"
            specs.append(
                AlgoSpec(
                    key=key,
                    label=label,
                    kind="discount",
                    split_fn_factory=exact_split_factory,
                    gamma_fn=None,
                    discount=float(discount),
                    discount_formula=discount_formula,
                )
            )

        else:
            raise ValueError(
                f"Unknown algo key: {k!r}. "
                "Use 'exact', 'maxsplit', 'geom_ends', 'const', 'const_maxsplit', "
                "'const_geom_ends', 'gcp', 'sliding', and/or 'discount'."
            )

    return specs


# -----------------------------
# MC experiment: average regret vs T
# -----------------------------
def mc_regret_for_T_multi_algos(
    *,
    T: int,
    scenario: str,
    algo_specs: Sequence[AlgoSpec],
    n_mc: int,
    sigma: float,
    alpha: float,
    seed: int,
    gcp_gamma_scale: float,
    s_frac: Optional[float],
    delta: float,
    debug_alarms: bool = False,
    debug_alarms_n: int = 5,
    debug_alarms_algo: Optional[str] = None,
):
    """
    Runs the SAME MC environment draws for all algorithms (common random numbers),
    and returns a list of result dicts (one per algorithm).
    """
    taus, mus = build_instance(
        T,
        scenario=scenario,
        short_len=10,
        s_frac=s_frac,
        delta=delta,
    )
    S = len(mus) - 1  # number of change points

    base_rng = np.random.default_rng(seed)
    rep_seeds = base_rng.integers(0, 2**32 - 1, size=n_mc, dtype=np.uint32)

    # Resolve split_fn once per algo for this horizon T
    split_fn_for_algo = {a.key: a.split_fn_factory(T) for a in algo_specs}

    # GCP: constants depending on T,S
    forced_period = max(int(T // (S + 1)), 1) if S >= 0 else max(T, 1)
    gcp_gamma = gcp_gamma_theory(T=T, S=S, sigma=sigma, alpha=alpha, scale=gcp_gamma_scale)

    regrets = {a.key: np.empty(n_mc, dtype=float) for a in algo_specs}
    misses = {a.key: np.empty(n_mc, dtype=float) for a in algo_specs}
    fas = {a.key: np.empty(n_mc, dtype=float) for a in algo_specs}
    delay_det = {a.key: np.full(n_mc, np.nan, dtype=float) for a in algo_specs}
    delay_cens = {a.key: np.full(n_mc, np.nan, dtype=float) for a in algo_specs}
    detect_rate = {a.key: np.full(n_mc, np.nan, dtype=float) for a in algo_specs}

    for i in range(n_mc):
        rng = np.random.default_rng(int(rep_seeds[i]))
        X, mu_t = simulate_env(T, taus, mus, sigma=sigma, rng=rng)

        for a in algo_specs:
            if a.kind == "atc":
                out = atc_run(
                    X,
                    sigma=sigma,
                    alpha=alpha,
                    split_fn=split_fn_for_algo[a.key],
                    gamma_fn=a.gamma_fn,  # type: ignore[arg-type]
                )
            elif a.kind == "gcp":
                out = gcp_run(
                    X,
                    sigma=sigma,
                    alpha=alpha,
                    S=S,
                    forced_period=forced_period,
                    gamma_const=gcp_gamma,
                    split_fn=split_fn_for_algo[a.key],
                )
            elif a.kind == "sliding":
                if a.window_len is None:
                    raise RuntimeError("sliding algorithm requires window_len.")
                W = resolve_window_len(
                    T=T,
                    S=S,
                    window_len=a.window_len,
                    window_formula=a.window_formula,
                )
                out = sliding_run(X, window_len=W)
            elif a.kind == "discount":
                if a.discount is None:
                    raise RuntimeError("discount algorithm requires discount.")
                rho = resolve_discount(
                    T=T,
                    S=S,
                    discount=a.discount,
                    discount_formula=a.discount_formula,
                )
                out = discount_run(X, discount=rho)
            else:
                raise RuntimeError(f"Unknown algo kind: {a.kind!r}")

            regrets[a.key][i] = regret_squared(out["hatmu"], mu_t)
            m, f = count_miss_fa(out["alarms"], taus)
            misses[a.key][i] = m
            fas[a.key][i] = f
            d_det, d_cens, d_rate = detection_delay_stats(out["alarms"], taus)
            delay_det[a.key][i] = d_det
            delay_cens[a.key][i] = d_cens
            detect_rate[a.key][i] = d_rate

            if debug_alarms and i < debug_alarms_n:
                if debug_alarms_algo is None or a.key == debug_alarms_algo:
                    _print_alarm_debug(
                        alarms=out["alarms"],
                        taus=taus,
                        algo_label=a.label,
                        rep_index=i,
                    )

    results = []
    for a in algo_specs:
        r = regrets[a.key]
        m = misses[a.key]
        f = fas[a.key]
        dd = delay_det[a.key]
        dc = delay_cens[a.key]
        dr = detect_rate[a.key]

        mean = float(np.mean(r))
        se = float(np.std(r, ddof=1) / math.sqrt(n_mc))

        mean_miss = float(np.mean(m))
        se_miss = float(np.std(m, ddof=1) / math.sqrt(n_mc))

        mean_fa = float(np.mean(f))
        se_fa = float(np.std(f, ddof=1) / math.sqrt(n_mc))
        mean_dd = float(np.mean(dd))
        se_dd = float(np.std(dd, ddof=1) / math.sqrt(n_mc)) if n_mc >= 2 else float("nan")
        mean_dc = float(np.mean(dc))
        se_dc = float(np.std(dc, ddof=1) / math.sqrt(n_mc)) if n_mc >= 2 else float("nan")
        mean_dr = float(np.mean(dr))
        se_dr = float(np.std(dr, ddof=1) / math.sqrt(n_mc)) if n_mc >= 2 else float("nan")

        row = {
            "scenario": scenario,
            "algo_key": a.key,
            "algo_label": a.label,
            "T": int(T),
            "logT": float(math.log(T)),
            "mean_regret": mean,
            "se": se,
            "mean_miss": mean_miss,
            "se_miss": se_miss,
            "mean_fa": mean_fa,
            "se_fa": se_fa,
            "mean_delay_det": mean_dd,
            "se_delay_det": se_dd,
            "mean_delay_cens": mean_dc,
            "se_delay_cens": se_dc,
            "mean_detect_rate": mean_dr,
            "se_detect_rate": se_dr,
        }

        # helpful debug info for GCP
        if a.kind == "gcp":
            row["gcp_forced_period"] = int(forced_period)
            row["gcp_gamma"] = float(gcp_gamma)

        results.append(row)

    return results


def scaling_experiment(
    *,
    T_list: Sequence[int],
    scenarios: Sequence[str],
    algo_specs: Sequence[AlgoSpec],
    n_mc: int,
    sigma: float,
    alpha: float,
    seed: int,
    gcp_gamma_scale: float,
    s_frac: Optional[float],
    delta: float,
    debug_alarms: bool = False,
    debug_alarms_n: int = 5,
    debug_alarms_algo: Optional[str] = None,
    debug_alarms_T: Optional[int] = None,
):
    results = []

    for scenario in scenarios:
        for T in T_list:
            perT_seed = int(seed + 17 * int(T))

            per_algo = mc_regret_for_T_multi_algos(
                T=T,
                scenario=scenario,
                algo_specs=algo_specs,
                n_mc=n_mc,
                sigma=sigma,
                alpha=alpha,
                seed=perT_seed,
                gcp_gamma_scale=gcp_gamma_scale,
                s_frac=s_frac,
                delta=delta,
                debug_alarms=debug_alarms and (debug_alarms_T is None or int(debug_alarms_T) == int(T)),
                debug_alarms_n=debug_alarms_n,
                debug_alarms_algo=debug_alarms_algo,
            )
            results.extend(per_algo)

            for row in per_algo:
                extra = ""
                if row["algo_key"] == "gcp":
                    extra = f"  (P={row['gcp_forced_period']}, γ={row['gcp_gamma']:.3f})"
                print(
                    f"[{row['scenario']}] {row['algo_label']:<52s} "
                    f"T={row['T']:6d}  mean={row['mean_regret']:10.3f}  SE={row['se']:8.3f}  "
                    f"miss={row['mean_miss']:6.2f}±{row['se_miss']:5.2f}  "
                    f"FA={row['mean_fa']:6.2f}±{row['se_fa']:5.2f}"
                    f"  delay_det={row['mean_delay_det']:.1f}±{row['se_delay_det']:.1f}"
                    f"  delay_cens={row['mean_delay_cens']:.1f}±{row['se_delay_cens']:.1f}"
                    f"  detect_rate={row['mean_detect_rate']:.2f}±{row['se_detect_rate']:.2f}"
                    f"{extra}"
                )

    return results


# -----------------------------
# Plotting
# -----------------------------
def _unique_in_order(items: Sequence[str]) -> list[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def plot_regret_vs_logT(
    results,
    *,
    scenario: str,
    out_path: str,
    add_fit: bool = True,
):
    rows = [r for r in results if r["scenario"] == scenario]
    if not rows:
        raise ValueError(f"No results for scenario={scenario!r}.")

    algo_order = _unique_in_order([r["algo_key"] for r in rows])

    plt.figure(figsize=(8, 5))
    for algo_key in algo_order:
        sub = [r for r in rows if r["algo_key"] == algo_key]
        sub = sorted(sub, key=lambda d: d["T"])

        x = np.array([r["logT"] for r in sub])
        y = np.array([r["mean_regret"] for r in sub])
        se = np.array([r["se"] for r in sub])
        label = sub[0]["algo_label"]

        plt.errorbar(
            x,
            y,
            yerr=1.96 * se,
            marker="o",
            linestyle="-",
            capsize=3,
            label=label,
        )

        if add_fit and len(x) >= 2:
            A = np.vstack([x, np.ones_like(x)]).T
            a, b = np.linalg.lstsq(A, y, rcond=None)[0]
            xx = np.linspace(x.min(), x.max(), 200)
            plt.plot(
                xx, a * xx + b,
                linestyle="--",
                linewidth=1,
                label=None,
            )

    plt.xlabel("log(T)", fontsize=16)
    plt.ylabel("Cumulative regret", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------------
# Cumulative regret vs time (single T)
# -----------------------------
def mc_cumregret_for_T_multi_algos(
    *,
    T: int,
    scenario: str,
    algo_specs: Sequence[AlgoSpec],
    n_mc: int,
    sigma: float,
    alpha: float,
    seed: int,
    gcp_gamma_scale: float,
    s_frac: Optional[float],
    delta: float,
):
    """
    For a single horizon T, compute mean cumulative regret over time:
      cumregret(t) = sum_{s=1}^t (hatmu_s - mu_s)^2
    Uses online mean/variance to avoid storing all MC traces.
    """
    taus, mus = build_instance(
        T,
        scenario=scenario,
        short_len=10,
        s_frac=s_frac,
        delta=delta,
    )
    S = len(mus) - 1

    base_rng = np.random.default_rng(seed)
    rep_seeds = base_rng.integers(0, 2**32 - 1, size=n_mc, dtype=np.uint32)

    split_fn_for_algo = {a.key: a.split_fn_factory(T) for a in algo_specs}

    forced_period = max(int(T // (S + 1)), 1) if S >= 0 else max(T, 1)
    gcp_gamma = gcp_gamma_theory(T=T, S=S, sigma=sigma, alpha=alpha, scale=gcp_gamma_scale)

    means = {a.key: np.zeros(T, dtype=float) for a in algo_specs}
    m2s = {a.key: np.zeros(T, dtype=float) for a in algo_specs}
    counts = {a.key: 0 for a in algo_specs}

    for i in range(n_mc):
        rng = np.random.default_rng(int(rep_seeds[i]))
        X, mu_t = simulate_env(T, taus, mus, sigma=sigma, rng=rng)

        for a in algo_specs:
            if a.kind == "atc":
                out = atc_run(
                    X,
                    sigma=sigma,
                    alpha=alpha,
                    split_fn=split_fn_for_algo[a.key],
                    gamma_fn=a.gamma_fn,  # type: ignore[arg-type]
                )
            elif a.kind == "gcp":
                out = gcp_run(
                    X,
                    sigma=sigma,
                    alpha=alpha,
                    S=S,
                    forced_period=forced_period,
                    gamma_const=gcp_gamma,
                    split_fn=split_fn_for_algo[a.key],
                )
            elif a.kind == "sliding":
                if a.window_len is None:
                    raise RuntimeError("sliding algorithm requires window_len.")
                W = resolve_window_len(
                    T=T,
                    S=S,
                    window_len=a.window_len,
                    window_formula=a.window_formula,
                )
                out = sliding_run(X, window_len=W)
            elif a.kind == "discount":
                if a.discount is None:
                    raise RuntimeError("discount algorithm requires discount.")
                out = discount_run(X, discount=a.discount)
            else:
                raise RuntimeError(f"Unknown algo kind: {a.kind!r}")

            err = out["hatmu"] - mu_t
            cum = np.cumsum(err * err)

            n = counts[a.key] + 1
            delta = cum - means[a.key]
            means[a.key] += delta / n
            m2s[a.key] += delta * (cum - means[a.key])
            counts[a.key] = n

    results = []
    for a in algo_specs:
        n = counts[a.key]
        if n <= 1:
            se = np.zeros(T, dtype=float)
        else:
            se = np.sqrt(m2s[a.key] / (n - 1)) / math.sqrt(n)

        results.append(
            {
                "scenario": scenario,
                "algo_key": a.key,
                "algo_label": a.label,
                "T": int(T),
                "mean_cumregret": means[a.key],
                "se_cumregret": se,
            }
        )

    return results


def plot_cumregret_vs_t(
    results,
    *,
    scenario: str,
    out_path: str,
    change_points: Optional[Sequence[int]] = None,
    alarms: Optional[Sequence[int]] = None,
    var_bias_results=None,
):
    rows = [r for r in results if r["scenario"] == scenario]
    if not rows:
        raise ValueError(f"No results for scenario={scenario!r}.")

    algo_order = _unique_in_order([r["algo_key"] for r in rows])

    plt.figure(figsize=(8, 5))
    for i, algo_key in enumerate(algo_order):
        sub = [r for r in rows if r["algo_key"] == algo_key]
        row = sub[0]
        y = row["mean_cumregret"]
        se = row["se_cumregret"]
        x = np.arange(1, len(y) + 1)

        lbl = "ATC algorithm regret" if i == 0 else None
        plt.plot(x, y, linewidth=2, label=lbl)
        if np.any(se > 0):
            plt.fill_between(x, y - 1.96 * se, y + 1.96 * se, alpha=0.2)

    if var_bias_results is not None:
        vb_rows = [r for r in var_bias_results if r["scenario"] == scenario]
        vb_order = _unique_in_order([r["algo_key"] for r in vb_rows])
        for i, algo_key in enumerate(vb_order):
            sub = [r for r in vb_rows if r["algo_key"] == algo_key]
            row = sub[0]
            x = np.arange(1, len(row["mean_cum_var"]) + 1)
            plt.plot(
                x,
                row["mean_cum_var"],
                linewidth=1.5,
                linestyle="--",
                label="variance" if i == 0 else None,
            )
            plt.plot(
                x,
                row["mean_cum_bias"],
                linewidth=1.5,
                linestyle=":",
                color="tab:red",
                label="bias" if i == 0 else None,
            )

    if change_points:
        first_cp = True
        for tau in change_points:
            lbl = "change points" if first_cp else None
            plt.axvline(
                tau,
                color="tab:green",
                alpha=0.9,
                linewidth=1,
                linestyle=":",
                label=lbl,
            )
            first_cp = False

    if alarms:
        first_alarm = True
        for a in alarms:
            lbl = "alarms" if first_alarm else None
            plt.axvline(a, color="black", alpha=0.8, linewidth=1.2, label=lbl)
            first_alarm = False

    plt.xlabel("time t", fontsize=16)
    plt.ylabel("Cumulative regret", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=16, loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def mc_var_bias_for_T_multi_algos(
    *,
    T: int,
    scenario: str,
    algo_specs: Sequence[AlgoSpec],
    n_mc: int,
    sigma: float,
    alpha: float,
    seed: int,
    gcp_gamma_scale: float,
    s_frac: Optional[float],
    delta: float,
):
    """
    For a single horizon T, compute mean cumulative variance-like and bias-like
    errors over time (see variance_bias_terms).
    """
    taus, mus = build_instance(
        T,
        scenario=scenario,
        short_len=10,
        s_frac=s_frac,
        delta=delta,
    )
    S = len(mus) - 1

    base_rng = np.random.default_rng(seed)
    rep_seeds = base_rng.integers(0, 2**32 - 1, size=n_mc, dtype=np.uint32)

    split_fn_for_algo = {a.key: a.split_fn_factory(T) for a in algo_specs}

    forced_period = max(int(T // (S + 1)), 1) if S >= 0 else max(T, 1)
    gcp_gamma = gcp_gamma_theory(T=T, S=S, sigma=sigma, alpha=alpha, scale=gcp_gamma_scale)

    mean_var = {a.key: np.zeros(T, dtype=float) for a in algo_specs}
    mean_bias = {a.key: np.zeros(T, dtype=float) for a in algo_specs}
    m2_var = {a.key: np.zeros(T, dtype=float) for a in algo_specs}
    m2_bias = {a.key: np.zeros(T, dtype=float) for a in algo_specs}
    counts = {a.key: 0 for a in algo_specs}

    for i in range(n_mc):
        rng = np.random.default_rng(int(rep_seeds[i]))
        X, mu_t = simulate_env(T, taus, mus, sigma=sigma, rng=rng)

        for a in algo_specs:
            if a.kind == "atc":
                out = atc_run(
                    X,
                    sigma=sigma,
                    alpha=alpha,
                    split_fn=split_fn_for_algo[a.key],
                    gamma_fn=a.gamma_fn,  # type: ignore[arg-type]
                )
            elif a.kind == "gcp":
                out = gcp_run(
                    X,
                    sigma=sigma,
                    alpha=alpha,
                    S=S,
                    forced_period=forced_period,
                    gamma_const=gcp_gamma,
                    split_fn=split_fn_for_algo[a.key],
                )
            elif a.kind == "sliding":
                raise ValueError("Variance/bias decomposition is not defined for sliding window.")
            elif a.kind == "discount":
                raise ValueError("Variance/bias decomposition is not defined for discount.")
            else:
                raise RuntimeError(f"Unknown algo kind: {a.kind!r}")

            var_term, bias_term = variance_bias_terms(out["hatmu"], mu_t, out["alarms"])
            cum_var = np.cumsum(var_term)
            cum_bias = np.cumsum(bias_term)

            n = counts[a.key] + 1
            delta_v = cum_var - mean_var[a.key]
            delta_b = cum_bias - mean_bias[a.key]
            mean_var[a.key] += delta_v / n
            mean_bias[a.key] += delta_b / n
            m2_var[a.key] += delta_v * (cum_var - mean_var[a.key])
            m2_bias[a.key] += delta_b * (cum_bias - mean_bias[a.key])
            counts[a.key] = n

    results = []
    for a in algo_specs:
        n = counts[a.key]
        if n <= 1:
            se_var = np.zeros(T, dtype=float)
            se_bias = np.zeros(T, dtype=float)
        else:
            se_var = np.sqrt(m2_var[a.key] / (n - 1)) / math.sqrt(n)
            se_bias = np.sqrt(m2_bias[a.key] / (n - 1)) / math.sqrt(n)

        results.append(
            {
                "scenario": scenario,
                "algo_key": a.key,
                "algo_label": a.label,
                "T": int(T),
                "mean_cum_var": mean_var[a.key],
                "se_cum_var": se_var,
                "mean_cum_bias": mean_bias[a.key],
                "se_cum_bias": se_bias,
            }
        )

    return results


def plot_var_bias_vs_t(
    results,
    *,
    scenario: str,
    out_path: str,
):
    rows = [r for r in results if r["scenario"] == scenario]
    if not rows:
        raise ValueError(f"No results for scenario={scenario!r}.")

    algo_order = _unique_in_order([r["algo_key"] for r in rows])

    plt.figure(figsize=(8, 5))
    for algo_key in algo_order:
        sub = [r for r in rows if r["algo_key"] == algo_key]
        row = sub[0]
        x = np.arange(1, len(row["mean_cum_var"]) + 1)

        plt.plot(x, row["mean_cum_var"], linewidth=2, label=f"variance: {row['algo_label']}")
        plt.plot(x, row["mean_cum_bias"], linewidth=2, linestyle="--", label=f"bias: {row['algo_label']}")

    plt.xlabel("time t")
    plt.ylabel("cumulative error")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_environment_example(
    *,
    T: int,
    scenario: str,
    sigma: float,
    seed: int,
    out_path: str,
    s_frac: Optional[float],
    delta: float,
):
    taus, mus = build_instance(
        T,
        scenario=scenario,
        short_len=10,
        s_frac=s_frac,
        delta=delta,
    )
    rng = np.random.default_rng(seed)
    X, mu_t = simulate_env(T, taus, mus, sigma=sigma, rng=rng)

    t = np.arange(1, T + 1)
    plt.figure(figsize=(9, 4))
    plt.plot(t, X, color="tab:gray", alpha=0.6, linewidth=1, label="observations")
    plt.plot(t, mu_t, color="tab:blue", linewidth=2.5, label="true mean")

    first_cp = True
    for tau in taus[1:-1]:
        lbl = "change point" if first_cp else None
        plt.axvline(
            tau,
            color="tab:green",
            alpha=0.9,
            linewidth=0.5,
            linestyle=":",
            label=lbl,
        )
        first_cp = False

    plt.title(f"Environment example (T={T}, scenario={scenario})")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _alarms_for_single_run(
    *,
    T: int,
    scenario: str,
    sigma: float,
    alpha: float,
    seed: int,
    algo_spec: AlgoSpec,
    gcp_gamma_scale: float,
    s_frac: Optional[float],
    delta: float,
):
    taus, mus = build_instance(
        T,
        scenario=scenario,
        short_len=10,
        s_frac=s_frac,
        delta=delta,
    )
    S = len(mus) - 1
    rng = np.random.default_rng(seed)
    X, _ = simulate_env(T, taus, mus, sigma=sigma, rng=rng)

    split_fn = algo_spec.split_fn_factory(T)
    if algo_spec.kind == "atc":
        out = atc_run(
            X,
            sigma=sigma,
            alpha=alpha,
            split_fn=split_fn,
            gamma_fn=algo_spec.gamma_fn,  # type: ignore[arg-type]
        )
    elif algo_spec.kind == "gcp":
        forced_period = max(int(T // (S + 1)), 1) if S >= 0 else max(T, 1)
        gcp_gamma = gcp_gamma_theory(T=T, S=S, sigma=sigma, alpha=alpha, scale=gcp_gamma_scale)
        out = gcp_run(
            X,
            sigma=sigma,
            alpha=alpha,
            S=S,
            forced_period=forced_period,
            gamma_const=gcp_gamma,
            split_fn=split_fn,
        )
    elif algo_spec.kind == "sliding":
        if algo_spec.window_len is None:
            raise RuntimeError("sliding algorithm requires window_len.")
        W = resolve_window_len(
            T=T,
            S=S,
            window_len=algo_spec.window_len,
            window_formula=algo_spec.window_formula,
        )
        out = sliding_run(X, window_len=W)
    elif algo_spec.kind == "discount":
        if algo_spec.discount is None:
            raise RuntimeError("discount algorithm requires discount.")
        rho = resolve_discount(
            T=T,
            S=S,
            discount=algo_spec.discount,
            discount_formula=algo_spec.discount_formula,
        )
        out = discount_run(X, discount=rho)
    else:
        raise RuntimeError(f"Unknown algo kind: {algo_spec.kind!r}")

    return taus, out["alarms"]


def _print_alarm_debug(*, alarms, taus, algo_label: str, rep_index: int):
    if len(taus) < 3:
        print(f"[debug-alarms] rep={rep_index} {algo_label}: insufficient taus")
        return

    pre_change = (taus[0], taus[1])
    windows = [(taus[i], taus[i + 1]) for i in range(1, len(taus) - 1)]
    alarms_sorted = sorted(alarms)

    pre_alarms = [a for a in alarms_sorted if pre_change[0] <= a < pre_change[1]]
    parts = [f"[debug-alarms] rep={rep_index} {algo_label}"]
    parts.append(f"pre_change={pre_change} pre_alarms={pre_alarms}")

    for idx, w in enumerate(windows, start=1):
        in_window = [a for a in alarms_sorted if w[0] <= a < w[1]]
        tp = in_window[0] if in_window else None
        extra = in_window[1:] if len(in_window) > 1 else []
        parts.append(f"seg{idx}={w} tp={tp} extra={extra}")

    print(" | ".join(parts))


def plot_alarms_on_environment(
    *,
    T: int,
    scenario: str,
    sigma: float,
    alpha: float,
    seed: int,
    algo_spec: AlgoSpec,
    gcp_gamma_scale: float,
    out_path: str,
    s_frac: Optional[float],
    delta: float,
):
    taus, mus = build_instance(
        T,
        scenario=scenario,
        short_len=10,
        s_frac=s_frac,
        delta=delta,
    )
    S = len(mus) - 1

    rng = np.random.default_rng(seed)
    X, mu_t = simulate_env(T, taus, mus, sigma=sigma, rng=rng)

    split_fn = algo_spec.split_fn_factory(T)
    if algo_spec.kind == "atc":
        out = atc_run(
            X,
            sigma=sigma,
            alpha=alpha,
            split_fn=split_fn,
            gamma_fn=algo_spec.gamma_fn,  # type: ignore[arg-type]
        )
    elif algo_spec.kind == "gcp":
        forced_period = max(int(T // (S + 1)), 1) if S >= 0 else max(T, 1)
        gcp_gamma = gcp_gamma_theory(T=T, S=S, sigma=sigma, alpha=alpha, scale=gcp_gamma_scale)
        out = gcp_run(
            X,
            sigma=sigma,
            alpha=alpha,
            S=S,
            forced_period=forced_period,
            gamma_const=gcp_gamma,
            split_fn=split_fn,
        )
    elif algo_spec.kind == "sliding":
        if algo_spec.window_len is None:
            raise RuntimeError("sliding algorithm requires window_len.")
        W = resolve_window_len(
            T=T,
            S=S,
            window_len=algo_spec.window_len,
            window_formula=algo_spec.window_formula,
        )
        out = sliding_run(X, window_len=W)
    elif algo_spec.kind == "discount":
        if algo_spec.discount is None:
            raise RuntimeError("discount algorithm requires discount.")
        rho = resolve_discount(
            T=T,
            S=S,
            discount=algo_spec.discount,
            discount_formula=algo_spec.discount_formula,
        )
        out = discount_run(X, discount=rho)
    else:
        raise RuntimeError(f"Unknown algo kind: {algo_spec.kind!r}")

    alarms = out["alarms"]
    if len(taus) < 3:
        windows = []
        pre_change = (1, T + 1)
    else:
        pre_change = (taus[0], taus[1])
        windows = [(taus[i], taus[i + 1]) for i in range(1, len(taus) - 1)]
    alarms_sorted = sorted(alarms)
    tp_set = set()
    fa_set = set()

    for a in alarms_sorted:
        if pre_change[0] <= a < pre_change[1]:
            fa_set.add(a)

    for w in windows:
        in_window = [a for a in alarms_sorted if w[0] <= a < w[1]]
        if in_window:
            tp_set.add(in_window[0])
            for a in in_window[1:]:
                fa_set.add(a)

    t = np.arange(1, T + 1)
    plt.figure(figsize=(9, 4))
    plt.plot(t, X, color="tab:gray", alpha=0.6, linewidth=1, label="observations")
    plt.plot(t, mu_t, color="tab:blue", linewidth=2.0, label="true mean")

    first_cp = True
    for tau in taus[1:-1]:
        lbl = "change points" if first_cp else None
        plt.axvline(
            tau,
            color="tab:green",
            alpha=0.9,
            linewidth=2,
            linestyle=":",
            label=lbl,
        )
        first_cp = False

    for a in alarms_sorted:
        if a in fa_set:
            lbl = "FA alarm"
        else:
            lbl = "alarms"
        plt.axvline(a, color="black", alpha=0.8, linewidth=1.5, label=lbl)

    plt.grid(True, alpha=0.3)

    handles, labels = plt.gca().get_legend_handles_labels()
    seen = set()
    uniq_h = []
    uniq_l = []
    for h, l in zip(handles, labels):
        if l not in seen:
            uniq_h.append(h)
            uniq_l.append(l)
            seen.add(l)
    plt.legend(uniq_h, uniq_l, fontsize=12)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------------
# CLI parsing helpers
# -----------------------------
def parse_max_splits_arg(v: Optional[str]) -> MaxSplitsSpec:
    """
    Accepts:
      - None (arg omitted)
      - positive integer as string, e.g. "200"
      - "logT" (case-insensitive)

    Returns:
      - None | int | "logT"
    """
    if v is None:
        return None
    s = str(v).strip()
    if s == "":
        return None
    s_low = s.lower()

    if s_low in {"none", "null"}:
        return None
    if s_low in {"logt", "log_t", "log"}:
        return "logT"

    try:
        n = int(s)
    except ValueError as e:
        raise argparse.ArgumentTypeError("max-splits must be an int, 'logT', or omitted.") from e

    if n <= 0:
        raise argparse.ArgumentTypeError("max-splits must be a positive integer.")
    return n


# -----------------------------
# Main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="ATC scaling experiment (modular) + GCP.")

    p.add_argument(
        "--algos",
        nargs="+",
        default=["exact"],
        choices=[
            "exact", "maxsplit", "geom_ends",
            "const", "const_maxsplit", "const_geom_ends",
            "gcp",
            "sliding",
            "discount",
        ],
        help=(
            "Which algorithm variants to run. Choose one or more.\n"
            "  exact            : ATC exact scan + paper threshold\n"
            "  maxsplit         : ATC uniform max_splits + paper threshold\n"
            "  geom_ends        : ATC geometric-ends scan + paper threshold\n"
            "  const            : ATC exact scan + constant threshold\n"
            "  const_maxsplit   : ATC uniform max_splits + constant threshold\n"
            "  const_geom_ends  : ATC geom_ends scan + constant threshold\n"
            "  gcp              : GCP (forced periodic restarts + theory constant threshold)\n"
            "  sliding          : Sliding window mean estimator (no alarms)\n"
            "  discount         : Discounted mean estimator (no alarms)\n"
        ),
    )

    p.add_argument(
        "--max-splits",
        type=parse_max_splits_arg,
        default=None,
        help=(
            "Used only if a maxsplit variant is selected. "
            "Provide a positive int (e.g. 200) or 'logT' to use max_splits=ceil(ln(T)). "
            "If omitted and maxsplit is used, defaults to 200."
        ),
    )

    p.add_argument(
        "--geom-base",
        type=float,
        default=2.0,
        help=(
            "Geometric base for geom_ends candidate offsets. Must be > 1. "
            "base=2.0 gives dyadic offsets (1,2,4,8,...). Smaller base => more candidates."
        ),
    )

    p.add_argument(
        "--const-gamma",
        type=float,
        default=6.0,
        help="Constant threshold parameter gamma0 used as γ = gamma0 * σ (default 6.0).",
    )

    p.add_argument(
        "--gcp-gamma-scale",
        type=float,
        default=1.0,
        help=(
            "Scale factor applied to the GCP theory threshold. "
            "GCP uses γ = scale * σ * sqrt(2 log(2 T^2 / ((S+1) α))). Default 1.0."
        ),
    )
    p.add_argument(
        "--window",
        type=int,
        default=50,
        help="Sliding window length (used only with --algos sliding).",
    )
    p.add_argument(
        "--window-formula",
        type=str,
        default="fixed",
        choices=["fixed", "sqrt_TlogT_over_S"],
        help="How to set the sliding window length when --algos sliding is used.",
    )
    p.add_argument(
        "--discount",
        type=float,
        default=0.98,
        help="Discount factor in (0,1) for the discounted mean estimator.",
    )
    p.add_argument(
        "--discount-formula",
        type=str,
        default="fixed",
        choices=["fixed", "one_minus_sqrt_S_over_T"],
        help="How to set the discount factor when --algos discount is used.",
    )

    p.add_argument(
        "--scenario",
        type=str,
        default="md",
        choices=["md", "easy", "adver"],
        help="Scenario to run. Default is 'md'.",
    )
    p.add_argument(
        "--S-frac",
        type=float,
        default=None,
        help=(
            "If set, use S = round(S-frac * T) change points with equal spacing "
            "and alternating means (overrides scenario)."
        ),
    )
    p.add_argument(
        "--delta",
        type=float,
        default=2.0,
        help="Mean magnitude used when --S-frac is provided (jumps are size 2*delta).",
    )

    p.add_argument(
        "--T-list",
        nargs="+",
        type=int,
        default=[600, 1200, 2400, 4800, 7000, 9000],
        help="List of horizons T.",
    )

    p.add_argument("--n-mc", type=int, default=1000, help="MC replications.")
    p.add_argument("--sigma", type=float, default=1.0, help="Noise std / sub-Gaussian proxy.")
    p.add_argument("--alpha", type=float, default=0.05, help="Alpha level (used by ATC and GCP theory threshold).")
    p.add_argument("--seed", type=int, default=123, help="Base seed.")

    p.add_argument("--out-dir", type=str, default="figures", help="Output directory for plots.")
    p.add_argument("--no-fit", action="store_true", help="Disable linear fit lines in the plot.")
    p.add_argument(
        "--skip-logT",
        action="store_true",
        help="Skip the regret_vs_logT plot.",
    )
    p.add_argument(
        "--logT-path",
        type=str,
        default=None,
        help="Optional full path (or filename) for the logT plot. Overrides --out-dir.",
    )
    p.add_argument(
        "--plot-cumregret",
        action="store_true",
        help="Also plot cumulative regret over time for a single horizon T.",
    )
    p.add_argument(
        "--cumregret-T",
        type=int,
        default=None,
        help="Horizon T for cumulative-regret plot (defaults to first T in --T-list).",
    )
    p.add_argument(
        "--plot-env",
        action="store_true",
        help="Plot a single environment example (observations + true mean) for one T.",
    )
    p.add_argument(
        "--env-T",
        type=int,
        default=None,
        help="Horizon T for the environment example plot (defaults to first T in --T-list).",
    )
    p.add_argument(
        "--env-seed",
        type=int,
        default=7,
        help="Seed for the environment example plot.",
    )
    p.add_argument(
        "--plot-alarms",
        action="store_true",
        help="Plot alarms over a single environment example for one T.",
    )
    p.add_argument(
        "--alarms-T",
        type=int,
        default=None,
        help="Horizon T for the alarms plot (defaults to first T in --T-list).",
    )
    p.add_argument(
        "--alarms-seed",
        type=int,
        default=7,
        help="Seed for the alarms plot.",
    )
    p.add_argument(
        "--alarms-algo",
        type=str,
        default=None,
        help="Algo key to use for the alarms plot (defaults to first in --algos).",
    )
    p.add_argument(
        "--debug-alarms",
        action="store_true",
        help="Print per-run alarm locations for a single algo (for debugging).",
    )
    p.add_argument(
        "--debug-alarms-n",
        type=int,
        default=5,
        help="Number of MC reps to print when --debug-alarms is set.",
    )
    p.add_argument(
        "--debug-alarms-algo",
        type=str,
        default=None,
        help="Algo key to debug (defaults to all algos).",
    )
    p.add_argument(
        "--debug-alarms-T",
        type=int,
        default=None,
        help="Only debug a specific T (defaults to all T in --T-list).",
    )
    p.add_argument(
        "--cumregret-annotate",
        action="store_true",
        help="Annotate the cumulative regret plot with change points and alarms.",
    )
    p.add_argument(
        "--cumregret-annotate-seed",
        type=int,
        default=7,
        help="Seed for the alarms used in cumregret annotations.",
    )
    p.add_argument(
        "--cumregret-alarms-algo",
        type=str,
        default=None,
        help="Algo key to use for cumregret alarm annotations (defaults to first in --algos).",
    )
    p.add_argument(
        "--cumregret-add-var-bias",
        action="store_true",
        help="Overlay variance/bias curves on the cumulative regret plot.",
    )
    p.add_argument(
        "--plot-var-bias",
        action="store_true",
        help="Plot cumulative variance-like and bias-like errors for a single T.",
    )
    p.add_argument(
        "--var-bias-T",
        type=int,
        default=None,
        help="Horizon T for variance/bias plot (defaults to first T in --T-list).",
    )

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Run ONLY the chosen scenario (default md).
    scenarios = (args.scenario,)

    algo_specs = make_algo_specs(
        args.algos,
        max_splits_spec=args.max_splits,
        const_gamma=args.const_gamma,
        geom_base=args.geom_base,
        gcp_gamma_scale=args.gcp_gamma_scale,
        window_len=args.window,
        window_formula=args.window_formula,
        discount=args.discount,
        discount_formula=args.discount_formula,
    )

    results = scaling_experiment(
        T_list=args.T_list,
        scenarios=scenarios,
        algo_specs=algo_specs,
        n_mc=args.n_mc,
        sigma=args.sigma,
        alpha=args.alpha,
        seed=args.seed,
        gcp_gamma_scale=args.gcp_gamma_scale,
        s_frac=args.S_frac,
        delta=args.delta,
        debug_alarms=args.debug_alarms,
        debug_alarms_n=args.debug_alarms_n,
        debug_alarms_algo=args.debug_alarms_algo,
        debug_alarms_T=args.debug_alarms_T,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    if not args.skip_logT:
        if args.logT_path:
            plot_path = args.logT_path
        else:
            plot_path = os.path.join(args.out_dir, f"regret_vs_logT_{args.scenario}.png")
        plot_regret_vs_logT(
            results,
            scenario=args.scenario,
            out_path=plot_path,
            add_fit=(not args.no_fit),
        )
        print(f"Wrote: {plot_path}")

    if args.plot_cumregret:
        if args.cumregret_T is not None:
            T_cum = int(args.cumregret_T)
        elif len(args.T_list) > 0:
            T_cum = int(args.T_list[0])
            if len(args.T_list) > 1:
                print(f"[cumregret] Using T={T_cum} (first in --T-list).")
        else:
            raise ValueError("No T available for cumulative regret plot.")

        cum_results = mc_cumregret_for_T_multi_algos(
            T=T_cum,
            scenario=args.scenario,
            algo_specs=algo_specs,
            n_mc=args.n_mc,
            sigma=args.sigma,
            alpha=args.alpha,
            seed=args.seed + 999,
            gcp_gamma_scale=args.gcp_gamma_scale,
            s_frac=args.S_frac,
            delta=args.delta,
        )
        cum_path = os.path.join(args.out_dir, f"cumregret_T{T_cum}_{args.scenario}.png")

        cp_times = None
        alarm_times = None
        vb_results = None
        if args.cumregret_annotate:
            if args.cumregret_alarms_algo is None:
                alarm_algo = algo_specs[0]
            else:
                match = [a for a in algo_specs if a.key == args.cumregret_alarms_algo]
                if not match:
                    raise ValueError(f"--cumregret-alarms-algo {args.cumregret_alarms_algo!r} not in --algos.")
                alarm_algo = match[0]

            taus, alarms = _alarms_for_single_run(
                T=T_cum,
                scenario=args.scenario,
                sigma=args.sigma,
                alpha=args.alpha,
                seed=args.cumregret_annotate_seed,
                algo_spec=alarm_algo,
                gcp_gamma_scale=args.gcp_gamma_scale,
                s_frac=args.S_frac,
                delta=args.delta,
            )
            cp_times = list(taus[1:-1])
            alarm_times = list(sorted(alarms))

        if args.cumregret_add_var_bias:
            vb_results = mc_var_bias_for_T_multi_algos(
                T=T_cum,
                scenario=args.scenario,
                algo_specs=algo_specs,
                n_mc=args.n_mc,
                sigma=args.sigma,
                alpha=args.alpha,
                seed=args.seed + 333,
                gcp_gamma_scale=args.gcp_gamma_scale,
                s_frac=args.S_frac,
                delta=args.delta,
            )

        plot_cumregret_vs_t(
            cum_results,
            scenario=args.scenario,
            out_path=cum_path,
            change_points=cp_times,
            alarms=alarm_times,
            var_bias_results=vb_results,
        )
        print(f"Wrote: {cum_path}")

    if args.plot_env:
        if args.env_T is not None:
            T_env = int(args.env_T)
        elif len(args.T_list) > 0:
            T_env = int(args.T_list[0])
            if len(args.T_list) > 1:
                print(f"[env] Using T={T_env} (first in --T-list).")
        else:
            raise ValueError("No T available for environment plot.")

        env_path = os.path.join(args.out_dir, f"env_example_T{T_env}_{args.scenario}.png")
        plot_environment_example(
            T=T_env,
            scenario=args.scenario,
            sigma=args.sigma,
            seed=args.env_seed,
            out_path=env_path,
            s_frac=args.S_frac,
            delta=args.delta,
        )
        print(f"Wrote: {env_path}")

    if args.plot_alarms:
        if args.alarms_T is not None:
            T_alarm = int(args.alarms_T)
        elif len(args.T_list) > 0:
            T_alarm = int(args.T_list[0])
            if len(args.T_list) > 1:
                print(f"[alarms] Using T={T_alarm} (first in --T-list).")
        else:
            raise ValueError("No T available for alarms plot.")

        if args.alarms_algo is None:
            alarm_algo = algo_specs[0]
        else:
            match = [a for a in algo_specs if a.key == args.alarms_algo]
            if not match:
                raise ValueError(f"--alarms-algo {args.alarms_algo!r} not in --algos.")
            alarm_algo = match[0]

        alarms_path = os.path.join(
            args.out_dir, f"alarms_T{T_alarm}_{alarm_algo.key}_{args.scenario}.png"
        )
        plot_alarms_on_environment(
            T=T_alarm,
            scenario=args.scenario,
            sigma=args.sigma,
            alpha=args.alpha,
            seed=args.alarms_seed,
            algo_spec=alarm_algo,
            gcp_gamma_scale=args.gcp_gamma_scale,
            out_path=alarms_path,
            s_frac=args.S_frac,
            delta=args.delta,
        )
        print(f"Wrote: {alarms_path}")

    if args.plot_var_bias:
        if args.var_bias_T is not None:
            T_vb = int(args.var_bias_T)
        elif len(args.T_list) > 0:
            T_vb = int(args.T_list[0])
            if len(args.T_list) > 1:
                print(f"[var-bias] Using T={T_vb} (first in --T-list).")
        else:
            raise ValueError("No T available for variance/bias plot.")

        vb_results = mc_var_bias_for_T_multi_algos(
            T=T_vb,
            scenario=args.scenario,
            algo_specs=algo_specs,
            n_mc=args.n_mc,
            sigma=args.sigma,
            alpha=args.alpha,
            seed=args.seed + 333,
            gcp_gamma_scale=args.gcp_gamma_scale,
            s_frac=args.S_frac,
            delta=args.delta,
        )
        vb_path = os.path.join(args.out_dir, f"var_bias_T{T_vb}_{args.scenario}.png")
        plot_var_bias_vs_t(
            vb_results,
            scenario=args.scenario,
            out_path=vb_path,
        )
        print(f"Wrote: {vb_path}")
