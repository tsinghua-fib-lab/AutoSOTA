#!/usr/bin/env python3
"""
Construct a length- and perplexity-matched OpenOrca (or TyDiQA, etc.) subset for
the mixed60 experiments, with optional language-aware sampling.

Pipeline:
  1. Load adversarial token stats, measure the post-prefix length IQR, and use
     it as the permissible length band for the “normal” pool.
  2. Load normal token stats, compute post-prefix lengths and per-prompt
     perplexities (skipping system-prompt tokens).
  3. Filter candidates to the adversarial length band and stratify their
     log10(PP) distribution into N bins (default 140). Optionally keep per-
     language quotas.
  4. Sample `target_size` prompts proportional to the empirical bin mass
     (with deterministic residual allocation) to preserve the first two
     moments of the filtered pool (mean ≈ 60, std ≈ 48). Optional Gaussian
     weighting around a target PP mean/std is supported.
  5. Write the sampled normals, optionally merge them with the adversarial
     rows from the base dataset, and emit a histogram JSON for auditability.

All outputs are deterministic for a fixed `--seed`.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def compute_perplexity_from_token_stream(token_stream: str) -> float:
    """Compute PP from a token_stream JSON string without allocating per-token arrays."""
    payload = json.loads(token_stream)
    total = 0.0
    count = 0
    for tok in payload.get("tokens", []):
        if int(tok.get("is_prefix", 0)) == 1:
            continue
        if tok.get("pos_postprefix") is None:
            continue
        nll = tok.get("nll")
        if nll is None:
            continue
        total += float(nll)
        count += 1
    if count == 0:
        return float("nan")
    mean_nll = total / count
    if not np.isfinite(mean_nll):
        return float("nan")
    return float(np.exp(mean_nll))


def compute_length_band(
    adv_stats_csv: Path,
    fence_mult: float = 1.5,
    adv_algorithms: Optional[Iterable[str]] = None,
) -> Tuple[int, int]:
    cols = ["is_adversarial", "prefix_len_tokens", "text_len_tokens", "algorithm"]
    df = pd.read_csv(adv_stats_csv, usecols=cols)
    adv = df[df["is_adversarial"] == 1].copy()
    if adv_algorithms:
        allowed = {str(a).strip().lower() for a in adv_algorithms if str(a).strip()}
        if allowed:
            adv = adv[adv["algorithm"].fillna("").astype(str).str.lower().isin(allowed)]
    if adv.empty:
        raise ValueError(f"No adversarial rows found in {adv_stats_csv}")
    post_lengths = adv["text_len_tokens"] - adv["prefix_len_tokens"]
    q1 = float(post_lengths.quantile(0.25))
    q3 = float(post_lengths.quantile(0.75))
    iqr = q3 - q1
    if fence_mult <= 0:
        low = float(post_lengths.min())
        high = float(post_lengths.max())
    else:
        fence_low = q1 - fence_mult * iqr
        fence_high = q3 + fence_mult * iqr
        low = max(fence_low, 0.0)
        high = min(fence_high, float(post_lengths.max()))
    if low >= high:
        low = float(post_lengths.min())
        high = float(post_lengths.max())
    return int(math.floor(low)), int(math.ceil(high))


def load_normal_candidates(normal_stats_csv: Path, language_col: Optional[str] = None) -> pd.DataFrame:
    usecols = [
        "full_prompt",
        "suffix",
        "token_stream",
        "prefix_len_tokens",
        "text_len_tokens",
    ]
    if language_col:
        usecols.append(language_col)
    try:
        reader = pd.read_csv(normal_stats_csv, usecols=usecols, chunksize=256)
    except ValueError as exc:
        if language_col and "do not match" in str(exc):
            raise ValueError(
                f"Language column '{language_col}' not found in {normal_stats_csv}"
            ) from exc
        raise

    rows: List[Dict[str, object]] = []
    for chunk in reader:
        for entry in chunk.itertuples(index=False):
            try:
                pp = compute_perplexity_from_token_stream(entry.token_stream)
            except Exception:
                continue
            if not np.isfinite(pp) or pp <= 0:
                continue
            post_len = int(entry.text_len_tokens) - int(entry.prefix_len_tokens)
            suffix = entry.suffix if isinstance(entry.suffix, str) else ""
            record = {
                "full_prompt": entry.full_prompt,
                "suffix": suffix,
                "post_len_tokens": post_len,
                "pp": pp,
                "log10_pp": float(np.log10(pp)),
            }
            if language_col:
                record["language"] = getattr(entry, language_col)
            rows.append(record)
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No valid prompts parsed from {normal_stats_csv}")
    return df


def load_adv_log10_pp(
    adv_stats_csv: Path,
    *,
    pp_multiplier: float = 1.0,
    adv_algorithms: Optional[Iterable[str]] = None,
) -> np.ndarray:
    """Return adversarial log10(PP), optionally shifted by a PP multiplier."""
    if pp_multiplier <= 0:
        raise ValueError("pp_multiplier must be > 0")
    usecols = [
        "token_stream",
        "is_adversarial",
        "algorithm",
    ]
    df = pd.read_csv(adv_stats_csv, usecols=usecols)
    adv = df[df["is_adversarial"].fillna(0).astype(int) == 1]
    if adv_algorithms:
        allowed = {str(a).strip().lower() for a in adv_algorithms if str(a).strip()}
        if allowed:
            adv = adv[adv["algorithm"].fillna("").astype(str).str.lower().isin(allowed)]
    if adv.empty:
        raise ValueError(f"No adversarial rows found in {adv_stats_csv}")

    out: List[float] = []
    shift = float(np.log10(pp_multiplier))
    for token_stream in adv["token_stream"].astype(str).tolist():
        try:
            pp = compute_perplexity_from_token_stream(token_stream)
        except Exception:
            continue
        if not np.isfinite(pp) or pp <= 0:
            continue
        out.append(float(np.log10(pp)) + shift)

    if not out:
        raise ValueError(f"Failed to compute adversarial PP values from {adv_stats_csv}")
    return np.asarray(out, dtype=float)


def _allocate_counts(
    proportions: np.ndarray,
    target_size: int,
    *,
    capacity: Optional[np.ndarray] = None,
) -> np.ndarray:
    proportions = np.asarray(proportions, dtype=float)
    if proportions.ndim != 1:
        raise ValueError("proportions must be a 1D array")
    if target_size <= 0:
        raise ValueError("target_size must be positive")
    if not np.isfinite(proportions).all() or proportions.sum() <= 0:
        raise ValueError("Invalid proportions array")

    proportions = proportions / proportions.sum()
    desired = proportions * target_size
    picks = np.floor(desired).astype(int)
    if capacity is not None:
        capacity = np.asarray(capacity, dtype=int)
        if capacity.shape != picks.shape:
            raise ValueError("capacity must match proportions shape")
        picks = np.minimum(picks, capacity)

    residual = target_size - picks.sum()
    if residual <= 0:
        if residual < 0:
            # Trim back deterministically.
            order = np.argsort(desired - np.floor(desired))
            for idx in order:
                if residual == 0:
                    break
                if picks[idx] > 0:
                    picks[idx] -= 1
                    residual += 1
        return picks

    frac = desired - np.floor(desired)
    order = np.argsort(frac)[::-1]
    for idx in order:
        if residual == 0:
            break
        if capacity is not None and picks[idx] >= capacity[idx]:
            continue
        picks[idx] += 1
        residual -= 1

    if residual > 0:
        if capacity is None:
            raise RuntimeError("Residual allocation failed unexpectedly.")
        spare = capacity - picks
        order = np.argsort(spare)[::-1]
        for idx in order:
            if residual == 0:
                break
            if spare[idx] <= 0:
                continue
            take = min(int(spare[idx]), residual)
            picks[idx] += take
            residual -= take
    if residual != 0:
        raise RuntimeError("Unable to allocate target_size within capacity constraints.")
    return picks


def _reservoir_update(
    reservoir: List[Dict[str, object]],
    record: Dict[str, object],
    seen: int,
    capacity: int,
    rng: np.random.Generator,
) -> None:
    """In-place reservoir sampling update."""
    if capacity <= 0:
        return
    if len(reservoir) < capacity:
        reservoir.append(record)
        return
    j = int(rng.integers(0, seen))
    if j < capacity:
        reservoir[j] = record


def sample_normals_match_adv_pp(
    normal_stats_csv: Path,
    *,
    edges_log10: np.ndarray,
    picks: np.ndarray,
    seed: int,
    length_band: Tuple[int, int],
    apply_length_band: bool,
    min_post_len: Optional[int],
    min_pp: Optional[float],
    min_log10_pp: Optional[float],
    language_col: Optional[str],
    default_language: Optional[str],
    language_allowlist: Optional[Iterable[str]],
    language_blocklist: Optional[Iterable[str]],
    source_dataset: Optional[str],
    fallback_size: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Stream-sample normals to match a target PP bin distribution."""
    bins = int(len(edges_log10) - 1)
    if bins <= 0:
        raise ValueError("edges_log10 must have at least 2 entries")
    if len(picks) != bins:
        raise ValueError("picks length must equal number of bins")

    usecols = [
        "full_prompt",
        "suffix",
        "token_stream",
        "prefix_len_tokens",
        "text_len_tokens",
    ]
    if language_col:
        usecols.append(language_col)

    reservoirs: List[List[Dict[str, object]]] = [[] for _ in range(bins)]
    seen_counts = np.zeros(bins, dtype=int)
    eligible_counts = np.zeros(bins, dtype=int)

    fallback: List[Dict[str, object]] = []
    fallback_seen = 0

    rng = np.random.default_rng(seed)

    allow_langs = None
    if language_allowlist:
        allow_langs = {str(x).strip().lower() for x in language_allowlist if str(x).strip()}
        if not allow_langs:
            allow_langs = None
    block_langs = None
    if language_blocklist:
        block_langs = {str(x).strip().lower() for x in language_blocklist if str(x).strip()}
        if not block_langs:
            block_langs = None

    reader = pd.read_csv(normal_stats_csv, usecols=usecols, chunksize=2048)
    for chunk in reader:
        for entry in chunk.itertuples(index=False):
            try:
                pp = compute_perplexity_from_token_stream(entry.token_stream)
            except Exception:
                continue
            if not np.isfinite(pp) or pp <= 0:
                continue

            log10_pp = float(np.log10(pp))
            if min_pp is not None and pp < float(min_pp):
                continue
            if min_log10_pp is not None and log10_pp < float(min_log10_pp):
                continue

            post_len = int(entry.text_len_tokens) - int(entry.prefix_len_tokens)
            if apply_length_band and not (length_band[0] <= post_len <= length_band[1]):
                continue
            if min_post_len is not None and post_len < int(min_post_len):
                continue

            if log10_pp < float(edges_log10[0]) or log10_pp > float(edges_log10[-1]):
                continue
            bin_idx = int(np.digitize(log10_pp, edges_log10[1:-1], right=False))
            if bin_idx < 0 or bin_idx >= bins:
                continue

            lang_val: Optional[str] = None
            if language_col:
                try:
                    raw = getattr(entry, language_col)
                except Exception:
                    raw = None
                if raw is not None and str(raw).strip():
                    lang_val = str(raw).strip()
            if lang_val is None and default_language and str(default_language).strip():
                lang_val = str(default_language).strip()

            if allow_langs is not None:
                if lang_val is None or lang_val.strip().lower() not in allow_langs:
                    continue
            if block_langs is not None and lang_val is not None:
                if lang_val.strip().lower() in block_langs:
                    continue

            record: Dict[str, object] = {
                "full_prompt": entry.full_prompt,
                "suffix": entry.suffix if isinstance(entry.suffix, str) else "",
                "post_len_tokens": post_len,
                "pp": pp,
                "log10_pp": log10_pp,
            }
            if lang_val is not None:
                record["language"] = lang_val
            if source_dataset is not None and str(source_dataset).strip():
                record["source_dataset"] = str(source_dataset).strip()

            eligible_counts[bin_idx] += 1
            seen_counts[bin_idx] += 1
            k = int(picks[bin_idx])
            if k > 0:
                _reservoir_update(reservoirs[bin_idx], record, int(seen_counts[bin_idx]), k, rng)

            if fallback_size > 0:
                fallback_seen += 1
                _reservoir_update(fallback, record, fallback_seen, fallback_size, rng)

    sampled_records: List[Dict[str, object]] = []
    underfilled: Dict[int, int] = {}
    for idx, k in enumerate(picks.tolist()):
        k = int(k)
        if k <= 0:
            continue
        got = len(reservoirs[idx])
        sampled_records.extend(reservoirs[idx])
        if got < k:
            underfilled[idx] = k - got

    # Fill deficits with fallback samples (best-effort) while avoiding duplicates.
    target_size = int(picks.sum())
    if len(sampled_records) < target_size:
        deficit = target_size - len(sampled_records)
        seen_prompts = {str(r.get("full_prompt", "")) for r in sampled_records}
        for rec in fallback:
            if deficit == 0:
                break
            key = str(rec.get("full_prompt", ""))
            if key in seen_prompts:
                continue
            sampled_records.append(rec)
            seen_prompts.add(key)
            deficit -= 1
        if deficit > 0:
            raise RuntimeError(
                f"Underfilled sample (missing {deficit} rows). "
                "Try lowering --bins or widening the PP range (reduce --adv-pp-multiplier)."
            )

    sampled = pd.DataFrame(sampled_records)
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    meta: Dict[str, object] = {
        "edges_log10": edges_log10,
        "eligible_counts": eligible_counts,
        "desired_counts": picks,
        "underfilled_bins": {int(k): int(v) for k, v in underfilled.items()},
    }
    return sampled, meta


def _build_bin_summary(values: np.ndarray, bins: int) -> Dict[str, np.ndarray]:
    lo, hi = float(values.min()), float(values.max())
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("Invalid log10(PP) range detected.")
    edges = np.linspace(lo, hi, bins + 1, dtype=float)
    bin_ids = np.digitize(values, edges[1:-1], right=False)
    counts = np.bincount(bin_ids, minlength=bins)
    proportions = counts / counts.sum()
    return {
        "edges_log10": edges,
        "edges_pp": np.power(10.0, edges),
        "counts": counts,
        "proportions": proportions,
        "bin_ids": bin_ids,
    }


def stratified_sample(
    df: pd.DataFrame,
    bins: int,
    target_size: int,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    if target_size > len(df):
        raise ValueError(f"target_size ({target_size}) exceeds pool ({len(df)})")

    summary = _build_bin_summary(df["log10_pp"].to_numpy(), bins)
    bin_ids = summary["bin_ids"]
    counts = summary["counts"]
    proportions = summary["proportions"]

    picks = _allocate_counts(proportions, target_size, capacity=counts)

    rng = np.random.default_rng(seed)
    indices = np.arange(len(df))
    selected: List[int] = []
    for bin_idx in range(bins):
        need = picks[bin_idx]
        if need <= 0:
            continue
        locs = indices[bin_ids == bin_idx]
        if locs.size == 0:
            continue
        if need > locs.size:
            need = locs.size
        chosen = rng.choice(locs, size=need, replace=False)
        selected.extend(chosen.tolist())

    selected = np.array(selected, dtype=int)
    if selected.size < target_size:
        deficit = target_size - selected.size
        remaining = np.setdiff1d(indices, selected, assume_unique=False)
        if remaining.size < deficit:
            raise RuntimeError("Not enough remaining candidates to reach target_size.")
        extra = rng.choice(remaining, size=deficit, replace=False)
        selected = np.concatenate([selected, extra])
    elif selected.size > target_size:
        selected = rng.choice(selected, size=target_size, replace=False)

    sampled = df.iloc[selected].copy().reset_index(drop=True)
    summary = {k: v for k, v in summary.items() if k != "bin_ids"}
    return sampled, summary


def stratified_sample_by_language(
    df: pd.DataFrame,
    bins: int,
    target_size: int,
    seed: int,
    lang_col: str,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    if lang_col not in df.columns:
        raise ValueError(f"Language column '{lang_col}' not found.")
    if target_size > len(df):
        raise ValueError(f"target_size ({target_size}) exceeds pool ({len(df)})")

    rng = np.random.default_rng(seed)
    lang_counts = df[lang_col].value_counts()
    proportions = lang_counts / lang_counts.sum()
    desired = proportions * target_size
    picks = np.floor(desired).astype(int)
    residual = target_size - picks.sum()

    if residual > 0:
        frac = desired - picks
        order = frac.sort_values(ascending=False).index
        for lang in order:
            if residual == 0:
                break
            if picks[lang] < lang_counts[lang]:
                picks[lang] += 1
                residual -= 1

    sampled_frames: List[pd.DataFrame] = []
    for lang, need in picks.items():
        if need <= 0:
            continue
        subset = df[df[lang_col] == lang]
        if need > len(subset):
            need = len(subset)
        lang_seed = int(rng.integers(0, 2**32 - 1))
        sub_sample, _ = stratified_sample(subset, bins, need, lang_seed)
        sampled_frames.append(sub_sample)

    sampled = pd.concat(sampled_frames, ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    bin_summary = _build_bin_summary(df["log10_pp"].to_numpy(), bins)
    bin_summary = {k: v for k, v in bin_summary.items() if k != "bin_ids"}
    return sampled, bin_summary


def gaussian_weighted_sample(
    df: pd.DataFrame,
    target_mean: float,
    target_std: float,
    target_size: int,
    bins: int,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    if target_size > len(df):
        raise ValueError(f"target_size ({target_size}) exceeds pool ({len(df)})")
    if target_std <= 0:
        raise ValueError("target_std must be positive.")

    values = df["pp"].to_numpy(dtype=float)
    weights = np.exp(-0.5 * ((values - target_mean) / target_std) ** 2)
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.any(weights > 0):
        raise ValueError("All sampling weights are zero; adjust target mean/std.")
    weights /= weights.sum()

    rng = np.random.default_rng(seed)
    try:
        indices = rng.choice(len(df), size=target_size, replace=False, p=weights)
    except ValueError as exc:
        raise ValueError(
            "Unable to sample without replacement given the current weights; "
            "try lowering target_size or adjusting mean/std."
        ) from exc

    sampled = df.iloc[indices].copy().reset_index(drop=True)
    summary = _build_bin_summary(df["log10_pp"].to_numpy(), bins)
    summary = {k: v for k, v in summary.items() if k != "bin_ids"}
    return sampled, summary


def gaussian_weighted_sample_by_language(
    df: pd.DataFrame,
    target_mean: float,
    target_std: float,
    target_size: int,
    bins: int,
    seed: int,
    lang_col: str,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    if lang_col not in df.columns:
        raise ValueError(f"Language column '{lang_col}' not found.")
    if target_size > len(df):
        raise ValueError(f"target_size ({target_size}) exceeds pool ({len(df)})")
    if target_std <= 0:
        raise ValueError("target_std must be positive.")

    rng = np.random.default_rng(seed)
    lang_counts = df[lang_col].value_counts()
    proportions = lang_counts / lang_counts.sum()
    desired = proportions * target_size
    picks = np.floor(desired).astype(int)
    residual = target_size - picks.sum()
    if residual > 0:
        frac = desired - picks
        order = frac.sort_values(ascending=False).index
        for lang in order:
            if residual == 0:
                break
            if picks[lang] < lang_counts[lang]:
                picks[lang] += 1
                residual -= 1

    sampled_frames: List[pd.DataFrame] = []
    for lang, need in picks.items():
        if need <= 0:
            continue
        subset = df[df[lang_col] == lang].copy()
        values = subset["pp"].to_numpy(dtype=float)
        weights = np.exp(-0.5 * ((values - target_mean) / target_std) ** 2)
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.any(weights > 0):
            continue
        weights /= weights.sum()
        if need > len(subset):
            need = len(subset)
        indices = rng.choice(len(subset), size=need, replace=False, p=weights)
        sub_sample = subset.iloc[indices].copy().reset_index(drop=True)
        sampled_frames.append(sub_sample)

    sampled = pd.concat(sampled_frames, ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    bin_summary = _build_bin_summary(df["log10_pp"].to_numpy(), bins)
    bin_summary = {k: v for k, v in bin_summary.items() if k != "bin_ids"}
    return sampled, bin_summary


def topk_sample(
    df: pd.DataFrame,
    target_size: int,
    *,
    lang_col: Optional[str] = None,
    seed: int = 7,
) -> pd.DataFrame:
    """Select the highest-PP rows, optionally respecting language proportions."""
    if target_size > len(df):
        raise ValueError(f"target_size ({target_size}) exceeds pool ({len(df)})")
    if lang_col is None:
        return df.sort_values("pp", ascending=False).head(target_size).copy().reset_index(drop=True)

    if lang_col not in df.columns:
        raise ValueError(f"Language column '{lang_col}' not found.")

    rng = np.random.default_rng(seed)
    lang_counts = df[lang_col].value_counts()
    proportions = lang_counts / lang_counts.sum()
    desired = proportions * target_size
    picks = np.floor(desired).astype(int)
    residual = target_size - picks.sum()

    if residual > 0:
        frac = desired - picks
        order = frac.sort_values(ascending=False).index
        for lang in order:
            if residual == 0:
                break
            if picks[lang] < lang_counts[lang]:
                picks[lang] += 1
                residual -= 1

    sampled_frames: List[pd.DataFrame] = []
    used_idx: List[int] = []
    for lang, need in picks.items():
        if need <= 0:
            continue
        subset = df[df[lang_col] == lang].sort_values("pp", ascending=False).head(int(need)).copy()
        used_idx.extend(subset.index.tolist())
        sampled_frames.append(subset)

    sampled = pd.concat(sampled_frames, ignore_index=True)
    if len(sampled) < target_size:
        deficit = target_size - len(sampled)
        remaining = df.drop(index=used_idx, errors="ignore").sort_values("pp", ascending=False).head(deficit)
        sampled = pd.concat([sampled, remaining], ignore_index=True)

    sampled = sampled.sample(frac=1.0, random_state=int(rng.integers(0, 2**32 - 1))).reset_index(drop=True)
    return sampled


def write_normals_csv(
    sampled: pd.DataFrame,
    out_path: Path,
    algorithm_label: str,
    *,
    include_pp_cols: bool = False,
) -> None:
    cols = ["full_prompt", "suffix", "is_adversarial", "algorithm"]
    if "source_dataset" in sampled.columns:
        cols.append("source_dataset")
    if "language" in sampled.columns:
        cols.append("language")
    if include_pp_cols:
        cols.extend(["post_len_tokens", "pp", "log10_pp"])
    payload = sampled.assign(
        is_adversarial=0,
        algorithm=algorithm_label,
    )[cols]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload.to_csv(out_path, index=False)


def write_combined_dataset(
    base_dataset: Path,
    normals: pd.DataFrame,
    out_path: Path,
    algorithm_label: str,
) -> None:
    base = pd.read_csv(base_dataset)
    adv = base[base["is_adversarial"] == 1].copy()
    if "source_dataset" not in adv.columns:
        adv["source_dataset"] = adv.get("algorithm", "adversarial")
    normals_out = normals.assign(is_adversarial=0, algorithm=algorithm_label)
    combined = pd.concat([adv, normals_out], ignore_index=True)
    combined = combined.drop_duplicates(subset=["full_prompt", "suffix", "algorithm"], keep="first")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)


def dump_histogram(
    filtered: pd.DataFrame,
    sampled: pd.DataFrame,
    bin_summary: Dict[str, np.ndarray],
    hist_path: Path,
    length_band: Tuple[int, int],
    bins: int,
) -> None:
    edges_log = bin_summary["edges_log10"].astype(float).tolist()
    edges_pp = bin_summary["edges_pp"].astype(float).tolist()
    proportions = bin_summary["proportions"].astype(float)
    interval_labels = []
    for a, b in zip(edges_pp[:-1], edges_pp[1:]):
        interval_labels.append(f"({a:.6f}, {b:.6f}]")

    lang_counts_filtered = (
        filtered["language"].value_counts().to_dict() if "language" in filtered.columns else {}
    )
    lang_counts_sampled = (
        sampled["language"].value_counts().to_dict() if "language" in sampled.columns else {}
    )

    payload = {
        "total_rows": int(len(filtered)),
        "sample_size": int(len(sampled)),
        "length_filter_tokens": {
            "lower": int(length_band[0]),
            "upper": int(length_band[1]),
        },
        "population_stats": {
            "mean_pp": float(filtered["pp"].mean()),
            "std_pp": float(filtered["pp"].std(ddof=0)),
            "min_pp": float(filtered["pp"].min()),
            "max_pp": float(filtered["pp"].max()),
        },
        "sample_stats": {
            "mean_pp": float(sampled["pp"].mean()),
            "std_pp": float(sampled["pp"].std(ddof=0)),
            "min_pp": float(sampled["pp"].min()),
            "max_pp": float(sampled["pp"].max()),
        },
        "binning": {
            "metric": "log10_pp",
            "num_bins": int(bins),
            "edges_log10": edges_log,
            "edges_pp": edges_pp,
            "proportions": proportions.tolist(),
        },
        "bin_proportions": {
            label: float(prop)
            for label, prop in zip(interval_labels, proportions)
        },
        "language_counts_filtered": lang_counts_filtered,
        "language_counts_sampled": lang_counts_sampled,
    }
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    hist_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def summarize(df: pd.DataFrame, label: str) -> None:
    stats = df["pp"].describe(percentiles=[0.25, 0.5, 0.75])
    print(
        f"[{label}] n={len(df):4d}  "
        f"mean={stats['mean']:.2f}  std={df['pp'].std(ddof=0):.2f}  "
        f"min={stats['min']:.2f}  median={stats['50%']:.2f}  max={stats['max']:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample normal prompts with PP/length matching.")
    parser.add_argument(
        "--normal-stats",
        type=Path,
        action="append",
        required=True,
        help="Token stats CSV for the normal pool. Repeat to sample from multiple pools (match-adv-pp mode only).",
    )
    parser.add_argument(
        "--normal-source-weights",
        nargs="+",
        type=float,
        default=None,
        help="Optional weights for each --normal-stats source (must match count). Defaults to uniform.",
    )
    parser.add_argument(
        "--normal-source-names",
        nargs="+",
        default=None,
        help="Optional human-readable names for each --normal-stats source (must match count). Stored in 'source_dataset'.",
    )
    parser.add_argument("--adv-stats", type=Path, required=True, help="Token stats containing adversarial rows.")
    parser.add_argument("--base-dataset", type=Path, required=True, help="Original mixed60 dataset (for adversarials).")
    parser.add_argument("--out-normals", type=Path, required=True, help="CSV to write the sampled normals.")
    parser.add_argument("--out-combined", type=Path, required=True, help="CSV for the combined dataset.")
    parser.add_argument("--hist-json", type=Path, required=True, help="JSON histogram describing the sampling.")
    parser.add_argument("--bins", type=int, default=140, help="Number of log10(PP) bins.")
    parser.add_argument("--target-size", type=int, default=700, help="Number of normals to sample.")
    parser.add_argument("--seed", type=int, default=7, help="RNG seed for reproducibility.")
    parser.add_argument(
        "--normal-algorithm",
        type=str,
        default="tydiqa",
        help="Algorithm/source label to stamp onto sampled normals.",
    )
    parser.add_argument(
        "--target-pp-mean",
        type=float,
        default=None,
        help="Optional target PP mean for weighted sampling.",
    )
    parser.add_argument(
        "--target-pp-std",
        type=float,
        default=None,
        help="Optional target PP std dev for weighted sampling.",
    )
    parser.add_argument(
        "--language-col",
        type=str,
        default=None,
        help="Optional language column name to enforce per-language sampling quotas.",
    )
    parser.add_argument(
        "--default-language",
        type=str,
        default=None,
        help="If a normal-stats source lacks --language-col, stamp this value into the output 'language' column (e.g., 'english' for OpenOrca).",
    )
    parser.add_argument(
        "--language-allowlist",
        nargs="+",
        default=None,
        help="Optional allowlist of languages (applied only to sources that provide language).",
    )
    parser.add_argument(
        "--language-blocklist",
        nargs="+",
        default=None,
        help="Optional blocklist of languages (applied only to sources that provide language).",
    )
    parser.add_argument(
        "--length-min",
        type=int,
        default=None,
        help="Optional override for the minimum post-prefix length used to filter normal candidates.",
    )
    parser.add_argument(
        "--length-max",
        type=int,
        default=None,
        help="Optional override for the maximum post-prefix length used to filter normal candidates.",
    )
    parser.add_argument(
        "--no-length-filter",
        action="store_true",
        help="Disable post-prefix length filtering entirely (use the full normal pool).",
    )
    parser.add_argument(
        "--min-pp",
        type=float,
        default=None,
        help="Optional minimum perplexity filter applied to normal candidates before sampling.",
    )
    parser.add_argument(
        "--min-log10-pp",
        type=float,
        default=None,
        help="Optional minimum log10(perplexity) filter applied to normal candidates before sampling.",
    )
    parser.add_argument(
        "--min-post-len",
        type=int,
        default=None,
        help="Optional minimum post-prefix token length filter for normal candidates.",
    )
    parser.add_argument(
        "--topk",
        action="store_true",
        help="Select the highest-PP normals (optionally per-language) instead of stratified/gaussian sampling.",
    )
    parser.add_argument(
        "--match-adv-pp",
        action="store_true",
        help="Sample normals to match the adversarial log10(PP) bin distribution (streaming; memory-safe for 1M stats).",
    )
    parser.add_argument(
        "--adv-pp-multiplier",
        type=float,
        default=1.0,
        help="Optional multiplier applied to adversarial PP before matching (e.g., 1.1 shifts the target PP up by 10%%).",
    )
    parser.add_argument(
        "--adv-algorithms",
        nargs="+",
        default=None,
        help="Optional subset of adversarial algorithm labels used to define the PP target distribution (e.g., autodan advprompter).",
    )
    parser.add_argument(
        "--fallback-size",
        type=int,
        default=0,
        help="Optional extra reservoir size used to backfill underfilled PP bins (0 disables).",
    )
    parser.add_argument(
        "--include-pp-cols",
        action="store_true",
        help="Include post_len_tokens/pp/log10_pp columns in --out-normals.",
    )
    args = parser.parse_args()

    adv_algorithms = [str(a).strip() for a in (args.adv_algorithms or []) if str(a).strip()]

    length_band = compute_length_band(args.adv_stats, adv_algorithms=adv_algorithms or None)
    if args.length_min is not None or args.length_max is not None:
        length_band = (
            int(args.length_min) if args.length_min is not None else length_band[0],
            int(args.length_max) if args.length_max is not None else length_band[1],
        )
    apply_length_band = not args.no_length_filter
    if apply_length_band:
        print(f"[INFO] Length band from adversarial Tukey fence -> {length_band}")
    else:
        print("[INFO] Length filtering disabled (--no-length-filter).")

    if args.match_adv_pp and args.topk:
        raise ValueError("Choose at most one of --match-adv-pp and --topk.")

    if args.match_adv_pp:
        normal_stats_paths = [Path(p) for p in (args.normal_stats or [])]
        if not normal_stats_paths:
            raise ValueError("At least one --normal-stats path is required.")

        source_names = args.normal_source_names
        if source_names is not None and len(source_names) != len(normal_stats_paths):
            raise ValueError("--normal-source-names must match the number of --normal-stats arguments.")
        if source_names is None:
            source_names = []
            for p in normal_stats_paths:
                low = p.name.lower()
                if "tydiqa" in low:
                    source_names.append("tydiqa")
                elif "orca" in low:
                    source_names.append("openorca")
                else:
                    source_names.append(p.stem)

        weights = args.normal_source_weights
        if weights is None:
            weights = [1.0] * len(normal_stats_paths)
        if len(weights) != len(normal_stats_paths):
            raise ValueError("--normal-source-weights must match the number of --normal-stats arguments.")
        weights_arr = np.asarray(weights, dtype=float)
        if not np.isfinite(weights_arr).all() or weights_arr.sum() <= 0:
            raise ValueError("Invalid --normal-source-weights; must be positive finite numbers.")
        weights_arr = weights_arr / weights_arr.sum()

        adv_log10 = load_adv_log10_pp(
            args.adv_stats,
            pp_multiplier=args.adv_pp_multiplier,
            adv_algorithms=adv_algorithms or None,
        )
        target_summary = _build_bin_summary(adv_log10, args.bins)
        edges_log10 = target_summary["edges_log10"]
        source_sizes = _allocate_counts(weights_arr, args.target_size)

        sampled_sources: List[pd.DataFrame] = []
        sources_payload: List[Dict[str, object]] = []

        for idx, (normal_stats, source_n) in enumerate(zip(normal_stats_paths, source_sizes.tolist())):
            source_n = int(source_n)
            if source_n <= 0:
                continue

            source_picks = _allocate_counts(target_summary["proportions"], source_n)

            fallback_size = int(args.fallback_size)
            if fallback_size <= 0:
                fallback_size = max(2000, int(math.ceil(0.10 * source_n)))

            # Only request language_col if the source actually has it.
            language_col = args.language_col
            if language_col:
                try:
                    cols = set(pd.read_csv(normal_stats, nrows=0).columns.tolist())
                except Exception:
                    cols = set()
                if language_col not in cols:
                    language_col = None

            sampled, meta = sample_normals_match_adv_pp(
                normal_stats,
                edges_log10=edges_log10,
                picks=source_picks,
                seed=args.seed + 10007 * idx,
                length_band=length_band,
                apply_length_band=apply_length_band,
                min_post_len=args.min_post_len,
                min_pp=args.min_pp,
                min_log10_pp=args.min_log10_pp,
                language_col=language_col,
                default_language=args.default_language if language_col is None else None,
                language_allowlist=args.language_allowlist if language_col is not None else None,
                language_blocklist=args.language_blocklist if language_col is not None else None,
                source_dataset=source_names[idx] if idx < len(source_names) else None,
                fallback_size=fallback_size,
            )

            summarize(sampled, f"{args.normal_algorithm}_sampled[{normal_stats.name}]")
            sampled_sources.append(sampled)

            sample_counts, _ = np.histogram(sampled["log10_pp"].to_numpy(dtype=float), bins=edges_log10)
            sources_payload.append(
                {
                    "path": str(normal_stats),
                    "target_size": int(source_n),
                    "weight": float(weights_arr[idx]),
                    "language_col": language_col,
                    "eligible_counts": meta["eligible_counts"].astype(int).tolist(),
                    "desired_counts": meta["desired_counts"].astype(int).tolist(),
                    "underfilled_bins": meta["underfilled_bins"],
                    "sample_counts": sample_counts.astype(int).tolist(),
                    "sample_stats": {
                        "mean_pp": float(sampled["pp"].mean()),
                        "std_pp": float(sampled["pp"].std(ddof=0)),
                        "min_pp": float(sampled["pp"].min()),
                        "max_pp": float(sampled["pp"].max()),
                    },
                    "language_counts_sampled": (
                        sampled["language"].value_counts().to_dict() if "language" in sampled.columns else {}
                    ),
                }
            )

        if not sampled_sources:
            raise ValueError("No normals were sampled. Check --normal-stats and weights.")

        sampled = pd.concat(sampled_sources, ignore_index=True)
        if len(sampled) > args.target_size:
            sampled = sampled.sample(n=int(args.target_size), random_state=args.seed).reset_index(drop=True)
        elif len(sampled) < args.target_size:
            raise RuntimeError(
                f"Sampled only {len(sampled)} normals but target_size={args.target_size}. "
                "Increase pool sizes, relax filters, or adjust --normal-source-weights."
            )
        sampled = sampled.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

        edges_pp = np.power(10.0, edges_log10)
        sample_counts, _ = np.histogram(sampled["log10_pp"].to_numpy(dtype=float), bins=edges_log10)
        payload = {
            "sample_size": int(len(sampled)),
            "target_size": int(args.target_size),
            "mode": "match_adv_pp",
            "adv_pp_multiplier": float(args.adv_pp_multiplier),
            "adv_algorithms": adv_algorithms,
            "length_filter_tokens": {
                "enabled": bool(apply_length_band),
                "lower": int(length_band[0]),
                "upper": int(length_band[1]),
            },
            "filters": {
                "min_post_len": int(args.min_post_len) if args.min_post_len is not None else None,
                "min_pp": float(args.min_pp) if args.min_pp is not None else None,
                "min_log10_pp": float(args.min_log10_pp) if args.min_log10_pp is not None else None,
            },
            "target_distribution": {
                "metric": "log10_pp",
                "num_bins": int(args.bins),
                "edges_log10": edges_log10.astype(float).tolist(),
                "edges_pp": edges_pp.astype(float).tolist(),
                "counts": target_summary["counts"].astype(int).tolist(),
                "proportions": target_summary["proportions"].astype(float).tolist(),
            },
            "sample_counts": sample_counts.astype(int).tolist(),
            "sources": sources_payload,
            "sample_stats": {
                "mean_pp": float(sampled["pp"].mean()),
                "std_pp": float(sampled["pp"].std(ddof=0)),
                "min_pp": float(sampled["pp"].min()),
                "max_pp": float(sampled["pp"].max()),
            },
            "language_counts_sampled": (
                sampled["language"].value_counts().to_dict() if "language" in sampled.columns else {}
            ),
        }
        args.hist_json.parent.mkdir(parents=True, exist_ok=True)
        args.hist_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        write_normals_csv(sampled, args.out_normals, args.normal_algorithm, include_pp_cols=args.include_pp_cols)
        write_combined_dataset(args.base_dataset, sampled, args.out_combined, args.normal_algorithm)
        print(f"[OK] Wrote sampled normals to {args.out_normals}")
        print(f"[OK] Wrote combined dataset to {args.out_combined}")
        print(f"[OK] Histogram metadata saved to {args.hist_json}")
        return

    if len(args.normal_stats) != 1:
        raise ValueError("Multiple --normal-stats sources are only supported with --match-adv-pp.")
    candidates = load_normal_candidates(args.normal_stats[0], args.language_col)
    if apply_length_band:
        mask = (
            (candidates["post_len_tokens"] >= length_band[0])
            & (candidates["post_len_tokens"] <= length_band[1])
        )
        filtered = candidates[mask].reset_index(drop=True)
    else:
        filtered = candidates.copy()

    if args.min_post_len is not None:
        filtered = filtered[filtered["post_len_tokens"] >= int(args.min_post_len)].reset_index(drop=True)
    if args.min_pp is not None:
        filtered = filtered[filtered["pp"] >= float(args.min_pp)].reset_index(drop=True)
    if args.min_log10_pp is not None:
        filtered = filtered[filtered["log10_pp"] >= float(args.min_log10_pp)].reset_index(drop=True)

    if filtered.empty:
        raise ValueError("No normal prompts satisfied the constraints.")

    summarize(candidates, f"{args.normal_algorithm}_full")
    summarize(filtered, f"{args.normal_algorithm}_filtered")

    if (args.target_pp_mean is None) ^ (args.target_pp_std is None):
        raise ValueError("Both --target-pp-mean and --target-pp-std must be provided together.")

    if args.topk:
        sampled = topk_sample(filtered, args.target_size, lang_col=args.language_col, seed=args.seed)
        bin_summary = _build_bin_summary(filtered["log10_pp"].to_numpy(), args.bins)
        bin_summary = {k: v for k, v in bin_summary.items() if k != "bin_ids"}
    elif args.target_pp_mean is not None:
        if args.language_col:
            sampled, bin_summary = gaussian_weighted_sample_by_language(
                filtered,
                args.target_pp_mean,
                args.target_pp_std,
                args.target_size,
                args.bins,
                args.seed,
                args.language_col,
            )
        else:
            sampled, bin_summary = gaussian_weighted_sample(
                filtered,
                args.target_pp_mean,
                args.target_pp_std,
                args.target_size,
                args.bins,
                args.seed,
            )
    else:
        if args.language_col:
            sampled, bin_summary = stratified_sample_by_language(
                filtered, args.bins, args.target_size, args.seed, args.language_col
            )
        else:
            sampled, bin_summary = stratified_sample(filtered, args.bins, args.target_size, args.seed)
    summarize(sampled, f"{args.normal_algorithm}_sampled")

    write_normals_csv(sampled, args.out_normals, args.normal_algorithm, include_pp_cols=args.include_pp_cols)
    write_combined_dataset(args.base_dataset, sampled, args.out_combined, args.normal_algorithm)
    dump_histogram(
        filtered,
        sampled,
        bin_summary,
        args.hist_json,
        length_band,
        args.bins,
    )

    print(f"[OK] Wrote sampled normals to {args.out_normals}")
    print(f"[OK] Wrote combined dataset to {args.out_combined}")
    print(f"[OK] Histogram metadata saved to {args.hist_json}")


if __name__ == "__main__":
    main()
