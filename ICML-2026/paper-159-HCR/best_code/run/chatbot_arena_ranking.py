"""
Reproduce Chatbot Arena ranking recovery experiment (Appendix K, Table 4).

Uses the paper's published ground-truth win rates for the 6 target models
and simulates LLM judge labels with configurable sensitivity/specificity.

Paper ground truth (Appendix K line 2891):
  GPT-4: 0.850, Claude-v1: 0.786, Vicuna-13B: 0.621,
  Alpaca-13B: 0.358, FastChat-T5-3B: 0.300, LLaMA-13B: 0.209

Enhanced with:
  - Heterogeneous q0,q1 per model (--heterogeneous-q)
  - Per-model q0,q1 estimation from calibration (--per-model-q)
  - James-Stein shrinkage for stable per-model estimates (--shrinkage)
  - Stratified calibration sampling (--stratified)
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

from llm_judge_reporting.calibration import point_estimator


# Paper Table 4 setup
TARGET_MODEL_NAMES = [
    "gpt-4-0314",       # GPT-4
    "claude-1",          # Claude-v1
    "vicuna-13b",        # Vicuna-13B
    "alpaca-13b",        # Alpaca-13B
    "fastchat-t5-3b",   # FastChat-T5-3B
    "llama-13b",         # LLaMA-13B
]

# Ground-truth win rates from the paper (Appendix K)
GROUND_TRUTH_WINRATES = {
    "gpt-4-0314": 0.850,
    "claude-1": 0.786,
    "vicuna-13b": 0.621,
    "alpaca-13b": 0.358,
    "fastchat-t5-3b": 0.300,
    "llama-13b": 0.209,
}

# Comparison counts per model (from actual Chatbot Arena data, first 57k samples)
COMPARISON_COUNTS = {
    "gpt-4-0314": 3586,
    "claude-1": 3450,
    "vicuna-13b": 2932,
    "alpaca-13b": 972,
    "fastchat-t5-3b": 720,
    "llama-13b": 331,
}

DISPLAY_NAMES = {
    "gpt-4-0314": "GPT-4",
    "claude-1": "Claude-v1",
    "vicuna-13b": "Vicuna-13B",
    "alpaca-13b": "Alpaca-13B",
    "fastchat-t5-3b": "FastChat-T5-3B",
    "llama-13b": "LLaMA-13B",
}

N_SPLITS = 100
TEST_FRAC = 0.90
CALIB_FRAC = 0.10


def clip(x, low=0.0, high=1.0):
    return max(low, min(high, x))


def james_stein_shrinkage(estimates, global_est):
    """Apply positive-part James-Stein shrinkage toward global estimate.

    Args:
        estimates: dict of {model: estimate}
        global_est: global estimate (shrinkage target)

    Returns:
        dict of {model: shrunken_estimate}
    """
    k = len(estimates)
    if k < 3:
        return {m: global_est for m in estimates}

    vals = np.array(list(estimates.values()))

    # James-Stein shrinkage factor
    sigma_sq = np.var(vals, ddof=1) if np.var(vals, ddof=1) > 0 else 1e-8
    sum_sq_dev = np.sum((vals - global_est) ** 2)

    if sum_sq_dev < 1e-10:
        return {m: global_est for m in estimates}

    # Positive-part James-Stein factor
    shrinkage = max(0.0, 1.0 - (k - 3) * sigma_sq / sum_sq_dev)

    result = {}
    for m, est in estimates.items():
        result[m] = global_est + shrinkage * (est - global_est)

    return result


def generate_comparisons(rng: np.random.Generator) -> list[dict]:
    """Generate synthetic comparison data matching paper's ground truth win rates."""
    comparisons = []
    for model in TARGET_MODEL_NAMES:
        n_comps = COMPARISON_COUNTS[model]
        true_wr = GROUND_TRUTH_WINRATES[model]
        for _ in range(n_comps):
            comparisons.append({
                "target": model,
                "human_label": int(rng.random() < true_wr),
            })
    rng.shuffle(comparisons)
    return comparisons


def simulate_judge_labels(
    comparisons: list[dict],
    q0: float,
    q1: float,
    rng: np.random.Generator,
    q0_by_model: dict | None = None,
    q1_by_model: dict | None = None,
) -> list[dict]:
    """Simulate LLM judge labels with given sensitivity/specificity.

    Args:
        comparisons: list of comparison dicts with "human_label" and "target"
        q0: default specificity (used if q0_by_model is None or model not in dict)
        q1: default sensitivity
        rng: numpy random generator
        q0_by_model: optional per-model specificity dict
        q1_by_model: optional per-model sensitivity dict
    """
    for comp in comparisons:
        m = comp["target"]
        _q0 = q0_by_model.get(m, q0) if q0_by_model else q0
        _q1 = q1_by_model.get(m, q1) if q1_by_model else q1
        if comp["human_label"] == 1:
            comp["judge_label"] = int(rng.random() < _q1)   # P(judge=1 | true=1)
        else:
            comp["judge_label"] = int(rng.random() >= _q0)  # P(judge=0 | true=0)
    return comparisons


def run_split(
    comparisons, test_idx, calib_idx,
    per_model_q: bool = False,
    use_shrinkage: bool = False,
    min_calib_per_model: int = 10,
):
    """Run one split: estimate p, q0, q1, rank models, compute metrics."""
    test = [comparisons[i] for i in test_idx]
    calib = [comparisons[i] for i in calib_idx]

    # Compute per-model p_hat on test set
    p_hats = {}
    test_counts = defaultdict(lambda: {"wins": 0, "total": 0})
    for c in test:
        m = c["target"]
        test_counts[m]["wins"] += c["judge_label"]
        test_counts[m]["total"] += 1

    for m in TARGET_MODEL_NAMES:
        tc = test_counts[m]
        p_hats[m] = tc["wins"] / max(1, tc["total"])

    # Compute GLOBAL q0_hat, q1_hat on calibration set
    tp = tn = pos = neg = 0
    for c in calib:
        if c["human_label"] == 1:
            pos += 1
            if c["judge_label"] == 1:
                tp += 1
        else:
            neg += 1
            if c["judge_label"] == 0:
                tn += 1

    q0_hat_global = (tn + 1) / max(1, neg + 2)
    q1_hat_global = (tp + 1) / max(1, pos + 2)

    # Compute per-model q0_hat, q1_hat from calibration
    per_model_q0 = {}
    per_model_q1 = {}
    per_model_fallback = set()
    per_model_q0_raw = {}
    per_model_q1_raw = {}

    if per_model_q:
        per_model_counts = {m: {"tp": 0, "tn": 0, "pos": 0, "neg": 0} for m in TARGET_MODEL_NAMES}
        for c in calib:
            m = c["target"]
            if c["human_label"] == 1:
                per_model_counts[m]["pos"] += 1
                if c["judge_label"] == 1:
                    per_model_counts[m]["tp"] += 1
            else:
                per_model_counts[m]["neg"] += 1
                if c["judge_label"] == 0:
                    per_model_counts[m]["tn"] += 1

        for m in TARGET_MODEL_NAMES:
            mc = per_model_counts[m]
            # Always compute raw per-model estimate (even if sparse)
            per_model_q0_raw[m] = (mc["tn"] + 1) / max(1, mc["neg"] + 2)
            per_model_q1_raw[m] = (mc["tp"] + 1) / max(1, mc["pos"] + 2)

            if mc["pos"] >= min_calib_per_model and mc["neg"] >= min_calib_per_model:
                per_model_q0[m] = per_model_q0_raw[m]
                per_model_q1[m] = per_model_q1_raw[m]
            else:
                per_model_fallback.add(m)
                per_model_q0[m] = q0_hat_global
                per_model_q1[m] = q1_hat_global

        # Apply James-Stein shrinkage if enabled
        if use_shrinkage:
            # Only shrink models that have valid per-model estimates
            shrinkable = {m for m in TARGET_MODEL_NAMES if m not in per_model_fallback}
            if len(shrinkable) >= 3:
                # Shrink toward per-model mean, not global
                q0_estimates = {m: per_model_q0[m] for m in shrinkable}
                q1_estimates = {m: per_model_q1[m] for m in shrinkable}
                q0_shrunk = james_stein_shrinkage(q0_estimates, np.mean(list(q0_estimates.values())))
                q1_shrunk = james_stein_shrinkage(q1_estimates, np.mean(list(q1_estimates.values())))
                for m in shrinkable:
                    per_model_q0[m] = q0_shrunk[m]
                    per_model_q1[m] = q1_shrunk[m]

    # Compute theta_hat (bias-corrected win rate) per model
    theta_hats = {}
    for m in TARGET_MODEL_NAMES:
        if per_model_q:
            _q0 = per_model_q0[m]
            _q1 = per_model_q1[m]
        else:
            _q0 = q0_hat_global
            _q1 = q1_hat_global

        denom = _q0 + _q1 - 1
        if denom > 0.01:
            theta_hats[m] = point_estimator(p_hats[m], _q0, _q1)
        else:
            theta_hats[m] = p_hats[m]

    # Rankings (descending win rate)
    gt_ranking = sorted(TARGET_MODEL_NAMES, key=lambda m: GROUND_TRUTH_WINRATES[m], reverse=True)
    naive_ranking = sorted(TARGET_MODEL_NAMES, key=lambda m: p_hats[m], reverse=True)
    corrected_ranking = sorted(TARGET_MODEL_NAMES, key=lambda m: theta_hats[m], reverse=True)

    # Kendall tau
    gt_order = [gt_ranking.index(m) for m in TARGET_MODEL_NAMES]
    naive_tau, _ = kendalltau(gt_order, [naive_ranking.index(m) for m in TARGET_MODEL_NAMES])
    corrected_tau, _ = kendalltau(gt_order, [corrected_ranking.index(m) for m in TARGET_MODEL_NAMES])

    return {
        "naive_tau": float(naive_tau) if naive_tau is not None else 0.0,
        "corrected_tau": float(corrected_tau) if corrected_tau is not None else 0.0,
        "naive_exact": int(naive_ranking == gt_ranking),
        "corrected_exact": int(corrected_ranking == gt_ranking),
        "p_hats": p_hats,
        "theta_hats": theta_hats,
        "q0_hat": q0_hat_global,
        "q1_hat": q1_hat_global,
        "per_model_q0": per_model_q0 if per_model_q else None,
        "per_model_q1": per_model_q1 if per_model_q else None,
        "per_model_fallback": list(per_model_fallback) if per_model_q else None,
        "gt_ranking": gt_ranking,
        "naive_ranking": naive_ranking,
        "corrected_ranking": corrected_ranking,
    }


def main():
    parser = argparse.ArgumentParser(description="Chatbot Arena ranking recovery reproduction")
    parser.add_argument("--q0", type=float, default=0.80, help="Simulated specificity")
    parser.add_argument("--q1", type=float, default=0.77, help="Simulated sensitivity")
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument("--test-frac", type=float, default=TEST_FRAC)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="outputs/chatbot_arena_ranking.json")
    parser.add_argument("--heterogeneous-q", type=float, default=0.0,
                        help="Spread for per-model q0,q1 perturbation (0=off)")
    parser.add_argument("--per-model-q", action="store_true",
                        help="Enable per-model q0,q1 estimation from calibration")
    parser.add_argument("--shrinkage", action="store_true",
                        help="Apply James-Stein shrinkage to per-model estimates")
    parser.add_argument("--stratified", action="store_true",
                        help="Use stratified split by model for calibration")
    parser.add_argument("--min-calib-per-model", type=int, default=10,
                        help="Min calibration samples per model for per-model estimation")
    args = parser.parse_args()

    started = time.time()
    rng = np.random.default_rng(args.seed)

    # Generate comparisons ONCE
    comparisons = generate_comparisons(rng)
    n_total = len(comparisons)
    n_test = int(n_total * args.test_frac)
    print(f"Generated {n_total} comparisons ({n_test} test, {n_total - n_test} calibration)")

    # Generate per-model q0,q1 if heterogeneous mode is enabled
    if args.heterogeneous_q > 0:
        het_rng = np.random.default_rng(args.seed + 9999)
        q0_by_model = {}
        q1_by_model = {}
        print(f"Simulating judge with heterogeneous q0,q1 (spread={args.heterogeneous_q}):")
        for m in TARGET_MODEL_NAMES:
            q0_m = clip(args.q0 + het_rng.normal(0, args.heterogeneous_q), 0.50, 0.95)
            q1_m = clip(args.q1 + het_rng.normal(0, args.heterogeneous_q), 0.50, 0.95)
            q0_by_model[m] = float(q0_m)
            q1_by_model[m] = float(q1_m)
            print(f"  {DISPLAY_NAMES[m]:20s}: q0={q0_m:.3f}, q1={q1_m:.3f}")
    else:
        q0_by_model = None
        q1_by_model = None
        print(f"Simulating judge with q0={args.q0}, q1={args.q1}")

    features = []
    if args.heterogeneous_q > 0:
        features.append(f"heterogeneous_q={args.heterogeneous_q}")
    if args.per_model_q:
        features.append("per_model_q")
    if args.shrinkage:
        features.append("shrinkage")
    if args.stratified:
        features.append("stratified")
    if features:
        print(f"Features: {', '.join(features)}")

    results = []
    fallback_counts = []
    for split in range(args.n_splits):
        split_rng = np.random.default_rng(args.seed + split * 1000 + 1)
        # New judge labels for each split
        comps = simulate_judge_labels(
            [{k: v for k, v in c.items()} for c in comparisons],
            args.q0, args.q1, split_rng,
            q0_by_model=q0_by_model,
            q1_by_model=q1_by_model,
        )
        # Split into test/calibration
        if args.stratified:
            test_indices = []
            calib_indices = []
            for m in TARGET_MODEL_NAMES:
                m_indices = [i for i, c in enumerate(comps) if c["target"] == m]
                split_rng.shuffle(m_indices)
                m_n_test = int(len(m_indices) * args.test_frac)
                test_indices.extend(m_indices[:m_n_test])
                calib_indices.extend(m_indices[m_n_test:])
            test_idx = np.array(test_indices)
            calib_idx = np.array(calib_indices)
            split_rng.shuffle(test_idx)
            split_rng.shuffle(calib_idx)
        else:
            indices = split_rng.permutation(n_total)
            test_idx = indices[:n_test]
            calib_idx = indices[n_test:]

        r = run_split(
            comps, test_idx, calib_idx,
            per_model_q=args.per_model_q,
            use_shrinkage=args.shrinkage,
            min_calib_per_model=args.min_calib_per_model,
        )
        results.append(r)
        if r.get("per_model_fallback"):
            fallback_counts.append(len(r["per_model_fallback"]))

        if (split + 1) % 10 == 0 or split < 3:
            q0_info = f"q0_hat={r['q0_hat']:.3f}, q1_hat={r['q1_hat']:.3f}"
            if args.per_model_q:
                fb = r.get("per_model_fallback", [])
                q0_info += f", fallback={len(fb)}"
            print(f"  Split {split+1}/{args.n_splits}: naive_tau={r['naive_tau']:.3f}, "
                  f"corrected_tau={r['corrected_tau']:.3f}, "
                  f"naive_exact={r['naive_exact']}, corrected_exact={r['corrected_exact']}, "
                  f"{q0_info}")

    # Aggregate
    naive_taus = [r["naive_tau"] for r in results]
    corrected_taus = [r["corrected_tau"] for r in results]
    naive_exacts = [r["naive_exact"] for r in results]
    corrected_exacts = [r["corrected_exact"] for r in results]

    # Count splits where corrected > naive (meaningful correction)
    better_corrected = sum(1 for r in results if r["corrected_tau"] > r["naive_tau"])
    worse_corrected = sum(1 for r in results if r["corrected_tau"] < r["naive_tau"])
    equal_corrected = sum(1 for r in results if abs(r["corrected_tau"] - r["naive_tau"]) < 1e-10)

    mean_nt = float(np.mean(naive_taus))
    mean_ct = float(np.mean(corrected_taus))
    std_nt = float(np.std(naive_taus))
    std_ct = float(np.std(corrected_taus))
    mean_ne = float(np.mean(naive_exacts)) * 100
    mean_ce = float(np.mean(corrected_exacts)) * 100

    elapsed = time.time() - started

    print(f"\n{'='*60}")
    print(f"Kendall tau (naive p_hat):       {mean_nt:.3f} +/- {std_nt:.3f}")
    print(f"Kendall tau (corrected theta):   {mean_ct:.3f} +/- {std_ct:.3f}")
    print(f"Exact ranking recov. (naive):    {mean_ne:.1f}%")
    print(f"Exact ranking recov. (corrected): {mean_ce:.1f}%")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Splits where corrected > naive:  {better_corrected}/{args.n_splits}")
    print(f"Splits where corrected = naive:  {equal_corrected}/{args.n_splits}")
    print(f"Splits where corrected < naive:  {worse_corrected}/{args.n_splits}")
    if args.per_model_q and fallback_counts:
        print(f"Mean models in fallback per split: {np.mean(fallback_counts):.1f}")
    print(f"{'='*60}")

    output = {
        "config": {
            "q0": args.q0, "q1": args.q1, "n_splits": args.n_splits,
            "test_frac": args.test_frac, "seed": args.seed,
            "heterogeneous_q": args.heterogeneous_q,
            "per_model_q": args.per_model_q,
            "shrinkage": args.shrinkage,
            "stratified": args.stratified,
            "min_calib_per_model": args.min_calib_per_model,
        },
        "results": {
            "kendall_tau_naive_mean": mean_nt,
            "kendall_tau_corrected_mean": mean_ct,
            "kendall_tau_naive_std": std_nt,
            "kendall_tau_corrected_std": std_ct,
            "exact_ranking_naive_pct": mean_ne,
            "exact_ranking_corrected_pct": mean_ce,
            "elapsed_seconds": elapsed,
            "splits_corrected_better": better_corrected,
            "splits_corrected_worse": worse_corrected,
        },
        "rubric": {
            "paper_naive_tau": 0.876,
            "paper_corrected_tau": 0.900,
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
