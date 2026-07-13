#!/usr/bin/env python3
"""
Analyze computational savings from using a lightweight detector as a gating mechanism for Llama Guard.

This script measures:
1. Guard-only performance: Run Llama Guard on all prompts
2. Detector-only performance: Use only a lightweight detector score
3. Hybrid performance: Run detector first, invoke Guard only when detector triggers

Outputs:
- Percentage of Guard calls saved at different CPD thresholds
- Detection performance (precision, recall, F1) for each approach
- Trade-off curves: detection performance vs. computational cost
"""

import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute precision, recall, F1, TPR, FPR."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    tpr = recall
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tpr": tpr,
        "fpr": fpr,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }


def analyze_hybrid_savings(
    detector_scores: np.ndarray,
    guard_predictions: np.ndarray,
    true_labels: np.ndarray,
    guard_runtime: float,
    detector_runtime: float = 1e-6,  # negligible
) -> pd.DataFrame:
    """
    Sweep detector threshold and compute:
    - Detection performance for detector-only, Guard-only, and Hybrid
    - Percentage of Guard calls saved
    - Total runtime savings
    """
    results = []

    # Sort unique detector scores for threshold sweep
    finite_scores = detector_scores[np.isfinite(detector_scores)]
    thresholds = np.sort(np.unique(finite_scores))
    if thresholds.size == 0:
        raise ValueError("Detector scores contain no finite values; cannot sweep thresholds.")

    # Guard-only metrics (baseline)
    guard_metrics = compute_metrics(true_labels, guard_predictions)
    total_prompts = len(true_labels)

    for threshold in thresholds:
        # Detector predictions at this threshold
        det_preds = (detector_scores >= threshold).astype(int)

        # Hybrid: CPD triggers AND Guard agrees
        det_triggers = detector_scores >= threshold
        hybrid_preds = np.zeros_like(det_preds)
        hybrid_preds[det_triggers] = guard_predictions[det_triggers]

        # Compute metrics
        det_metrics = compute_metrics(true_labels, det_preds)
        hybrid_metrics = compute_metrics(true_labels, hybrid_preds)

        # Computational savings
        num_guard_calls = int(np.sum(det_triggers))
        guard_calls_saved = total_prompts - num_guard_calls
        pct_saved = 100.0 * guard_calls_saved / total_prompts

        # Runtime analysis
        guard_only_time = total_prompts * guard_runtime
        hybrid_time = total_prompts * detector_runtime + num_guard_calls * guard_runtime
        time_saved = guard_only_time - hybrid_time
        pct_time_saved = 100.0 * time_saved / guard_only_time

        results.append({
            "threshold": threshold,
            # Detector-only
            "det_precision": det_metrics["precision"],
            "det_recall": det_metrics["recall"],
            "det_f1": det_metrics["f1"],
            "det_fpr": det_metrics["fpr"],
            # Guard-only (constant)
            "guard_precision": guard_metrics["precision"],
            "guard_recall": guard_metrics["recall"],
            "guard_f1": guard_metrics["f1"],
            "guard_fpr": guard_metrics["fpr"],
            # Hybrid
            "hybrid_precision": hybrid_metrics["precision"],
            "hybrid_recall": hybrid_metrics["recall"],
            "hybrid_f1": hybrid_metrics["f1"],
            "hybrid_fpr": hybrid_metrics["fpr"],
            # Savings
            "num_guard_calls": num_guard_calls,
            "guard_calls_saved": int(guard_calls_saved),
            "pct_guard_calls_saved": pct_saved,
            "hybrid_runtime_sec": hybrid_time,
            "guard_only_runtime_sec": guard_only_time,
            "time_saved_sec": time_saved,
            "pct_time_saved": pct_time_saved,
        })

    return pd.DataFrame(results)


def pick_best_with_tiebreak(
    df: pd.DataFrame,
    primary_col: str,
    *,
    maximize_primary: bool = True,
    tiebreak_col: str = "pct_guard_calls_saved",
    atol: float = 1e-12,
) -> pd.Series:
    """
    Select a row optimizing ``primary_col`` and break ties by maximizing ``tiebreak_col``.

    This avoids degenerate selections when the primary metric is flat over a range
    of thresholds (e.g., when the hybrid matches Guard performance for multiple
    thresholds but some save more Guard calls than others).
    """
    if df.empty:
        raise ValueError("Cannot pick best row from empty DataFrame.")
    primary = df[primary_col].astype(float)
    best_val = primary.max() if maximize_primary else primary.min()
    if maximize_primary:
        candidates = df[primary >= best_val - atol]
    else:
        candidates = df[primary <= best_val + atol]
    if candidates.empty:
        candidates = df
    tiebreak = candidates[tiebreak_col].astype(float)
    return candidates.loc[tiebreak.idxmax()]


def plot_savings_vs_performance(df: pd.DataFrame, output_dir: Path):
    """Create visualizations of the trade-off."""
    output_dir.mkdir(parents=True, exist_ok=True)
    method_label = str(df["method"].iloc[0]) if "method" in df.columns and len(df) else "detector"

    # 1. F1 vs. Guard Calls Saved
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(df["pct_guard_calls_saved"], df["det_f1"], label="Detector-only", marker='o', alpha=0.7)
    ax.plot(df["pct_guard_calls_saved"], df["hybrid_f1"], label="Hybrid (Detector+Guard)", marker='s', alpha=0.7)
    ax.axhline(df["guard_f1"].iloc[0], color='red', linestyle='--', label="Guard-only", linewidth=2)
    ax.set_xlabel("% of Guard Calls Saved", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title(f"Detection Performance vs. Computational Savings ({method_label})", fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "f1_vs_guard_savings.png", dpi=300)
    plt.close()

    # 2. Precision-Recall curve for all three approaches
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(df["det_recall"], df["det_precision"], label="Detector-only", marker='o', alpha=0.7)
    ax.plot(df["hybrid_recall"], df["hybrid_precision"], label="Hybrid", marker='s', alpha=0.7)
    ax.scatter([df["guard_recall"].iloc[0]], [df["guard_precision"].iloc[0]],
               color='red', s=100, marker='*', label="Guard-only", zorder=5)
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision-Recall: Detector vs. Guard vs. Hybrid ({method_label})", fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_comparison.png", dpi=300)
    plt.close()

    # 3. Runtime savings bar chart at selected operating points
    # Pick 3 operating points: high recall, balanced F1, low FPR
    high_recall_row = pick_best_with_tiebreak(df, "hybrid_recall", tiebreak_col="pct_guard_calls_saved")
    balanced_f1_row = pick_best_with_tiebreak(df, "hybrid_f1", tiebreak_col="pct_guard_calls_saved")
    if (df["hybrid_fpr"] < 0.1).any():
        low_fpr_row = pick_best_with_tiebreak(df[df["hybrid_fpr"] < 0.1], "hybrid_f1", tiebreak_col="pct_guard_calls_saved")
    else:
        low_fpr_row = balanced_f1_row

    selected = pd.DataFrame([high_recall_row, balanced_f1_row, low_fpr_row]).reset_index(drop=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    x = np.arange(len(selected))
    width = 0.35

    guard_times = selected["guard_only_runtime_sec"].values
    hybrid_times = selected["hybrid_runtime_sec"].values

    ax.bar(x - width/2, guard_times, width, label="Guard-only", alpha=0.8)
    ax.bar(x + width/2, hybrid_times, width, label="Hybrid (Detector+Guard)", alpha=0.8)

    ax.set_ylabel("Runtime (seconds)", fontsize=12)
    ax.set_title("Runtime Comparison at Different Operating Points", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"High Recall\n(F1={selected.iloc[0]['hybrid_f1']:.2f})",
                        f"Best F1\n(F1={selected.iloc[1]['hybrid_f1']:.2f})",
                        f"Low FPR\n(F1={selected.iloc[2]['hybrid_f1']:.2f})"])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add percentage saved labels
    for i, (g, h) in enumerate(zip(guard_times, hybrid_times)):
        pct = 100 * (g - h) / g
        ax.text(i, h + 0.5, f"{pct:.1f}% saved", ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / "runtime_comparison.png", dpi=300)
    plt.close()

    print(f"Saved plots to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze computational savings from detector+Guard hybrid approach"
    )
    parser.add_argument(
        "--cpd-csv",
        default=None,
        help="CPD results CSV (must have 'online_max_W_plus' column). If provided, computes savings for CPD gating.",
    )
    parser.add_argument(
        "--pp-csv",
        default=None,
        help="Per-prompt PP CSV (from perplexity_detector_metrics_paper_f1.py). If provided, computes savings for PP + window-PP gating.",
    )
    parser.add_argument(
        "--window-sizes",
        nargs="+",
        type=int,
        default=[5, 10, 15, 20],
        help="Window sizes to analyze for window-PP gating (default: 5 10 15 20).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["cpd", "pp", "window"],
        default=["cpd", "pp", "window"],
        help="Which detectors to include when the corresponding CSV is provided (default: cpd pp window).",
    )
    parser.add_argument(
        "--guard-csv",
        required=True,
        help="Llama Guard results CSV (must have 'LG1_Prediction' or 'LG2_Prediction', 'True_Label', 'LG*_Runtime')"
    )
    parser.add_argument(
        "--output-csv",
        default="results/hybrid_guard_savings.csv",
        help="Output CSV with threshold sweep results"
    )
    parser.add_argument(
        "--output-dir",
        default="results/figures/hybrid_guard",
        help="Directory for output plots"
    )
    parser.add_argument(
        "--guard-version",
        choices=["lg1", "lg2", "lg3"],
        default="lg2",
        help="Which Llama Guard version to use (lg1, lg2, lg3)"
    )
    args = parser.parse_args()

    if not args.cpd_csv and not args.pp_csv:
        raise ValueError("Provide at least one of --cpd-csv or --pp-csv.")

    # Load data
    print(f"Loading Guard results from {args.guard_csv}...")
    guard_df = pd.read_csv(args.guard_csv)

    guard_pred_col = f"{args.guard_version.upper()}_Prediction"
    guard_runtime_col = f"{args.guard_version.upper()}_Runtime"

    if guard_pred_col not in guard_df.columns:
        raise ValueError(f"Guard CSV must have '{guard_pred_col}' column")
    if guard_runtime_col not in guard_df.columns:
        raise ValueError(f"Guard CSV must have '{guard_runtime_col}' column")

    if "row_index" not in guard_df.columns:
        guard_df = guard_df.copy()
        guard_df["row_index"] = np.arange(len(guard_df), dtype=int)

    guard_df = guard_df.copy()
    guard_df["row_index"] = guard_df["row_index"].astype(int)

    # Extract data
    guard_predictions = guard_df[guard_pred_col].fillna(0).astype(int).values
    if "True_Label" in guard_df.columns:
        true_labels = guard_df["True_Label"].fillna(0).astype(int).values
    else:
        raise ValueError("Guard CSV must have 'True_Label' column for this analysis.")
    avg_guard_runtime = guard_df[guard_runtime_col].mean()
    if not np.isfinite(avg_guard_runtime):
        raise ValueError(f"Average Guard runtime is not finite (column {guard_runtime_col}).")

    base_index = guard_df[["row_index"]].copy()

    print(f"\nDataset size: {len(true_labels)} prompts")
    print(f"Adversarial: {np.sum(true_labels)} ({100*np.sum(true_labels)/len(true_labels):.1f}%)")
    print(f"Average Guard runtime: {avg_guard_runtime:.4f} seconds per prompt")

    def ensure_row_index(df: pd.DataFrame) -> pd.DataFrame:
        if "row_index" in df.columns:
            out = df.copy()
            out["row_index"] = out["row_index"].astype(int)
            return out
        out = df.copy()
        out["row_index"] = np.arange(len(out), dtype=int)
        return out

    def align_scores(det_df: pd.DataFrame, score_col: str, *, name: str) -> np.ndarray:
        det_df = ensure_row_index(det_df)
        if score_col not in det_df.columns:
            raise ValueError(f"{name} CSV missing score column '{score_col}'")
        merged = pd.merge(base_index, det_df[["row_index", score_col]], on="row_index", how="left")
        if len(merged) != len(base_index):
            raise ValueError(f"{name} alignment failed: expected {len(base_index)} rows, got {len(merged)}")
        if merged[score_col].isna().any():
            missing = int(merged[score_col].isna().sum())
            raise ValueError(f"{name} alignment missing {missing} score values for column '{score_col}'")
        return merged[score_col].astype(float).to_numpy()

    methods_results: List[pd.DataFrame] = []

    if args.cpd_csv and "cpd" in args.methods:
        print(f"\nLoading CPD results from {args.cpd_csv}...")
        cpd_df = pd.read_csv(args.cpd_csv)
        scores = align_scores(cpd_df, "online_max_W_plus", name="CPD")
        print("Analyzing hybrid savings across CPD thresholds...")
        cpd_results = analyze_hybrid_savings(scores, guard_predictions, true_labels, avg_guard_runtime)
        cpd_results.insert(0, "method", "cpd_online")
        methods_results.append(cpd_results)

    if args.pp_csv:
        print(f"\nLoading PP results from {args.pp_csv}...")
        pp_df = pd.read_csv(args.pp_csv)

        if "pp" in args.methods:
            scores = align_scores(pp_df, "global_mean_nll", name="PP")
            print("Analyzing hybrid savings across PP thresholds...")
            pp_results = analyze_hybrid_savings(scores, guard_predictions, true_labels, avg_guard_runtime)
            pp_results.insert(0, "method", "pp_global")
            methods_results.append(pp_results)

        if "window" in args.methods:
            window_sizes = sorted({int(w) for w in args.window_sizes if int(w) > 0})
            for w in window_sizes:
                col = f"window_mean_nll_w{w}"
                if col not in pp_df.columns:
                    print(f"[WARN] PP CSV missing {col}; skipping window_pp_w{w}")
                    continue
                scores = align_scores(pp_df, col, name=f"window-PP w={w}")
                print(f"Analyzing hybrid savings across window-PP (w={w}) thresholds...")
                w_results = analyze_hybrid_savings(scores, guard_predictions, true_labels, avg_guard_runtime)
                w_results.insert(0, "method", f"window_pp_w{w}")
                methods_results.append(w_results)

    if not methods_results:
        raise ValueError("No detectors selected to analyze. Check --methods and provided CSVs.")

    results_df = pd.concat(methods_results, ignore_index=True)

    # Save results
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")

    # Print summary at key operating points
    print("\n" + "="*80)
    print("SUMMARY: Computational Savings at Key Operating Points")
    print("="*80)

    for method in sorted(results_df["method"].unique()):
        method_df = results_df[results_df["method"] == method]
        if method_df.empty:
            continue

        f1_opt = pick_best_with_tiebreak(method_df, "hybrid_f1", tiebreak_col="pct_guard_calls_saved")
        print(f"\n[{method}] F1-optimal threshold = {f1_opt['threshold']:.4f}")
        print(f"  Detector-only: P={f1_opt['det_precision']:.3f} R={f1_opt['det_recall']:.3f} F1={f1_opt['det_f1']:.3f}")
        print(f"  Guard-only:    P={f1_opt['guard_precision']:.3f} R={f1_opt['guard_recall']:.3f} F1={f1_opt['guard_f1']:.3f}")
        print(f"  Hybrid:        P={f1_opt['hybrid_precision']:.3f} R={f1_opt['hybrid_recall']:.3f} F1={f1_opt['hybrid_f1']:.3f}")
        print(f"  Guard calls saved: {f1_opt['guard_calls_saved']:.0f} / {len(true_labels)} ({f1_opt['pct_guard_calls_saved']:.1f}%)")

        fpr_subset = method_df[method_df["hybrid_fpr"] <= 0.10]
        if not fpr_subset.empty:
            fpr_10 = pick_best_with_tiebreak(fpr_subset, "hybrid_f1", tiebreak_col="pct_guard_calls_saved")
            print(f"  FPR≤10%: thr={fpr_10['threshold']:.4f} F1={fpr_10['hybrid_f1']:.3f} saved={fpr_10['pct_guard_calls_saved']:.1f}%")

    print("\n" + "="*80)

    # Generate plots
    print("\nGenerating visualizations...")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for method in sorted(results_df["method"].unique()):
        method_df = results_df[results_df["method"] == method]
        if method_df.empty:
            continue
        plot_savings_vs_performance(method_df, out_dir / method)

    # Combined comparison plot (hybrid F1 vs savings).
    plt.figure(figsize=(9, 6))
    for method in sorted(results_df["method"].unique()):
        method_df = results_df[results_df["method"] == method]
        if method_df.empty:
            continue
        plt.plot(method_df["pct_guard_calls_saved"], method_df["hybrid_f1"], label=method, alpha=0.8)
    plt.axhline(results_df["guard_f1"].iloc[0], color="red", linestyle="--", label="Guard-only", linewidth=2)
    plt.xlabel("% of Guard Calls Saved")
    plt.ylabel("Hybrid F1")
    plt.title("Hybrid F1 vs Guard Savings (all methods)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "hybrid_f1_vs_guard_savings_all_methods.png", dpi=300)
    plt.close()

    print("\nDone!")


if __name__ == "__main__":
    main()
