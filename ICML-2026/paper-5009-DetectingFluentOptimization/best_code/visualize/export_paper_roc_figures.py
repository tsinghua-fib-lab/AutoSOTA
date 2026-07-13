#!/usr/bin/env python3
"""
Export vector PDF ROC curves used in the paper without rerunning
any detection pipeline.

This script reads existing per-prompt score CSVs (PP / CPD scans) from
results/changepoints and writes paper-ready ROC PDFs with the filenames
expected by the LaTeX source (or similar naming).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    positives = y_true == 1
    negatives = ~positives
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores)
    ranked = np.empty_like(order, dtype=float)
    ranked[order] = np.arange(len(scores))
    sum_ranks_pos = float(ranked[positives].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos - 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def compute_roc_curve(y_true: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(scores)[::-1]
    y_sorted = y_true[order]
    pos = y_sorted == 1
    neg = ~pos
    n_pos = float(pos.sum())
    n_neg = float(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    tps = np.cumsum(pos)
    fps = np.cumsum(neg)
    tpr = np.concatenate(([0.0], tps / n_pos, [1.0]))
    fpr = np.concatenate(([0.0], fps / n_neg, [1.0]))
    return fpr, tpr


def _paper_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "lines.linewidth": 2.2,
        }
    )


def plot_roc_segmented(
    df: pd.DataFrame,
    *,
    score_col: str,
    label_col: str,
    algo_col: str,
    title: str,
    out_pdf: str,
    out_png: Optional[str] = None,
    dpi: int = 300,
) -> Dict[str, float]:
    algo = df[algo_col].fillna("").astype(str).str.lower()
    y_all = df[label_col].fillna(0).astype(int).to_numpy()
    scores_all = df[score_col].astype(float).to_numpy()

    mask_gcg = algo == "gcg"
    segments: Dict[str, np.ndarray] = {
        "overall": np.ones(len(df), dtype=bool),
        "no_gcg": ~mask_gcg,
        "gcg_only": mask_gcg | ((~mask_gcg) & (y_all == 0)),
    }

    _paper_style()
    plt.figure(figsize=(4.6, 4.2))
    ax = plt.gca()

    aucs: Dict[str, float] = {}
    for seg_name, seg_mask in segments.items():
        if seg_mask.sum() == 0:
            continue
        y = y_all[seg_mask]
        s = scores_all[seg_mask]
        finite = np.isfinite(s)
        y = y[finite]
        s = s[finite]
        if len(y) < 2 or len(np.unique(y)) < 2:
            continue
        fpr, tpr = compute_roc_curve(y, s)
        auc_val = compute_auroc(y, s)
        aucs[seg_name] = auc_val
        ax.plot(fpr, tpr, label=f"{seg_name} (AUC={auc_val:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5, linewidth=1.6)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    plt.savefig(out_pdf, bbox_inches="tight")
    if out_png is not None:
        plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close()
    return aucs


@dataclass(frozen=True)
class RocSpec:
    in_csv: str
    score_col: str
    label_col: str
    algo_col: str
    title: str
    out_base: str  # filename without extension


def _specs(changepoints_dir: str) -> Iterable[RocSpec]:
    def cp(path: str) -> str:
        return os.path.join(changepoints_dir, path)

    return [
        RocSpec(
            in_csv=cp("llama-7b_benign_mix_ppgap5_700_k_0_cpd_scan.csv"),
            score_col="online_max_W_plus",
            label_col="is_adversarial",
            algo_col="algorithm",
            title="ROC (adv vs benign) — CPD Online",
            out_base="cpd_online_roc_segmented_ppgap5_700_k0",
        ),
        RocSpec(
            in_csv=cp("llama-7b_benign_mix_ppgap5_700_k_0.5_cpd_scan.csv"),
            score_col="online_max_W_plus",
            label_col="is_adversarial",
            algo_col="algorithm",
            title="ROC (adv vs benign) — CPD Online",
            out_base="cpd_online_roc_segmented_ppgap5_700_k0p5",
        ),
        RocSpec(
            in_csv=cp("llama-7b_benign_mix_ppgap1_800_k_0_cpd_scan.csv"),
            score_col="online_max_W_plus",
            label_col="is_adversarial",
            algo_col="algorithm",
            title="ROC (adv vs benign) — CPD Online",
            out_base="cpd_online_roc_segmented_ppgap1_800_k0",
        ),
        RocSpec(
            in_csv=cp("llama-7b_benign_mix_ppgap2_800_k_0_cpd_scan.csv"),
            score_col="online_max_W_plus",
            label_col="is_adversarial",
            algo_col="algorithm",
            title="ROC (adv vs benign) — CPD Online",
            out_base="cpd_online_roc_segmented_ppgap2_800_k0",
        ),
        RocSpec(
            in_csv=cp("llama-7b_benign_mix_ppgap3_800_k_0_cpd_scan.csv"),
            score_col="online_max_W_plus",
            label_col="is_adversarial",
            algo_col="algorithm",
            title="ROC (adv vs benign) — CPD Online",
            out_base="cpd_online_roc_segmented_ppgap3_800_k0",
        ),
        RocSpec(
            in_csv=cp("llama-7b_benign_mix_ppgap1_800_pp.csv"),
            score_col="window_mean_nll_w15",
            label_col="is_adversarial",
            algo_col="algorithm",
            title="ROC (adv vs benign) — WPP15",
            out_base="window_pp_w15_roc_segmented_ppgap1_800_k0",
        ),
        RocSpec(
            in_csv=cp("llama-7b_benign_mix_ppgap2_800_pp.csv"),
            score_col="window_mean_nll_w15",
            label_col="is_adversarial",
            algo_col="algorithm",
            title="ROC (adv vs benign) — WPP15",
            out_base="window_pp_w15_roc_segmented_ppgap2_800_k0",
        ),
        RocSpec(
            in_csv=cp("llama-7b_benign_mix_ppgap3_800_pp.csv"),
            score_col="window_mean_nll_w15",
            label_col="is_adversarial",
            algo_col="algorithm",
            title="ROC (adv vs benign) — WPP15",
            out_base="window_pp_w15_roc_segmented_ppgap3_800_k0",
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--changepoints-dir",
        default="results/changepoints",
        help="Directory containing *_pp.csv and *_cpd_scan.csv outputs.",
    )
    parser.add_argument(
        "--out-dir",
        default="paper_figures",
        help="Output directory for paper ROC PDFs (default: paper_figures).",
    )
    parser.add_argument("--also-png", action="store_true", help="Also export high-DPI PNGs alongside PDFs.")
    parser.add_argument("--dpi", type=int, default=300, help="PNG dpi when --also-png is enabled.")
    args = parser.parse_args()

    for spec in _specs(args.changepoints_dir):
        df = pd.read_csv(spec.in_csv)
        out_pdf = os.path.join(args.out_dir, f"{spec.out_base}.pdf")
        out_png = os.path.join(args.out_dir, f"{spec.out_base}.png") if args.also_png else None
        aucs = plot_roc_segmented(
            df,
            score_col=spec.score_col,
            label_col=spec.label_col,
            algo_col=spec.algo_col,
            title=spec.title,
            out_pdf=out_pdf,
            out_png=out_png,
            dpi=args.dpi,
        )
        auc_str = ", ".join(f"{k}={v:.3f}" for k, v in aucs.items()) if aucs else "no curves"
        print(f"[OK] {os.path.basename(out_pdf)} ({auc_str})")


if __name__ == "__main__":
    main()
