#!/usr/bin/env python3
"""
Plot probe-budget ROC summary figures from evidence/bbob_noisy_probe_budget_roc/.

Produces:
- auc_vs_lam.png (AUC and accuracy at report threshold vs probe λ)
Optionally:
- roc_curves.png (TPR vs FPR curves per λ)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _project import BASE_DIR, repo_relpath


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-dir",
        default="evidence/bbob_noisy_probe_budget_roc",
        help="Input directory containing summary.json and roc.csv.",
    )
    parser.add_argument(
        "--out-auc",
        default="evidence/bbob_noisy_probe_budget_roc/auc_vs_lam.png",
        help="Output AUC-vs-lam PNG.",
    )
    parser.add_argument(
        "--out-roc",
        default="evidence/bbob_noisy_probe_budget_roc/roc_curves.png",
        help="Output ROC curves PNG.",
    )
    parser.add_argument("--skip-roc", action="store_true", help="Skip ROC curve plot.")
    args = parser.parse_args()

    os.chdir(BASE_DIR)

    in_dir = Path(str(args.in_dir))
    summary_path = in_dir / "summary.json"
    roc_path = in_dir / "roc.csv"
    if not summary_path.exists():
        raise SystemExit(f"Missing: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    by_lam = summary.get("by_lam", {})
    lams = sorted(int(k) for k in by_lam.keys())
    aucs = [float(by_lam[str(l)]["auc"]) for l in lams]
    accs = [float(by_lam[str(l)]["accuracy_at_report_threshold"]) for l in lams]
    rep_t = float(next(iter(by_lam.values()))["report_threshold"]) if by_lam else float("nan")

    fig, ax1 = plt.subplots(figsize=(6.2, 3.6), constrained_layout=True)
    ax1.plot(lams, aucs, marker="o", label="AUC")
    ax1.set_xlabel("probe population size λ (≈ 2λ evals)")
    ax1.set_ylabel("AUC")
    ax1.set_ylim(0.5, 1.0)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(lams, accs, marker="s", color="tab:orange", label=f"accuracy @ t={rep_t:g}")
    ax2.set_ylabel("accuracy")
    ax2.set_ylim(0.5, 1.0)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="lower right")
    ax1.set_title("Misranking-probe reliability vs probe budget (bbob-noisy)")

    out_auc = Path(str(args.out_auc))
    out_auc.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_auc, dpi=200)
    plt.close(fig)
    print("Wrote:", repo_relpath(str(out_auc)))

    if args.skip_roc:
        return
    if not roc_path.exists():
        print("Skip ROC curves (missing roc.csv):", str(roc_path))
        return

    import csv

    rows: list[dict[str, str]] = []
    with roc_path.open("r", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    if not rows:
        print("Skip ROC curves (empty roc.csv).")
        return

    fig, ax = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
    colors = ["tab:blue", "tab:green", "tab:purple", "tab:red", "tab:brown"]
    for idx, lam in enumerate(lams):
        rr = [r for r in rows if int(float(r["lam"])) == int(lam)]
        rr.sort(key=lambda x: float(x["fpr"]))
        fpr = np.asarray([float(r["fpr"]) for r in rr], dtype=float)
        tpr = np.asarray([float(r["tpr"]) for r in rr], dtype=float)
        ax.plot(fpr, tpr, color=colors[idx % len(colors)], label=f"λ={lam} (AUC={by_lam[str(lam)]['auc']:.3f})")

    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC curves (misranking probe)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    out_roc = Path(str(args.out_roc))
    out_roc.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_roc, dpi=200)
    plt.close(fig)
    print("Wrote:", repo_relpath(str(out_roc)))


if __name__ == "__main__":
    main()
