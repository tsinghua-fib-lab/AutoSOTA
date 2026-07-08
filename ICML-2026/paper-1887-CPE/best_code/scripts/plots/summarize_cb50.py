#!/usr/bin/env python3
# summarize_cb50.py
# Summarize CausalBench-50 HITL results: curves + budget tables + LaTeX + init/final bar chart.
# Keeps original functionality/outputs, while applying camera-ready cosmetics consistently.

import argparse
import glob
import json
import os
from collections import defaultdict
from typing import List

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import re


METRICS_DEFAULT = ["auprc_dir", "auroc_dir", "topk_prec", "avg_pred_entropy"]

# -----------------------------
# Policy aliases + colors (editable)
# -----------------------------
# POLICY_ALIASES = {
#     "eig": "CaPE",
#     "uncertainty": "UNC",
#     "random": "RND",
# }
#
# # Colorblind-safe palette (Okabe–Ito-ish)
# POLICY_COLORS = {
#     "eig": "#0072B2",          # blue
#     "uncertainty": "#E69F00",  # orange
#     "random": "#999999",       # gray
# }
# Policy aliases + colors (editable)
# -----------------------------
POLICY_ALIASES = {
    "eig": "CaPE",
    "uncertainty": "UNC",
    "random": "RND",
    "static_eig": "STE",
    "static_uncertainty": "STU",
    "static_random": "STR",
}


# Colorblind-safe palette (Okabe–Ito-ish)
POLICY_COLORS = {
    # Adaptive methods
    "eig": "#0072B2",                 # CaPE – strong blue
    "uncertainty": "#E69F00",          # UNC – strong orange

    # Static baselines (same hues, lighter)
    "static_eig": "#56B4E9",           # light blue
    "static_uncertainty": "#F0E442",   # light yellow/orange

    # Random baselines
    "random": "#999999",               # dark gray
    "static_random": "#CCCCCC",        # light gray
}

def apply_mpl_style(font_size: int, line_width: float) -> None:
    """Apply global matplotlib settings for camera-ready figures."""
    mpl.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 1.5,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "legend.frameon": False,
    })
    mpl.rcParams["font.size"] = font_size
    mpl.rcParams["lines.linewidth"] = line_width


def _safe_float(x):
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return np.nan
        return v
    except Exception:
        return np.nan


def load_runs(outdir: str, policies: List[str]) -> dict:
    """Load runs from outdir/**/cb50_<policy>_seed*.json."""
    patterns = [
        os.path.join(outdir, "cb50_*_seed*.json"),
        os.path.join(outdir, "**", "cb50_*_seed*.json"),
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))
    files = sorted(set(files))

    runs_by_policy = defaultdict(list)
    for fp in files:
        base = os.path.basename(fp)

        # parts = base.split("_")
        # # cb50_<policy>_seedX.json
        # if len(parts) < 3 or not base.startswith("cb50_"):
        #     continue
        # policy = parts[1]
        match = re.match(r"cb50_(.+)_seed\d+\.json", base)
        policy = match.group(1)

        if policy in policies:
            with open(fp, "r") as f:
                r = json.load(f)
            r["_path"] = fp
            runs_by_policy[policy].append(r)

    return dict(runs_by_policy)


def stack_metric(runs, key: str):
    """Returns 2D array (n_runs, T) for metric key. Pads with nan if lengths differ."""
    seqs = []
    maxT = 0
    for run in runs:
        vals = [_safe_float(entry.get(key, np.nan)) for entry in run.get("logs", [])]
        seqs.append(vals)
        maxT = max(maxT, len(vals))

    if maxT == 0:
        return np.zeros((len(runs), 0), dtype=float)

    arr = np.full((len(runs), maxT), np.nan, dtype=float)
    for i, vals in enumerate(seqs):
        arr[i, : len(vals)] = np.asarray(vals, dtype=float)
    return arr


def mean_std(arr: np.ndarray):
    m = np.nanmean(arr, axis=0)
    s = np.nanstd(arr, axis=0)
    return m, s


def _plot_with_band(x, m, s, label, *, color=None, line_width=3.5):
    plt.plot(x, m, label=label, color=color, linewidth=line_width)
    plt.fill_between(x, m - s, m + s, alpha=0.2, color=color, linewidth=0)


def save_curves(
    runs_by_policy,
    outdir: str,
    policies: List[str],
    metrics=None,
    *,
    font_size: int,
    line_width: float,
    fig_width: float,
    fig_height: float,
    show_legend: bool,
    show_titles: bool,
):
    os.makedirs(outdir, exist_ok=True)
    if metrics is None:
        metrics = METRICS_DEFAULT

    # policies = sorted(runs_by_policy.keys())
    if not policies:
        raise RuntimeError("No policies found to plot.")

    for key in metrics:
        plt.figure(figsize=(fig_width, fig_height))

        for p in policies:
            a = stack_metric(runs_by_policy[p], key)
            if a.shape[1] == 0:
                continue
            m, s = mean_std(a)
            xx = np.arange(1, len(m) + 1)
            label = POLICY_ALIASES.get(p, p)
            color = POLICY_COLORS.get(p, None)
            _plot_with_band(xx, m, s, label=label, color=color, line_width=line_width)

        # No titles by default (camera-ready)
        if show_titles:
            plt.title(key, fontsize=font_size)

        plt.xlabel("Queries", fontsize=font_size)
        if key == "auprc_dir":
            plt.ylabel("AUPRC (directed) (↑)", fontsize=font_size)
            fname = "fig_cb50_auprc.pdf"
        elif key == "auroc_dir":
            plt.ylabel("AUROC (directed) (↑)", fontsize=font_size)
            fname = "fig_cb50_auroc.pdf"
        elif key == "topk_prec":
            plt.ylabel("Top-K Precision (↑)", fontsize=font_size)
            fname = "fig_cb50_topk_prec.pdf"
        elif key == "avg_pred_entropy":
            plt.ylabel("Avg predictive entropy (↓)", fontsize=font_size)
            fname = "fig_cb50_entropy.pdf"
        else:
            plt.ylabel(key, fontsize=font_size)
            fname = f"fig_cb50_{key}.pdf"

        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.locator_params(axis="x", nbins=5)
        plt.locator_params(axis="y", nbins=4)

        if show_legend:
            plt.legend(fontsize=font_size)

        plt.tight_layout(pad=0.15)
        plt.savefig(os.path.join(outdir, fname))
        plt.close()


def save_budget_table(runs_by_policy, outdir: str, budgets=(10, 25, 50, 100, 200)):
    """Writes CSV with mean±std at specific query budgets for core metrics."""
    import csv

    os.makedirs(outdir, exist_ok=True)

    policies = sorted(runs_by_policy.keys())
    rows = []

    for p in policies:
        arrs = {
            "auprc_dir": stack_metric(runs_by_policy[p], "auprc_dir"),
            "auroc_dir": stack_metric(runs_by_policy[p], "auroc_dir"),
            "topk_prec": stack_metric(runs_by_policy[p], "topk_prec"),
            "avg_pred_entropy": stack_metric(runs_by_policy[p], "avg_pred_entropy"),
        }

        maxT = max((a.shape[1] for a in arrs.values()), default=0)

        for b in budgets:
            idx = b - 1
            if maxT == 0 or idx < 0 or idx >= maxT:
                continue

            row = {"policy": p, "budget": int(b)}
            for k, a in arrs.items():
                vals = a[:, idx] if a.shape[1] > idx else np.full((a.shape[0],), np.nan)
                row[f"{k}_mean"] = float(np.nanmean(vals))
                row[f"{k}_std"] = float(np.nanstd(vals))
            rows.append(row)

    if not rows:
        raise RuntimeError("No rows produced for budget table (budgets exceed available T?).")

    fp = os.path.join(outdir, "table_cb50_budget.csv")
    with open(fp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    return fp


def save_final_summary(runs_by_policy, outdir: str):
    """Writes summary JSON of final (last-iteration) metrics mean±std per policy."""
    os.makedirs(outdir, exist_ok=True)
    policies = sorted(runs_by_policy.keys())
    summary = {}

    for p in policies:
        finals = []
        for r in runs_by_policy[p]:
            logs = r.get("logs", [])
            if logs:
                finals.append(logs[-1])
        if not finals:
            continue

        def _mean(key):
            vals = np.asarray([_safe_float(f.get(key, np.nan)) for f in finals], dtype=float)
            return float(np.nanmean(vals))

        def _std(key):
            vals = np.asarray([_safe_float(f.get(key, np.nan)) for f in finals], dtype=float)
            return float(np.nanstd(vals))

        summary[p] = {
            "n_runs": int(len(finals)),
            "final_auprc_mean": _mean("auprc_dir"),
            "final_auprc_std": _std("auprc_dir"),
            "final_auroc_mean": _mean("auroc_dir"),
            "final_auroc_std": _std("auroc_dir"),
            "final_topk_prec_mean": _mean("topk_prec"),
            "final_topk_prec_std": _std("topk_prec"),
            "final_entropy_mean": _mean("avg_pred_entropy"),
            "final_entropy_std": _std("avg_pred_entropy"),
        }

    out_path = os.path.join(outdir, "summary_cb50_final.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    return out_path, summary


def save_init_final_bar(
    runs_by_policy,
    outdir: str,
    metric_key="auprc_dir",
    *,
    font_size: int,
    fig_width: float,
    fig_height: float,
    show_titles: bool,
):
    """Bar chart: init vs final metric per policy (mean±std)."""
    os.makedirs(outdir, exist_ok=True)
    policies = sorted(runs_by_policy.keys())
    if not policies:
        return None

    init_means, init_stds = [], []
    final_means, final_stds = [], []

    for p in policies:
        inits = []
        finals = []
        for r in runs_by_policy[p]:
            if "init" in r and metric_key in r["init"]:
                inits.append(_safe_float(r["init"].get(metric_key, np.nan)))
            logs = r.get("logs", [])
            if logs and metric_key in logs[-1]:
                finals.append(_safe_float(logs[-1].get(metric_key, np.nan)))

        inits = np.asarray(inits, dtype=float)
        finals = np.asarray(finals, dtype=float)

        init_means.append(float(np.nanmean(inits)))
        init_stds.append(float(np.nanstd(inits)))
        final_means.append(float(np.nanmean(finals)))
        final_stds.append(float(np.nanstd(finals)))

    x = np.arange(len(policies))
    width = 0.38

    plt.figure(figsize=(fig_width, fig_height))

    # Keep bar colors default; align naming cosmetics via aliases in xlabels
    xticklabels = [POLICY_ALIASES.get(p, p) for p in policies]

    plt.bar(x - width / 2, init_means, width, yerr=init_stds, capsize=3, label="init")
    plt.bar(x + width / 2, final_means, width, yerr=final_stds, capsize=3, label="final")

    plt.xticks(x, xticklabels, rotation=0, fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.ylabel(metric_key, fontsize=font_size)

    if show_titles:
        plt.title(f"Init vs Final ({metric_key})", fontsize=font_size)

    plt.legend(fontsize=font_size)
    plt.tight_layout(pad=0.15)

    fp = os.path.join(outdir, f"fig_cb50_init_final_{metric_key}.pdf")
    plt.savefig(fp)
    plt.close()
    return fp


def latex_table_from_budget_csv(
    csv_path: str,
    out_tex_path: str,
    budgets=(10, 25, 50, 100, 200),
    *,
    table_digits: int = 3,
):
    """Convert budget CSV into a LaTeX table (booktabs) with mean±std."""
    import csv

    # Load CSV into dict[policy][budget] -> metrics
    data = defaultdict(dict)
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = row["policy"]
            b = int(row["budget"])
            data[p][b] = row

    policies = sorted(data.keys())
    budgets = [b for b in budgets if any(b in data[p] for p in policies)]

    def fmt(row, key):
        m = float(row[f"{key}_mean"])
        s = float(row[f"{key}_std"])
        return f"{m:.{table_digits}f} $\\pm$ {s:.{table_digits}f}"

    lines = []
    lines.append("% Auto-generated by summarize_cb50.py")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{6pt}")
    lines.append("\\begin{tabular}{l" + "c" * len(budgets) + "}")
    lines.append("\\toprule")
    lines.append("Policy & " + " & ".join([f"$T={b}$" for b in budgets]) + " \\\\")
    lines.append("\\midrule")

    # AUPRC block
    lines.append("\\multicolumn{" + str(1 + len(budgets)) + "}{l}{\\textbf{AUPRC (directed)}} \\\\")
    for p in policies:
        cells = []
        for b in budgets:
            if b in data[p]:
                cells.append(fmt(data[p][b], "auprc_dir"))
            else:
                cells.append("--")
        pname = POLICY_ALIASES.get(p, p)
        lines.append(pname + " & " + " & ".join(cells) + " \\\\")
    lines.append("\\midrule")

    # TopK precision block
    lines.append("\\multicolumn{" + str(1 + len(budgets)) + "}{l}{\\textbf{Top-K Precision}} \\\\")
    for p in policies:
        cells = []
        for b in budgets:
            if b in data[p]:
                cells.append(fmt(data[p][b], "topk_prec"))
            else:
                cells.append("--")
        pname = POLICY_ALIASES.get(p, p)
        lines.append(pname + " & " + " & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{CausalBench-50 HITL performance (mean $\\pm$ std over seeds) at different query budgets.}")
    lines.append("\\label{tab:cb50_budget}")
    lines.append("\\end{table}")

    with open(out_tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return out_tex_path


def main():
    ap = argparse.ArgumentParser()

    # Keep functionality, but make args optional by default
    ap.add_argument(
        "--outdir",
        default="results_cb50",
        help="Root directory containing cb50_<policy>_seed*.json files (possibly in subfolders).",
    )
    ap.add_argument(
        "--policies", nargs="+",
        default=["eig", "uncertainty", "random"],
        help="List of policies to compare"
    )

    ap.add_argument(
        "--budgets",
        type=str,
        default="10,25,50,100,200",
        help="Comma-separated budgets for the CSV/LaTeX table.",
    )
    ap.add_argument(
        "--metrics",
        type=str,
        default=",".join(METRICS_DEFAULT),
        help="Comma-separated metrics to plot.",
    )
    ap.add_argument(
        "--latex_tex",
        type=str,
        default="table_cb50_budget.tex",
        help="Output LaTeX filename (saved inside outdir unless absolute).",
    )

    # Camera-ready styling (defaults match synthetic and Sachs plotting scripts)
    ap.add_argument("--font_size", type=int, default=20)
    ap.add_argument("--line_width", type=float, default=3.5)
    ap.add_argument("--fig_width", type=float, default=6.0)
    ap.add_argument("--fig_height", type=float, default=4.0)
    ap.add_argument("--legend", action="store_true", help="Show legend (default: on)")
    ap.set_defaults(legend=True)
    ap.add_argument("--titles", action="store_true", help="Enable titles (default: off)")

    # Table formatting
    ap.add_argument("--table_digits", type=int, default=3, help="Decimal places for LaTeX mean±std")

    # Bar chart options
    ap.add_argument("--bar_metric", type=str, default="auprc_dir", help="Metric for init-vs-final bar chart")
    ap.add_argument("--bar_fig_width", type=float, default=6.0)
    ap.add_argument("--bar_fig_height", type=float, default=4.0)

    args = ap.parse_args()

    apply_mpl_style(args.font_size, args.line_width)

    budgets = tuple(int(x) for x in args.budgets.split(",") if x.strip())
    metrics = [x.strip() for x in args.metrics.split(",") if x.strip()]

    runs = load_runs(args.outdir, policies=args.policies)
    assert runs, f"No run json files found under {args.outdir}"

    print("Found runs:")
    for p, rs in sorted(runs.items()):
        print(f" - {p}: {len(rs)} runs")

    save_curves(
        runs,
        args.outdir,
        policies=args.policies,
        metrics=metrics,
        font_size=args.font_size,
        line_width=args.line_width,
        fig_width=args.fig_width,
        fig_height=args.fig_height,
        show_legend=args.legend,
        show_titles=args.titles,
    )
    table_fp = save_budget_table(runs, args.outdir, budgets=budgets)
    summary_fp, _summary = save_final_summary(runs, args.outdir)

    # Init vs final bar chart
    bar_fp = save_init_final_bar(
        runs,
        args.outdir,
        metric_key=args.bar_metric,
        font_size=args.font_size,
        fig_width=args.bar_fig_width,
        fig_height=args.bar_fig_height,
        show_titles=args.titles,
    )

    # LaTeX table
    latex_path = args.latex_tex
    if not os.path.isabs(latex_path):
        latex_path = os.path.join(args.outdir, latex_path)
    tex_fp = latex_table_from_budget_csv(table_fp, latex_path, budgets=budgets, table_digits=args.table_digits)

    print("Wrote:")
    expected = [
        os.path.join(args.outdir, "fig_cb50_auprc.pdf"),
        os.path.join(args.outdir, "fig_cb50_auroc.pdf"),
        os.path.join(args.outdir, "fig_cb50_topk_prec.pdf"),
        os.path.join(args.outdir, "fig_cb50_entropy.pdf"),
        bar_fp,
        table_fp,
        tex_fp,
        summary_fp,
    ]
    for fp in expected:
        if fp and os.path.exists(fp):
            print(" -", fp)


if __name__ == "__main__":
    main()
