#!/usr/bin/env python3
# summarize_sachs.py
# Summarize results of sachs experiments + generate before/after posterior heatmaps.

import argparse, glob, json, os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import re


# -----------------------------
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


def load_runs(outdir, policies):
    runs = {}
    for fp in glob.glob(os.path.join(outdir, "sachs_*_seed*.json")):
        base = os.path.basename(fp)
        # policy = base.split("_")[1]  # sachs_<policy>_seedX.json
        match = re.match(r"sachs_(.+)_seed\d+\.json", base)
        policy = match.group(1)
        # print(policy)
        if policy in policies:
            with open(fp, "r") as f:
                r = json.load(f)
            runs.setdefault(policy, []).append(r)
    return runs


def get_T(runs_by_policy):
    # choose the first run we see
    for _, runs in runs_by_policy.items():
        if runs:
            return len(runs[0]["logs"])
    raise ValueError("No runs found.")


def stack_metric(runs, key, T):
    """
    Returns array shape (n_runs, T) float.
    Missing entries are filled with np.nan.
    """
    out = np.full((len(runs), T), np.nan, dtype=float)
    for r_i, run in enumerate(runs):
        logs = run.get("logs", [])
        if len(logs) != T:
            raise ValueError(f"Inconsistent T: expected {T}, got {len(logs)} in seed={run.get('seed')}")
        for t, entry in enumerate(logs):
            if key in entry:
                out[r_i, t] = float(entry[key])
    return out


def mean_std(arr):
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


def save_curves(
    runs_by_policy,
    outdir,
    policies,
    font_size: int,
    line_width: float,
    fig_width: float,
    fig_height: float,
    show_legend: bool,
):
    # policies = sorted(runs_by_policy.keys()
    # policies = runs_by_policy.keys()
    T = get_T(runs_by_policy)
    x = np.arange(1, T + 1)

    def _plot(metric_key, ylabel, fname):
        plt.figure(figsize=(fig_width, fig_height))

        for p in policies:
            a = stack_metric(runs_by_policy[p], metric_key, T)
            m, s = mean_std(a)

            color = POLICY_COLORS.get(p, None)
            label = POLICY_ALIASES.get(p, p)

            plt.fill_between(x, m - s, m + s, alpha=0.2, color=color, linewidth=0)
            plt.plot(x, m, label=label, color=color, linewidth=line_width)

        # No titles (camera-ready)
        plt.xlabel("Queries", fontsize=font_size)
        plt.ylabel(ylabel, fontsize=font_size)
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)

        # fewer ticks for compact plots
        plt.locator_params(axis="x", nbins=5)
        plt.locator_params(axis="y", nbins=4)

        if show_legend:
            plt.legend(fontsize=font_size)

        plt.tight_layout()
        fout = os.path.join(outdir, fname)
        plt.savefig(f"{fout}.png" , dpi=300)
        plt.savefig(f"{fout}.pdf")
        plt.close()


    metrics = [
        ("avg_pred_entropy", "Entropy"),
        ("exp_true_class_prob", "ETCP"),
        ("brier", "Brier"),
        ("samp_skel_f1", "Skeleton F1"),
        ("samp_orient_f1", "Orientation F1"),
        ("samp_shd", "SHD"),
    ]
    for key, ylabel in metrics:
        _plot(metric_key=key, ylabel=ylabel, fname=f"fig_sachs_{key}")
    #_plot("samp_shd", "SHD (↓)", "fig_sachs_shd.pdf")
    #_plot("samp_orient_f1", "Orientation F1 (↑)", "fig_sachs_orient_f1.pdf")
    #_plot("avg_pred_entropy", "Avg predictive entropy (↓)", "fig_sachs_entropy.pdf")


def save_budget_table(runs_by_policy, outdir, budgets=(5, 10, 20, 40)):
    policies = sorted(runs_by_policy.keys())
    T = get_T(runs_by_policy)
    rows = []

    for p in policies:
        runs = runs_by_policy[p]
        shd = stack_metric(runs, "samp_shd", T)
        f1 = stack_metric(runs, "samp_orient_f1", T)
        ent = stack_metric(runs, "avg_pred_entropy", T)

        # init metrics (shape n_runs,)
        shd0 = np.asarray([r.get("init", {}).get("samp_shd", np.nan) for r in runs], dtype=float)
        f10 = np.asarray([r.get("init", {}).get("samp_orient_f1", np.nan) for r in runs], dtype=float)
        ent0 = np.asarray([r.get("init", {}).get("avg_pred_entropy", np.nan) for r in runs], dtype=float)  # if saved; else nan

        for b in budgets:
            if b > T:
                continue
            idx = b - 1

            shd_b = shd[:, idx]
            f1_b = f1[:, idx]
            ent_b = ent[:, idx]

            rows.append({
                "policy": p,
                "budget": int(b),

                "shd_mean": float(np.nanmean(shd_b)),
                "shd_std": float(np.nanstd(shd_b)),
                "shd_delta_mean": float(np.nanmean(shd_b - shd0)),
                "shd_delta_std": float(np.nanstd(shd_b - shd0)),

                "orient_f1_mean": float(np.nanmean(f1_b)),
                "orient_f1_std": float(np.nanstd(f1_b)),
                "orient_f1_delta_mean": float(np.nanmean(f1_b - f10)),
                "orient_f1_delta_std": float(np.nanstd(f1_b - f10)),

                "entropy_mean": float(np.nanmean(ent_b)),
                "entropy_std": float(np.nanstd(ent_b)),
            })

    import csv
    fp = os.path.join(outdir, "table_sachs_budget.csv")
    with open(fp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_latex_table_budget(csv_path: str, tex_path: str, budgets=(5, 10, 20, 40), policy_order=None):
    """
    Reads table_sachs_budget.csv and writes a LaTeX table.
    Assumes CSV has columns at least:
      policy, budget, shd_mean, shd_std, orient_f1_mean, orient_f1_std
    If you added deltas, it will include them too (if present).
    """
    import csv

    with open(csv_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    budgets = [int(b) for b in budgets]
    rows = [r for r in rows if int(r["budget"]) in budgets]

    if policy_order is None:
        policy_order = sorted({r["policy"] for r in rows})

    def fmt(mean_key, std_key, r, ndp=2):
        m = float(r[mean_key]); s = float(r[std_key])
        return f"{m:.{ndp}f} $\\pm$ {s:.{ndp}f}"

    def has(*cols):
        return all(c in rows[0] for c in cols) if rows else False

    has_shd_delta = has("shd_delta_mean", "shd_delta_std")
    has_f1_delta = has("orient_f1_delta_mean", "orient_f1_delta_std")

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")

    colspec = "ll" + "c" * (len(budgets))
    lines.append(rf"\begin{{tabular}}{{{colspec}}}")
    lines.append(r"\toprule")

    budget_hdr = " & ".join([rf"$T={b}$" for b in budgets])
    lines.append(rf"\textbf{{Metric}} & \textbf{{Policy}} & {budget_hdr} \\")
    lines.append(r"\midrule")

    def get_row(policy, budget):
        for r in rows:
            if r["policy"] == policy and int(r["budget"]) == budget:
                return r
        raise KeyError(f"Missing row for policy={policy}, budget={budget}")

    # Pretty policy names (editable)
    pretty = {
        "eig": POLICY_ALIASES.get("eig", "eig"),
        "uncertainty": POLICY_ALIASES.get("uncertainty", "uncertainty"),
        "random": POLICY_ALIASES.get("random", "random"),
        "static_eig": POLICY_ALIASES.get("static_eig", "static_eig"),
        "static_random": POLICY_ALIASES.get("static_random", "static_random"),
        "static_uncertainty": POLICY_ALIASES.get("static_uncertainty", "static_uncertainty"),
    }

    for policy in policy_order:
        vals = [fmt("shd_mean", "shd_std", get_row(policy, b), ndp=2) for b in budgets]
        pname = pretty.get(policy, policy)
        lines.append(rf"\multirow{{1}}{{*}}{{SHD $\downarrow$}} & {pname} & " + " & ".join(vals) + r" \\")
    lines.append(r"\addlinespace")

    for policy in policy_order:
        vals = [fmt("orient_f1_mean", "orient_f1_std", get_row(policy, b), ndp=3) for b in budgets]
        pname = pretty.get(policy, policy)
        lines.append(rf"\multirow{{1}}{{*}}{{Orient.\ F1 $\uparrow$}} & {pname} & " + " & ".join(vals) + r" \\")
    lines.append(r"\addlinespace")

    if has_shd_delta:
        for policy in policy_order:
            vals = [fmt("shd_delta_mean", "shd_delta_std", get_row(policy, b), ndp=2) for b in budgets]
            pname = pretty.get(policy, policy)
            lines.append(rf"\multirow{{1}}{{*}}{{$\Delta$SHD $\downarrow$}} & {pname} & " + " & ".join(vals) + r" \\")
        lines.append(r"\addlinespace")

    if has_f1_delta:
        for policy in policy_order:
            vals = [fmt("orient_f1_delta_mean", "orient_f1_delta_std", get_row(policy, b), ndp=3) for b in budgets]
            pname = pretty.get(policy, policy)
            lines.append(rf"\multirow{{1}}{{*}}{{$\Delta$Orient.\ F1 $\uparrow$}} & {pname} & " + " & ".join(vals) + r" \\")
        lines.append(r"\addlinespace")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Sachs observational-only benchmark: mean $\pm$ std across runs at different expert-query budgets.}")
    lines.append(r"\label{tab:sachs_budget}")
    lines.append(r"\end{table}")

    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _load_nodes(outdir):
    meta_fp = os.path.join(outdir, "sachs_meta.json")
    if os.path.exists(meta_fp):
        with open(meta_fp, "r") as f:
            meta = json.load(f)
        return meta.get("nodes", None)
    return None


def _avg_matrix(mats):
    return np.mean(np.stack(mats, axis=0), axis=0)


def save_heatmaps(
    runs_by_policy,
    outdir,
    *,
    policy="eig",
    which="directed",
    make_delta=True,
    font_size: int = 20,
    heatmap_fig_width: float = 6,
    heatmap_fig_height: float = 5,
    show_titles: bool = False,
    show_cbar_labels: bool = False
):
    """
    Creates before/after posterior marginal heatmaps averaged across runs.
    Assumes run json contains posterior_marginals_init and posterior_marginals_final.
    which: "directed" or "skeleton"
    """
    if policy not in runs_by_policy:
        print(f"[heatmaps] policy '{policy}' not found; skipping heatmaps.")
        return

    runs = runs_by_policy[policy]
    if not runs or "posterior_marginals_init" not in runs[0]:
        print("[heatmaps] posterior_marginals_init/final missing; did you patch the experiment script? Skipping.")
        return

    P0s, PTs = [], []
    for r in runs:
        P0 = np.asarray(r["posterior_marginals_init"], dtype=float)
        PT = np.asarray(r["posterior_marginals_final"], dtype=float)
        P0s.append(P0)
        PTs.append(PT)

    P0 = _avg_matrix(P0s)
    PT = _avg_matrix(PTs)

    if which == "skeleton":
        P0 = np.maximum(P0, P0.T)
        PT = np.maximum(PT, PT.T)

    nodes = _load_nodes(outdir)
    D = P0.shape[0]

    def _plot_mat(M, title, fname):
        plt.figure(figsize=(heatmap_fig_width, heatmap_fig_height))

        im = plt.imshow(
            M,
            vmin=0.0,
            vmax=1.0,
            cmap="Greys",
            interpolation="nearest",
        )
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        if show_cbar_labels:
            cbar.set_label("Posterior edge probability", rotation=90, fontsize=font_size)
        cbar.ax.tick_params(labelsize=font_size)

        if show_titles:
            plt.title(title, fontsize=font_size)

        if nodes is not None and len(nodes) == D:
            plt.xticks(range(D), nodes, rotation=45, ha="right", fontsize=font_size)
            plt.yticks(range(D), nodes, fontsize=font_size)
        else:
            plt.xticks(range(D), fontsize=font_size)
            plt.yticks(range(D), fontsize=font_size)

        plt.tight_layout(pad=0.15)
        plt.savefig(os.path.join(outdir, fname))
        plt.close()

    suffix = "skel" if which == "skeleton" else "dir"
    _plot_mat(P0, f"Sachs posterior marginals (t=0, {policy}, {which})", f"fig_sachs_heatmap_before_{suffix}.pdf")
    _plot_mat(PT, f"Sachs posterior marginals (t=T, {policy}, {which})", f"fig_sachs_heatmap_after_{suffix}.pdf")

    # Combined before/after in one PDF (two panels)
    fig = plt.figure(figsize=(2 * heatmap_fig_width, heatmap_fig_height))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    im1 = ax1.imshow(P0, vmin=0.0, vmax=1.0, cmap="Greys", interpolation="nearest")
    im2 = ax2.imshow(PT, vmin=0.0, vmax=1.0, cmap="Greys", interpolation="nearest")

    cbar = fig.colorbar(im2, ax=[ax1, ax2], fraction=0.046, pad=0.04)
    if show_cbar_labels:
        cbar.set_label("Posterior edge probability", rotation=90, fontsize=font_size)
    cbar.ax.tick_params(labelsize=font_size)

    if show_titles:
        fig.suptitle(f"Sachs posterior edge probabilities ({policy}, {which})", fontsize=font_size)
        ax1.set_title("t = 0", fontsize=font_size)
        ax2.set_title("t = T", fontsize=font_size)

    for ax in (ax1, ax2):
        if nodes is not None and len(nodes) == D:
            ax.set_xticks(range(D)); ax.set_yticks(range(D))
            ax.set_xticklabels(nodes, rotation=45, ha="right", fontsize=font_size)
            ax.set_yticklabels(nodes, fontsize=font_size)
        else:
            ax.set_xticks(range(D)); ax.set_yticks(range(D))
            ax.tick_params(labelsize=font_size)

    fig.tight_layout(pad=0.15)
    fig.savefig(os.path.join(outdir, f"fig_sachs_heatmap_before_after_{suffix}.pdf"))
    plt.close(fig)

    if make_delta:
        delta = PT - P0
        m = float(np.max(np.abs(delta)))

        plt.figure(figsize=(heatmap_fig_width, heatmap_fig_height))
        im = plt.imshow(delta, vmin=-m, vmax=m, cmap="coolwarm", interpolation="nearest")
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        if show_cbar_labels:
            cbar.set_label("Δ posterior edge probability", rotation=90, fontsize=font_size)
        cbar.ax.tick_params(labelsize=font_size)

        if show_titles:
            plt.title(f"Δ posterior marginals (t=T − t=0, {policy}, {which})", fontsize=font_size)

        if nodes is not None and len(nodes) == D:
            plt.xticks(range(D), nodes, rotation=45, ha="right", fontsize=font_size)
            plt.yticks(range(D), nodes, fontsize=font_size)
        else:
            plt.xticks(range(D), fontsize=font_size)
            plt.yticks(range(D), fontsize=font_size)

        plt.tight_layout(pad=0.15)
        plt.savefig(os.path.join(outdir, f"fig_sachs_heatmap_delta_{suffix}.pdf"))
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results_sachs", help="Directory containing sachs_*_seed*.json")

    ap.add_argument(
        "--policies", nargs="+",
        default=["eig", "uncertainty", "random"],
        help="List of policies to compare"
    )

    # Camera-ready styling (defaults chosen to match synthetic plotting script)
    ap.add_argument("--font_size", type=int, default=20)
    ap.add_argument("--line_width", type=float, default=3.5)
    ap.add_argument("--dpi", type=int, default=200, help="Only affects PNG outputs (if enabled later)")
    ap.add_argument("--fig_width", type=float, default=6.0)
    ap.add_argument("--fig_height", type=float, default=6.0)
    ap.add_argument("--legend", action="store_true", help="Show legend (default: on)")
    ap.set_defaults(legend=True)

    # Heatmaps
    ap.add_argument("--heatmap_policy", default="eig")
    ap.add_argument("--heatmap_which", default="directed", choices=["directed", "skeleton"])
    ap.add_argument("--heatmap_fig_width", type=float, default=6.0)
    ap.add_argument("--heatmap_fig_height", type=float, default=6.0)
    ap.add_argument("--titles", action="store_true", help="Enable titles/suptitles (default: off)")
    ap.add_argument("--no_delta", action="store_true")
    args = ap.parse_args()

    apply_mpl_style(args.font_size, args.line_width)

    runs = load_runs(args.outdir, policies=args.policies)
    assert runs, f"No run json files found in {args.outdir}"

    save_curves(
        runs,
        args.outdir,
        policies=args.policies,
        font_size=args.font_size,
        line_width=args.line_width,
        fig_width=args.fig_width,
        fig_height=args.fig_height,
        show_legend=args.legend,
    )

    save_budget_table(runs, args.outdir)
    csv_fp = os.path.join(args.outdir, "table_sachs_budget.csv")
    tex_fp = os.path.join(args.outdir, "table_sachs_budget.tex")
    write_latex_table_budget(csv_fp, tex_fp, budgets=(5, 10, 20, 40), policy_order=["eig", "uncertainty", "random"])
    print(" -", tex_fp)

    save_heatmaps(
        runs,
        args.outdir,
        policy=args.heatmap_policy,
        which=args.heatmap_which,
        make_delta=(not args.no_delta),
        font_size=args.font_size,
        heatmap_fig_width=args.heatmap_fig_width,
        heatmap_fig_height=args.heatmap_fig_height,
        show_titles=args.titles,
    )

    print("Wrote:")
    for fn in [
        "fig_sachs_shd.pdf",
        "fig_sachs_orient_f1.pdf",
        "fig_sachs_entropy.pdf",
        "table_sachs_budget.csv",
    ]:
        print(" -", os.path.join(args.outdir, fn))


if __name__ == "__main__":
    main()
