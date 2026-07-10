import os
import ast
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import json
from collections import Counter

ALPHA          = 0.10

UNWEIGHTED_LABEL = "CP"
WEIGHTED_LABEL   = "DS-CP"

DISPLAY_NAMES = {
    "Yi-34B":"Yi-34B","Qwen-72B":"Qwen-72B","Qwen-14B":"Qwen-14B",
    "Llama-2-70b-hf":"Llama-2-70B","deepseek-llm-67b-base":"DeepSeek-67B",
    "Yi-6B":"Yi-6B","Mistral-7B-v0.1":"Mistral-7B","Llama-2-13b-hf":"Llama-2-13B",
    "Qwen-7B":"Qwen-7B","InternLM-7B":"InternLM-7B","Llama-2-7b-hf":"Llama-2-7B",
    "deepseek-llm-7b-base":"DeepSeek-7B","Qwen-1_8B":"Qwen-1.8B",
    "Falcon-40B":"Falcon-40B","MPT-7B":"MPT-7B","Falcon-7B":"Falcon-7B",
}

# Plotting (5 figures × 4×4 grid)
# Matplotlib styling
plt.rcParams.update({
    "font.size": 19, "axes.titlesize": 19, "axes.labelsize": 19,
    "xtick.labelsize": 17, "ytick.labelsize": 17, "legend.fontsize": 17
})
TITLE_FS, TITLE_WEIGHT = 22, "bold"
LABEL_FS, TICK_FS, TICK_WEIGHT, LEGEND_FS = 22, 17, "bold", 17

def _style_ax(ax, title=None, xlabel=None, ylabel=None, show_legend=False, legend_loc="upper left"):
    if title:  ax.set_title(title, fontsize=TITLE_FS, fontweight=TITLE_WEIGHT)
    if xlabel: ax.set_xlabel(xlabel, fontsize=LABEL_FS, fontweight="bold")
    if ylabel: ax.set_ylabel(ylabel, fontsize=LABEL_FS, fontweight="bold")
    ax.tick_params(axis="both", labelsize=TICK_FS)
    for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        t.set_fontweight(TICK_WEIGHT); t.set_fontsize(TICK_FS)
    if show_legend:
        leg = ax.legend(loc=legend_loc)
        for txt in leg.get_texts():
            txt.set_fontweight("bold"); txt.set_fontsize(LEGEND_FS)

def _coerce_cell(x):
    if isinstance(x, (int, float, np.floating)): return float(x)
    try:
        d = ast.literal_eval(str(x))
        if isinstance(d, dict) and len(d): return float(next(iter(d.values())))
        return float(d)
    except Exception:
        try:    return float(x)
        except Exception: return np.nan

def _grid_fixed(): return 4, 4

CELL_W, CELL_H = 4, 4
FIG_SCALE = float(os.getenv("FIG_SCALE", ".8"))
def _figsize(r, c, scale=0.5):
    s = FIG_SCALE * scale
    return (CELL_W * c * s, CELL_H * r * s)

def _grid_span(fig, axes):
    axes = np.atleast_1d(axes).ravel()
    pos = [ax.get_position().frozen() for ax in axes if ax.get_visible()]
    if not pos: return 0.1, 0.9, (0.1, 0.9)
    lefts=[p.x0 for p in pos]; rights=[p.x1 for p in pos]
    bottoms=[p.y0 for p in pos]; tops=[p.y1 for p in pos]
    return min(lefts), max(rights), (min(bottoms), max(tops))

def _legend_footer(fig, axes, series_handles, series_labels, axis_desc_texts,
                   bottom_pad=0.12, yoff=0.095):
    fig.tight_layout(rect=[0.03, bottom_pad, 0.97, 0.98])
    left, right, _ = _grid_span(fig, axes); center_x = (left + right) / 2.0
    desc_handles = [Line2D([], [], linestyle='None', label=t) for t in axis_desc_texts]
    handles = (series_handles or []) + desc_handles
    labels  = (series_labels  or []) + axis_desc_texts
    leg = fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(center_x, yoff),
                     ncol=max(1, len(labels)), frameon=False,
                     prop={"weight":"bold","size":LEGEND_FS},
                     handlelength=1.8, handletextpad=0.6, columnspacing=1.2, borderaxespad=0.0)
    for txt in leg.get_texts(): txt.set_fontweight("bold"); txt.set_fontsize(LEGEND_FS)

# Violin styling
VIOLIN_EDGE_LW   = 0.5
VIOLIN_MEDIAN_LW = 1.9
VIOLIN_EXTREMA_LW= 1.9
ZERO_LINE_LW     = 2.0
VIOLIN_SEP       = 0.60
VIOLIN_W_SCALE   = 0.90
CP_COLOR, DS_COLOR, EDGE_COLOR = "#4DAAED", "#F98F39", "#000000"

def _styled_violin(ax, a, b, show_target_zero=False):
    pos_cp = 1.0 - VIOLIN_SEP/2.0
    pos_ds = 1.0 + VIOLIN_SEP/2.0
    width  = VIOLIN_SEP * VIOLIN_W_SCALE
    a = np.asarray(a, float); b = np.asarray(b, float)
    vp = ax.violinplot([a, b], positions=[pos_cp, pos_ds], widths=width,
                       showmeans=False, showmedians=True, showextrema=True)
    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor(CP_COLOR if i == 0 else DS_COLOR)
        body.set_alpha(0.65); body.set_edgecolor(EDGE_COLOR); body.set_linewidth(VIOLIN_EDGE_LW)
    if "cmedians" in vp: vp["cmedians"].set_color('black'); vp["cmedians"].set_linewidth(VIOLIN_EXTREMA_LW)
    if "cmins"    in vp: vp["cmins"].set_color('black');    vp["cmins"].set_linewidth(VIOLIN_EXTREMA_LW)
    if "cmaxes"   in vp: vp["cmaxes"].set_color('black');   vp["cmaxes"].set_linewidth(VIOLIN_EXTREMA_LW)
    if "cbars"    in vp: vp["cbars"].set_color('black');    vp["cbars"].set_linewidth(VIOLIN_EXTREMA_LW)
    if show_target_zero:
        ax.axhline(0, linestyle="--", linewidth=ZERO_LINE_LW, color="#D50000", alpha=0.9, zorder=0)
    ax.set_xticks([pos_cp, pos_ds])

def _get_items(csv_paths, model_names, alpha, pair_prefix):
    """
    Load per-model CSVs and extract the relevant coverage/set-size columns
    for plotting. Skips any missing, empty, or all-NaN files gracefully.
    """
    cov_A_col   = f"coverage_{pair_prefix}"
    cov_B_col   = f"coverage_{pair_prefix}_W"
    size_A_col  = f"setsize_{pair_prefix}"
    size_B_col  = f"setsize_{pair_prefix}_W"
    items = []

    for p, internal_name in zip(csv_paths, model_names):
        if not os.path.exists(p):
            print(f"[skip] missing CSV: {p}")
            continue

        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[skip] failed to read CSV ({p}): {e}")
            continue

        if df.empty:
            print(f"[skip] empty CSV: {p}")
            continue

        # Coerce values to floats
        for col in [cov_A_col, cov_B_col, size_A_col, size_B_col]:
            if col in df.columns:
                df[col] = df[col].apply(_coerce_cell)

        # Drop rows where all key columns are NaN
        df = df.dropna(subset=[cov_A_col, cov_B_col, size_A_col, size_B_col], how="all")
        if df.empty:
            print(f"[skip] all-NaN CSV: {p}")
            continue

        # Compute coverage deltas (difference from target 1-alpha)
        if cov_A_col in df.columns:
            df[f"cov_diff_{pair_prefix}"] = df[cov_A_col] - (1 - alpha)
        else:
            df[f"cov_diff_{pair_prefix}"] = np.nan

        if cov_B_col in df.columns:
            df[f"cov_diff_{pair_prefix}_W"] = df[cov_B_col] - (1 - alpha)
        else:
            df[f"cov_diff_{pair_prefix}_W"] = np.nan

        items.append((internal_name, df))

    if not items:
        print(f"[warn] no valid data for {pair_prefix} — skipping plots.")
    else:
        print(f"[ok] loaded {len(items)} CSV(s) for {pair_prefix} plotting.")

    return items


def _title_for(internal_name: str) -> str:
    return DISPLAY_NAMES.get(internal_name, internal_name)

def _five_plots_for_pair(csv_paths, model_names, alpha, outdir, pair_prefix,
                         label_A=UNWEIGHTED_LABEL, label_B=WEIGHTED_LABEL):
    os.makedirs(outdir, exist_ok=True)
    items = _get_items(csv_paths, model_names, alpha, pair_prefix)
    if not items:
        print(f"[plot:{pair_prefix}] no valid CSVs — skipping plots."); return

    cov_A   = f"cov_diff_{pair_prefix}"
    cov_B   = f"cov_diff_{pair_prefix}_W"
    size_A  = f"setsize_{pair_prefix}"
    size_B  = f"setsize_{pair_prefix}_W"
    r, c = _grid_fixed()

    # 1) Coverage violin (shared Y; x labels only on last row)
    fig, axes = plt.subplots(r, c, figsize=_figsize(r, c, scale=1.00), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, df) in zip(axes, items):
        a = df[cov_A].dropna(); b = df[cov_B].dropna()
        _styled_violin(ax, a, b, show_target_zero=True)
        ax.set_xticklabels([label_A, label_B], fontsize=TICK_FS)
        for t in ax.get_xticklabels(): t.set_fontweight("bold"); t.set_fontsize(LEGEND_FS)
        _style_ax(ax, title=_title_for(name))
    for ax in axes[len(items):]: ax.axis("off")
    for i, ax in enumerate(axes): ax.tick_params(labelbottom=(i // c == r - 1))
    fig.supylabel(f"Coverage − {1-alpha:.2f}", fontsize=LABEL_FS, fontweight="bold", x=0.005)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{pair_prefix.lower()}_coverage_delta_violin.pdf"), dpi=300)
    plt.show()

    # 2) Set size violin
    fig, axes = plt.subplots(r, c, figsize=_figsize(r, c, scale=1.00), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, df) in zip(axes, items):
        a = df[size_A].dropna() if size_A in df.columns else pd.Series(dtype=float)
        b = df[size_B].dropna() if size_B in df.columns else pd.Series(dtype=float)
        _styled_violin(ax, a, b, show_target_zero=False)
        ax.set_xticklabels([label_A, label_B], fontsize=TICK_FS)
        for t in ax.get_xticklabels(): t.set_fontweight("bold"); t.set_fontsize(LEGEND_FS)
        _style_ax(ax, title=_title_for(name))
    for ax in axes[len(items):]: ax.axis("off")
    for i, ax in enumerate(axes): ax.tick_params(labelbottom=(i // c == r - 1))
    fig.supylabel("Set Size", fontsize=LABEL_FS, fontweight="bold", x=0.005)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{pair_prefix.lower()}_setsize_violin.pdf"), dpi=300)
    plt.show()

    # 3) Coverage Δ histogram (shared bins across panels)
    fig, axes = plt.subplots(r, c, figsize=_figsize(r, c, scale=0.95), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    all_vals = []
    for _, df in items:
        all_vals += df[cov_A].dropna().tolist(); all_vals += df[cov_B].dropna().tolist()
    x_min, x_max = (min(all_vals), max(all_vals)) if all_vals else (-0.1, 0.1)

    first_handles, first_labels = None, None
    for ax, (name, df) in zip(axes, items):
        h1 = ax.hist(df[cov_A].dropna(), bins=20, alpha=0.6, label=label_A)
        h2 = ax.hist(df[cov_B].dropna(), bins=20, alpha=0.6, label=label_B)
        ax.set_xlim(x_min, x_max)
        _style_ax(ax, title=_title_for(name))
        if first_handles is None:
            handles, labels = [], []
            if h1[-1]: handles.append(h1[-1][0]); labels.append(label_A)
            if h2[-1]: handles.append(h2[-1][0]); labels.append(label_B)
            first_handles, first_labels = handles, labels
    for ax in axes[len(items):]: ax.axis("off")
    for i, ax in enumerate(axes): ax.tick_params(labelbottom=(i // c == r - 1))
    _legend_footer(fig, axes, (first_handles or []), (first_labels or []),
                   [f"X-axis: Coverage − {1-alpha:.2f}", "Y-axis: Count"])
    fig.savefig(os.path.join(outdir, f"{pair_prefix.lower()}_coverage_delta_hist.pdf"), dpi=300, bbox_inches="tight")
    plt.show()

    # 4) Set size histogram
    fig, axes = plt.subplots(r, c, figsize=_figsize(r, c, scale=0.95), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    all_sizes = []
    for _, df in items:
        if size_A in df.columns: all_sizes += df[size_A].dropna().tolist()
        if size_B in df.columns: all_sizes += df[size_B].dropna().tolist()
    xs_min, xs_max = (min(all_sizes), max(all_sizes)) if all_sizes else (1, 6)

    first_handles, first_labels = None, None
    for ax, (name, df) in zip(axes, items):
        hlist = []
        if size_A in df.columns: hlist.append(ax.hist(df[size_A].dropna(), bins=20, alpha=0.6, label=label_A))
        if size_B in df.columns: hlist.append(ax.hist(df[size_B].dropna(), bins=20, alpha=0.6, label=label_B))
        ax.set_xlim(xs_min, xs_max)
        _style_ax(ax, title=_title_for(name))
        if first_handles is None and hlist:
            handles, labels = [], []
            for h, lab in zip(hlist, [label_A, label_B][:len(hlist)]):
                if h[-1]: handles.append(h[-1][0]); labels.append(lab)
            first_handles, first_labels = handles, labels
    for ax in axes[len(items):]: ax.axis("off")
    for i, ax in enumerate(axes): ax.tick_params(labelbottom=(i // c == r - 1))
    _legend_footer(fig, axes, (first_handles or []), (first_labels or []),
                   ["X-axis: Set Size", "Y-axis: Count"])
    fig.savefig(os.path.join(outdir, f"{pair_prefix.lower()}_setsize_hist.pdf"), dpi=300, bbox_inches="tight")
    plt.show()

    # 5) Paired coverage Δ scatter
    fig, axes = plt.subplots(r, c, figsize=_figsize(r, c, scale=1.05), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    xy = []
    for _, df in items:
        xy += df[cov_A].dropna().tolist(); xy += df[cov_B].dropna().tolist()
    lim_min = min(min(xy), 0) if xy else -0.1
    lim_max = max(max(xy), 0) if xy else 0.1

    first_handles, first_labels = None, None
    for ax, (name, df) in zip(axes, items):
        x = df[cov_A].dropna()
        y = df.loc[x.index, cov_B].dropna()
        common_idx = x.index.intersection(y.index)
        x, y = x.loc[common_idx], y.loc[common_idx]
        scat_all = ax.scatter(x, y, alpha=0.3, label="CP > Target")
        mask = x < 0
        scat_lt  = ax.scatter(x[mask], y[mask], alpha=0.7, label=f"{UNWEIGHTED_LABEL} < target")
        ax.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--", linewidth=1.6)
        ax.set_xlim(lim_min, lim_max); ax.set_ylim(lim_min, lim_max)
        _style_ax(ax, title=_title_for(name))
        if first_handles is None:
            first_handles = [scat_all, scat_lt]; first_labels = ["CP > Target", f"{UNWEIGHTED_LABEL} < target"]
    for ax in axes[len(items):]: ax.axis("off")
    _legend_footer(fig, axes, (first_handles or []), (first_labels or []),
                   [f"X-axis: Cov − {1-ALPHA:.2f} ({UNWEIGHTED_LABEL})",
                    f"Y-axis: Cov − {1-ALPHA:.2f} ({WEIGHTED_LABEL})"])
    fig.savefig(os.path.join(outdir, f"{pair_prefix.lower()}_paired_coverage_scatter.pdf"), dpi=300, bbox_inches="tight")
    plt.show()

def plot_cp_comparisons(csv_paths, model_names, alpha=0.1, outdir="figs",
                        label_unweighted=UNWEIGHTED_LABEL, label_weighted=WEIGHTED_LABEL):
    _five_plots_for_pair(csv_paths, model_names, alpha, outdir, pair_prefix="LAC",
                         label_A=label_unweighted, label_B=label_weighted)
    _five_plots_for_pair(csv_paths, model_names, alpha, outdir, pair_prefix="APS",
                         label_A=label_unweighted, label_B=label_weighted)

def _infer_models_from_csvs(csv_paths):
    names = []
    for p in csv_paths:
        base = os.path.basename(p)
        if base.startswith("coverage_") and base.endswith(".csv"):
            names.append(base[len("coverage_"):-len(".csv")])
    return names

def rerun_plots_only(results_dir="results-mmlu", figs_dir=None, alpha=0.10,
                     label_unweighted=UNWEIGHTED_LABEL, label_weighted=WEIGHTED_LABEL):
    figs_dir = figs_dir or os.path.join(results_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)
    csv_paths = sorted(glob.glob(os.path.join(results_dir, "coverage_*.csv")))
    if not csv_paths:
        print(f"[plots-only] no CSVs found in {results_dir}"); return
    model_names = _infer_models_from_csvs(csv_paths)
    print(f"[plots-only] plotting {len(csv_paths)} CSVs with labels: {label_unweighted} vs {label_weighted}")
    plot_cp_comparisons(csv_paths, model_names, alpha=alpha, outdir=figs_dir,
                        label_unweighted=label_unweighted, label_weighted=label_weighted)

def plot_mmlu_subject_distribution(
    file_path="../CP/data/mmlu_10k.json",
    key="subcategory",
    out_path="./results-mmlu/figs/mmlu_subject_distribution.pdf",
    color="#4DAAED",
    figsize=(12, 6),
    rotate_xticks=45,
    top_n=None,
    show=True,
    save=True,
):
    """
    Load an MMLU-style JSON, count unique values of `key`, print a summary,
    and plot a bar chart.
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    counts = Counter(item.get(key, "UNKNOWN") for item in data)
    items_sorted = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    if top_n is not None and top_n > 0:
        items_sorted = items_sorted[:top_n]

    labels = [k for k, _ in items_sorted]
    values = [v for _, v in items_sorted]
    total = sum(values)

    print(f"Number of unique {key}s:", len(counts))
    for subcat, count in items_sorted:
        pct = 100.0 * count / total if total else 0.0
        print(f"{subcat:20s} {count:6d}  ({pct:5.1f}%)")

    plt.figure(figsize=figsize)
    bars = plt.bar(labels, values, color=color)

    # Value labels
    for bar, val in zip(bars, values):
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            h,
            f"{val}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax = plt.gca()
    for t in ax.get_xticklabels():
        t.set_fontweight("bold")
        t.set_fontsize(16)
    for t in ax.get_yticklabels():
        t.set_fontweight("bold")
        t.set_fontsize(16)

    plt.xticks(rotation=rotate_xticks, ha="right")
    plt.ylabel("Number of Items", fontsize=19, fontweight="bold")
    plt.xlabel("Subject", fontsize=19, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    if save and out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()

    return {
        "num_unique": len(counts),
        "total": total,
        "items_sorted": items_sorted,
    }
