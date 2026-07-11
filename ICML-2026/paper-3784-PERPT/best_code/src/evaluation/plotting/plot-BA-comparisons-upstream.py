#!/usr/bin/env python3
import os
import argparse
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# -------------------------------------------------------------------
# config
# -------------------------------------------------------------------

TSV_DEFAULT = "/home/mica/gamba/data_processing/data/240-mammalian/global_balacc_combined/balacc_upstream_per_category.tsv"

# gamba / bigamba models (for labeling + facets)
GAMBA_MODELS = {
    # old naming
    # "gamba_seq_only_step_44000":  ("Gamba", "seq_only"),
    # "gamba_cons_only_step_44000": ("Gamba", "cons_only"),
    # "gamba_dual_step_44000":      ("Gamba", "dual"),
    # new naming (no underscore before step)
    "gamba_seq_only_step44000":   ("Gamba", "seq_only"),
    "gamba_cons_only_step44000":  ("Gamba", "cons_only"),
    "gamba_dual_step44000":       ("Gamba", "dual"),
}

BIGAMBA_MODELS = {
    # "caduceus_seq_only_step_44000":  ("Bi-Gamba", "seq_only"),
    # "caduceus_cons_only_step_44000": ("Bi-Gamba", "cons_only"),
    # "caduceus_dual_step_44000":      ("Bi-Gamba", "dual"),
    "caduceus_seq_only_step44000":   ("Bi-Gamba", "seq_only"),
    "caduceus_cons_only_step44000":  ("Bi-Gamba", "cons_only"),
    "caduceus_dual_step44000":       ("Bi-Gamba", "dual"),
}


# exact per-category sample sizes 
CATEGORY_N = {
    "vista_enhancer": 1978,
    "UCNE": 2000,
    "repeats": 2000,
    "exons": 2000,
    "introns": 1988,
    "noncoding_regions": 2000,
    "coding_regions": 2000,
    "upstream_TSS": 2000,
    "UTR5": 2000,
    "UTR3": 2000,
    "promoters": 2000,
}

# nice ordering for legends
CATEGORY_ORDER = [
    "vista_enhancer",
    "UCNE",
    "repeats",
    "exons",
    "introns",
    "noncoding_regions",
    "coding_regions",
    "upstream_TSS",
    "UTR5",
    "UTR3",
    "promoters",
]

# simple color map per category
CATEGORY_COLORS = {
    "vista_enhancer":   "#1f77b4",
    "UCNE":             "#ff7f0e",
    "repeats":          "#2ca02c",
    "exons":            "#d62728",
    "introns":          "#9467bd",
    "noncoding_regions":"#8c564b",
    "coding_regions":   "#e377c2",
    "upstream_TSS":     "#7f7f7f",
    "UTR5":             "#bcbd22",
    "UTR3":             "#17becf",
    "promoters":        "#000000",
}


# -------------------------------------------------------------------
# stats: per-category diff vs reference model
# -------------------------------------------------------------------

def per_category_diff_test(acc_model, acc_ref, n):
    """
    acc_model, acc_ref: proportions (0–1)
    n: number of samples per classifier for this category
    returns: delta, se, z, p, ci_low, ci_high (all in *proportion* units, not %)
    """
    p1, p2 = acc_model, acc_ref
    delta = p1 - p2

    se = math.sqrt(
        p1 * (1.0 - p1) / n +
        p2 * (1.0 - p2) / n
    )

    if se == 0:
        return delta, 0.0, float("inf"), 0.0, delta, delta

    z = delta / se
    p = 2.0 * (1.0 - norm.cdf(abs(z)))
    ci_low = delta - 1.96 * se
    ci_high = delta + 1.96 * se
    return delta, se, z, p, ci_low, ci_high


# -------------------------------------------------------------------
# data prep
# -------------------------------------------------------------------

def load_per_category_table(path_tsv: str) -> pd.DataFrame:
    df = pd.read_csv(path_tsv, sep="\t")
    required = {"Model", "Group", "Category", "Scope", "BA_pct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns in {path_tsv}: {missing}")

    # normalize types
    df["BA_pct"] = df["BA_pct"].astype(float)

    # optional sample sizes
    if "N_pos" in df.columns:
        df["N_pos"] = pd.to_numeric(df["N_pos"], errors="coerce")
    if "N_neg" in df.columns:
        df["N_neg"] = pd.to_numeric(df["N_neg"], errors="coerce")

    return df


def _row_n_from_counts(r: pd.Series) -> int | None:
    if "N_pos" in r and "N_neg" in r:
        npos = r["N_pos"]
        nneg = r["N_neg"]
        if pd.notna(npos) and pd.notna(nneg):
            n = int(npos) + int(nneg)
            if n > 0:
                return n
    return None


def _infer_family_objective(model_name: str):
    """
    map a model name to (family, objective) for labeling.
    general default: ('Other', model_name)
    gamba / bi-gamba use configured maps.
    """
    if model_name in GAMBA_MODELS:
        return GAMBA_MODELS[model_name]
    if model_name in BIGAMBA_MODELS:
        return BIGAMBA_MODELS[model_name]
    # everything else
    return "Other", model_name


def build_per_category_diffs(
    df: pd.DataFrame,
    ref_model: str,
    model_whitelist: list | None = None,
) -> pd.DataFrame:
    """
    for each (model, category) compute:
      - BA_model, BA_ref (in %)
      - delta, se, z, p, ci (in %)
    restricts to Scope == 'roi'
    ref_model: name in df['Model'] to use as x-axis
    model_whitelist: optional list of models to include (excluding ref_model)
    """
    df_roi = df[df["Scope"] == "roi"].copy()

    # reference baseline rows
    base = df_roi[df_roi["Model"] == ref_model].copy()
    if base.empty:
        raise ValueError(f"no rows for ref_model='{ref_model}' with Scope=='roi' found in TSV")

    # index baseline by (Group, Category)
    base_idx = base.set_index(["Group", "Category"])

    # which models to compare?
    all_models = sorted(m for m in df_roi["Model"].unique() if m != ref_model)
    if model_whitelist is not None and len(model_whitelist) > 0:
        target_models = [m for m in all_models if m in model_whitelist]
    else:
        target_models = all_models

    if not target_models:
        raise RuntimeError("no target models to compare (check --models and --ref-model).")

    rows = []
    bonf_factor = df_roi["Category"].nunique()

    alpha_bonf = 0.05 / bonf_factor

    for model_name in target_models:
        family, objective = _infer_family_objective(model_name)

        sub = df_roi[df_roi["Model"] == model_name].copy()
        if sub.empty:
            print(f"[warn] no rows for model {model_name} in TSV")
            continue

        # if there is a 'test' group, prefer that; otherwise use whatever exists
        if "test" in sub["Group"].unique():
            sub = sub[sub["Group"] == "test"].copy()

        for _, r in sub.iterrows():
            cat = r["Category"]
            group = r["Group"]
            ba_model = float(r["BA_pct"])

            # find matching reference row
            r_base = None
            # 1) same group, same category
            key1 = (group, cat)
            # 2) group 'all', same category
            key2 = ("all", cat)
            # 3) any group, same category (take first)
            if key1 in base_idx.index:
                r_base = base_idx.loc[key1]
            elif key2 in base_idx.index:
                r_base = base_idx.loc[key2]
            else:
                # try any baseline row for this category
                base_cat = base[base["Category"] == cat]
                if not base_cat.empty:
                    r_base = base_cat.iloc[0]
                else:
                    print(f"[warn] no baseline row for ref_model={ref_model}, category={cat}, skipping")
                    continue

            ba_ref = float(r_base["BA_pct"])

            # sample size for this category: prefer per-row N_pos/N_neg, else fallback
            n_cat = _row_n_from_counts(r)
            if n_cat is None:
                n_cat = CATEGORY_N.get(cat, 2000)

            acc_m = ba_model / 100.0
            acc_r = ba_ref / 100.0

            delta, se, z, p, ci_lo, ci_hi = per_category_diff_test(acc_m, acc_r, n_cat)

            rows.append(
                dict(
                    RefModel=ref_model,
                    Family=family,
                    Model=model_name,
                    Objective=objective,
                    Group=group,
                    Category=cat,
                    N=n_cat,
                    BA_model=ba_model,
                    BA_ref=ba_ref,
                    Delta_pct=delta * 100.0,
                    SE_pct=se * 100.0,
                    CI95_low_pct=ci_lo * 100.0,
                    CI95_high_pct=ci_hi * 100.0,
                    z=z,
                    p_two_sided=p,
                    Significant=(p < alpha_bonf),
                )
            )

    if not rows:
        raise RuntimeError("no per-category rows constructed; check config/TSV/ref/model list.")

    out = pd.DataFrame(rows)
    out["Category"] = pd.Categorical(out["Category"], CATEGORY_ORDER, ordered=True)
    return out


# -------------------------------------------------------------------
# plotting
# -------------------------------------------------------------------

def plot_family_facets(df: pd.DataFrame, family: str, outdir: str):
    """
    df: per-category diff table filtered to this family
    one row per (Model, Objective, Category)
    makes a 3-panel plot for cons_only / dual / seq_only (if present)
    """
    sub = df[df["Family"] == family].copy()
    if sub.empty:
        print(f"[warn] no rows for family {family}, skipping plot")
        return

    ref_model = sub["RefModel"].iloc[0]
    objectives = ["cons_only", "dual", "seq_only"]
    titles = {
        "cons_only": "cons_only",
        "dual": "dual",
        "seq_only": "seq_only",
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)
    fig.suptitle(f"{family}: per-category BA vs {ref_model} (roi)")

    for ax, obj in zip(axes, objectives):
        sub_obj = sub[sub["Objective"] == obj].copy()
        ax.set_title(titles[obj])

        if sub_obj.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            ax.set_xlabel(f"{ref_model} BA (%)")
            continue

        # identity line extents based on this objective
        xmin = min(sub_obj["BA_ref"].min(), sub_obj["BA_model"].min()) - 2
        xmax = max(sub_obj["BA_ref"].max(), sub_obj["BA_model"].max()) + 2
        ax.plot([xmin, xmax], [xmin, xmax], ls="--", color="0.7", label="y=x")

        # scatter points
        for _, r in sub_obj.iterrows():
            cat = r["Category"]
            x = r["BA_ref"]
            y = r["BA_model"]
            c = CATEGORY_COLORS.get(str(cat), "k")
            ax.scatter(x, y, s=40, color=c, edgecolor="black", linewidth=0.5, zorder=3)

            # label near point (category name)
            ax.annotate(
                str(cat),
                xy=(x, y),
                xytext=(3, 2),
                textcoords="offset points",
                fontsize=8,
            )

            # star for significance
            if bool(r["Significant"]):
                ax.text(
                    x,
                    y + 0.8,   # small vertical offset in BA %
                    "*",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    color="black",
                    zorder=4,
                )

        ax.set_xlabel(f"{ref_model} BA (%)")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Model BA (%)")

    # common limits across facets for this family
    all_x = sub["BA_ref"].to_numpy()
    all_y = sub["BA_model"].to_numpy()
    xmin = min(all_x.min(), all_y.min()) - 2
    xmax = max(all_x.max(), all_y.max()) + 2
    for ax in axes:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(xmin, xmax)

    # legend for categories
    handles = []
    labels = []
    for cat in CATEGORY_ORDER:
        if cat in sub["Category"].unique():
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="None",
                    markersize=6,
                    markerfacecolor=CATEGORY_COLORS.get(cat, "k"),
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                )
            )
            labels.append(cat)
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        title="Category",
        frameon=False,
        fontsize=8,
    )

    fig.tight_layout(rect=[0, 0, 0.85, 0.95])

    os.makedirs(outdir, exist_ok=True)
    tag = f"{family.lower()}_vs_{ref_model}"
    out_png = os.path.join(outdir, f"per_category_vs_{ref_model}_{tag}.png")
    out_svg = os.path.join(outdir, f"per_category_vs_{ref_model}_{tag}.svg")
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_svg)
    print(f"[info] saved {family} facet plot to:\n  {out_png}\n  {out_svg}")
    plt.close(fig)


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Per-category BA vs reference model (roi) with per-category significance and stars."
    )
    ap.add_argument(
        "--tsv",
        type=str,
        default=TSV_DEFAULT,
        help="binary_upstream_balacc_per_category.tsv path",
    )
    ap.add_argument(
        "--ref-model",
        type=str,
        default="evo2",
        help="model name in TSV to use as reference (x-axis), e.g. phylop, kmer6, nt-ms, etc.",
    )
    ap.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=None,
        help="optional list of model names (as in TSV 'Model' column) to include; "
             "if omitted, use all models except ref-model",
    )
    ap.add_argument(
        "-o",
        "--outdir",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/figures_upstream_binary",
        help="output directory for plots and tables",
    )
    args = ap.parse_args()

    df_percat = load_per_category_table(args.tsv)
    df_diff = build_per_category_diffs(df_percat, ref_model=args.ref_model, model_whitelist=args.models)

    # save table of per-category diffs
    os.makedirs(args.outdir, exist_ok=True)
    out_tsv = os.path.join(
        args.outdir,
        f"per_category_diffs_vs_{args.ref_model}.tsv",
    )
    df_diff.to_csv(out_tsv, sep="\t", index=False)
    print(f"[info] wrote per-category diffs to {out_tsv}")

    # facet plots only for Gamba / Bi-Gamba families (if present)
    for fam in ["Gamba", "Bi-Gamba"]:
        if fam in df_diff["Family"].unique():
            plot_family_facets(df_diff, fam, args.outdir)


if __name__ == "__main__":
    main()



#  python /home/mica/gamba/src/evaluation/plotting/plot-BA-comparisons-upstream.py\
#   --ref-model caduceus_seq_only_step44000 \
#   --models \
#     gamba_seq_only_step44000 \
#     gamba_cons_only_step44000 \
#     gamba_dual_step44000 \
#     caduceus_cons_only_step44000 \
#     caduceus_dual_step44000


#  python /home/mica/gamba/src/evaluation/plotting/plot-BA-comparisons-upstream.py\
#   --ref-model phylop \
#   --models \
#     gamba_seq_only_step44000 \
#     gamba_cons_only_step44000 \
#     gamba_dual_step44000 \
#     caduceus_seq_only_step44000 \
#     caduceus_cons_only_step44000 \
#     caduceus_dual_step44000


#  python /home/mica/gamba/src/evaluation/plotting/plot-BA-comparisons-upstream.py\
#   --ref-model caduceus-theirs \
#   --models \
#     gamba_seq_only_step44000 \
#     gamba_cons_only_step44000 \
#     gamba_dual_step44000 \
#     caduceus_seq_only_step44000 \
#     caduceus_cons_only_step44000 \
#     caduceus_dual_step44000

#  python /home/mica/gamba/src/evaluation/plotting/plot-BA-comparisons-upstream.py\
#   --ref-model evo2 \
#   --models \
#     gamba_seq_only_step44000 \
#     gamba_cons_only_step44000 \
#     gamba_dual_step44000 \
#     caduceus_seq_only_step44000 \
#     caduceus_cons_only_step44000 \
#     caduceus_dual_step44000
