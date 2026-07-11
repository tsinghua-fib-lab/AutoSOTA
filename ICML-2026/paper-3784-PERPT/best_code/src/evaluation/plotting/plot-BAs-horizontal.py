#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---------------------------------------------------------------------
# MODEL META
# ---------------------------------------------------------------------
MODEL_META = {
    "gamba_seq_only_step44000": dict(
        label="ArGamba NTP-only", family="Gamba", kind="seq_only",
        params=66_492_392, context=2048, random_init=False),
    "gamba_seq_only_step0": dict(
        label="ArGamba NTP-only Random-Init", family="Gamba", kind="seq_only",
        params=66_492_392, context=2048, random_init=True),

    "gamba_cons_only_step44000": dict(
        label="ArGamba CEP-only", family="Gamba", kind="phy_only",
        params=66_492_392, context=2048, random_init=False),
    "gamba_cons_only_step0": dict(
        label="ArGamba CEP-only Random-Init", family="Gamba", kind="phy_only",
        params=66_492_392, context=2048, random_init=True),

    "gamba_dual_step44000": dict(
        label="ArGamba NTP+CEP", family="Gamba", kind="seq_plus_phy",
        params=66_493_418, context=2048, random_init=False),
    "gamba_dual_step0": dict(
        label="ArGamba NTP+CEP Random-Init", family="Gamba", kind="seq_plus_phy",
        params=66_493_418, context=2048, random_init=True),

    # bi-gamba
    "caduceus_seq_only_step44000": dict(
        label="Bi-Gamba MLM-only", family="Bi-Gamba", kind="seq_only",
        params=3_864_832, context=2048, random_init=False),
    "caduceus_seq_only_step0": dict(
        label="Bi-Gamba MLM-only Random-Init", family="Bi-Gamba", kind="seq_only",
        params=3_864_832, context=2048, random_init=True),

    "caduceus_cons_only_step44000": dict(
        label="Bi-Gamba MEM-only", family="Bi-Gamba", kind="phy_only",
        params=3_864_832, context=2048, random_init=False),
    "caduceus_cons_only_step0": dict(
        label="Bi-Gamba MEM-only Random-Init", family="Bi-Gamba", kind="phy_only",
        params=3_864_832, context=2048, random_init=True),

    "caduceus_dual_step44000": dict(
        label="Bi-Gamba MLM+MEM", family="Bi-Gamba", kind="seq_plus_phy",
        params=3_869_442, context=2048, random_init=False),
    "caduceus_dual_step0": dict(
        label="Bi-Gamba MLM+MEM Random-Init", family="Bi-Gamba", kind="seq_plus_phy",
        params=3_869_442, context=2048, random_init=True),

    # NT / HyenaDNA / PhyloGPN / others
    "nt-ms": dict(
        label="NT multi-species", family="Other", kind="seq_only",
        params=498_345_436, context=1000, random_init=False),
    "nt-ms-random-init": dict(
        label="NT multi-species Random-Init", family="Other", kind="seq_only",
        params=498_345_436, context=1000, random_init=True),

    "nt-human": dict(
        label="NT human-ref", family="Other", kind="seq_only",
        params=480_438_241, context=1000, random_init=False),
    "nt-human-random-init": dict(
        label="NT human-ref Random-Init", family="Other", kind="seq_only",
        params=480_438_241, context=1000, random_init=True),

    "phyloGPN": dict(
        label="PhyloGPN", family="Other", kind="seq_only",
        params=83_185_924, context=481, random_init=False),
    "phyloGPN-random-init": dict(
        label="PhyloGPN Random-Init", family="Other", kind="seq_only",
        params=83_185_924, context=481, random_init=True),

    # "hyenaDNA": dict(
    #     label="HyenaDNA", family="Other", kind="seq_only",
    #     params=6_551_040, context=160_000, random_init=False),
    # "hyenaDNA-random-init": dict(
    #     label="HyenaDNA Random-Init", family="Other", kind="seq_only",
    #     params=6_551_040, context=160_000, random_init=True),

    "caduceus": dict(
        label="Caduceus", family="Other", kind="seq_only",
        params=7_725_312, context=131_000, random_init=False),
    "caduceus-random-init": dict(
        label="Caduceus Random-Init", family="Other", kind="seq_only",
        params=7_725_312, context=131_000, random_init=True),

    "caduceus-theirs": dict(
        label="Caduceus", family="Other", kind="seq_only",
        params=7_725_312, context=131_000, random_init=False),
    "caduceus-theirs-random-init": dict(
        label="Caduceus Random-Init", family="Other", kind="seq_only",
        params=7_725_312, context=131_000, random_init=True),

    "evo2": dict(
        label="Evo2", family="Other", kind="seq_only",
        params=7_000_000_000, context=2048, random_init=False),

    # baselines (now bars too)
    "kmer6": dict(
        label="K-mer (k=6)", family="Other", kind="baseline_kmer",
        params=0, context=2048, random_init=False),
    "phylop": dict(
        label="PhyloP (6D)", family="Other", kind="baseline_phylop",
        params=0, context=2048, random_init=False),
}



# colors
BLUE   = "#4287f5"   # seq+phy
PURPLE = "#6F2DA8"   # phy only
ORANGE = "#FF8C32"   # seq only
DARK   = "#6A6A6A"   # baselines

def _hex_to_rgb01(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def _rgb01_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(
        int(round(rgb[0] * 255)),
        int(round(rgb[1] * 255)),
        int(round(rgb[2] * 255)),
    )

def lighten_hex(hex_color: str, amount: float = 0.55) -> str:
    """
    Mix with white. amount=0 -> original, amount=1 -> white.
    """
    r, g, b = _hex_to_rgb01(hex_color)
    r = r + (1.0 - r) * amount
    g = g + (1.0 - g) * amount
    b = b + (1.0 - b) * amount
    return _rgb01_to_hex((r, g, b))

def base_color_for(kind: str):
    if kind.startswith("baseline"):
        return DARK
    if kind == "seq_plus_phy":
        return BLUE
    if kind == "phy_only":
        return PURPLE
    if kind == "seq_only":
        return ORANGE
    return "#B0B0B0"

# ---------------------------------------------------------------------
# eval-type config
# ---------------------------------------------------------------------
EVAL_CFG = {
    "upstream": dict(
        scope="roi",
        y_label="global ROI 1-vs-upstream balanced accuracy (%)",
        title="global ROI 1-vs-upstream balanced accuracy",
        filename="plot_global_balacc_upstream_barh.svg",
    ),
    "random": dict(
        scope="roi",
        y_label="global ROI feature-vs-random balanced accuracy (%)",
        title="global ROI feature-vs-random balanced accuracy",
        filename="plot_global_balacc_random_barh.svg",
    ),
    "multiclass": dict(
        scope="roi",
        y_label="global multiclass balanced accuracy (%)",
        title="global multiclass balanced accuracy",
        filename="plot_global_balacc_multiclass_barh.svg",
    ),
    "random_noannot": dict(
        scope="roi",
        y_label="global ROI feature-vs-random (no annotation) balanced accuracy (%)",
        title="global ROI feature-vs-random (no annotation) balanced accuracy",
        filename="plot_global_balacc_random_noannot_barh.svg",
    ),
    "multiclass100bproi": dict(
        scope="roi100bp",
        y_label="global multiclass (100bp sampled from ROI) balanced accuracy (%)",
        title="global multiclass (100bp sampled from ROI) balanced accuracy",
        filename="plot_global_balacc_multiclass100bproi_barh.svg",
    ),
}

def pick_roi_row(sub: pd.DataFrame, scope: str) -> pd.DataFrame:
    if "test" in sub["Group"].unique():
        roi = sub[(sub["Group"] == "test") & (sub["Scope"] == scope)]
    else:
        roi = sub[(sub["Group"] == "all") & (sub["Scope"] == scope)]
    return roi

def _base_label(label: str) -> str:
    if label.endswith(" Random-Init"):
        return label[: -len(" Random-Init")]
    return label

def load_table(tsv_path: str, eval_type: str) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t")
    cfg = EVAL_CFG[eval_type]
    scope = cfg["scope"]

    # collapse trained + random-init onto one row (overlay later)
    rows_by_label = {}

    for model_folder, meta in MODEL_META.items():
        sub = df[df["Model"] == model_folder]
        if sub.empty:
            continue

        roi = pick_roi_row(sub, scope=scope)
        if len(roi) != 1:
            print(f"[warn] {eval_type}: {model_folder}: expected 1 row for scope={scope}, got {len(roi)}")
            continue

        r = roi.iloc[0]
        base_label = _base_label(meta["label"])
        entry = rows_by_label.get(base_label, dict(
            label=base_label,
            kind=meta["kind"],
            trained_BA=np.nan, trained_SE=np.nan,
            rand_BA=np.nan, rand_SE=np.nan,
        ))

        if meta["random_init"]:
            entry["rand_BA"] = float(r["GlobalBalancedAccuracyPct"])
            entry["rand_SE"] = float(r["GlobalBalancedAccuracySEPct"])
        else:
            entry["trained_BA"] = float(r["GlobalBalancedAccuracyPct"])
            entry["trained_SE"] = float(r["GlobalBalancedAccuracySEPct"])

        rows_by_label[base_label] = entry

    tbl = pd.DataFrame(list(rows_by_label.values()))
    if tbl.empty:
        raise SystemExit(f"no models loaded for eval_type={eval_type} (check tsv and MODEL_META keys).")

    # sort by trained if available else random-init
    sort_key = tbl["trained_BA"].copy()
    sort_key = sort_key.fillna(tbl["rand_BA"])
    tbl = tbl.assign(_sort=sort_key).sort_values("_sort", ascending=True).drop(columns=["_sort"]).reset_index(drop=True)
    return tbl

def plot_barh(tbl: pd.DataFrame, eval_type: str, out_path: str):
    cfg = EVAL_CFG[eval_type]
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    })


    n = len(tbl)
    fig_h = max(4.0, 0.35 * n)
    fig, ax = plt.subplots(figsize=(7.5, fig_h))

    y = np.arange(n)

    # trained bars (full color)
    trained_vals = tbl["trained_BA"].values
    trained_err  = tbl["trained_SE"].values
    trained_mask = ~np.isnan(trained_vals)

    trained_colors = [base_color_for(k) for k in tbl["kind"]]

    ax.barh(
        y[trained_mask],
        trained_vals[trained_mask],
        xerr=trained_err[trained_mask],
        color=np.array(trained_colors, dtype=object)[trained_mask],
        height=0.78,
        edgecolor="none",
        linewidth=0,
        error_kw=dict(ecolor="black", lw=1.0, capsize=2),
        zorder=2,
    )

    # random-init overlay (lighter shade of the same base color)
    rand_vals = tbl["rand_BA"].values
    rand_err  = tbl["rand_SE"].values
    rand_mask = ~np.isnan(rand_vals)

    rand_colors = [lighten_hex(base_color_for(k), amount=0.60) for k in tbl["kind"]]

    ax.barh(
        y[rand_mask],
        rand_vals[rand_mask],
        xerr=rand_err[rand_mask],
        color=np.array(rand_colors, dtype=object)[rand_mask],
        height=0.46,  # slightly smaller so you can see the trained bar under it
        edgecolor="none",
        linewidth=0,
        error_kw=dict(ecolor="black", lw=1.0, capsize=2),
        zorder=3,
    )

    # x-lims based on whichever exists
    all_vals = np.nan_to_num(np.vstack([trained_vals, rand_vals]), nan=np.nan)
    flat = all_vals[~np.isnan(all_vals)]
    xmin = max(0, float(flat.min()) - 2) if flat.size else 0
    xmax = (float(flat.max()) + 1) if flat.size else 1
    ax.set_xlim(xmin, xmax)

    ax.set_yticks(y)
    ax.set_yticklabels(tbl["label"].tolist(), fontsize=9)

    ax.set_xlabel(cfg["y_label"])
    ax.set_title(cfg["title"])

    legend_items = [
        Patch(facecolor=ORANGE, edgecolor="none", label="MLM-only / NTP-only"),
        Patch(facecolor=PURPLE, edgecolor="none", label="MEM-only / CEP-only"),
        Patch(facecolor=BLUE,   edgecolor="none", label="MLM+MEM / NTP+CEP"),
        Patch(facecolor=lighten_hex(BLUE, 0.60), edgecolor="none", label="Random-Init (lighter shade)"),
        Patch(facecolor=DARK, edgecolor="none", label="Baseline"),
    ]
    # ax.legend(handles=legend_items, loc="lower right", frameon=True, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print("saved to:", out_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_type",
        choices=list(EVAL_CFG.keys()) + ["all"],
        default="upstream",
        help="which plot to generate (or 'all')",
    )

    parser.add_argument("--tsv", default=None, help="input TSV (used when eval_type != all)")

    parser.add_argument(
        "--tsv_upstream",
        default="/home/mica/gamba/data_processing/data/240-mammalian/global_balacc_combined/balacc_upstream_global.tsv",
    )
    parser.add_argument(
        "--tsv_random",
        default="/home/mica/gamba/data_processing/data/240-mammalian/global_balacc_combined/balacc_random_global.tsv",
    )
    parser.add_argument(
        "--tsv_multiclass",
        default="/home/mica/gamba/data_processing/data/240-mammalian/global_balacc_combined/balacc_multiclass_global.tsv",
    )
    parser.add_argument(
        "--tsv_random_noannot",
        default="/home/mica/gamba/data_processing/data/240-mammalian/global_balacc_combined/balacc_random_noannot_global.tsv",
    )
    parser.add_argument(
        "--tsv_multiclass100bproi",
        default="/home/mica/gamba/data_processing/data/240-mammalian/global_balacc_combined/balacc_multiclass100bproi_global.tsv",
    )

    parser.add_argument("-o", "--outdir", default=".", help="directory to save plots")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.eval_type != "all":
        if args.tsv is None:
            raise SystemExit("--tsv is required unless --eval_type all")
        tbl = load_table(args.tsv, args.eval_type)
        out_path = os.path.join(args.outdir, EVAL_CFG[args.eval_type]["filename"])
        plot_barh(tbl, args.eval_type, out_path)
        return

    tsv_map = {
        "upstream": args.tsv_upstream,
        "random": args.tsv_random,
        "multiclass": args.tsv_multiclass,
        "random_noannot": args.tsv_random_noannot,
        "multiclass100bproi": args.tsv_multiclass100bproi,
    }

    for et, path in tsv_map.items():
        if not os.path.exists(path):
            print(f"[warn] missing tsv for {et}: {path} (skipping)")
            continue
        tbl = load_table(path, et)
        out_path = os.path.join(args.outdir, EVAL_CFG[et]["filename"])
        plot_barh(tbl, et, out_path)

if __name__ == "__main__":
    main()

# # all plots (recommended)
# python src/evaluation/plotting/plot-BAs-horizontal.py --eval_type all -o /home/mica/gamba/data_processing/data/240-mammalian/global_balacc_figs

# # single plot (still supported)
# python src/evaluation/plotting/plot-BAs-horizontal.py \
#   --eval_type upstream \
#   --tsv /home/mica/gamba/data_processing/data/240-mammalian/global_balacc_combined/balacc_upstream_global.tsv \
#   -o /home/mica/gamba/data_processing/data/240-mammalian/global_balacc_figs
