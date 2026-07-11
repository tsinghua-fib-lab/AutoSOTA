# #!/usr/bin/env python3
# import os
# import argparse
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D

# # ---------------------------------------------------------------------
# # MODEL META (same as before)
# # ---------------------------------------------------------------------
# MODEL_META = {
#     "gamba_seq_only_ALLPOSstep_44000": dict(
#         label="Gamba NTP-only", family="Gamba", kind="seq_only",
#         params=66_492_392, context=2048, random_init=False),
#     "gamba_seq_only_step_random_init": dict(
#         label="Gamba NTP-only Random-Init", family="Gamba", kind="seq_only",
#         params=66_492_392, context=2048, random_init=True),

#     "gamba_cons_only_ALLPOSstep_44000": dict(
#         label="Gamba CEP-only", family="Gamba", kind="phy_only",
#         params=66_492_392, context=2048, random_init=False),
#     "gamba_cons_only_step_random_init": dict(
#         label="Gamba CEP-only Random-Init", family="Gamba", kind="phy_only",
#         params=66_492_392, context=2048, random_init=True),

#     "gamba_dual_ALLPOSstep_44000": dict(
#         label="Gamba NTP+CEP", family="Gamba", kind="seq_plus_phy",
#         params=66_493_418, context=2048, random_init=False),
#     "gamba_dual_step_random_init": dict(
#         label="Gamba NTP+CEP Random-Init", family="Gamba", kind="seq_plus_phy",
#         params=66_493_418, context=2048, random_init=True),

#     # bi-gamba
#     "caduceus_seq_only_ALLPOSstep_44000": dict(
#         label="Bi-Gamba MLM-only", family="Bi-Gamba", kind="seq_only",
#         params=3_864_832, context=2048, random_init=False),
#     "caduceus_seq_only_step_random_init": dict(
#         label="Bi-Gamba MLM-only Random-Init", family="Bi-Gamba", kind="seq_only",
#         params=3_864_832, context=2048, random_init=True),

#     "caduceus_cons_only_ALLPOSstep_44000": dict(
#         label="Bi-Gamba MEM-only", family="Bi-Gamba", kind="phy_only",
#         params=3_864_832, context=2048, random_init=False),
#     "caduceus_cons_only_step_random_init": dict(
#         label="Bi-Gamba MEM-only Random-Init", family="Bi-Gamba", kind="phy_only",
#         params=3_864_832, context=2048, random_init=True),

#     "caduceus_dual_ALLPOSstep_44000": dict(
#         label="Bi-Gamba MLM+MEM", family="Bi-Gamba", kind="seq_plus_phy",
#         params=3_869_442, context=2048, random_init=False),
#     "caduceus_dual_step_random_init": dict(
#         label="Bi-Gamba MLM+MEM Random-Init", family="Bi-Gamba", kind="seq_plus_phy",
#         params=3_869_442, context=2048, random_init=True),

#     # NT / HyenaDNA / PhyloGPN / baselines
#     "nt-ms": dict(
#         label="NT multi-species", family="Other", kind="seq_only",
#         params=498_345_436, context=6000, random_init=False),
#     "nt-ms-random-init": dict(
#         label="NT multi-species Random-Init", family="Other", kind="seq_only",
#         params=498_345_436, context=6000, random_init=True),

#     "nt-human": dict(
#         label="NT human-ref", family="Other", kind="seq_only",
#         params=480_438_241, context=6000, random_init=False),
#     "nt-human-random-init": dict(
#         label="NT human-ref Random-Init", family="Other", kind="seq_only",
#         params=480_438_241, context=6000, random_init=True),

#     "phyloGPN": dict(
#         label="PhyloGPN", family="Other", kind="seq_only",
#         params=83_185_924, context=481, random_init=False),
#     "phyloGPN-random-init": dict(
#         label="PhyloGPN Random-Init", family="Other", kind="seq_only",
#         params=83_185_924, context=481, random_init=True),

#     "hyenaDNA": dict(
#         label="HyenaDNA", family="Other", kind="seq_only",
#         params=6_551_040, context=160_000, random_init=False),
#     "hyenaDNA-random-init": dict(
#         label="HyenaDNA Random-Init", family="Other", kind="seq_only",
#         params=6_551_040, context=160_000, random_init=True),

#     "kmer6": dict(
#         label="K-mer (k=6)", family="Other", kind="baseline_kmer",
#         params=0, context=2048, random_init=False),
#     "phylop": dict(
#         label="PhyloP (6D)", family="Other", kind="baseline_phylop",
#         params=0, context=2048, random_init=False),
# }

# BASELINE_LABELS = {
#     "K-mer (k=6)": "K-mer (k=6) baseline",
#     "PhyloP (6D)": "PhyloP (6D) baseline",
# }

# # colors
# BLUE   = "#4287f5"
# PURPLE = "#6F2DA8"
# ORANGE = "#FF8C32"
# GREY   = "#B0B0B0"

# GAMBA_MARK   = "s"
# BIGAMBA_MARK = "o"
# OTHER_MARK   = "^"


# def color_for(kind, random_init):
#     if random_init:
#         return GREY
#     if kind == "seq_plus_phy":
#         return BLUE
#     if kind == "phy_only":
#         return PURPLE
#     if kind == "seq_only":
#         return ORANGE
#     return GREY


# def marker_for(family):
#     if family == "Gamba":
#         return GAMBA_MARK
#     if family == "Bi-Gamba":
#         return BIGAMBA_MARK
#     return OTHER_MARK


# # ---------------------------------------------------------------------
# # MAIN
# # ---------------------------------------------------------------------
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--tsv",
#         default="/home/mica/gamba/data_processing/data/240-mammalian/global_balacc_recomputed/binary_upstream_balacc_global.tsv",
#         help="input global-balacc tsv (upstream / random / multiclass)",
#     )
#     parser.add_argument(
#         "--eval_type",
#         choices=["upstream", "random", "multiclass"],
#         default="upstream",
#         help="controls y-label/title text; file schema is assumed the same",
#     )
#     parser.add_argument(
#         "-o", "--outdir",
#         default=".",
#         help="directory to save plots",
#     )
#     args = parser.parse_args()

#     os.makedirs(args.outdir, exist_ok=True)

#     # load tsv (use the user-provided path, not a hard-coded one)
#     df = pd.read_csv(args.tsv, sep="\t")

#     rows = []

#     for model_folder, meta in MODEL_META.items():
#         sub = df[df["Model"] == model_folder]

#         if sub.empty:
#             print(f"[warn] no rows for {model_folder}")
#             continue

#         # all of your global_* tsvs have Group + Scope
#         # for upstream & random: ROI is (Group=='all' or 'test', Scope=='roi')
#         # for multiclass: we also stored by Scope, so same logic applies.
#         if "test" in sub["Group"].unique():
#             roi = sub[(sub["Group"] == "test") & (sub["Scope"] == "roi")]
#         else:
#             roi = sub[(sub["Group"] == "all") & (sub["Scope"] == "roi")]

#         if len(roi) != 1:
#             print(f"[warn] {model_folder}: expected 1 ROI row, got {len(roi)}")
#             continue

#         r = roi.iloc[0]

#         rows.append(dict(
#             model_key=model_folder,
#             label=meta["label"],
#             family=meta["family"],
#             kind=meta["kind"],
#             params=meta["params"],
#             context=meta["context"],
#             random_init=meta["random_init"],
#             BA=r["GlobalBalancedAccuracyPct"],
#             SE=r["GlobalBalancedAccuracySEPct"],
#         ))

#     tbl = pd.DataFrame(rows)
#     if tbl.empty:
#         raise SystemExit("no models loaded (check MODEL_META keys vs tsv Model column).")

#     # avoid log(0) for baselines
#     tbl["params_plot"] = tbl["params"].replace(0, 1)
#     tbl = tbl.sort_values("BA", ascending=False)

#     # choose labels depending on evaluation type
#     if args.eval_type == "upstream":
#         y_label = "global ROI 1-vs-upstream balanced accuracy (%)"
#         title = "global ROI 1-vs-upstream BA vs parameter count"
#         filename = "plot_global_balacc_upstream.png"
#     elif args.eval_type == "random":
#         y_label = "global ROI feature-vs-random balanced accuracy (%)"
#         title = "global ROI feature-vs-random BA vs parameter count"
#         filename = "plot_global_balacc_random.png"
#     else:  # multiclass
#         y_label = "global multiclass balanced accuracy (%)"
#         title = "global multiclass BA vs parameter count"
#         filename = "plot_global_balacc_multiclass.png"

#     # ------------------------------------------------------------------
#     # plot
#     # ------------------------------------------------------------------
#     fig, ax = plt.subplots(figsize=(7, 6))

#     for _, r in tbl.iterrows():
#         col = color_for(r["kind"], r["random_init"])
#         mkr = marker_for(r["family"])

#         ax.errorbar(
#             r["params_plot"],
#             r["BA"],
#             yerr=r["SE"],
#             fmt=mkr,
#             mfc=col,
#             mec="black",
#             ms=6,
#             mew=0.9,
#             capsize=2,
#             ecolor="black",
#         )

#         # label every point
#         ax.annotate(
#             r["label"],
#             xy=(r["params_plot"], r["BA"]),
#             xytext=(4, 2),
#             textcoords="offset points",
#             fontsize=8,
#         )

#     # baseline horizontal lines (k-mer / phyloP) if present
#     for _, r in tbl.iterrows():
#         if r["label"] in BASELINE_LABELS:
#             ax.axhline(r["BA"], ls="--", lw=1.2, color="0.35")

#     ax.set_xscale("log")
#     ax.set_xlabel("parameters")
#     ax.set_ylabel(y_label)
#     ax.set_title(title)

#     fig.tight_layout()
#     out_path = os.path.join(args.outdir, filename)
#     fig.savefig(out_path, dpi=300)
#     print("saved to:", out_path)


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------
# MODEL META
# ---------------------------------------------------------------------
MODEL_META = {
    "gamba_seq_only_step44000": dict(
        label="Gamba NTP-only", family="Gamba", kind="seq_only",
        params=66_492_392, context=2048, random_init=False),
    "gamba_seq_only_step0": dict(
        label="Gamba NTP-only Random-Init", family="Gamba", kind="seq_only",
        params=66_492_392, context=2048, random_init=True),

    "gamba_cons_only_step44000": dict(
        label="Gamba CEP-only", family="Gamba", kind="phy_only",
        params=66_492_392, context=2048, random_init=False),
    "gamba_cons_only_step0": dict(
        label="Gamba CEP-only Random-Init", family="Gamba", kind="phy_only",
        params=66_492_392, context=2048, random_init=True),

    "gamba_dual_step44000": dict(
        label="Gamba NTP+CEP", family="Gamba", kind="seq_plus_phy",
        params=66_493_418, context=2048, random_init=False),
    "gamba_dual_step0": dict(
        label="Gamba NTP+CEP Random-Init", family="Gamba", kind="seq_plus_phy",
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
    "caduceus_cons_only_step_random_init": dict(
        label="Bi-Gamba MEM-only Random-Init", family="Bi-Gamba", kind="phy_only",
        params=3_864_832, context=2048, random_init=True),

    "caduceus_dual_step44000": dict(
        label="Bi-Gamba MLM+MEM", family="Bi-Gamba", kind="seq_plus_phy",
        params=3_869_442, context=2048, random_init=False),
    "caduceus_dual_step0": dict(
        label="Bi-Gamba MLM+MEM Random-Init", family="Bi-Gamba", kind="seq_plus_phy",
        params=3_869_442, context=2048, random_init=True),

    # NT / HyenaDNA / PhyloGPN / baselines
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

    "hyenaDNA": dict(
        label="HyenaDNA", family="Other", kind="seq_only",
        params=6_551_040, context=160_000, random_init=False),
    "hyenaDNA-random-init": dict(
        label="HyenaDNA Random-Init", family="Other", kind="seq_only",
        params=6_551_040, context=160_000, random_init=True),

    "caduceus": dict(
        label="Caduceus",
        family="Other",
        kind="seq_only",
        params=7_725_312,
        context=131_000,
        random_init=False
    ),

    "caduceus-random-init": dict(
        label="Caduceus Random-Init",
        family="Other",
        kind="seq_only",
        params=7_725_312,
        context=131_000,
        random_init=True,
    ),

    "caduceus-theirs": dict(
        label="Caduceus",
        family="Other",
        kind="seq_only",
        params=7_725_312,
        context=131_000,
        random_init=False
    ),

    "caduceus-theirs-random-init": dict(
        label="Caduceus Random-Init",
        family="Other",
        kind="seq_only",
        params=7_725_312,
        context=131_000,
        random_init=True,
    ),

    "evo2": dict(
        label="Evo2",
        family="Other",
        kind="seq_only",
        params=7_000_000_000,
        context=2048,
        random_init=False,
    ),

            

    "kmer6": dict(
        label="K-mer (k=6)", family="Other", kind="baseline_kmer",
        params=0, context=2048, random_init=False),
    "phylop": dict(
        label="PhyloP (6D)", family="Other", kind="baseline_phylop",
        params=0, context=2048, random_init=False),
}

BASELINE_LABELS = {
    "K-mer (k=6)": "K-mer (k=6) baseline",
    "PhyloP (6D)": "PhyloP (6D) baseline",
}

# colors
BLUE   = "#4287f5"
PURPLE = "#6F2DA8"
ORANGE = "#FF8C32"
GREY   = "#B0B0B0"

GAMBA_MARK   = "s"
BIGAMBA_MARK = "o"
OTHER_MARK   = "^"


def color_for(kind, random_init):
    if random_init:
        return GREY
    if kind == "seq_plus_phy":
        return BLUE
    if kind == "phy_only":
        return PURPLE
    if kind == "seq_only":
        return ORANGE
    return GREY


def marker_for(family):
    if family == "Gamba":
        return GAMBA_MARK
    if family == "Bi-Gamba":
        return BIGAMBA_MARK
    return OTHER_MARK


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tsv",
        default="/home/mica/gamba/data_processing/data/240-mammalian/global_balacc_upstream/binary_upstream_balacc_global.tsv",
        help="input global-balacc tsv (upstream / random / multiclass)",
    )
    parser.add_argument(
        "--eval_type",
        choices=["upstream", "random", "multiclass", "multiclass100bproi","random_noannot"],
        default="upstream",
        help="controls y-label/title text; file schema is assumed the same",
    )
    parser.add_argument(
        "-o", "--outdir",
        default=".",
        help="directory to save plots",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.tsv, sep="\t")

    rows = []
    for model_folder, meta in MODEL_META.items():
        sub = df[df["Model"] == model_folder]
        if sub.empty:
            print(f"[warn] no rows for {model_folder}")
            continue

        # choose ROI: prefer test/roi if present, else all/roi
        if args.eval_type =="multiclass100bproi":
            if "test" in sub["Group"].unique():
                roi = sub[(sub["Group"] == "test") & (sub["Scope"] == "roi100bp")]
            else:
                roi = sub[(sub["Group"] == "all") & (sub["Scope"] == "roi100bp")]
        else:
            if "test" in sub["Group"].unique():
                roi = sub[(sub["Group"] == "test") & (sub["Scope"] == "roi")]
            else:
                roi = sub[(sub["Group"] == "all") & (sub["Scope"] == "roi")]

        if len(roi) != 1:
            print(f"[warn] {model_folder}: expected 1 ROI row, got {len(roi)}")
            continue

        r = roi.iloc[0]
        rows.append(dict(
            model_key=model_folder,
            label=meta["label"],
            family=meta["family"],
            kind=meta["kind"],
            params=meta["params"],
            context=meta["context"],
            random_init=meta["random_init"],
            BA=r["GlobalBalancedAccuracyPct"],
            SE=r["GlobalBalancedAccuracySEPct"],
        ))

    tbl = pd.DataFrame(rows)
    if tbl.empty:
        raise SystemExit("no models loaded (check MODEL_META keys vs tsv Model column).")

    # sort by BA for nicer ordering on x-axis
    tbl = tbl.sort_values("BA", ascending=False).reset_index(drop=True)
    x = np.arange(len(tbl))

    # labels / titles by eval_type
    if args.eval_type == "upstream":
        y_label = "global ROI 1-vs-upstream balanced accuracy (%)"
        title = "global ROI 1-vs-upstream balanced accuracy"
        filename = "plot_global_balacc_upstream_by_model.svg"
    elif args.eval_type == "random":
        y_label = "global ROI feature-vs-random balanced accuracy (%)"
        title = "global ROI feature-vs-random balanced accuracy"
        filename = "plot_global_balacc_random_by_model.svg"
    elif args.eval_type == "multiclass":
        y_label = "global multiclass balanced accuracy (%)"
        title = "global multiclass balanced accuracy"
        filename = "plot_global_balacc_multiclass_by_model.svg"
    elif args.eval_type == "random_noannot":
        y_label = "global ROI feature-vs-random (no annotation) balanced accuracy (%)"
        title = "global ROI feature-vs-random (no annotation) balanced accuracy"
        filename = "plot_global_balacc_random_noannot_by_model.svg"
    else: # eval_type == "multiclass100bproi":
        y_label= "global multiclass (100bp sampled from ROI) balanced accuracy (%)"
        title = "global multiclass (100bp sampled from ROI) balanced accuracy"
        filename = "plot_global_balacc_multiclass100bproi_by_model.svg"

    # ------------------------------------------------------------------
    # plot: x-axis = model (categorical)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (_, r) in enumerate(tbl.iterrows()):
        col = color_for(r["kind"], r["random_init"])
        mkr = marker_for(r["family"])

        ax.errorbar(
            x[i],
            r["BA"],
            yerr=r["SE"],
            fmt=mkr,
            mfc=col,
            mec="black",
            ms=6,
            mew=0.9,
            capsize=2,
            ecolor="black",
        )

    # x-ticks = model labels
    ax.set_xticks(x)
    ax.set_xticklabels(tbl["label"], rotation=45, ha="right", fontsize=8)

    # baseline horizontal lines if present
    for _, r in tbl.iterrows():
        if r["label"] in BASELINE_LABELS:
            ax.axhline(r["BA"], ls="--", lw=1.2, color="0.35")

    ax.set_xlabel("model")
    ax.set_ylabel(y_label)
    ax.set_title(title)

    fig.tight_layout()
    out_path = os.path.join(args.outdir, filename)
    fig.savefig(out_path, dpi=300)
    print("saved to:", out_path)


if __name__ == "__main__":
    main()


#  python src/evaluation/plotting/plot-BAs.py \
#   --eval_type upstream \
#   --tsv /home/mica/gamba/data_processing/data/240-mammalian/global_balacc_combined/balacc_upstream_global.tsv \
#   -o/home/mica/gamba/data_processing/data/240-mammalian/global_balacc_upstream

#  python src/evaluation/plotting/plot-BAs.py \
#    --eval_type random \
#   --tsv /home/mica/gamba/data_processing/data/240-mammalian/global_balacc_combined/balacc_random_global.tsv\
#   -o /home/mica/gamba/data_processing/data/240-mammalian/global_balacc_random

#  python src/evaluation/plotting/plot-BAs.py \
#   --eval_type multiclass \
#   --tsv /home/mica/gamba/data_processing/data/240-mammalian/global_balacc_combined/balacc_multiclass_global.tsv \
#   -o /home/mica/gamba/data_processing/data/240-mammalian/global_balacc_multiclass

#  python src/evaluation/plotting/plot-BAs.py \
#   --eval_type random_noannot \
#   --tsv /home/mica/gamba/data_processing/data/240-mammalian/global_balacc_combined/balacc_random_noannot_global.tsv \
#   -o /home/mica/gamba/data_processing/data/240-mammalian/global_balacc_random_noannot

#  python src/evaluation/plotting/plot-BAs.py \
#   --eval_type multiclass100bproi \
#   --tsv /home/mica/gamba/data_processing/data/240-mammalian/global_balacc_combined/balacc_multiclass100bproi_global.tsv \
#   -o /home/mica/gamba/data_processing/data/240-mammalian/global_balacc_multiclass100bproi




