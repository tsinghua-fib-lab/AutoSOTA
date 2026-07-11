#!/usr/bin/env python3
import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# categories
RENAME_MAP = {"upstream_TSS": "2kb upstream of TSS"}
CORE_ORDER_RAW = ["repeats","introns","UTR5","UTR3","vista_enhancer","UCNE","exons","coding_regions","upstream_TSS","promoters"]
CORE_ORDER = [RENAME_MAP.get(c, c) for c in CORE_ORDER_RAW]

# colors
PALETTE = ["#4287f5", "#FF8C32", "#616161"]

# def load_df(path: str, label: str) -> pd.DataFrame:
#     df = pd.read_parquet(path)
#     need = ["category", "loss"]
#     for c in need:
#         if c not in df.columns:
#             raise ValueError(f"missing column {c} in {path}")
#     df = df[need].copy()
#     df["category"] = df["category"].replace(RENAME_MAP)
#     df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
#     df["model"] = label
#     return df.dropna(subset=["category","loss"])
def load_full_df(path: str, label: str, base: str) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()
    # keep all original columns
    if "loss" not in df.columns:
        raise ValueError(f"missing column 'loss' in {path}")
    df["category"] = df["category"].replace(RENAME_MAP)
    df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
    df["ppl"]  = ce_to_ppl(df["loss"], base)
    df["model"] = label
    return df.dropna(subset=["category", "loss", "ppl"])


def subset_and_order(df: pd.DataFrame, wanted: list[str]) -> tuple[pd.DataFrame, list[str]]:
    present = [c for c in wanted if c in set(df["category"])]
    return df[df["category"].isin(present)].copy(), present

def summarize_sem(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["category","model"], as_index=False).agg(
        mean=("loss","mean"),
        std=("loss","std"),
        n=("loss","count"),
    )
    g["sem"] = g["std"] / np.sqrt(g["n"]).replace(0, np.nan)
    g["sem"] = g["sem"].fillna(0.0)
    return g

# add near imports
from brokenaxes import brokenaxes

# new helper
def ce_to_ppl(x: pd.Series, base: str) -> pd.Series:
    if base == "e":
        return np.exp(x)
    if base == "2":
        return np.power(2.0, x)
    if base == "10":
        return np.power(10.0, x)
    raise ValueError("loss_logbase must be one of: e, 2, 10")

# # update load_df to create 'ppl'
# def load_df(path: str, label: str, base: str) -> pd.DataFrame:
#     df = pd.read_parquet(path)
#     need = ["category", "loss"]
#     for c in need:
#         if c not in df.columns:
#             raise ValueError(f"missing column {c} in {path}")
#     df = df[need].copy()
#     df["category"] = df["category"].replace(RENAME_MAP)
#     df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
#     df["ppl"]  = ce_to_ppl(df["loss"], base)
#     df["model"] = label
#     return df.dropna(subset=["category","loss","ppl"])

# make summarizer generic (default to 'ppl')
def summarize_sem(df: pd.DataFrame, value_col: str = "ppl") -> pd.DataFrame:
    g = df.groupby(["category","model"], as_index=False).agg(
        mean=(value_col,"mean"),
        std=(value_col,"std"),
        n=(value_col,"count"),
    )
    g["sem"] = g["std"] / np.sqrt(g["n"]).replace(0, np.nan)
    g["sem"] = g["sem"].fillna(0.0)
    return g


# add near imports
from brokenaxes import brokenaxes

# new helper
def ce_to_ppl(x: pd.Series, base: str) -> pd.Series:
    if base == "e":
        return np.exp(x)
    if base == "2":
        return np.power(2.0, x)
    if base == "10":
        return np.power(10.0, x)
    print(highest_ppl)

    raise ValueError("loss_logbase must be one of: e, 2, 10")


def ppl_means(df: pd.DataFrame, order: list[str], models: list[str]) -> pd.DataFrame:
    t = (df.groupby(["category","model"], as_index=False)
           .agg(mean_ppl=("ppl","mean"), n=("ppl","count")))
    cats = [c for c in order if c in set(t["category"])]
    mods = [m for m in models if m in set(t["model"])]
    t["category"] = pd.Categorical(t["category"], categories=cats, ordered=True)
    t["model"]    = pd.Categorical(t["model"],    categories=mods, ordered=True)
    return t.sort_values(["category","model"])


def load_df(path: str, label: str, base: str) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()             # keep everything
    if "loss" not in df.columns:
        raise ValueError(f"missing 'loss' in {path}")
    if "category" not in df.columns:
        raise ValueError(f"missing 'category' in {path}")
    df["category"] = df["category"].replace(RENAME_MAP)
    df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
    df["ppl"]  = ce_to_ppl(df["loss"], base)
    df["model"] = label
    # if a file lacks data_split, mark it (so the column exists post-concat)
    if "data_split" not in df.columns:
        df["data_split"] = "Unknown"
    return df.dropna(subset=["category","loss","ppl"])


# # update load_df to create 'ppl'
# def load_df(path: str, label: str, base: str) -> pd.DataFrame:
#     df = pd.read_parquet(path)
#     need = ["category", "loss"]
#     for c in need:
#         if c not in df.columns:
#             raise ValueError(f"missing column {c} in {path}")
#     df = df[need].copy()
#     df["category"] = df["category"].replace(RENAME_MAP)
#     df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
#     df["ppl"]  = ce_to_ppl(df["loss"], base)
#     df["model"] = label
#     return df.dropna(subset=["category","loss","ppl"])

# make summarizer generic (default to 'ppl')
def summarize_sem(df: pd.DataFrame, value_col: str = "ppl") -> pd.DataFrame:
    g = df.groupby(["category","model"], as_index=False).agg(
        mean=(value_col,"mean"),
        std=(value_col,"std"),
        n=(value_col,"count"),
    )
    g["sem"] = g["std"] / np.sqrt(g["n"]).replace(0, np.nan)
    g["sem"] = g["sem"].fillna(0.0)
    return g

from brokenaxes import brokenaxes


# add import
from scipy.stats import t

def summarize_err(df: pd.DataFrame, value_col: str = "ppl", err_type: str = "ci95") -> pd.DataFrame:
    g = df.groupby(["category","model"], as_index=False).agg(
        mean=(value_col,"mean"),
        std=(value_col,"std"),
        n=(value_col,"count"),
    )
    se = g["std"] / np.sqrt(g["n"]).replace(0, np.nan)
    if err_type == "sem":
        err = se
    elif err_type == "ci95":
        # t critical per group; use 1.96 approx if df>30
        dfree = g["n"] - 1
        tcrit = np.where(dfree > 30, 1.96, t.ppf(0.975, np.clip(dfree, 1, None)))
        err = se * tcrit
    else:
        raise ValueError("err_type must be 'sem' or 'ci95'")
    g["err"] = pd.Series(err).fillna(0.0)
    return g

# compute a thin top band from the summary table (which is in ppl units)
def auto_ybreak_from_summary(summary: pd.DataFrame, low_max: float = 4.0, pad_frac: float = 0.002):
    top = float(summary["mean"].max())
    pad = max(1.0, top * pad_frac)   # tiny strip around the max
    lo0, lo1 = 0.0, low_max
    hi0, hi1 = top - pad, top + pad
    if hi0 <= lo1:  # avoid touching panels
        lo1 = hi0 * 0.9
    return (lo0, lo1, hi0, hi1)
    
def bar_plot_sem(summary: pd.DataFrame, order: list[str], models: list[str],
                 title: str, out_path: str, ylim=(0.0, 4.0),
                 value_fmt="{:.2f}"):
    if summary.empty:
        print(f"[warn] no data for {title}")
        return

    cats = [c for c in order if c in set(summary["category"])]
    models = [m for m in models if m in set(summary["model"])]

    width = 1.6 * max(6, len(cats))
    fig, ax = plt.subplots(figsize=(width, 6))

    x = np.arange(len(cats), dtype=float)
    bar_w = 0.8 / max(1, len(models))
    left = x - 0.4 + bar_w/2

    y_min, y_max = ylim
    clamp = lambda y: np.minimum(y, y_max)

    handles, labels = [], []

    for j, model in enumerate(models):
        sub = summary[summary["model"] == model].set_index("category")
        h = np.array([sub.loc[c, "mean"] if c in sub.index else np.nan for c in cats], float)
        e = np.array([sub.loc[c, "err"]  if c in sub.index else np.nan for c in cats], float)
        mask = np.isfinite(h)
        xpos = (left + j*bar_w)[mask]
        h_plot = clamp(h[mask])

        bars = ax.bar(xpos, h_plot, bar_w,
                      color=PALETTE[j % len(PALETTE)],
                      linewidth=0, label=model, zorder=2)

        e_plot = np.where(h[mask] >= y_max, 0.0, np.minimum(e[mask], y_max - h_plot))
        ax.errorbar(xpos, h_plot, yerr=e_plot, fmt="none",
                    ecolor="black", capsize=3, linewidth=1, zorder=3)

        # white rotated labels inside bars
        y_text = np.minimum(h_plot/2.0, y_max - 0.1*(y_max - y_min))
        for xb, yt, hv in zip(xpos, y_text, h[mask]):
            ax.text(xb, float(yt), value_fmt.format(hv),
                    ha="center", va="center", fontsize=8,
                    color="white", rotation=90, clip_on=True, zorder=4)

        handles.append(bars[0]); labels.append(model)

    ax.set_ylim(*ylim)
    ax.set_xlabel("feature category")
    ax.set_ylabel("perplexity")
    ax.set_title(title)
    ax.set_xticks(x, cats, rotation=30, ha="right")

    # clean background, keep border
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)

    # legend outside right
    fig.legend(handles, labels, loc="center left",
               bbox_to_anchor=(1.02, 0.5), frameon=False)

    plt.tight_layout(rect=[0,0,0.88,1])  # leave room for legend
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.savefig(out_path.replace(".png", ".svg"))
    plt.close()

from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests
import numpy as np
import pandas as pd

def paired_ttests_by_category_index(
    df: pd.DataFrame,
    model_a: str = "Bi-Gamba MLM+MEM",
    model_b: str = "Bi-Gamba MLM-only",
    value_col: str = "ppl",
    out_csv: str | None = None,
) -> pd.DataFrame:
    d = df[df["model"].isin([model_a, model_b])].copy()
    if d.empty:
        return pd.DataFrame()

    # category-wise row index to “pair by position”
    d["row_id"] = d.groupby(["model","category"]).cumcount()

    wide = (d.pivot_table(index=["category","row_id"], columns="model", values=value_col, aggfunc="first")
              .dropna(subset=[model_a, model_b]))

    rows = []
    for cat, g in wide.groupby(level=0, sort=False):
        a = g.droplevel(0)[model_a].to_numpy(float)
        b = g.droplevel(0)[model_b].to_numpy(float)
        n = min(len(a), len(b))
        if n < 2:
            rows.append({"category": cat, "n_pairs": n, "mean_a": np.nan, "mean_b": np.nan,
                         "mean_diff": np.nan, "t_stat": np.nan, "p_value": np.nan,
                         "p_adj_BH": np.nan, "cohen_dz": np.nan})
            continue
        diff = a - b
        t_stat, p_val = ttest_rel(a, b, nan_policy="omit")
        dz = (np.nanmean(diff) / np.nanstd(diff, ddof=1)) if np.nanstd(diff, ddof=1) > 0 else np.nan
        rows.append({
            "category": cat,
            "n_pairs": int(n),
            "mean_a": float(np.nanmean(a)),
            "mean_b": float(np.nanmean(b)),
            "mean_diff": float(np.nanmean(diff)),  # >0 ⇒ model_a lower ppl if lower=better
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "cohen_dz": float(dz),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        rej, p_adj, *_ = multipletests(out["p_value"].values, method="fdr_bh")
        out["p_adj_BH"] = p_adj
        out = out.sort_values("p_adj_BH")
        if out_csv:
            out.to_csv(out_csv, index=False)
    return out

def summarize_split_err(df: pd.DataFrame, value_col: str = "ppl", err_type: str = "sem") -> pd.DataFrame:
    if "data_split" not in df.columns:
        print("[warn] no data_split column found; skipping split plot.")
        return pd.DataFrame()
    d = df[df["data_split"].isin(["Held Out","Training"])].copy()
    g = (d.groupby(["category","model","data_split"], as_index=False)
           .agg(mean=(value_col,"mean"), std=(value_col,"std"), n=(value_col,"count")))
    se = g["std"] / np.sqrt(g["n"]).replace(0, np.nan)
    g["err"] = se.fillna(0.0) if err_type=="sem" else (se*1.96).fillna(0.0)
    return g

from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

def paired_ttests_by_category_split(
    df: pd.DataFrame,
    split: str = "Held Out",
    model_a: str = "Bi-Gamba MLM+MEM",
    model_b: str = "Bi-Gamba MLM-only",
    value_col: str = "ppl",
    out_csv: str | None = None,
) -> pd.DataFrame:
    # ID columns to align the *same* region across models
    ignore = {"model","category",value_col,"loss","ppl","sem","std","n","err","data_split"}
    cand = [c for c in df.columns if c not in ignore]
    prefer = [c for c in ["chrom","start","end","feature_id"] if c in cand]
    keys = prefer or cand
    if not keys:
        raise ValueError("No identifier columns to align pairs (need e.g. chrom/start/end/feature_id).")

    dsplit = df[df["data_split"] == split].copy()
    rows = []
    cats = sorted(dsplit["category"].dropna().unique().tolist()) + ["All"]

    for cat in cats:
        d = dsplit if cat == "All" else dsplit[dsplit["category"] == cat]
        if d.empty: 
            continue
        da = d[d["model"] == model_a][keys + [value_col,"category"]].rename(columns={value_col:"a"})
        db = d[d["model"] == model_b][keys + [value_col,"category"]].rename(columns={value_col:"b"})
        m  = pd.merge(da, db, on=keys+["category"], how="inner")

        n = len(m)
        if n < 2:
            rows.append({"category": cat, "n_pairs": n, "mean_a": np.nan, "mean_b": np.nan,
                         "mean_diff": np.nan, "t_stat": np.nan, "p_value": np.nan,
                         "cohen_dz": np.nan, "split": split})
            continue

        a = m["a"].to_numpy(float); b = m["b"].to_numpy(float)
        diff = a - b
        t_stat, p_val = ttest_rel(a, b, nan_policy="omit")
        dz = np.nan if np.nanstd(diff, ddof=1) == 0 else np.nanmean(diff)/np.nanstd(diff, ddof=1)

        rows.append({
            "category": cat, "split": split, "n_pairs": int(n),
            "mean_a": float(np.nanmean(a)), "mean_b": float(np.nanmean(b)),
            "mean_diff": float(np.nanmean(diff)),  # >0 ⇒ model_a higher than model_b
            "t_stat": float(t_stat), "p_value": float(p_val), "cohen_dz": float(dz),
        })

    out = pd.DataFrame(rows)
    mask = (out["category"] != "All")
    if mask.any():
        out.loc[mask, "p_adj_BH"] = multipletests(out.loc[mask,"p_value"].values, method="fdr_bh")[1]
    else:
        out["p_adj_BH"] = np.nan

    out = pd.concat([out[out["category"]=="All"],
                     out[out["category"]!="All"].sort_values("p_adj_BH")], ignore_index=True)

    if out_csv: out.to_csv(out_csv, index=False)
    return out


def bar_plot_split_by_model(summary: pd.DataFrame, order: list[str], models: list[str],
                            title: str, out_path: str, ylim=(0.0, 4.0), value_fmt="{:.2f}"):
    if summary.empty:
        print(f"[warn] no data for {title}"); return

    cats  = [c for c in order  if c in set(summary["category"])]
    mods  = [m for m in models if m in set(summary["model"])]
    splits = ["Held Out","Training"]

    width = 1.6 * max(6, len(cats))
    fig, ax = plt.subplots(figsize=(width, 6))

    x = np.arange(len(cats), dtype=float)
    n_bars = len(mods) * 2
    bar_w = 0.8 / max(1, n_bars)
    left = x - 0.4 + bar_w/2

    handles, labels = [], []
    for j, model in enumerate(mods):
        for k, split in enumerate(splits):
            sub = summary[(summary["model"]==model) & (summary["data_split"]==split)].set_index("category")
            h = np.array([sub.loc[c, "mean"] if c in sub.index else np.nan for c in cats], float)
            e = np.array([sub.loc[c, "err"]  if c in sub.index else np.nan for c in cats], float)
            mask = np.isfinite(h)
            xpos = (left + (j*2 + k)*bar_w)[mask]

            bars = ax.bar(
                xpos, h[mask], bar_w,
                color=PALETTE[j % len(PALETTE)],
                alpha=0.95 if split=="Held Out" else 0.60,
                linewidth=0, zorder=2,
                label=f"{model} — {split}"
            )
            ax.errorbar(xpos, h[mask], yerr=e[mask], fmt="none",
                        ecolor="black", capsize=3, linewidth=1, zorder=3)

            # white vertical labels inside bars
            y_min, y_max = ylim
            y_text = np.minimum(h[mask]/2.0, y_max - 0.1*(y_max - y_min))
            for xb, yt, hv in zip(xpos, y_text, h[mask]):
                ax.text(xb, float(yt), value_fmt.format(hv),
                        ha="center", va="center", fontsize=8,
                        color="white", rotation=90, clip_on=True, zorder=4)

            handles.append(bars[0]); labels.append(f"{model} — {split}")

    ax.set_ylim(*ylim)
    ax.set_xlabel("feature category")
    ax.set_ylabel("perplexity")
    ax.set_title(title)
    ax.set_xticks(x, cats, rotation=30, ha="right")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)

    # legend outside right
    fig.legend(handles, labels, loc="center left",
               bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.tight_layout(rect=[0,0,0.88,1])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.savefig(out_path.replace(".png",".svg"))
    plt.close()


def main():
    p = argparse.ArgumentParser(description="Grouped bar plots with SEM for CE loss.")
    p.add_argument("--parquet_a", default="/home/mica/gamba/data_processing/data/240-mammalian/phylop_corr_analysis/caduceus_dual_step_44000/all_results.parquet")
    p.add_argument("--parquet_b", default="/home/mica/gamba/data_processing/data/240-mammalian/phylop_corr_analysis/caduceus_seq_only_step_44000/all_results.parquet")
    p.add_argument("--parquet_c", default="/home/mica/gamba/data_processing/data/240-mammalian/phylop_corr_analysis/caduceus-theirs/all_results.parquet")
    p.add_argument("--loss_logbase", choices=["e","2","10"], default="e",
                   help="log base of CE loss; PyTorch CE uses natural log (e).")
    p.add_argument("--label_a", default="Bi-Gamba MLM+MEM")
    p.add_argument("--label_b", default="Bi-Gamba MLM-only")
    p.add_argument("--label_c", default="Caduceus")
    p.add_argument("--outdir", default="/home/mica/gamba/data_processing/data/240-mammalian/figures")
    p.add_argument("--ybreak", type=float, nargs=4, default=[0.0, 4.0, 10000000.0, 10000004.0],
                   help="low_min low_max high_min high_max for broken y-axis")
    p.add_argument("--ylim", type=float, nargs=2, default=[0.0, 1.8])
    args = p.parse_args()

    dfa = load_df(args.parquet_a, args.label_a, args.loss_logbase)
    dfb = load_df(args.parquet_b, args.label_b, args.loss_logbase)
    dfc = load_df(args.parquet_c, args.label_c, args.loss_logbase)
    df = pd.concat([dfa, dfb, dfc], ignore_index=True)

    # print("available columns:", sorted(df.columns.tolist()))
    # # after df = concat([...]) and ppl computed
    # tt = paired_ttests_by_category_index(
    #     df, model_a="Bi-Gamba MLM+MEM", model_b="Bi-Gamba MLM-only",
    #     value_col="ppl", out_csv=os.path.join(args.outdir, "paired_ttests_by_category.csv")
    # )
    # print(tt.to_string(index=False, float_format=lambda x: f"{x:.3g}"))


    # #print the t test results per category
    # print("\npaired t-test results by category (Bi-Gamba MLM+MEM vs. Bi-Gamba MLM-only):")
    # if not tt.empty:
    #     print(tt.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    # else:
    #     print("No t-test results available.")

    hue_order = [args.label_a, args.label_b, args.label_c]
    df_core, order_core = subset_and_order(df, CORE_ORDER)
    tbl = ppl_means(df_core, order_core, hue_order)

    print("\nperplexity means by category (lower is better):")
    for cat, sub in tbl.groupby("category", sort=False):
        print(cat)
        for _, r in sub.iterrows():
            print(f"  {r['model']}: {r['mean_ppl']:.3f}  (n={int(r['n'])})")

    wide = tbl.pivot(index="category", columns="model", values="mean_ppl")
    print("\nwide table:")
    print(wide.to_string(float_format=lambda x: f'{x:.3f}'))

    os.makedirs(args.outdir, exist_ok=True)
    out_csv = os.path.join(args.outdir, "perplexity_means_by_category.csv")
    tbl.to_csv(out_csv, index=False)
    print(f"[ok] wrote {out_csv}")

    # 1) confirm base
    print("loss base:", args.loss_logbase)

    # 2) top offenders
    print(df.sort_values("ppl", ascending=False).head(10)[["model","category","loss","ppl"]])

    # 3) log-view sanity plot
    df_log = df.assign(log10_ppl=np.log10(df["ppl"]))
    df_log.boxplot(column="log10_ppl", by=["model","category"], rot=90)
    plt.ylabel("log10 perplexity"); plt.tight_layout(); plt.show()


    # in main(), compute ppl and summary, then call with a very thin top band
    core_summary = summarize_err(df_core, value_col="ppl", err_type="sem")
    ybreak = auto_ybreak_from_summary(core_summary, low_max=4.0, pad_frac=0.001)  # 0.1% band
    bar_plot_sem(core_summary, order_core, hue_order,
             "perplexity by category (biological)",
             os.path.join(args.outdir, "bar_core_categories_ppl.png"),
             ylim=(0.0, 4.0))

    tt_heldout = paired_ttests_by_category_split(
        df,
        split="Held Out",
        model_a=args.label_a,
        model_b=args.label_b,
        value_col="ppl",
        out_csv=os.path.join(args.outdir, "paired_ttests_by_category_heldout.csv"),
    )
    print("\nHeld-Out paired t-tests (A vs B):")
    print(tt_heldout.to_string(index=False, float_format=lambda x: f"{x:.3g}"))


    # Split-specific summary & plot (extra figure)
    split_summary = summarize_split_err(df_core, value_col="ppl", err_type="sem")
    bar_plot_split_by_model(
        split_summary, order_core, hue_order,
        title="perplexity by category — Held Out vs Training (per model)",
        out_path=os.path.join(args.outdir, "bar_core_categories_ppl_split.png"),
        ylim=(0.0, 4.0),
    )
if __name__ == "__main__":
    main()
