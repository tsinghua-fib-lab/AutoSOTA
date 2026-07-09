"""
Generate rebuttal tables (matching rebuttal_tables.md):
  Table 1a/b: HSIC × MMD kernel combinations (TSLS bias), excl. RBF rows/cols
  Table 2   : Different IV estimators (TSLS, LIML, DML)
  Table 3a  : Mis-specified latent dim — GCM rejection counts
  Table 3b  : Mis-specified latent dim — val/inv_loss MMD aggregates
  Table 4   : lam3 (relatedness loss weight) ablation — TSLS bias

Note: RBF rows/columns were not included in the rebuttal, so dropped here.
"""

import os
import numpy as np
import pandas as pd

THETA = 1.0
CKPT = "last"
METRIC = "val_inv_loss"
EXCLUDE = "no048"
EXCLUDE_SIM_IDS = [0, 4, 8]
DATA_SEEDS = list(range(42, 62))
N_SIMS = 12
LAST_K = 20
INSTRUMENTS = ["hW", "hWchV"]
POP_NUM = -1  # combined population
OUTPUTS_DIR = "outputs"


def score_sim_val_inv_loss(metrics_csv, last_k=LAST_K):
    """Mirror evaluate.py ModelSelector 'last' strategy: mean of last_k epochs of val/inv_loss."""
    if not os.path.exists(metrics_csv):
        return None
    df = pd.read_csv(metrics_csv)
    df = df[df["metric"] == "val/inv_loss"].copy()
    if df.empty:
        return None
    df = df.sort_values(["epoch", "timestamp"]).drop_duplicates("epoch", keep="last")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df["value"].tail(last_k).mean()


def best_val_inv_loss_per_seed(exp_id, outputs_dir=OUTPUTS_DIR,
                               data_seeds=DATA_SEEDS, n_sims=N_SIMS,
                               exclude_sim_ids=EXCLUDE_SIM_IDS):
    """For each seed, select the sim with min val/inv_loss (last_k mean). Returns list of scores."""
    out = []
    for ds in data_seeds:
        scores = []
        for sim in range(n_sims):
            if sim in exclude_sim_ids:
                continue
            p = os.path.join(outputs_dir, f"{exp_id}-ds{ds}", str(sim),
                             "metrics", "val_metrics.csv")
            s = score_sim_val_inv_loss(p)
            if s is not None:
                scores.append(s)
        if scores:
            out.append(min(scores))
    return out


def load_extras(results_dir, exp_grp, exp_id, ckpt=CKPT, metric=METRIC, suffix=None):
    if suffix is None:
        suffix = f"insample_{EXCLUDE}"
    path = os.path.join(results_dir, exp_grp,
                        f"extras_{exp_id}_{ckpt}_{metric}_{suffix}.csv")
    if not os.path.exists(path):
        print(f"  Missing: {path}")
        return None
    return pd.read_csv(path)


def load_summary(results_dir, exp_grp, exp_id, ckpt, metric, suffix):
    path = os.path.join(
        results_dir,
        exp_grp,
        f"summary_{exp_id}_{ckpt}_{metric}_{suffix}.csv",
    )
    if not os.path.exists(path):
        print(f"  Missing: {path}")
        return None
    return pd.read_csv(path)


def bias_stats(df, instruments=INSTRUMENTS, pop_num=POP_NUM, theta=THETA):
    """Return mean bias and std across seeds for given instruments."""
    sub = df[(df["pop_num"] == pop_num) & df["instrument"].isin(instruments)]
    sub = sub.copy()
    sub["bias"] = sub["estimate"] - theta
    stats = sub.groupby("instrument")["bias"].agg(["mean", "std"])
    return stats


# ── Table 1: HSIC × MMD kernels ─────────────────────────────────────────────

def table1_kernels():
    print("=" * 70)
    print("Table 1: Estimation bias for different HSIC × MMD kernel combinations")
    print("       (TSLS estimator, polymix degree 3)")
    print("=" * 70)

    hsic_kernels = {
        "Poly 2": ("polyind", "new_mlpnormenc_inv_polyind_ms100"),
        "Poly 3": ("poly3ind", "new_mlpnormenc_inv_poly3ind_ms100"),
        "Orth": ("orthnullind", "new_mlpnormenc_inv_poly3ind_ms100"),
    }
    mmd_kernels = ["meanvar", "poly2", "poly3"]
    mmd_labels = ["Mean-Var", "Poly 2", "Poly 3"]

    rows = []
    for hsic_label, (ind_tag, exp_grp) in hsic_kernels.items():
        for mmd_key, mmd_label in zip(mmd_kernels, mmd_labels):
            exp_id = f"new_normalclamppolymix3_mlpnormenc_{mmd_key}inv_{ind_tag}_ms100"
            suffix = f"insample_{EXCLUDE}"
            df = load_summary("results", exp_grp, exp_id, CKPT, METRIC, suffix)
            if df is None:
                rows.append({
                    "HSIC": hsic_label, "MMD": mmd_label,
                    "instrument": "—", "mean_bias": np.nan, "std_bias": np.nan,
                })
                continue
            stats = bias_stats(df)
            for inst in INSTRUMENTS:
                if inst in stats.index:
                    rows.append({
                        "HSIC": hsic_label,
                        "MMD": mmd_label,
                        "instrument": inst,
                        "mean_bias": stats.loc[inst, "mean"],
                        "std_bias": stats.loc[inst, "std"],
                    })

    df_table = pd.DataFrame(rows)
    # Pivot for nicer display
    for inst in INSTRUMENTS:
        print(f"\nInstrument: {inst}")
        sub = df_table[df_table["instrument"] == inst].copy()
        pivot = sub.pivot(index="HSIC", columns="MMD", values="mean_bias")
        pivot_std = sub.pivot(index="HSIC", columns="MMD", values="std_bias")
        # Format as "mean (std)"
        formatted = pivot.copy()
        for col in formatted.columns:
            formatted[col] = [
                f"{pivot.loc[r, col]:.3f} ({pivot_std.loc[r, col]:.3f})"
                if not np.isnan(pivot.loc[r, col]) else "—"
                for r in formatted.index
            ]
        formatted = formatted.reindex(index=["Orth", "Poly 2", "Poly 3"])
        formatted = formatted.reindex(columns=["Mean-Var", "Poly 2", "Poly 3"])
        print(formatted.to_string())

    # Also save as CSV
    df_table.to_csv("results/rebuttal_table1_kernels.csv", index=False)
    print(f"\nSaved to results/rebuttal_table1_kernels.csv")


# ── Table 2: Different IV estimators ─────────────────────────────────────────

def table2_iv_methods():
    print("\n" + "=" * 70)
    print("Table 2: Estimation bias for different IV estimators")
    print("       (poly2inv, polyind, polymix3)")
    print("=" * 70)

    exp_grp = "new_mlpnormenc_inv_polyind_ms100"
    exp_id = "new_normalclamppolymix3_mlpnormenc_poly2inv_polyind_ms100"

    # (label, method_suffix, csv_instrument)
    estimators = [
        (r"2SLS($\widehat{W}$)", f"insample_{EXCLUDE}", "hW"),
        (r"PO($\widehat{V}$)-2SLS($\widehat{W}$)", f"insample_{EXCLUDE}", "hWchV"),
        (r"LIML($\widehat{W}$)", f"insample_liml_{EXCLUDE}", "hW"),
        (r"LIML($\widehat{W}$, $\widehat{V}$)", f"insample_liml_{EXCLUDE}", "hWchV"),
        (r"DML($\widehat{W}$)", f"insample_dml_{EXCLUDE}", "hW"),
        (r"DML($\widehat{W}$, $\widehat{V}$)", f"insample_dml_{EXCLUDE}", "hWchV"),
    ]

    rows = []
    for label, suffix, inst in estimators:
        df = load_summary("results", exp_grp, exp_id, CKPT, METRIC, suffix)
        if df is None:
            rows.append({
                "Estimator": label,
                "mean_bias": np.nan, "std_bias": np.nan,
            })
            continue
        stats = bias_stats(df, instruments=[inst])
        if inst in stats.index:
            rows.append({
                "Estimator": label,
                "mean_bias": stats.loc[inst, "mean"],
                "std_bias": stats.loc[inst, "std"],
            })

    df_table = pd.DataFrame(rows)
    df_table["Mean Bias (Std)"] = [
        f"{row['mean_bias']:.3f} ({row['std_bias']:.3f})"
        if not np.isnan(row["mean_bias"]) else "—"
        for _, row in df_table.iterrows()
    ]
    print()
    print(df_table[["Estimator", "Mean Bias (Std)"]].to_string(index=False))

    df_table.to_csv("results/rebuttal_table2_iv_methods.csv", index=False)
    print("\nSaved to results/rebuttal_table2_iv_methods.csv")


# ── Helpers shared by Tables 3a/3b ──────────────────────────────────────────

def _mmd_summary_row(label, exp_id):
    vals = best_val_inv_loss_per_seed(exp_id)
    if not vals:
        return {"label": label, "mean": np.nan, "min": np.nan, "max": np.nan, "n": 0}
    s = pd.Series(vals)
    return {"label": label, "mean": s.mean(), "min": s.min(), "max": s.max(), "n": len(s)}


def _gcm_reject_row(label, exp_grp, exp_id, col="gp_w"):
    df = load_extras("results", exp_grp, exp_id)
    if df is None:
        return {"label": label, "reject": "—", "n": 0}
    sub = df[df["pop"] == -1]
    rej = int((sub[col] < 0.05).sum())
    return {"label": label, "reject": f"{rej}/{len(sub)}", "n": len(sub)}


# ── Table 3a/3b: mis-specified latent dimensions ────────────────────────────

MISSPEC_HW = {
    1: "new_normalclamppolymix3_mlpnormenc_inv_polyind_hw1_ms100",
    2: "new_normalclamppolymix3_mlpnormenc_inv_polyind_hw2_ms100",
    3: "new_normalclamppolymix3_mlpnormenc_inv_polyind_hw3_ms100",
    4: "new_normalclamppolymix3_mlpnormenc_inv_polyind_hw4_ms100",
}


def table3a_misspec_gcm():
    print("\n" + "=" * 70)
    print("Table 3a: GCM rejection counts for mis-specified latent dim")
    print("       (H0: hW ⊥ V | W, reject at p<0.05, combined pop)")
    print("=" * 70)

    EXP_GRP = "new_normalclamppolymix3_mlpnormenc_polyinv_polyind_hw"
    rows = [_gcm_reject_row(f"{k}", EXP_GRP, eid) for k, eid in MISSPEC_HW.items()]
    df = pd.DataFrame(rows).rename(columns={"label": "p_hat", "reject": "Reject at 0.05"})
    print()
    print(df[["p_hat", "Reject at 0.05"]].to_string(index=False))
    df.to_csv("results/rebuttal_table3a_misspec_gcm.csv", index=False)
    print("\nSaved to results/rebuttal_table3a_misspec_gcm.csv")
    return df


def table3b_misspec_mmd():
    print("\n" + "=" * 70)
    print("Table 3b: val/inv_loss MMD for mis-specified latent dim")
    print("       (polymix degree 3, best sim per seed by val/inv_loss)")
    print("=" * 70)

    rows = []
    for k, eid in MISSPEC_HW.items():
        r = _mmd_summary_row(str(k), eid)
        rows.append(r)
    df = pd.DataFrame(rows).rename(columns={"label": "p_hat"})
    print()
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    df.to_csv("results/rebuttal_table3b_misspec_mmd.csv", index=False)
    print("\nSaved to results/rebuttal_table3b_misspec_mmd.csv")
    return df


# ── Table 4: lam3 (relatedness loss weight) ablation ────────────────────────

# lam3=0 baseline lives under a different exp_grp/exp_id (polyind was the old name for poly2ind).
LAM3_FILES = {
    "lam3=0":     ("new_mlpnormenc_inv_polyind_ms100",
                   "new_normalclamppolymix3_mlpnormenc_poly2inv_polyind_ms100"),
    "lam3=0.001": ("new_normalclamppolymix3_mlpnormenc_polyinv_polyind_hw",
                   "new_normalclamppolymix3_mlpnormenc_poly2inv_poly2ind_0.001lam3_ms100"),
    "lam3=0.01":  ("new_normalclamppolymix3_mlpnormenc_polyinv_polyind_hw",
                   "new_normalclamppolymix3_mlpnormenc_poly2inv_poly2ind_0.01lam3_ms100"),
    "lam3=0.1":   ("new_normalclamppolymix3_mlpnormenc_polyinv_polyind_hw",
                   "new_normalclamppolymix3_mlpnormenc_poly2inv_poly2ind_0.1lam3_ms100"),
}


def table4_lam3():
    print("\n" + "=" * 70)
    print("Table 4: Estimation bias vs lam3 (relatedness loss weight)")
    print("       (TSLS, poly2inv, poly2ind, polymix degree 3, combined pop)")
    print("=" * 70)

    rows = []
    for col, (exp_grp, eid) in LAM3_FILES.items():
        df = load_summary("results", exp_grp, eid, CKPT, METRIC, f"insample_{EXCLUDE}")
        if df is None:
            for inst in INSTRUMENTS:
                rows.append({"lam3": col, "instrument": inst, "mean": np.nan, "std": np.nan})
            continue
        stats = bias_stats(df)
        for inst in INSTRUMENTS:
            if inst in stats.index:
                rows.append({"lam3": col, "instrument": inst,
                             "mean": stats.loc[inst, "mean"],
                             "std": stats.loc[inst, "std"]})

    df_long = pd.DataFrame(rows)
    df_long["entry"] = [
        f"{r['mean']:.3f} ({r['std']:.3f})" if not np.isnan(r["mean"]) else "—"
        for _, r in df_long.iterrows()
    ]
    pivot = df_long.pivot(index="instrument", columns="lam3", values="entry")
    pivot = pivot.reindex(index=INSTRUMENTS, columns=list(LAM3_FILES.keys()))
    print()
    print(pivot.to_string())
    df_long.to_csv("results/rebuttal_table4_lam3.csv", index=False)
    print("\nSaved to results/rebuttal_table4_lam3.csv")
    return df_long


if __name__ == "__main__":
    table1_kernels()
    table2_iv_methods()
    table3a_misspec_gcm()
    table3b_misspec_mmd()
    table4_lam3()
