"""
BRCA driver gene recovery: read and plot outputs of run_brca.py.
"""

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

RESULTS_DIR = Path("../results_brca")
MODEL_NAME_list = ["mlp", "logreg_l2"]
VIM_METHOD = "loco"
N_SEEDS = 10

GROUND_TRUTH_GENES = [
    "BCL11A",
    "EZH2",
    "IGF1R",
    "LFNG",
    "BRCA1",
    "SLC22A5",
    "CDK6",
    "BRCA2",
    "TEX14",
    "CCND1",
]
palette = {"ensemble": "#648fff", "sub_models": "#fe6100", "sub-models": "#fe6100"}


# %%
###################################
# Load results across seeds
###################################

all_imp = []
all_metrics = []
all_scores = []
for MODEL_NAME in MODEL_NAME_list:
    for s in range(N_SEEDS):
        seed_dir = RESULTS_DIR / MODEL_NAME / f"seed_{s}"
        if not (seed_dir / "metrics.csv").exists():
            continue

        metrics_s = pd.read_csv(seed_dir / "metrics.csv")
        metrics_s["model_name"] = MODEL_NAME
        all_metrics.append(metrics_s)

        imp_path = seed_dir / f"{VIM_METHOD}_importances.csv"
        if imp_path.exists():
            imp_s = pd.read_csv(imp_path)
            imp_s["seed"] = s
            imp_s["model_name"] = MODEL_NAME
            all_imp.append(imp_s)

        scores_path = seed_dir / "scores.csv"
        if scores_path.exists():
            scores_s = pd.read_csv(scores_path)
            scores_s["seed"] = s
            scores_s["model_name"] = MODEL_NAME
            all_scores.append(scores_s)

df_metrics = pd.concat(all_metrics, ignore_index=True)
df_imp = pd.concat(all_imp, ignore_index=True)
df_scores = pd.concat(all_scores, ignore_index=True)
print(f"Loaded {len(df_metrics)} seeds for {MODEL_NAME_list} / {VIM_METHOD}")

for model_name in MODEL_NAME_list:
    df_m = df_metrics[df_metrics["model_name"] == model_name]
    print(f"Model: {model_name}")
    for col in ["p10_ens", "p10_sub", "auc_ens", "auc_sub"]:
        print(f"  {col}: {df_m[col].mean():.3f} +/- {df_m[col].std():.3f}")

# Gene metadata
gene_names = (
    df_imp[["feature", "gene"]]
    .drop_duplicates()
    .sort_values("feature")["gene"]
    .tolist()
)
n_features = len(gene_names)
support = set(i for i, g in enumerate(gene_names) if g in GROUND_TRUTH_GENES)
n_drivers = len(support)


# %%
###################################
# Figure
###################################


def topk_recovery(importances, support, n_features):
    ranking = np.argsort(importances)[::-1]
    recovered = np.cumsum([1 if r in support else 0 for r in ranking])
    return recovered / len(support)


fig, axes = plt.subplots(
    2, 3, figsize=(7, 6), gridspec_kw={"width_ratios": [1, 2, 1]}, sharey=False
)

for i, MODEL_NAME in enumerate(MODEL_NAME_list):
    # ── (a) Prediction performance: ensemble vs sub-models ──
    ax = axes[i, 0]
    df_pred = df_scores.copy()[df_scores["model_name"] == MODEL_NAME]
    df_pred["strategy"] = df_pred["strategy"].map(
        {"ensemble": "ensemble", "sub_models": "sub-models"}
    )
    df_pred_agg = df_pred.groupby(["seed", "strategy"])["roc_auc"].mean().reset_index()
    sns.boxplot(
        data=df_pred_agg,
        x="strategy",
        y="roc_auc",
        palette=palette,
        width=0.5,
        ax=ax,
        order=["ensemble", "sub-models"],
    )
    sns.stripplot(
        data=df_pred_agg,
        x="strategy",
        y="roc_auc",
        color="black",
        size=4,
        alpha=0.5,
        ax=ax,
        order=["ensemble", "sub-models"],
    )
    ax.set_xlabel("")
    ax.set_ylabel("Cancer type prediction (AUC)")
    ax.axhline(0.5, color="tab:grey", lw=0.8, ls="--", alpha=0.5)
    ax.set_xticklabels(["ensemble", "sub-models"], fontsize=7)
    sns.despine(ax=ax)

    # ── (b) Top-K recovery curve ──
    ens_df = df_imp[
        (df_imp["model"] == "ensemble") & (df_imp["model_name"] == MODEL_NAME)
    ]
    sub_df = df_imp[
        df_imp["model"].str.startswith("sub_model")
        & (df_imp["model_name"] == MODEL_NAME)
    ]

    ens_per_seed = (
        ens_df.groupby(["seed", "feature"])["importance"].mean().reset_index()
    )
    sub_per_seed = (
        sub_df.groupby(["seed", "fold", "feature"])["importance"]
        .mean()
        .reset_index()
        .groupby(["seed", "feature"])["importance"]
        .mean()
        .reset_index()
    )

    ax = axes[i, 1]
    seeds = sorted(ens_per_seed["seed"].unique())
    ks = np.arange(1, n_features + 1)

    ens_curves = []
    sub_curves = []
    for s in seeds:
        ens_imp_s = (
            ens_per_seed[ens_per_seed["seed"] == s]
            .sort_values("feature")["importance"]
            .values
        )
        sub_imp_s = (
            sub_per_seed[sub_per_seed["seed"] == s]
            .sort_values("feature")["importance"]
            .values
        )
        ens_curves.append(topk_recovery(ens_imp_s, support, n_features))
        sub_curves.append(topk_recovery(sub_imp_s, support, n_features))

    ens_curves = np.array(ens_curves)
    sub_curves = np.array(sub_curves)

    for curves, label, color in [
        (ens_curves, "ensemble", palette["ensemble"]),
        (sub_curves, "sub-models", palette["sub_models"]),
    ]:
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        ax.plot(ks, mean, color=color, lw=2, label=label)
        ax.fill_between(ks, mean - std, mean + std, color=color, alpha=0.15)

    ax.plot(ks, ks / n_features, "k--", lw=0.8, alpha=0.5, label="chance")
    ax.axvline(n_drivers, color="grey", lw=0.8, ls=":", alpha=0.6)
    ax.set_xlabel("Top-K genes selected")
    ax.set_ylabel("Fraction of drivers recovered")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_xlim(1, n_features)
    ax.set_ylim(0, 1.05)
    sns.despine(ax=ax)

    # ── (c) AUC boxplot (driver gene identification) ──
    ax = axes[i, 2]
    df_metrics_m = df_metrics[df_metrics["model_name"] == MODEL_NAME]
    df_box = pd.DataFrame(
        {
            "AUC": list(df_metrics_m["auc_ens"]) + list(df_metrics_m["auc_sub"]),
            "strategy": (
                ["ensemble"] * len(df_metrics_m) + ["sub-models"] * len(df_metrics_m)
            ),
        }
    )
    sns.boxplot(
        data=df_box,
        x="strategy",
        y="AUC",
        palette=palette,
        width=0.5,
        ax=ax,
        order=["ensemble", "sub-models"],
    )
    sns.stripplot(
        data=df_box,
        x="strategy",
        y="AUC",
        color="black",
        size=4,
        alpha=0.5,
        ax=ax,
        order=["ensemble", "sub-models"],
    )
    ax.set_xlabel("")
    ax.set_ylabel("Driver gene ranking (AUC)")
    ax.set_xticklabels(["ensemble", "sub-models"], fontsize=7)
    ax.axhline(0.5, color="grey", lw=0.8, ls="--", alpha=0.5)
    sns.despine(ax=ax)


# ── Panel labels ──
for ax, label in zip(axes.ravel(), ["a", "b", "c", "d", "e", "f"]):
    ax.text(
        -0.2,
        1.15,
        f"$\\bf{{{label}}}$",
        transform=ax.transAxes,
        fontsize=14,
        va="top",
    )

for i, MODEL_NAME in enumerate(MODEL_NAME_list):
    axes[i, 1].set_title(
        MODEL_NAME.upper(),
        fontsize=10,
        fontweight="bold",
        bbox={
            "boxstyle": "round",
            "facecolor": "tab:gray",
            "edgecolor": "0.3",
            "alpha": 0.1,
        },
        y=1.05,
    )


plt.tight_layout()
plt.savefig("./figures_pdf/brca_results.pdf", bbox_inches="tight")
plt.show()

# %%
