# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from utils import compute_auc_per_seed, compute_mse_per_seed, read_outputs, read_scores

# %%
results_dir = Path("/path/to/your/results/directory")


dataset_names = ["friedman1", "ishigami", "g_function"]
model_name = ["mlp"]
ensemble_name = ["bagging"]
method_name = ["loco"]
df_merged = pl.concat(
    [
        read_outputs(
            results_dir,
            dataset,
            model_name,
            ensemble_name,
            method_name,
            n_list=[512],
            seeds=np.arange(1, 100),
            n_jobs=10,
        )
        for dataset in dataset_names
    ]
)
df_merged

# %%
df_roc_auc = compute_auc_per_seed(df_merged, threshold=1e-3)
df_auc_plot = df_roc_auc.to_pandas().copy()

# df_roc_auc = compute_roc_auc(df_merged.to_pandas(), threshold=1e-3)
# df_auc_plot = df_roc_auc.copy()
df_auc_plot["model_name"] = df_auc_plot["model_name"].str.upper()
df_auc_plot["method_name"] = df_auc_plot["method_name"].str.upper()


# %%

df_mse = compute_mse_per_seed(df_merged)

df_plot_mse = (
    df_mse.filter(pl.col("support") == True)
    .with_columns(pl.col("model_name").str.to_uppercase().alias("model_name"))
    .with_columns(pl.col("method_name").str.to_uppercase().alias("method_name"))
    .to_pandas()
)


# %%

scores_df_list = [
    read_scores(
        results_dir=results_dir,
        dataset_name=dataset_name,
        model_name=model_name,
        ensemble_name=ensemble_name,
        n_list=[512],
        seeds=np.arange(1, 100),
        n_jobs=10,
    )
    for dataset_name in dataset_names
]
df_scores = pl.concat(scores_df_list).to_pandas().rename(columns={"model": "strategy"})
df_scores["strategy"] = df_scores["strategy"].replace({"sub_models": "sub-models"})


# %%
_, axes = plt.subplots(
    1, 3, sharex=False, sharey=True, figsize=(6.2, 2.0), gridspec_kw={"wspace": 0.1}
)
palette = "Set2"
palette = {
    "ensemble": "#648fff",
    "sub-models": "#fe6100",
}
# plot prediction R2 for each dataset
sns.boxplot(
    df_scores.query("metric == 'r2'"),
    y="dataset_name",
    x="score",
    ax=axes[0],
    hue="strategy",
    showfliers=False,
    palette=palette,
)
axes[0].set_xlabel(r"R2 Score $\uparrow$")
axes[0].set_yticklabels(
    ["Fried.1", "Ishig.", "G-func"],
    rotation=90,
    va="center",
)


# Plot the MSE for each dataset
sns.boxplot(
    df_plot_mse[df_plot_mse["support"] == True],
    y="dataset_name",
    x="mse",
    ax=axes[1],
    hue="strategy",
    showfliers=False,
    legend=False,
    palette=palette,
)
axes[1].set_xscale("log")
axes[1].set_xlabel("MSE $\downarrow$")

# Plot the ROC AUC for each dataset
sns.boxplot(
    df_auc_plot,
    y="dataset_name",
    x="roc_auc",
    ax=axes[2],
    hue="strategy",
    showfliers=False,
    legend=False,
    palette=palette,
)
axes[2].set_xlabel(r"ROC AUC $\uparrow$")

axes[0].set_ylabel("")
axes[0].legend(title="", fontsize=8)

axes[0].axhspan(0.5, 1.5, color="tab:gray", alpha=0.25, zorder=-10)
axes[1].axhspan(0.5, 1.5, color="tab:gray", alpha=0.25, zorder=-10)
axes[2].axhspan(0.5, 1.5, color="tab:gray", alpha=0.25, zorder=-10)

sns.despine()
plt.tight_layout()
