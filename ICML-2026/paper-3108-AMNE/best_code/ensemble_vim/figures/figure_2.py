# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from utils import compute_auc_per_seed, compute_mse_per_seed, read_outputs

# %%
results_dir = Path("/path/to/your/results/directory")

dataset_name = "friedman1"
model_name = ["mlp", "rf"]
ensemble_name = ["bagging"]
method_name = ["loco", "sage"]
df_merged = read_outputs(
    results_dir,
    dataset_name,
    model_name,
    ensemble_name,
    method_name,
    n_list=[128, 256, 512, 1024, 2048],
    seeds=np.arange(100),
    n_jobs=10,
)


# %%
###########################
# MSE computation
###########################


bootstrap_result = compute_mse_per_seed(df_merged)

df_plot_mse = (
    bootstrap_result.filter(pl.col("support") == True)
    .with_columns(pl.col("model_name").str.to_uppercase().alias("model_name"))
    .with_columns(pl.col("method_name").str.to_uppercase().alias("method_name"))
    .to_pandas()
)

# %%
###########################
# AUC computation
###########################

df_roc_auc = compute_auc_per_seed(df_merged, threshold=1e-3)


df_auc_plot = df_roc_auc.to_pandas().copy()
df_auc_plot["model_name"] = df_auc_plot["model_name"].str.upper()
df_auc_plot["method_name"] = df_auc_plot["method_name"].str.upper()


# %%

from matplotlib.ticker import ScalarFormatter

support = True
palette = {
    "ensemble": "#648fff",
    "sub-models": "#fe6100",
    "sub_models": "#fe6100",
}
_, axes = plt.subplots(
    2, 2, sharex=True, sharey=False, figsize=(4.5, 4.0), gridspec_kw={"hspace": 0.1}
)


y_lim = (df_plot_mse["mse"].min(), df_plot_mse["mse"].max())
for i, method in enumerate(["LOCO", "SAGE"]):
    ax = axes[0, i]
    df_tmp = df_plot_mse[
        (df_plot_mse["method_name"] == method) & (df_plot_mse["support"] == support)
    ].copy()
    print(df_tmp.shape)
    sns.lineplot(
        data=df_tmp,
        x="n",
        y="mse",
        hue="strategy",
        hue_order=["ensemble", "sub-models"],
        style="model_name",
        style_order=["RF", "MLP"],
        markers=["o", "^"],
        ax=ax,
        legend=False,
        palette=palette,
        errorbar=("sd", 0.5),
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_ylabel("MSE $\downarrow$", fontsize=9)
    ax.xaxis.set_major_formatter(ScalarFormatter())

    y_formatter = ScalarFormatter(useMathText=True)
    y_formatter.set_scientific(True)
    y_formatter.set_powerlimits((-0.1, 0.1))
    ax.yaxis.set_major_formatter(y_formatter)
    ax.set_title(method, y=0.9, fontsize=10)
    ax.set_xlabel("Number of samples", fontsize=9)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    if i > 0:
        ax.set_ylabel("")


for i, method in enumerate(["LOCO", "SAGE"]):
    ax = axes[1, i]
    sns.lineplot(
        data=df_auc_plot[df_auc_plot["method_name"] == method],
        x="n",
        y="roc_auc",
        hue="strategy",
        hue_order=["ensemble", "sub-models"],
        style="model_name",
        style_order=["RF", "MLP"],
        markers=["o", "^"],
        ax=ax,
        legend=method == "SAGE",
        palette=palette,
        errorbar=("sd", 0.5),
    )
    if method == "SAGE":
        el, labels = ax.get_legend_handles_labels()
        el.pop(3)
        labels.pop(3)
        el.pop(0)
        labels.pop(0)
        ax.legend(el, labels, bbox_to_anchor=(0.3, 0.55), loc="upper left", fontsize=8)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_ylabel(r"ROC AUC $\uparrow$", fontsize=9)
    ax.set_xlabel("Number of samples", fontsize=9)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    if i > 0:
        ax.set_ylabel("")


sns.despine()
plt.tight_layout()
