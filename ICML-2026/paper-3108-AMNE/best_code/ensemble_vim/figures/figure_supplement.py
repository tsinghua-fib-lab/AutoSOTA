# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from utils import compute_auc_per_seed, compute_mse_per_seed, read_outputs

# %%
results_dir = Path("/path/to/your/results/directory")

dataset_name = ["friedman1", "ishigami", "g_function"]
model_name = ["mlp", "rf"]
ensemble_name = ["bagging", "voting"]
method_name = ["cfi"]
df_list = [
    read_outputs(
        results_dir,
        dataset_n,
        model_name,
        ensemble_name,
        method_name,
        n_list=[128, 256, 512, 1024, 2048],
        seeds=np.arange(1, 100),
        n_jobs=10,
    )
    for dataset_n in dataset_name
]
df_merged = pl.concat(df_list)


# %%
###########################
# MSE computation
###########################


result = compute_mse_per_seed(df_merged)

df_plot_mse = (
    result.with_columns(pl.col("model_name").str.to_uppercase().alias("model_name"))
    .with_columns(pl.col("method_name").str.to_uppercase().alias("method_name"))
    .to_pandas()
)

# %%
import matplotlib.ticker as ticker

palette = {
    "ensemble": "#648fff",
    "sub-models": "#fe6100",
    "sub_models": "#fe6100",
}
_, axes = plt.subplots(
    2, 6, sharex=True, sharey=False, gridspec_kw={"hspace": 0.2}, figsize=(10, 4)
)


limits = {
    "SAGE": [(0.5 * 1e-5, 0.02), (0.75 * 1e-3, 0.2)],
    "LOCO": [(1e-4, 0.2), (0.5 * 1e-3, 0.8)],
    "CFI": [(1e-5, 0.05), (1e-2, 2)],
}
method = method_name[0].upper()
support = True
for j, ensembling in enumerate(["bagging", "voting"]):
    for i, dataset in enumerate(["friedman1", "ishigami", "g_function"]):
        ax = axes[0, i + 3 * j]
        df_tmp = df_plot_mse[
            (df_plot_mse["dataset_name"] == dataset)
            & (df_plot_mse["support"] == support)
            & (df_plot_mse["ensemble_name"] == ensembling)
        ]
        do_legend = (j == 1) & (i == 2)
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
            legend=do_legend,
            palette=palette,
            alpha=0.9,
            errorbar=("sd", 0.5),
        )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        if i == 0 and j == 0:
            ax.set_ylabel("MSE $\downarrow$", fontsize=9)
        else:
            print(dataset, ensembling)
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False, labelright=False)

        ax.set_title(dataset, fontsize=10)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        # ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlabel("Nb of samples", fontsize=9)
        ax.tick_params(axis="x", labelsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.set_ylim(limits[method][1][0], limits[method][1][1])

        if i == 1:
            ax.annotate(
                ensembling + " (support features)",
                xy=(0.5, 1.2),
                xycoords="axes fraction",
                ha="center",
                fontsize=10,
                fontweight="bold",
            )
        if do_legend:
            el, labels = ax.get_legend_handles_labels()
            el.pop(3)
            labels.pop(3)
            el.pop(0)
            labels.pop(0)
            ax.legend(
                el, labels, bbox_to_anchor=(0.3, 0.55), loc="upper left", fontsize=8
            )

support = False
for j, ensembling in enumerate(["bagging", "voting"]):
    for i, dataset in enumerate(["friedman1", "ishigami", "g_function"]):
        ax = axes[1, i + 3 * j]
        df_tmp = df_plot_mse[
            (df_plot_mse["dataset_name"] == dataset)
            & (df_plot_mse["support"] == support)
            & (df_plot_mse["ensemble_name"] == ensembling)
        ]

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
            alpha=0.9,
            errorbar=("sd", 0.5),
        )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        if (i == 0) and (j == 0):
            ax.set_ylabel("MSE $\downarrow$", fontsize=9)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False, labelright=False)

        ax.xaxis.set_major_formatter(ScalarFormatter())
        # ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlabel("Nb of samples", fontsize=9)
        ax.tick_params(axis="x", labelsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.set_ylim(limits[method][0][0], limits[method][0][1])

        if i == 1:
            ax.annotate(
                ensembling + " (null features)",
                xy=(0.5, 1.05),
                xycoords="axes fraction",
                ha="center",
                fontsize=10,
                fontweight="bold",
            )
        ax.set_xticks([256, 1024])


sns.despine()
plt.tight_layout()


# %%
###########################
# AUC computation
###########################

df_roc_auc = compute_auc_per_seed(df_merged, threshold=1e-3)


df_auc_plot = df_roc_auc.to_pandas().copy()
df_auc_plot["model_name"] = df_auc_plot["model_name"].str.upper()
df_auc_plot["method_name"] = df_auc_plot["method_name"].str.upper()

# %%

_, axes = plt.subplots(
    1, 6, sharex=True, sharey=False, gridspec_kw={"hspace": 0.2}, figsize=(10, 2)
)

for j, ensembling in enumerate(["bagging", "voting"]):
    for i, dataset in enumerate(["friedman1", "ishigami", "g_function"]):
        ax = axes[i + 3 * j]
        df_tmp = df_auc_plot[
            (df_auc_plot["dataset_name"] == dataset)
            & (df_auc_plot["ensemble_name"] == ensembling)
        ]
        do_legend = (j == 1) & (i == 2)
        sns.lineplot(
            data=df_tmp,
            x="n",
            y="roc_auc",
            hue="strategy",
            hue_order=["ensemble", "sub-models"],
            style="model_name",
            style_order=["RF", "MLP"],
            markers=["o", "^"],
            ax=ax,
            palette=palette,
            legend=do_legend,
            alpha=0.9,
            errorbar=("sd", 0.5),
        )
        if do_legend:
            el, labels = ax.get_legend_handles_labels()
            el.pop(3)
            labels.pop(3)
            el.pop(0)
            labels.pop(0)
            ax.legend(
                el, labels, bbox_to_anchor=(0.3, 0.55), loc="upper left", fontsize=8
            )
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_major_formatter(ScalarFormatter())
        if i == 0 and j == 0:
            ax.set_ylabel(r"ROC AUC $\uparrow$", fontsize=9)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False, labelright=False)
        ax.set_xlabel("Nb of samples", fontsize=9)
        ax.tick_params(axis="x", labelsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.set_title(dataset, fontsize=10)

        ax.set_ylim(0.45, 1.02)
        ax.axhline(0.5, ls="--", color="tab:gray", alpha=0.7, zorder=-10)

        if i == 1:
            ax.annotate(
                ensembling,
                xy=(0.5, 1.2),
                xycoords="axes fraction",
                ha="center",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_xticks([256, 1024])

sns.despine()
plt.tight_layout()
