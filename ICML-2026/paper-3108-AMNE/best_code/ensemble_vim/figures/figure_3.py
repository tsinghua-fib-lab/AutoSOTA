# %%
#
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from utils import read_outputs

# %%
results_dir = Path("/path/to/your/results/directory")

dataset_name = "friedman1"
model_name = ["mlp"]
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
# Squarred bias computation
###########################

df_tmp = df_merged.with_columns(
    bias=(pl.col("importance") - pl.col("asymptotic_importance"))
).drop(["asymptotic_importance"])

df_plot = (
    df_tmp.filter(pl.col("support") == True)
    .group_by(
        [
            "feature",
            "dataset_name",
            "n",
            "model_name",
            "ensemble_name",
            "strategy",
            "method_name",
        ]
    )
    .agg(
        [
            pl.col("bias").mean().alias("mean_bias"),
            pl.col("importance").var().alias("variance"),
        ]
    )
    .group_by(
        [
            "dataset_name",
            "n",
            "model_name",
            "ensemble_name",
            "strategy",
            "method_name",
        ]
    )
    .agg(
        [
            (pl.col("mean_bias").mean() ** 2).alias("bias_squared"),
            pl.col("variance").mean().alias("variance"),
        ]
    )
)
df_plot = df_plot.to_pandas()
df_plot = (
    df_plot.groupby(["n", "strategy", "method_name", "dataset_name"])[
        ["bias_squared", "variance"]
    ]
    .mean()
    .reset_index()
)
# %%
df_plot = df_plot.rename(columns={"bias_squared": "bias_squarred"}).melt(
    id_vars=[
        # "dataset_name",
        "n",
        # "model_name",
        # "ensemble_name",
        "strategy",
        "method_name",
    ],
    value_vars=["bias_squarred", "variance"],
    var_name="component",
    value_name="value",
)
df_plot["method_name"] = df_plot["method_name"].str.upper()
# %%
import matplotlib.ticker as ticker
from matplotlib.patches import Patch

_, axes = plt.subplots(
    1, 2, sharex=True, sharey=False, figsize=(6.2, 2.5), gridspec_kw={"wspace": 0.2}
)

palette = {
    "ensemble": "#648fff",
    "sub-models": "#fe6100",
}
# Get colors from the Paired colormap
paired_colors = sns.color_palette("tab20", 20)
colors_ensemble = ["#69cc647a", "#6acc64"]  # Light and dark green
colors_sub_models = ["#ee864a83", "#ee854a"]  # Light and dark blue
colors_ensemble = ["#6490ff83", "#648fff"]  # Light and dark green
colors_sub_models = ["#fe610083", "#fe6100"]  # Light and dark blue

for i, method in enumerate(["LOCO", "SAGE"]):
    for j, strat in enumerate(["ensemble", "sub-models"]):
        df_model = df_plot[df_plot["strategy"] == strat]
        ax = axes[i]
        df_method = df_model[df_model["method_name"] == method]

        # Pivot data for stacked bar plot
        df_pivot = df_method.pivot(index="n", columns="component", values="value")

        # Select colors based on model
        colors = colors_ensemble if strat == "ensemble" else colors_sub_models

        df_pivot.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color=colors,
            width=0.33,
            legend=i == 1,
            position=j,
        )

        # ax.set_yscale("log")
        if method == "LOCO":
            ax.set_ylabel("MSE $\downarrow$", fontsize=12, labelpad=-1)
        else:
            ax.set_ylabel("")
        ax.set_xlabel("Number of samples", fontsize=12)
        ax.set_title(method, y=0.95, fontsize=12)
        # ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.set_xticklabels(df_pivot.index, rotation=0)
        ax.tick_params(axis="y", labelsize=12)
        ax.tick_params(axis="x", labelsize=12)

        if i == 1:
            # Create custom legend with hierarchical structure
            legend_elements = [
                Patch(facecolor="none", edgecolor="none", label="ensemble"),
                Patch(facecolor=colors_ensemble[0], label="  Bias²"),
                Patch(facecolor=colors_ensemble[1], label="  Variance"),
                Patch(facecolor="none", edgecolor="none", label="sub-models"),
                Patch(facecolor=colors_sub_models[0], label="  Bias²"),
                Patch(facecolor=colors_sub_models[1], label="  Variance"),
            ]
            ax.legend(
                handles=legend_elements,
                loc="upper right",
                ncol=2,
                fontsize=8.5,
                columnspacing=0.4,
                handletextpad=0.3,
                bbox_to_anchor=(1.05, 1),
            )

        ax.set_xlim(-0.5, 4.33)
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits(
            (-2, 2)
        )  # Triggers scientific notation if < 0.01 or > 100

        ax.yaxis.set_major_formatter(formatter)

sns.despine()
plt.tight_layout()
