import os
import numpy as np
import pandas as pd
import seaborn as sns
from src.datagen import get_nested_circles

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PATH = "results/exp_circles/"
FONTSIZE = 20

# generate data once for all experiments
(X, Y), (S_X, S_Y) = get_nested_circles(
    n_x=250,
    n_y=25,
    p_X0=0.5,
    p_Y0=0.5,
    noise_0=0.15,
    noise_1=0.2,
    diameter=4.0,
    rng=42,
    n_outliers_x=4,
    n_outliers_y=2,
)

results_mlp = pd.read_pickle(f"{PATH}/results_mlp.pkl")
results_penalized = pd.read_pickle(f"{PATH}/results_penalized.pkl")
results_mahalanobis = pd.read_pickle(f"{PATH}/results_mahalanobis.pkl")
results_entropic_ot = pd.read_pickle(f"{PATH}/results_entropic_ot.pkl")

results_penalized["final_fairness"] = results_penalized["fairness_loss_value"]
results_entropic_ot["final_fairness"] = results_entropic_ot[
    "fairness_loss_value"
]


# Function to compute F_hat from transport plan by aggregating over groups
def compute_F_hat(pi, S_X, S_Y):
    """Compute the group-aggregated transport matrix F_hat from the transport
    plan pi."""
    if hasattr(pi, "numpy"):
        pi = pi.numpy()
    if hasattr(S_X, "numpy"):
        S_X = S_X.numpy()
    if hasattr(S_Y, "numpy"):
        S_Y = S_Y.numpy()

    unique_S_X = np.unique(S_X)
    unique_S_Y = np.unique(S_Y)
    k_s = len(unique_S_X)
    k_w = len(unique_S_Y)

    F_hat = np.zeros((k_s, k_w))
    for i, s in enumerate(unique_S_X):
        for j, w in enumerate(unique_S_Y):
            mask_X = S_X == s
            mask_Y = S_Y == w
            F_hat[i, j] = pi[np.ix_(mask_X, mask_Y)].sum()

    return F_hat


plt.rc("font", family="serif")
sns.set_style("whitegrid")

fig, axes = plt.subplots(1, 4, figsize=(20, 4), sharey=True, sharex=True)

# 4 subplots for results
results_list = [
    results_penalized,
    results_mlp,
    results_mahalanobis,
    results_entropic_ot,
]
titles = ["Penalized OT", "MLP", "Mahalanobis", "Entropic OT"]
cmaps = ["Oranges", "Blues", "Greens", "Purples"]

# F matrix heatmap settings
inset_size = 0.25  # size of inset axes as fraction of main axes
heatmap_cmap = "Blues"
vmin_F = 0.04
vmax_F = 0.62


def get_penalty_indices(results, levels=["low", "mid", "high"]):
    """Get indices for lowest, middle, and highest penalty values."""
    sorted_df = results.sort_values("penalty").reset_index(drop=True)
    n = len(sorted_df)
    indices = {
        "low": 0,
        "mid": n // 2,
        "high": n - 1,
    }
    return {level: sorted_df.iloc[indices[level]] for level in levels}


for i, (results, title, cmap) in enumerate(zip(results_list, titles, cmaps)):
    ax = axes[i]

    # Use separate normalization for each method based on its penalty range
    vmin_penalty = results["penalty"].min()
    vmax_penalty = results["penalty"].max()
    norm = plt.matplotlib.colors.LogNorm(vmin=vmin_penalty, vmax=vmax_penalty)

    scatter = ax.scatter(
        results["final_fairness"],
        results["cost_diff"],
        c=results["penalty"],
        cmap=cmap,
        marker="o",
        s=200,
        edgecolor="black",
        norm=norm,
    )

    ax.set_xlabel("Fairness Loss" if i == 1 else None, fontsize=20)
    ax.set_ylabel("Cost Difference" if i == 0 else None, fontsize=20)
    ax.set_title(title, fontsize=20, fontweight="bold")
    ax.grid(True, alpha=1)
    ax.tick_params(axis="both", labelsize=20)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1)
    ax.tick_params(axis="both", which="both", width=2, color="black")

    # Add colorbar for each subplot with its own normalization
    cbar = plt.colorbar(scatter, ax=ax)
    if i < 3:
        cbar.set_label(label="Fairness penalty", fontsize=16)

    else:
        cbar.set_label(label="Entropic penalty", fontsize=16)
    cbar.ax.tick_params(labelsize=16)

    # Get the three penalty levels (low, mid, high)
    selected_rows = get_penalty_indices(results)

    # Inset positions: [x, y, width, height] in axes coordinates
    # Position insets in reverse order (high to low) to avoid crossing arrows
    if i == 0:
        inset_positions = {
            "high": [0.03, 0.6, inset_size, inset_size],
            "mid": [0.25, 0.08, inset_size, inset_size],
            "low": [0.7, 0.6, inset_size, inset_size],
        }
    elif i == 1:
        inset_positions = {
            "high": [0.03, 0.3, inset_size, inset_size],
            "mid": [0.3, 0.2, inset_size, inset_size],
            "low": [0.6, 0.05, inset_size, inset_size],
        }
    else:
        inset_positions = {
            "high": [0.2, 0.7, inset_size, inset_size],
            "mid": [0.2, 0.4, inset_size, inset_size],
            "low": [0.2, 0.1, inset_size, inset_size],
        }

    for level, row in selected_rows.items():
        # Get the transport plan and compute F_hat
        pi = row["fair_ot_plan"]
        F_hat = compute_F_hat(pi, S_X, S_Y)

        # Create inset axes
        inset_ax = ax.inset_axes(inset_positions[level])

        # Plot heatmap in inset
        sns.heatmap(
            F_hat,
            ax=inset_ax,
            cmap=heatmap_cmap,
            cbar=False,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 14},
            vmin=vmin_F,
            vmax=vmax_F,
            linewidths=0.5,
            linecolor="black",
        )
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])

        for spine in inset_ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.5)
        # Force all spines to be on top and visible
        inset_ax.patch.set_edgecolor("black")
        inset_ax.patch.set_linewidth(1.5)
        # Add rectangle to ensure full border is visible
        rect = Rectangle(
            (0, 0),
            1,
            1,
            transform=inset_ax.transAxes,
            fill=False,
            edgecolor="black",
            linewidth=1.5,
            clip_on=False,
            zorder=10,
        )
        inset_ax.add_patch(rect)

        # Get the scatter point coordinates
        point_x = row["final_fairness"]
        point_y = row["cost_diff"]

        # Draw arrow from inset to scatter point
        # Convert inset position to data coordinates for the arrow
        inset_center_x = (
            inset_positions[level][0] + inset_positions[level][2] / 2
        )
        inset_center_y = inset_positions[level][1]

        ax.annotate(
            "",
            xy=(point_x, point_y),
            xytext=(inset_center_x, inset_center_y),
            textcoords="axes fraction",
            arrowprops=dict(
                arrowstyle="->",
                color="black",
                lw=1.5,
                connectionstyle="arc3,rad=-0.2",
            ),
        )
        ax.set_xscale("log")

plt.tight_layout()
os.makedirs("figures/exp_circles/", exist_ok=True)
plt.savefig("figures/exp_circles/cost_diff_vs_fairness_circles.pdf")
plt.show()
