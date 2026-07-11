import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# matplotlib style
fontsize = 20
rc = {
    "font.size": fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,
    "font.family": "serif",
    "font.serif": ["Times"],
}
plt.rcParams.update(rc)

# parameters 
nb_seeds = 100

# read dataframe
# simulation_dir = "/storage/store4/work/aheurteb/LiMVAM/experiments_synthetic/"
simulation_dir = "/Users/ambroiseheurtebise/Desktop/LiMVAM/experiments_synthetic/"
results_dir = simulation_dir + "results/results_diversity_correlation_assumption/"
save_name = f"DataFrame_with_{nb_seeds}_seeds"
save_path = results_dir + save_name
df = pd.read_csv(save_path)

# add a column
df["success_P_exact"] = 1 - df["error_P_exact"]

# ---------- group definitions ----------
groups = [
    ("Assumption is\nfulfilled", (True,  True)),
    ("No correlation and\ndiversity across\nviews", (False, True)),
    ("No correlation and\ndiversity across\nvariables", (True,  False)),
]

# COLOR_SCORE = "#1f77b4"
COLOR_SCORE = "#17becf"
COLOR_ERROR = "#ff7f0e"

fig, ax1 = plt.subplots(figsize=(10, 4))
ax2 = ax1.twinx()

positions = np.array([1, 2, 3])
width = 0.35
gap = 0.03
CLIP_PERCENTILE = 99

for ax, col, color, side in [
    (ax1, "score_assump",  COLOR_SCORE, "left"),
    (ax2, "success_P_exact", COLOR_ERROR, "right"),
]:
    for pos, (label, (cv, var)) in zip(positions, groups):
        mask = (
            (df["cross_view_correlations"] == cv) &
            (df["cross_variable_correlations"] == var)
        )
        data = df.loc[mask, col].dropna().values

        # separate outliers from main distribution
        is_third_group = (cv, var) == (True, False)
        p = CLIP_PERCENTILE if (side == "right" and is_third_group) else 100

        outlier_threshold = np.percentile(data, p)
        outliers = data[data > outlier_threshold]
        data_clipped = data[data <= outlier_threshold]
        
        if np.allclose(data_clipped, data_clipped[0]):
            ax.hlines(data_clipped[0], pos, pos + width, colors=color, linewidths=4)
            continue

        parts = ax.violinplot(
            data_clipped,
            positions=[pos],
            widths=width * 2,
            showmedians=False,
            showextrema=False,
        )

        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_edgecolor(color)
            pc.set_alpha(0.75)

            # keep only the left or right half
            verts = pc.get_paths()[0].vertices
            if side == "left":
                verts[:, 0] = np.minimum(verts[:, 0], pos - gap)
            else:
                verts[:, 0] = np.maximum(verts[:, 0], pos + gap)
        
    # # plot outliers as dots
    # x_outliers = pos - gap if side == "left" else pos + gap
    # ax.scatter(
    #     [x_outliers] * len(outliers),
    #     outliers,
    #     color=color, zorder=5, s=20, alpha=0.6
    # )

for x in [1.5, 2.5]:
    ax1.axvline(x, color="gray", linestyle=":", linewidth=1, alpha=0.7)

ax1.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.7)
ax1.axhline(1, color="gray", linestyle=":", linewidth=1, alpha=0.7)

# ---------- axes styling ----------
eps = 0.1
ax1.set_ylim([-eps, 1+eps])
ax2.set_ylim([-eps, 1+eps])
ax1.set_ylabel("Score on assumption",  color=COLOR_SCORE, fontsize=fontsize)
ax2.set_ylabel("Success rate on P", color=COLOR_ERROR, fontsize=fontsize)
ax1.tick_params(axis="y", colors=COLOR_SCORE)
ax2.tick_params(axis="y", colors=COLOR_ERROR)

ax1.set_xticks(positions)
ax1.set_xticklabels([g[0] for g in groups], fontsize=fontsize)
ax1.set_xlim(0.5, 3.5)

# caption
# caption = (
#     r"Caption: Data are generated with $m=10$ views and $p=5$ disturbances," + "\n"
#     "and the experiment is ran with 100 different seeds."
# )
# fig.text(0.5, -0.07, caption, ha='center', va='center', fontsize=fontsize)


plt.tight_layout()

# save figure
figures_dir = simulation_dir + "figures/"
plt.savefig(
    figures_dir + f"simulation_diversity_correlation_assumption.pdf", 
    bbox_inches="tight")
plt.savefig(
    figures_dir + f"simulation_diversity_correlation_assumption.png", 
    bbox_inches="tight", dpi=300)
# plt.show()
