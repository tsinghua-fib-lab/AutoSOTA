import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from pathlib import Path


# matplotlib and seaborn style
fontsize = 18
rc = {
    "font.size": fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,
    "font.family": "serif",
    "font.serif": ["Times"],
}
sns.set(style="white")
plt.rcParams.update(rc)

# parameters 
nb_seeds = 200
median_or_mean = "mean"

# read dataframe
# simulation_dir = Path("/storage/store4/work/aheurteb/LiMVAM/experiments_synthetic")
simulation_dir = Path("/Users/ambroiseheurtebise/Desktop/LiMVAM/experiments_synthetic")
results_dir = simulation_dir / "results/results_execution_time"
save_name = f"DataFrame_with_{nb_seeds}_seeds_new_methods_no_shared_disturbances_gaussian"
save_path = results_dir / save_name
df = pd.read_csv(save_path)

# Use color palette and markers
palette_sns = sns.color_palette()
palette = {
    'pairwise': palette_sns[0],
    'direct_limvam': palette_sns[1],
    'shica_ml': palette_sns[2],
    'shica_j': palette_sns[3],
    'lingam': palette_sns[4],
    'multi_group_direct_lingam': palette_sns[5],
}
marker_styles = {
    'pairwise': 'o',
    'direct_limvam': 's',
    'shica_ml': 'P',
    'shica_j': 'X',
    'lingam': 'D',
    'multi_group_direct_lingam': '*',
}
marker_sizes = {
    'pairwise': 40,
    'direct_limvam': 36,
    'shica_ml': 40,
    'shica_j': 40,
    'lingam': 32,
    'multi_group_direct_lingam': 56,
}

# Create the plot
fig, ax = plt.subplots(figsize=(6, 3.2))

for method, group_df in df.groupby("ica_algo"):
    # cloud of points
    sns.scatterplot(
        data=group_df,
        x="error_B",
        y="execution_time",
        color=palette.get(method, "gray"),
        alpha=0.2,
        label=None,
        marker=marker_styles.get(method, "o"),
        s=marker_sizes.get(method, 40),
    )
    
    if median_or_mean == "median":
        # median
        x_marker = group_df["error_B"].median()
        y_marker = group_df["execution_time"].median()
        # error bars
        x_low = group_df["error_B"].quantile(0.16)
        x_high = group_df["error_B"].quantile(0.84)
        y_low = group_df["execution_time"].quantile(0.16)
        y_high = group_df["execution_time"].quantile(0.84)
        xerr = [[x_marker - x_low], [x_high - x_marker]]
        yerr = [[y_marker - y_low], [y_high - y_marker]]
    elif median_or_mean == "mean":
        # mean
        log_x = np.log(group_df["error_B"])
        log_y = np.log(group_df["execution_time"])
        x_marker = np.exp(log_x.mean())
        y_marker = np.exp(log_y.mean())
        # quantiles
        x_low = np.exp(log_x.quantile(0.05))
        x_high = np.exp(log_x.quantile(0.95))
        xerr = [[x_marker - x_low], [x_high - x_marker]]
        y_low = np.exp(log_y.quantile(0.05))
        y_high = np.exp(log_y.quantile(0.95))
        yerr = [[y_marker - y_low], [y_high - y_marker]]
        # # std
        # x_std = np.std(group_df["error_B"])
        # xerr = [[x_std], [x_std]]
        # y_std = np.std(group_df["execution_time"])
        # yerr = [[y_std], [y_std]]
        
        
    # markersize
    if method == "direct_limvam":
        markersize = 9
    elif method == "lingam":
        markersize = 8
    elif method == "multi_group_direct_lingam":
        markersize = 14
    else:
        markersize = 10

    # markers and error bars
    ax.errorbar(
        x_marker,
        y_marker,
        xerr=xerr,
        yerr=yerr,
        fmt=marker_styles.get(method),
        markersize=markersize,
        color=palette.get(method, "gray"),
        capsize=5,
        capthick=2,
        elinewidth=2,
        markerfacecolor=palette.get(method, "gray"),
        markeredgecolor='white'
    )

# legend
legend_styles = [
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor=palette_sns[0], markeredgecolor='white', markersize=10),
    Line2D([0], [0], marker='s', color='w',
           markerfacecolor=palette_sns[1], markeredgecolor='white', markersize=9),
    Line2D([0], [0], marker='P', color='w',
           markerfacecolor=palette_sns[2], markeredgecolor='white', markersize=10),
    Line2D([0], [0], marker='X', color='w',
           markerfacecolor=palette_sns[3], markeredgecolor='white', markersize=10),
    Line2D([0], [0], marker='D', color='w',
           markerfacecolor=palette_sns[4], markeredgecolor='white', markersize=8),
    Line2D([0], [0], marker='*', color='w',
           markerfacecolor=palette_sns[5], markeredgecolor='white', markersize=14),
]
# labels = [
#     'LR-DirectLiMVAM', 'CC-DirectLiMVAM', 'ICSL-ML', 'ICSL-J', 'ICA-LiNGAM', 'MultiGroupDirectLiNGAM']
labels = [
    'PairwiseLiMVAM', 'DirectLiMVAM', 'ICA-LiMVAM-ML', 'ICA-LiMVAM-J', 'ICA-LiNGAM', 'MultiGroupDirectLiNGAM']
# fig.legend(
#     legend_styles, labels, bbox_to_anchor=(0.5, 1.15), loc="center",
#     ncol=2, borderaxespad=0., fontsize=fontsize
# )

fig.legend(
    legend_styles, labels,
    bbox_to_anchor=(0.5, 1.15), loc="center", ncol=2, fontsize=fontsize,
    # spacing tweaks:
    handletextpad=0.5,   # marker <-> text
    columnspacing=0.8,   # space between columns
    handlelength=0.6,    # shrink handle length
    # labelspacing=0.2,    # vertical space between rows
    borderpad=0.2,       # padding inside legend box
    borderaxespad=0.0,   # space between legend and axes (you already set this)
    # frameon=False
)

plt.xscale("log")
plt.yscale("log")
xmin, xmax = ax.get_xlim()
# ax.set_xlim([10**(-4.1), 10**0.1])
ax.set_xlim([xmin, 10**3.8])
ax.set_ylim([10**(-0.9), 10**2.2])
plt.xlabel(r"Error on $B^i$", fontsize=fontsize, family="serif")
plt.ylabel("Fitting time (in s)", fontsize=fontsize, family="serif")
ax.tick_params(which='major', bottom=True, left=True, length=4, width=0.8, color='black')
# ax.tick_params(which='minor', bottom=True, left=True, length=2.2, width=0.8, color='black')
ax.grid(linewidth=0.5, alpha=0.5)
plt.tight_layout()

# save figure
figures_dir = simulation_dir / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(figures_dir / f"simulation_execution_time.pdf", bbox_inches="tight")
# plt.show()
