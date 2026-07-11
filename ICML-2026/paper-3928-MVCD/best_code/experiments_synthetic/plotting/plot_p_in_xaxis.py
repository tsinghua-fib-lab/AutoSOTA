import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")


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
nb_seeds = 50
metric = "error_B"  # or "error_T", "error_P_exact", "error_P_spearmanr", "amari_distance"
beta1 = 1.5
beta2 = 2.5

# read dataframe
beta1_str = str(beta1).replace('.', '')
beta2_str = str(beta2).replace('.', '')
# results_dir = "/storage/store4/work/aheurteb/LiMVAM/experiments_synthetic/results/results_p_in_xaxis/"
results_dir = "/Users/ambroiseheurtebise/Desktop/LiMVAM/experiments_synthetic/results/results_p_in_xaxis/"
save_name = f"DataFrame_with_{nb_seeds}_seeds_beta_{beta1_str}_{beta2_str}_no_shared_disturbances"
save_path = results_dir + save_name
df = pd.read_csv(save_path)

# number of views and disturbances
m_list = np.sort(np.unique(df["m"]))
p_list = np.sort(np.unique(df["p"]))

# metric name
if metric == "error_B":
    metric_name = r"Error on $B^i$"
elif metric == "error_T":
    metric_name = r"Error on $T^i$"
elif metric == "error_P_exact":
    metric_name = r"Error on $P$"
elif metric == "error_P_spearmanr":
    metric_name = "Spearman's rank\ncorrelation on" + r" $P$"
elif metric == "amari_distance":
    metric_name = "Amari distance"

# labels, dashes and curves order
labels = [
    'PairwiseLiMVAM', 'DirectLiMVAM', 'ICA-LiMVAM-ML', 'ICA-LiMVAM-J', 'ICA-LiNGAM', 'MultiGroupDirectLiNGAM']
dashes = ['', '', (2, 2), (2, 2), (2, 2), (2, 2)]
hue_order = ["pairwise", "direct_limvam", "shica_ml", "shica_j", "lingam", "multi_group_direct_lingam"]

marker_styles = {
    'pairwise': 'o',
    'direct_limvam': 's',
    'shica_ml': 'P',
    'shica_j': 'X',
    'lingam': 'D',
    'multi_group_direct_lingam': '*',
}
marker_sizes = {
    'pairwise': 5,
    'direct_limvam': 4.5,
    'shica_ml': 5,
    'shica_j': 5,
    'lingam': 4,
    'multi_group_direct_lingam': 7,
}

# remove MVICA LiNGAM and MV-NOTEARS curves
filtered_df = df[(df["ica_algo"] != "multiviewica") & (df["ica_algo"] != "mv_notears")]

# add a column in logscale
filtered_df["_log_metric"] = np.log10(filtered_df[metric])

# subplots
fig, axes = plt.subplots(2, 3, figsize=(12, 4.8), sharex=True, sharey=True)
palette = sns.color_palette()

for i, ax in enumerate(axes.flat):
    m = m_list[i]
    data = filtered_df[filtered_df["m"] == m].copy()

    # work in log10 space for the metric
    data["_logy"] = np.log10(data[metric])

    for method_idx, method in enumerate(hue_order):
        dm = data[data["ica_algo"] == method]
        if dm.empty:
            continue

        # group by x (p) and compute median & CI in log space
        g = (
            dm.groupby("p")["_logy"]
              .agg(med="median",
                   lo=lambda x: np.quantile(x, 0.25),
                   hi=lambda x: np.quantile(x, 0.75))
              .reset_index()
              .sort_values("p")
        )

        x = g["p"].to_numpy()
        y_med = 10 ** g["med"].to_numpy()
        y_lo  = 10 ** g["lo"].to_numpy()
        y_hi  = 10 ** g["hi"].to_numpy()

        ax.fill_between(x, y_lo, y_hi,
                    color=palette[method_idx], alpha=0.2, linewidth=0)
        
        # line + markers
        ax.plot(
            x, y_med,
            linewidth=2.5,
            color=palette[method_idx],
            linestyle='-' if dashes[method_idx] == '' else (0, dashes[method_idx]),
            marker=marker_styles[method],
            markersize=marker_sizes[method],
            markeredgecolor="white",
            label=method
        )

    ax.set_yscale("log")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(f"m = {m}", fontsize=fontsize)
    ax.grid(which='both', linewidth=0.5, alpha=0.5)
    ax.set_xticks(p_list)

xlabel = fig.supxlabel(r"Number of disturbances $p$", fontsize=fontsize)
ylabel = fig.supylabel(metric_name, fontsize=fontsize)
xlabel.set_position((0.5, 0.09))
ylabel.set_position((0.04, 0.5))

for ax in axes.flat:
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(max(ymin, 1e-4), 1e3)

plt.tight_layout()
plt.subplots_adjust(hspace=0.3, wspace=0.1)

# legend
legend_styles = [
    Line2D([0], [0], color=palette[0], linewidth=2.5, marker='o', markersize=6, markeredgecolor="white", linestyle='-'),
    Line2D([0], [0], color=palette[1], linewidth=2.5, marker='s', markersize=5.5, markeredgecolor="white", linestyle='-'),
    Line2D([0], [0], color=palette[2], linewidth=2.5, marker='P', markersize=6, markeredgecolor="white", linestyle=(0, (2, 2))),
    Line2D([0], [0], color=palette[3], linewidth=2.5, marker='X', markersize=6, markeredgecolor="white", linestyle=(0, (2, 2))),
    Line2D([0], [0], color=palette[4], linewidth=2.5, marker='D', markersize=5, markeredgecolor="white", linestyle=(0, (2, 2))),
    Line2D([0], [0], color=palette[5], linewidth=2.5, marker='*', markersize=9, markeredgecolor="white", linestyle=(0, (2, 2))),
]
fig.legend(
    legend_styles, labels, bbox_to_anchor=(0.5, 1.05), loc="center",
    ncol=3, borderaxespad=0., fontsize=fontsize
)


# # caption
# caption = (
#     "Caption: Separation performance of ICA-LiNGAM, MultiGroupDirectLiNGAM, and two versions \n"
#     "of our method when we vary the number of views $m$ and disturbances $p$. The disturbances are \n"
#     "evenly divided into sub-Gaussian, Gaussian, and super-Gaussian groups."
# )
# fig.text(0.5, -0.05, caption, ha='center', va='center', fontsize=fontsize)

# save figure
# figures_dir = Path("/storage/store4/work/aheurteb/LiMVAM/experiments_synthetic/figures")
figures_dir = Path("/Users/ambroiseheurtebise/Desktop/LiMVAM/experiments_synthetic/figures")
plt.savefig(figures_dir / f"simulation_p_in_xaxis.pdf", bbox_inches="tight")
# plt.show()
