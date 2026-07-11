import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")


# matplotlib style
fontsize = 16
rc = {
    "font.size": fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,
    "font.family": "serif",
    "font.serif": ["Times"],
}
plt.rcParams.update(rc)

# parameters 
nb_seeds = 30
metric = "error_P_spearmanr"  # or "error_B", "error_P_exact", "amari_distance"

# read dataframe
# results_dir = "/storage/store4/work/aheurteb/LiMVAM/experiments_synthetic/results/results_sparsity_of_Ti/"
results_dir = "/Users/ambroiseheurtebise/Desktop/LiMVAM/experiments_synthetic/results/results_sparsity_of_Ti/"
save_name = f"DataFrame_with_{nb_seeds}_seeds_time_and_scale"
save_path = results_dir + save_name
df = pd.read_csv(save_path)

# metric names
if metric == "error_B" or metric == "error_B_abs":
    metric_name = r"Error on $B^i$"
elif metric == "error_T" or metric == "error_T_abs":
    metric_name = r"Error on $T^i$"
elif metric == "error_P_exact":
    metric_name = r"Error on $P$"
elif metric == "error_P_spearmanr":
    metric_name = "Spearman's rank\ncorrelation of the\ncausal ordering(s)"
elif metric == "amari_distance":
    metric_name = "Amari distance"

# labels, dashes, curves order and titles
labels = ['PairwiseLiMVAM', 'ICA-LiMVAM-ML', 'ICA-LiMVAM-J']
dashes = ['', (2, 2), (2, 2)]
hue_order = ["pairwise", "shica_ml", "shica_j"]
marker_styles = {
    'pairwise': 'o',
    'shica_ml': 'P',
    'shica_j': 'X',
}
marker_sizes = {
    'pairwise': 5,
    'shica_ml': 5,
    'shica_j': 5,
}
palette = sns.color_palette()
colors = {
    'pairwise': palette[0],
    'shica_ml': palette[2],
    'shica_j': palette[3],
}

# # Use color palette
# palette_sns = sns.color_palette()
# palette = {
#     'pairwise': palette_sns[0],
#     'shica_ml': palette_sns[1],
#     'shica_j': palette_sns[2],
# }

# subplots
fig, axes = plt.subplots(1, 2, figsize=(8, 2.8), sharex=True, sharey=True)

# first subplot
data1 = df[(df["shared_causal_ordering"] == 0) & (df["ica_algo"] != "pairwise")]

for method in hue_order:
    data = data1[data1["ica_algo"] == method]
    sns.lineplot(
        data=data, x="nb_zeros_Ti", y=metric, linewidth=2.5,
        estimator=np.median, errorbar=('ci', 95),
        color=colors[method],
        dashes=dashes[hue_order.index(method)],
        marker=marker_styles[method],
        markersize=marker_sizes[method],
        label=method,
        ax=axes[0]
    )

# sns.lineplot(
#     data=data1, x="nb_zeros_Ti", y=metric, linewidth=2.5,
#     hue="ica_algo", style="ica_algo", style_order=["shica_ml", "shica_j"],
#     dashes=['', ''], markers={"shica_ml": "X", "shica_j": "s"}, 
#     palette={"shica_ml": palette_sns[1], "shica_j": palette_sns[2]},
#     estimator=np.median, errorbar=('ci', 95), ax=axes[0])
axes[0].set_xlabel("")
ylabel = axes[0].set_ylabel(metric_name)
axes[0].yaxis.set_label_coords(-0.12, 0.5)
axes[0].set_title("Multiple causal orderings", fontsize=fontsize)
axes[0].grid(which='both', linewidth=0.5, alpha=0.5)
axes[0].get_legend().remove()

# second subplot
data2 = df[df["shared_causal_ordering"] == 1]

for method in hue_order:
    data = data2[data2["ica_algo"] == method]
    sns.lineplot(
        data=data, x="nb_zeros_Ti", y=metric, linewidth=2.5,
        estimator=np.median, errorbar=('ci', 95),
        color=colors[method],
        dashes=dashes[hue_order.index(method)],
        marker=marker_styles[method],
        markersize=marker_sizes[method],
        label=method,
        ax=axes[1]
    )

# sns.lineplot(
#     data=data2, x="nb_zeros_Ti", y=metric, linewidth=2.5, hue="ica_algo", estimator=np.median,
#     errorbar=('ci', 95), ax=axes[1], hue_order=hue_order, style_order=hue_order, style="ica_algo",
#     dashes=['', '', ''], markers=True)
axes[1].set_xlabel("")
axes[1].set_title("Shared causal ordering", fontsize=fontsize)
axes[1].grid(which='both', linewidth=0.5, alpha=0.5)
axes[1].get_legend().remove()
axes[1].set_yticks([0, 1])
axes[1].set_yticklabels([0, 1])

# ax.set_ylim(-1.1, 1.1)
# ax.set_yticks([-1, 0, 1])
# ax.set_yticklabels([-1, 0, 1])
xlabel = fig.supxlabel(r"Number of sparse entries in each $T^i$", fontsize=fontsize)
xlabel.set_position((0.5, 0.16))
plt.tight_layout()
plt.subplots_adjust(hspace=0.15)

# legend
palette = sns.color_palette()
legend_styles = [
    Line2D([0], [0], color=palette[0], linewidth=2.5, linestyle='-', marker='o', 
           markeredgecolor="white", markersize=6),
    Line2D([0], [0], color=palette[2], linewidth=2.5, linestyle=(0, (2, 2)), marker='P', 
           markeredgecolor="white", markersize=6),
    Line2D([0], [0], color=palette[3], linewidth=2.5, linestyle=(0, (2, 2)), marker='X', 
           markeredgecolor="white", markersize=6),
]
fig.legend(
    legend_styles, labels, bbox_to_anchor=(0.5, 1.03), loc="center",
    ncol=3, borderaxespad=0., fontsize=fontsize
)

# # caption
# caption = (
#     "Caption: Data are generated with $m=8$ views and $p=6$ disturbances, consisting \n"
#     "of 2 sub-Gaussian, 2 Gaussian, and 2 super-Gaussian disturbances. The x-axis \n"
#     "represents the number of sparse entries in the strictly lower triangular part of \n"
#     "each " + r"$T^i$" + ", while the y-axis shows the Spearman's rank correlation between true and \n"
#     "estimated causal orderings. Results justify Assumptions 2 and 3, since recovering \n"
#     "the causal ordering is significantly easier in the shared causal ordering scenario \n"
#     "(Assumption 3) than in the multiple causal orderings scenario (Assumption 2)."
# )
# fig.text(0.5, -0.2, caption, ha='center', va='center', fontsize=fontsize)

# save figure
# figures_dir = Path("/storage/store2/work/aheurteb/LiMVAM/experiments_synthetic/figures")
figures_dir = Path("/Users/ambroiseheurtebise/Desktop/LiMVAM/experiments_synthetic/figures")
plt.savefig(figures_dir / f"simulation_sparsity_of_Ti.pdf", bbox_inches="tight")
# plt.show()
