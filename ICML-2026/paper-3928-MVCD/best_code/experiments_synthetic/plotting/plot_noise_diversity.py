import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")


# matplotlib style
fontsize = 19
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

# read dataframe
results_dir = "/Users/ambroiseheurtebise/Desktop/LiMVAM/experiments_synthetic/results/results_noise_diversity/"
# results_dir = "/storage/store4/work/aheurteb/LiMVAM/experiments_synthetic/results/results_noise_diversity/"
save_name = f"DataFrame_with_{nb_seeds}_seeds_time_and_scale"
save_path = results_dir + save_name
df = pd.read_csv(save_path)

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
# labels = ['PRaLiNE', 'MICaDo-ML', 'MICaDo-J']
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

# plot
fig, ax = plt.subplots(figsize=(6, 2.7))

for method in hue_order:
    data = df[df["ica_algo"] == method]
    sns.lineplot(
        data=data, x="nb_equal_variances", y=metric, linewidth=2.5,
        estimator=np.median, errorbar=('ci', 95),
        color=colors[method],
        dashes=dashes[hue_order.index(method)],
        marker=marker_styles[method],
        markersize=marker_sizes[method],
        label=method
    )
# sns.lineplot(
#     data=df, x="nb_equal_variances", y=metric, linewidth=2.5, hue="ica_algo", estimator=np.median,
#     errorbar=('ci', 95), hue_order=hue_order, style_order=hue_order, style="ica_algo",
#     dashes=dashes, markers=True)
ax.set_yscale("log")
ax.set_xticks(np.arange(6))
xlabel = r"Number of views $i$ s.t. $\frac{\Sigma^i_{jj}}{(D^i_{jj})^2} = \frac{\Sigma^i_{j'j'}}{(D^i_{j'j'})^2}$"
ax.set_xlabel(xlabel, fontsize=fontsize)
ax.xaxis.set_label_coords(0.5, -0.17)
ax.set_ylabel(metric_name, fontsize=fontsize)
ax.yaxis.set_label_coords(-0.155, 0.5)
ax.grid(which='major', linewidth=0.5, alpha=0.5)
ax.get_legend().remove()

# legend
legend_styles = [
    Line2D([0], [0], color=palette[0], linewidth=2.5, linestyle='-', marker='o', 
           markeredgecolor="white", markersize=6),
    Line2D([0], [0], color=palette[2], linewidth=2.5, linestyle=(0, (2, 2)), marker='P', 
           markeredgecolor="white", markersize=6),
    Line2D([0], [0], color=palette[3], linewidth=2.5, linestyle=(0, (2, 2)), marker='X', 
           markeredgecolor="white", markersize=6),
]
fig.legend(
    legend_styles, labels, bbox_to_anchor=(0.43, 1.05), loc="center",
    ncol=2, fontsize=fontsize,
    handletextpad=0.7,   # marker <-> text
    columnspacing=0.7,   # space between columns
    # handlelength=0.6,    # shrink handle length
    # # labelspacing=0.2,    # vertical space between rows
    # borderpad=0.2,       # padding inside legend box
    # borderaxespad=0.0,   # space between legend and axes (you already set this)
    # # frameon=False
)

# # caption
# caption = (
#     "Caption: Data are generated with $m=5$ views and $p=4$\ndisturbances, "
#     "consisting of 2 Gaussian and 2 non-Gaussian \n"
#     "disturbances. We vary the number of views in which the \n"
#     "2 Gaussian disturbances have equal variances. The error \n"
#     "increases abruptly only when variances are equal in all \n"
#     "views, which justifies Assumption 1."
# )
# fig.text(0.5, -0.42, caption, ha='center', va='center', fontsize=fontsize)

# save figure
figures_dir = Path("/Users/ambroiseheurtebise/Desktop/LiMVAM/experiments_synthetic/figures")
# figures_dir = Path("/storage/store2/work/aheurteb/LiMVAM/experiments_synthetic/figures")
plt.savefig(figures_dir / f"simulation_noise_diversity.pdf", bbox_inches="tight")
# plt.show()
