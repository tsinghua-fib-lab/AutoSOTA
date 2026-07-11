import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")


# matplotlib style
fontsize = 15
rc = {
    "font.size": fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,
    "font.family": "serif",
}
plt.rcParams.update(rc)

# parameters 
nb_seeds = 100
metric = "error_B"  # or "error_T", "error_P_exact", "error_P_spearmanr", "amari_distance"

# read dataframe
results_dir = "/storage/store4/work/aheurteb/LiMVAM/experiments_synthetic/results/results_with_non_linear_activation/"
save_name = f"DataFrame_with_{nb_seeds}_seeds"
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
labels = ['PRaLiNE', 'MICaDo-ML', 'MICaDo-J', 'ICA-LiNGAM', 'MultiGroupDirectLiNGAM']
dashes = ['', '', '', (2, 2), (2, 2)]
hue_order = ["pairwise", "shica_ml", "shica_j", "lingam", "multi_group_direct_lingam"]

# plot
fig, ax = plt.subplots(figsize=(6, 3))
sns.lineplot(
    data=df, x="non_linearity_alpha", y=metric, linewidth=2.5, hue="ica_algo", estimator=np.median,
    errorbar=('ci', 95), hue_order=hue_order, style_order=hue_order, style="ica_algo",
    dashes=dashes, markers=True)
ax.set_yscale("log")
xlabel = r"Parameter $\alpha$ of the non-linear activation $f$"
ax.set_xlabel(xlabel, fontsize=fontsize)
ax.xaxis.set_label_coords(0.5, -0.17)
ax.set_ylabel(metric_name, fontsize=fontsize)
ax.yaxis.set_label_coords(-0.155, 0.5)
ax.grid(which='both', linewidth=0.5, alpha=0.5)
ax.get_legend().remove()

# legend
palette = sns.color_palette()
legend_styles = [
    Line2D([0], [0], color=palette[0], linewidth=2.5, linestyle='-', marker='o', 
           markeredgecolor="white", markersize=6),
    Line2D([0], [0], color=palette[1], linewidth=2.5, linestyle='-', marker='X', 
           markeredgecolor="white", markersize=7),
    Line2D([0], [0], color=palette[2], linewidth=2.5, linestyle='-', marker='s', 
           markeredgecolor="white", markersize=6),
        Line2D([0], [0], color=palette[3], linewidth=2.5, linestyle='--', marker='P', 
           markeredgecolor="white", markersize=6),
    Line2D([0], [0], color=palette[4], linewidth=2.5, linestyle='--', marker='D', 
           markeredgecolor="white", markersize=5),
]
fig.legend(
    legend_styles, labels, bbox_to_anchor=(0.5, 1.03), loc="center",
    ncol=3, borderaxespad=0., fontsize=fontsize
)

# caption
caption = (
    # "Caption: Data are generated with $m=6$ views and $p=5$ disturbances,\n"
    # "consisting of sub-Gaussian, Gaussian, and super-Gaussian disturbances.\n"
    "Caption: We apply a non-linear function $f$ to the observations,\n"
    r"where $f(x) = (1 - \alpha) x + \alpha (2$ logistic$(x) - 1)$." + "\n"
    r"Recall that the observations are $\boldsymbol{x}^i = (\boldsymbol{I} - \boldsymbol{B}^i)^{-1}$"
    r"$(\boldsymbol{D}^i \boldsymbol{s} + \boldsymbol{n}^i)$."
)
fig.text(0.5, -0.3, caption, ha='center', va='center', fontsize=fontsize)

# save figure
figures_dir = Path("/storage/store4/work/aheurteb/LiMVAM/experiments_synthetic/figures")
plt.savefig(figures_dir / f"simulation_with_non_linear_activation.pdf", bbox_inches="tight")
plt.show()
