import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")


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
nb_seeds = 50
errors = ["error_B", "error_T", "error_P_exact"]
error_names = [r"Error on $B^i$", r"Error on $T^i$", r"Error rate on $P$"]
estimator = "mean"
labels = ['MICaDo-ML', 'MICaDo-J', 'ICA-LiNGAM', 'MultiGroupDirectLiNGAM', 'MICaDo-MVICA']
hue_order = ["shica_ml", "shica_j", "lingam", "multi_group_direct_lingam", "multiviewica"]

# read dataframe
results_dir = "/Users/ambroiseheurtebise/Desktop/LiMVAM/experiments_synthetic/results/"
# results_dir = "/storage/store2/work/aheurteb/MICaDo/experiments_synthetic/results/"
parent_dir = "results_gaussian_disturbances_in_xaxis/"
save_name = f"DataFrame_with_{nb_seeds}_seeds_and_7_metrics"
save_path = results_dir + parent_dir + save_name
df = pd.read_csv(save_path)

# subplots
fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharex=True)
for i, ax in enumerate(axes):
    y = errors[i]
    sns.lineplot(
        data=df, x="nb_gaussian_disturbances", y=y, linewidth=2.5, hue="ica_algo", ax=ax, 
        errorbar=('ci', 95), hue_order=hue_order, style_order=hue_order, style="ica_algo",
        dashes=['', '', (2, 2), (2, 2), ''])
    if i != 2:
        ax.set_yscale("log")
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, 1e3)
    ax.set_xticks(np.unique(df["nb_gaussian_disturbances"]))
    ax.set_xticklabels(np.unique(df["nb_gaussian_disturbances"]))
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(error_names[i], fontsize=fontsize)
    ax.grid(which='both', linewidth=0.5, alpha=0.5)
    ax.get_legend().remove()
xlabel = fig.supxlabel("Number of Gaussian disturbances", fontsize=fontsize)
xlabel.set_position((0.5, 0.09))
ylabel = fig.supylabel("Errors", fontsize=fontsize)
ylabel.set_position((0.025, 0.5))
plt.tight_layout()
plt.subplots_adjust(hspace=0.15)
# legend
palette = sns.color_palette()[:5]
legend_styles = [
    Line2D([0], [0], color=palette[0], linewidth=2.5, linestyle='-'),
    Line2D([0], [0], color=palette[1], linewidth=2.5, linestyle='-'),
    Line2D([0], [0], color=palette[2], linewidth=2.5, linestyle='--'),
    Line2D([0], [0], color=palette[3], linewidth=2.5, linestyle='--'),
    Line2D([0], [0], color=palette[4], linewidth=2.5, linestyle='-'),
]
fig.legend(
    legend_styles, labels, bbox_to_anchor=(0.5, 1.07), loc="center",
    ncol=3, borderaxespad=0., fontsize=fontsize
)

# save figure
figures_dir = Path("/Users/ambroiseheurtebise/Desktop/LiMVAM/experiments_synthetic/figures/")
# figures_dir = Path("/storage/store2/work/aheurteb/MICaDo/experiments_synthetic/figures")
figures_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(figures_dir / f"simulation_gaussian_disturbances.pdf", bbox_inches="tight")
plt.show()
