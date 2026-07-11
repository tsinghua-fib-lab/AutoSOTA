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
nb_gaussian_disturbances_list = [4, 0, 2]
errors = ["error_B", "error_T", "error_P_exact"]
# errors = ["error_B", "error_T", "error_P_spearmanr"]
error_names = [r"Error on $B^i$", r"Error on $T^i$", r"Error rate on $P$"]
# error_names = [r"Error on $B^i$", r"Error on $T^i$", "Spearman's rank\ncorrelation on" + r" $P$"]
titles = ["Gaussian", "Non-Gaussian", "Half-G / Half-NG"]
estimator = "mean"
labels = ['PRaLiNE', 'MICaDo-ML', 'MICaDo-J', 'ICA-LiNGAM', 'MultiGroupDirectLiNGAM']

# read dataframe
results_dir = "/storage/store2/work/aheurteb/LiMVAM/experiments_synthetic/results/"
parent_dir = "results_noise_in_xaxis/"
save_name = f"DataFrame_with_{nb_seeds}_seeds_time_and_scale"
save_path = results_dir + parent_dir + save_name
df = pd.read_csv(save_path)

# change the curves order
hue_order = ["pairwise", "shica_ml", "shica_j", "lingam", "multi_group_direct_lingam"]

# subplots
fig, axes = plt.subplots(3, 3, figsize=(12, 6), sharex="col", sharey="row")
for i, ax in enumerate(axes.flat):
    # number of Gaussian sources; one for each of the 3 columns
    nb_gaussian_disturbances = nb_gaussian_disturbances_list[i % 3]
    data = df[df["nb_gaussian_disturbances"] == nb_gaussian_disturbances]
    # error; one for each of the 3 rows
    y = errors[i // 3]
    # subplot
    if i // 3 != 2 and estimator == "median":
        sns.lineplot(
            data=data, x="noise_level", y=y, linewidth=2.5, hue="ica_algo", estimator=np.mean,
            ax=ax, errorbar=lambda x: (np.quantile(x, 0.025), np.quantile(x, 0.975)),
            hue_order=hue_order, style_order=hue_order, style="ica_algo",
            dashes=['', '', '', (2, 2), (2, 2)])
    else:
        sns.lineplot(
            data=data, x="noise_level", y=y, linewidth=2.5, hue="ica_algo", ax=ax,
            errorbar=('ci', 95), hue_order=hue_order, style_order=hue_order,
            style="ica_algo", dashes=['', '', '', (2, 2), (2, 2)])
    # set axis in logscale, except for the yaxis of the middle row
    ax.set_xscale("log")
    if i // 3 != 2:
        ax.set_yscale("log")
    # correct ylim in the second row
    if i == 0 or i == 3:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, 1e5)
    # ylabel
    ax.set_xlabel("")
    ax.set_ylabel("")
    if i % 3 == 0:
        ax.set_ylabel(error_names[i // 3])
    # title, grid, and legend
    if i // 3 == 0:
        ax.set_title(titles[i], fontsize=fontsize)
    ax.grid(which='both', linewidth=0.5, alpha=0.5)
    ax.get_legend().remove()
label = fig.supxlabel("Noise level", fontsize=fontsize)
label.set_position((0.5, 0.055))
plt.gcf().align_labels()
plt.tight_layout()
plt.subplots_adjust(hspace=0.15)
# legend
palette = sns.color_palette()[:5]
legend_styles = [
    Line2D([0], [0], color=palette[0], linewidth=2.5, linestyle='-'),
    Line2D([0], [0], color=palette[1], linewidth=2.5, linestyle='-'),
    Line2D([0], [0], color=palette[2], linewidth=2.5, linestyle='-'),
    Line2D([0], [0], color=palette[3], linewidth=2.5, linestyle='--'),
    Line2D([0], [0], color=palette[4], linewidth=2.5, linestyle='--'),
]
fig.legend(
    legend_styles, labels, bbox_to_anchor=(0.5, 1.05), loc="center",
    ncol=3, borderaxespad=0., fontsize=fontsize
)

# save figure
figures_dir = Path("/storage/store2/work/aheurteb/LiMVAM/experiments_synthetic/figures")
figures_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(figures_dir / "simulation_noise_in_xaxis.pdf", bbox_inches="tight")
plt.show()
