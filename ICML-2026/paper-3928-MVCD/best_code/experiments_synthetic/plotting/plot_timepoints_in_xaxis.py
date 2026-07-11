import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator
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
n_metrics = 7
nb_gaussian_disturbances_list = [4, 0, 2]
shared_permutation = True
errors = ["error_B", "error_P_exact"]  # "error_P_spearmanr"
if shared_permutation:
    error_names = [r"Error on $B^i$", r"Error rate on $P$"]
    # error_names = [r"Error on $B^i$", r"Error on $T^i$", "Spearman's rank\ncorrelation on" + r" $P$"]
else:
    error_names = [r"Error on $B^i$", r"Error rate on $P^i$"]
    # error_names = [r"Error on $B^i$", r"Error on $T^i$", "Spearman's rank\ncorrelation on" + r" $P^i$"]
titles = ["Gaussian", "Non-Gaussian", "Half-G / Half-NG"]
estimator = "mean"
# labels = ['LR-DirectLiMVAM', 'CC-DirectLiMVAM', 'ICSL-ML', 'ICSL-J', 'ICA-LiNGAM', 'MultiGroupDirectLiNGAM']
labels = [
    'PairwiseLiMVAM', 'DirectLiMVAM', 'ICA-LiMVAM-ML', 'ICA-LiMVAM-J', 'ICA-LiNGAM', 'MultiGroupDirectLiNGAM']

# read dataframe
results_dir = "/Users/ambroiseheurtebise/Desktop/LiMVAM/experiments_synthetic/results/results_timepoints_in_xaxis/"
# results_dir = "/storage/store4/work/aheurteb/LiMVAM/experiments_synthetic/results/results_timepoints_in_xaxis/"
if shared_permutation:
    parent_dir = "shared_P"
else:
    parent_dir = "multiple_Pi"
save_name = f"/DataFrame_with_{nb_seeds}_seeds"
save_path = results_dir + parent_dir + save_name
df = pd.read_csv(save_path)

# remove MVICA-LiNGAM and MV-NOTEARS curves
filtered_df = df[(df["ica_algo"] != "multiviewica") & (df["ica_algo"] != "mv_notears")]

# change the curves order
hue_order = ["pairwise", "direct_limvam", "shica_ml", "shica_j", "lingam", "multi_group_direct_lingam"]

# subplots
fig, axes = plt.subplots(2, 3, figsize=(12, 4.5), sharex="col", sharey="row")
for i, ax in enumerate(axes.flat):
    # number of Gaussian sources; one for each of the 3 columns
    nb_gaussian_disturbances = nb_gaussian_disturbances_list[i % 3]
    data = filtered_df[filtered_df["nb_gaussian_disturbances"] == nb_gaussian_disturbances]
    # error; one for each of the 3 rows
    y = errors[i // 3]
    # subplot
    dashes = ['', (5, 5), (2, 2), (2, 2), (2, 2), (2, 2)]
    palette = sns.color_palette(None, n_colors=len(hue_order))

    if i // 3 == 0:
        # ---- FIRST ROW: compute median + 95% CI in log space and draw bands ----
        dm = data.copy()
        dm["_logy"] = np.log10(dm[y])

        for k, method in enumerate(hue_order):
            dmk = dm[dm["ica_algo"] == method]
            if dmk.empty:
                continue

            g = (
                dmk.groupby("n")["_logy"]
                   .agg(med="median",
                        lo=lambda x: np.quantile(x, 0.25),
                        hi=lambda x: np.quantile(x, 0.75))
                   .reset_index()
                   .sort_values("n")
            )

            x_vals = g["n"].to_numpy()
            y_med  = 10 ** g["med"].to_numpy()
            y_lo   = 10 ** g["lo"].to_numpy()
            y_hi   = 10 ** g["hi"].to_numpy()

            # CI band (like seaborn)
            ax.fill_between(x_vals, y_lo, y_hi,
                            color=palette[k], alpha=0.2, linewidth=0)

            # median line
            ax.plot(
                x_vals, y_med,
                color=palette[k], linewidth=2.5,
                linestyle='-' if dashes[k] == '' else (0, dashes[k]),
                label=method
            )
    else:
        # ---- SECOND ROW: keep seaborn with default (linear y) CI ----
        sns.lineplot(
            data=data, x="n", y=y, linewidth=2.5, hue="ica_algo", ax=ax,
            errorbar=('ci', 95),
            hue_order=hue_order, style_order=hue_order, style="ica_algo",
            dashes=dashes, legend=False
        )

    # set axis in logscale, except for the yaxis of the middle row
    ax.set_xscale("log")
    if i // 3 != 1:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=None, numticks=10))
        ax.set_yticks([1e-1, 1e-3])
    if i // 3 == 1:
        ax.set_yticks([1, 0.5, 0])
    # correct ylim in first and second rows
    if i == 0:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, 5)
    # ylabel
    ax.set_xlabel("")
    ax.set_ylabel("")
    if i == 0:
        ylabel1 = ax.set_ylabel(error_names[i // 3])
        ylabel1.set_position((0, 0.55))
    if i == 3:
        ylabel2 = ax.set_ylabel(error_names[i // 3])
        ylabel2.set_position((0, 0.4))
    # title, grid, and legend
    if i // 3 == 0:
        ax.set_title(titles[i], fontsize=fontsize)
    ax.grid(which='both', linewidth=0.5, alpha=0.5)
label = fig.supxlabel("Number of samples", fontsize=fontsize)
label.set_position((0.5, 0.1))
plt.gcf().align_labels()
plt.tight_layout()
plt.subplots_adjust(hspace=0.15)
# legend
palette = sns.color_palette()
legend_styles = [
    Line2D([0], [0], color=palette[0], linewidth=2.5, linestyle=(3, (4, 3))),
    Line2D([0], [0], color=palette[1], linewidth=2.5, linestyle=(0, (4, 4))),
    Line2D([0], [0], color=palette[2], linewidth=2.5, linestyle=(0, (2, 2))),
    Line2D([0], [0], color=palette[3], linewidth=2.5, linestyle=(0, (2, 2))),
    Line2D([0], [0], color=palette[4], linewidth=2.5, linestyle=(0, (2, 2))),
    Line2D([0], [0], color=palette[5], linewidth=2.5, linestyle=(0, (2, 2))),
]
# fig.legend(
#     legend_styles, labels, bbox_to_anchor=(0.5, 1.05), loc="center",
#     ncol=3, borderaxespad=0., fontsize=fontsize
# )
fig.legend(
    legend_styles, labels, bbox_to_anchor=(0.98, 0.57), loc="center left",
    ncol=1, borderaxespad=0., fontsize=fontsize, handletextpad=0.4, handlelength=1.5,
    labelspacing=0.7,
)

# save figure
figures_dir = Path("/Users/ambroiseheurtebise/Desktop/LiMVAM/experiments_synthetic/figures")
# figures_dir = Path("/storage/store4/work/aheurteb/LiMVAM/experiments_synthetic/figures")
figures_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(figures_dir / f"simulation_{parent_dir}.pdf", bbox_inches="tight")
# plt.show()
