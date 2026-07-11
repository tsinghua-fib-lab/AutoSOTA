# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
from pathlib import Path
from scipy.stats import pearsonr

# %%
# Parameters
n_subjects_batch = 30
n_runs = 50
algo = "pairwise_limvam"
n_arrows = 10

# Load results
expes_dir = Path("/storage/store2/work/aheurteb/LiMVAM/experiments_meg")
results_dir = Path(expes_dir / f"4_results/aparc_sub_{n_subjects_batch}_random_subjects_{n_runs}_runs_{algo}")

P_total = np.load(results_dir / "P_total.npy")
T_total = np.load(results_dir / "T_total.npy")
B_total = np.load(results_dir / "B_total.npy")
with open(results_dir / f"labels.pkl", "rb") as f:
    labels = pickle.load(f)

# %%
# matplotlib style
fontsize = 26
rc = {
    "font.size": fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,
    "font.family": "serif",
}
plt.rcParams.update(rc)

# %%
# Compute Pearson coefficients for the Bi matrices
spearmanr_matrix = np.zeros((n_runs, n_runs))
for i in range(n_runs):
    for j in range(n_runs):
        B1_median = np.median(B_total[i], axis=0)
        B2_median = np.median(B_total[j], axis=0)
        rho, p_value = pearsonr(B1_median.flatten(), B2_median.flatten())
        spearmanr_matrix[i, j] = rho
np.fill_diagonal(spearmanr_matrix, 0)

# Compute the average Pearson coefficient
upper_triangular_values = spearmanr_matrix[np.triu_indices(n_runs, k=1)]
avg_corr = np.mean(upper_triangular_values)

# Plot obtained coefficients
fig, ax = plt.subplots(figsize=(5, 5))
norm = TwoSlopeNorm(vmin=-1, vmax=1, vcenter=0)
im = plt.imshow(spearmanr_matrix, norm=norm, cmap="coolwarm")
# plt.title(f"Average = {avg_corr:.2f}")
# ax.set_xticks(np.arange(n_runs))
# ax.set_yticks(np.arange(n_runs))
ax.set_xlabel("Runs")
ax.set_ylabel("Runs")
ax.set_xticks([])
ax.set_yticks([])

# colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.2)
cbar = fig.colorbar(im, cax=cax, ticks=[-1, 0, 1])
cbar.ax.set_yticklabels(["-1", "0", "1"])

save = True
if save:
    figures_dir = "/storage/store4/work/aheurteb/LiMVAM/experiments_meg/6_figures/"
    plt.savefig(figures_dir + f"pearson_coefs_B_{algo}.pdf", bbox_inches="tight")
plt.show()

print(f"Average = {avg_corr:.2f}")

# %%
