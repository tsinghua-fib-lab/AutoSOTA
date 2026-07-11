# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import pickle
from pathlib import Path


# %%
# Parameters
n_arrows = 10
algo = "pairwise_limvam"
plot_which_B = "median"  # choose among "median", "positive_normalized", and "random"

# Load results
expes_dir = Path("/storage/store2/work/aheurteb/LiMVAM/experiments_meg")
results_dir = Path(expes_dir / f"4_results/aparc_sub_98_subjects_{algo}")

P = np.load(results_dir / "P.npy")
T = np.load(results_dir / "T.npy")
B = np.load(results_dir / "B.npy")
with open(results_dir / f"labels.pkl", "rb") as f:
    labels = pickle.load(f)

# %%
# Plot average matrix T (should be lower triangular)
plt.imshow(np.mean(np.abs(T), axis=0))
plt.colorbar()
plt.title("Average absolute value lower triangular matrix T")
plt.show()

# %%
# Define matrix B_avg
if plot_which_B == "median":
    # Get the median matrix B over subjects
    B_avg = np.median(B, axis=0)
elif plot_which_B == "positive_normalized":    
    # Normalize matrices Bi: divide each Bi by its max in absolute value
    B_maxs = np.array([np.max(Bi_abs) for Bi_abs in np.abs(B)])[:, np.newaxis, np.newaxis]
    B_norm = B / B_maxs
    B_avg = np.mean(B_norm, axis=0)
elif plot_which_B == "random":
    # Pick one of the 98 subjects
    n_subjects = len(B)
    random_state = 42
    rng = np.random.RandomState(random_state)
    idx = rng.randint(0, n_subjects)
    B_avg = B[idx]

# %%
# Plot average normalized adjacency matrix
fig, ax = plt.subplots()
norm = TwoSlopeNorm(vmin=np.min(B_avg), vmax=np.max(B_avg), vcenter=0)
plt.imshow(B_avg, norm=norm, cmap="coolwarm")
plt.colorbar()
plt.title("Average normalized adjacency matrix B")
label_names = [label.name for label in labels]
ax.set_xticks(np.arange(len(label_names)))
ax.set_yticks(np.arange(len(label_names)))
ax.set_xticklabels(label_names, rotation=45, ha="right")
ax.set_yticklabels(label_names)
plt.show()

# %%
# Only keep the most important effects
M = np.abs(B_avg)
indices = np.argsort(M.flatten())[::-1]
ranked_flat = np.zeros(M.size)
ranked_flat[indices] = np.arange(M.size)
B_avg_rank = ranked_flat.reshape(M.shape)
B_avg_subset = B_avg * (B_avg_rank < n_arrows)

homogenise = False
if homogenise:
    B_avg_subset = np.sign(B_avg_subset) * np.sqrt(np.abs(B_avg_subset))
fig, ax = plt.subplots()
norm = TwoSlopeNorm(vmin=min(-1, np.min(B_avg_subset)), vmax=max(1, np.max(B_avg_subset)), vcenter=0)
plt.imshow(B_avg_subset, norm=norm, cmap="coolwarm")
plt.colorbar()
plt.title(f"Average adjacency matrix B ({n_arrows} highest effects)")
ax.set_xticks(np.arange(len(label_names)))
ax.set_yticks(np.arange(len(label_names)))
ax.set_xticklabels(label_names, rotation=45, ha="right")
ax.set_yticklabels(label_names)

save = True
if save:
    figures_dir = "/storage/store4/work/aheurteb/LiMVAM/experiments_meg/6_figures/"
    plt.savefig(figures_dir + f"top_coefs_B_{algo}.pdf", bbox_inches="tight")
plt.show()

# %%
