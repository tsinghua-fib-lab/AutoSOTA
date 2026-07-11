import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


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
nb_seeds = 100

# read dataframe
# simulation_dir = Path("/storage/store4/work/aheurteb/LiMVAM/experiments_synthetic")
simulation_dir = Path("/Users/ambroiseheurtebise/Desktop/LiMVAM/experiments_synthetic")
results_dir = simulation_dir / "results/results_cross_view_correlations_decrease"
save_name = f"DataFrame_with_{nb_seeds}_seeds"
save_path = results_dir / save_name
df = pd.read_csv(save_path)

# Use color palette and markers
palette_sns = sns.color_palette()

fig, ax = plt.subplots(figsize=(6, 3))

# plot
sns.lineplot(
    data=df, x="cross_view_correlations_alpha", y="error_B", linewidth=2.5,
    estimator=np.median, errorbar=('ci', 95), label="DirectLiMVAM", color=palette_sns[1],
)

# ax.set_yscale("log")
ax.set_xticks([0, 1])
ax.set_yticks([0, 0.5])
ax.set_xticks(np.arange(0, 1.1, 0.1), minor=True)
ax.set_yticks(np.arange(0, 0.6, 0.1), minor=True)
ax.grid(which="both", linewidth=0.5, alpha=0.5)
ax.set_xlabel("Level of cross-view correlations")
ax.set_ylabel(r"Error on $B^i$")

# caption
# caption = (
#     "Caption: Cross-view correlation level is 0 when views are\n"
#     "uncorrelated, and 1 for random cross-view correlations as\n"
#     "in Section 6 of the paper."
# )
# fig.text(0.5, -0.2, caption, ha='center', va='center', fontsize=fontsize)


# save figure
# figures_dir = Path("/storage/store4/work/aheurteb/LiMVAM/experiments_synthetic/figures")
figures_dir = simulation_dir / "figures"
plt.savefig(figures_dir / f"simulation_cross_view_correlations_decrease.pdf", bbox_inches="tight")
plt.savefig(figures_dir / f"simulation_cross_view_correlations_decrease.png", bbox_inches="tight", dpi=300)
