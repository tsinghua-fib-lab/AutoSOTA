import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from pathlib import Path


# matplotlib and seaborn style
fontsize = 19
rc = {
    "font.size": fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,
    "font.family": "serif",
    "font.serif": ["Times"],
}
sns.set(style="white")
plt.rcParams.update(rc)

# parameters
m = 6
p = 4
n = 500
nb_seeds = 30
max_interventions = p*(p-1)//2

labels = ["DirectLiMVAM", "MSS (KCI)"]

# read dataframe
# simulation_dir = Path("/storage/store4/work/aheurteb/LiMVAM/experiments_synthetic")
simulation_dir = Path("/Users/ambroiseheurtebise/Desktop/LiMVAM/experiments_synthetic")
results_dir = simulation_dir / "results/results_comparison_with_perry"
save_name = f"DataFrame_with_m{m}_p{p}_n{n}_seeds{nb_seeds}"
save_path = results_dir / save_name
df = pd.read_csv(save_path)

# Use color palette and markers
palette_sns = sns.color_palette()
palette = {
    'directlimvam': palette_sns[1],  # or direct_limvam?
    'mss': palette_sns[7],
}

fig, ax = plt.subplots(figsize=(6, 2.7))

sns.lineplot(
    data=df, x="nb_interventions", y="score", hue="method", linewidth=2.5,
    errorbar=('ci', 95), palette=palette,
    # hue_order=hue_order, style_order=hue_order, style="ica_algo",
    # dashes=dashes, legend=False
)
ax.get_legend().remove()
ax.grid(alpha=0.3)
ax.set_xlim([0, max_interventions])
ax.set_xticks(np.arange(max_interventions+1, dtype=int))
ax.set_yticks([0, 0.5, 1])
ax.set_yticklabels(["0\%", "50\%", "100\%"])
ax.set_xlabel("Number of interventions", fontsize=fontsize)
ax.set_ylabel("Proportion", fontsize=fontsize)
# ax.set_title(
#     "Proportion of entirely recovered orderings (higher is better)", 
#     fontsize=fontsize, y=1.02)

# legend
legend_styles = [
    Line2D([0], [0], color=palette["directlimvam"], linewidth=2.5, linestyle='-'),
    Line2D([0], [0], color=palette["mss"], linewidth=2.5, linestyle='-'),
]
fig.legend(
    legend_styles, labels, bbox_to_anchor=(0.5, 1.02), loc="center",
    ncol=3, borderaxespad=0., fontsize=fontsize
)


# save figure
figures_dir = simulation_dir / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(figures_dir / f"simulation_comparison_with_perry.pdf", bbox_inches="tight")
# plt.show()