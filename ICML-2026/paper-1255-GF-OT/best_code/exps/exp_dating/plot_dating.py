import os
import pandas as pd
import matplotlib.pyplot as plt

results_penalized = pd.read_pickle("results/exp_dating/results_penalized.pkl")
results_cost_learning = pd.read_pickle(
    "results/exp_dating/results_cost_learning.pkl"
)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, sharex=True)

ax = axes[0]
vmin_penalty = results_penalized["penalty"].min()
vmax_penalty = results_penalized["penalty"].max()
norm = plt.matplotlib.colors.LogNorm(vmin=vmin_penalty, vmax=vmax_penalty)
scatter = ax.scatter(
    results_penalized["fairness_loss_value"],
    results_penalized["cost_diff"],
    c=results_penalized["penalty"],
    cmap="Oranges",
    marker="o",
    s=100,
    edgecolor="black",
    norm=norm,
)
ax.set_xlabel("Fairness Loss", fontsize=16)
ax.set_ylabel("Cost Difference", fontsize=16)
ax.set_title("Penalized OT", fontsize=16)

plt.colorbar(scatter, ax=ax)
ax = axes[1]
vmin_penalty = results_cost_learning["penalty"].min()
vmax_penalty = results_cost_learning["penalty"].max()
norm = plt.matplotlib.colors.LogNorm(vmin=vmin_penalty, vmax=vmax_penalty)
scatter = ax.scatter(
    results_cost_learning["fairness_loss_value"].apply(lambda x: x[-1]),
    results_cost_learning["cost_diff"],
    c=results_cost_learning["penalty"],
    cmap="Oranges",
    marker="o",
    s=100,
    edgecolor="black",
    norm=norm,
)
ax.set_xlabel("Fairness Loss", fontsize=16)
ax.set_title("Cost learning", fontsize=16)

plt.colorbar(scatter, ax=ax)
os.makedirs("figures/exp_dating/", exist_ok=True)
plt.savefig("figures/exp_dating/cost_vs_fairness.pdf",
            bbox_inches="tight", dpi=300)
