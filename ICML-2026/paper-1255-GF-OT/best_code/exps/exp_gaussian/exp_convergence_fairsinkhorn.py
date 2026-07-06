import os
import torch
import numpy as np
import seaborn as sns
from ot import sinkhorn
import matplotlib.pyplot as plt

from src.solvers import fair_sinkhorn_knopp
from src.datagen import get_gaussian_mixture

rng = np.random.RandomState(0)

n_grid = torch.linspace(10, 200, 5, dtype=torch.int32)
n_runs = 5
F_target = torch.tensor([[0.25, 0.25], [0.25, 0.25]])

eps = 2.0
n = 1000
d = 5
random_center_x = rng.randn(d)
random_center_y = rng.randn(d)


(X, Y), (S_X, S_Y) = get_gaussian_mixture(
    d=d,
    n_x=n,
    n_y=n,
    scale=3,
    p_x0=0.5,
    p_y0=0.5,
    centers_X=[random_center_x, random_center_x + np.ones(d)],
    centers_Y=[random_center_y, random_center_y + np.ones(d)],
    rng=42,
)


a = torch.ones((X.shape[0])) / X.shape[0]
b = torch.ones((Y.shape[0])) / Y.shape[0]

cost = torch.sum((X[:, None, :] - Y[None, :, :]) ** 2, dim=2)

true_fair_plan, fair_log = fair_sinkhorn_knopp(
    a=a.clone(),
    b=b.clone(),
    M=cost,
    F=F_target,
    S_X=S_X,
    S_Y=S_Y,
    reg=eps,
    log=True,
    numItermax=int(1e4),
    stopThr=1e-16,
)

true_plan, log = sinkhorn(
    a=a.clone(),
    b=b.clone(),
    M=cost,
    reg=eps,
    log=True,
    numItermax=int(1e4),
    stopThr=1e-16,
)
print("Number of iterations done:", log["niter"])

# in the log object, compute the KL divergence between elements of the log and
# the final element of the log

fair_kl_divergences = []
kl_divergences = []
a = torch.ones((X.shape[0])) / X.shape[0]
b = torch.ones((Y.shape[0])) / Y.shape[0]
final_fair_plan = true_fair_plan.clone()
final_plan = true_plan.clone()
reg_value = 1e-16
for i in range(1, 200, 10):
    plan, fair_log = fair_sinkhorn_knopp(
        a=a.clone(),
        b=b.clone(),
        M=cost,
        F=F_target,
        S_X=S_X,
        S_Y=S_Y,
        reg=eps,
        log=True,
        numItermax=int(i),
        stopThr=1e-16,
    )
    fair_kl_divergences.append(
        torch.sum(
            plan
            * (
                torch.log(plan + reg_value)
                - torch.log(final_fair_plan + reg_value)
            )
        )
    )
for i in range(1, 200, 10):
    plan, fair_log = sinkhorn(
        a=a.clone(),
        b=b.clone(),
        M=cost,
        reg=eps,
        log=True,
        numItermax=int(i),
        stopThr=1e-16,
    )
    kl_divergences.append(
        torch.sum(
            plan
            * (torch.log(plan + reg_value) - torch.log(final_plan + reg_value))
        )
    )

grid = np.arange(1, 200, 10)
fair_kl_to_plot = fair_kl_divergences[: len(grid)]
kl_to_plot = kl_divergences[: len(grid)]
# convert to np array
fair_kl_to_plot = np.array(fair_kl_to_plot)
kl_to_plot = np.array(kl_to_plot)

plt.figure(figsize=(4, 3))
sns.lineplot(
    x=grid,
    y=kl_to_plot,
    linewidth=3,
    marker="o",
    markersize=8,
    label="Sinkhorn",
)
sns.lineplot(
    x=grid,
    y=fair_kl_to_plot,
    linewidth=3,
    marker="o",
    markersize=6,
    label="Fair Sinkhorn",
)
plt.xlabel("Iteration")
plt.ylabel("KL Divergence")
plt.yscale("log")
plt.xscale("log")


plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
plt.minorticks_on()
plt.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.5)

# make the axis black and bold

for a in ["top", "bottom", "left", "right"]:
    plt.gca().spines[a].set_linewidth(1)
    plt.gca().spines[a].set_color("black")

plt.legend().remove()

# remove x and y labels
plt.xlabel("")
plt.ylabel("")
os.makedirs("figures/exp_gaussian/", exist_ok=True)
plt.savefig("figures/exp_gaussian/convergence_fair_vs_vanilla.pdf", dpi=300)
