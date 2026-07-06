import os
import ot
import tqdm
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from time import perf_counter
from sklearn.utils import check_random_state

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.loss_funcs import quota_loss
from src.penalized_ot import PenalizedOT
from src.datagen import get_gaussian_mixture
from src.cost_learning import TorchFairCostOT

entropic_grid = [1.0]
F_target = torch.tensor([[0.2, 0.3], [0.28, 0.22]])
n_runs = 10

# Generate a big dataset for cost learning
n_train_samples = 1000
results_mahalanobis = []
results_mlp = []

(X_train, Y_train), (S_X_train, S_Y_train) = get_gaussian_mixture(
    d=2,
    n_x=n_train_samples,
    n_y=n_train_samples // 10,
    scale=0.2,
    p_x0=0.5,
    p_y0=0.5,
    centers_X=[np.array([0, 0]), np.array([2.0, 0.0])],
    centers_Y=[np.array([1.0, 1.0]), np.array([2.5, 0.5])],
    rng=42,
)

(X_test, Y_test), (S_X_test, S_Y_test) = get_gaussian_mixture(
    d=2,
    n_x=10000,
    n_y=1000,
    scale=0.2,
    p_x0=0.5,
    p_y0=0.5,
    centers_X=[np.array([0, 0]), np.array([2.0, 0.0])],
    centers_Y=[np.array([1.0, 1.0]), np.array([2.5, 0.5])],
    rng=12,
)

# Fairness of the original OT plan on train data
cost_matrix = ot.dist(X_train, Y_train)
ot_plan = ot.sinkhorn(
    torch.ones(X_train.shape[0]) / X_train.shape[0],
    torch.ones(Y_train.shape[0]) / Y_train.shape[0],
    cost_matrix,
    reg=entropic_grid[0],
    warn=False,
    verbose=False,
    stopThr=1e-6,
    numItermax=1000,
    method="sinkhorn",
    log=True,
)[0]

original_fairness = quota_loss(ot_plan, S_X_train, S_Y_train, F_target)
print(f"Original fairness of OT plan: {original_fairness.item()}")

# Cost Learning

fixed_cost_model_mahalanobis = TorchFairCostOT(
    penalty_grid=[1000.0],
    entropic_grid=entropic_grid,
    lr_grid=[1e-1],
    fairness_loss="quota_loss",
    cost_model_name="mahalanobis",
    verbose=True,
    optimizer="Adam",
)

fixed_cost_model_mlp = TorchFairCostOT(
    penalty_grid=[500.0],
    entropic_grid=entropic_grid,
    lr_grid=[5e-2],
    fairness_loss="quota_loss",
    cost_model_name="mlp",
    verbose=True,
    optimizer="Adam",
    d_hidden=4,
    d_out=2,
    n_layers=2,
)

results_mahalanobis = fixed_cost_model_mahalanobis.solve(
    X=X_train,
    Y=Y_train,
    S_X=S_X_train,
    S_Y=S_Y_train,
    F=F_target,
    n_iter=2000,
    tol=1e-6,
    use_cache=False,
    auto_stop=False,
)
results_mahalanobis["final_fairness"] = results_mahalanobis[
    "fairness_loss_value"
].apply(lambda x: x[-1])

results_mlp = fixed_cost_model_mlp.solve(
    X=X_train,
    Y=Y_train,
    S_X=S_X_train,
    S_Y=S_Y_train,
    F=F_target,
    pretrained_weights="mlp_seed42.pt",
    n_iter=2000,
    tol=1e-6,
    use_cache=False,
)

results_mlp["final_fairness"] = results_mlp["fairness_loss_value"].apply(
    lambda x: x[-1]
)

results_mahalanobis = pd.concat(results_mahalanobis, ignore_index=True)
results_mlp = pd.concat(results_mlp, ignore_index=True)

n_test_samples = 500

all_fairness_penalized = []
all_fairness_learnt_cost = []

df = pd.DataFrame(
    columns=[
        "fairness_mahalanobis_train",
        "fairness_mlp_train",
        "fairness_mahalanobis_test",
        "fairness_mlp_test",
        "fairness_vanilla_ot",
        "fairness_penalized",
        "time_penalized",
    ]
)

rng = check_random_state(22)
for i, n in enumerate(n_train_samples):
    x_sample_size = n_test_samples
    y_sample_size = n_test_samples // 10
    for k in tqdm.tqdm(range(n_runs)):

        id_X = rng.choice(X_test.shape[0], x_sample_size, replace=False)
        id_Y = rng.choice(Y_test.shape[0], y_sample_size, replace=False)
        X_sampled = X_test[id_X]
        Y_sampled = Y_test[id_Y]
        S_X_sampled = S_X_test[id_X]
        S_Y_sampled = S_Y_test[id_Y]

        # Compute the OT plan with learnt cost
        start = perf_counter()
        cost_matrix_mahalanobis = (
            results_mahalanobis["model"].iloc[0].cpu()(X_sampled, Y_sampled)
        )

        ot_plan = ot.sinkhorn(
            torch.ones(x_sample_size) / x_sample_size,
            torch.ones(y_sample_size) / y_sample_size,
            cost_matrix_mahalanobis,
            reg=entropic_grid[0],
            warn=False,
            verbose=False,
            stopThr=1e-6,
            numItermax=1000,
            method="sinkhorn",
            log=True,
        )[0]
        time_mahalanobis = perf_counter() - start
        fairness_mahalanobis = quota_loss(
            ot_plan, S_X_sampled, S_Y_sampled, F_target
        )

        start = perf_counter()
        cost_matrix_mlp = (
            results_mlp["model"].iloc[0].cpu()(X_sampled, Y_sampled)
        )

        ot_plan = ot.sinkhorn(
            torch.ones(x_sample_size) / x_sample_size,
            torch.ones(y_sample_size) / y_sample_size,
            cost_matrix_mlp,
            reg=entropic_grid[0],
            warn=False,
            verbose=False,
            stopThr=1e-6,
            numItermax=1000,
            method="sinkhorn",
            log=True,
        )[0]
        time_mlp = perf_counter() - start

        fairness_mlp = quota_loss(ot_plan, S_X_sampled, S_Y_sampled, F_target)

        cost_matrix_vanilla = ot.dist(X_sampled, Y_sampled)
        ot_plan = ot.sinkhorn(
            torch.ones(x_sample_size) / x_sample_size,
            torch.ones(y_sample_size) / y_sample_size,
            cost_matrix_vanilla,
            reg=entropic_grid[0],
            warn=False,
            verbose=False,
            stopThr=1e-6,
            numItermax=1000,
            method="sinkhorn",
            log=True,
        )[0]
        fairness_vanilla_ot = quota_loss(
            ot_plan, S_X_sampled, S_Y_sampled, F_target
        )

        # Solve penalized OT
        start = perf_counter()
        penalized_ot = PenalizedOT(
            penalty_grid=[90.0],
            entropic_grid=entropic_grid,
            fairness_loss="quota_loss",
        )

        results = penalized_ot.solve(
            X=X_sampled,
            Y=Y_sampled,
            S_X=S_X_sampled,
            S_Y=S_Y_sampled,
            F=F_target,
            use_cache=False,
        )
        time_penalized = perf_counter() - start
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "fairness_mahalanobis_train": [
                            results_mahalanobis["final_fairness"]
                            .iloc[0]
                            .item()
                        ],
                        "fairness_mlp_train": [
                            results_mlp["final_fairness"].iloc[0].item()
                        ],
                        "n_train": [n_train_samples],
                        "fairness_mahalanobis_test": [
                            fairness_mahalanobis.item()
                        ],
                        "fairness_mlp_test": [fairness_mlp.item()],
                        "fairness_penalized": [
                            results.fairness_loss_value.item()
                        ],
                        "fairness_vanilla_ot": [fairness_vanilla_ot.item()],
                        "time_penalized": [time_penalized],
                        "time_mahalanobis": [time_mahalanobis],
                        "time_mlp": [time_mlp],
                    }
                ),
            ],
            ignore_index=True,
        )
        print(df.tail(1))

# Plotting the results
sns.set_style("whitegrid")

plt.rcParams["font.family"] = "serif"
legend_font_size = 18
label_font_size = 18

f, ax = plt.subplots(
    2,
    2,
    figsize=(10, 4),
    gridspec_kw={"height_ratios": [1, 2], "width_ratios": [1, 1]},
)
sns.set_palette("viridis", n_colors=4)
viridis_colors = sns.color_palette("viridis", n_colors=4)

# Plot test fairness vs time in ax[1, 0]
sns.scatterplot(
    x=df["fairness_mahalanobis_test"],
    y=df["time_mahalanobis"],
    ax=ax[1, 0],
    s=140,
    edgecolor="black",
    color=viridis_colors[0],
    marker="o",
    label="Mahalanobis",
)
sns.scatterplot(
    x=df["fairness_mlp_test"],
    y=df["time_mlp"],
    ax=ax[1, 0],
    s=140,
    edgecolor="black",
    color=viridis_colors[1],
    marker="X",
    label="MLP",
)
sns.scatterplot(
    x=df["fairness_penalized"],
    y=df["time_penalized"],
    ax=ax[1, 0],
    s=140,
    edgecolor="black",
    color=viridis_colors[2],
    marker="D",
    label="Penalized OT",
)

ax[1, 0].set_yscale("log")

for spine in ax[1, 0].spines.values():
    spine.set_edgecolor("black")
    spine.set_linewidth(1)
ax[1, 0].set_xlabel("Fairness loss", fontsize=label_font_size)
ax[1, 0].set_ylabel("Time (s)", fontsize=label_font_size)
ax[1, 0].tick_params(axis="both", labelsize=label_font_size)

handles, labels = ax[1, 0].get_legend_handles_labels()
ax[0, 0].legend(
    handles,
    labels,
    fontsize=legend_font_size,
    title="Method",
    title_fontsize=legend_font_size + 1,
    loc="center",
    frameon=True,
    ncol=2,
)
ax[0, 0].axis("off")
ax[1, 0].get_legend().remove()

# Create boxplot for generalization in ax[1, 1]
sns.boxplot(
    data=df[
        [
            "fairness_mahalanobis_test",
            "fairness_mlp_test",
            "fairness_vanilla_ot",
        ]
    ],
    ax=ax[1, 1],
    saturation=0.5,
)

# add a line in violinplot indicating training fairness
ax[1, 1].axhline(
    y=results_mahalanobis["final_fairness"].iloc[0].item(),
    xmin=0.03,
    xmax=0.31,
    color="red",
    linestyle="--",
)
ax[1, 1].axhline(
    y=results_mlp["final_fairness"].iloc[0].item(),
    xmin=0.36,
    xmax=0.64,
    color="red",
    linestyle="--",
)

for spine in ax[1, 1].spines.values():
    spine.set_edgecolor("black")
    spine.set_linewidth(1)
ax[1, 1].set_ylabel("Test fairness loss", fontsize=label_font_size)

ax[1, 1].set_xticklabels(
    ["Mahalanobis", "MLP", "Vanilla OT"],
    rotation=0,
    fontsize=legend_font_size,
)
ax[1, 1].set_yscale("log")
ax[1, 1].tick_params(axis="y", labelsize=legend_font_size)

# Create legend for boxplot in ax[0, 1] (first row, second column)

legend_elements = [
    Line2D([0], [0], color="red", linestyle="--", label="Train fairness loss")
]
ax[0, 1].legend(
    handles=legend_elements,
    fontsize=legend_font_size,
    loc="center",
    frameon=False,
)
ax[0, 1].axis("off")

plt.tight_layout()

# compute the variance and add as text to the plot
var_mahalanobis = df["fairness_mahalanobis_test"].var()
var_mlp = df["fairness_mlp_test"].var()
var_penalized = df["fairness_penalized"].var()
var_vanilla = df["fairness_vanilla_ot"].var()

print(f"Variance Mahalanobis: {var_mahalanobis}")
print(f"Variance MLP: {var_mlp}")
print(f"Variance Penalized OT: {var_penalized}")
print(f"Variance Vanilla OT: {var_vanilla}")

os.makedirs("figures/exp_gaussian/", exist_ok=True)
plt.savefig("figures/exp_gaussian/generalization.pdf")
