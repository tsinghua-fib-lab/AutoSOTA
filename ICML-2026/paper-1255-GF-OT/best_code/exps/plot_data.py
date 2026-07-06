import os
import numpy as np
import matplotlib.pyplot as plt

from src.datagen import get_nested_circles, get_gaussian_mixture

(X_gaussian, Y_gaussian), (S_X_gaussian, S_Y_gaussian) = get_gaussian_mixture(
    d=2,
    n_x=250,
    n_y=25,
    scale=0.2,
    p_x0=0.5,
    p_y0=0.5,
    centers_X=[np.array([0, 0]), np.array([2.0, 0.0])],
    centers_Y=[np.array([1.0, 1.0]), np.array([2.5, 0.5])],
    rng=42,
)

(X_circles, Y_circles), (S_X_circles, S_Y_circles) = get_nested_circles(
    n_x=250,
    n_y=25,
    p_X0=0.5,
    p_Y0=0.5,
    noise_0=0.15,
    noise_1=0.2,
    diameter=4.0,
    rng=42,
    n_outliers_x=4,
    n_outliers_y=2,
)

# Create figure with GridSpec for custom layout
plt.rc("font", family="serif")
# sns.set_style("white")

fig = plt.figure(figsize=(8, 4))
gs = fig.add_gridspec(2, 2, height_ratios=[0.15, 1], hspace=0.1)

# Define colors for groups
colors_X = {0: "tab:blue", 1: "tab:orange"}
colors_Y = {0: "tab:green", 1: "tab:red"}

# Row 0: Single shared legend spanning both columns
ax_legend = fig.add_subplot(gs[0, :])
ax_legend.axis("off")
legend_elements = [
    plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=colors_X[0],
        markersize=12,
        label=r"$X$, $S=0$",
    ),
    plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=colors_X[1],
        markersize=12,
        label=r"$X$, $S=1$",
    ),
    plt.Line2D(
        [0],
        [0],
        marker="s",
        color="w",
        markerfacecolor=colors_Y[0],
        markersize=12,
        label=r"$Y$, $W=0$",
    ),
    plt.Line2D(
        [0],
        [0],
        marker="s",
        color="w",
        markerfacecolor=colors_Y[1],
        markersize=12,
        label=r"$Y$, $W=1$",
    ),
]
ax_legend.legend(
    handles=legend_elements,
    loc="center",
    fontsize=14,
    ncol=4,
    frameon=True,
    fancybox=True,
)

# Row 1: Point clouds
# Plot Gaussian point cloud (1, 0)
ax_gaussian = fig.add_subplot(gs[1, 0])
for group in [0, 1]:
    mask_X = S_X_gaussian == group
    ax_gaussian.scatter(
        X_gaussian[mask_X, 0],
        X_gaussian[mask_X, 1],
        c=colors_X[group],
        marker="o",
        s=50,
        alpha=0.7,
        label=f"X - Group {group}",
    )
    mask_Y = S_Y_gaussian == group
    ax_gaussian.scatter(
        Y_gaussian[mask_Y, 0],
        Y_gaussian[mask_Y, 1],
        c=colors_Y[group],
        marker="s",
        s=100,
        alpha=0.9,
        edgecolor="black",
        linewidth=1,
        label=f"Y - Group {group}",
    )
ax_gaussian.tick_params(axis="both", labelsize=14)
ax_gaussian.set_box_aspect(1)
for spine in ax_gaussian.spines.values():
    spine.set_color("black")
    spine.set_linewidth(1)

# Plot Nested Circles point cloud (1, 1)
ax_circles = fig.add_subplot(gs[1, 1])
for group in [0, 1]:
    mask_X = S_X_circles == group
    ax_circles.scatter(
        X_circles[mask_X, 0],
        X_circles[mask_X, 1],
        c=colors_X[group],
        marker="o",
        s=50,
        alpha=0.7,
        label=f"X - Group {group}",
    )
    mask_Y = S_Y_circles == group
    ax_circles.scatter(
        Y_circles[mask_Y, 0],
        Y_circles[mask_Y, 1],
        c=colors_Y[group],
        marker="s",
        s=100,
        alpha=0.9,
        edgecolor="black",
        linewidth=1,
        label=f"Y - Group {group}",
    )
ax_circles.tick_params(axis="both", labelsize=14)
ax_circles.set_box_aspect(1)
for spine in ax_circles.spines.values():
    spine.set_color("black")
    spine.set_linewidth(1)

plt.tight_layout()
os.makedirs("figures/", exist_ok=True)
plt.savefig("figures/data.pdf")
