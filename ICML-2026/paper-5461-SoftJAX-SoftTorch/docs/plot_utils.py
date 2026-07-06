from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D


linewidth = 1.0
SMALL_SIZE = 6
MEDIUM_SIZE = 6
BIGGER_SIZE = 6


def configure_plots() -> None:
    plt.rcParams["figure.dpi"] = 100
    plt.rc("font", size=SMALL_SIZE)
    plt.rc("axes", titlesize=SMALL_SIZE)
    plt.rc("axes", labelsize=MEDIUM_SIZE)
    plt.rc("xtick", labelsize=SMALL_SIZE)
    plt.rc("ytick", labelsize=SMALL_SIZE)
    plt.rc("legend", fontsize=SMALL_SIZE)
    plt.rc("figure", titlesize=BIGGER_SIZE)


def plot(
    fn,
    modes,
    title="",
    softnesses=[1.0, 0.5, 0.1],
    xs=jnp.linspace(-2, 2, 1001),
    **kwargs,
):
    xs = jnp.array(xs)

    colormap = LinearSegmentedColormap.from_list(
        "blue_red", ["dodgerblue", "gold", "lightcoral"]
    )
    colors = colormap(jnp.array(softnesses) / max(softnesses))

    # Create plotting canvas
    fig, axes = plt.subplots(
        2,
        len(modes),
        figsize=(3 * len(modes), 3.5),
        sharex=True,
        sharey="row",
        squeeze=False,
    )

    softnesses_arr = jnp.asarray(softnesses) if softnesses else None

    for col_idx, mode in enumerate(modes):
        ax_f = axes[0][col_idx]
        ax_g = axes[1][col_idx]

        if softnesses:

            def eval_for_softness(s):
                f = lambda x: fn(x, mode=mode, softness=s, **kwargs)
                return jax.vmap(jax.value_and_grad(f))(xs)

            ys_soft, grads_soft = jax.vmap(eval_for_softness)(softnesses_arr)

            for i, (softness, color) in enumerate(zip(softnesses, colors)):
                ax_f.plot(xs, ys_soft[i], linewidth=linewidth, color=color)
                ax_g.plot(
                    xs,
                    np.asarray(grads_soft[i]),
                    linewidth=linewidth,
                    color=color,
                )

        ys, grad_vals = jax.vmap(
            lambda x: jax.value_and_grad(fn)(x, mode="hard", softness=None, **kwargs)
        )(xs)
        ax_f.plot(xs, ys, linewidth=linewidth, linestyle="--", color="black")
        ax_g.plot(
            xs,
            grad_vals,
            linewidth=linewidth,
            linestyle="--",
            color="black",
        )

        ax_f.text(
            0.01, 0.99, f"[{mode}]", ha="left", va="top", transform=ax_f.transAxes
        )

        for ax in (ax_f, ax_g):
            ax.axhline(0, color="black", linewidth=0.5, alpha=0.7)
            ax.axvline(0, color="black", linewidth=0.5, alpha=0.7)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.set_xticks([-1.0, 0.0, 1.0])
            ax.margins(x=0)

    # Label axes
    for ax in axes[-1]:
        ax.set_xlabel("x")

    axes[0][0].set_ylabel("function")
    axes[1][0].set_ylabel("gradient")
    axes[0][0].set_yticks([min(min(ys), 0), max(ys)])

    # Build a shared legend for softnesses
    handles = [
        Line2D([0], [0], color=color, lw=1, label=str(s))
        for s, color in zip(softnesses, colors)
    ]
    handles.append(Line2D([0], [0], color="black", lw=1, label=f"{fn.__name__}"))
    handles.reverse()
    fig.legend(
        handles=handles,
        title="softness",
        loc="upper right",
        bbox_to_anchor=(1.0, 0.98),
        ncol=min(len(softnesses) + 1, 6),
        frameon=False,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    plt.locator_params(axis="both", nbins=3)
    plt.show()


def plot_value_and_grad(x, values, grads, label_func="function", label_grad="gradient"):
    fig, ax = plt.subplots(1, 1, figsize=(4, 2))
    plt.plot(x, values, label=label_func, color="black", linewidth=1.0)
    plt.plot(
        x, grads, label=label_grad, color="dodgerblue", linewidth=1.0, linestyle="--"
    )
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.7)
    ax.axvline(0, color="black", linewidth=0.5, alpha=0.7)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    xmin = round(min(min(x), 0), 1)
    xmax = round(max(x), 1)
    ax.set_xticks([xmin, xmax])
    ax.margins(x=0)
    ax.set_xlabel("x")
    plt.legend()
    plt.xlim(xmin - 0.1, xmax + 0.1)

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    plt.locator_params(axis="both", nbins=3)
    plt.show()


def plot_array(x, plot_text=True, title=""):
    plt.figure(figsize=(4, 2))
    plt.imshow(x, cmap="coolwarm")

    if plot_text:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                plt.text(
                    j,
                    i,
                    f"{x[i, j]:0.2f}",
                    ha="center",
                    va="center",
                    color="white",
                )

    if title:
        plt.title(f"{title}")

    plt.tight_layout()
    plt.xticks(range(x.shape[1]))
    plt.yticks(range(x.shape[0]))
    plt.show()


def plot_softness_sweep(
    fn,
    modes,
    *,
    x=None,
    axis: int | None = 0,
    softness_min: float = 1e-2,
    softness_max: float = 1e2,
    num_softness: int = 200,
    descending: bool = False,
    show_hard: bool = True,
    mark_endpoints: bool = True,
    title: str = "",
    **kwargs,
):
    if x is None:
        x = jnp.array([10.0, 1.0, 2.0, 9.0, 4.0, 8.0, 3.0, 7.0, 5.0, 6.0]) / 10.0
    x = jnp.asarray(x)

    eps = jnp.asarray(
        np.logspace(np.log10(softness_min), np.log10(softness_max), num_softness),
        dtype=x.dtype,
    )

    def eval_mode(mode):
        def f(e):
            y = fn(
                x,
                axis=axis,
                descending=descending,
                softness=e,
                mode=mode,
                **kwargs,
            )
            y = jnp.asarray(y)
            y = y.reshape(-1, y.shape[-1])  # (M, D), D = last-dim curves
            return y

        return jax.vmap(f)(eps)  # (S, M, D)

    ys = [eval_mode(mode) for mode in modes]  # list of (S, M, D)

    hard = None
    if show_hard:
        hard = jnp.asarray(
            fn(
                x,
                axis=axis,
                descending=descending,
                softness=None,
                mode="hard",
                **kwargs,
            )
        )
        hard = np.asarray(hard.reshape(-1, hard.shape[-1]))  # (M, D)

    fig, axes = plt.subplots(
        1,
        len(modes),
        figsize=(3.2 * len(modes), 3.0),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    xs = np.asarray(eps)

    for ax, mode, y in zip(axes, modes, ys):
        y = np.asarray(y)  # (S, M, D)
        S, M, D = y.shape

        y2 = y.reshape(S, M * D)  # (S, M*D)
        lines = ax.plot(xs, y2)

        if mark_endpoints:
            for j, ln in enumerate(lines):
                c = ln.get_color()
                ax.plot(
                    [xs[0], xs[-1]],
                    [y2[0, j], y2[-1, j]],
                    marker="o",
                    linestyle="None",
                    color=c,
                    markersize=3,
                )

        if show_hard:
            hard2 = hard.reshape(M * D)
            for j in range(M * D):
                ax.plot(
                    [xs[0], xs[-1]],
                    [hard2[j], hard2[j]],
                    linestyle="--",
                    color="black",
                    linewidth=1.0,
                )

        ax.set_xscale("log")
        ax.set_xlabel("softness")
        ax.set_title(f"[{mode}]")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    axes[0].set_ylabel("output value")
    if title:
        fig.suptitle(title, y=1.02)

    fig.tight_layout()
    plt.show()
    return fig, axes


def plot_softindices_1D(x, title=None, log=False):
    # Convert JAX array to numpy array if needed
    if hasattr(x, "__array__"):
        x = np.array(x)

    # Apply log transform if requested
    if log:
        x = np.log(np.maximum(x, 1e-10))  # Add small epsilon to avoid log(0)

    my_cmap = plt.get_cmap("coolwarm")

    # Rescale function that handles constant arrays
    def rescale(arr):
        arr = np.array(arr)
        arr_min = np.min(arr)
        arr_max = np.max(arr)
        if arr_max - arr_min < 1e-10:  # All values are the same
            return np.zeros_like(arr)  # Return zeros (will map to middle of colormap)
        return (arr - arr_min) / (arr_max - arr_min)

    fig, ax = plt.subplots(1, 1, figsize=(4, 1))
    colors = my_cmap(rescale(x))
    ax.bar(range(x.shape[0]), x, color=colors)
    ax.set_xticks(range(x.shape[0]))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlabel("indices")  # fontsize=14)
    ax.set_yticks([round(min(min(x), 0), 1), round(max(x), 1)])
    if title:
        ax.set_title(f"{title}")
    plt.tight_layout()
    plt.show()


def plot_softbool_operation(fn, title=""):
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)

    # F = -X * Y - Y - X
    F = fn(X, Y)
    plt.figure(figsize=(5, 4))
    cf = plt.contourf(X, Y, F, levels=50, cmap="coolwarm")
    levels = np.arange(0, 1, 0.1)
    c = plt.contour(X, Y, F, colors="k", levels=levels, linewidths=0.5)
    plt.clabel(c, inline=True, fontsize=8)
    plt.colorbar(cf)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(label=f"{fn.__name__}")
    # plt.xticks(np.arange(0,1,0.2))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect("equal", "box")
    plt.tight_layout()
    plt.show()


def plot_value_grad_2D(
    fn: Callable,
    min_val=-1,
    max_val=1,
    title="",
):
    x = jnp.linspace(min_val, max_val, 100)
    y = jnp.linspace(min_val, max_val, 100)
    X, Y = jnp.meshgrid(x, y)
    F = fn(X, Y)
    # Compute the gradient field on the grid so we can quiver-plot it
    grad_fn = jax.grad(lambda a, b: fn(a, b), argnums=(0, 1))
    grad_over_row = jax.vmap(
        lambda x_row, y_row: jax.vmap(lambda _x, _y: grad_fn(_x, _y))(x_row, y_row)
    )
    grad_x, grad_y = grad_over_row(X, Y)

    # normalize the gradient vectors for better visualization
    grad_magnitude = jnp.sqrt(grad_x**2 + grad_y**2)
    grad_x = jnp.where(grad_magnitude > 0, grad_x / grad_magnitude, 0)
    grad_y = jnp.where(grad_magnitude > 0, grad_y / grad_magnitude, 0)

    plt.figure(figsize=(5, 4))
    cf = plt.contourf(np.array(X), np.array(Y), np.array(F), levels=50, cmap="coolwarm")

    stride = max(1, X.shape[0] // 20)  # thin the arrows to keep the plot readable
    plt.quiver(
        np.array(X[::stride, ::stride]),
        np.array(Y[::stride, ::stride]),
        np.array(grad_x[::stride, ::stride]),
        np.array(grad_y[::stride, ::stride]),
        color="white",
        alpha=0.9,
        linewidth=0.5,
        scale=50,
    )

    plt.colorbar(cf)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(label=title or f"{fn.__name__}")
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.gca().set_aspect("equal", "box")
    plt.tight_layout()
    plt.show()
