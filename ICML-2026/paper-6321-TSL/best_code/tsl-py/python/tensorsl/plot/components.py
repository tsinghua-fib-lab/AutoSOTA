"""Plot one-dimensional components of GridTensors (raw, pre-aggregation)."""

from typing import Optional

from ._common import _require_matplotlib
from ._theme import (
    TOKENS,
    airy,
    axis_label,
    card_inset,
    figure_title,
    flat_background,
    flat_legend,
    grid_card_layout,
    grid_figsize,
    header,
    setup_fonts,
)

# Calm categorical cycle for overlaid step curves: indigo lead, then the warm /
# cool signed tokens, rounding out with two teal/violet picks and muted greys.
FLAT_CYCLE = [
    TOKENS["accent"], TOKENS["pos"], TOKENS["neg"], "#0D9488", "#9333EA",
    "#D97706", TOKENS["greys"][2], TOKENS["greys"][1],
]


def _step_points(intervals, values):
    """Bounded step coordinates for one component: drop the ±inf tail intervals."""
    x_points = []
    y_points = []
    for (x_start, x_end), y in zip(intervals, values):
        if x_start == float("-inf") or x_end == float("inf"):
            continue
        x_points.extend([x_start, x_end])
        y_points.extend([y, y])
    return x_points, y_points


def _component_card(plt, title, badge="plot_grid_tensor_components()", cell_h_in=5.6):
    """A single full-width card for a component figure: dotted ground, one card,
    figure title, and an inset axes. Returns ``(fig, ax, bgax, cards)``."""
    figsize = grid_figsize(1, 1, cell_w_in=10.0, cell_h_in=cell_h_in)
    fig = plt.figure(figsize=figsize)
    fw, fh = fig.get_size_inches()
    cards = grid_card_layout(fw, fh, 1, 1)
    bgax = flat_background(fig, cards)
    figure_title(fig, "TSL / diagnostics", title,
                 badge=badge, badge_color=TOKENS["accent"])
    ax = card_inset(fig, cards, (0, 0))
    return fig, ax, bgax, cards


def _style_axes(fig, ax, bgax, cards, disp, mono, kicker, title, legend=True):
    """Flat styling shared by every component card: optional mono legend, airy
    spines with a faint dot-grid, muted x/value axis labels, and a card header."""
    if legend and ax.get_legend_handles_labels()[1]:
        flat_legend(ax, mono, loc="upper right", fontsize=8)
    airy(ax, mono, grid_axis="both")
    axis_label(ax, mono, xlabel="x", ylabel="value")
    header(fig, bgax, cards, (0, 0), kicker, title, "", disp, mono)


def plot_grid_tensor_components(grid_tensor, individual_plots: bool = False, axis: Optional[int] = None):
    """Plot one-dimensional components of a GridTensor.

    Parameters
    ----------
    grid_tensor : GridTensor
        A fitted GridTensor instance.
    individual_plots : bool, default=False
        If True, each component gets its own figure.
    axis : int, optional
        If provided, only plot the component for this feature index.
    """
    plt = _require_matplotlib()
    disp, mono = setup_fonts()

    n_components = len(grid_tensor.intervals)

    if axis is not None:
        if not 0 <= axis < n_components:
            raise ValueError(f"axis must be between 0 and {n_components - 1}")
        axes_to_plot = [(axis, (grid_tensor.intervals[axis], grid_tensor.mean_factor[axis]))]
    else:
        axes_to_plot = list(enumerate(zip(grid_tensor.intervals, grid_tensor.mean_factor)))

    base_w = TOKENS["base_w"] + 0.3

    if not individual_plots:
        fig, ax, bgax, cards = _component_card(plt, "GridTensor components")

    for axis_idx, (intervals, values) in axes_to_plot:
        if individual_plots:
            fig, ax, bgax, cards = _component_card(plt, "GridTensor component")

        color = FLAT_CYCLE[axis_idx % len(FLAT_CYCLE)]
        x_points, y_points = _step_points(intervals, values)
        if x_points:
            ax.step(x_points, y_points, where="post", lw=base_w, color=color,
                    solid_capstyle="round", label=f"axis {axis_idx}")

        if individual_plots:
            _style_axes(fig, ax, bgax, cards, disp, mono, f"axis {axis_idx}",
                        "One-dimensional component", legend=False)

    if not individual_plots:
        _style_axes(fig, ax, bgax, cards, disp, mono, "all axes",
                    "One-dimensional components")


def plot_combined_grid_tensors(model, individual_plots: bool = True, axis: Optional[int] = None):
    """Plot combined grid-tensor components for each stage of a TSL model."""
    for tgf in model.stage_predictors:
        plot_grid_tensor_components(tgf.combined_grid_tensor, individual_plots=individual_plots, axis=axis)


def plot_epoch_components(model, epoch: int) -> None:
    """Plot all per-tree grid components for a given stage.

    Parameters
    ----------
    model : TSL
        A fitted TSL instance.
    epoch : int
        Zero-based stage index to visualize.
    """
    plt = _require_matplotlib()
    disp, mono = setup_fonts()

    families = model.stage_predictors
    total_epochs = len(families)

    if epoch < 0 or epoch >= total_epochs:
        raise ValueError(f"epoch must be between 0 and {total_epochs - 1}")

    epoch_grid_tensors = families[epoch].grid_tensors

    if len(epoch_grid_tensors) == 0:
        raise ValueError(f"No tree grids found for epoch {epoch}")

    num_components = len(epoch_grid_tensors[0].intervals)
    base_w = TOKENS["base_w"] + 0.3

    for component_index in range(num_components):
        fig, ax, bgax, cards = _component_card(
            plt, "Per-tree components", badge="plot_epoch_components()",
            cell_h_in=4.2)

        for grid_index, grid in enumerate(epoch_grid_tensors):
            intervals = grid.intervals[component_index]
            values = grid.mean_factor[component_index]
            x_points, y_points = _step_points(intervals, values)
            if x_points:
                ax.step(x_points, y_points, where="post", lw=base_w,
                        color=FLAT_CYCLE[grid_index % len(FLAT_CYCLE)],
                        solid_capstyle="round", label=f"grid {grid_index}")

        _style_axes(fig, ax, bgax, cards, disp, mono,
                    f"Stage {epoch + 1} · component {component_index}",
                    "Per-tree grid components")
