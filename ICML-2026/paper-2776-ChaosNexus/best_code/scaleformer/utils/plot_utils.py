import os
import warnings
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches as mpatches
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from omegaconf import OmegaConf

from scaleformer.utils import safe_standardize

DEFAULT_COLORS = list(plt.rcParams["axes.prop_cycle"].by_key()["color"])
DEFAULT_MARKERS = ["o", "s", "v", "D", "X", "P", "H", "h", "d", "p", "x"]


def apply_custom_style(config_path: str):
    """
    Apply custom matplotlib style from config file with rcparams
    """
    if os.path.exists(config_path):
        cfg = OmegaConf.load(config_path)
        plt.style.use(cfg.base_style)

        custom_rcparams = OmegaConf.to_container(cfg.matplotlib_style, resolve=True)
        for category, settings in custom_rcparams.items():
            if isinstance(settings, dict):
                for param, value in settings.items():
                    if isinstance(value, dict):
                        for subparam, subvalue in value.items():
                            plt.rcParams[f"{category}.{param}.{subparam}"] = subvalue
                    else:
                        plt.rcParams[f"{category}.{param}"] = value
    else:
        print(f"Warning: Plotting config not found at {config_path}")


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, _ = proj_transform(xs3d, ys3d, zs3d, self.axes.get_proj())
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, self.axes.get_proj())
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def make_clean_projection(ax_3d):
    ax_3d.grid(False)
    ax_3d.set_facecolor("white")
    ax_3d.set_xticks([])
    ax_3d.set_yticks([])
    ax_3d.set_zticks([])
    ax_3d.axis("off")


def make_arrow_axes(ax_3d):
    ax_3d.grid(False)
    ax_3d.set_facecolor("white")
    ax_3d.set_xticks([])
    ax_3d.set_yticks([])
    ax_3d.set_zticks([])
    ax_3d.axis("off")

    # Get axis limits
    x0, x1 = ax_3d.get_xlim3d()
    y0, y1 = ax_3d.get_ylim3d()
    z0, z1 = ax_3d.get_zlim3d()

    ax_3d.set_box_aspect((x1 - x0, y1 - y0, z1 - z0))
    # Define arrows along the three frame edges
    edges = [
        ((x0, y0, z0), (x1, y0, z0), "X"),
        ((x0, y0, z0), (x0, y1, z0), "Y"),
        ((x0, y0, z0), (x0, y0, z1), "Z"),
    ]

    for (xs, ys, zs), (xe, ye, ze), label in edges:
        arr = Arrow3D(
            [xs, xe],
            [ys, ye],
            [zs, ze],
            mutation_scale=20,
            lw=1.5,
            arrowstyle="-|>",
            color="black",
        )
        ax_3d.add_artist(arr)
        ax_3d.text(xe * 1.03, ye * 1.03, ze * 1.03, label, fontsize=12)

    # Hide the default frame and ticks
    for pane in (ax_3d.xaxis.pane, ax_3d.yaxis.pane, ax_3d.zaxis.pane):
        pane.set_visible(False)
    ax_3d.view_init(elev=30, azim=30)


def plot_trajs_multivariate(
    trajectories: np.ndarray,
    save_dir: str | None = None,
    plot_name: str = "dyst",
    samples_subset: list[int] | None = None,
    plot_projections: bool = False,
    standardize: bool = False,
    dims_3d: list[int] = [0, 1, 2],
    figsize: tuple[int, int] = (6, 6),
    max_samples: int = 6,
    show_plot: bool = False,
) -> None:
    """
    Plot multivariate timeseries from dyst_data

    Args:
        trajectories (np.ndarray): Array of shape (n_samples, n_dimensions, n_timesteps) containing the multivariate time series data.
        save_dir (str, optional): Directory to save the plots. Defaults to None.
        plot_name (str, optional): Base name for the saved plot files. Defaults to "dyst".
        samples_subset (list[int] | None): Subset of sample indices to plot. If None, all samples are used. Defaults to None.
        plot_projections (bool): Whether to plot 2D projections on the coordinate planes
        standardize (bool): Whether to standardize the trajectories
        dims_3d (list[int]): Indices of dimensions to plot in 3D visualization. Defaults to [0, 1, 2]
        figsize (tuple[int, int]): Figure size in inches (width, height). Defaults to (6, 6)
        max_samples (int): Maximum number of samples to plot. Defaults to 6.
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    assert trajectories.shape[1] >= len(dims_3d), (
        f"Data has {trajectories.shape[1]} dimensions, but {len(dims_3d)} dimensions were requested for plotting"
    )

    n_samples_plot = min(max_samples, trajectories.shape[0])

    if samples_subset is not None:
        if n_samples_plot > len(samples_subset):
            warnings.warn(
                f"Number of samples to plot is greater than the number of samples in the subset. Plotting all {len(samples_subset)} samples in the subset."
            )
            n_samples_plot = len(samples_subset)

    if standardize:
        trajectories = safe_standardize(trajectories)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    if n_samples_plot == 1:
        linewidth = 1
    else:
        linewidth = 0.5

    for sample_idx in range(n_samples_plot):
        label_sample_idx = (
            samples_subset[sample_idx] if samples_subset is not None else sample_idx
        )
        label = f"Sample {label_sample_idx}"
        curr_color = DEFAULT_COLORS[sample_idx % len(DEFAULT_COLORS)]

        xyz = trajectories[sample_idx, dims_3d, :]
        ax.plot(*xyz, alpha=0.5, linewidth=linewidth, color=curr_color, label=label)

        ic_pt = xyz[:, 0]
        ax.scatter(*ic_pt, marker="*", s=100, alpha=0.5, color=curr_color)

        end_pt = xyz[:, -1]
        ax.scatter(*end_pt, marker="x", s=100, alpha=0.5, color=curr_color)

    if plot_projections:
        x_min, x_max = ax.get_xlim3d()  # type: ignore
        y_min, y_max = ax.get_ylim3d()  # type: ignore
        z_min, z_max = ax.get_zlim3d()  # type: ignore
        palpha = 0.1  # whatever

        for sample_idx in range(n_samples_plot):
            label_sample_idx = (
                samples_subset[sample_idx] if samples_subset is not None else sample_idx
            )
            curr_color = DEFAULT_COLORS[sample_idx % len(DEFAULT_COLORS)]
            xyz = trajectories[sample_idx, dims_3d, :]
            ic_pt = xyz[:, 0]
            end_pt = xyz[:, -1]

            # XY plane projection (bottom)
            ax.plot(xyz[0], xyz[1], z_min, alpha=palpha, linewidth=1, color=curr_color)
            ax.scatter(
                ic_pt[0], ic_pt[1], z_min, marker="*", alpha=palpha, color=curr_color
            )
            ax.scatter(
                end_pt[0], end_pt[1], z_min, marker="x", alpha=palpha, color=curr_color
            )

            # XZ plane projection (back)
            ax.plot(xyz[0], y_max, xyz[2], alpha=palpha, linewidth=1, color=curr_color)
            ax.scatter(
                ic_pt[0], y_max, ic_pt[2], marker="*", alpha=palpha, color=curr_color
            )
            ax.scatter(
                end_pt[0], y_max, end_pt[2], marker="x", alpha=palpha, color=curr_color
            )

            # YZ plane projection (right)
            ax.plot(x_min, xyz[1], xyz[2], alpha=palpha, linewidth=1, color=curr_color)
            ax.scatter(
                x_min, ic_pt[1], ic_pt[2], marker="*", alpha=palpha, color=curr_color
            )
            ax.scatter(
                x_min, end_pt[1], end_pt[2], marker="x", alpha=palpha, color=curr_color
            )

    save_path = (
        os.path.join(save_dir, f"{plot_name}_3D.png") if save_dir is not None else None
    )
    if save_path is not None:
        print(f"Saving 3D plot to {save_path}")
    ax.set_xlabel(f"dim_{dims_3d[0]}")
    ax.set_ylabel(f"dim_{dims_3d[1]}")
    ax.set_zlabel(f"dim_{dims_3d[2]}")  # type: ignore
    plt.legend()
    ax.tick_params(pad=3)
    ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="both")
    plt.title(plot_name.replace("_", " "))
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    if show_plot:
        print("Showing plot")
        plt.show()
    plt.close()


def plot_grid_trajs_multivariate(
    ensemble: dict[str, np.ndarray],
    save_path: str | None = None,
    dims_3d: list[int] = [0, 1, 2],
    sample_indices: list[int] | np.ndarray | None = None,
    n_rows_cols: tuple[int, int] | None = None,
    subplot_size: tuple[int, int] = (3, 3),
    row_col_padding: tuple[float, float] = (0.0, 0.0),
    plot_kwargs: dict[str, Any] = {},
    title_kwargs: dict[str, Any] = {},
    custom_colors: list[str] = [],
    show_titles: bool = True,
    show_axes: bool = False,
    plot_projections: bool = False,
    projections_alpha: float = 0.1,
) -> None:
    n_systems = len(ensemble)
    if n_rows_cols is None:
        n_rows = int(np.ceil(np.sqrt(n_systems)))
        n_cols = int(np.ceil(n_systems / n_rows))
    else:
        n_rows, n_cols = n_rows_cols

    row_padding, column_padding = row_col_padding
    # Reduce spacing by using smaller padding multipliers
    figsize = (
        n_cols * subplot_size[0] * (1 + column_padding),
        n_rows * subplot_size[1] * (1 + row_padding),
    )
    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(wspace=column_padding, hspace=row_padding)

    if sample_indices is None:
        sample_indices = np.zeros(len(ensemble), dtype=int)
    # Keep track of the last used color index to avoid consecutive same colors
    last_color_idx = -1

    for i, (system_name, trajectories) in enumerate(ensemble.items()):
        assert trajectories.shape[1] >= len(dims_3d), (
            f"Data has {trajectories.shape[1]} dimensions, but {len(dims_3d)} dimensions were requested for plotting"
        )

        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection="3d")

        sample_idx = sample_indices[i]
        xyz = trajectories[sample_idx, dims_3d, :]

        # Select a color that's different from the last one used
        if len(custom_colors) > 0:
            if len(custom_colors) > 1:
                # Get a new color index that's different from the last one
                available_indices = [
                    j for j in range(len(custom_colors)) if j != last_color_idx
                ]
                color_idx = np.random.choice(available_indices)
                last_color_idx = color_idx
            else:
                # If only one color is available, use it
                color_idx = 0
        else:
            color_idx = 0
        ax.plot(
            *xyz,
            **plot_kwargs,
            color=custom_colors[color_idx] if len(custom_colors) > 0 else None,
            zorder=10,
        )

        if show_titles:
            system_name_title = system_name.replace("_", " + ")
            ax.set_title(f"{system_name_title}", **title_kwargs)
        fig.patch.set_facecolor("white")  # Set the figure's face color to white
        ax.set_facecolor("white")  # Set the axes' face color to white
        # Hide tick marks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])  # type: ignore

        if not show_axes:
            ax.set_axis_off()
        ax.grid(False)

        if plot_projections:
            x_min, x_max = ax.get_xlim3d()  # type: ignore
            y_min, y_max = ax.get_ylim3d()  # type: ignore
            z_min, z_max = ax.get_zlim3d()  # type: ignore

            proj_color = "black"
            proj_linewidth = 0.3

            # XY plane projection (bottom)
            ax.plot(
                xyz[0],
                xyz[1],
                z_min,
                alpha=projections_alpha,
                linewidth=proj_linewidth,
                color=proj_color,
            )

            # XZ plane projection (back)
            ax.plot(
                xyz[0],
                y_max,
                xyz[2],
                alpha=projections_alpha,
                linewidth=proj_linewidth,
                color=proj_color,
            )

            # YZ plane projection (right)
            ax.plot(
                x_min,
                xyz[1],
                xyz[2],
                alpha=projections_alpha,
                linewidth=proj_linewidth,
                color=proj_color,
            )

    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()
    plt.close()


def make_box_plot(
    unrolled_metrics: dict[str, dict[int, dict[str, list[float]]]],
    prediction_length: int,
    metric_to_plot: str = "smape",
    selected_run_names: list[str] | None = None,
    ylim: tuple[float, float] | None = None,
    verbose: bool = False,
    run_names_to_exclude: list[str] = [],
    use_inv_spearman: bool = False,
    title: str | None = None,
    fig_kwargs: dict[str, Any] = {},
    title_kwargs: dict[str, Any] = {},
    colors: list[str] = DEFAULT_COLORS,
    sort_runs: bool = False,
    save_path: str | None = None,
    order_by_metric: str | None = None,
    ylabel_fontsize: int = 8,
    show_xlabel: bool = True,
    show_legend: bool = False,
    legend_kwargs: dict[str, Any] = {},
    alpha_val: float = 0.8,
    box_percentile_range: tuple[int, int] = (25, 75),
    whisker_percentile_range: tuple[float, float] = (5, 95),
    box_width: float = 0.6,
    has_nans: dict[str, bool] | None = None,
) -> list[mpatches.Patch] | None:
    if fig_kwargs == {}:
        fig_kwargs = {"figsize": (3, 5)}

    if selected_run_names is None:
        selected_run_names = list(unrolled_metrics.keys())

    run_names = [
        name for name in selected_run_names if name not in run_names_to_exclude
    ]
    has_nans = has_nans or {}

    if len(run_names) == 0:
        print("No run names to plot after exclusions!")
        return

    plt.figure(**fig_kwargs)
    plot_data = []

    ordering_metric_data = {}

    for run_name in run_names:
        try:
            if prediction_length not in unrolled_metrics[run_name]:
                warnings.warn(
                    f"Warning: prediction_length {prediction_length} not found for {run_name}"
                )
                continue

            # Check if this run has the specified metric
            if metric_to_plot not in unrolled_metrics[run_name][prediction_length]:
                warnings.warn(
                    f"Warning: metric '{metric_to_plot}' not found for {run_name}"
                )
                continue

            values = unrolled_metrics[run_name][prediction_length][metric_to_plot]
            has_nans[run_name] = bool(np.isnan(values).any()) or has_nans.get(
                run_name, False
            )

            # Process values based on metric type
            if metric_to_plot == "spearman" and use_inv_spearman:
                values = [1 - x for x in values]

            # Filter out NaN values
            values = [v for v in values if not np.isnan(v)]

            if len(values) == 0:
                warnings.warn(f"Warning: All values for {run_name} are NaN")
                continue

            median_value = np.median(values)
            plot_data.extend([(run_name, v) for v in values])

            # If we need to order by a different metric, collect that data too
            if order_by_metric is not None and order_by_metric != metric_to_plot:
                if order_by_metric in unrolled_metrics[run_name][prediction_length]:
                    order_values = unrolled_metrics[run_name][prediction_length][
                        order_by_metric
                    ]

                    # Apply same processing as we would for the plotting metric
                    if order_by_metric == "spearman" and use_inv_spearman:
                        order_values = [1 - x for x in order_values]

                    # Filter out NaN values
                    order_values = [v for v in order_values if not np.isnan(v)]

                    if order_values:
                        ordering_metric_data[run_name] = np.median(order_values)

            if verbose:
                print(f"{run_name} median {metric_to_plot}: {median_value}")

        except Exception as e:
            warnings.warn(f"Error processing {run_name}: {e}")

    df = pd.DataFrame(plot_data, columns=["Run", "Value"])

    if order_by_metric is not None and ordering_metric_data:
        run_order = [
            run for run, _ in sorted(ordering_metric_data.items(), key=lambda x: x[1])
        ]
        run_order = [run for run in run_order if run in df["Run"].unique()]
        df["Run"] = pd.Categorical(df["Run"], categories=run_order, ordered=True)
    elif sort_runs:
        # Use the existing sort_runs logic if order_by_metric isn't specified
        median_by_run = df.groupby("Run")["Value"].median().sort_values()
        run_order = median_by_run.index.tolist()
        df["Run"] = pd.Categorical(df["Run"], categories=run_order, ordered=True)

    metric_title = metric_to_plot
    if metric_to_plot in ["mse", "mae", "rmse", "mape"]:
        metric_title = metric_to_plot.upper()
    elif metric_to_plot == "smape":
        metric_title = "sMAPE"
    elif metric_to_plot == "spearman":
        metric_title = "1 - Spearman" if use_inv_spearman else "Spearman"
    else:
        metric_title = metric_to_plot.capitalize()

    ax = plt.gca()
    unique_runs = (
        df["Run"].unique()
        if not isinstance(df["Run"].dtype, pd.CategoricalDtype)
        else df["Run"].cat.categories
    )

    for i, run in enumerate(unique_runs):
        run_data = df[df["Run"] == run]["Value"].to_numpy()
        if len(run_data) == 0:
            continue

        # Calculate the percentiles
        lower_box, upper_box = np.percentile(run_data, box_percentile_range)
        lower_whisker, upper_whisker = np.percentile(run_data, whisker_percentile_range)
        median_val = np.median(run_data)
        if isinstance(colors, dict):
            color = colors[run]
        else:
            color = colors[i % len(colors)]  # type: ignore
        box_half_width = box_width / 2
        whisker_cap_width = box_half_width * 0.5

        box = plt.Rectangle(
            (i - box_half_width, lower_box),
            box_width,
            upper_box - lower_box,
            fill=True,
            facecolor=color,
            alpha=alpha_val,
            linewidth=1,
            edgecolor="black",
            zorder=5,
        )
        ax.add_patch(box)

        ax.hlines(
            median_val,
            i - box_half_width,
            i + box_half_width,
            colors="black",
            linewidth=2.5,
            zorder=10,
        )

        ax.vlines(
            i,
            lower_box,
            lower_whisker,
            colors="black",
            linestyle="-",
            linewidth=1,
            zorder=5,
        )
        ax.vlines(
            i,
            upper_box,
            upper_whisker,
            colors="black",
            linestyle="-",
            linewidth=1,
            zorder=5,
        )

        ax.hlines(
            lower_whisker,
            i - whisker_cap_width,
            i + whisker_cap_width,
            colors="black",
            linewidth=1,
            zorder=5,
        )
        ax.hlines(
            upper_whisker,
            i - whisker_cap_width,
            i + whisker_cap_width,
            colors="black",
            linewidth=1,
            zorder=5,
        )

    if ylim:
        plt.ylim(ylim)

    plt.ylabel(metric_title, fontweight="bold", fontsize=ylabel_fontsize)
    plt.xlabel("")
    if show_xlabel:
        plt.xticks(
            range(len(unique_runs)),
            unique_runs.tolist(),
            rotation=45,
            ha="right",
            fontsize=5,
            fontweight="bold",
        )
    else:
        plt.xticks([])

    if title is not None:
        title_with_metric = f"{title}: {metric_title}" if title == "Metrics" else title
        plt.title(title_with_metric, fontweight="bold", **title_kwargs)

    plt.tight_layout()
    if isinstance(df["Run"].dtype, pd.CategoricalDtype):
        runs = df["Run"].cat.categories.tolist()
    else:
        runs = df["Run"].unique().tolist()

    # Add a dagger to the run name if it has NaNs
    runs = [f"{run}$^\dagger$" if has_nans[run] else run for run in runs]

    if isinstance(colors, dict):
        legend_handles = [
            mpatches.Patch(color=colors[run], label=run, alpha=alpha_val)  # type: ignore
            for run in runs
        ]
    else:
        legend_handles = [
            mpatches.Patch(color=colors[i % len(colors)], label=run, alpha=alpha_val)  # type: ignore
            for i, run in enumerate(runs)
        ]

    if show_legend:
        plt.legend(handles=legend_handles, **legend_kwargs)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()
    return legend_handles


def plot_all_metrics_by_prediction_length(
    all_metrics_dict: dict[str, dict[str, dict[str, list[float]]]],
    metric_names: list[str],
    stat_to_plot: Literal["mean", "median"] = "median",
    metrics_to_show_envelope: list[str] = [],
    percentile_range: tuple[int, int] = (25, 75),
    n_rows: int = 2,
    n_cols: int = 3,
    individual_figsize: tuple[int, int] = (4, 4),
    save_path: str | None = None,
    ylim: tuple[float | None, float | None] = (None, None),
    show_legend: bool = True,
    legend_kwargs: dict = {},
    colors: list[str] | dict[str, str] = DEFAULT_COLORS,
    markers: list[str] = DEFAULT_MARKERS,
    use_inv_spearman: bool = False,
    model_names_to_exclude: list[str] = [],
    has_nans: dict[str, dict[str, bool]] | None = None,
    replace_nans_with_val: float | None = None,
) -> list[plt.Line2D]:
    """
    Plot multiple metrics across different prediction lengths for various models.

    Parameters:
    -----------
    all_metrics_dict : dict[str, dict[str, dict[str, list[float]]]]
        A nested dictionary with the following structure:
        - First level keys are metric names (e.g., 'mse', 'mae', 'smape', 'spearman')
        - Second level keys are model names (e.g., 'Our Model', 'Chronos 20M', 'TimesFM 200M')
        - Third level contains metric data with keys like:
          - 'prediction_lengths': list of prediction lengths
          - 'means': mean values for each prediction length
          - 'medians': median values for each prediction length
          - 'stds': standard deviations for each prediction length
          - 'stes': standard error of the mean for each prediction length
          - 'all_vals': list of all values for each prediction length

    metric_names : list[str]
        List of metric names to plot (must be keys in all_metrics_dict)

    model_names_to_exclude : list[str], default=[]
        List of model names to exclude from the plot

    stat_to_plot : Literal["mean", "median"], default="median"
        Statistic to plot

    metrics_to_show_envelope : list[str]
        List of metrics for which to show envelopes
        if stat_to_plot == "mean", then show standard *error* envelopes
        if stat_to_plot == "median", then show median envelopes

    percentile_range : tuple[int, int], default=(25, 75)
        Percentile range to use for the median envelope, only used when stat_to_plot == "median"

    n_rows : int, default=2
        Number of rows in the subplot grid

    n_cols : int, default=3
        Number of columns in the subplot grid

    individual_figsize : tuple[int, int], default=(4, 4)
        Size of each individual subplot

    save_path : str | None, default=None
        Path to save the figure, if None, the figure is not saved

    ylim : tuple[float | None, float | None], default=(None, None)
        Y-axis limits for all subplots

    show_legend : bool, default=True
        Whether to show the legend

    legend_kwargs : dict, default={}
        Additional keyword arguments for the legend

    colors : list[str] | dict[str, str], default=default_colors
        List of colors to use for different models
        Or, dict of model names to colors

    markers : list[str], default=markers
        List of markers to use for different models

    use_inv_spearman : bool, default=False
        If True, plot 1 - Spearman correlation instead of Spearman correlation

    Returns:
    --------
    list[plt.Line2D]
        Legend handles for the plotted lines
    """
    has_nans = has_nans or {}
    num_metrics = len(metric_names)
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(individual_figsize[0] * n_cols, individual_figsize[1] * n_rows),
    )
    legend_handles = []
    # Handle the case where axes might be a single element or already a list
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif hasattr(axes, "flatten"):  # Check if axes has flatten method
        axes = axes.flatten()  # Flatten the axes array for easy iteration

    for i, (ax, metric_name) in enumerate(zip(axes, metric_names)):
        metrics_dict = all_metrics_dict[metric_name]
        nan_models = has_nans.get(metric_name, {})
        for j, (model_name, metrics) in enumerate(metrics_dict.items()):
            has_nan = nan_models.get(model_name, False)
            if has_nan:
                print(f"{model_name} has NaNs for {metric_name}")
            if model_name in model_names_to_exclude:
                continue
            mean_vals = np.array(metrics["means"])
            median_vals = np.array(metrics["medians"])
            all_vals = metrics[
                "all_vals"
            ]  # Keep as list of arrays to avoid inhomogeneous shape error
            if replace_nans_with_val is not None:
                # Filter NaN values from each array in all_vals
                all_vals = [
                    np.array(
                        [v if not np.isnan(v) else replace_nans_with_val for v in val]
                    )
                    for val in all_vals
                ]
                # recompute the mean and median and standard error
                mean_vals = np.array([np.mean(val) for val in all_vals])
                median_vals = np.array([np.median(val) for val in all_vals])
                se_envelope = np.array(
                    [np.std(val) / np.sqrt(len(val)) for val in all_vals]
                )

            if metric_name == "spearman" and use_inv_spearman:
                mean_vals = 1 - mean_vals
                median_vals = 1 - median_vals
                all_vals = [1 - val for val in all_vals]

            color = colors[j] if isinstance(colors, list) else colors[model_name]
            if stat_to_plot == "mean":
                ax.plot(
                    metrics["prediction_lengths"],
                    mean_vals,
                    marker=markers[j],
                    label=model_name,
                    markersize=6,
                    color=color,
                    markerfacecolor="none" if has_nan else color,
                    linestyle="-." if has_nan else "-",
                )
                if metric_name in metrics_to_show_envelope:
                    se_envelope = np.array(metrics["stes"])
                    ax.fill_between(
                        metrics["prediction_lengths"],
                        mean_vals - se_envelope,
                        mean_vals + se_envelope,
                        alpha=0.1,
                        color=color,
                    )
            elif stat_to_plot == "median":
                ax.plot(
                    metrics["prediction_lengths"],
                    median_vals,
                    marker=markers[j],
                    label=model_name,
                    markersize=6,
                    color=color,
                    markerfacecolor="none" if has_nan else color,
                    linestyle="-." if has_nan else "-",
                )
                if metric_name in metrics_to_show_envelope:
                    percentile_range_lower = [
                        np.percentile(all_vals[pred_len_idx], percentile_range[0])
                        for pred_len_idx in range(len(all_vals))
                    ]
                    percentile_range_upper = [
                        np.percentile(all_vals[pred_len_idx], percentile_range[1])
                        for pred_len_idx in range(len(all_vals))
                    ]

                    ax.fill_between(
                        metrics["prediction_lengths"],
                        percentile_range_lower,
                        percentile_range_upper,
                        alpha=0.1,
                        color=color,
                    )

        if i == 0:
            legend_handles = [
                plt.Line2D(
                    [0],
                    [0],
                    color=colors[j] if isinstance(colors, list) else colors[model_name],
                    marker=markers[j],
                    markersize=6,
                    label=model_name
                    if not nan_models.get(model_name, False)
                    else f"{model_name}$^\dagger$",
                    linestyle="-." if nan_models.get(model_name, False) else "-",
                    markerfacecolor="none"
                    if nan_models.get(model_name, False)
                    else colors[j]
                    if isinstance(colors, list)
                    else colors[model_name],
                )
                for j, model_name in enumerate(metrics_dict.keys())
            ]
            if show_legend:
                legend_handles = ax.legend(handles=legend_handles, **legend_kwargs)
        ax.set_xlabel("Prediction Length", fontweight="bold", fontsize=12)
        ax.set_xticks(metrics["prediction_lengths"])
        name = metric_name.replace("_", " ")
        if name in ["mse", "mae", "rmse", "mape"]:
            name = name.upper()
        elif name == "smape":
            name = "sMAPE"
        elif name == "spearman" and use_inv_spearman:
            name = "1 - Spearman"
        else:
            name = name.capitalize()
        ax.set_title(name, fontweight="bold", fontsize=16)

    # Hide any unused subplots
    for ax in axes[num_metrics:]:
        ax.set_visible(False)
    if ylim is not None:
        for ax in axes:
            ax.set_ylim(ylim)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()

    return legend_handles
