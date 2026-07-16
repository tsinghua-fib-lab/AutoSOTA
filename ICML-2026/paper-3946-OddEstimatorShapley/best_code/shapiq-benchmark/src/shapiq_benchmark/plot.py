"""This module contains the plotting utilities for the benchmark results."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from collections.abc import Callable

    from shapiq.approximator.base import Approximator

__all__ = [
    "abbreviate_application_name",
    "create_application_name",
    "get_game_title_name",
    "plot_approximation_quality",
]


STYLE_DICT: dict[str, dict[str, str]] = {
    # permutation sampling
    "PermutationSamplingSII": {"color": "#7d53de", "marker": "o"},
    "PermutationSamplingSTII": {"color": "#7d53de", "marker": "o"},
    "PermutationSamplingSV": {"color": "#7d53de", "marker": "o"},
    # KernelSHAP-IQ
    #"KernelSHAP": {"color": "#ff6f00", "marker": "o"},
    "KernelSHAPIQ": {"color": "#ff6f00", "marker": "o"},
    # inconsistent KernelSHAP-IQ
    "InconsistentKernelSHAPIQ": {"color": "#ffba08", "marker": "o"},
    "kADDSHAP": {"color": "#ffba08", "marker": "o"},
    # SVARM-based
    "SVARMIQ": {"color": "#00b4d8", "marker": "o"},
    #"SVARM": {"color": "#00b4d8", "marker": "o"},
    # shapiq
    "SHAPIQ": {"color": "#ef27a6", "marker": "o"},
    "UnbiasedKernelSHAP": {"color": "#ef27a6", "marker": "o"},
    # misc SV
    "OwenSamplingSV": {"color": "#7DCE82", "marker": "o"},
    "StratifiedSamplingSV": {"color": "#4B7B4E", "marker": "o"},
    # Regression MSR
    # PolySHAP plots
    "PermutationSampling": {"color": "#555555", "marker": "o"},
    "MSR": {"color": "#D0D0D0", "marker": "o"},
    "SVARM": {"color": "#8C8C8C", "marker": "o"},
    "ProxySpex": {"color": "#AA0060", "marker": "o"},
    "KernelSHAP": {"color": "#d62728", "marker": "o"},
    "LeverageSHAP": {"color": "#009688", "marker": "o"},
    "RegressionMSR": {"color": "#0F0F0F", "marker": "o"},
    "PolySHAP-3ADD": {"color": "#7CB342", "marker": "o"},
    "RegressionMSR-LGBM": {"color": "#3F51B5", "marker": "o"},
    "Proxy-LGBM": {"color": "#B0BEC5", "marker": "o"},
    "OddSHAP-Fourier-ProxySPEX": {"color": "#AA0060", "marker": "o"},
    "OddSHAP-Fourier-Random": {"color": "#6e6e6e", "marker": "o", "linestyle":"dashed"},
    "OddSHAP-Fourier-ProxySPEX-NoCV": {"color": "#E64A19", "marker": "o"},
    "OddSHAP-Fourier-ProxySPEX-NoCV-5": {"color": "#FFB74D", "marker": "o"},
    "OddSHAP-Fourier-ProxySPEX-NoCV-20": {"color": "#BF360C", "marker": "o"},
    "All-Fourier-ProxySPEX-NoCV": {"color": "black", "marker": "o"},
    "FFD-RD": {"color": "#6A0DAD", "marker": "D"},
    "FFD-RD-Corrected": {"color": "#C084FC", "marker": "D"},
    "FourierSHAP": {"color": "#F9A825", "marker": "P"},
}
STYLE_DICT = defaultdict(lambda: {"color": "black", "marker": "o"}, STYLE_DICT)
MARKERS = []
LIGHT_GRAY = "#d3d3d3"
LINE_STYLES_ORDER = {
    0: "solid",
    1: "solid",
    2: "solid",
    3: "dashed",
    4: "dashdot",
    "all": "solid",
}
LINE_MARKERS_ORDER = {0: "o", 1: "o", 2: "o", 3: "X", 4: "d", "all": "o"}
LINE_THICKNESS = 2
MARKER_SIZE = 7


LOG_SCALE_MAX = 1e5
LOG_SCALE_MIN = 1e-9

METRICS_LIMITS = {
    "Precision@10": (0, 1),
    "Precision@5": (0, 1),
    "KendallTau": (-1, 1),
    "KendallTau@5": (-1, 1),
    "KendallTau@10": (-1, 1),
    "KendallTau@50": (-1, 1),
}
METRICS_NOT_TO_LOG_SCALE = list(METRICS_LIMITS.keys())


def create_application_name(setup: str, *, abbrev: bool = False) -> str:
    """Create an application name from the setup string."""
    application_name = "".join(setup.split("_")[0:2])
    application_name = application_name.replace("Game", "")
    application_name = application_name.replace("SynthData", "")
    application_name = application_name.replace("AdultCensus", "")
    application_name = application_name.replace("CaliforniaHousing", "")
    application_name = application_name.replace("BikeSharing", "")
    application_name = application_name.replace("ImageClassifier", "LocalExplanation")
    application_name = application_name.replace("SentimentAnalysis", "LocalExplanation")
    application_name = application_name.replace("TreeSHAPIQXAI", "LocalExplanation")
    application_name = application_name.replace(
        "RandomForestEnsembleSelection",
        "EnsembleSelection",
    )
    if abbrev:
        application_name = abbreviate_application_name(application_name)
    return application_name


def abbreviate_application_name(
    application_name: str, *, new_line: bool = False
) -> str:
    """Abbreviate the application name.

    Abbreviate the application name by taking the first three characters after each capital
    letter and adding a dot. The last character is not abbreviated.

    Args:
        application_name: The application name to abbreviate.
        new_line: Whether to add a new line after each abbreviation. Defaults to ``False``.

    Returns:
        The abbreviated application name.

    Example:
        >>> abbreviate_application_name("LocalExplanation")
        "Loc. Exp."

    """
    abbreviations = []
    count_char = 0
    for char in application_name:
        if char.isupper():
            count_char = 0
            abbreviations.append(char)
        else:
            count_char += 1
            if count_char == 3:
                abbreviations.append(".")
            elif count_char > 3:
                continue
            else:
                abbreviations.append(char)
    abbreviation = "".join(abbreviations)
    if application_name == "DatasetValuation":
        abbreviation = "Dst. Val."
    if application_name == "SOUM":
        abbreviation = "SOUM"
    if application_name == "SOUM (low)" and new_line:
        abbreviation = "SOUM\n(low)"
    if application_name == "SOUM (high)":
        abbreviation = "SOUM\n(high)"
    if new_line:
        abbreviation = abbreviation.replace(".", ".\n")
    return abbreviation.strip()


def get_game_title_name(game_name: str) -> str:
    """Changes the game name to a more readable title.

    Args:
        game_name: The game name to change.

    Returns:
        The game title name.

    Example:
        >>> get_game_title_name("ImageClassifierLocalXAI")
        "Image Classifier Local XAI"
        >>> get_game_title_name("AdultCensusClusterExplanation")
        "Adult Census Cluster Explanation"

    """
    # split words by capital letters
    words = ""
    for char in game_name:
        if char.isupper():
            words += " "
        words += char
    words = words.replace("Tree S H A P I Q", "TreeSHAPIQ")  # TreeSHAPIQ
    words = words.replace("X A I", "XAI")  # XAI
    words = words.replace("S O U M", "SOUM")  # SOUM
    return words.strip()


def agg_percentile(q: float) -> Callable[[np.ndarray], np.floating]:
    """Get the aggregation function for the given percentile.

    Args:
        q: The percentile to compute.

    Returns:
        The aggregation function.

    """

    def quantile(x: np.ndarray) -> np.floating:
        """Performs the aggregation function for the given percentile."""
        return np.percentile(x, q)

    quantile.__name__ = f"quantile_{q}"
    return quantile


def plot_approximation_quality(
    data: pd.DataFrame | None = None,
    *,
    data_path: str | None = None,
    metric: str = "MSE",
    orders: list[int | str] | None = None,
    approximators: list[str] | None = None,
    aggregation: str = "median",
    confidence_metric: str | None = "IQR",
    log_scale_y: bool = False,
    log_scale_min: float = LOG_SCALE_MIN,
    log_scale_max: float = LOG_SCALE_MAX,
    legend: bool = True,
    remove_spines: bool = False,
    x_column: str = "used_budget",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the approximation quality curves.

    Args:
        data: The data to plot the values from (if `None`, the data_path must be provided).
        data_path: The path to the data to plot the values from (if `None`, the data must be
            provided).
        metric: The metric to plot. Defaults to "MSE".
        orders: The orders to plot. If `None`, all orders are plotted. Defaults to `None`.
            Can be a list of integers or a single integer.
        approximators: The approximators to plot. Defaults to `None`. When `None`, all approximators
            are plotted.
        aggregation: The aggregation function to plot for the metric values. Defaults to "mean".
            Available options are "mean", "median", "quantile_95", "quantile_5".
        confidence_metric: The metric to use for the confidence interval. Defaults to "sem".
            Available options are "sem", "std", "var", "quantile_95", "quantile_5".
        log_scale_y: Whether to use a log scale for the y-axis. Defaults to `False`.
        log_scale_min: The minimum value for the log scale. Defaults to 1e-7.
        log_scale_max: The maximum value for the log scale. Defaults to 1e2.
        legend: Whether to add a legend to the plot. Defaults to `True`.
        remove_spines: Whether to remove the spines in the top and right of the plot. Defaults to
            `False`.

    Returns:
        The figure and axes of the plot.

    """
    if data_path is None and data is None:
        msg = "Either data or data_path must be provided."
        raise ValueError(msg)

    if data is None:
        data = pd.read_csv(data_path)
    # remove exact
    data = data[~data["approximator"].str.contains("Exact")]

    aggregation_keys = ["approximator", "used_budget", "iteration","id_config_approximator"]
    if x_column == "runtime_median":
        # add median runtime to aggregation keys, if x-axis should be runtime
        aggregation_keys.append("runtime_median")
    # get the metric data
    metric_data = get_metric_data(data, aggregation_keys, metric)
    sorted_budget = list(data["budget"].sort_values(ascending=False).unique())

    min_value_y = data[metric].min()
    bot_lim = min_value_y/10
    log_scale_min = max(bot_lim,LOG_SCALE_MIN)

    # make sure orders is a list
    if orders is None:
        orders = ["all"]
    if isinstance(orders, int | str):
        orders = [orders]

    # make sure approximators is a list
    if approximators is None:
        approximators = list(metric_data["approximator"].unique())

    # set the confidence metrics
    confidence_metric_low, confidence_metric_high = confidence_metric, confidence_metric
    if confidence_metric is not None and "quantile" in confidence_metric:
        confidence_metric_low = "quantile_5"
        confidence_metric_high = "quantile_95"
    if confidence_metric is not None and "IQR" in confidence_metric:
        confidence_metric_low = "quantile_25"
        confidence_metric_high = "quantile_75"

    # create the plot
    fig, ax = plt.subplots()
    approx_max_budget = 0
    for approximator in approximators:
        for order in metric_data["order"].unique():
            if orders is not None and order not in orders:
                continue
            data_order = metric_data[
                (metric_data["approximator"] == approximator)
                & (metric_data["order"] == order)
            ].copy()

            if log_scale_y:
                # manually set all below log_scale_min to log_scale_min (to avoid log(0))
                # clamp aggregation (median/mean) itself
                data_order[aggregation] = data_order[aggregation].apply(lambda x: max(x, log_scale_min))

                # clamp the lower/higher confidence columns individually (if they exist)
                if confidence_metric_low in data_order.columns:
                    data_order[confidence_metric_low] = data_order[confidence_metric_low].apply(
                        lambda x: max(x, log_scale_min)
                    )
                if confidence_metric_high in data_order.columns:
                    data_order[confidence_metric_high] = data_order[confidence_metric_high].apply(
                        lambda x: max(x, log_scale_min)
                    )

            # base color for this approximator
            color = STYLE_DICT[approximator]["color"]

            cfg_ids = sorted(data_order["id_config_approximator"].unique())
            for cfg in cfg_ids:
                data_cfg = data_order[data_order["id_config_approximator"] == cfg]
                if data_cfg.empty:
                    continue

                # choose linestyle: override for 37 and 39, otherwise keep order default
                if int(cfg) == 39:
                    line_style = "dashed"
                elif int(cfg) == 37:
                    line_style = "solid"
                else:
                    line_style = LINE_STYLES_ORDER[order]

                line_marker = LINE_MARKERS_ORDER[order]

                x_axis_values = data_cfg[x_column]

                # plot the mean values for this config (no label so legend remains per-approximator)
                ax.plot(
                    x_axis_values,
                    data_cfg[aggregation],
                    color=color,
                    linestyle=line_style,
                    marker=line_marker,
                    linewidth=LINE_THICKNESS,
                    mec="white",
                    markersize=MARKER_SIZE,
                )

                # error band for this config
                if confidence_metric is not None:
                    if "IQR" in confidence_metric:
                        ax.fill_between(
                            x_axis_values,
                            data_cfg[confidence_metric_low],
                            data_cfg[confidence_metric_high],
                            alpha=0.1,
                            color=color,
                        )
                    else:
                        ax.fill_between(
                            x_axis_values,
                            data_cfg[aggregation] - data_cfg[confidence_metric_low],
                            data_cfg[aggregation] + data_cfg[confidence_metric_high],
                            alpha=0.1,
                            color=color,
                        )

                approx_max_budget = max(approx_max_budget, int(x_axis_values.max()))

    # add x/y labels
    if x_column == "runtime_median":
        ax.set_xlabel(r"Runtime (seconds)",fontsize=20)
    else:
        ax.set_xlabel(r"Budget ($m$)", fontsize=20)

    ax.set_ylabel(metric+" (Median $\pm$ IQR Band)",fontsize=20)
    plt.yticks(fontsize=18)

    # add grid
    ax.grid(True, color=LIGHT_GRAY, linestyle="-")

    # add the legend
    if legend:
        add_legend(ax, approximators, orders=orders)

    # set the y-axis limits
    if log_scale_y and metric not in METRICS_NOT_TO_LOG_SCALE:
        _set_y_axis_log_scale(ax, log_scale_min, log_scale_max)

    # set the y-axis limits for specific metrics
    if metric in METRICS_LIMITS:
        ax.set_ylim(METRICS_LIMITS[metric])

    # determine global x-range (use aggregated metric_data when available)
    if x_column in metric_data.columns:
        global_min_x = float(metric_data[x_column].min())
        global_max_x = float(metric_data[x_column].max())
    else:
        global_min_x = float(data[x_column].min())
        global_max_x = float(data[x_column].max())

    # ensure positive min for log scale
    if global_min_x <= 0:
        # pick a small positive value (not zero) to avoid log issues
        global_min_x = max(LOG_SCALE_MIN, abs(global_max_x) * 1e-6)

    # set x-scale first, then compute/format ticks so tick positions align with scale
    ax.set_xscale("log")
    # set sensible axis limits so the plotted curves start at the actual minimum
    ax.set_xlim(left=global_min_x * 0.8, right=global_max_x * 1.2)



    if remove_spines:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # resize the figure and remove padding
    plt.tight_layout()

    return fig, ax


def _set_y_axis_log_scale(
    ax: plt.Axes, log_scale_min: float, log_scale_max: float
) -> None:
    """Sets the y-axis to a log scale and adjusts the limits."""
    # adjust the top limit to be one order of magnitude higher than the current top limit
    top_lim = ax.get_ylim()[1]
    top_lim = f"{top_lim:.2e}"  # get the top limi in scientific notation
    top_lim = top_lim.split("e")[1]  # get the exponent
    top_lim = int(top_lim) + 1  # get the top limit as the exponent + 1
    top_lim = 10**top_lim  # get the top limit in scientific notation

    top_lim = min(top_lim, log_scale_max)

    # set the y-axis limits
    ax.set_ylim(top=top_lim)
    ax.set_ylim(bottom=log_scale_min*10)
    ax.set_yscale("log")


def get_metric_data(results_df: pd.DataFrame, aggregation_keys: list[str], metric: str = "MSE") -> pd.DataFrame:
    """Get the metric data for the given results.

    Args:
        results_df: The results dataframe.
        metric: The metric to get the data for. Defaults to "MSE".

    Returns:
        The metric data.

    """
    # get the metric columns for each order in the results
    metric_columns = [col for col in results_df.columns if metric == col]
    print(f"Metric columns: {metric_columns}: Metric: {metric}")
    metric_dfs = []
    for metric_col in metric_columns:
        data_order = (
            results_df.groupby(aggregation_keys)
            .agg(
                {
                    metric_col: [
                        "mean",
                        "std",
                        "var",
                        "count",
                        "median",
                        agg_percentile(25),
                        agg_percentile(75),
                        agg_percentile(95),
                        agg_percentile(5),
                    ],
                },
            )
            .reset_index()
        )
        data_order["order"] = (
            "all" if "_" not in metric_col else int(metric_col.split("_")[0])
        )
        # rename the columns of grouped data
        new_columns = [
            "_".join(col).strip() if col[1] != "" else col[0]
            for col in data_order.columns
        ]
        new_columns = [col.replace(f"{metric_col}_", "") for col in new_columns]

        data_order.columns = new_columns
        metric_dfs.append(data_order)

    # concat the dataframes along the row
    metric_df = pd.concat(metric_dfs)

    # compute the standard error
    metric_df["sem"] = (
        metric_df["std"] / metric_df["count"] ** 0.5
    )  # compute standard error

    return metric_df


def add_legend(
    axis: plt.Axes,
    approximators: list[str | Approximator],
    *,
    orders: list[int | str] | None = None,
    legend_subtitle: bool = True,
    loc: str = "best",
) -> None:
    """Add the legend to the plot.

    Args:
        axis: The axes of the plot.
        approximators: The list of approximators to add to the legend.
        orders: The orders to add to the legend. Can be a list of integers or a single integer.
            Defaults to `None`.
        legend_subtitle: Whether to add a subtitle to the legend. Defaults to `True`.
        loc: The location of the legend. Defaults to "upper right".

    """
    if orders is None and approximators is None:
        return

    # convert approximators to strings if they are not strings
    if approximators is not None:
        approximators_str = []
        for approx in approximators:
            approx_str = type(approx).__name__
            if approx_str == "str":
                approx_str = approx
            approximators_str.append(approx_str)
        approximators = approximators_str


    # plot the approximator elements
    if legend_subtitle:
        axis.plot([], [], label="$\\bf{Method}$", color="none")
    for approximator in approximators:
        axis.plot(
            [],
            [],
            label=approximator,
            color=STYLE_DICT[approximator]["color"],
            linewidth=LINE_THICKNESS,
        )
    axis.legend(loc=loc,ncol=1)
