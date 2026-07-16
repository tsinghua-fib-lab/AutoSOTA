from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapiq_benchmark.plot import plot_approximation_quality, STYLE_DICT
import argparse
from collections.abc import Callable

parser = argparse.ArgumentParser(
    description="Run benchmark approximations on explanation games."
)
parser.add_argument(
    "--config_approximators",
    type=int,
    default=37,
    help="Configuration ID for approximators: 40 (PAIRING=False, REPLACEMENT=True), "
    "39 (PAIRING=False, REPLACEMENT=False), "
    "38 (PAIRING=True, REPLACEMENT=True), "
    "37 (PAIRING=True, REPLACEMENT=False). Default is 37.",
)
parser.add_argument(
    "--index",
    type=str,
    default="SV",
    help="Index to compute (default: SV). Options: SV, SII",
)
parser.add_argument(
    "--order", type=int, default=1, help="Order of interaction index (default: 1)."
)
args = parser.parse_args()

DATA_NAMES = {
    "BreastCancerLocalXAI": "Cancer ($d=30$)",
    "CommunitiesAndCrimeLocalXAI": "Crime ($d=101$)",
    "Corrgroups60LocalXAI": "CG60 ($d=60$)",
    "ForestFiresLocalXAI": "Forest ($d=13$)",
    "IndependentLinear60LocalXAI": "IL60 ($d=60$)",
    "NHANESILocalXAI": "NHANES ($d=79$)",
    "RealEstateLocalXAI": "Estate ($d=15$)",
    "wine_quality": "Wine ($d=11$)",
    "AdultCensusLocalXAI": "Adult ($d=14$)",
    "CaliforniaHousingLocalXAI": "Housing ($d=8$)",
    "BikeSharingLocalXAI": "Bike ($d=12$)",
    "ViT4by4Patches": "ViT16 ($d=16$)",
    "ViT3by3Patches": "ViT9 ($d=9$)",
    "ResNet18w14Superpixel": "ResNet18 ($d=14$)",
    "SentimentIMDBDistilBERT14": "DistilBERT ($d=14$)",
    "SOUM": "soum",
}

APPROXIMATOR_RENAMING = {
    "PermutationSampling": "Permutation Sampling",
    "KernelSHAP": "1-PolySHAP (KernelSHAP) ",
    "LeverageSHAP": "LeverageSHAP",
    "Proxy-LGBM": "Proxy (LGBM)",
    "RegressionMSR-LGBM": "RegressionMSR (LGBM)",
    "OddSHAP-Fourier-Random": "OddSHAP ($\eta=10$,random)",
    "OddSHAP-Fourier-ProxySPEX-NoCV": "OddSHAP ($\eta=10$)",
    "OddSHAP-Fourier-ProxySPEX-NoCV-5": "OddSHAP ($\eta=5$)",
    "OddSHAP-Fourier-ProxySPEX-NoCV-20": "OddSHAP ($\eta=20$)",
    "PolySHAP-2ADD": "2-PolySHAP",
    "PolySHAP-3ADD": "3-PolySHAP",
    "PolySHAP-4ADD": "4-PolySHAP",
    "FFD-RD": "FFD-RD",
    "FFD-RD-Corrected": "FFD-RD (corrected)",
    "FourierSHAP": "FourierSHAP",
}

TITLE_FONT_SIZE = 24


# Function to extract p and q
def parse_approximator(s):
    if pd.isna(s):
        return np.nan, np.nan
    match = re.search(r"PolySHAP-(\d+)ADD(?:-(\d+)%)*", s)
    if not match:
        return np.nan, np.nan
    if match:
        p = int(match.group(1))
        q = int(match.group(2)) if match.group(2) else 100
        return p, q
    return None, None


def max_value(row):
    n = int(row["n_players"])
    return 2**n

def compute_value(row):
    p = row["p"]
    q = row["q"]
    n = int(row["n_players"])

    # treat NaN (or any missing) as "not a PolySHAP" -> return 0
    if pd.isna(p) or pd.isna(q):
        return 0.0

    # now safe to cast p,q to int
    p = int(p)
    q = int(q)

    # guard if p > n: combinations for k>n are zero, so sum_{i=1}^{p-1} reduces to i=1..n
    if p > n:
        return float(sum(math.comb(n, i) for i in range(1, n + 1)))

    total = sum(math.comb(n, i) for i in range(1, p))
    total += math.comb(n, p) * (q / 100.0)
    return float(total)

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

if __name__ == "__main__":
    PLOT_BUDGET = True
    PLOT_RUNTIME = True

    INDEX = args.index
    ORDER = args.order
    # Load the results from the CSV file
    results_df = pd.read_csv(f"results_benchmark_{INDEX}_{ORDER}.csv")
    results_df = results_df.sort_values(by="n_players")

    GAME_IDS = results_df["game_id"].unique()
    GAME_TYPES = results_df["game_type"].unique()

    info = results_df[["game_id", "n_players"]].drop_duplicates()

    results_df[["p", "q"]] = results_df["approximator"].apply(
        lambda x: pd.Series(parse_approximator(x))
    )
    results_df["minimum_budget_to_plot"] = results_df.apply(compute_value, axis=1)
    results_df = results_df[
        results_df["used_budget"] >= results_df["minimum_budget_to_plot"]
    ]

    results_df["maximum_budget_to_plot"] = results_df.apply(max_value, axis=1)
    # results_df = results_df[
    #     results_df["used_budget"] < results_df["maximum_budget_to_plot"]
    #     ]

    # Plot approximation quality for standard
    if ORDER == 1:
        print(results_df["approximator"].unique())
        plot_df = results_df[
            (results_df["approximator"] == "MSR")
            | (results_df["approximator"] == "SVARM")
            | (results_df["approximator"] == "PermutationSampling")
            | (results_df["approximator"] == "Proxy-LGBM")
            | (results_df["approximator"] == "RegressionMSR-LGBM")
            | (results_df["approximator"] == "LeverageSHAP")
            | (results_df["approximator"] == "PolySHAP-3ADD")
            # | (results_df["approximator"] == "OddSHAP-Fourier-Random")
            | (results_df["approximator"] == "OddSHAP-Fourier-ProxySPEX-NoCV")
            # | (results_df["approximator"] == "OddSHAP-Fourier-ProxySPEX-NoCV-5")
            # | (results_df["approximator"] == "OddSHAP-Fourier-ProxySPEX-NoCV-20")
            # | (results_df["approximator"] == "All-Fourier-ProxySPEX-NoCV")
            | (results_df["approximator"] == "FFD-RD")
            | (results_df["approximator"] == "FFD-RD-Corrected")
            | (results_df["approximator"] == "FourierSHAP")
        ]

    # Create and save a legend for the plots
    fig, ax = plot_approximation_quality(
        data=plot_df,
        metric="MSE",
        log_scale_y=True,
        # log_scale_x=False,
        legend=True,
    )
    # Get handles and labels
    handles, labels = ax.get_legend_handles_labels()
    # Replace old labels with new ones
    labels = [APPROXIMATOR_RENAMING.get(l, l) for l in labels]
    # Build a name->index map so ordering is robust to data insertion order
    _lbl_idx = {lbl: i for i, lbl in enumerate(labels)}
    # Desired display order (after renaming):
    # Row1: header, MSR, SVARM, Permutation Sampling, LeverageSHAP, 3-PolySHAP
    # Row2: RegressionMSR(LGBM), Proxy(LGBM), FourierSHAP, FFD-RD, FFD-RD(corrected), OddSHAP
    _desired_ncol = [
        "$\\bf{Method}$",
        "MSR", "SVARM", "Permutation Sampling", "LeverageSHAP", "3-PolySHAP",
        "RegressionMSR (LGBM)", "Proxy (LGBM)", "FFD-RD", "FFD-RD (corrected)",
        "FourierSHAP", "OddSHAP ($\\eta=10$)",
    ]
    _desired_1col = [n for n in _desired_ncol if n != "$\\bf{Method}$"]
    order_ncol = [_lbl_idx[n] for n in _desired_ncol if n in _lbl_idx]
    order_1col  = [_lbl_idx[n] for n in _desired_1col  if n in _lbl_idx]

    # Clear the entire figure (removes all axes/artists)
    legend_fig = plt.figure(figsize=(6, 0.5))
    # Create a new axes that will contain only the legend
    legend_ax = legend_fig.add_subplot(111)
    legend_ax.axis("off")
    legend = legend_ax.legend([handles[i] for i in order_ncol],
               [labels[i] for i in order_ncol], ncol=6, frameon=False, loc="center")
    # ensure layout on canvas, get renderer and legend bbox in inches
    legend_fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = legend.get_window_extent(renderer=renderer)
    bbox_inches = bbox.transformed(legend_fig.dpi_scale_trans.inverted())

    legend_fig.savefig("plots/legend.pdf", bbox_inches=bbox_inches,pad_inches=0, transparent=True)
    # fig_legend.show()

    # Clear the entire figure (removes all axes/artists)
    legend_fig = plt.figure(figsize=(6, 0.5))
    # Create a new axes that will contain only the legend
    legend_ax = legend_fig.add_subplot(111)
    legend_ax.axis("off")
    legend = legend_ax.legend([handles[i] for i in order_1col],
               [labels[i] for i in order_1col], ncol=1, frameon=False, loc="center")
    #ax.legend(handles, labels)
    # ensure layout on canvas, get renderer and legend bbox in inches
    legend_fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = legend.get_window_extent(renderer=renderer)
    bbox_inches = bbox.transformed(legend_fig.dpi_scale_trans.inverted())

    legend_fig.savefig("plots/legend_1col.pdf", bbox_inches=bbox_inches,pad_inches=0, transparent=True)
    # fig_legend.show()

    # Create dataframe with paired and non-paired sampling
    plot_df_standard_and_paired = plot_df.copy()

    # Filter for paired sampling
    plot_df = plot_df[plot_df["id_config_approximator"].astype(int) == 37] # filter for paired sampling
    print(plot_df["approximator"].unique())
    print(plot_df["id_config_approximator"].unique())
    print(plot_df["game_id"].unique())

    # FFD methods have a fixed but instance-varying budget.
    # Collapse all instances to a single x-point at the mean budget so the
    # aggregation step groups them together and produces one (median, IQR) point.
    _ffd_names = {"FFD-RD", "FFD-RD-Corrected"}
    plot_df = plot_df.copy()
    # Collapse per game_id so different games keep their own mean budget
    game_ids = plot_df["game_id"].unique()
    for _ffd in _ffd_names:
        for gid in game_ids:
            _mask = (plot_df["approximator"] == _ffd) & (plot_df["game_id"] == gid)
            if _mask.any():
                _mean_budget = int(round(plot_df.loc[_mask, "used_budget"].mean()))
                plot_df.loc[_mask, "used_budget"] = _mean_budget
                plot_df.loc[_mask, "budget"] = _mean_budget


    if PLOT_BUDGET:
        # APPROXIMATION QUALITY PLOTS
        for game_type in GAME_TYPES:
            plot_df_game_type = plot_df[plot_df["game_type"] == game_type]
            for game_id in GAME_IDS:
                plot_df_game_id = plot_df_game_type[plot_df_game_type["game_id"] == game_id]
                if len(plot_df_game_id) > 0:
                    # print("Plotting", game_type, game_id)
                    metric = "MSE"
                    dataset = plot_df_game_id["game"].unique()[0]
                    fig, ax = plot_approximation_quality(
                        data=plot_df_game_id,
                        metric=metric,
                        log_scale_y=True,
                        legend=False,
                    )
                    ax.set_title(DATA_NAMES[dataset], fontsize=TITLE_FONT_SIZE)

                    # Add explicit IQR error bars for FFD single-point methods
                    for _ffd in ("FFD-RD", "FFD-RD-Corrected"):
                        _fdf = plot_df_game_id[plot_df_game_id["approximator"] == _ffd]
                        if len(_fdf) == 0:
                            continue
                        _x = _fdf["used_budget"].iloc[0]
                        _y_med = _fdf[metric].median()
                        _y_lo = max(_fdf[metric].quantile(0.25), 1e-9)
                        _y_hi = _fdf[metric].quantile(0.75)
                        ax.errorbar(
                            _x, _y_med,
                            yerr=[[_y_med - _y_lo], [_y_hi - _y_med]],
                            color=STYLE_DICT[_ffd]["color"],
                            fmt="none", capsize=5, capthick=2,
                            elinewidth=2, zorder=5,
                        )

                    fig.tight_layout()
                    fig.savefig(
                        f"plots/{game_type}/{game_id}_{metric}_{INDEX}_{ORDER}_standard.pdf"
                        #f"plots/{game_type}/{game_id}_{metric}_{INDEX}_{ORDER}_standard.png"
                    )
                    plt.close(fig)

    if PLOT_RUNTIME:
        # RUNTIME PLOTS

        EVALUATION_CONSTANTS = [0.001,0.01,0.1,1]

        plot_df["runtime_without_evaluations"] = plot_df["total_runtime"] - plot_df["evaluations"]
        plot_df["runtimeraw"] = plot_df["total_runtime"]

        for game_type in GAME_TYPES:
            plot_df_game_type = plot_df[plot_df["game_type"] == game_type]
            for game_id in GAME_IDS:
                plot_df_game_id = plot_df_game_type[plot_df_game_type["game_id"] == game_id]
                if len(plot_df_game_id) > 0:
                    for evaluation_cost in EVALUATION_CONSTANTS:
                        plot_cost = plot_df_game_id.copy()
                        # aggregate runtime over iterations
                        plot_cost["evaluation_time"] = plot_cost["used_budget"] * evaluation_cost
                        plot_cost["runtime"] = plot_cost["runtime_without_evaluations"] + plot_cost["evaluation_time"]
                        # compute group medians (one row per group)
                        keys = ["game_type", "game_id", "approximator", "used_budget"]
                        runtime_medians = (
                            plot_cost
                            .groupby(keys, as_index=False)["runtime"]
                            .median()
                            .rename(columns={"runtime": "runtime_median"})
                        )
                        # merge medians back into the original DataFrame (preserves original rows/order)
                        plot_cost = plot_cost.merge(runtime_medians, on=keys, how="left", sort=False)

                        metric = "MSE"
                        # runtime on x-axis, MSE on y-axis
                        fig, ax = plot_approximation_quality(
                            data=plot_cost,
                            metric=metric,
                            log_scale_y=True,
                            legend=False,
                            x_column="runtime_median"
                        )
                        dataset = plot_cost["game"].unique()[0]
                        ax.set_title(DATA_NAMES[dataset], fontsize=TITLE_FONT_SIZE)
                        fig.tight_layout()
                        # Add explicit IQR error bars for FFD single-point methods
                        for _ffd in ("FFD-RD", "FFD-RD-Corrected"):
                            _fdf = plot_cost[plot_cost["approximator"] == _ffd]
                            if len(_fdf) == 0:
                                continue
                            _x = _fdf["runtime_median"].iloc[0]
                            _y_med = _fdf[metric].median()
                            _y_lo = max(_fdf[metric].quantile(0.25), 1e-9)
                            _y_hi = _fdf[metric].quantile(0.75)
                            ax.errorbar(
                                _x,
                                _y_med,
                                yerr=[[_y_med - _y_lo], [_y_hi - _y_med]],
                                color=STYLE_DICT[_ffd]["color"] if _ffd in STYLE_DICT else "black",
                                fmt="none",
                                capsize=5,
                                capthick=2,
                                elinewidth=2,
                                zorder=5,
                            )
                        # print("Saving figure...", f"plots/{game_type}/{game_id}_{metric}_{INDEX}_{ORDER}_standard.png")
                        fig.savefig(
                            f"plots/{game_type}/{game_id}_{metric}_{INDEX}_{ORDER}_runtime_{evaluation_cost}.pdf"
                        )
                        plt.close(fig)

                        if "Crime" in game_id and evaluation_cost == 0.001:
                            # Add crime plot with legend for main paper
                            # runtime on x-axis, MSE on y-axis
                            fig, ax = plot_approximation_quality(
                                data=plot_cost,
                                metric=metric,
                                log_scale_y=True,
                                legend=True,
                                x_column="runtime_median"
                            )
                            # Get handles and labels
                            handles, labels = ax.get_legend_handles_labels()
                            # Replace old labels with new ones
                            labels = [APPROXIMATOR_RENAMING.get(l, l) for l in labels]
                            _lbl_idx = {lbl: i for i, lbl in enumerate(labels)}
                            _desired_ncol = [
                                "$\\bf{Method}$",
                                "MSR", "SVARM", "Permutation Sampling", "LeverageSHAP",
                                "RegressionMSR (LGBM)", "Proxy (LGBM)", "FFD-RD", "FFD-RD (corrected)",
                                "FourierSHAP", "OddSHAP ($\\eta=10$)",
                            ]
                            _desired_1col = [n for n in _desired_ncol if n != "$\\bf{Method}$"]
                            order_1col = [_lbl_idx[n] for n in _desired_1col if n in _lbl_idx]

                            dataset = plot_cost["game"].unique()[0]
                            ax.set_title(DATA_NAMES[dataset], fontsize=TITLE_FONT_SIZE)
                            ax.legend([handles[i] for i in order_1col], [labels[i] for i in order_1col], fontsize=10)
                            fig.tight_layout()
                            # Add explicit IQR error bars for FFD single-point methods
                            for _ffd in ("FFD-RD", "FFD-RD-Corrected"):
                                _fdf = plot_cost[plot_cost["approximator"] == _ffd]
                                if len(_fdf) == 0:
                                    continue
                                _x = _fdf["runtime_median"].iloc[0]
                                _y_med = _fdf[metric].median()
                                _y_lo = max(_fdf[metric].quantile(0.25), 1e-9)
                                _y_hi = _fdf[metric].quantile(0.75)
                                ax.errorbar(
                                    _x,
                                    _y_med,
                                    yerr=[[_y_med - _y_lo], [_y_hi - _y_med]],
                                    color=STYLE_DICT[_ffd]["color"] if _ffd in STYLE_DICT else "black",
                                    fmt="none",
                                    capsize=5,
                                    capthick=2,
                                    elinewidth=2,
                                    zorder=5,
                                )
                            fig.savefig(
                                f"plots/custom/{game_id}_{metric}_{INDEX}_{ORDER}_runtime_{evaluation_cost}_main_paper_legend.pdf"
                            )
                            plt.close(fig)

                    # PLOT RUNTIME y-AXIS, BUDGET x-AXIS
                    metric = "Runtime"
                    plot_df_game_id[metric] = plot_df_game_id["runtime_without_evaluations"]
                    # budget on x-axis, runtime on y-axis
                    fig, ax = plot_approximation_quality(
                        data=plot_df_game_id,
                        metric=metric,
                        log_scale_y=True,
                        legend=False,
                        x_column="used_budget"
                    )

                    dataset = plot_df_game_id["game"].unique()[0]
                    ax.set_title(DATA_NAMES[dataset], fontsize=TITLE_FONT_SIZE)
                    fig.tight_layout()
                    # Add explicit IQR error bars for FFD single-point methods on runtime plots
                    for _ffd in ("FFD-RD", "FFD-RD-Corrected"):
                        _fdf = plot_df_game_id[plot_df_game_id["approximator"] == _ffd]
                        if len(_fdf) == 0:
                            continue
                        # use the collapsed/mean budget as x-position (same approach as above)
                        _x = _fdf["used_budget"].iloc[0]
                        _y_med = _fdf[metric].median()
                        _y_lo = max(_fdf[metric].quantile(0.25), 1e-9)
                        _y_hi = _fdf[metric].quantile(0.75)
                        ax.errorbar(
                            _x,
                            _y_med,
                            yerr=[[_y_med - _y_lo], [_y_hi - _y_med]],
                            color=STYLE_DICT[_ffd]["color"] if _ffd in STYLE_DICT else "black",
                            fmt="none",
                            capsize=5,
                            capthick=2,
                            elinewidth=2,
                            zorder=5,
                        )

                    # print("Saving figure...", f"plots/{game_type}/{game_id}_{metric}_{INDEX}_{ORDER}_standard.png")
                    fig.savefig(
                    #    f"plots/{game_type}/{game_id}_{metric}_{INDEX}_{ORDER}_runtime.png"
                        f"plots/{game_type}/{game_id}_{metric}_{INDEX}_{ORDER}_runtime.pdf"
                    )
                    plt.close(fig)


                    # PLOT RUNTIME y-AXIS, X-AXIS for OddSHAP and the different runtimes
                    metric = "Runtime"
                    plot_oddshap = plot_df_game_id[
                        (plot_df_game_id["approximator"] == "OddSHAP-Fourier-ProxySPEX-NoCV")
                    ].copy()
                    plot_oddshap["Proxy"] = plot_oddshap["proxy_fit"]
                    plot_oddshap["Regression"] = plot_oddshap["regression"]
                    plot_oddshap["Sampling"] = plot_oddshap["sampling"]
                    plot_oddshap["Extraction"] = plot_oddshap["extraction"]
                    plot_oddshap["Total"] = plot_oddshap["runtime_without_evaluations"]


                    # keys already defined in the surrounding code
                    columns_to_aggregate = ["Sampling","Proxy","Extraction","Regression","Total"]

                    # compute grouped median and 25/75 percentiles
                    aggregated = (
                        plot_oddshap
                        .groupby(keys)
                        .agg({col: ["median", agg_percentile(25), agg_percentile(75)] for col in columns_to_aggregate})
                        .reset_index()
                    )

                    # flatten multiindex columns -> e.g. 'Proxy_median', 'Proxy_quantile_25', 'Proxy_quantile_75'
                    aggregated.columns = [
                        "_".join(filter(None, map(str, c))).strip() if isinstance(c, tuple) else c
                        for c in aggregated.columns
                    ]


                    marker = "o"
                    # prepare plotting
                    fig, ax = plt.subplots(figsize=(8, 5))
                    df_plot = aggregated.sort_values(by="used_budget")

                    # determine threshold: skip regression for budgets < `n_players * 10`
                    n_players = int(plot_df_game_id["n_players"].unique()[0])
                    threshold = n_players * 10

                    #colors = ["C0", "C1", "C2", "C3"]
                    colors = ['#4477AA', '#66CCEE', '#228833',
                        '#CCBB44', 'black', '#AA3377', '#BBBBBB']

                    LIGHT_GRAY = "#d3d3d3"
                    # add grid
                    ax.grid(True, color=LIGHT_GRAY, linestyle="-")

                    # PLOT LEGEND FIRST
                    linewidth = 2
                    markersize = 7
                    # create compact legend-only figure
                    legend_fig = plt.figure(figsize=(6, 0.45))
                    legend_ax = legend_fig.add_subplot(111)
                    legend_ax.axis("off")

                    from matplotlib.lines import Line2D

                    # prepare handles and labels without plotting on the main axes
                    handles = [
                        Line2D([0], [0], color=c, marker=marker, linewidth=linewidth, markersize=markersize,mec="white")
                        for c in colors
                    ]
                    labels = columns_to_aggregate

                    # draw centered legend, no frame, 4 columns
                    legend = legend_fig.legend(handles, labels, loc="center", ncol=5, frameon=False, fontsize=12)

                    # save tightly (removes excess white space)
                    legend_fig.savefig("plots/legend_runtime_oddshap_components.pdf", bbox_inches="tight", pad_inches=0)
                    plt.close(legend_fig)

                    LINE_THICKNESS = 2
                    MARKER_SIZE = 7
                    # Plot runtime components
                    for i, col in enumerate(columns_to_aggregate):
                        # for Regression, only keep points with used_budget >= threshold
                        if col == "Regression":
                            df_col = df_plot[df_plot["used_budget"] >= threshold]
                            if df_col.empty:
                                continue  # nothing to plot for Regression
                        else:
                            df_col = df_plot

                        med = df_col[f"{col}_median"]
                        low = df_col[f"{col}_quantile_25"]
                        high = df_col[f"{col}_quantile_75"]
                        x = df_col["used_budget"]

                        ax.plot(x, med, label=col, color=colors[i], linewidth=LINE_THICKNESS,markersize=MARKER_SIZE, marker="o", mec="white")
                        ax.fill_between(x, low, high, color=colors[i], alpha=0.2)

                    ax.set_xscale("log")
                    ax.set_yscale("log")
                    ax.set_xlabel("Budget ($m$)", fontsize=20)
                    ax.set_ylabel("Runtime (seconds)", fontsize=20)
                    #ax.legend() # hide or show legend
                    ax.grid(True,linestyle="-", color=LIGHT_GRAY, alpha=0.8)

                    dataset = plot_df_game_id["game"].unique()[0]
                    ax.set_title(DATA_NAMES[dataset], fontsize=TITLE_FONT_SIZE)
                    plt.xticks(fontsize=18)
                    plt.yticks(fontsize=18)
                    fig.tight_layout()
                    # print("Saving figure...", f"plots/{game_type}/{game_id}_{metric}_{INDEX}_{ORDER}_standard.png")
                    fig.savefig(
                    #    f"plots/{game_type}/{game_id}_{metric}_{INDEX}_{ORDER}_runtime_oddshap.png"
                        f"plots/{game_type}/{game_id}_{metric}_{INDEX}_{ORDER}_runtime_oddshap.pdf"
                    )
                    plt.close(fig)



