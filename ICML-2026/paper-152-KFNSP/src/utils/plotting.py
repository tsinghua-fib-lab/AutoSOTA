import numpy as np
from matplotlib import rc
import matplotlib as mpl


def get_all_legend_handles_labels(axes):
    """
    Collects all unique legend handles and labels from a list of axes.
    """
    all_handles = []
    all_labels = []

    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for h, lab in zip(handles, labels):
            if lab not in all_labels:  # Avoid duplicates
                all_handles.append(h)
                all_labels.append(lab)

    return all_handles, all_labels


## Disctionaries used to coordinate colors, markers, linestyles etc.
dict_translate = {
    "ACSIncome": "ACS Inc.",
    "ACSTravelTime": "ACS Tra.",
    "Crime": "Crimes",
    "": None,
}

color_dict = {
    "KRR-FKD": "#17becf",
    "SVR-FKD": "#ff7f0e",
    "SVR-INPL": "#e29552",
    "KRR-FKL": "#8cbb26",
    "NN-HGR": "#d46a6a",
    "NN-HGR-L": "#4F4195",
    "SVR-FKD-CrimeSingle": "#ff7f0e",
    "SVR-FKD-CrimeMulti": "#17becf",
    "DUMMY": "#e377c2",
     "NN-FREM" : "tab:red",
}

marker_dict = {
    "KRR-FKD": "s",
    "SVR-FKD-CrimeSingle": "o",
    "SVR-FKD-CrimeMulti": "s",
    "SVR-FKD": "o",
    "SVR-INPL": "X",
    "NN-HGR": "^",
    "NN-HGR-L": "^",
    "NN-FREM" : "x",
    "KRR-FKL": "P",
    "DUMMY": "D",
}

linestyle_dict = {
    "KRR-FKD": "dashed",
    "SVR-FKD": "solid",
    "SVR-INPL": "solid",
    "NN-HGR": "dotted",
    "NN-HGR-L": "dotted",
    "KRR-FKL": "dashdot",
    "DUMMY": "solid",
    "NN-FREM" : "solid",
}
##


def plot_statistics(axs, result, label, str_dataset=None, str_model=None, alpha=0.85):
    """
    Plotting helper that takes a number of axes and plots the results in ''res'' as errorbars
    for each different measure into a different axis in ''axs''.
    """

    rc("font", **{"family": "serif", "serif": ["times"]})
    rc("text", usetex=True)
    mpl.rcParams["xtick.labelsize"] = 30
    mpl.rcParams["ytick.labelsize"] = 30

    means = np.mean(np.array(result), axis=0)
    stds = np.std(np.array(result), axis=0)

    str_dataset_short = dict_translate[str_dataset]

    for i in range(len(means[0, :]) - 1):
        ebar = axs[i].errorbar(
            means[:, i + 1],
            means[:, 0],
            xerr=stds[:, i + 1],
            yerr=stds[:, 0],
            capsize=4.75,
            ms=10,
            dash_joinstyle="round",
            linestyle="dotted",  # linestyle_dict[str_model],
            marker=marker_dict[str_model],
            linewidth=2,
            capthick=2.25,
            label=label,
            alpha=alpha,
            markeredgecolor="black",
            c=color_dict[str_model],
        )
        for bar in ebar[2]:
            bar.set_linestyle("dashed")

        if str_dataset_short is not None:
            axs[i].text(
                0.94,
                0.95,
                str_dataset_short,
                fontsize=20,
                alpha=0.55,
                transform=axs[i].transAxes,
                bbox=dict(edgecolor="black", facecolor="white", alpha=0.25),
                ha="right",
                va="top",
            )
