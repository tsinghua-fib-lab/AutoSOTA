import matplotlib
import matplotlib.pyplot as plt

from utils.plotting import get_all_legend_handles_labels
from utils.plotting import dict_translate, color_dict


import numpy as np
import json

plt.rc("font", **{"family": "serif", "serif": ["times"]})
plt.rc("text", usetex=True)

plt.rc("axes", titlesize=25)
plt.rc("axes", labelsize=22)

plt.rcParams["axes.grid"] = True
plt.rcParams["axes.axisbelow"] = True

matplotlib.rc("xtick", labelsize=22)
matplotlib.rc("ytick", labelsize=22)

f = open(
    "./results/" + "Benchmark_Nystroem_Crime" + ".json"
)

tmp_data = json.load(f)

f.close()


fig = plt.figure(figsize=(10, 7.5), constrained_layout=True)
gs = fig.add_gridspec(2, 3, height_ratios=[0.08, 1])


f_ax1 = fig.add_subplot(gs[1, :])


l_axs = fig.add_subplot(gs[0, :])

axs = [f_ax1]


alpha_dict = {
    "SVR-FKD(25.0%)": 0.25,
    "SVR-FKD(50.0%)": 0.5,
    "SVR-FKD(75.0%)": 0.75,
    "SVR-FKD(100%)": 1,
    "DUMMY": 1,
}

mod_dict = {
    "SVR-FKD(25.0%)": "SVR-FKD",
    "SVR-FKD(50.0%)": "SVR-FKD",
    "SVR-FKD(75.0%)": "SVR-FKD",
    "SVR-FKD(100%)": "SVR-FKD",
    "DUMMY": "DUMMY",
}

marker_dict = {
    "SVR-FKD(25.0%)": "P",
    "SVR-FKD(50.0%)": "^",
    "SVR-FKD(75.0%)": "o",
    "SVR-FKD(100%)": "s",
    "DUMMY": "D",
}

label_dict = {
    "SVR-FKD(25.0%)": "25\%",
    "SVR-FKD(50.0%)": "50\%",
    "SVR-FKD(75.0%)": "75\%",
    "SVR-FKD(100%)": "100\%",
    "DUMMY": "DUMMY",
}



for mod in tmp_data["MODELS"]:
    res = np.array(tmp_data[mod]["result"])

    result = res[:, :, (0, 2)]  # Pick MAE and GDP only

    means = np.mean(np.array(result), axis=0)
    stds = np.std(np.array(result), axis=0)

    str_dataset_short = dict_translate[tmp_data["DATASET"]]

    for i in range(len(means[0, :]) - 1):
        ebar = axs[i].errorbar(
            means[:, i + 1],
            means[:, 0],
            xerr=stds[:, i + 1],
            yerr=stds[:, 0],
            capsize=6.75,
            ms=12.5,
            dash_joinstyle="round",
            linestyle="dotted",
            marker=marker_dict[mod],
            linewidth=3.5,
            capthick=3.5,
            label=label_dict[mod],
            alpha=alpha_dict[mod],
            markeredgecolor="black",
            c=color_dict[mod_dict[mod]],
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


handles, labels = get_all_legend_handles_labels([f_ax1])

f_ax1.set_xlabel("GDP [DP]")
f_ax1.set_ylabel("MAE")

l_axs.legend(
    handles,
    labels,
    fontsize=20,
    loc="center",
    ncol=int(len(labels)),
    frameon=False,
    bbox_to_anchor=(0.5, -0.1),
)
l_axs.axis("off")

plt.savefig("./imgs/" + "Nystroem_Demonstration" + ".pdf", bbox_inches="tight")
plt.show()
