import matplotlib
import matplotlib.pyplot as plt

from utils.plotting import plot_statistics

import numpy as np
import json

from utils.plotting import get_all_legend_handles_labels

# Plotting Settings
plt.rc("font", **{"family": "serif", "serif": ["times"]})
plt.rc("text", usetex=True)

plt.rc("axes", titlesize=25)
plt.rc("axes", labelsize=22)

plt.rcParams["axes.grid"] = True
plt.rcParams["axes.axisbelow"] = True

matplotlib.rc("xtick", labelsize=22)
matplotlib.rc("ytick", labelsize=22)

data_dicts = []

# 5 Fold results

to_bench = "ComparisonLinear"
one_row = False

if to_bench == "Appendix":
    fnames = ["Benchmark_Crime_FullAppendix", "Benchmark_ACSIncome_FullAppendix", "Benchmark_ACSTravelTime_FullAppendix"] 
elif to_bench == "ComparisonLinear":
    fnames = ["Benchmark_ACSIncome_Linear_Comparison"]
    one_row = True
elif to_bench == "MainResult":
    fnames = ["Benchmark_Crime", "Benchmark_ACSIncome", "Benchmark_ACSTravelTime"] 
else:
    fnames = [] # Errors

fnames_total = [x + ".json" for x in fnames]

# Load results from every json file
for fname_total in fnames_total:

    f = open("./results/" + fname_total)
    tmp_data = json.load(f)
    f.close()

    data_dicts.append(tmp_data)


# Construct plot format
if one_row:
    fig = plt.figure(figsize=(15, 4.7), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[0.08, 1])

    f_ax1 = fig.add_subplot(gs[1, 0])
    f_ax2 = fig.add_subplot(gs[1, 1])
    f_ax3 = fig.add_subplot(gs[1, 2])

else:
    fig = plt.figure(figsize=(15, 12), constrained_layout=True)
    gs = fig.add_gridspec(4, 3, height_ratios=[0.08, 1, 1, 1])

    f_ax1 = fig.add_subplot(gs[1, 0])
    f_ax2 = fig.add_subplot(gs[1, 1])
    f_ax3 = fig.add_subplot(gs[1, 2])

    f_ax4 = fig.add_subplot(gs[2, 0])
    f_ax5 = fig.add_subplot(gs[2, 1])
    f_ax6 = fig.add_subplot(gs[2, 2])

    f_ax7 = fig.add_subplot(gs[3, 0])
    f_ax8 = fig.add_subplot(gs[3, 1])
    f_ax9 = fig.add_subplot(gs[3, 2])

l_axs = fig.add_subplot(gs[0, :])

if one_row:
    axs = np.array([[f_ax1, f_ax2, f_ax3]])
else:   
    axs = np.array([[f_ax1, f_ax2, f_ax3], [f_ax4, f_ax5, f_ax6], [f_ax7, f_ax8, f_ax9]])




for k, data_dict in enumerate(data_dicts):

    dataset = data_dict["DATASET"]
    models = data_dict["MODELS"]
    measures = data_dict["MEASURES"]

    for i, model in enumerate(models):
        model_dict = data_dict[model]

        result = np.array(model_dict["result"])

        if model == "SVR-FKD" or model == "KRR-FKD":
            s_model = model + " (ours)"
        else:
            s_model = model

        plot_statistics(
            axs[k, :], result, label=s_model, str_dataset=dataset, str_model=model
        )


        if one_row:
            for j, ax in enumerate([f_ax1, f_ax2, f_ax3]):
                ax.set_xlabel(measures[j])  # + " ($\downarrow$)"
        else:
            for j, ax in enumerate([f_ax7, f_ax8, f_ax9]):
                ax.set_xlabel(measures[j])  # + " ($\downarrow$)"

    if one_row:
        f_ax1.set_ylabel("MAE")
    else:
        for ax in [f_ax1, f_ax4, f_ax7]:
            ax.set_ylabel("MAE")

    if one_row:
         for ax in [f_ax2, f_ax3]:
            ax.set_yticklabels([])
    else:
        for ax in [f_ax2, f_ax3, f_ax5, f_ax6, f_ax8, f_ax9]:
            ax.set_yticklabels([])

# Get all handles and labels for one global legend

if one_row:
    handles, labels = get_all_legend_handles_labels(
        [f_ax1, f_ax2, f_ax3]
    )
else:
    handles, labels = get_all_legend_handles_labels(
        [f_ax1, f_ax2, f_ax3, f_ax4, f_ax5, f_ax6, f_ax7, f_ax8, f_ax9]
    )


if l_axs != None:
    l_axs.legend(
        handles,
        labels,
        fontsize=19,
        loc="center",
        ncol=int(len(labels)),
        frameon=False,
        bbox_to_anchor=(0.5, -0.1),
    )
    l_axs.axis("off")


plt.savefig(
    "./imgs/" + to_bench + "_Experimental_Evaluation_" + str(fnames[:]) + ".pdf",
    bbox_inches="tight",
)
plt.show()
