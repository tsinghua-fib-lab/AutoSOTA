import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib

import json

from utils.plotting import plot_statistics, get_all_legend_handles_labels

# Plotting Settings
plt.rc("font", **{"family": "serif", "serif": ["times"]})
plt.rc("text", usetex=True)

plt.rc("axes", titlesize=25)
plt.rc("axes", labelsize=22)

plt.rcParams["axes.grid"] = True
plt.rcParams["axes.axisbelow"] = True

matplotlib.rc("xtick", labelsize=22)
matplotlib.rc("ytick", labelsize=22)

# Create figure and grid
fig = plt.figure(figsize=(17, 8.5))
gs = gridspec.GridSpec(
    4,
    3,
    figure=fig,
    width_ratios=[0.5, 0.0125, 0.4],
    height_ratios=[0.1, 0.8, 0.8, 0.25],
)


ax_right = fig.add_subplot(gs[1:3, 2])

ax_middle = fig.add_subplot(gs[:, 1])

ax_left_bottom = fig.add_subplot(gs[2, 0])
ax_left_top = fig.add_subplot(gs[1, 0], sharex=ax_left_bottom)

ax_letter_left = fig.add_subplot(gs[3, 0])
ax_letter_right = fig.add_subplot(gs[3, 2])


fname_multi = "Benchmark_multi_protected"

f = open("./results/" + fname_multi + ".json")
tmp_data = json.load(f)
f.close()


l_ax_multi = fig.add_subplot(gs[0, 0])  # Legend axis

axs_multi = [ax_left_top, ax_left_bottom]

translate_dict = {"CrimeSingle": "Single", "CrimeMulti": "Multi", "DUMMY": "Dummy"}

for name in ["CrimeSingle", "CrimeMulti", "DUMMY"]:
    # FOLDS x PARAMS x MEASURES
    res = np.array(tmp_data[name]["result"])

    if name == "DUMMY":
        s_temp = "DUMMY"
    else:
        s_temp = "SVR-FKD" + "-" + name

    plot_statistics(
        axs_multi, res, str_dataset="", label=translate_dict[name], str_model=s_temp
    )


handles, labels = get_all_legend_handles_labels(axs_multi)

l_ax_multi.legend(
    handles,
    labels,
    fontsize=22,
    loc="center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, -0.1),
)
l_ax_multi.axis("off")

ax_left_bottom.set_xlabel("GDP [DP]")
ax_left_bottom.set_ylabel("MAE")

ax_left_top.set_ylabel("MAE")
ax_left_top.tick_params(labelbottom=False)

ax_left_top.text(
    0.7,
    0.85,
    r"Black pop. (\%)",
    fontsize=20,
    alpha=0.85,
    transform=ax_left_top.transAxes,
    bbox=dict(edgecolor="black", facecolor="white", alpha=0.85),
)

ax_left_bottom.text(
    0.7,
    0.85,
    r"White pop. (\%)",
    fontsize=20,
    alpha=0.85,
    transform=ax_left_bottom.transAxes,
    bbox=dict(edgecolor="black", facecolor="white", alpha=0.85),
)


############################################################### END PLOT (a)

# Plotting Settings
plt.rc("font", **{"family": "serif", "serif": ["times"]})
plt.rc("text", usetex=True)

plt.rc("axes", titlesize=25)
plt.rc("axes", labelsize=22)

plt.rcParams["axes.grid"] = True
plt.rcParams["axes.axisbelow"] = True

matplotlib.rc("xtick", labelsize=22)
matplotlib.rc("ytick", labelsize=22)

fname_alpha = "Benchmark_alpha_prime_influence"

f = open("./results/" + fname_alpha + ".json")
data = json.load(f)
f.close()


l_ax_alpha = fig.add_subplot(gs[0, 2])

res = np.array(data["KRR"])
alphas = np.array(data["alpha_primes"])

for IT in range(len(data["ITER"])):
    res_per_iter = res[:, :, IT, :]

    t = 0

    avg_over_folds = np.mean(res_per_iter, axis=1)
    std_over_folds = np.std(res_per_iter, axis=1)

    if IT == 0:
        mk = "o"
        c = "#8cbb26"
    else:
        mk = "s"
        c = "#d46a6a"

    ebar = ax_right.errorbar(
        avg_over_folds[:, 1],
        avg_over_folds[:, 0],
        xerr=std_over_folds[:, 1],
        yerr=std_over_folds[:, 0],
        linewidth=3,
        capsize=7.75,
        dash_joinstyle="round",
        linestyle="dotted",
        marker=mk,
        ms=10,
        color=c,
        capthick=2.5,
        alpha=0.85,
        markeredgecolor="black",
        label="$m=" + str(data["ITER"][IT]) + "$",
    )
    for bar in ebar[2]:
        bar.set_linestyle("dashed")

    if IT == 1:
        ax_right.annotate(
            r"$\tilde{\alpha} =" + str(alphas[0]) + "$ ",
            xy=(avg_over_folds[0, 1], avg_over_folds[0, 0]),  # point to annotate
            xytext=(
                avg_over_folds[0, 1] - 0.022,
                avg_over_folds[0, 0] - 0.005,
            ),  # position of text
            arrowprops=dict(arrowstyle="-", linewidth=2, linestyle="--"),
            fontsize=25,
        )

        ax_right.annotate(
            r"$\tilde{\alpha} =" + str(alphas[-1]) + "$ ",
            xy=(avg_over_folds[-1, 1], avg_over_folds[-1, 0]),  # point to annotate
            xytext=(
                avg_over_folds[-1, 1] + 0.004,
                avg_over_folds[-1, 0] + 0.003,
            ),  # position of text
            arrowprops=dict(arrowstyle="-", linewidth=2, linestyle="--"),
            fontsize=25,
        )


ax_right.set_xlabel("GDP [DP]")
ax_right.set_ylabel("MAE")


handles, labels = get_all_legend_handles_labels([ax_right])

l_ax_alpha.legend(
    handles,
    labels,
    fontsize=22,
    loc="center",
    ncol=len(data["ITER"]),
    frameon=False,
    bbox_to_anchor=(0.5, -0.1),
)
l_ax_alpha.axis("off")

################################################################################ END plot (b)


# Use the middle axis for a vertical dotted line
# ax_middle.axvline(x=0.5, linestyle=':', color='gray', linewidth=2)

# Remove ticks and frame from the middle axis
ax_middle.set_xticks([])
ax_middle.set_yticks([])
ax_middle.set_frame_on(False)

# Optional: adjust limits so the line is centered vertically
ax_middle.set_xlim(0, 1)
ax_middle.set_ylim(0, 1)


ax_letter_left.text(
    0.5,
    0,
    "(a)",
    transform=ax_letter_left.transAxes,
    fontsize=30,
    fontweight="bold",
    ha="center",
    va="top",
)

ax_letter_right.text(
    0.5,
    0,
    "(b)",
    transform=ax_letter_right.transAxes,
    fontsize=30,
    fontweight="bold",
    ha="center",
    va="top",
)

ax_letter_left.set_frame_on(False)
ax_letter_right.set_frame_on(False)

ax_letter_left.set_xticks([])
ax_letter_left.set_yticks([])

ax_letter_right.set_xticks([])
ax_letter_right.set_yticks([])


plt.savefig("./imgs/" + "OtherExperiments.pdf", bbox_inches="tight")
plt.show()
