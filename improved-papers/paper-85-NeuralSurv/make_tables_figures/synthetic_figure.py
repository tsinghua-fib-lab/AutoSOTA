import pandas as pd
import os
import dill
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from data.data_loading import load_synthetic_data
from scipy.stats import lognorm
from sksurv.nonparametric import kaplan_meier_estimator

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "Times New Roman",
        "font.size": 16,  # <-- Increase this value to scale all fonts
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    }
)
output_dir = "/Users/melodiemonod/projects/2025/neuralsurv"

configs = [
    {
        "subsample_n": 25,
        "date": "2025-05-12",
        "jobid": "sub_25_layers_2_hidden_16",
        "jobid_neuralsurv": "synthetic_25",
    },
    {
        "subsample_n": 50,
        "date": "2025-05-12",
        "jobid": "sub_50_layers_2_hidden_16",
        "jobid_neuralsurv": "synthetic_50",
    },
    {
        "subsample_n": 100,
        "date": "2025-05-13",
        "jobid": "sub_100_layers_2_hidden_16",
        "jobid_neuralsurv": "synthetic_100",
    },
    {
        "subsample_n": 150,
        "date": "2025-05-10",
        "jobid": "sub_150_layers_2_hidden_16",
        "jobid_neuralsurv": "synthetic_150",
    },
]

selected_methods = ["True", "SumoNet", "CoxCC", "Neuralsurv"]

colors = {
    "True": "black",
    "SumoNet": "#1f77b4",
    "CoxCC": "#2ca02c",
    "Neuralsurv": "#ff7f0e",
}

linestyles = {
    "True": "solid",
    "SumoNet": "dashed",
    "CoxCC": "dashed",
    "Neuralsurv": "dashed",
}

fig, axs = plt.subplots(1, len(configs), figsize=(14, 4), sharex=True, sharey=True)

for i, config in enumerate(configs):
    subsample_n = config["subsample_n"]
    date = config["date"]
    jobid = config["jobid"]
    jobid_neuralsurv = config["jobid_neuralsurv"]
    dataset_name = "synthetic_data"

    data = load_synthetic_data(subsample_n=subsample_n)
    data_test = data["test"]["np.array"]
    data_train = data["train"]["np.array"]
    time_train, event_train, x_train = (
        data_train["time"],
        data_train["event"],
        data_train["x"],
    )
    time_test, event_test, x_test = (
        data_test["time"],
        data_test["event"],
        data_test["x"],
    )

    time_max = max(115.54582, time_test.max())
    delta_time = time_max / 40
    num = int(time_max // delta_time) + 1
    times = np.linspace(1e-6, time_max, num=num)

    surv_0_true = 1 - lognorm.cdf(times, s=0.8, scale=np.exp(3))
    surv_1_true = 1 - lognorm.cdf(times, s=1.0, scale=np.exp(3.5))
    index_0 = np.where(x_test[:, 0] == 0)[0]
    index_1 = np.where(x_test[:, 0] == 1)[0]
    surv_test = surv_0_true * len(index_0) / len(time_test) + surv_1_true * len(
        index_1
    ) / len(time_test)
    true = pd.DataFrame({"surv": surv_test, "times": times, "method": "True"})

    # Load benchmarks
    benchmark = []
    benchmark_metric = []
    input_dir = os.path.join(output_dir, "benchmark", date)
    for method in [
        "CoxCC",
        "sumo_net",
    ]:
        file_path = os.path.join(
            input_dir, f"{dataset_name}_{jobid}_{method}_surv_new_times.npy"
        )
        file_path_metric = os.path.join(
            input_dir, f"{dataset_name}_{jobid}_{method}_evaluation_metrics.csv"
        )
        if os.path.exists(file_path):
            surv_new_times = np.load(file_path)
            df = pd.DataFrame({"surv": np.mean(surv_new_times, axis=0), "times": times})
            df["method"] = method
            benchmark.append(df)
            df = pd.read_csv(file_path_metric)
            df["method"] = method
            benchmark_metric.append(df)
    benchmark = pd.concat(benchmark, ignore_index=True)
    benchmark_metric = pd.concat(benchmark_metric, ignore_index=True)
    benchmark["method"] = benchmark["method"].replace("sumo_net", "SumoNet")
    benchmark_metric["method"] = benchmark_metric["method"].replace(
        "sumo_net", "SumoNet"
    )

    # Load neuralsurv
    file_path = os.path.join(output_dir, jobid_neuralsurv, "survival_function_test.npy")
    surv_new_times = np.load(file_path)
    surv = np.mean(surv_new_times, axis=0)
    surv_median = np.median(surv, axis=1)
    if subsample_n == 100:
        lower, upper = 0.5, 99.5
    else:
        lower, upper = 5, 95
    surv_lower = np.percentile(surv, lower, axis=1)
    surv_upper = np.percentile(surv, upper, axis=1)
    # surv_median = np.median(surv_new_times, axis=(0, 2))
    # surv_lower = np.percentile(surv_new_times, 5, axis=(0, 2))
    # surv_upper = np.percentile(surv_new_times, 95, axis=(0, 2))
    neuralsurv = pd.DataFrame(
        {
            "surv": surv_median,
            "surv_lower": surv_lower,
            "surv_upper": surv_upper,
            "times": times,
            "method": "Neuralsurv",
        }
    )

    file_path_metric = os.path.join(
        output_dir, jobid_neuralsurv, "evaluation_metrics.csv"
    )
    neuralsurv_metric = pd.read_csv(file_path_metric)
    neuralsurv_metric["method"] = "Neuralsurv"
    neuralsurv_metric.rename(columns={"index": "metric"}, inplace=True)

    full_df = pd.concat([benchmark, neuralsurv, true], axis=0)
    full_df_metric = pd.concat([benchmark_metric, neuralsurv_metric], axis=0)

    filtered_df = full_df[full_df["method"].isin(selected_methods)]
    filtered_df_metric = full_df_metric[full_df_metric["method"].isin(selected_methods)]
    filtered_df_metric = filtered_df_metric[filtered_df_metric["metric"] == "ipcw_ibs"]

    text_lines = ["IPCW IBS ↓"]
    for method in selected_methods[1:]:  # Skip "True"
        row = filtered_df_metric[filtered_df_metric["method"] == method]
        if not row.empty:
            val = row["value"].values[0]
            color = colors[method]
            text_lines.append(f"{method}: {val:.3f}")

    ax = axs[i]
    for method in selected_methods:

        method_data = filtered_df[filtered_df["method"] == method]
        ax.plot(
            method_data["times"],
            method_data["surv"],
            color=colors[method],
            linestyle=linestyles[method],
            label=method,
        )
        if method == "Neuralsurv":
            ax.fill_between(
                method_data["times"],
                method_data["surv_lower"],
                method_data["surv_upper"],
                color=colors[method],
                alpha=0.15,
            )

    ax.set_title(rf"$N = {subsample_n}$")
    if i == 0:
        ax.set_ylabel("Survival Function")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Time")

    # Optional: color per line (advanced)
    ax.text(
        0.01,
        0.2 - (-1 + 1) * 0.075,  # Y offset per line
        text_lines[0],
        transform=ax.transAxes,
        fontsize=12,
        color="black",
        verticalalignment="bottom",
        horizontalalignment="left",
    )
    for j, line in enumerate(text_lines[1:]):
        method = line.split(":")[0]
        ax.text(
            0.01,
            0.2 - (j + 1) * 0.06,  # Y offset per line
            line,
            transform=ax.transAxes,
            fontsize=12,
            color=colors[method],
            verticalalignment="bottom",
            horizontalalignment="left",
        )


plt.tight_layout()
# plt.show()

fpath = os.path.join(output_dir, configs[0]["jobid_neuralsurv"])
plt.savefig(f"{fpath}/synthetic_experiment.pdf", dpi=300, bbox_inches="tight")
plt.savefig(f"{fpath}/synthetic_experiment.png", dpi=300, bbox_inches="tight")
