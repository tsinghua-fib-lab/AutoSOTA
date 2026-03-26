import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

output_dir = "/Users/melodiemonod/projects/2025/neuralsurv"

date = "2025-07-28"
dataset_name = "synthetic_data"
jobid = "sub_25_layers_2_hidden_16"
jobid_neuralsurv = "synthetic_25"

methods = [
    "LogisticHazard",
    "PCHazard",
    "Deepsurv",
    "CoxCC",
    "DeepHitSingle",
    "BCESurv",
    "MTLR",
    "PMF",
    "CoxTime",
    "Dysurv",
    "sumo_net",
    "dqs",
]

# Load benchmark
benchmark = []

input_dir = os.path.join(output_dir, "benchmark", date)
for method in methods:
    file_path = os.path.join(
        input_dir, f"{dataset_name}_{jobid}_{method}_evaluation_metrics.csv"
    )
    df = pd.read_csv(file_path)  # or sep="," if comma-delimited
    df["method"] = method
    benchmark.append(df)
benchmark = pd.concat(benchmark, ignore_index=True)

# Load neuralsurv
neuralsurv = []

file_path = os.path.join(output_dir, jobid_neuralsurv, "evaluation_metrics.csv")
df = pd.read_csv(file_path)
df["method"] = "Neuralsurv"
neuralsurv.append(df)
neuralsurv = pd.concat(neuralsurv, ignore_index=True)
neuralsurv.rename(columns={"index": "metric"}, inplace=True)

#  Concatenate
full_df = pd.concat([benchmark, neuralsurv], axis=0)

# Step 3: Plot one boxplot per metric
metrics = ["c_index_antolini", "ipcw_ibs", "d_calibration", "km_calibration"]

for metric in metrics:
    metric_df = full_df[full_df["metric"] == metric]

    # Remove method with nan
    methods_with_nan = metric_df[metric_df["value"].isna()]["method"].unique()
    metric_df = metric_df[~metric_df["method"].isin(methods_with_nan)]

    # Compute ordering by mean value
    method_order = (
        metric_df.groupby("method")["value"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=metric_df, x="method", y="value", order=method_order)
    plt.title(f"Boxplot of {metric}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Make table
    methods_ordered = [
        "MTLR",
        "DeepHitSingle",
        "Deepsurv",
        "LogisticHazard",
        "CoxTime",
        "CoxCC",
        "PMF",
        "PCHazard",
        "BCESurv",
        "Dysurv",
        "sumo_net",
        "dqs",
        "Neuralsurv",
    ]
    filtered_df = metric_df[metric_df["method"].isin(methods_ordered)].dropna(
        subset=["value"]
    )
    filtered_df["method"] = pd.Categorical(
        filtered_df["method"], categories=methods_ordered, ordered=True
    )
    sorted_df = filtered_df.sort_values("method")
    sorted_df["value"] = sorted_df["value"].apply(lambda x: float(f"{x:.3f}"))
    print(sorted_df["value"].to_numpy())
    print(method_order)
