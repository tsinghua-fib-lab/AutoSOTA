import time
import numpy as np
import pandas as pd
import os
import json
import logging

LOSS_COLS = [] # ["train/CrossEntropyLoss"] # disabling loss columns for now since it's not good and maybe missing for some tasks

RC_METRICS = [
    "primary_metric",
    "correct_prob",
    "correct_prob_per_token",
    "correct_prob_per_char",
    "margin",
    "margin_per_token",
    "margin_per_char",
    "total_prob",
    "total_prob_per_token",
    "total_prob_per_char",
    # "uncond_correct_prob",  # uncond not supported in all tasks
    # "uncond_correct_prob_per_token",
    # "uncond_correct_prob_per_char",
    "norm_correct_prob",
    "norm_correct_prob_per_token",
    "norm_correct_prob_per_char",
]

METRICS_GENERATE = [
    "logit",
    "logit_per_token",
    "logit_per_char",
    "score",
    "score_per_token",
    "score_per_char"
]

RANDOM_BASELINES = {
    'arc_challenge': 1/4,
    'arc_easy': 1/4,
    'boolq': 1/2,
    'hellaswag': 1/4,
    'csqa': 1/5,
    'openbookqa': 1/4,
    'socialiqa': 1/3,
    'piqa': 1/2,
    'winogrande': 1/2,
    'MMLU': 1/4,
}

ACC_METRICS = [
    "acc_raw",
    "acc_per_char",
    "acc_per_token",
    # "acc_uncond" # uncond not supported in all tasks
]

task_groups = {
    "mmlu": "MMLU",
    "bbh": "BBH",
    "tydiqa": "TyDiQA",
    "minerva_math": "Minerva Math",
}

model_order = ['4M',"6M","8M","10M","14M", "16M", '20M', '60M', '90M', '150M', '300M', '530M', '750M', '1B']

def save_data(data, file_name, out_dir="outputs") -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Split file name and extension
    base_name, file_type = os.path.splitext(file_name)

    # Detect file type if not provided
    if not file_type:
        if isinstance(data, pd.DataFrame):
            file_type = ".csv"
        elif isinstance(data, (dict, list)):
            file_type = ".json"
        else:
            raise ValueError("Unsupported data type")

    # Construct final file path
    file_type = file_type if file_type.startswith(".") else f".{file_type}"
    file_path = os.path.join(out_dir, f"{base_name}{file_type}")

    try:
        if file_type == ".csv":
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Data must be a DataFrame for CSV files.")
            data.to_csv(file_path, index=False)
        elif file_type == ".json":
            if not isinstance(data, (dict, list)):
                raise ValueError("Data must be JSON-compatible for JSON files.")
            with open(file_path, "w") as f:
                json.dump(data, f, indent=4)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        logging.info(f"Data saved to '{file_path}'...")
    except Exception as e:
        logging.error(f"Error saving data to '{file_path}': {e}")
        raise e

def load_data(file_name, in_dir="outputs"):
    file_path = os.path.join(in_dir, file_name)
    try:
        # Split file name and extension
        base_name, file_type = os.path.splitext(file_name)

        # Detect file type if not provided
        if not file_type:
            if file_name.endswith(".csv"):
                file_type = ".csv"
            elif file_name.endswith(".json"):
                file_type = ".json"
            else:
                raise ValueError("Unsupported file type")

        if file_type == ".csv":
            data = pd.read_csv(file_path)
        elif file_type == ".json":
            with open(file_path, "r") as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        logging.info(f"Data loaded from '{file_path}'...")
        return data
    except Exception as e:
        logging.error(f"Error loading data from '{file_path}': {e}")
        raise e

def timeit(func):
    """
    A decorator to measure the execution time of a function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        duration = end_time - start_time
        logging.info(f"{func.__name__} executed in {duration:.4f} seconds")
        return result
    return wrapper

def get_tasks(file_path):
    with open(file_path, "r") as f:
        tasks = [t.split(":")[0] for t in f.readlines()]
    return tasks

def safe_eval(x):
    """Utility for reading 'metrics' col which is a dict in DataFrame"""
    try:
        result = eval(x)
        # Traverse dict to replace NaN values
        if isinstance(result, dict):
            result = {key: (None if (isinstance(value, float) and np.isnan(value)) else value)
                      for key, value in result.items()}
        return result
    except:
        # If fails, return the original string or handle it as needed
        return x

def unpack_dict_column(df, col_name):
    """
    Unpack a dictionary column in a DataFrame using json_normalize.
    Return a new DataFrame with the unpacked columns joined.
    """
    temp = pd.json_normalize(df[col_name], max_level=1)
    temp = temp.reset_index(drop=True)
    df = df.reset_index(drop=True).drop(columns=[col_name]).join(temp)
    # print(f"Columns from unpacking: {df.columns}")
    return df

def format_tokens(tokens: int):
    if tokens >= 1_000_000_000:  # Check for billions
        return f"{tokens / 1_000_000_000:.1f}B"
    elif tokens >= 1_000_000:  # Check for millions
        return f"{tokens / 1_000_000:.1f}M"
    else:
        return str(tokens)

def find_common_checkpoints(metric_values1, metric_values2):
    """Find all common checkpoints between two metric arrays."""
    # Identify non-NaN indices for both arrays
    valid_indices1 = ~np.isnan(metric_values1)
    valid_indices2 = ~np.isnan(metric_values2)
    common_indices = np.where(valid_indices1 & valid_indices2)[0]
    if not len(common_indices):
        raise ValueError("No common checkpoints found between the two mixes.")
    return common_indices

def clean_nans(arr1, arr2):
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    mask = np.isfinite(arr1) & np.isfinite(arr2)
     # Check if any NaNs were removed by comparing the original and filtered lengths
    changed = not np.all(mask)
    # Apply the mask to filter out NaN indices
    filtered_arr1 = arr1[mask].tolist()
    filtered_arr2 = arr2[mask].tolist()
    return filtered_arr1, filtered_arr2, changed

def groupby_mean(df, groupby_cols, metric_names) -> pd.DataFrame:
    """For each metric col, groupby and calculate its average and count of non-null metric values"""
    agg_dict = {}
    # Define aggregation rules for each metric
    for col in metric_names:
        agg_dict[col] = ['mean']
    grouped = df.groupby(groupby_cols).agg(agg_dict)
    # # Flatten multi-level column names
    grouped.columns = [col for col, agg in grouped.columns]
    return grouped.reset_index()

def generate_pivot_table(df, model_order, csv_name):
    """
    Generate a pivot table for cross-scale data, ensuring safety and accuracy in metric calculations.
    """
    # Order by model_order
    df['model'] = pd.Categorical(df['model'], categories=model_order, ordered=True)
    sorted_metric_names = list(sorted(df['metric'].unique()))
    df = df.round(4)

    # Prepare to build the pivot table
    pivot_data = []
    metric_set = False

    for model in model_order:
        model_df = df[df['model'] == model]

        for (prop,), model_prop_df in model_df.groupby(['proportion']):
            model_prop_df = model_prop_df.sort_values(by=['metric', 'proportion']).reset_index(drop=True)

            # Check unique
            props = model_prop_df["proportion"].unique()
            computes = model_prop_df["compute"].unique()
            prop_targets = model_prop_df["proportion_target"].unique()
            # tasks_with_metrics = model_prop_df["tasks_with_metric"].unique()
            tokenss = model_prop_df["tokens"].unique()
            # for name, arr in [("proportion", props), ("proportion_target", prop_targets), ("tasks_with_metric", tasks_with_metrics), ("tokens", tokenss)]:
                # if len(arr) > 1:
                #     logging.warning(f"({model}, {prop}) has multiple values for '{name}': {arr}")

            # Assign
            pb = float(prop)
            # tasks_with_metric = tasks_with_metrics[0]
            pb_compute = computes[0]
            pb_target = prop_targets[0]
            tokens_formatted = format_tokens(tokenss[0])

            if model_prop_df.empty:
                continue

            # temp = model_prop_df[["metric", "tasks_with_metric"]]
            # assert list(cross_scale_df["metric"].unique()) == sorted_metric_names

            # Initialize dictionaries for metric data
            metrics_data = {metric: {
                'correct': 0,
                'incorrect': 0,
                'abstain': 0,
                'total': 0,
                'precision': '-',
            } for metric in sorted_metric_names}

            # Iterate through each row to populate metrics_data
            for _, row in model_prop_df.iterrows():
                metric = row['metric']
                if metric in metrics_data:
                    metrics_data[metric]['correct'] = row['correct_count']
                    metrics_data[metric]['incorrect'] = row['incorrect_count']
                    metrics_data[metric]['abstain'] = row['abstain_count']
                    metrics_data[metric]['binary_accuracy'] = row['binary_accuracy']

            # Calculate totals and precision
            for metric, data in metrics_data.items():
                total = data['correct'] + data['incorrect']
                correct = data['correct']
                data['total'] = total
                if total > 0:
                    precision = correct / total
                    data['precision'] = f"{precision:.4f}"
                    assert round(precision, 4) == round(data['binary_accuracy'], 4), f"{precision:.4f} vs. {data['binary_accuracy']:.4f}"

            # Ensure metrics_data is in the correct order using sorted_metric_names
            formatted = {
                metric: f"{metrics_data[metric]['precision']} ({metrics_data[metric]['correct']}/{metrics_data[metric]['incorrect']}/{metrics_data[metric]['abstain']})"
                for metric in sorted_metric_names
            }
            # Build pivot data
            if not metric_set:
                # Include metric names as the first column
                pivot_data.append(pd.DataFrame({
                    ('Metric', '', '', '', '', ''): sorted_metric_names,
                    ('Cross Scales', model, pb, pb_compute, pb_target, tokens_formatted): list(formatted.values())
                }))
                metric_set = True
            else:
                # Append only data columns
                pivot_data.append(pd.DataFrame({
                    ('Cross Scales', model, pb, pb_compute, pb_target, tokens_formatted): list(formatted.values())
                }))

    # Concat and write pivot table
    pivot_df = pd.concat(pivot_data, axis=1)
    pivot_df.columns = pd.MultiIndex.from_tuples(pivot_df.columns)
    pivot_df.to_csv(csv_name, index=False)
    logging.info(f"Wrote pivot to '{csv_name}'...")
