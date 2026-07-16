"""
Script to read S3 bucket_name/prefix and process results to csv.
By default, always update csv with new data.
> python process_s3_to_csv.py --s3 s3://ai2-llm/eval-results/downstream/eval-for-consistent-ranking-preemption-fixed --csv_name results_ladder_5xC.csv

Add --no_refresh to skip refreshing the data from S3 and just plot existing data.

Paths:
- s3://ai2-llm/eval-results/downstream/eval-for-consistent-ranking
- s3://ai2-llm/eval-results/downstream/eval-for-consistent-ranking-preemption-fixed
- s3://ai2-llm/checkpoints/benb/results_oe-eval-internal_ladder_oldstylehf
"""
import os
import re
import boto3
import argparse
import random
import warnings
import pandas as pd
import numpy as np
import wandb
import plotly
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.colors as mcolors
from tqdm import tqdm
from util_metrics import compute_metrics_from_file, safe_eval, task_groups, get_tasks, METRICS_RC, METRICS_GENERATE, RANDOM_BASELINES
from concurrent.futures import ThreadPoolExecutor, as_completed
from olmes.tasks.oe_eval_tasks import TASK_REGISTRY
from plotly.subplots import make_subplots

random.seed(42)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

wandb_val_losses = [
    'eval/c4_en-validation/CrossEntropyLoss',
    'eval/dolma_common-crawl-validation/CrossEntropyLoss',
    'eval/pile-validation/CrossEntropyLoss',
    'eval/wikitext_103-validation/CrossEntropyLoss',
]

wandb_stats = [
    "train/CrossEntropyLoss",
    "throughput/total_tokens"
]

dtypes = {
    "group": str,
    "model": str,
    "task": str,
    "chinchilla": str,
    "step": int,
    "tokens": float,
    "compute": float,
    "metrics": object,
    **{stat: float for stat in wandb_val_losses + wandb_stats}
}

RC_TASKS = get_tasks("../all_olmes_rc_tasks.txt")

api = wandb.Api()

def get_prediction_paths(s3, bucket_name, prefix) -> list[str]:
    """Get all prediction.jsonl from all checkpoints stored in S3"""
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter='/')
    prediction_paths = []
    if 'CommonPrefixes' in response:
        for model_prefix in response['CommonPrefixes']:
            model_prefix_str = model_prefix["Prefix"]
            model_response = s3.list_objects_v2(Bucket=bucket_name, Prefix=model_prefix_str, Delimiter='/')
            if 'CommonPrefixes' in model_response:
                for step_prefix in model_response['CommonPrefixes']:
                    if step_prefix["Prefix"].endswith("s/"):  # weird ckpt
                        continue
                    step_prefix_str = step_prefix["Prefix"]
                    step_response = s3.list_objects_v2(Bucket=bucket_name, Prefix=step_prefix_str, Delimiter='/')
                    if 'CommonPrefixes' in step_response:
                        is_new_eval = all("all_olmes_" in r['Prefix'] for r in step_response['CommonPrefixes'])
                        if is_new_eval:
                            step_responses = [r for r in step_response['CommonPrefixes'] if r['Prefix'].endswith("all_olmes_rc_tasks/")]
                            assert len(step_responses) == 1, f"Multiple all_olmes_rc_tasks/ found: {step_responses}"
                            all_task_prefix = step_responses[0]['Prefix']
                            all_tasks_response = s3.list_objects_v2(Bucket=bucket_name, Prefix=all_task_prefix, Delimiter='/')
                            pred_paths = [p['Key'] for p in all_tasks_response['Contents'] if p['Key'].endswith("predictions.jsonl")]
                        # Tai's old logic on n=500
                        else:
                            pred_paths = [task_prefix["Prefix"] + "predictions.jsonl" for task_prefix in step_response['CommonPrefixes']]
                        prediction_paths.extend(pred_paths)
    return list(set(prediction_paths))

def process_prediction_path(s3, bucket_name, path):
    obj = s3.get_object(Bucket=bucket_name, Key=path)
    predictions_content = obj['Body'].read().decode('utf-8')
    group, model, chinchilla, task, step, seed = parse_train_name(path)

    # Extract metrics
    rows_list = compute_metrics_from_file(predictions_content, task)
    if rows_list is None:
        print(f"Skipping results for: {task}")
        return None

    task_config = TASK_REGISTRY[task].__dict__.get('TASK_CONFIG_DEFAULTS', {})
    # Get primary_metric in this order
    primary_metric = task_config.get("primary_metric", None)
    possible_metrics = ["primary_metric", "acc_raw", "exact_match", "f1", \
                        "mc1", "pass_at_1", "prompt_level_loose_acc", "maj_at_1"]
    aggregated_metrics = {}
    for mrow in rows_list:
        if "em" in mrow:
            mrow["exact_match"] = mrow.pop("em")
        if primary_metric is None:
            for metric in possible_metrics:
                if metric in mrow:
                    # Set name forprimary_metric
                    primary_metric = metric
                    break
        if primary_metric is None:
            print(f"Skipping task {task} due to missing primary metric: {mrow}")
            continue

        mrow["primary_metric"] = mrow[primary_metric]
        mrow["acc_raw"] = mrow["acc_raw"]
        mrow["acc_per_char"] = mrow["acc_per_char"]
        mrow["acc_per_token"] = mrow["acc_per_token"]
        mrow["acc_uncond"] = mrow["acc_uncond"]
        for key, value in mrow.items():
            if value is None or isinstance(value, str):
                continue
            if key in aggregated_metrics:
                aggregated_metrics[key].append(value)
            else:
                aggregated_metrics[key] = [value]

    mean_metrics = {k: np.mean(v) for k, v in aggregated_metrics.items()}

    row = {
        "group": group,
        "model": model,
        "task": task,
        "chinchilla": chinchilla,
        "step": step,
        "seed": seed,
        "metrics": mean_metrics
    }
    return row

def unpack_dict_column(df, col_name):
    """
    Unpack a dictionary column in a DataFrame using json_normalize.
    Return a new DataFrame with the unpacked columns joined.
    """
    # Normalize the specified column and extract the desired keys
    temp = pd.json_normalize(df[col_name], max_level=1)
    temp = temp.reset_index(drop=True)
    # # Automatically detect and drop overlapping columns
    # overlap_cols = df.columns.intersection(temp.columns)
    # if not overlap_cols.empty:
    #     temp = temp.drop(columns=overlap_cols)
    # Reset the index of the original DataFrame and join the unpacked data
    df = df.reset_index(drop=True).drop(columns=[col_name]).join(temp)
    print(f"Columns from unpacking: {df.columns}")
    return df

def calc_compute(model_size_str: str, tokens: int):
    if model_size_str[-1].lower() == "m":
        scale = 1e6
    elif model_size_str[-1].lower() == "b":
        scale = 1e9
    else:
        raise ValueError(f"Unknown model size unit {model_size_str[-1]} from `{model_size_str}`")

    model_size_str = model_size_str.replace("olmo-", "")
    num_parameters = int(float(model_size_str[:-1]) * scale)
    return 6 * num_parameters * tokens

def parse_train_name(path):
    """
    Parse the S3 path to extract the group, model, chinchilla, task, and step.
    Example input path structure: "checkpoints/benb/olmo-150M-no_math_no_code-1xC/step500/mmlu/predictions.jsonl"
    """
    parts = path.split('/')
    assert re.match(r'.*-\d+xC(-\d+)?$', parts[3]), f"Invalid model name format: {parts[3]}"

    if re.match(r'.*-\d+xC-\d+$', parts[3]):
        group_model_chinchilla = parts[3].rsplit('-', 3)
        seed = group_model_chinchilla[3]
        assert re.match(r'\d', seed), f"Invalid model name parsing: {parts[3]} -> {group_model_chinchilla}"
        seed = int(seed)
    elif re.match(r'.*-\d+xC$', parts[3]):
        group_model_chinchilla = parts[3].rsplit('-', 2)
        seed = None
    else:
        raise ValueError(f"Invalid model name format: {parts}")

    group = group_model_chinchilla[0]
    model = group_model_chinchilla[1]
    assert re.match(r'\d+[M|B]', model), f"Invalid model size parsing: {model}"
    chinchilla = group_model_chinchilla[2]
    assert re.match(r'\d+xC', chinchilla), f"Invalid chinchilla parsing: {chinchilla}"
    step = int(re.search(r'step(\d+)', parts[4]).group(1))
    if "all_olmes" in path:
        task = None
        if "_rc_tasks" in path:
            task_re = re.search(r'task-\d+-(.*?)-predictions\.jsonl', parts[6])
            if task_re:
                task = task_re.group(1)
    else:
        task = parts[5]
    return group, model, chinchilla, task, step, seed

def parse_s3_path(s3_path):
    if not s3_path.startswith("s3://"):
        raise ValueError("Path must start with 's3://'")
    # Remove the 's3://' prefix
    s3_path = s3_path[5:]
    bucket_name, _, prefix = s3_path.partition("/")
    if not prefix.endswith("/"):
        prefix += "/"
    return bucket_name, prefix

# Write Wandb stats
def fetch_wandb_data(project, filters, keys):
    """Fetch all the WandB data based on filters."""

    def _parse_exp_name(exp_name):
        """
        Parse the experiment name to extract the group, model, chinchilla, and step.
        ie. "olmo-150M-no_math_no_code-1xC-step500"
        """
        group_model_chinchilla = exp_name.rsplit('-', 2)
        assert len(group_model_chinchilla) == 3, f"Invalid experiment name parsing: {exp_name} -> {group_model_chinchilla}"
        group = group_model_chinchilla[0]
        model = group_model_chinchilla[1]
        chinchilla = group_model_chinchilla[2]
        return group, model, chinchilla

    experiments = api.runs(f"ai2-llm/{project}", filters=filters, order="-created_at")
    data = {}
    for exp in experiments:
        history = exp.history(keys=keys)
        group, model, chinchilla = _parse_exp_name(exp.name)
        # Iterate through the history and store data using the parsed key
        for _, history_row in history.iterrows():
            wandb_step = int(history_row["_step"])
            key = (group, model, chinchilla, wandb_step)
            data[key] = {k: history_row[k] for k in keys}
    return data

def get_stat_from_data(row, data, stat_name):
    key = (row['group'], row['model'], row['chinchilla'], int(row['step']))
    return data.get(key, {}).get(stat_name, np.nan)

def create_plotly_plots(df, s3_name):
    def sort_by_order(df, sorting_params):
        """
        Sorts the DataFrame based on the specified categorical order for multiple columns.
        """
        for column, categories_order in sorting_params.items():
            df[column] = pd.Categorical(df[column], categories=categories_order, ordered=True)
            df = df.sort_values(by=column)
        df = df.sort_values(by="compute")
        return df

    # Shared settings
    model_order = ['150M', '300M', '530M', '750M', '1B']
    metrics_all = METRICS_RC + ["train/CrossEntropyLoss"] + wandb_val_losses
    mix_order = sorted(df["group"].unique())
    sorting_params = {
        "metric": metrics_all,
        "model": model_order,
        "group": mix_order
    }

    # Unique colors for each group
    groups = sorted(df["group"].unique())
    tasks = ["All Tasks"] + sorted(df["task_suite"].unique())
    # Randomly get 200 colors for each data mix
    colors = sns.color_palette("hsv", 200)
    random.shuffle(colors)
    palette = [mcolors.rgb2hex(c) for c in colors[:len(groups)]]
    assert len(palette) >= len(groups), f"Insufficient colors for {len(groups)} groups: {len(palette)}"
    color_map = {group: palette[i % len(palette)] for i, group in enumerate(groups)}

    def melt_df_and_add_aggregate(df):
        """
        Melt the DataFrame to get the metrics as individual columns.
        Also get a grouped DataFrame with the overall mean for OLMES
        """
        df_filtered = df.copy()
        df_filtered = unpack_dict_column(df, col_name="metrics")

        # Melt
        df_melted = df_filtered.melt(id_vars=["group", "model", "task_suite", "tokens", "compute", "step"],
                                     value_vars=metrics_all,
                                     var_name="metric", value_name="value").dropna(subset=["value"])

        decimals = 6
        df_melted["tokens"] = df_melted["tokens"].round(decimals)
        df_melted["compute"] = df_melted["compute"].round(decimals)
        df_melted["step"] = df_melted["step"].round(decimals)

        # Group by both individual tasks and the aggregated
        df_grouped_task = df_melted.groupby(["group", "model", "tokens", "compute", "step", "metric", "task_suite"])["value"].mean().reset_index()
        df_grouped_all = df_melted.groupby(["group", "model", "tokens", "compute", "step", "metric"])["value"].mean().reset_index()
        df_grouped_all["task_suite"] = "All Tasks"

        # Concatenate the two dataframes
        df_ret = pd.concat([df_grouped_task, df_grouped_all], axis=0)
        df_ret = df_ret.sort_values(by=['compute'])
        return df_ret

    # Part 1: Plotting Across Multiple Models
    def plot_full(df_melted):
        fig = make_subplots(
            rows=len(metrics_all), cols=len(model_order), shared_xaxes=False, shared_yaxes=False,
            subplot_titles=[f"{model}" for model in model_order] * len(metrics_all),
            vertical_spacing=0.02
        )

        task_traces = {task: [] for task in tasks}

        # Create traces for each combination of metric and model
        for i, metric in enumerate(metrics_all):
            for j, model in enumerate(model_order):
                df_model_metric = df_melted[(df_melted["metric"] == metric) & (df_melted["model"] == model)]
                for group in groups:
                    for task in tasks:
                        df_group_task = df_model_metric[(df_model_metric["group"] == group) &
                                                        (df_model_metric["task_suite"] == task)]
                        if df_group_task.empty:
                            continue
                        df_group_task = sort_by_order(df_group_task, sorting_params)
                        trace = go.Scatter(
                            x=df_group_task["compute"],
                            y=df_group_task["value"],
                            mode='lines+markers',
                            name=group,
                            legendgroup=group,
                            line=dict(color=color_map[group]),
                            opacity=0.7,
                            showlegend=(i == 0 and j == 0),
                            visible=True if task == "All Tasks" else False
                        )
                        fig.add_trace(trace, row=i + 1, col=j + 1)
                        task_traces[task].append(len(fig.data) - 1)

                        # Add baseline if it's the primary metric
                        if metric == "primary_metric" and task != "All Tasks":
                            baseline_value = RANDOM_BASELINES.get(task, None)
                            if baseline_value is not None:
                                shape = go.Scatter(
                                    x=[df_group_task["compute"].min(), df_group_task["compute"].max()],
                                    y=[baseline_value, baseline_value],
                                    mode="lines",
                                    line=dict(color="black", dash="dash"),
                                    name=f"{task} baseline",
                                    showlegend=False,
                                    visible=True if task == "All Tasks" else False
                                )
                                fig.add_trace(shape, row=i + 1, col=j + 1)
                                task_traces[task].append(len(fig.data) - 1)


        # Create dropdown for task selection
        dropdown_buttons = create_dropdown_buttons(task_traces, fig)

        # Set y-axis titles for metrics
        for i, metric in enumerate(metrics_all):
            metric = metric.replace("eval/", " ").replace("CrossEntropyLoss", "CELoss").replace("-validation", "-val")
            fig.update_yaxes(title_text=metric, row=i + 1, col=1)

        # Update layout
        fig.update_layout(
            height=200 * len(metrics_all), width=300 * len(model_order),
            showlegend=True, title_x=0.5,
            legend=dict(yanchor="top", y=1, xanchor="left", x=1),
            updatemenus=[dict(
                active=0,  # Default to the first task
                buttons=dropdown_buttons,
                direction="down",
                showactive=True,
                x=0.17, xanchor="left",
                y=1.03, yanchor="top"
            )]
        )
        # Modify layout annotations (subplot titles) to only show on the first row
        for i, annotation in enumerate(fig.layout.annotations):
            if i >= len(model_order):
                annotation.text = ""
        plotly.io.write_html(fig, file=f"{s3_name}_olmes_curves_task_filter.html", auto_open=True)
        print(f"Plot saved to '{s3_name}_olmes_curves_task_filter.html'")

    # Part 2: Plotting Latest Step per Group and Task
    def plot_latest_step(df_melted):
        df_melted = df_melted.sort_values(by=["group", "model", "metric", "task_suite", "step"])
        df_latest_step = df_melted.groupby(["group", "model", "metric", "task_suite"], as_index=False).apply(
            lambda x: x.loc[x["step"].idxmax()]).reset_index(drop=True)

        fig = make_subplots(
            rows=len(metrics_all), cols=1, shared_xaxes=False, shared_yaxes=False,
            vertical_spacing=0.02
        )

        task_traces = {task: [] for task in tasks}

        # Iterate through each metric and group
        for i, metric in enumerate(metrics_all):
            df_metric = df_latest_step[df_latest_step["metric"] == metric]
            for group in groups:
                for task in tasks:
                    df_group_task = df_metric[(df_metric["group"] == group) &
                                              (df_metric["task_suite"] == task)]
                    if df_group_task.empty:
                        continue
                    df_group_task = sort_by_order(df_group_task, sorting_params)
                    trace = go.Scatter(
                        x=df_group_task["compute"],
                        y=df_group_task["value"],
                        mode='lines+markers',
                        name=group,
                        legendgroup=group,
                        line=dict(color=color_map[group]),
                        opacity=0.7,
                        showlegend=(i == 0),
                        visible=True if task == "All Tasks" else False
                    )
                    fig.add_trace(trace, row=i + 1, col=1)
                    task_traces[task].append(len(fig.data) - 1)

                    # Add baseline if it's the primary metric
                    if metric == "primary_metric" and task != "All Tasks":
                        baseline_value = RANDOM_BASELINES.get(task, None)
                        if baseline_value is not None:
                            shape = go.Scatter(
                                x=[df_group_task["compute"].min(), df_group_task["compute"].max()],
                                y=[baseline_value, baseline_value],
                                mode="lines",
                                line=dict(color="black", dash="dash"),
                                name=f"{task} baseline",
                                showlegend=False,
                                visible=True if task == "All Tasks" else False
                            )
                            fig.add_trace(shape, row=i + 1, col=1)
                            task_traces[task].append(len(fig.data) - 1)

        # Create dropdown for task selection
        dropdown_buttons = create_dropdown_buttons(task_traces, fig)

        # Set y-axis titles for metrics
        for i, metric in enumerate(metrics_all):
            metric = metric.replace("eval/", " ").replace("CrossEntropyLoss", "CELoss")
            fig.update_yaxes(title_text=metric, row=i + 1, col=1)

        # Update layout
        fig.update_layout(
            height=200 * len(metrics_all), width=800,
            showlegend=True, title_x=0.5,
            legend=dict(yanchor="top", y=1, xanchor="left", x=1),
            updatemenus=[dict(
                active=0,  # Default to the first task
                buttons=dropdown_buttons,
                direction="down",
                showactive=True,
                x=0.17, xanchor="left",
                y=1.03, yanchor="top"
            )]
        )
        # Modify layout annotations (subplot titles) to only show on the first row
        for i, annotation in enumerate(fig.layout.annotations):
            if i >= len(model_order):
                annotation.text = ""
        plotly.io.write_html(fig, file=f"{s3_name}_olmes_curves_last_step_task_filter.html", auto_open=True)
        print(f"Plot saved to '{s3_name}_olmes_curves_last_step_task_filter.html'")

    # Create dropdown buttons for task selection
    def create_dropdown_buttons(task_traces, fig):
        dropdown_buttons = []
        for task in tasks:
            button = dict(
                args=[{'visible': [False] * len(fig.data)}],  # Set all traces to invisible
                label=task,
                method="update"
            )
            for trace_index in task_traces[task]:
                button['args'][0]['visible'][trace_index] = True
            dropdown_buttons.append(button)
        return dropdown_buttons

    # Melt DataFrame and generate plots
    df_melted = melt_df_and_add_aggregate(df)

    # save
    df_melted.to_csv("results_ladder_5xC_melted.csv", index=False)

    plot_full(df_melted)
    plot_latest_step(df_melted)


def main(bucket_name, prefix, csv_name, no_refresh, batch_size):
    from botocore.config import Config

    # Create a custom configuration with a higher max_pool_connections value
    config = Config(
        max_pool_connections=batch_size
    )
    s3 = boto3.client('s3', config=config)

    print(f"Batch size: {batch_size}")
    existing_combinations = set()
    df = pd.DataFrame(columns=list(dtypes.keys()))
    if os.path.exists(csv_name):
        df = pd.read_csv(csv_name, dtype=dtypes)
        existing_combinations = set(df.apply(
            lambda row: (
                row['task'],
                row['group'],
                row['model'],
                row['chinchilla'],
                row['step'],
                row['seed'] if 'seed' in row else None),
            axis=1
        ))

    df = df.astype(dtypes)
    df = df[df["chinchilla"] == "5xC"]
    print("Filter to only 5xC...")
    rows_before = df.shape[0]

    if not no_refresh:
        print("Reading all available result files from S3 prefix path...")
        prediction_paths = get_prediction_paths(s3, bucket_name, prefix)
        print(f"Found total files: {len(prediction_paths)} - Existing: {df.shape[0]}")
        new_prediction_paths = []
        for path in prediction_paths:
            group, model, chinchilla, task, step, seed = parse_train_name(path)
            if task is None or task not in RC_TASKS:
                continue
            if (task, group, model, chinchilla, step, seed) not in existing_combinations:
                new_prediction_paths.append(path)
        print("Processing new prediction files: ", len(new_prediction_paths))

        new_data = []
        for i in tqdm(range(0, len(new_prediction_paths), batch_size), desc="Processing batches"):
            batch_paths = new_prediction_paths[i:i+batch_size]
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = {executor.submit(process_prediction_path, s3, bucket_name, path): path for path in batch_paths}
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        new_data.append(result)

        if new_data:
            # Cache the new data as a temp file
            temp_file = "temp_new_data.csv"
            while os.path.exists(temp_file):
                temp_file = temp_file.replace(".csv", "_temp.csv")
            pd.DataFrame(new_data).to_csv(temp_file, index=False)
            print(f"New data cached to '{temp_file}'")
            df_new = pd.DataFrame(new_data)

            # Call wandb
            print("Fetching WandB data...")
            wandb_data = {}
            for mix in tqdm(df_new['group'].unique()):
                for size in df_new['model'].unique():
                    for chinchilla in df_new['chinchilla'].unique():
                        filters = {"display_name": f"{mix}-{size}-{chinchilla}"}
                        if chinchilla == "2xC":
                            if mix in ["no_math_no_code", "no_code", "no_reddit", "no_flan"]:
                                project = "olmo-ladder-benb"
                            else:
                                project = "olmo-ladder"
                        else:
                            if any(substring in mix for substring in ["DCLM", "dolma17"]):
                                project = "olmo-ladder-ianm"
                            else:
                                project = "olmo-ladder-benb"
                            filters.update({"group": f"{mix}-{size}-{chinchilla}"})

                        # Fetch and accumulate data for each combination
                        stat_data = fetch_wandb_data(project, filters, wandb_stats + wandb_val_losses)
                        wandb_data.update(stat_data)
            # Apply data to df
            for stat in wandb_val_losses + wandb_stats:
                df_new[stat] = df_new.apply(lambda row: get_stat_from_data(row, wandb_data, stat), axis=1)

            # Concat and calculate C
            df = pd.concat([df, df_new], ignore_index=True)
            df["tokens"]  = df["throughput/total_tokens"]
            df["compute"] = df.apply(lambda row: calc_compute(row['model'], row['tokens']), axis=1)
            df = df.astype(dtypes)

            df.to_csv(csv_name, index=False)
            print("Rows before: ", rows_before)
            print("Rows after: ", df.shape[0])
            print(f"Results saved to '{csv_name}'")
    if df.empty:
        print("No data found...")
        return

    # Create interactive viz
    print("Creating interactive Plotly...")
    df.loc[df['step'] == 0, 'compute'] = 0.0
    df.loc[df['step'] == 0, 'tokens'] = 0.0
    df["task_suite"] = df["task"].apply(lambda x: next((name for prefix, name in task_groups.items() if x.startswith(prefix)), x))
    df["metrics"] = df["metrics"].apply(lambda x: safe_eval(str(x)))
    create_plotly_plots(df, os.path.basename(os.path.normpath(prefix)))

    # Clean up the temp file
    if os.path.exists(temp_file):
        os.remove(temp_file)
        print(f"Temporary file '{temp_file}' has been removed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process results from S3 path and save to CSV.')
    parser.add_argument('--s3', type=str, required=True, help='Full S3 path in the format s3://bucket/prefix/')
    parser.add_argument('--csv_name', type=str, default='results_ladder_5xC.csv', help='File to save the results')
    parser.add_argument('--no_refresh', action='store_true', help='Skip refreshing the data from S3')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for processing S3 files')
    args = parser.parse_args()

    bucket_name, prefix = parse_s3_path(args.s3)
    main(bucket_name, prefix, args.csv_name, args.no_refresh, args.batch_size)
