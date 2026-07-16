
import yaml
import argparse
import logging
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from target_filter import TargetFilter
from data_method import Raw, ExponentialSmoothing, ScalingLawMethod
from evaluator import Evaluator, get_transformed_values_compute_and_seeds
from visualization import Visualization

from utils import RC_METRICS,LOSS_COLS, \
    model_order, ACC_METRICS, groupby_mean, \
    task_groups, safe_eval, get_tasks, \
    unpack_dict_column, clean_nans, save_data, \
    timeit, generate_pivot_table, load_data

RC_TASKS = get_tasks("../../all_olmes_rc_tasks.txt")

def load_yaml_config(config_file):
    """Load configuration from a YAML file."""
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def main(config_file):
    # Load YAML configuration
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    csv_name = config['input_csv']
    out_dir = config["out_dir"]
    proportions = config['proportions']
    # compute_bin_width_as_proportion = config.get('compute_bin_width_as_proportion', None)
    target_model = config['target_model']
    bad_mixes = config['bad_mixes']
    task_aggregation = config['task_aggregation']
    plots_dir = config['plots_dir']

    groupby_cols = ['model', 'group', 'seed', 'step', 'tokens', 'compute', 'task_group']
    task_metrics = RC_METRICS + LOSS_COLS + ACC_METRICS

    # Init target filter
    target_filter_config = config['target_filter']
    method = target_filter_config['method']
    if method == "log_fit":
        target_filter = TargetFilter(
            method=method,
            target_model=target_model,
            method_args=target_filter_config.get('method_args', {})
        )
    elif method == "seed":
        target_filter = TargetFilter(
            method=method,
            target_model=target_model
        )
    elif method is None:
        target_filter = TargetFilter(target_model=target_model)
    else:
        raise ValueError(f"Unsupported target_filter method: {method}")

    # Init data transform methods
    primary_transform_method_name = config['data_methods']['primary']
    metric_transform_method_name = config['data_methods']['metric']

    if primary_transform_method_name == "exponential_smoothing":
        primary_transform_method = ExponentialSmoothing()
    elif primary_transform_method_name == "scaling_law":
        primary_transform_method = ScalingLawMethod()
    elif primary_transform_method_name is None:
        primary_transform_method = Raw()
    else:
        raise ValueError(f"Unsupported data_method primary: {primary_transform_method_name}")

    if metric_transform_method_name == "exponential_smoothing":
        metric_transform_method = ExponentialSmoothing()
    elif metric_transform_method_name == "scaling_law":
        metric_transform_method = ScalingLawMethod()
    elif metric_transform_method_name is None:
        metric_transform_method = Raw()
    else:
        raise ValueError(f"Unsupported data_method metric: {metric_transform_method_name}")

    df = pd.read_csv(csv_name)

    # Replace seed 14 and 15 with 4 and 5 #TODO: handle this better
    assert not df[(df['model'] == '1B') & (df['seed'].isin([14, 15]))].any().any(), "There are rows where 'model' is 1B and 'seed' is 14 or 15"
    df['seed'] = df['seed'].replace({14: 4, 15: 5})

    assert all(len(d) == 1 for n,d in df.groupby(['model', 'group', 'task', 'step','seed']) ), f"There are duplicates in the data; max size per models X group X steps X seed X task was {max((len(d) for n, d in df.groupby(['model', 'group', 'task', 'step','seed'])))}"
    assert all(len(d) == 3 for n, d in df.groupby(['model', 'group', 'task', 'step'])['seed']), f"Not all models X group X steps X task have 3 seeds; min size was {min((len(d) for n, d in df.groupby(['model', 'group', 'task', 'step'])))}"
    assert all(d['seed'].nunique() ==3 for n, d in df.groupby(['model', 'group', 'task', 'step'])), f"Not all models X group X steps X task have 3 seeds; min size was  {min((d['seed'].nunique() for n, d in df.groupby(['model', 'group', 'task', 'step'])))}"
    df = preprocess_df(df, bad_mixes)
    df = df[groupby_cols + task_metrics]

    # Ensure no NaNs in groupby columns
    assert df[groupby_cols].isna().sum().sum() == 0, "NaNs found in groupby columns"

    # take the mean of any task_groups that have multiple tasks (task_groups with just one task will not be affected)
    df = groupby_mean(df, groupby_cols, task_metrics)

    # Task aggregation
    logging.info(f"available tasks: {df.task_group.unique()}")
    if task_aggregation == "olmes":
        # OLMES mean
        df = groupby_mean(df, groupby_cols[:-1], task_metrics)
    else:
        assert task_aggregation in df.task_group.unique(), f"Task aggregation {task_aggregation} not found in tasks"
        df = df[df['task_group'] == task_aggregation].reset_index(drop=True)
        df = df.drop(columns=['task_group'])
    assert df.isna().sum().sum() == 0, f"nans still present after aggregation"

    max_C, max_tokens = get_max_compute_and_tokens(df)
    available_seeds = list(df['seed'].unique())
    logging.info(f"Available seeds: {available_seeds}")
    assert all(len(d) == 1 for n,d in df.groupby(['model', 'group', 'step','seed']) ), f"There are duplicates in the data; max size per models X group X steps X seed X task was {max((len(d) for n, d in df.groupby(['model', 'group', 'step','seed'])))}"
    assert all(len(d) == 3 for n, d in df.groupby(['model', 'group', 'step'])['seed']), f"Not all models X group X steps X task have 3 seeds; min size was {min((len(d) for n, d in df.groupby(['model', 'group', 'step'])))}"
    assert all(d['seed'].nunique() == 3 for n, d in df.groupby(['model', 'group', 'step'])), f"Not all models X group X steps X task have 3 seeds; min size was {min((d['seed'].nunique() for n, d in df.groupby(['model', 'group', 'step'])))}"

    # Determine target pairs
    target_pairs_kept = target_filter.apply(df, primary_metric="primary_metric")
    save_data(target_pairs_kept, "0_target_pairs.json", out_dir)

    # Transform data for all computes
    primary_transformed_df = primary_transform_method(df[df["model"]==target_model], ["primary_metric"])
    metric_transformed_df = metric_transform_method(df, task_metrics)
    save_data(primary_transformed_df, "1_primary_transformed.csv", out_dir)
    primary_transformed_df = load_data("1_primary_transformed.csv", out_dir)
    save_data(metric_transformed_df, "1_metric_transformed.csv", out_dir)
    metric_transformed_df = load_data("1_metric_transformed.csv", out_dir)

    # Process predictions:
    evaluator_config = config['evaluator']

    # Use Compute by model scale experiments (each seed separately, which means that there are never abstentions and 2 class == 3 class)
    evaluator_model_scale = Evaluator(evaluator_config)
    results_model_scale = process_predictions(
        metric_transformed_df=metric_transformed_df,
        primary_transformed_df=take_mean_over_seeds(primary_transformed_df),
        target_pairs_kept=target_pairs_kept,
        evaluator=evaluator_model_scale,
        multi_seed=False,
        proportions=proportions,
        max_C=max_C,
        max_tokens=max_tokens,
        model_order=model_order,
        available_seeds=available_seeds,
        task_metrics=task_metrics,
        target_model=target_model,
        # compute_bin_width=compute_bin_width_as_proportion * max_C if compute_bin_width_as_proportion else None
    )
    df_model_scale = pd.DataFrame(results_model_scale)
    save_data(df_model_scale, "2_prediction_model_scale.csv", out_dir)

    # # Use Compute by training many models (all seeds are used together, which means that abstentions are possibly credited in 3 class)
    # evaluator_multi_seed = Evaluator(evaluator_config, multi_seed=True)
    # results_seeds = []
    # for i in [len(available_seeds)]:
    #     curr_seeds = available_seeds[:i]
    #     curr_result = process_predictions(
    #         metric_transformed_df=metric_transformed_df[metric_transformed_df["seed"].isin(curr_seeds)],
    #         primary_transformed_df=primary_transformed_df[primary_transformed_df["seed"].isin(curr_seeds)],
    #         target_pairs_kept=target_pairs_kept,
    #         evaluator=evaluator_multi_seed,
    #         multi_seed=True, # All seeds are used to determine winner
    #         proportions=proportions,
    #         max_C=max_C,
    #         max_tokens=max_tokens,
    #         model_order=model_order,
    #         available_seeds=available_seeds[:i],
    #         task_metrics=task_metrics,
    #         target_model=target_model,
    #         # compute_bin_width=compute_bin_width_as_proportion * max_C if compute_bin_width_as_proportion else None
    #     )
    #     for r in curr_result: r["seed_count"] = i
    #     results_seeds.extend(curr_result)
    # df_seeds = pd.DataFrame(results_seeds)
    # save_data(df_seeds, "2_prediction_seeds.csv", out_dir)

    df_model_scale = pd.read_csv(f"{out_dir}/2_prediction_model_scale.csv")
    # df_seeds = pd.read_csv(f"{out_dir}/2_prediction_seeds.csv")

    # Viz
    viz = Visualization(plot_dir=plots_dir)
    viz.plot_heatmap_model_scale(data=df_model_scale, metrics=task_metrics, value_col="binary_accuracy", save_name="heatmap_model_scale")
    # viz.plot_heatmap_model_scale(data=df_model_scale, metrics=task_metrics, value_col="magnitude_correlation", save_name="heatmap_model_scale_correlation")
    viz.plot_heatmap_model_scale(data=df_model_scale, metrics=task_metrics, value_col="pearson_correlation", save_name="heatmap_model_scale_pearson_correlation")
    # viz.plot_heatmap_model_scale(data=df_model_scale, metrics=task_metrics, value_col="recall_at_k", save_name="heatmap_model_scale_recall_at_k")
    viz.plot_heatmap_model_scale(data=df_model_scale, metrics=task_metrics, value_col="weighted_pearson_correlation", save_name="heatmap_model_scale_weighted_pearson_correlation")
    viz.plot_heatmap_model_scale(data=df_model_scale, metrics=task_metrics, value_col="NDCG", save_name="heatmap_model_scale_NDCG")
    # viz.plot_heatmap_multi_seed(data=df_seeds, metrics=task_metrics, value_col="binary_accuracy", save_name="heatmap_multi_seed")
    # viz.plot_heatmap_model_scale(data=df_seeds[df_seeds['seed_count'] == 3], metrics=task_metrics, value_col="three_way_accuracy", save_name="heatmap_model_scale_all_seeds_3_class", seed='all seeds')

def take_mean_over_seeds(df):
    """
    This works on 1_*_transformed.csv dfs like:
    model,group,seed,metric,models,compute_latest,token_latest,raw_values,value
    1B,DCLM-baseline,4,primary_metric,['1B'],0.0,0.0,[0.32872815549214207],0.32872815549214207
    """
    df = df.copy()
    df['raw_values'] = df['raw_values'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    df = df.groupby(['model', 'group', 'metric', 'models', 'compute_latest', 'token_latest']).agg({
        'value': 'mean',
        'raw_values': lambda x: list(pd.DataFrame(x.tolist()).mean())
    }).reset_index()
    df['seed'] = 'mean_seeds'
    return df

def get_max_compute_and_tokens(df):
    max_C = df['compute'].max()
    assert max_C == max([float(c) for c in df['compute'].unique()])
    argmax = df['compute'].argmax()
    max_steps = df.loc[argmax, 'step']
    max_model = df.loc[argmax, 'model']
    max_tokens = df.loc[argmax, 'tokens']
    logging.info(f"Max compute: {max_C} at step {max_steps} for model {max_model} with tokens {max_tokens}")

    max_tokens = df['tokens'].max()
    assert max_tokens == max([float(t) for t in df['tokens'].unique()])
    argmax = df['tokens'].argmax()
    max_tokens_max_C = df.loc[argmax, 'compute']
    assert max_tokens_max_C == max_C, f"max_tokens_max_C: {max_tokens_max_C} != max_C: {max_C}"
    return max_C, max_tokens

def preprocess_df(df, bad_mixes):
    df['tokens'] = df['tokens'] / 1e9
    df = df[df['task'].isin(RC_TASKS)]
    logging.info(f"Tasks: {len(df['task'].unique())} out of {len(RC_TASKS)} possible RC tasks")
    df = df[~df['group'].isin(bad_mixes)]
    df["metrics"] = df["metrics"].apply(lambda x: safe_eval(str(x)))
    df.loc[df['step'] == 0, 'compute'] = 0.0
    df.loc[df['step'] == 0, 'tokens'] = 0.0
    mixes = list(df["group"].unique())
    logging.info(f"Mixes: {len(mixes)}")

    # Group tasks, mmlu_* -> 'mmlu'
    df['task_group'] = df['task'].apply(lambda x: next((name for prefix, name in task_groups.items() if x.startswith(prefix)), x))
    logging.info(f"Identified task groups: {df['task_group'].unique()}")
    # Unpack dict column
    df = unpack_dict_column(df, 'metrics')
    num_nans = df.isnull().sum().sum()
    logging.info(f"setting nans to None; nans found: {num_nans}")
    df = df.where(df.notnull(), None)
    return df

def process_single_combination(
    proportion, seed, model, metric, compute_bin_width_as_proportion, target_model,
    target_pairs_kept, metric_transformed_df, primary_transformed_df, evaluator,
    max_C, max_tokens, multi_seed
):
    compute_bin_width = compute_bin_width_as_proportion * max_C if compute_bin_width_as_proportion else None
    mark_C = proportion * max_C #TODO proportion is the propotion from the config which is now the same thing as target proportion, unless this gets called with different values of max_C
    mark_tokens = proportion * max_tokens

    # these dfs are the same as 1_*_transformed.csv
    metric_seed_df = metric_transformed_df[
        ((metric_transformed_df['seed'] == seed) if not multi_seed else slice(None))
    ]

    primary_seed_df = primary_transformed_df # primary_transformed_df is already aggregated over seeds or this is multi_seed=True

    # prepare a null result in case we dont get any data
    failed_res = {
        'binary_accuracy': None,
        'three_way_accuracy': None,
        'correct_count': None,
        'incorrect_count': None,
        'abstain_count': None,
        'total_count': None,
        'primary_abstain': None,
        'metric': metric,
        'model': model,
        'seed': seed if not multi_seed else "all seeds",
        'compute': mark_C,
        'proportion': proportion,
        'tokens': mark_tokens,
        'proportion_target': round(mark_C / max_C, 2),
    }

    raw_results = []
    for mix1, mix2 in target_pairs_kept:
        metric_values1, metric_compute_latest1 = get_transformed_values_compute_and_seeds(
            metric_seed_df, mix1, model, metric, mark_C, compute_bin_width=compute_bin_width
        )
        metric_values2, metric_compute_latest2 = get_transformed_values_compute_and_seeds(
            metric_seed_df, mix2, model, metric, mark_C, compute_bin_width=compute_bin_width
        )
        # Get the latest C at target model scale
        primary_values1, primary__compute_latest1 = get_transformed_values_compute_and_seeds(
            primary_seed_df, mix1, target_model, "primary_metric", float("inf"), compute_bin_width=compute_bin_width
        )
        primary_values2, primary__compute_latest2 = get_transformed_values_compute_and_seeds(
            primary_seed_df, mix2, target_model, "primary_metric", float("inf"), compute_bin_width=compute_bin_width
        )
        

        if not metric_values1 or not metric_values2 or not primary_values1 or not primary_values2:
            # logging.warning(f"Failed to get complete data for {model} at {proportion} proportion (mark_c: {mark_C}). closest compute: {metric_compute_latest1} {metric_compute_latest2} {primary__compute_latest1} {primary__compute_latest2}.")
            return failed_res
        
        assert metric_compute_latest1 == metric_compute_latest2, f"metric_compute_latest1: {metric_compute_latest1} != metric_compute_latest2: {metric_compute_latest2}"
        assert primary__compute_latest1 == primary__compute_latest2, f"primary__compute_latest1: {primary__compute_latest1} != primary__compute_latest2: {primary__compute_latest2}"
        assert primary__compute_latest1 == max_C, f"primary__compute_latest1: {primary__compute_latest1} != max_C: {max_C}"

        raw_results.append({
            'mix1': mix1,
            'mix2': mix2,
            'primary_values1': primary_values1,
            'primary_values2': primary_values2,
            'metric_values1': metric_values1,
            'metric_values2': metric_values2,
        })

    res = evaluator.calculate_metrics(raw_results)
    res.update({
        'metric': metric,
        'model': model,
        'seed': seed if not multi_seed else "all seeds",
        'compute_limit': mark_C,
        'compute_latest': metric_compute_latest1,
        'proportion': proportion,
        'tokens': mark_tokens,
        # 'proportion_target': round(mark_C / max_C, 2),
    })
    return res

@timeit
def process_predictions(
    metric_transformed_df,
    primary_transformed_df,
    target_pairs_kept,
    evaluator,
    multi_seed,
    proportions,
    target_model,
    max_C,
    max_tokens,
    model_order,
    available_seeds,
    task_metrics
):
    results = []

    if multi_seed:
        seed_iter = [None]
    else:
        seed_iter = available_seeds
        assert primary_transformed_df['seed'].nunique() == 1, "for multi_seed=False, primary_metrics should already be aggregated to a single seed"

    if not proportions:
        proportion_per_model = {model: get_proportions_for_model(model, metric_transformed_df, max_C) for model in model_order}
        combinations = [
            (proportion, seed, model, metric, proportion - last_proportion)
            for seed in seed_iter
            for model in model_order
            for metric in task_metrics
            for last_proportion, proportion in zip([0.0]+ proportion_per_model[model][:-1], proportion_per_model[model])
        ]
    else:
        # Generate all combinations of inputs
        combinations = [
            (proportion, seed, model, metric, proportion - last_proportion)
            for last_proportion, proportion in zip([0.0]+ proportions[:-1], proportions)
            for seed in seed_iter
            for model in model_order
            for metric in task_metrics
        ]

    # Call multiprocess pool
    with ProcessPoolExecutor() as executor:
        process_func = partial(
            process_single_combination,
            target_pairs_kept=target_pairs_kept,
            metric_transformed_df=metric_transformed_df,
            primary_transformed_df=primary_transformed_df,
            evaluator=evaluator,
            max_C=max_C,
            max_tokens=max_tokens,
            multi_seed=multi_seed,
            target_model=target_model
        )
        # Wrap in tqdm
        for res in tqdm(executor.map(process_func, *zip(*combinations)), total=len(combinations)):
            results.append(res)

    return results

def get_proportions_for_model(model, metric_transformed_df, max_C, eps=1e-6):
    # Get the compute values for the model
    model_df = metric_transformed_df[metric_transformed_df['model'] == model]
    compute_latests = model_df['compute_latest'].unique()
    compute_latests.sort()
    # Get the proportion of compute for each compute value
    proportions = [(c / max_C) + eps for c in compute_latests]
    print(f"Proportions for {model}: {proportions}")
    print(f"with compute_latests: {compute_latests.tolist()}")
    return proportions

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Process experiment results")
    parser.add_argument('--config', type=str, default="config.yaml", help="YAML config file")
    args = parser.parse_args()
    main(args.config)
