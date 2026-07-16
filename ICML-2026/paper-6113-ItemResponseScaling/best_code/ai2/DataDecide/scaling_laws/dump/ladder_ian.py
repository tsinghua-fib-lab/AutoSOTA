import os
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

from scaling.utils import FinalConfig

from fitting.step1 import fit_step1, predict_step1, plot_step1, str_chinchilla_n_d_fit
from fitting.step2 import fit_step2, predict_step2, plot_step2
from fitting.predict import predict_chained, plot_chained, str_chained_fit

from fitting.step1_flops import fit_step1 as fit_step1_flops, predict_step1 as predict_step1_flops, plot_step1 as plot_step1_flops, str_chinchilla_flops_fit
from fitting.predict_flops import predict_chained_flops, plot_chained as plot_chained_flops, str_chained_fit as str_chained_fit_flops
from fitting.single_step import fit_single_step, predict_single_step, plot_single_step, str_combined_fit

from concurrent.futures import ProcessPoolExecutor

from scipy.interpolate import UnivariateSpline

FONTSIZE = 8

TASK_KEY_MAP = {
    "arc_challenge": "arc_challenge_test_5shot",
    "arc_easy": "arc_easy_test_5shot",
    "boolq": "boolq_val_5shot",
    "socialiqa": "socialiqa_val_5shot",
    "csqa": "csqa_val_5shot",
    "hellaswag": "hellaswag_val_5shot",
    "openbookqa": "openbookqa_test_5shot",
    "winogrande": "winogrande_val_5shot",
    "piqa": "piqa_val_5shot",
}

SIZE_COLORS = {
    '4M': 'brown',
    '20M': 'black',
    '60M': 'teal',
    '90M': 'pink',
    '150M': '#1f77b4',
    '300M': '#2ca02c',
    '530M': '#ff7f0e',
    '750M': '#d62728',
    '1B': '#9467bd'
}

FULL_SCHEDULE = {
    '4M': 5725,
    '20M': 14584,
    '60M': 29042,
    '90M': 29901,
    '150M': 38157,
    '300M': 45787,
    '530M': 57786,
    '750M': 63589,
    '1B': 69369,
}

MODEL_TO_BATCH = {
    '4M': 32,
    '6M': 32,
    '8M': 32,
    '10M': 32,
    '14M': 32,
    '16M': 32,
    '20M': 64,
    '60M': 96,
    '90M': 160,
    '150M': 192,
    '300M': 320,
    '530M': 448,
    '750M': 576,
    '1B': 704
}

MODEL_TO_PARAMETERS = {
    '4M': 3744832,
    '6M': 6010464,
    '8M': 8538240,
    '10M': 9900432,
    '12M': 12066600,
    '14M': 14380224,
    '16M': 16004560,
    '20M': 19101888,
    '60M': 57078144,
    '90M': 97946640,
    '150M': 151898880,
    '300M': 319980544,
    '530M': 530074944,
    '750M': 681297408,
    '1B': 1176832000
}


def get_compute(scale):
    return 2048 * 6 * MODEL_TO_BATCH[scale] * MODEL_TO_PARAMETERS[scale] * FULL_SCHEDULE[scale]


def compute_2_class(ranking_a, ranking_b):
    """ Compute 2-class accuracy """
    ranking_a = list(ranking_a)
    ranking_b = list(ranking_b)

    assert len(ranking_b) == len(ranking_b)
    
    n = len(ranking_a)
    same_order_count = 0
    total_pairs = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            i_pred = ranking_b.index(ranking_a[i])
            j_pred = ranking_b.index(ranking_a[j])
            
            if (i < j and i_pred < j_pred) or (i > j and i_pred > j_pred):
                same_order_count += 1
            total_pairs += 1
    
    return same_order_count / total_pairs if total_pairs > 0 else 0.0


def add_ladder_data_cheap_decisions(data_by_name):
    """ From Ian """
    sequence_length = 2048

    def model_and_step_to_tokens(model, step):
        return MODEL_TO_BATCH[model] * step * sequence_length

    def model_and_step_to_compute(model, step):
        return MODEL_TO_PARAMETERS[model] * model_and_step_to_tokens(model, step) * 6
    
    for k, v in data_by_name.items():
        step = v['step'][-1]
        c = model_and_step_to_compute(k, step)
        n = MODEL_TO_PARAMETERS[k]
        d = model_and_step_to_tokens(k, step)
        f = float(n * d * 6)
        data_by_name[k]['ns'] = [n]
        data_by_name[k]['fs'] = [f]
        data_by_name[k]["ds"] = [d]

    # raise RuntimeError(data_by_name)

    return data_by_name


def get_slice(df, model, task):
    try:
        df = df.loc[(task, model)]
    except KeyError:
        return df.iloc[0:0]
    df = df.reset_index()
    return df


def get_ladder_data(df, task_name, x_metric, y_metric, train_models, eval_models, step='max'):
    data_by_name = defaultdict(dict)

    for model in train_models + eval_models:
        if model in eval_models:
            mode = 'eval'
        elif model in train_models:
            mode = 'train'

        split_name = model.split('-')
        size, chinchilla = split_name[-2:]
        group = '-'.join(split_name[:-2])

        is_multiindex = isinstance(df.index, pd.MultiIndex)

        if is_multiindex:
            _slice = get_slice(df, model, task_name)
        else:
            # Filter the DataFrame based on the criteria
            _slice = df[
                (df['task'] == task_name) &
                (df['model'] == model)
                # (df['group'] == group) &
                # (df['size'] == size) &
                # (df['chinchilla'] == chinchilla)
            ]

        if len(_slice) == 0:
            raise RuntimeError(f'Got empty slice for {(task_name, group, size, chinchilla)}.')
        
        if step == 'max':
            # Find the entry with the largest value of 'step'
            max_step_entry = _slice.loc[_slice['step'].idxmax()]
            x_val = max_step_entry[x_metric].tolist()
            y_val = max_step_entry[y_metric].tolist()
            step_val = _slice['step'].max().tolist()

            # Remove duplicates
            if isinstance(x_val, list):
                assert len(np.unique(x_val)) == 1
                x_val = x_val[0]
            if isinstance(y_val, list):
                assert len(np.unique(y_val)) == 1
                y_val = y_val[0]
            if isinstance(step_val, list):
                assert len(np.unique(step_val)) == 1
                step_val = step_val[0]

            x_val = [x_val]
            y_val = [y_val]
            step_val = [step_val]
        else:
            _slice = _slice.sort_values(by='step', ascending=True)
            x_val = _slice[x_metric].tolist()
            y_val = _slice[y_metric].tolist()
            step_val = _slice['step'].tolist()

            x_val = [x_val]
            y_val = [y_val]
            step_val = [step_val]
        
        if 'xs' not in data_by_name[size]: data_by_name[size]['xs'] = []
        if 'ys' not in data_by_name[size]: data_by_name[size]['ys'] = []
        if 'step' not in data_by_name[size]: data_by_name[size]['step'] = []

        data_by_name[size]['step'] += step_val
        data_by_name[size]['xs'] += x_val
        data_by_name[size]['ys'] += y_val
        data_by_name[size]['mode'] = mode

    return data_by_name


def create_ladder_config(task_name, train_models, eval_models, color=None):
    # arc_easy:enlarge => arc_easy
    task_root = task_name.split(':')[0] if isinstance(task_name, str) else None

    # Create config
    configs = {}
    for model in train_models + eval_models:
        size = model.split('-')[-2]
        if color == None: color = SIZE_COLORS.get(size, 'k')
        mode = 'eval' if model in eval_models else 'train'
        
        # Create dummy config for new eval points
        configs[size] = FinalConfig(
            paths=None, mode=mode, n=0, label=size, color=color, use_last_n_percentage=100
        )

    task_key = TASK_KEY_MAP.get(task_root, None) # the task key is used to get min/max perf and plot title

    return task_key, configs


def run_ladder(
        df, task_name, train_models, eval_models, x_metric, y_metric,
        use_flops=False, use_single_step=False, use_two_param=False, use_helper_points=False, 
        last_perc_step_2=0.9,
        run_step1=True, run_step2=True, run_stacked=True,
        axes=None, add_texts=False, color=None, extrapolate_ratio=[0.8, 1.5], # plotting
        return_preds=False, return_reals=False, return_coeff=False):
    abs_error_step_1, abs_error_step_2, abs_error_stacked = None, None, None
    step_1_y_pred, step_2_y_pred, stacked_y_pred = None, None, None
    step_1_y, step_2_y, stacked_y = None, None, None

    # Get config
    task_key, configs = create_ladder_config(task_name, train_models, eval_models, color=color)

    # Get data
    # data_by_name = get_ladder_data(df, task_name, x_metric, y_metric, train_models, eval_models)

    # Use avg of final 10% of checkpoints to fit step 1
    data_by_name = get_ladder_data(df, task_name, x_metric, y_metric, train_models, eval_models, step='all')
    for k1, v1 in data_by_name.items():
        for k2, v2 in v1.items():
            if k2 == 'step': 
                # grab final step
                data_by_name[k1][k2] = [data_by_name[k1][k2][0][-1]]
                continue
            if k2 == 'mode': continue
            if isinstance(v2, list):
                import math
                data_by_name[k1][k2] = [np.mean(v2[0][math.ceil(0.9 * len(v2[0])):])]

    # which functional form to use for step 1 prediction
    if 'byte' in x_metric:
        y_metric_func = 'rc_bpb'
    else:
        y_metric_func = 'rc_acc'

    assert len(data_by_name) != 0, train_models
    if use_two_param:
        assert use_flops, 'we only have a 2 param function for flops version'

    if use_single_step:
        assert not run_stacked and not run_step2, 'Single step prediction will only run step 1!'

    ax_i = 0
    if run_step1 or run_stacked:
        # Add token data
        data_by_name = add_ladder_data_cheap_decisions(data_by_name)
        
        # Fit step 1
        if use_single_step:
            step1_coefficients = fit_single_step(data_by_name, y_metric_func, use_flops=use_flops)
        elif use_flops:
            step1_coefficients, cov = fit_step1_flops(data_by_name, y_metric_func, use_two_param=use_two_param)
        else:
            step1_coefficients, cov = fit_step1(data_by_name, y_metric_func)

        if use_single_step:
            (
                predicted_data_by_name, plotted_predicted_data,
                (step_1_y, step_1_y_pred, rel_error_step_1),
            ) = predict_single_step(
                # configs, data_by_name, step1_coefficients, y_metric=y_metric_func, 
                data_by_name, step1_coefficients, use_flops=use_flops
            )
        elif use_flops:
            (
                predicted_data_by_name, plotted_predicted_data,
                (step_1_y, step_1_y_pred, rel_error_step_1), all_rel_errors,
            ) = predict_step1_flops(
                configs, data_by_name, step1_coefficients, y_metric=y_metric_func, use_two_param=use_two_param
            )
        else:
            (
                predicted_data_by_name, plotted_predicted_data,
                (step_1_y, step_1_y_pred, rel_error_step_1), all_rel_errors,
            ) = predict_step1(
                configs, data_by_name, step1_coefficients, y_metric=y_metric_func, 
            )
        abs_error_step_1 = abs(step_1_y_pred - step_1_y)

        # Plot step 1
        if axes is not None and run_step1:
            ax = axes[ax_i]
            ax_i += 1
            if use_single_step:
                plot_single_step(
                    # configs, data_by_name, predicted_data_by_name, plotted_predicted_data,
                    # task_name, str_combined_fit(step1_coefficients), y_metric_func,
                    # step1_coefficients, cov, ax,
                    configs, data_by_name, predicted_data_by_name, plotted_predicted_data,
                    task_name, str_combined_fit(step1_coefficients), use_flops, ax,
                )
            elif use_flops:
                plot_step1_flops(
                    configs, data_by_name, predicted_data_by_name, plotted_predicted_data,
                    task_name, str_chinchilla_flops_fit(step1_coefficients), y_metric_func,
                    step1_coefficients, cov, ax,
                )
            else:
                plot_step1(
                    configs, data_by_name, predicted_data_by_name, plotted_predicted_data,
                    task_name, str_chinchilla_n_d_fit(step1_coefficients), y_metric_func,
                    step1_coefficients, cov, ax,
                )
            ax.set_ylabel(x_metric)

    # Use intermediate checkpoints to fit step 2
    data_by_name = get_ladder_data(df, task_name, x_metric, y_metric, train_models, eval_models, step='all')
    for k1, v1 in data_by_name.items():
        for k2, v2 in v1.items():
            if isinstance(v2, list):
                import math
                # use last 90% of checkpoints
                data_by_name[k1][k2] = v2[0][math.ceil((1-last_perc_step_2) * len(v2[0])):]
    
    # data_by_name = {k: v for k, v in data_by_name.items() if '150M' not in k and '300M' not in k and '530M' not in k}

    if run_step2 or run_stacked:
        task_key, configs = create_ladder_config(task_name, train_models, eval_models, color=color)

        _min, _max = None, None
        if task_key is None and use_helper_points:
            _min, _max = 0, 1 # TODO: Use utils.constants_task to get correct values

        # Fit step 2
        step2_coefficients, cov = fit_step2(data_by_name, task_key, y_metric=y_metric_func, _min=_min, _max=_max, use_log_sigmoid=False, use_helper_points=use_helper_points)

        (
            predicted_data_by_name, plotted_predicted_data,
            (step_2_y, step_2_y_pred, rel_error_step_2, delta_error), all_rel_errors,
        ) = predict_step2(
            configs, data_by_name, step2_coefficients, cov, y_metric=y_metric_func, use_log_sigmoid=False
        )
        abs_error_step_2 = abs(step_2_y_pred - step_2_y)

        # Plot step 2
        if axes is not None and run_step2:
            ax = axes[ax_i]
            ax_i += 1
            plot_step2(
                configs, data_by_name, predicted_data_by_name, plotted_predicted_data, task_key, None, y_metric_func, 'rc_acc',
                step2_coefficients, cov, use_log_sigmoid=False, add_texts=add_texts, ax=ax
            )
            ax.set_xlabel(x_metric)
            ax.set_ylabel(y_metric)
        
    # Get step 1 data again (necessary if running with intermediate checkpoints)
    data_by_name = get_ladder_data(df, task_name, x_metric, y_metric, train_models, eval_models)
    data_by_name = add_ladder_data_cheap_decisions(data_by_name)
    
    if run_stacked:
        # Predict stacked
        if use_flops:
            (
                predicted_data_by_name, plotted_predicted_data_by_name, 
                (stacked_y, stacked_y_pred, rel_error_stacked)
            ) = predict_chained_flops(
                data_by_name, step1_coefficients, step2_coefficients, 
                use_two_param=use_two_param, y_metric=y_metric_func,
                extrapolate_ratio=extrapolate_ratio
            )
        else:
            (
                predicted_data_by_name, plotted_predicted_data_by_name, 
                (stacked_y, stacked_y_pred, rel_error_stacked)
            ) = predict_chained(
                data_by_name, step1_coefficients, step2_coefficients, y_metric=y_metric_func, use_log_sigmoid=False
            )
        abs_error_stacked = abs(stacked_y_pred - stacked_y)

        # For stacked predictions, the x axis is now the y axis
        for key in data_by_name:
            data_by_name[key]['xs'] = data_by_name[key]['ys']

        # Plot stacked prediction
        if axes is not None:
            ax = axes[ax_i]
            if use_flops:
                plot_chained_flops(
                    configs,
                    data_by_name,
                    predicted_data_by_name,
                    plotted_predicted_data_by_name,
                    task_name,
                    str_chained_fit_flops(step1_coefficients, step2_coefficients),
                    ax,
                )
            else:
                plot_chained(
                    configs,
                    data_by_name,
                    predicted_data_by_name,
                    plotted_predicted_data_by_name,
                    task_name,
                    str_chained_fit(step1_coefficients, step2_coefficients, use_log_sigmoid=False),
                    ax,
                )
            ax.legend(loc='upper left')
            ax.set_ylabel(y_metric)

    if axes is not None:
        for ax in axes:
            ax.set_title(task_name)
            ax.legend(fontsize=8)

    if return_coeff:
        return (step1_coefficients, step2_coefficients), (abs_error_step_1, abs_error_step_2, abs_error_stacked), (step_1_y_pred, step_2_y_pred, stacked_y_pred)
    if return_reals:
        return (abs_error_step_1, abs_error_step_2, abs_error_stacked), (step_1_y, step_2_y, stacked_y), (step_1_y_pred, step_2_y_pred, stacked_y_pred)
    if return_preds:
        return (abs_error_step_1, abs_error_step_2, abs_error_stacked), (step_1_y_pred, step_2_y_pred, stacked_y_pred)
    return abs_error_step_1, abs_error_step_2, abs_error_stacked


def compute_intersection(p1s1, p1s2, p2s1, p2s2, x_range):
    """ Compute the intersection between two scaling law curves """
    from scipy.optimize import root_scalar
    from scaling.fitting_functions import sigmoid, chinchilla_flops_fit

    def diff(x):
        y1 = sigmoid(chinchilla_flops_fit(x, p1s1), *p1s2)
        y2 = sigmoid(chinchilla_flops_fit(x, p2s1), *p2s2)
        return y1 - y2
    
    # Split the range into multiple segments to check for multiple intersections
    num_segments = 100
    x_segments = np.logspace(np.log10(x_range[0] + 1e-10), np.log10(x_range[1]), num_segments+1)
    intersections = []
    
    for i in range(num_segments):
        segment_range = (x_segments[i], x_segments[i+1])
        f_a = diff(segment_range[0])
        f_b = diff(segment_range[1])
        
        if np.sign(f_a) != np.sign(f_b):
            try:
                result = root_scalar(diff, bracket=segment_range, method='brentq')
                if result.converged:
                    intersections.append(result.root)
            except ValueError:
                continue
    
    if not intersections:
        raise ValueError(f"No intersections found in the specified range.")
    
    return max(intersections)  # Return the last (rightmost) intersection


def pairwise_intersections(coeffs, x_range):
    """ Compute pairwise intersection between N scaling law fits """
    n = len(coeffs)
    intersections = np.zeros((n, n))

    x_range = (float(x_range[0]), float(x_range[1])) # convert range to floats
    
    for i in range(n):
        for j in range(n):
            if i == j:
                intersections[i, j] = 0
                continue
            
            p1s1, p1s2 = coeffs[i][0], coeffs[i][1]
            p2s1, p2s2 = coeffs[j][0], coeffs[j][1]
            
            try:
                x_intersection = compute_intersection(p1s1, p1s2, p2s1, p2s2, x_range)
                intersections[i, j] = x_intersection
            except ValueError:
                intersections[i, j] = 0
    
    return intersections


def get_perf(coeffs, C):
    """ Get the performance of a scaling law fit for some compute C """
    from scaling.fitting_functions import sigmoid, chinchilla_flops_fit

    n = len(coeffs)
    perf = np.zeros((n))
    
    for i in range(n):
        p1s1, p1s2 = coeffs[i][0], coeffs[i][1]
        
        try:
            y1 = sigmoid(chinchilla_flops_fit(C, p1s1), *p1s2)
        except ValueError:
            perf = 0

        perf[i] = y1
    
    return perf


def process_mix(mix, df_multi_index, all_models, all_tasks, setup, x_metric, y_metric):
    warnings.filterwarnings("ignore", category=RuntimeWarning) # supress function fitting warnings

    results = {
        'mix': mix,
        'fitting_results_step_1': {},
        'fitting_results_step_2': {},
        'fitting_results_stacked': {},
        'step_1_y': {},
        'step_2_y': {},
        'stacked_y': {},
        'step_1_y_preds': {},
        'step_2_y_preds': {},
        'stacked_y_preds': {}
    }
    
    models = [model for model in all_models if '-'.join(model.split('-')[:-2]) == mix]
    
    import re

    for task in all_tasks:
        # Parse all `no_*M` from setup and use these to filter
        exclusions = re.findall(r'no_\d+M', setup)
        train_models = [m for m in models if '1B' not in m and all(excl[3:] not in m for excl in exclusions)]
        eval_models = [m for m in models if '1B' in m and all(excl[3:] not in m for excl in exclusions)]

        assert len(train_models) != 0 and len(eval_models) != 0, f'{mix}: ({train_models}, {eval_models}) {models}'

        use_helper_points = 'helper_points' in setup
        use_single_step   = '1_step' in setup
        use_flops         = '3_param' in setup or '2_param' in setup
        use_two_param     = '2_param' in setup
        use_intermediate_feature = 'intermediate' in setup

        last_perc_step_2 = 0.9
        if 'step2=0.5' in setup:
            last_perc_step_2 = 0.5

        run_step1, run_step2, run_stacked = True, True, True
        if use_single_step:
            # Only run 1 step, and have the 1 step be the downstream metric
            run_step2, run_stacked = False, False
            x_metric = y_metric

        if use_intermediate_feature:
            # Predict FLOPs -> [metric] -> primary_metric
            x_metric = y_metric
            y_metric = 'primary_metric'
            assert use_single_step == False, 'Must be 2-step prediction!'

        try:
            (abs_error_step_1, abs_error_step_2, abs_error_step_stacked), \
                (step_1_y, step_2_y, stacked_y), \
                (step_1_y_pred, step_2_y_pred, stacked_y_pred) = run_ladder(
                df_multi_index,
                task_name=task,
                train_models=train_models,
                eval_models=eval_models,
                use_helper_points=use_helper_points,
                last_perc_step_2=last_perc_step_2,
                x_metric=x_metric,
                y_metric=y_metric,
                use_flops=use_flops,
                use_single_step=use_single_step,
                use_two_param=use_two_param,
                return_reals=True,
                run_step1=run_step1, run_step2=run_step2, run_stacked=run_stacked
            )
        except Exception as e:
            abs_error_step_1 = abs_error_step_2 = abs_error_step_stacked = float('inf')
            step_1_y = step_2_y = stacked_y = float('inf')
            step_1_y_pred = step_2_y_pred = stacked_y_pred = float('inf')
            if task != 'winograde' and task != 'boolq':
                # raise RuntimeError(f'{task}, {setup}: {e}')
                # We expect some of Winograde and BoolQ to fail due to missing data
                print(f'Failed to fit ({setup, mix, task, x_metric, y_metric}): {e}')
                # raise e

        results['fitting_results_step_1'][task] = abs_error_step_1
        results['fitting_results_step_2'][task] = abs_error_step_2
        results['fitting_results_stacked'][task] = abs_error_step_stacked
        results['step_1_y'][task] = step_1_y
        results['step_2_y'][task] = step_2_y
        results['stacked_y'][task] = stacked_y
        results['step_1_y_preds'][task] = step_1_y_pred
        results['step_2_y_preds'][task] = step_2_y_pred
        results['stacked_y_preds'][task] = stacked_y_pred

    return results


def fit_all_mixes(df, all_models, mixes, tasks, y_metrics, setups, x_metric='correct_logit_per_byte', quiet=True):
    all_predictions = []

    df_multi_index = df.set_index(['task', 'model']).sort_index()

    # Use ProcessPoolExecutor for CPU-intensive mix processing
    cpus = os.cpu_count()
    results = []
    with ProcessPoolExecutor(max_workers=cpus) as process_executor:
        total_jobs = len(setups)*len(mixes)

        # Submit all futures upfront
        futures = []
        future_info = {}
        for y_metric in tqdm(y_metrics, desc=f"Submitting {total_jobs} fitting jobs on {cpus} CPUs", total=len(y_metrics)):
            for setup in setups:
                for mix in mixes:
                    future = process_executor.submit(process_mix, mix, df_multi_index, all_models, tasks, setup, x_metric, y_metric)
                    futures.append(future)
                    future_info[future] = (mix, y_metric, setup)

            try:
                for future in tqdm(futures, desc=f"Processing results for {y_metric}", total=total_jobs):
                    result = future.result()
                    mix, y_metric, setup = future_info[future]
                    results.append((result, mix, y_metric, setup))
                    future.done()  # Ensure all futures are complete
                    del future  # Immediately remove future
            finally:
                for future in futures:
                    future.cancel()  # Cancel any remaining futures
                del futures[:]  # Free memory
        process_executor.shutdown(wait=True)  # Shutdown the executor and wait for cleanup

    print(f'Done processing jobs!')

    # Process results as they complete
    for result, mix, y_metric, setup in tqdm(results, desc='Processing all predictions', total=len(results), disable=quiet):
        fitting_results_step_1 = pd.DataFrame(index=[], columns=tasks)
        fitting_results_step_2 = pd.DataFrame(index=[], columns=tasks)
        fitting_results_stacked = pd.DataFrame(index=[], columns=tasks)
        step_1_y = pd.DataFrame(index=[], columns=tasks)
        step_2_y = pd.DataFrame(index=[], columns=tasks)
        stacked_y = pd.DataFrame(index=[], columns=tasks)
        step_1_y_preds = pd.DataFrame(index=[], columns=tasks)
        step_2_y_preds = pd.DataFrame(index=[], columns=tasks)
        stacked_y_preds = pd.DataFrame(index=[], columns=tasks)

        for task in tasks:
            fitting_results_step_1.loc[mix, task] = result['fitting_results_step_1'].get(task, float('inf'))
            fitting_results_step_2.loc[mix, task] = result['fitting_results_step_2'].get(task, float('inf'))
            fitting_results_stacked.loc[mix, task] = result['fitting_results_stacked'].get(task, float('inf'))
            step_1_y.loc[mix, task] = result['step_1_y'].get(task, float('inf'))
            step_2_y.loc[mix, task] = result['step_2_y'].get(task, float('inf'))
            stacked_y.loc[mix, task] = result['stacked_y'].get(task, float('inf'))
            step_1_y_preds.loc[mix, task] = result['step_1_y_preds'].get(task, float('inf'))
            step_2_y_preds.loc[mix, task] = result['step_2_y_preds'].get(task, float('inf'))
            stacked_y_preds.loc[mix, task] = result['stacked_y_preds'].get(task, float('inf'))

        def process_dataframe(df, calculate_abs=False):
            pd.set_option('future.no_silent_downcasting', True)
            df = df.fillna(value=np.nan)
            df['avg'] = df.mean(axis=1, skipna=True)
            if calculate_abs:
                df = df.abs()
            return df.sort_values(by='avg', ascending=False)

        # Process DataFrames sequentially
        fitting_results_step_1 = process_dataframe(fitting_results_step_1, True)
        fitting_results_step_2 = process_dataframe(fitting_results_step_2, True)
        fitting_results_stacked = process_dataframe(fitting_results_stacked, True)
        step_1_y = process_dataframe(step_1_y)
        step_2_y = process_dataframe(step_2_y)
        stacked_y = process_dataframe(stacked_y)
        step_1_y_preds = process_dataframe(step_1_y_preds)
        step_2_y_preds = process_dataframe(step_2_y_preds)
        stacked_y_preds = process_dataframe(stacked_y_preds)

        if not quiet:
            print('Absolute unsigned error for predicting 1B-5xC (stacked):')
            # display(fitting_results_stacked.map(lambda x: f'{round(x * 100, 1)}%'))
            print('Predicted performance for 1B-5xC on all mixes:')
            # display(stacked_y_preds.map(lambda x: f'{round(x * 100, 1)}%'))

        (step_1_abs_error, step_2_abs_error, stacked_abs_error), \
            (step_1_y_preds, step_2_y_preds, stacked_y_preds), \
            (step_1_y, step_2_y, stacked_y) = \
            (fitting_results_step_1, fitting_results_step_2, fitting_results_stacked), \
            (step_1_y_preds, step_2_y_preds, stacked_y_preds), \
            (step_1_y, step_2_y, stacked_y)
        
        all_predictions += [(
            x, y, y_metric, setup,
            step_1_y.loc[y, x], step_2_y.loc[y, x], stacked_y.loc[y, x],
            step_1_y_preds.loc[y, x], step_2_y_preds.loc[y, x], stacked_y_preds.loc[y, x], 
            step_1_abs_error.loc[y, x], step_2_abs_error.loc[y, x], stacked_abs_error.loc[y, x],
        ) for x in stacked_y_preds.columns for y in stacked_y_preds.index]

    results = pd.DataFrame(all_predictions, columns=[
        'task', 'mix', 'metric', 'setup', 
        'step_1_y', 'step_2_y', 'stacked_y',
        'step_1_pred', 'step_2_pred', 'stacked_pred', 
        'abs_error_step_1', 'abs_error_step_2', 'abs_error_stacked'
    ])

    # For single step setups, only the step_1 pred is reported. Copy over the values
    for col_base in ['y', 'pred', 'abs_error']:
        if col_base in ['y', 'pred']:
            step_1_col = f'step_1_{col_base}'
            step_2_col = f'step_2_{col_base}'
            stacked_col = f'stacked_{col_base}'
        if col_base in ['abs_error']:
            step_1_col = f'{col_base}_step_1'
            step_2_col = f'{col_base}_step_2'
            stacked_col = f'{col_base}_stacked'
        
        # Where step_1 exists but step_2 and stacked are NaN, copy the values
        mask = results[step_1_col].notna() & results[step_2_col].isna() & results[stacked_col].isna()
        results.loc[mask, step_2_col] = results.loc[mask, step_1_col]
        results.loc[mask, stacked_col] = results.loc[mask, step_1_col]

    # Re-compute rel error
    results['rel_error_stacked'] = results['abs_error_stacked'] / results['stacked_pred']

    # remove "avg" task before returning
    results = results[results['task'] != "avg"]

    return results


def render_result_table(results, index, agg_col='setup', only_use_default_scaling_law=False, raw_values=False):
    """ Convert a df of results to LaTeX """
    from utils.constants_ian import SETUP_NAME_LATEX

    if index != 'metric':
        filtered_results = results[results['metric'] == 'primary_metric']
    else:
        filtered_results = results

    # Select numeric columns for aggregation
    agg_cols = ['abs_error_stacked', 'rel_error_stacked', 'stacked_y']

    # Compute averages for each unique value in 'setup'
    average_by_setup = filtered_results.groupby([index, agg_col])[agg_cols].mean().reset_index()

    # Pivot the table to have 'setup' as columns for comparison
    pivoted_results = average_by_setup.pivot(index=index, columns=agg_col, values=agg_cols).reset_index()

    # Flatten the multi-index columns
    pivoted_results.columns = ['_'.join(col).strip('_') for col in pivoted_results.columns.values]

    # Format percentage columns
    for col in pivoted_results.columns:
        if 'abs_error_stacked' in col or 'rel_error_stacked' in col:
            pivoted_results[col] = (pivoted_results[col] * 100).round(2)

    # Calculate averages for each column and append as a new row
    avg_row = {col: pivoted_results[col].mean() if 'abs_error_stacked' in col or 'rel_error_stacked' in col else 'Average' for col in pivoted_results.columns}
    pivoted_results = pd.concat([pivoted_results, pd.DataFrame([avg_row])], ignore_index=True)

    # Display results
    pivoted_results = pivoted_results.set_index(index)

    # Multiply OLMES Avg col by 100
    for col in pivoted_results.columns:
        if 'stacked_y_' in col and index != 'metric':
            pivoted_results[col] = pivoted_results[col].apply(lambda x: x * 100 if isinstance(x, float) else x)
    
    if not raw_values:
        pivoted_results = pivoted_results.map(lambda x: (f"{x:.2f}" if isinstance(x, float) else x) + "%")

    if only_use_default_scaling_law:
        pivoted_results = pivoted_results[[col for col in pivoted_results.columns if col.endswith("3_param-default") or col.endswith("3_param-no_750M") or col.endswith("3_param-no_750M_no_530M")]]
        pivoted_results = pivoted_results.sort_values('abs_error_stacked_3_param-default')
        pivoted_results.rename(columns=lambda col: col.replace("stacked_y_", "OLMES Avg. ").replace("abs_error_stacked_", "Abs Error ").replace("rel_error_stacked_", "Rel Error ").replace("3_param-", "").replace("no_", "-").replace("_", " "), inplace=True)
        pivoted_results = pd.concat([pivoted_results.loc[pivoted_results.index != "Average"], pivoted_results.loc[["Average"]]])

    if index == 'setup':
        # Create a mapping from index values to their order in SETUP_NAME_LATEX
        setup_order = {k: i for i, k in enumerate(SETUP_NAME_LATEX.keys())}
        
        # Sort the index based on the order in SETUP_NAME_LATEX
        pivoted_results = pivoted_results.reindex(sorted(pivoted_results.index, 
                                                       key=lambda x: setup_order.get(x, float('inf')) if x != 'Average' else float('inf')))

    cols_to_drop = [col for col in pivoted_results.columns if "OLMES Avg." in col and 'default' not in col]
    pivoted_results = pivoted_results.drop(columns=cols_to_drop)

    if index == 'metric':
        cols_to_drop = [col for col in pivoted_results.columns if "Rel Error" in col]
        pivoted_results = pivoted_results.drop(columns=cols_to_drop)
    
    return pivoted_results


def fix_table_rendering(table):
    """ Fix formatting issues with pandas table formatter """
    from utils.constants_ian import DATA_NAME_LATEX, SETUP_NAME_LATEX, TASK_NAME_LATEX

    lines = table.split('\n')
    # Process each line of the table
    for i, line in enumerate(lines):
        # if 'Task' in table: continue
        if not line or line.startswith(' ') or line.startswith('\\'): continue
        if 'Recipe' in line: continue
        
        # Map the data mix name to the latex name
        for key, value in DATA_NAME_LATEX.items():
            if line.split(' ')[0] == key: 
                lines[i] = lines[i].replace(key, value)
        
        # Map the data mix name to the latex name
        for key, value in SETUP_NAME_LATEX.items():
            if line.split(' ')[0] == key: 
                lines[i] = lines[i].replace(key, value)
        
        # Map the data mix name to the latex name
        for key, value in TASK_NAME_LATEX.items():
            if line.split(' ')[0] == key: 
                lines[i] = lines[i].replace(key, value)

    # Add midrule before Average row
    for i, line in enumerate(lines):
        if line.strip().startswith('Average'):
            lines.insert(i, '\\midrule')
            break
    
    # Rejoin and print
    table_str = '\n'.join(lines).replace('%', '\%')

    if 'Scaling Law Functional Form' in table:
        table_str = table_str.replace('\n    ', '\n\\hspace{1em}\\hspace{1em}').replace('\n  ', '\n\\hspace{1em}')

    return table_str

def plot_task_accuracy(ax, two_class_results, task, sizes, show_legend=False, size_colors=SIZE_COLORS):
    # First plot all scatter points
    all_x = []
    all_y = []
    for size in list(size_colors.keys()):
        data = two_class_results.loc[size]
        x = np.array(two_class_results.columns, dtype=np.float64)
        y = np.array(data.values, dtype=np.float64)
        
        # Plot scatter points with consistent colors
        ax.scatter(x, y, marker='o', label=f'{size}', s=5, color=size_colors[size])
        
        # Collect valid points for overall spline
        mask = ~np.isnan(y) & ~np.isnan(x) & ~np.isneginf(y) & ~np.isneginf(x)
        all_x.extend(x[mask])
        all_y.extend(y[mask])
    
    # Add interpolating spline, ignoring nans
    mask = ~np.isnan(all_y) & ~np.isnan(all_x)
    if np.sum(mask) >= 3:  # Need at least 4 points for cubic spline
        all_x = np.array(np.array(all_x)[mask]) # exclude compute=0
        all_y = np.array(np.array(all_y)[mask]) # exclude compute=0

        x_nonzero = all_x != 0
        all_x = all_x[x_nonzero] # exclude x=0 values
        all_y = all_y[x_nonzero] # exclude x=0 values
        
        # Sort points by x value
        sort_idx = np.argsort(all_x)
        all_x = all_x[sort_idx]
        all_y = all_y[sort_idx]
        
        # Fit smoothed B-spline with high smoothing parameter
        x_smooth = np.logspace(np.log10(min(all_x)), np.log10(max(all_x)), len(all_x))
        # Use UnivariateSpline with high smoothing for a smoother fit
        spline = UnivariateSpline(np.log10(all_x), all_y, s=len(all_x))
        y_smooth = spline(np.log10(x_smooth))

        ax.plot(x_smooth, y_smooth, color='k', linestyle='--', label='spline', linewidth=1)
    
    # Add random baseline
    ax.axhline(y=0.5, color='r', linestyle='-', label='random', linewidth=0.5)
    
    ax.set_xlabel('Compute')
    ax.set_ylabel('2-class Accuracy')
    ax.set_title(f'{task}')
    ax.set_xscale('log', base=10)
    if show_legend: ax.legend(loc='lower right', fontsize=10, ncols=2)

    # Add vertical lines at specific FLOPS values with matching colors and accuracies
    # for flops, size in zip(sizes, ['150M', '300M', '530M', '750M', '1B']):
    for flops, size in zip(sizes, list(size_colors.keys())):
        try:
            acc = two_class_results.loc[size].get(np.float64(flops), np.nan)
            if not np.isnan(acc) and not np.isneginf(acc):
                ax.axvline(x=flops, color=size_colors[size], linestyle=':', alpha=0.7)
                ax.text(
                    flops, 0.98, ' ' + ('1.' if acc == 1 else f'{acc:.2f}').lstrip('0'), 
                    rotation=0, color=size_colors[size], ha='left', va='bottom', fontsize=8)
            else:
                raise FileNotFoundError(f'Not all results found for task={task}, size={size}')
        except Exception as e:
            # raise RuntimeError(f'Cant graph cheap decisions lines: {e}')
            print(f'Cant graph cheap decisions lines: {e}')