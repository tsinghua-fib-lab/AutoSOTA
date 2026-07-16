from pathlib import Path
import os, sys, warnings
sys.path.append(os.path.dirname(os.getcwd()))
from utils.constants import DATA_DIR

import warnings
import numpy as np
import pandas as pd
from utils.scaling_laws import fit_all_mixes
from utils.stats import compute_decision_accuracy
from utils.constants import ALL_METRICS
from remote.hf import push_parquet_to_hf, pull_predictions_from_hf

warnings.filterwarnings("ignore", category=RuntimeWarning) # supress function fitting warnings
pd.set_option('display.max_columns', None) # display all pandas cols

# SCALING_LAW_CONFIGS = [
#     '3_param',
#     '3_param-helper_points',
#     '3_param-step2=0.5',
#     '3_param-helper_points-step2=0.5',
#     '5_param-ai2',
#     '5_param-1_step-ai2',
#     '3_param-1_step',
#     '2_param',
#     '3_param-intermediate',
#     '3_param-intermediate-helper_points'
# ]

SCALING_LAW_CONFIGS = [
    # 3-parameter setup (FLOPs)
    '3_param-default',
    # '3_param-no_750M',
    # '3_param-no_750M_no_530M',

    #     # + helper points
    #     '3_param-default-helper_points',
    #     '3_param-no_750M-helper_points',
    #     '3_param-no_750M_no_530M-helper_points',

    #     # + only use 50% points for step 2
    #     '3_param-default-step2=0.5',
    #     '3_param-no_750M-step2=0.5',
    #     '3_param-no_750M_no_530M-step2=0.5',

    #     # + only use 50% points for step 2 + helper points
    #     '3_param-default-helper_points-step2=0.5',
    #     '3_param-no_750M-helper_points-step2=0.5',
    #     '3_param-no_750M_no_530M-helper_points-step2=0.5',

    # # Full AI2 scaling law
    # '5_param-ai2',
    # '5_param-ai2-no_750M',
    # '5_param-ai2-no_750M_no_530M',

    # # Single step version of AI2 scaling law
    # '5_param-1_step-ai2',
    # '5_param-1_step-ai2-no_750M',
    # '5_param-1_step-ai2-no_750M_no_530M',

    # # Single step version of 3-parameter setup (FLOPs)
    # '3_param-1_step',
    # '3_param-1_step-no_750M',
    # '3_param-1_step-no_750M_no_530M',

    # # Use 2 parameter setup (FLOPs prediction without bias term E)
    # '2_param-default',
    # '2_param-no_750M',
    # '2_param-no_750M_no_530M',

    # 3-parameter setup using metric as intermediate feature (FLOPs -> [metric] -> primary_metric)
    # '3_param-intermediate-default',
    # '3_param-intermediate-no_750M',
    # '3_param-intermediate-no_750M_no_530M',

        # # + helper points
        # '3_param-intermediate-default-helper_points',
        # '3_param-intermediate-no_750M-helper_points',
        # '3_param-intermediate-no_750M_no_530M-helper_points',
]

TARGET_SIZE = '1B'

def generate_setups(setup_types, model_sizes):
    """
    Generate fitting configurations by excluding different combinations of small models.
    >>> [
    >>>     3_param-intermediate
    >>>     3_param-intermediate-no_8M
    >>>     3_param-intermediate-no_8M_no_10M
    >>>     3_param-intermediate-no_8M_no_10M_no_14M
    >>>     3_param-intermediate-no_8M_no_10M_no_14M_no_16M
    >>>     ...
    >>> ]
    """
    def all_combinations(lst):
        result = []
        for end in range(3, len(lst)+1):
            result.append(lst[:end])
        for start in range(1, len(lst) - 2):
            result.append(lst[start:])
        return result

    setups = []
    for setup_type in setup_types:
        for combination in all_combinations(model_sizes):
            compliment = [size for size in model_sizes if size not in combination]
            setup_name = f"{setup_type}" + (f"-no_{'_no_'.join(compliment)}" if compliment else "")
            setups.append(setup_name)
    return setups

def run_ladder_fits(data_path, results_path, dry_run=False):
    local_path = pull_predictions_from_hf(data_path, split_name='macro_avg')
    df = pd.read_parquet(local_path)
    print(f'Loaded {len(df):,} model evaluations')

    # For my results, I'm only using the last seed for analysis
    df = df[(df['seed'] == 6198) | (df['seed'] == 'default')]
    
    # Restore the original col names in this code
    if 'params' in df.columns: 
        df['size'] = df['params']
    if 'data' in df.columns: 
        from utils.constants.constants_recepies import DATA_NAME_CLEAN
        DATA_NAME_CLEAN_REV = {v: k for k, v in DATA_NAME_CLEAN.items()}
        df['group'] = df['data'].map(DATA_NAME_CLEAN_REV)
    if 'model' not in df.columns: 
        df['model'] = (df['group'] + '-' + df['params'].astype(str) + '-' + df['chinchilla'].astype(str))

    # Incomplete results
    # df = df[~df['size'].isin(['4M', '6M'])]
    df = df[df['size'].isin(['150M', '300M', '530M', '750M', '1B'])]

    MIXES  = df['group'].unique()
    SIZES  = df['size'].unique()
    MULT   = df['chinchilla'].unique()
    MODELS = df['model'].unique()
    TASKS  = df['task'].unique()

    # sort sizes
    SIZES = sorted(SIZES, key=lambda x: float(x.replace('M', '').replace('B', '')) * (1000 if 'B' in x else 1))
    
    # SETUPS = generate_setups(SCALING_LAW_CONFIGS, SIZES[:-1])
    SETUPS = SCALING_LAW_CONFIGS

    if dry_run:
        # Example table on two setups
        results = fit_all_mixes(
            df,
            all_models=MODELS,
            mixes=MIXES[:2], # only test with 2 data mixes
            tasks=TASKS[:2], # only test with 2 tasks
            setups=SETUPS,
            y_metrics=["primary_metric", "acc_per_char", "correct_prob_per_char"], # only test with 3 metrics
            x_metric="correct_logit_per_byte",
        )
    else:
        results = fit_all_mixes(
            df,
            all_models=MODELS,
            mixes=MIXES,
            tasks=TASKS,
            setups=SETUPS,
            # y_metrics=ALL_METRICS,
            y_metrics= ["primary_metric", "acc_per_char", "correct_prob_per_char", "acc_raw"],
            x_metric="correct_logit_per_byte",
            quiet=True
        )

    # Remove any results where we did not have data to fit the scaling prediction (some of WinoGrande, BoolQ)
    results = results[np.isfinite(results['step_1_pred'])].reset_index(drop=True)

    results = compute_decision_accuracy(df, results, TARGET_SIZE)

    results.to_csv(results_path, index=False)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='allenai/DataDecide-eval-results',
                      help='HuggingFace dataset to load results from')
    parser.add_argument('--result-path', type=str, default='ladder_predictions.csv',
                      help='Path to save results CSV (default: ladder_predictions.csv)')
    parser.add_argument('--dry-run', action='store_true',
                      help='Run on subset of data for testing')
    parser.add_argument('--push-to-hf', type=str,
                      help='HuggingFace dataset name to push results to (optional)')
    args = parser.parse_args()
    
    data_path    = args.data_path
    results_path = Path(DATA_DIR) / args.result_path
    
    run_ladder_fits(data_path, results_path, dry_run=args.dry_run)

    if args.push_to_hf:
        push_parquet_to_hf(
            parquet_file_path=results_path,
            hf_dataset_name=args.push_to_hf,
            split_name='scaling_law_fit',
            overwrite=True,
            private=True,
        )