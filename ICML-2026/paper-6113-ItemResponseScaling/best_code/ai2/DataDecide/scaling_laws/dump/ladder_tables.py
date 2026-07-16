# %%
import os, sys, warnings
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from utils import DATA_DIR, ROOT_DIR

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import ladder_ian
plt.close()

warnings.filterwarnings("ignore", category=RuntimeWarning) # supress function fitting warnings
pd.set_option('display.max_columns', None) # display all pandas cols

# %%
from download.hf import pull_predictions_from_hf
local_path = pull_predictions_from_hf("davidheineman/consistent-ranking-evals", split_name='benchmarks')

# %%
df = pd.read_parquet(local_path)

if 'HF_TOKEN' not in os.environ:
    raise EnvironmentError("HF_TOKEN environment variable is not set")

local_path = pull_predictions_from_hf("allenai/ianm-datadecide-small-seed-extras", split_name='benchmarks')
df1 = pd.read_parquet(local_path)

df = pd.concat([df, df1])

print(f'Loaded {len(df):,} model evaluations')

# %%
# For my results, I'm only using the last seed for analysis
df = df[df['seed'] == 6198]

# %%
MIXES = df['group'].unique()
SIZES = df['size'].unique()
MULT  = df['chinchilla'].unique()

MODELS = df['model'].unique()
TASKS  = df['task'].unique()

# METRICS_RC= [
#     'primary_metric', 'correct_prob', 'correct_prob_per_token', 'correct_prob_per_char', 'margin', 'margin_per_token', 'margin_per_char', 'total_prob', 'total_prob_per_token', 'total_prob_per_char', 'uncond_correct_prob', 'uncond_correct_prob_per_token', 'uncond_correct_prob_per_char', 'norm_correct_prob', 'norm_correct_prob_per_token', 'norm_correct_prob_per_char',
#     'acc_raw', 'acc_per_token', 'acc_per_char', 'acc_uncond'
# ]

METRICS_RC= [
    'primary_metric', 'correct_prob', 'correct_prob_per_token', 'correct_prob_per_char', 'margin', 'margin_per_token', 'margin_per_char', 'total_prob', 'total_prob_per_token', 'total_prob_per_char', 'norm_correct_prob', 'norm_correct_prob_per_token', 'norm_correct_prob_per_char',
    'acc_raw', 'acc_per_token', 'acc_per_char'
]
# %%
# Get the S3 paths of the latest ckpt of 1B
s3_paths = []

for size in df['size'].unique():
    df_size = df[df['size'] == size]
    s3_paths.extend(df_size.loc[df_size.groupby('model')['step'].idxmax()]['s3_path'].tolist())

print(f'Largest model of each size (total={len(s3_paths)})')
s3_paths[:3]

setup_types = [
    '3_param',
    '3_param-helper_points',
    '3_param-step2=0.5',
    '3_param-helper_points-step2=0.5',
    '5_param-ai2',
    '5_param-1_step-ai2',
    '3_param-1_step',
    '2_param',
    '3_param-intermediate',
    '3_param-intermediate-helper_points'
]   

def generate_setups(setup_types, model_sizes):
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

# Example usage
model_sizes = ['4M',"6M","8M","10M","14M", "16M", '20M', '60M', '90M', '150M', '300M', '530M', '750M', '1B']
SETUPS = generate_setups(setup_types, model_sizes[:-1])
for setup in SETUPS:
    print(setup)


# %%
from ladder_ian import fit_all_mixes

# # Example table on two setups
# results = fit_all_mixes(
#     df,
#     all_models=MODELS,
#     # mixes=MIXES[:2], # only test with 2 data mixes
#     mixes=MIXES,
#     # tasks=TASKS[:2], # only test with 2 tasks
#     tasks=TASKS,
#     setups=SETUPS,
#     # y_metrics=["primary_metric"], #,"acc_per_char", "correct_logit_per_char"], # only test with 3 metrics
#     y_metrics=["correct_prob_per_token"],
#     x_metric="correct_logit_per_byte",
#     # quiet=False
# )
# display(results)

# %%
results = fit_all_mixes(
    df,
    all_models=MODELS,
    mixes=MIXES,
    tasks=TASKS,
    setups=SETUPS,
    y_metrics=METRICS_RC,
    x_metric="correct_logit_per_byte",
    quiet=True
)

# Remove any results where we did not have data to fit the scaling prediction (some of WinoGrande, BoolQ)
results = results[np.isfinite(results['step_1_pred'])].reset_index(drop=True)

# %%
results.to_csv('cheap_decisions_stacked_rc_pred_all.csv', index=False)
