import numpy as np
import pandas as pd
import argparse
import os, sys
import json
from pathlib import Path

from utils_v2 import *
from utils_data_v2 import *

parser = argparse.ArgumentParser()
parser.add_argument('--d', default=6, type=int)
parser.add_argument('--n_val', default=1000, type=int)
parser.add_argument('--n_rct', default=10000, type=int)
parser.add_argument('--n_obs', default=10000, type=int)
parser.add_argument('--num_trials', default=2, type=int)
parser.add_argument('--seed', default=99, type=int)
parser.add_argument('--res_dir', default="dummy_results", type=str)
parser.add_argument('--bias_S', action="store_true")
parser.add_argument('--bias_A', action="store_true")
parser.add_argument('--bias_Y0', action="store_true")
parser.add_argument('--bias_Y1', action="store_true")
parser.add_argument('--bias_trs', action="store_true")
args = parser.parse_args()

np.random.seed(args.seed)

d = args.d
d_meas = d - 1
predictors = [f"Xp{i + 1}" for i in range(2 ** d_meas)]

n_rct, n_rct_val = args.n_rct, args.n_val * 20
n_obs, n_obs_val = args.n_obs, args.n_val * 20

pl_range = (0.1, 0.2)
ph_range = (0.8, 0.9)
pX = {"R=0": [0.4, 0.6], "R=1": [0.6, 0.4]}

bias_values = {"Y0": args.bias_Y0, "Y1": args.bias_Y1, "A": args.bias_A, "S": args.bias_S}

res_keys = ["SE_Y0", "SE_Y1", "SE_A", "SE_S"]
cov_res = pd.DataFrame(columns=[key + suffix for key in res_keys for suffix in ["_r", "_p"]], 
                       index=np.arange(args.n_val))

for k in range(args.num_trials):

    if args.bias_trs:
        p_os_u0, p_rct_u0 = np.random.uniform(0.1,0.4), np.random.uniform(0.6,0.9)
        pU = {"R=0": [p_os_u0, 1 - p_os_u0], "R=1": [p_rct_u0, 1 - p_rct_u0]}
    else:
        pU = pX

    obs_probs = sample_all_probs(d, pl_range, ph_range, bias_values)
    rct_probs = {"Y0": obs_probs["Y0"], "Y1": obs_probs["Y1"],  "A": (2 ** d) * [0.5], "S": (2 ** d) * [0.95]}

    df_rct     = init_df(n_rct, d, d_meas, 1, rct_probs, pX, pU)
    df_obs     = init_df(n_obs, d, d_meas, 0, obs_probs, pX, pU)
    df_rct_val = init_df(n_rct_val, d, d_meas, 1, rct_probs, pX, pU)
    df_obs_val = init_df(n_obs_val, d, d_meas, 0, obs_probs, pX, pU)

    rct_models = fit_models(df_rct, predictors)
    make_preds(df_rct_val, predictors, rct_models)
    
    obs_models = fit_models(df_obs, predictors)
    make_preds(df_obs_val, predictors, obs_models)

    pr_model = fit_model(pd.concat([df_rct, df_obs]), predictors, "R")
    df_val = merge_df_val(df_rct_val, df_obs_val, predictors, pr_model, rct_models, obs_models)

    for key in res_keys:
        r, p = pearsonr(df_val, "abs(b1(X))", key, args.n_val)
        cov_res.loc[k, key + "_r"] = r
        cov_res.loc[k, key + "_p"] = p

save_dir = Path(os.path.dirname(os.path.abspath(__file__)) + f'/results/{args.res_dir}')
save_dir.mkdir(parents=True, exist_ok=True)

cov_res.to_csv(os.path.join(save_dir, 'r_p.csv'), index=False)    

with open(os.path.join(save_dir, 'args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)
