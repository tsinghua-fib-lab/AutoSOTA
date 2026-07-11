import numpy as np
import pandas as pd
import argparse
import os
import json
from pathlib import Path
from tqdm import tqdm

from utils_data_gen import *
from utils_estimate import *

parser = argparse.ArgumentParser()
parser.add_argument("--d", nargs="+", default=6, type=int)
parser.add_argument("--n_val", default=1000, type=int)
parser.add_argument("--n_rct", default=5000, type=int)
parser.add_argument("--n_obs", default=50000, type=int)
parser.add_argument("--num_trials", default=200, type=int)
parser.add_argument("--seed", default=99, type=int)
parser.add_argument("--bias_S", action="store_true")
parser.add_argument("--bias_A", action="store_true")
parser.add_argument("--bias_Y0", action="store_true")
parser.add_argument("--bias_Y1", action="store_true")
parser.add_argument("--bias_trs", action="store_true")
parser.add_argument("--bias_sel_type2_probs", nargs=4, type=float, default=[])
parser.add_argument("--full_adj", action="store_true")
parser.add_argument("--bias_type", default="dummy_results", type=str)
parser.add_argument("--u_type", default="disc", type=str)
args = parser.parse_args()

np.random.seed(args.seed)
ds = args.d if isinstance(args.d, list) else [args.d]

p_positivity = 0.1
# Use fixed extreme ranges matching old code behavior
pl_range = (0.1, 0.2)
ph_range = (0.8, 0.9)

pX = {"R=0": [0.4, 0.6], "R=1": [0.6, 0.4]}
bias_values = {"Y0": args.bias_Y0, "Y1": args.bias_Y1, "A": args.bias_A, "S": args.bias_S}

res_keys = ["SE_Y0", "SE_Y1", "SE_A", "SE_S"]

for d in ds:
    d_meas = d if args.full_adj else d - 1
    predictors = [f"Xp{i + 1}" for i in range(2 ** d_meas)]
    cov_res = pd.DataFrame(columns=[key + suffix for key in res_keys for suffix in ["_r", "_p"]], index=np.arange(args.num_trials))

    for k in tqdm(range(args.num_trials)):
        obs_probs = sample_all_probs(d, pl_range, ph_range, bias_values)
        rct_probs = {"Y0": obs_probs["Y0"], "Y1": obs_probs["Y1"],  "A": (2 ** d) * [0.5], "S": (2 ** d) * [0.95]}

        df_rct = init_df(args.n_rct, d, d_meas, 1, rct_probs, pX, False, [], args.u_type)
        df_obs = init_df(args.n_obs, d, d_meas, 0, obs_probs, pX, args.bias_trs, args.bias_sel_type2_probs, args.u_type)

        rct_models = fit_models(df_rct, predictors)
        obs_models = fit_models(df_obs, predictors)

        df_rct_val = init_df(args.n_val * 20, d, d_meas, 1, rct_probs, pX, False, [], args.u_type)
        df_obs_val = init_df(args.n_val * 20, d, d_meas, 0, obs_probs, pX, args.bias_trs, args.bias_sel_type2_probs, args.u_type)

        make_preds(df_rct_val, predictors, rct_models)
        make_preds(df_obs_val, predictors, obs_models)

        df_val = merge_df_val(df_rct_val, df_obs_val, predictors, rct_models, obs_models)

        for key in res_keys:
            r, p = pearsonr(df_val, "abs(b1(X))", key, args.n_val, random_state=args.seed + k)
            cov_res.loc[k, key + "_r"] = r
            cov_res.loc[k, key + "_p"] = p

    save_dir = Path(os.path.dirname(os.path.abspath(__file__)) + f"/results_U_{args.u_type}_ntrain-{args.n_rct}_nval-{args.n_val}/{args.bias_type}/d{d}")
    save_dir.mkdir(parents=True, exist_ok=True)

    cov_res.to_csv(os.path.join(save_dir, "results.csv"), index=False)    

    with open(os.path.join(save_dir, "args.txt"), "w") as f:
        json.dump(args.__dict__, f, indent=2)
