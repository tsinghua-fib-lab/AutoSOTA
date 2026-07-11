import numpy as np
import pandas as pd
import argparse
import os, sys
import json
import pandas.api.types as ptypes
from pathlib import Path
from tqdm import tqdm
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from scipy.stats import zscore

from utils_models_v2 import fit_models, make_preds, merge_df_val, fit_model
from collections import defaultdict
from utils_v2 import pearsonr
from utils_whi_data import process_ct_df, process_os_df, combine_ct_os, add_target_variables

def run_experiment(df_ctos, predictors, args, save_file=True): 
    seeds = [42]
    seeds += [x for x in range(19)]

    df_rct_train_list = []
    df_obs_train_list = []
    df_rct_val_list = [] 
    df_obs_val_list = []

    for seed in seeds:  
        # split into train and val
        df_ctos_train, df_ctos_val = train_test_split(df_ctos, test_size=0.25, random_state=seed)

        # split into RCT and OBS
        df_rct_train = df_ctos_train.query('R == 1') 
        df_obs_train = df_ctos_train.query('R == 0')
        df_rct_val   = df_ctos_val.query('R == 1')
        df_obs_val   = df_ctos_val.query('R == 0')

        # add into lists
        df_rct_train_list.append(df_rct_train)
        df_obs_train_list.append(df_obs_train)
        df_rct_val_list.append(df_rct_val) 
        df_obs_val_list.append(df_obs_val)

    num_trials = len(seeds)
    bias_res = list() 
    cov_res = defaultdict(lambda: defaultdict(list))
    from tqdm import tqdm
    model_type = args.model_type
    for i in tqdm(range(num_trials)): 
        df_rct_train = df_rct_train_list[i]
        df_obs_train = df_obs_train_list[i]
        df_rct_val   = df_rct_val_list[i]
        df_obs_val   = df_obs_val_list[i]

        rct_models = fit_models(df_rct_train, predictors, is_rct=True, model=model_type)
        make_preds(df_rct_val, predictors, rct_models)
        
        obs_models = fit_models(df_obs_train, predictors, is_rct=False, model=model_type)
        make_preds(df_obs_val, predictors, obs_models)

        pr_model = fit_model(pd.concat([df_rct_train, df_obs_train]), predictors, "R")
        df_val = merge_df_val(df_rct_val, df_obs_val, predictors, pr_model, rct_models, obs_models)
        bias_res.append(df_val['b1(X)'].mean())
        for key in ['SE_Y0', 'SE_Y1', 'SE_A', 'SE_S']:
            cov_res['Pearson'][key].append(pearsonr(df_val, 'abs(b1(X))', key, df_val.shape[0]))


    cov_res_final = defaultdict(list)
    keys = ['SE_Y0', 'SE_Y1', 'SE_A', 'SE_S']
    for key in keys: 
        l = cov_res['Pearson'][key]
        # res = [x[0] for x in l if x[1] < alpha]
        # mean, standard deviation, sample size 
        mean = np.mean(l); std = np.std(l); n = len(l)
        lower = mean - 1.96 * (std / np.sqrt(n))
        upper = mean + 1.96 * (std / np.sqrt(n))
        cov_res_final[key].append(mean)
        cov_res_final[key].append(lower)
        cov_res_final[key].append(upper)

    df = pd.DataFrame.from_dict(cov_res_final, orient='index', columns=['mean', 'lower', 'upper'])
    if save_file:
        os.makedirs('./results', exist_ok=True)
        filename_save = f'./results/{args.drop_mechanism}_{args.selection_flag}_includecensored_{args.censored}_{args.outcome_name}_{model_type}.csv'
        print(f'Saving {filename_save}....')
        df.to_csv(
            filename_save,          # File name
            sep=',',               # Delimiter (comma)
            index=True,            # Include index
            header=True,           # Include headers
            float_format='%.6f'    # Floating-point format
        )
    else: 
        return df

if __name__ == "__main__":
    # Code to execute when the script is run
    parser = argparse.ArgumentParser()
    parser.add_argument('--selection_flag', default='biased', type=str)
    parser.add_argument('--drop_mechanism', default='drop_some_excluded', type=str)
    parser.add_argument('--censored', action='store_true')
    parser.add_argument('--outcome_name', default='CHD', type=str)
    parser.add_argument('--model_type', default='RF', type=str)
    args = parser.parse_args()

    ct_df = process_ct_df(args)
    os_df = process_os_df(args)
    ctos  = combine_ct_os(ct_df, os_df, args)
    ctos_df, predictors = add_target_variables(ctos, args)
    run_experiment(ctos_df, predictors, args)
