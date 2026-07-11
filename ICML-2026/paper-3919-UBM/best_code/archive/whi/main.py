import numpy as np
import pandas as pd
import argparse
import os, sys
import json
from pathlib import Path
from tqdm import tqdm
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from scipy.stats import zscore

sys.path.append('../synthetic/')
from utils_data_v2 import fit_models, make_preds, merge_df_val, fit_model
from collections import defaultdict
from utils_v2 import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument('--selection_flag', default='biased', type=str)
parser.add_argument('--censored', action='store_true')
parser.add_argument('--outcome_name', default='CHD', type=str)
parser.add_argument('--model_type', default='RF', type=str)
args = parser.parse_args()

# read tables
dir_path = '/Users/zeshanmh/Documents/research/benchmarking-os/'
out   = pd.read_csv(os.path.join(dir_path, 'whi/data/data/main_study/csv/outc_adj_bio.csv'))
ct_fu = pd.read_csv(os.path.join(dir_path, 'whi/data/data/main_study/csv/adh_ht_pub.csv'))[['ID', 'ADHRATE', 'ENDDY', 'STARTDY', 'LOST', 'STOPHRT']] 
std_trt = pd.read_csv(os.path.join(dir_path, 'whi/data/data/main_study/csv/dem_ctos_bio.csv'))[['ID', 'HRTARM', 'OSFLAG']]

# List of outcomes     
glbl_list = ['CHD', 'BREAST', 'STROKE', 'PE', 'ENDMTRL', 'COLORECTAL', 'BKHIP', 'DEATH']    
other_list = ['PTCA', 'DVT']

# Get end of follow-up for CT patients 
# BTW, do we have to consider START-DAY? what about LOST for censoring?
def get_lost_day(group):
    lost_rows = group[group['LOST'] == 'Yes']
    return lost_rows['ENDDY'].iloc[0] if not lost_rows.empty else None

ct_end = (ct_fu[ct_fu['ADHRATE'].notna()]
          .groupby('ID')
          .agg({
              'ENDDY': 'max',
              'LOST': lambda x: 1 if 'Yes' in x.values else 0,
          })
          .reset_index())

# ADD LOST_DY column
lost_days = (ct_fu[ct_fu['ADHRATE'].notna()]
             .groupby('ID')[['LOST','ENDDY']]
             .apply(get_lost_day)
             .rename('LOST_DY'))

ct_end = ct_end.merge(lost_days.to_frame(), on='ID', how='left')
ct_end = ct_end.rename(columns={'ENDDY': 'END_DY'})

ct_df = std_trt.drop_duplicates('ID')
ct_df = ct_df[ct_df['HRTARM'].isin(['E+P intervention', 'E+P control'])]
ct_df = ct_df.merge(ct_end, on='ID', how='left')
ct_df = ct_df.merge(out, on='ID', how='left')

# code variables HRTARM and OS 
ct_df['OS'] = 0 
ct_df['HRTARM'] = ct_df['HRTARM'].map({'E+P intervention': 1, 'E+P control': 0})

# print out first 10 rows
print(ct_df.shape)
print(ct_df[ct_df['HRTARM'] == 1].shape)
print(ct_df[ct_df['HRTARM'] == 0].shape)

diff_selection_for_CT = False

# process outcomes 
for i in glbl_list + other_list: 
    ct_df[i+'_E']  = ((ct_df[i] == 1) & (ct_df[i+'DY'] <= ct_df['END_DY'])).astype(int)
    ct_df[i+'_DY'] = np.where(ct_df[i+'_E'] == 1, ct_df[i+'DY'], ct_df['END_DY'])
    ct_df[i+'_EDY'] = np.where(ct_df[i+'_E'] == 1, ct_df[i+'DY'], np.nan) 

# Global index
ct_df['GLBL_E'] = (ct_df[[j+'_E' for j in glbl_list]].sum(axis=1) > 0).astype(int)
ct_df['GLBL_DY'] = np.where(ct_df['GLBL_E'] == 1,
                            ct_df[[j+'_EDY' for j in glbl_list]].min(axis=1),
                            ct_df[[j+'_DY' for j in glbl_list]].min(axis=1))

# Selection variable 
ct_df['S'] = 1

# Add different selection variables for each outcome (this is because S = 0 for censored patients)
for i in glbl_list + other_list: 
    ct_df['S_'+i] = ct_df['S']
    if diff_selection_for_CT:
        ct_df['S_'+i] = np.where(ct_df[i+'DY'] > ct_df['END_DY'], 0, ct_df['S_'+i])
        ct_df['S_'+i] = np.where(((ct_df['LOST'] == 1) & (ct_df[i+'DY'] > ct_df['LOST_DY'])), 0, ct_df['S_'+i])
ct_df['S_GLBL'] = ct_df['S']

# Select needed columns
ct_df = ct_df[['ID', 'OS', 'HRTARM'] + 
                ['S_'+j for j in glbl_list + other_list + ['GLBL']] +
                [j+'_E' for j in glbl_list + other_list + ['GLBL']] + 
                [j+'_DY' for j in glbl_list + other_list + ['GLBL']]]

dir_path = '/Users/zeshanmh/Documents/research/benchmarking-os/'
hyst    = pd.read_csv(os.path.join(dir_path, 'whi/data/data/main_study/csv/f2_ctos_bio.csv'))[['ID','HYST']]
pre_hrt  = pd.read_csv(os.path.join(dir_path, 'whi/data/data/main_study/csv/f43_ctos_bio.csv'))[['ID', 'TOTESTAT','TOTPSTAT']]
post_hrt = pd.read_csv(os.path.join(dir_path, 'whi/data/data/main_study/csv/f48_av1_os_pub.csv'))[['ID','ELSTYR','PLSTYR','HRTCMBP']]
unc_hf   = pd.read_csv(os.path.join(dir_path, 'whi/data/data/main_study/csv/unc_hf_bio.csv'))[['ID','CHDYRHX','CHDEVERHX','HYPERTNHX','MIHX','PVDHX','DIABHX','STROKEHX']]

# construct os_df 
selection_flag = args.selection_flag
'''
drop_all_excluded: this drops all patients who had hysterectomy OR are on unopposed estrogen; thus, selection, S = 0 and S = 1, is based on censoring only
drop_some_excluded: this keeps patients who had hyseterectomy OR are on unopposed estrogen but were past users of combined HRT, assigns them to be S = 0;
censored patients are additionally S = 0
drop_no_excluded: keeps all patients who had hysterectomy OR are on unopposed estrogen, and assigns them S = 0; censored patients are additionally S = 0
'''
additional_selection_processing = 'drop_some_excluded' # 'drop_some_excluded', 'drop_no_excluded', 'drop_all_excluded'
'''
if censored_patients_sel0 = True, then censored patients are additionally S = 0
'''
censored_patients_sel0 = args.censored 

os_df = std_trt.drop_duplicates('ID')
os_df = os_df[os_df['OSFLAG'] == 'Yes']
os_df = os_df.merge(hyst, on='ID', how='left')
os_df = os_df.merge(pre_hrt, on='ID', how='left')
print(os_df['TOTESTAT'].value_counts())
print(os_df['HYST'].value_counts())
os_df = os_df.merge(post_hrt, on='ID', how='left')
if additional_selection_processing == 'drop_some_excluded': 
    os_df = os_df.merge(unc_hf, on='ID', how='left')
    condition_dict = {
        'CHDEVERHX': ('!=', 1.),
        'HYPERTNHX': ('!=', 1.),
        'MIHX': ('!=', 1.),
        'PVDHX': ('!=', 1.),
        'DIABHX': ('!=', 1.),
        'STROKEHX': ('!=', 1.)
    }
    
    condition = (os_df['HYST'] == 'Yes') & ((os_df['TOTPSTAT'] == 'Never used') | (os_df['TOTPSTAT'] == 'Current user'))
    os_df = os_df[~condition]
    condition2 = (os_df['TOTESTAT'] == 'Current user') & ((os_df['TOTPSTAT'] == 'Never used') | (os_df['TOTPSTAT'] == 'Current user'))
    os_df = os_df[~condition2]
    os_df['S'] = os_df.apply(
        lambda row: 0 if (row['HYST'] == 'Yes' or row['TOTESTAT'] == 'Current user') else 1,
        axis=1
    )
elif additional_selection_processing == 'drop_no_excluded': 
    # Selected patients
    os_df['S'] = os_df.apply(
        lambda row: 1 if (row['HYST'] == 'No' and row['TOTESTAT'] in ['Never used', 'Past user']) else 0,
        axis=1
    )
elif additional_selection_processing == 'drop_all_excluded':
    os_df = os_df[os_df['HYST'] == 'No']
    os_df = os_df[os_df['TOTESTAT'].isin(['Never used', 'Past user'])]
    os_df['S'] = 1

os_df = os_df.merge(out, on='ID', how='left')

# 35551 (control) + 17503 (intervention) = 53054

print(os_df[os_df['TOTPSTAT'].isin(['Current user'])].shape)
print(os_df[os_df['TOTPSTAT'].isin(['Never used', 'Past user'])].shape)

if selection_flag == 'biased': 
    os_df = os_df[os_df['TOTPSTAT'].isin(['Never used', 'Past user','Current user'])]
    os_df['HRTARM'] = os_df['TOTPSTAT'].map({'Current user': 1, 'Never used': 0, 'Past user': 0})
elif selection_flag == 'unbiased' or selection_flag == 'manually_biased': 
    os_df = os_df[os_df['TOTPSTAT'].isin(['Never used', 'Past user','Current user'])]
    conditions = [
        (((os_df['ELSTYR'] == 'Yes') & (os_df['PLSTYR'] == 'Yes')) | (os_df['HRTCMBP'] == 'Yes')),
        ((os_df['ELSTYR'] == 'No') & (os_df['PLSTYR'] == 'No')),
        (((os_df['ELSTYR'] == 'Yes') & (os_df['PLSTYR'] == 'No')) | ((os_df['ELSTYR'] == 'No') & (os_df['PLSTYR'] == 'Yes')))
    ]
    choices = [1, 0, -1]
    os_df['HRTGRP'] = np.select(conditions, choices, default=-2)
    os_df = os_df[os_df['HRTGRP'] != -2]
    os_df['HRTARM'] = (os_df['HRTGRP'] == 1).astype(int)
    os_df['S'] = os_df.apply(lambda row: 0 if row['TOTPSTAT'] == 'Current user' else row['S'], axis=1)
os_df['OS'] = 1

# os_end_day = None
os_end_day = 6*365
os_df['END_DY'] = os_end_day if os_end_day is not None else os_df['ENDFOLLOWDY']
# os_df['END_DY'] = os_df.apply(lambda x: x['DEATHDY'] if x['DEATHDY'] < os_end_day else os_end_day, axis=1)


# Process outcomes (same as CT)
for i in glbl_list + other_list:
    os_df[i+'_E'] = ((os_df[i] == 1) & (os_df[i+'DY'] <= os_df['END_DY'])).astype(int)
    os_df[i+'_DY'] = np.where(os_df[i+'_E'] == 1, os_df[i+'DY'], os_df['END_DY'])
    os_df[i+'_EDY'] = np.where(os_df[i+'_E'] == 1, os_df[i+'_DY'], np.nan)

# Global index
os_df['GLBL_E'] = (os_df[[j+'_E' for j in glbl_list]].sum(axis=1) > 0).astype(int)
os_df['GLBL_DY'] = np.where(os_df['GLBL_E'] == 1,
                            os_df[[j+'_EDY' for j in glbl_list]].min(axis=1),
                            os_df[[j+'_DY' for j in glbl_list]].min(axis=1))

# Selection variable adjustment
for i in glbl_list + other_list:
    os_df['S_'+i] = os_df['S']
    if censored_patients_sel0: 
        os_df['S_'+i] = np.where(os_df[i+'DY'] > os_df['END_DY'], 0, os_df['S_'+i])
os_df['S_GLBL'] = os_df['S']

# Select needed columns
os_df = os_df[['ID', 'OS', 'HRTARM'] + 
                ['S_'+j for j in glbl_list + other_list + ['GLBL']] + 
                [j+'_E' for j in glbl_list + other_list + ['GLBL']] + 
                [j+'_DY' for j in glbl_list + other_list + ['GLBL']]]

ctos_df = pd.concat([ct_df, os_df], ignore_index=True)
if selection_flag == 'manually_biased': 
    # removing age and menopausal status
    categorical_features = {
        'dem_ctos_bio.csv': {'ETHNIC': True, 'EDUC': True}, 
        'f80_ctos_bio.csv': {'BMI': False}, 
        'f34_ctos_bio.csv': {'SMOKING': True}, 
        'f151_ctos_bio.csv': {'PHYSFUN': False}    
    }
    
    new_feature_dict = { 
        'dem_ctos_bio.csv': ['ETHNIC_White', \
                             'EDUC_Some post-graduate or professional', \
                             'EDUC_Some college or Associate Degree'],
        'f80_ctos_bio.csv': ['BMI'],
        'f34_ctos_bio.csv': ['SMOKING_Past Smoker', 'SMOKING_Current Smoker'],
        'f151_ctos_bio.csv': ['PHYSFUN']
    }
else:
    categorical_features = {
        'dem_ctos_bio.csv': {'AGE': False, 'ETHNIC': True, 'EDUC': True}, 
        'f80_ctos_bio.csv': {'BMI': False}, 
        'f34_ctos_bio.csv': {'SMOKING': True}, 
        'f31_ctos_bio.csv': {'MENO': False}, 
        'f151_ctos_bio.csv': {'PHYSFUN': False}    
    }
    
    new_feature_dict = { 
        'dem_ctos_bio.csv': ['AGE', 'ETHNIC_White', \
                             'EDUC_Some post-graduate or professional', \
                             'EDUC_Some college or Associate Degree'],
        'f80_ctos_bio.csv': ['BMI'],
        'f34_ctos_bio.csv': ['SMOKING_Past Smoker', 'SMOKING_Current Smoker'],
        'f31_ctos_bio.csv': ['MENO'],
        'f151_ctos_bio.csv': ['PHYSFUN']
    }
    
import pandas.api.types as ptypes

ctos_temp = ctos_df.copy()
# Dictionary to specify which features are categorical

# dfs = []  # Store all dataframes to concatenate later
new_dir_path = dir_path + 'whi/data/data/main_study/csv'

for filename, f_dict in categorical_features.items():
    # Read the data
    df = pd.read_csv(os.path.join(new_dir_path, filename))
    if filename == 'f80_ctos_bio.csv': 
        df = df.query('F80VTYP == "Screening"')
    elif filename == 'f151_ctos_bio.csv': 
        idx = df.groupby('ID')['F151DAYS'].idxmin().reset_index(drop=True)
        df = df.loc[idx, :].reset_index(drop=True)[['ID','PHYSFUN']]
    # Select needed columns
    features = list(f_dict.keys())
    df = df[['ID'] + features]
    
    # Separate ID column
    id_col = df['ID']
    print(f"Processed {filename}")
    print(df.shape)

    orig_cols = ctos_temp.columns.tolist()
    ctos_temp = ctos_temp.merge(df, on='ID', how='left')

    # Handle continuous and categorical features separately
    cont_features = [f for f in features if not f_dict[f]]
    cat_features = [f for f in features if f_dict[f]]
    
    # Handle continuous features
    if cont_features:
        cont_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        ctos_temp[cont_features] = cont_imputer.fit_transform(ctos_temp[cont_features])
    
    # Handle categorical features
    if cat_features:
        cat_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        ctos_temp[cat_features] = cat_imputer.fit_transform(ctos_temp[cat_features])
        
        # One-hot encode categorical features
        ctos_temp = pd.get_dummies(ctos_temp, columns=cat_features, prefix=cat_features)

    if filename == 'dem_ctos_bio.csv': 
        ctos_temp = ctos_temp.rename(columns={'ETHNIC_White (not of Hispanic origin)': 'ETHNIC_White'})

    ctos_temp = ctos_temp[orig_cols + new_feature_dict[filename]]

ctos_temp = ctos_temp.astype({col: int for col in ctos_temp.select_dtypes(include='bool').columns})

from sklearn.model_selection import train_test_split 

df_ctos = ctos_temp.copy()
if selection_flag == 'manually_biased': 
    predictors = ['ETHNIC_White', 'EDUC_Some post-graduate or professional', 
          'EDUC_Some college or Associate Degree', 'BMI', 'SMOKING_Past Smoker', 
          'SMOKING_Current Smoker', 'PHYSFUN'] 
else: 
    predictors = ['AGE', 'ETHNIC_White', 'EDUC_Some post-graduate or professional', 
          'EDUC_Some college or Associate Degree', 'BMI', 'SMOKING_Past Smoker', 
          'SMOKING_Current Smoker', 'MENO', 'PHYSFUN'] 
outcome_name = args.outcome_name # STROKE, BREAST

outcome= outcome_name + '_E'
trt    = 'HRTARM'
select = f'S_{outcome_name}'

drop_columns = [x for x in df_ctos.columns if x not in predictors + [outcome, trt, 'ID', 'S']]
df_ctos.rename(columns={trt: 'A', outcome: 'Y'}, inplace=True) 
df_ctos['S']  = df_ctos[select]
df_ctos['Y0'] = df_ctos['Y']
df_ctos['Y1'] = df_ctos['Y']
df_ctos['R'] = 1 - df_ctos['OS']
df_ctos.drop(columns=drop_columns, inplace=True)

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
filename_save = f'./results/DROPSOME_{selection_flag}_includecensored_{censored_patients_sel0}_{outcome_name}_{model_type}.csv'
print(f'Saving {filename_save}....')
df.to_csv(
    filename_save,          # File name
    sep=',',               # Delimiter (comma)
    index=True,            # Include index
    header=True,           # Include headers
    float_format='%.6f'    # Floating-point format
)
