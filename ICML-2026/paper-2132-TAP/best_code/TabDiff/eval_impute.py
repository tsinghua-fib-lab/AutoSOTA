import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import root_mean_squared_error
import argparse
import json


parser = argparse.ArgumentParser(description='Missing Value Imputation')

parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
parser.add_argument('--exp_name', type=str, default=None)
parser.add_argument('--col', type=int, default=0, help='Numerical Column to Impute')
parser.add_argument('--non_learnable_schedule', action='store_true')

args = parser.parse_args()

dataname = args.dataname
exp_name = args.exp_name
if exp_name is None:
    exp_name = "non_learnable_schedule" if args.non_learnable_schedule else "learnable_schedule"
col = args.col

dataname = args.dataname

data_dir = f'data/{dataname}'

real_path = f'{data_dir}/test.csv'

info_path = f'data/{dataname}/info.json'
with open(info_path, 'r') as f:
    info = json.load(f)
task_type = info['task_type']


encoder = OneHotEncoder()

real_data = pd.read_csv(real_path)
target_col = real_data.columns[info['target_col_idx'][0]]

if task_type == "binclass":
    real_target = real_data[target_col].to_numpy().reshape(-1,1)
    real_y = encoder.fit_transform(real_target).toarray()
    
    syn_y = []
    for i in range(50):
        syn_path = f'impute/{dataname}/{exp_name}/{i}.csv'
        syn_data = pd.read_csv(syn_path)
        target = syn_data[target_col].to_numpy().reshape(-1, 1)
        syn_y.append(encoder.transform(target).toarray())

    syn_y_prob = np.stack(syn_y).mean(0)
    syn_y_oh = np.argmax(syn_y_prob, axis=1)
    num_classes = np.max(syn_y_oh) + 1
    syn_y_oh = np.eye(num_classes)[syn_y_oh]

    


    micro_f1 = f1_score(real_y.argmax(axis=1), syn_y_prob.argmax(axis=1), average='micro')
    auc = roc_auc_score(real_y, syn_y_prob, average='micro')
    auc_argmaxed = roc_auc_score(real_y, syn_y_oh, average='micro')
    print("AUC: ", round(auc*100, 3))
else:
    y_test = real_data[target_col].to_numpy()
    y_test = np.log(np.clip(y_test, 1, 20000))
    
    syn_y_ = []
    error = []
    for i in range(50):
        syn_path = f'impute/{dataname}/{exp_name}/{i}.csv'
        syn_data = pd.read_csv(syn_path)
        syn_y = syn_data[target_col].to_numpy()
        syn_y = np.log(np.clip(syn_y, 1, 20000))
        syn_y_.append(syn_y)
        
    pred = np.stack(syn_y_).mean(0)
    root_mean_squared = root_mean_squared_error(y_test, pred)   # mean_squared_error with squared=False is deprecated
    
    print("RMSE:", round(root_mean_squared, 4))
    