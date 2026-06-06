import numpy as np
import pandas as pd
import os 

import json

# Metrics
from sdmetrics.reports.single_table import QualityReport, DiagnosticReport


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataname', type=str, default='adult')
parser.add_argument('--model', type=str, default='tabsyn')
parser.add_argument('--path', type=str, default = None, help='The file path of the synthetic data')

args = parser.parse_args()


def reorder(real_data, syn_data, info):
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    task_type = info['task_type']
    if task_type == 'regression':
        num_col_idx += target_col_idx
    else:
        cat_col_idx += target_col_idx

    real_num_data = real_data[num_col_idx]
    real_cat_data = real_data[cat_col_idx]

    new_real_data = pd.concat([real_num_data, real_cat_data], axis=1)
    new_real_data.columns = range(len(new_real_data.columns))

    syn_num_data = syn_data[num_col_idx]
    syn_cat_data = syn_data[cat_col_idx]
    
    new_syn_data = pd.concat([syn_num_data, syn_cat_data], axis=1)
    new_syn_data.columns = range(len(new_syn_data.columns))

    
    metadata = info['metadata']

    columns = metadata['columns']
    metadata['columns'] = {}

    inverse_idx_mapping = info['inverse_idx_mapping']


    for i in range(len(new_real_data.columns)):
        if i < len(num_col_idx):
            metadata['columns'][i] = columns[num_col_idx[i]]
        else:
            metadata['columns'][i] = columns[cat_col_idx[i-len(num_col_idx)]]
    

    return new_real_data, new_syn_data, metadata

if __name__ == '__main__':

    dataname = args.dataname
    model = args.model

    if not args.path:
        syn_path = f'synthetic/{dataname}/{model}.csv'
    else:
        syn_path = args.path

    real_path = f'synthetic/{dataname}/real.csv'

    data_dir = f'data/{dataname}' 
    print(syn_path)

    with open(f'{data_dir}/info.json', 'r') as f:
        info = json.load(f)

    syn_data = pd.read_csv(syn_path)
    real_data = pd.read_csv(real_path)

    save_dir = f'eval/density/{dataname}/{model}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    real_data.columns = range(len(real_data.columns))
    syn_data.columns = range(len(syn_data.columns))

    metadata = info['metadata']
    metadata['columns'] = {int(key): value for key, value in metadata['columns'].items()}

    new_real_data, new_syn_data, metadata = reorder(real_data, syn_data, info)

    qual_report = QualityReport()
    qual_report.generate(new_real_data, new_syn_data, metadata)

    diag_report = DiagnosticReport()
    diag_report.generate(new_real_data, new_syn_data, metadata)

    quality =  qual_report.get_properties()
    diag = diag_report.get_properties()

    Shape = quality['Score'][0]
    Trend = quality['Score'][1]

    with open(f'{save_dir}/quality.txt', 'w') as f:
        f.write(f'{Shape}\n')
        f.write(f'{Trend}\n')

    Quality = (Shape + Trend) / 2

    shapes = qual_report.get_details(property_name='Column Shapes')
    trends = qual_report.get_details(property_name='Column Pair Trends')
    coverages = diag_report.get_details('Coverage')


    shapes.to_csv(f'{save_dir}/shape.csv')
    trends.to_csv(f'{save_dir}/trend.csv')
    coverages.to_csv(f'{save_dir}/coverage.csv')
