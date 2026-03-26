import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors
import pickle
import pandas as pd
import argparse
import os
import numpy as np
import re
from functools import reduce
from utils import get_trainable_params


def load_pkl(alpha_train, dir):
    with open(dir+f"atrain-{alpha_train:.2f}.pkl", 'rb') as f:
        return pickle.load(f)
    
def process_results(alpha_train, dir):
    res = load_pkl(alpha_train, dir)
    flattened_data = []
    for key, metrics in res.items():
        for metric, values in metrics.items():
            for i, value in enumerate(values):
                flattened_data.append({'model': key, 'iteration': i, 'metric': metric, 'value': value})

    # Create DataFrame
    df = pd.DataFrame(flattened_data)
    df = df.assign(alpha_train=alpha_train)
    #df = df.iloc[::-1].reset_index(drop=True)

    return df

def extract_maskrate(text, m, nparams):
    # Default idx
    np = 0.
    if 'original' in text:
        # idx = 0
        np = 0.
    elif 'classifier' in text:
        np = nparams['classifier']
    elif 'emb' in text:
        np = nparams['total']
    else:
        # if encoder layer, extract the layer number from the text
        match = re.search(r'layer(\d+)\.masked_dementia', text)
        if match:
            layer_num = int(match.group(1))
            idx = 14 - layer_num
            np = nparams['classifier'] + idx * nparams['encoder']
    # Calculate maskrate
    prop = np/nparams['total']
    return m * prop

def extract_ratio(text):
    match = re.search(r'\d+\.\d+', text)
    return match.group() if match else 0.

def extract_type(text):
    match = re.search(r'^(.+?)\.', text)
    return match.group(1) if match else None


    
def process_df(res_df):
    res_df['ratio'] = res_df['model'].apply(extract_ratio).astype(float)
    res_df['type'] = res_df['model'].apply(extract_type)
    res_df['alpha_train'] = res_df['alpha_train'].astype(float)
    res_df['ratio'] = res_df['ratio'].astype(float)
    extened_df = []
    for type in ['all', 'compliment', 'intersection']:
        intactdf = res_df[res_df['type'].isnull()]
        intactdf['type'] = type
        extened_df.append(intactdf)
    intact_df = pd.concat(extened_df, ignore_index=True)
    res_df = res_df.dropna()
    res_df = pd.concat([res_df, intact_df], ignore_index=True)
    res_df = res_df.sort_values(by=['alpha_train', 'type', 'ratio', 'iteration'])
    return res_df


def join_dfs(df_local, df_global, mr):
    nparams = get_trainable_params('bert-base-uncased')
    df_local['ratio'] = df_local['model'].apply(extract_maskrate, m=mr, nparams=nparams).astype(float)

    df_local['type'] = 'ecf'
    df_global['ratio'] = df_global['size_percentage'].astype(float)
    df_global.type = df_global.type.where(df_global.type != 'compliment', 'difference')


    df_global = df_global[['type', 'ratio', 'alpha_train', 'value']]
    df_local = df_local[['type', 'ratio', 'alpha_train', 'value']]    
    df = pd.concat([df_local, df_global], ignore_index=True)
    df['alpha_train'] = df['alpha_train'].astype(str)

    return df


def draw_plot(plotdf, mr, data_name):
    sns.set_style("white")
    sns.set_context("poster", rc={"grid.linewidth": 1.0})    

    alpha_train_values = plotdf['alpha_train'].unique()

    fig, axes = plt.subplots(nrows=3, ncols=len(alpha_train_values), figsize=(15, 10), sharey=True, sharex=True)

    comparison_types = ['intersection', 'difference', 'all']

    for row_idx, comp_type in enumerate(comparison_types):
        for col_idx, alpha_train in enumerate(alpha_train_values):
            df_plot = plotdf[(plotdf['type'].isin([comp_type, 'ecf'])) & (plotdf['alpha_train'] == alpha_train)]
            ax = axes[row_idx, col_idx]
            sns.lineplot(x='ratio', y='value', hue = 'type', style='type', data=df_plot, ax=ax,
                          palette='Set2', alpha=0.8, errorbar=None)
            if row_idx == 0:
                ax.set_title(r'$\alpha_{train}$:'+f'{alpha_train}', fontsize=22)
            if col_idx == 0:
                ax.set_ylabel(comp_type)
            ax.set_xlabel('')
            ax.set_xticks(np.arange(0, int(mr)+5, 5))
            ax.set_yticks(np.arange(0.4, 1.0, 0.2))
            ax.tick_params(axis='both', labelsize=14)

            ax.set_xlim(0, int(mr)+5)
            if row_idx == 2 and col_idx == len(alpha_train_values) - 1:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels=['ECF', 'DF'], loc='lower right', fontsize=18, frameon=False)
            else:
                ax.legend().remove()
            ax.grid()
    
    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels = ['ECF', 'DF'], loc='lower center', fontsize=25)
    plt.tight_layout()

    plt.savefig(f'figs/comparisons{mr}-{data_name}.pdf', format='pdf', bbox_inches="tight", dpi=300)



def main():
    parser = argparse.ArgumentParser(description='Make figrue of different metrics')
    parser.add_argument('-m', '--metrics', default='aps', type=str)
    parser.add_argument('-t', '--tag', default='dementia', type=str)
    parser.add_argument('-r', '--mask_rate', default=15, type=int)
    parser.add_argument('-d', '--data_name', default='pitts', type=str)

    args = parser.parse_args()

    #alpha_trains = [0.20, 0.25, 0.33, 0.50, 1.00, 2.00, 3.00, 4.00, 5.00]
    alpha_trains = [0.20, 0.33, 1.00, 3.0, 5.00]
    metric = args.metrics
    mr = args.mask_rate
    data_name = args.data_name

    global_dir = f'/home/sheng136/DeconDTN/code/output/global/global-{data_name}/global_'
    local_dir = f'/home/sheng136/DeconDTN/code/output/local/{data_name}/ratio{mr}/local_'

    num_param = pd.read_csv('/home/sheng136/DeconDTN/code/misc/csv/mask_sizes.csv')
    summary = num_param.groupby(['mask_type', 'model', 'top_x_percent', 'alpha_train'])[['size', 'size_percentage']].mean().reset_index()
    summary['ratio'] = summary['top_x_percent']/100.
    tag = args.tag
    global_df_lst = []
    local_df_lst = []

    for alpha_train in alpha_trains:
        global_df = process_results(alpha_train, dir = global_dir )
        global_df = process_df(global_df)
        global_df = global_df[global_df['ratio'] < 0.41] # keep difference mask monotonic
        global_df = global_df[global_df.model.str.contains(f'{tag}$', regex = True)]
        global_sub_df = global_df[global_df['metric'] == metric]

        
        local_df = process_results(alpha_train, dir = local_dir)
        local_df = local_df[local_df.model.str.contains(f'{tag}$', regex = True)]
        local_sub_df = local_df[local_df['metric'] == metric]
        # only keep first 4 iteration for direct comparison
        local_df = local_df[local_df['iteration']<=4]
        global_df_lst.append(global_sub_df)
        local_df_lst.append(local_sub_df)

        #get global plotdf
    g = reduce(lambda ldf, rdf: pd.concat([ldf, rdf], ignore_index=True), 
                global_df_lst)
    g.to_csv('csv/temp_global_df-1.csv')
    global_df = pd.merge(g[['type', 'ratio', 'alpha_train','value']],
            summary[['mask_type', 'ratio', 'alpha_train', 'size', 'size_percentage']], 
            left_on = ['type', 'ratio', 'alpha_train'], 
            right_on = ['mask_type', 'ratio', 'alpha_train'])

    local_df = reduce(lambda ldf, rdf: pd.concat([ldf, rdf], ignore_index=True), 
                      local_df_lst)
    
    #print(local_df)
    plotdf = join_dfs(local_df, global_df, mr=mr)
    plotdf.to_csv(f'./csv/{data_name}-joined.csv', index=False)

    draw_plot(plotdf, mr=mr, data_name=data_name)

if __name__ == '__main__':
    main()