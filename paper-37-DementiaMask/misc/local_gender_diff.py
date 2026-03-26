import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors
import pickle
import pandas as pd
import argparse
import os
import numpy as np
from functools import reduce
import argparse


def load_pkl(alpha_train, dir='../output/local/'):
    with open(os.path.join(dir,f"ratio15/local_atrain-{alpha_train:.2f}.pkl"), 'rb') as f:
        return pickle.load(f)

def process_results(alpha_train, dir='../output/local/'):
    res = load_pkl(alpha_train, dir)
    flattened_data = []
    for key, metrics in res.items():
        for metric, values in metrics.items():
            for i, value in enumerate(values):
                flattened_data.append({'model': key, 'iteration': i, 'metric': metric, 'value': value})

    # Create DataFrame
    df = pd.DataFrame(flattened_data)
    df = df.assign(alpha_train=alpha_train)
    #df = get_fpr(df)
    return df


def main():
    #alpha_trains = [0.20, 0.25, 0.33, 0.50, 1.00, 2.00, 3.00, 4.00, 5.00]
    alpha_trains = [0.20, 0.33, 1.00, 3.00, 5.00]
    parser = argparse.ArgumentParser(description='Make figrue for gender difference')
    parser.add_argument('-m', '--metrics', default='aps', type=str, required=True)
    parser.add_argument('-d', '--data', default='pitts', type=str, required=True)
    args = parser.parse_args()

    metric = args.metrics
    data_name = args.data
    dir = f"../output/local/{data_name}"

    msk_df_lst = []
    orig_df_lst = []
    for alpha_train in alpha_trains:
        res_df = process_results(alpha_train, dir=dir)
        sub_df1 = res_df[res_df.model.str.contains(f'masked_gender$', regex = True)]
        sub_df2 = res_df[res_df.model.str.contains(f'orig_gender$', regex = True)]

        mask_df = sub_df1[sub_df1['metric'] == metric]
        orig_df = sub_df2[sub_df2['metric'] == metric]
        msk_df_lst.append(mask_df)
        orig_df_lst.append(orig_df)

    masked_gender = reduce(lambda ldf, rdf: pd.concat([ldf, rdf], ignore_index=True), msk_df_lst)
    masked_gender['model'] = masked_gender['model'].str.split('.').str[0]

    orig_gender = reduce(lambda ldf, rdf: pd.concat([ldf, rdf], ignore_index=True), orig_df_lst)
    orig_gender['model'] = orig_gender['model'].str.split('.').str[0]

    gender_roc = pd.merge(orig_gender,masked_gender, on = ['model', 'alpha_train', 'iteration'], suffixes=['_orig', '_mask'] )
    gender_roc['met_diff'] = gender_roc['value_orig'] - gender_roc['value_mask']

    sns.set_style("white", {'axes.grid' : True})
    sns.set_context("notebook", rc={"grid.linewidth": 0.5})
    g = sns.FacetGrid(gender_roc, col="alpha_train", col_wrap=2, sharey=True, sharex=True)
    g.map(sns.lineplot, 'model', 'met_diff',  color = 'skyblue', alpha = 1.0, errorbar=None)
    for ax in g.axes.flatten(): # Loop directly on the flattened axes 
        for _, spine in ax.spines.items():
            spine.set_visible(True) # You have to first turn them on
            spine.set_color('black')
            spine.set_linewidth(1)

    def add_hline(*args, **kwargs):
        plt.axhline(y=0, color='red', linestyle='--')  # Example line at y=6

    # Apply the function to each subplot
    g.map(add_hline)

    g.set_axis_labels('', f'{metric.upper()} diff', fontsize = 14)
    g.set(xticklabels=[], xlabel = None)
    g.set_titles(col_template=r"$\alpha_{{train}}$={col_name}", size = 16)

    plt.savefig(f'figs/gender_diff_{data_name}_{metric}.pdf', bbox_inches='tight', dpi=300, format='pdf')

if __name__ == "__main__":
    main()