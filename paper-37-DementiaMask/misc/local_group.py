import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors
import pickle
import pandas as pd
import argparse
import os
import numpy as np
from functools import reduce


def load_pkl(alpha_train, dir='../output/local/'):
    with open(os.path.join(dir,f"ratio15/local_atrain-{alpha_train:.2f}.pkl"), 'rb') as f:
        return pickle.load(f)

def process_results(alpha_train, dir = '../output/local/'):
    res = load_pkl(alpha_train, dir = dir)
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



def draw_group_lineplot(maledf, femaledf, metric='sp'):
    maledf['layer'] = maledf['model'].str.replace('_male', '')
    femaledf['layer'] = femaledf['model'].str.replace('_female', '')

    if metric == 'sp':
        ylabel = 'Statistical Parity'
    elif metric == 'tpr':
        ylabel = 'TPR diff'
    elif metric == 'fpr':
        # |P(y_hat = 1 | female, y = 0) - P(y_hat = 1 | male, y = 0)|
        # that is false positive rate of dementia
        # equivalent to 1 - recall_healthy
        ylabel = 'FPR diff'

    #palette = sns.diverging_palette(150, 275, s=80, l=55, center="dark", as_cmap=True)

    plotdf = pd.merge(maledf, femaledf, on=['layer', 'alpha_train', 'iteration', 'metric'], suffixes=('_male', '_female'))
    plotdf[metric] = np.absolute(plotdf['value_male'] - plotdf['value_female'])
    plotdf['alpha_train'] = plotdf['alpha_train'].astype(str)
    plotdf.to_csv('./csv/local_group.csv', index=False)
    sns.set_style("white")
    sns.set_context("notebook", rc={"grid.linewidth": 0.5})
    sns.lineplot(data=plotdf, x='layer', y=metric, hue='alpha_train', marker='o',  
                 markersize=8, errorbar=None, palette='icefire')

    
    #plt.xlabel('')
    plt.xticks([])
    plt.ylabel(ylabel)
    plt.title('')
    plt.legend(title=r'$\alpha_{train}$', frameon=False)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'figs/local_group_{metric}.pdf', format='pdf', bbox_inches="tight", dpi=300)



def main():
    parser = argparse.ArgumentParser(description='Make figrue of different metrics between groups')
    parser.add_argument('-m', '--metrics', default='sp', type=str, required=True)
    args = parser.parse_args()
    metric_name = args.metrics
    if metric_name == 'sp':
        metric = 'dementia_rate'
    elif metric_name == 'tpr':
        metric = 'recall_dementia'
    elif metric_name == 'fpr':
        # false positive rate of dementia equals 1 - true positive rate of healthy patient
        metric = 'recall_healthy'


    alpha_trains = [0.20, 0.25, 0.33, 0.50, 1.00, 2.00, 3.00, 4.00, 5.00]
    df_lst_male = []
    df_lst_female = []

    for alpha_train in alpha_trains:
        res_df = process_results(alpha_train)
        sub_df_male = res_df[res_df.model.str.contains(f'_male$', regex = True)]
        metric_df_male = sub_df_male[sub_df_male['metric'] == metric]
        df_lst_male.append(metric_df_male)

        sub_df_female = res_df[res_df.model.str.contains(f'_female$', regex = True)]
        metric_df_female = sub_df_female[sub_df_female['metric'] == metric]
        df_lst_female.append(metric_df_female)

    plotdf_male = reduce(lambda ldf, rdf: pd.concat([ldf, rdf], ignore_index=True), df_lst_male)
    plotdf_female = reduce(lambda ldf, rdf: pd.concat([ldf, rdf], ignore_index=True), df_lst_female)
    # alpha train tested
    
    draw_group_lineplot(plotdf_male, plotdf_female, metric=metric_name)

if __name__ == '__main__':
    main()



