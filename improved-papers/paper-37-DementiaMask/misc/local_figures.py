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


def draw_plot(plotdf, metric, data, n_layers=13, rep=10):
    """
    Draw a barplot with the last value highlighted
    Input:
    df: DataFrame with columns ['model', 'iteration', 'metric', 'value']
    n_layers: number of layers (12 for bertbase + 1 embedding ) in the model plus embedding and classifer
    rep: number of repetitions
    """
    def mybar(x, y , **kwargs):
        
        ax = sns.barplot(x=x, y=y, **kwargs)
        # Adjust bar width
        sns.despine(left=True, bottom=False)
        ax.spines['bottom'].set_color('0.1')
        ax.spines['top'].set_color('0.1')



    sns.set_style("whitegrid")
    sns.set_context("poster", rc={"grid.linewidth": 1.0})    

    g = sns.FacetGrid(plotdf, col="alpha_train", col_wrap=5, sharey=True, sharex=True)
    g.map(mybar, 'model', 'value',  alpha = 1.0,
          width=0.5, palette = ['darkred'] + ['darkorange'] + ['skyblue']* n_layers , errwidth=1)
    if metric == 'aps':
        ylabel = 'AUPRC'
    g.set_axis_labels('', f'{ylabel}', fontsize = 16)


    
    for ax in g.axes.flat:
        ax.set_ylabel(ax.get_ylabel(), fontsize=20)
        ax.tick_params(axis='y', labelsize=16)
        #ax.set_xlim(-1, len(plotdf['model'].unique()) - 1)
        #ax.yaxis.grid(True, linestyle='-', color='black', linewidth=0.8)
    g.fig.subplots_adjust(wspace=0.1)  # Reduce the space between the plots

    g.set(xticklabels=[], xlabel = None, ylim=(0.45,0.98))

    g.set_titles(col_template=r"$\alpha_{{train}}$={col_name}", size = 25)
    # Create custom legend patches
    darkred_patch = plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='darkred', markersize=25, label='Intact Model')
    darkorange_patch = plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='darkorange', markersize=25, label='Confounding Filter')

    # Add legend to the plot
    g.add_legend(handles=[darkred_patch, darkorange_patch], 
                 loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5, prop={'size': 20})

    # Adjust the layout
    g.fig.tight_layout(w_pad=1)
    plt.margins(0.8)
    plt.savefig(f'figs/local_{metric}_{data}.pdf', format='pdf', bbox_inches="tight", dpi=300)


def main():
    parser = argparse.ArgumentParser(description='Make figrue of different metrics')
    parser.add_argument('-m', '--metrics', default='aps', type=str, required=True)
    parser.add_argument('-t', '--tag', default='dementia', type=str, required=True)
    parser.add_argument('-d', '--data', default='pitts', type=str)

    args = parser.parse_args()
    metric = args.metrics
    tag = args.tag
    data_name = args.data
    # alpha train tested
    #alpha_trains = [0.20, 0.25, 0.33, 0.50, 1.00, 2.00, 3.00, 4.00, 5.00]
    alpha_trains = [0.20, 0.33, 1.00, 3.00, 5.00]
    df_lst = []
    dir = f"../output/local/{data_name}"
    for alpha_train in alpha_trains:
        res_df = process_results(alpha_train, dir)
        sub_df = res_df[res_df.model.str.contains(f'{tag}$', regex = True)]
        metric_df = sub_df[sub_df['metric'] == metric]
        df_lst.append(metric_df)
    
    plotdf = reduce(lambda ldf, rdf: pd.concat([ldf, rdf], ignore_index=True), df_lst)
    plotdf.to_csv(f'./csv/{metric}_{data_name}_local.csv')
    draw_plot(plotdf, metric,data_name)



if __name__ == '__main__':
    main()



