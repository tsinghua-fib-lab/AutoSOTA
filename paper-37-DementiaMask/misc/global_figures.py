import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors
import pickle
import pandas as pd
import argparse
import os
import numpy as np
from functools import reduce
import re


def extract_ratio(text):
    match = re.search(r'\d+\.\d+', text)
    return match.group() if match else None

def extract_type(text):
    match = re.search(r'^(.+?)\.', text)
    return match.group(1) if match else None


def load_pkl(alpha_train, dir='../output/global'):
    with open(os.path.join(dir,f"global_atrain-{alpha_train:.2f}.pkl"), 'rb') as f:
        return pickle.load(f)
    
def process_df(res_df):
    res_df['ratio'] = res_df['model'].apply(extract_ratio).astype(float)
    res_df['type'] = res_df['model'].apply(extract_type)
    res_df = res_df.dropna()
    res_df['alpha_train'] = res_df['alpha_train'].astype(float)
    res_df['ratio'] = res_df['ratio'].astype(float)

    return res_df
    
def process_results(alpha_train, dir):
    res = load_pkl(alpha_train,dir)
    flattened_data = []
    for key, metrics in res.items():
        for metric, values in metrics.items():
            for i, value in enumerate(values):
                flattened_data.append({'model': key, 'iteration': i, 'metric': metric, 'value': value})

    # Create DataFrame
    df = pd.DataFrame(flattened_data)
    df = df.assign(alpha_train=alpha_train)
    df = df.iloc[::-1].reset_index(drop=True)
    return df

def draw_group_lineplot(maledf, femaledf, summary, ylab, metric='sp', window_size=1):
    maledf['model'] = maledf['model'].str.replace('_male', '')
    femaledf['model'] = femaledf['model'].str.replace('_female', '')


    plotdf = pd.merge(maledf, femaledf, on=['model', 'alpha_train', 'iteration', 'type', 'ratio', 'metric'], suffixes=('_male', '_female'))

    plotdf[metric] = np.absolute(plotdf['value_male'] - plotdf['value_female'])
    plotdf = pd.merge(plotdf[['type', 'ratio', 'alpha_train',metric]], summary[['mask_type', 'ratio', 'alpha_train', 'size', 'size_percentage']], 
         left_on = ['type', 'ratio', 'alpha_train'], right_on = ['mask_type', 'ratio', 'alpha_train'])
    
    plotdf.type = plotdf.type.where(plotdf.type != 'compliment', 'difference')
    plotdf['alpha_train'] = plotdf['alpha_train'].astype(str)
    plotdf['svalue'] = plotdf[metric].rolling(window=window_size, center=True).mean()

    plotdf.to_csv('./csv/global_group.csv', index=False)

    sns.set_style("white", {'axes.grid' : True})
    sns.set_context("notebook", rc={"grid.linewidth": 0.5})
    g = sns.relplot(
    data=plotdf,
    x="size_percentage", y='svalue',
    hue="alpha_train", col="type",
    alpha = 0.8, linewidth = 3, markers=True, markersize = 4,
    kind="line", palette='icefire', errorbar = None,
    height=5, aspect=1., facet_kws=dict(sharex=False),
)
    g.set_axis_labels('Zero Out Ratio (%)', ylab)
    g.set(xlim = (1, 15.5))

    x_limits = {
     'difference': (1, 10.5),
}
    # for ax in g.axes.flatten():
    #     mtype = ax.get_title().split('=')[1].strip()
    #     if mtype == 'difference':
    #         ax.set_xlim(x_limits.get(mtype, (0, 1))) 

    plt.savefig(f'figs/global_group_{metric}.pdf', bbox_inches='tight', dpi=300, format='pdf')

def draw_global_type(plotdf, metric, tag, window_size=1):
    
    plotdf.type = plotdf.type.where(plotdf.type != 'compliment', 'difference')
    # Ensure correct data types
    plotdf['alpha_train'] = plotdf['alpha_train'].astype(str)
    plotdf['ratio'] = plotdf['ratio'].astype(float)
    # Apply moving average for smoothing
    plotdf['svalue'] = plotdf['value'].rolling(window=window_size, center=True).mean()

    # Set style and context for the plot
    sns.set_style("white",{'axes.grid' : True})
    sns.set_context("notebook", rc={"grid.linewidth": 0.5, "lines.linewidth": 1.5})


    # Create the FacetGrid

    # Map the lineplot
    g = sns.relplot(
    data=plotdf,
    x="size_percentage", y="svalue",
    hue="alpha_train", col="type",
    alpha = 0.8, linewidth = 3,
    kind="line", palette='icefire', errorbar = None,
    height=5, aspect=1., facet_kws=dict(sharex=False), legend='auto',
)
    g.set_axis_labels('Zero Out Ratio (%)', f'{metric}')
    g.set(xlim = (1, 15.5))
#     x_limits = {
#      'difference': (1, 15),
# }
#     for ax in g.axes.flatten():
#         mtype = ax.get_title().split('=')[1].strip()
#         if mtype == 'difference':
#             ax.set_xlim(x_limits.get(mtype, (0, 1))) 
    plt.savefig(f'figs/global_filter_{tag}-{metric}.pdf', bbox_inches='tight', dpi=300, format='pdf')



def main():
    parser = argparse.ArgumentParser(description='Make figrue of different metrics')
    parser.add_argument('-m', '--metrics', default='aps', type=str)
    parser.add_argument('-g', '--group_metrics', default='sp', type=str)
    parser.add_argument('-t', '--tag', default='dementia', type=str)
    parser.add_argument('-i', '--inputdir', default='/home/sheng136/DeconDTN/code/output/global-a0', type=str)

    args = parser.parse_args()

    #alpha_trains = [0.20, 0.25, 0.33, 0.50, 1.00, 2.00, 3.00, 4.00, 5.00]
    alpha_trains = [0.20, 0.33, 1.00, 3.0, 5.00]
    metric = args.metrics
    group_metric = args.group_metrics

    if group_metric == 'sp':
        group_metric_name = 'dementia_rate'
        ylab = 'Statistical Parity'
    elif group_metric == 'tpr':
        group_metric_name = 'recall_dementia'
        ylab = 'True Positive Rate Diff'
    elif group_metric == 'fpr':
        # false positive rate of dementia equals 1 - true positive rate of healthy patient
        group_metric_name = 'recall_healthy'
        ylab = 'False Positive Rate Diff'



    num_param = pd.read_csv('/home/sheng136/DeconDTN/code/misc/csv/mask_sizes.csv')
    summary = num_param.groupby(['mask_type', 'model', 'top_x_percent', 'alpha_train'])[['size', 'size_percentage']].mean().reset_index()
    summary['ratio'] = summary['top_x_percent']/100

    tag = args.tag
    global_df_lst = []
    df_lst_male = []
    df_lst_female = []
    for alpha_train in alpha_trains:
        res_df = process_results(alpha_train, dir = args.inputdir)
        res_df = process_df(res_df)
        res_df = res_df[res_df['ratio'] < 0.42]
        sub_df = res_df[res_df.model.str.contains(f'{tag}$', regex = True)]
        male_subdf = res_df[res_df.model.str.contains(f'_male$', regex = True)]
        female_subdf = res_df[res_df.model.str.contains(f'_female$', regex = True)]

        metric_df = sub_df[sub_df['metric'] == metric]
        global_df_lst.append(metric_df)

        male_df  = male_subdf[male_subdf['metric'] == group_metric_name]
        df_lst_male.append(male_df)

        female_df  = female_subdf[female_subdf['metric'] == group_metric_name]
        df_lst_female.append(female_df)

    g = reduce(lambda ldf, rdf: pd.concat([ldf, rdf], ignore_index=True), global_df_lst)

    global_df = pd.merge(g[['type', 'ratio', 'alpha_train','value']],
                summary[['mask_type', 'ratio', 'alpha_train', 'size', 'size_percentage']], 
                left_on = ['type', 'ratio', 'alpha_train'], 
                right_on = ['mask_type', 'ratio', 'alpha_train'])
    
    draw_global_type(global_df, metric=metric, tag=tag, window_size = 1)

    # plot between group metrics
    mdf = reduce(lambda ldf, rdf: pd.concat([ldf, rdf], ignore_index=True), df_lst_male)
    fdf = reduce(lambda ldf, rdf: pd.concat([ldf, rdf], ignore_index=True), df_lst_female)

    draw_group_lineplot(mdf, fdf, summary=summary, metric=group_metric_name, ylab=ylab, window_size=1)
    

    # three sets trajectories
    #make_masks_fig(num_param)
   

if __name__ == '__main__':
    main()