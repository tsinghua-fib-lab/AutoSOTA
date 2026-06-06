import pickle
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
from statannotations.Annotator import Annotator
import numpy as np



def flat_result(res, setting):
    flattened_data = []
    for idx, data_dict in enumerate(res):
        for key in ['overall', 'male', 'female']:
            for metric, values in data_dict[key].items():
                for value in values:
                    flattened_data.append({'category': key, 'metric': metric, 'value': value})

    # Create DataFrame
    df = pd.DataFrame(flattened_data).assign(exp=setting)

    return df

def make_boxplot(df, metric, data):
    df = df[df['metric'] == metric]
    palette = sns.color_palette("Set2")[2:]
    sns.set_style("whitegrid")
    #sns.set_context("poster", rc={"grid.linewidth": 0.5})    
    hue_order=['original', 'balanced']
    order = ['overall', 'male', 'female']
    #plt.figure(figsize=(6, 4))
    ax = sns.boxplot(x='category', y='value', hue='exp', data=df, palette=palette, 
                     showfliers=False, width=0.6, hue_order=hue_order, order=order)
    
    
    # print group mean difference for table 1
    # print('orig:', df[(df['category'] == 'male') & (df['exp']=='original')]['value'].mean() - df[(df['category'] == 'female') & (df['exp']=='original')]['value'].mean())
    # print('balanced:', df[(df['category'] == 'male') & (df['exp']=='balanced')]['value'].mean() - df[(df['category'] == 'female') & (df['exp']=='balanced')]['value'].mean())
    # exit()

    # Customize boxplot to remove edges
   
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[2:], labels=labels[2:])
    leg = plt.legend(fontsize=20)
    leg.get_frame().set_edgecolor('black')
    ax.spines['top'].set_visible(True)
    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_color('black')

    sns.despine(left=True, top=False)

    pairs=[(("male", "balanced"), ("female", "balanced")), 
           (("male", "original"), ("female", "original"))]

    annot = Annotator(ax, pairs, data=df, x='category', y='value', order=order, hue='exp', hue_order=hue_order)
    annot.new_plot(ax, pairs, data=df, x='category', y='value', order=order, hue='exp', hue_order=hue_order)
    annot.configure(test='Mann-Whitney', text_format='star', loc='inside')
    annot.apply_and_annotate()

    # plt.text(0.05, 0.05, 'ns: p > 5e-02\n*: p <= 5e-02',
    #     horizontalalignment='left',  
    #     verticalalignment='bottom',  
    #     fontsize=11,                  
    #     transform=plt.gca().transAxes) 

    # Adding labels and title
    plt.xlabel('', fontsize=12)
    plt.xticks(fontsize=14)
    plt.ylabel(metric, fontsize=14)
    plt.legend(title='', fontsize=14)
    plt.title('')
    plt.tight_layout()

    plt.savefig(f'figs/bert-raw_disparity-{metric}-{data}.pdf', format = 'pdf', bbox_inches="tight", dpi=300)

def main():
    parser = argparse.ArgumentParser(description='Train a binary classification model use bert-base')
    parser.add_argument('--metric', type=str, default='accuracy', help='metric to plot')
    parser.add_argument('--data', type=str, default='pitts', help='data to plot')
    parser.add_argument('--model', type=str, default='bert-base-uncased', help='model to plot')
    args = parser.parse_args()

    with open(f'../output/raw_cv/results_{args.data}_{args.model}_sample0.pkl', 'rb') as f:
        no_subsample=pickle.load(f)
    with open(f'../output/raw_cv/results_{args.data}_{args.model}_sample1.pkl', 'rb') as f:
        subsample=pickle.load(f)

    subdf = flat_result(subsample, 'balanced')
    no_subdf = flat_result(no_subsample, 'original')

    plotdf = pd.concat([subdf, no_subdf], ignore_index=True)
    print(plotdf[plotdf['metric']==args.metric].reset_index())
    # Filter the DataFrame for the desired metric (e.g., f1)

    make_boxplot(plotdf, metric=args.metric, data=args.data)

if __name__ == '__main__':
    main()