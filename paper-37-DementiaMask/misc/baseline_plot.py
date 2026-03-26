import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os
import pickle

def plot_baseline(plotdf, data_name, atrain):
    sns.set_style("white")
    #sns.set_context("poster", rc={"grid.linewidth": 1.0}) 

    # Create the line plot
    # plotdf['size'] = np.where(((plotdf['type'] == 'ConGater')|(plotdf['type'] == 'ModDiffy')), 100, 60).astype(float)
    lineplot = sns.scatterplot(data=plotdf, x='fpr', y='aps', 
                               hue='type', style='type', markers=True, 
                               palette='Dark2',s = 120,  alpha = 0.8, legend='brief',
                               hue_order=[ "DF(all)", "DF(intersection)",
                                      "DF(difference)", "ECF", "CF",
                                      "ConGater", "ModDiffy"])

    # Set plot labels and title
    lineplot.set_title('', fontsize=16)
    lineplot.set_xlabel('FPR Difference', fontsize=16)
    lineplot.set_ylabel('AUPRC', fontsize=16)
    lineplot.grid()
    lineplot.legend(title = 'Method', frameon=True, fontsize=11, title_fontsize=12,
                     handles=lineplot.legend_.legendHandles[:len(plotdf['type'].unique())])
    plt.xticks(fontsize=10)
    # lineplot.legend().set_visible(False)
    # Show the plot
    plt.savefig(f'/home/sheng136/DeconDTN/code/misc/figs/{data_name}_baseline_{atrain}.pdf', 
               format='pdf', bbox_inches="tight", dpi=300)

def main():
    parser = argparse.ArgumentParser(description='Plot baseline results')
    parser.add_argument('-d', '--data', type=str, default='', help='baseline to plot')
    parser.add_argument('-alpha', '--alpha_train', type=float, default=3.0)

    args = parser.parse_args()
    data_name = args.data
    atrain = args.alpha_train

    aps_df = pd.read_csv(f'/home/sheng136/DeconDTN/code/misc/csv/{data_name}-joined.csv')
    fpr_df = pd.read_csv(f'/home/sheng136/DeconDTN/code/misc/csv/{data_name}-joined_group.csv')


    aps_df = aps_df[aps_df['alpha_train'] == atrain].rename(columns={'value': 'aps'})
    fpr_df = fpr_df[fpr_df['alpha_train'] == atrain].rename(columns={'value': 'fpr'})

    fprdf = fpr_df.groupby(['type', 'ratio'])['fpr'].mean().reset_index()
    aps_df = aps_df.groupby(['type', 'ratio'])['aps'].mean().reset_index()
    plotdf = pd.merge(fprdf, aps_df, on=['type', 'ratio'], how = 'left')
    plotdf.type = plotdf.type.map({'all':'DF(all)', 'intersection': 'DF(intersection)',
                                'difference': 'DF(difference)', 'ecf': 'ECF'})
    # add baseline
    # 1. congater
    congater = pd.read_csv(f'/home/sheng136/DeconDTN/code/baseline/data/gater_evaluation_{data_name}.csv')
    congaterdf = congater[['w_eval_task_auprc', 'w_eval_task_fprdiff']]
    congaterdf['type'] = 'ConGater'
    congaterdf.rename(columns={'w_eval_task_auprc': 'aps', 'w_eval_task_fprdiff': 'fpr'}, inplace=True)
    plotdf = pd.concat([plotdf, congaterdf], axis = 0, ignore_index=True)

    # 2. moddiffy
    # prerecorded results
    if data_name == 'pitts':
        apr_mod = [0.7592592592592593]
        fpr_mod = [0.07272727272727275]
    elif data_name == 'ccc':
        apr_mod = [0.6647058823529413, 0.8836734693877552]
        fpr_mod = [0.48, 0.46153846153846156]
    plotdf = pd.concat([plotdf, pd.DataFrame({'type': ['ModDiffy']*len(apr_mod), 'aps': apr_mod, 'fpr': fpr_mod})], axis = 0, ignore_index=True)
    
    # confounding filter
    def get_plotdf(cf_dir, metric, suffix=''):
        cf = pd.DataFrame()
        for alpha_train in [3.00]:
            with open(os.path.join(cf_dir,f"CF_local_atrain-{alpha_train:.2f}.pkl"), 'rb') as f:
                res = pickle.load(f)
            masked_params = list(res.keys())
            result_dict = {}
            for param in masked_params:
                result_dict[param] = res[param][f'classifier.masked_dementia{suffix}'][metric]
                result_dict[0] = res[param][f'orig_dementia{suffix}'][metric]
            df = pd.DataFrame(result_dict)
            df_melted = df.reset_index().melt(id_vars='index', var_name='masked_num', value_name='value')
            df_melted = df_melted.rename(columns={'index': 'Repeat'})
            df_melted['masked_ratio'] = df_melted['masked_num'].apply(lambda x: float(x)/1536)
            df_melted['alpha_train'] = alpha_train
            cf = pd.concat([cf, df_melted], ignore_index=True)
        return cf
    cf_dir = f'/home/sheng136/DeconDTN/code/output/local/{data_name}'
    aps_df = get_plotdf(cf_dir, 'aps') # AUPRC
    male_recall_df = get_plotdf(cf_dir, 'recall_healthy', '_male')
    female_recall_df = get_plotdf(cf_dir, 'recall_healthy', '_female')
    fairness_df = pd.merge(male_recall_df, female_recall_df, on=['Repeat', 'masked_num', 'masked_ratio', 'alpha_train'])
    fairness_df['fpr'] = np.abs(fairness_df['value_x'] - fairness_df['value_y'])
    aps_df = aps_df.groupby(['masked_ratio', 'alpha_train'])['value'].mean().reset_index().rename(columns={'value': 'aps'})
    fairness_df = fairness_df.groupby(['masked_ratio', 'alpha_train'])['fpr'].mean().reset_index()
    cf_df = pd.merge(aps_df, fairness_df, on=['masked_ratio', 'alpha_train'], how = 'left')
    cf_df['type'] = 'CF'
    # quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # quantile_values = cf_df.masked_ratio.quantile(quantiles).values
    # closest_in_dat = [min(sorted(cf_df.masked_ratio), key=lambda x: abs(x - q)) for q in quantile_values]
    # cf_df = cf_df[cf_df.masked_ratio.isin(closest_in_dat)]

    plotdf = pd.concat([plotdf, cf_df], axis = 0, ignore_index=True)
    
    plot_baseline(plotdf, data_name, atrain)

if __name__ == '__main__':
    main()