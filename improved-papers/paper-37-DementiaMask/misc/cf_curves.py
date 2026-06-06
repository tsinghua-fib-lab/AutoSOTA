import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from functools import reduce
import numpy as np
import os

def get_plotdf(cf_dir, metric, suffix=''):
    cf = pd.DataFrame()
    for alpha_train in [0.20, 0.33, 1.00, 3.00, 5.00]:
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

def main():
    cf_dir = '/home/sheng136/DeconDTN/code/output/local/pitts'
    aps_df = get_plotdf(cf_dir, 'aps') # AUPRC
    male_recall_df = get_plotdf(cf_dir, 'recall_healthy', '_male')
    female_recall_df = get_plotdf(cf_dir, 'recall_healthy', '_female')

    
    # subset
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    quantile_values = aps_df.masked_ratio.quantile(quantiles)
    aps_df = aps_df[aps_df.masked_ratio.isin(quantile_values)]

    
    ax = sns.lineplot(x="masked_ratio", y='value', data=aps_df, hue = "alpha_train",
                            linewidth = 2, palette='tab10', alpha=0.8, errorbar=None, marker = 'o',
                            markerfacecolor='white', markeredgecolor='black', markersize = 4)

    ax.set_xlabel('% of abalated weights in classification layer', fontsize=16)
    ax.set_ylabel('AUPRC', fontsize=16)
    ax.tick_params(axis='both', labelsize=16)
    ax.legend(title = r'$\alpha_{train}$', fontsize=14, title_fontsize=14, frameon=False)
    ax.grid()

    plt.savefig(f'/home/sheng136/DeconDTN/code/misc/figs/cf_curves.pdf', 
                format='pdf', bbox_inches="tight", dpi=300)
    plt.close()

    fairness_df = pd.merge(male_recall_df, female_recall_df, on=['Repeat', 'masked_num', 'masked_ratio', 'alpha_train'])
    fairness_df['value'] = np.abs(fairness_df['value_x'] - fairness_df['value_y'])
    fairness_df = fairness_df[fairness_df.masked_ratio.isin(quantile_values)]

    
    ax1 = sns.lineplot(x="masked_ratio", y='value', data=fairness_df, hue = "alpha_train",
                            linewidth = 2, palette='tab10', alpha=0.8, errorbar=None, marker = 'o',
                            markerfacecolor='white', markeredgecolor='black', markersize = 4)

    ax1.set_xlabel('% of abalated weights in classification layer', fontsize=16)
    ax1.set_ylabel('FPR Diff', fontsize=16)
    ax1.tick_params(axis='both', labelsize=16)
    ax1.legend(title = r'$\alpha_{train}$', fontsize=14, title_fontsize=14, frameon=False)
    ax1.grid()

    plt.savefig(f'/home/sheng136/DeconDTN/code/misc/figs/cf_fair.pdf', 
                format='pdf', bbox_inches="tight", dpi=300)
    

if __name__ == '__main__':
    main()