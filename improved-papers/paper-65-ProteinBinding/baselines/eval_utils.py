import pandas as pd
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import math
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score, average_precision_score
from scipy.stats import ttest_rel

THRESHOLD=10
BOOTSTRAP=300
def compute_metrics(df, bootstrap=True):
    ddG = df['ddG'].to_numpy()
    ddG_pred = df['ddG_pred'].to_numpy()
    sp, _ = spearmanr(ddG_pred, ddG)
    pr, _ = pearsonr(ddG_pred, ddG)
    pred = df['ddG_pred'] < 0
    real = df['ddG'] < 0

    per_ppi = df.groupby('#Pdb')
    sp_sum, pr_sum = 0.0, 0.0
    rmse_sum = 0.0
    num_ppis = 0
    sp_array = []
    pr_array = []
    rmse_array = []

    for name, group in per_ppi:
        ddG = group['ddG'].to_numpy()
        ddG_pred = group['ddG_pred'].to_numpy()
        if ddG.shape[0] < THRESHOLD: continue
        try:
            sp, _ = spearmanr(ddG_pred, ddG)
            pr, _ = pearsonr(ddG_pred, ddG)
            rmse = np.sqrt(np.mean((ddG - ddG_pred) ** 2))
            sp_array.append(sp)
            pr_array.append(pr)
            rmse_array.append(rmse)
            sp_sum += sp
            pr_sum += pr
            rmse_sum += rmse
            num_ppis += 1

        except Exception as e:
            print(e)
            print(name)
            continue
        bootstrap_spearman =[]

    bootstrap_pearson = []
    bootstrap_rmse = []
    bootstrap_auroc = []
    bootstrap_iter = BOOTSTRAP if bootstrap else 2

    for _ in tqdm(range(bootstrap_iter)):
        group_keys = list(per_ppi.groups.keys())
        N = len(group_keys)

        # Sample N keys with replacement
        sampled_keys = np.random.choice(group_keys, size=N, replace=True)

        # Collect and concatenate the sampled groups
        sampled_dfs = [per_ppi.get_group(k) for k in sampled_keys]
        bootstrapped_df = pd.concat(sampled_dfs, ignore_index=True)
        bootstrap_spearman.append(bootstrapped_df['ddG'].corr(bootstrapped_df['ddG_pred'], method='spearman'))
        bootstrap_pearson.append(bootstrapped_df['ddG'].corr(bootstrapped_df['ddG_pred'], method='pearson'))
        bootstrap_rmse.append(math.sqrt((bootstrapped_df['ddG'] - bootstrapped_df['ddG_pred']).pow(2).mean()))
        roc_auc = roc_auc_score(bootstrapped_df['ddG'] < 0, -bootstrapped_df['ddG_pred'])
        if roc_auc is not np.nan:
            bootstrap_auroc.append(roc_auc)

    metrics = {
        'Spearman': df['ddG'].corr(df['ddG_pred'], method='spearman'),
        'Spearman Standard Error': np.std(bootstrap_spearman),
        'Per Structure Spearman': sp_sum/num_ppis,
        'Per Structure Spearman Standard Error': np.std(sp_array)/np.sqrt(num_ppis),
        'Pearson': df['ddG'].corr(df['ddG_pred'], method='pearson'),
        'Pearson Standard Error': np.std(bootstrap_pearson),
        'Per Structure Pearson': pr_sum/num_ppis,
        'Per Structure Pearson Standard Error': np.std(pr_array)/np.sqrt(num_ppis),
        'RMSE': math.sqrt((df['ddG'] - df['ddG_pred']).pow(2).mean()),
        'RMSE Standard Error': np.std(bootstrap_rmse),
        'Per Structure RMSE': rmse_sum/num_ppis,
        'Per Structure RMSE Standard Error': np.std(rmse_array)/np.sqrt(num_ppis),
        'MAE': (df['ddG'] - df['ddG_pred']).abs().mean(),
        'Precision': precision_score(real, pred, zero_division=0),
        'Recall': recall_score(real, pred, zero_division=0),
        'ROC AUC': roc_auc_score(real, -df['ddG_pred']) if len(df) and real.nunique() > 1 else np.nan,
        'ROC AUC Standard Error': np.std(bootstrap_auroc),
        'PR AUC': average_precision_score(real, -df['ddG_pred']) if len(df) and real.nunique() > 1 else np.nan,
    }
    return metrics

def latex_table_format(metric, name):
    """
    Format a metrics dict into a LaTeX table row with per‐structure ± stderr
    and overall metrics including ROC AUC.
    """
    # Per‐structure values ± standard error
    per_p   = metric['Per Structure Pearson']
    per_p_se = metric['Per Structure Pearson Standard Error']
    per_s   = metric['Per Structure Spearman']
    per_s_se = metric['Per Structure Spearman Standard Error']
    per_r   = metric['Per Structure RMSE']
    per_r_se = metric['Per Structure RMSE Standard Error']
    # Overall values
    over_p   = metric['Pearson']
    over_p_se = metric['Pearson Standard Error']
    over_s   = metric['Spearman']
    over_s_se = metric['Spearman Standard Error']
    over_r   = metric['RMSE']
    over_r_se = metric['RMSE Standard Error']
    over_roc = metric['ROC AUC']
    over_roc_se = metric['ROC AUC Standard Error']
    # Build the LaTeX row
    return (
        f"{name}  & "
        f"{per_p:.2f} $\\pm$ {per_p_se:.2f}  & "
        f"{per_s:.2f} $\\pm$ {per_s_se:.2f}  & "
        f"{per_r:.2f} $\\pm$ {per_r_se:.2f}  & "
        f"{over_p:.2f} $\\pm$ {over_p_se:.2f} & "
        f"{over_s:.2f}  $\\pm$ {over_s_se:.2f} & "
        f"{over_r:.2f}  $\\pm$ {over_r_se:.2f} & "
        f"{over_roc:.2f} $\\pm$ {over_roc_se:.2f} \\\\"
    )

def struct_metrics(df):
    ddG = df['ddG'].to_numpy()
    ddG_pred = df['ddG_pred'].to_numpy()
    per_ppi = df.groupby('#Pdb')
    sp_array = []
    pr_array = []
    rmse_array = []

    for name, group in per_ppi:
        ddG = group['ddG'].to_numpy()
        ddG_pred = group['ddG_pred'].to_numpy()
        if ddG.shape[0] < THRESHOLD: continue
        try:
            sp, _ = spearmanr(ddG_pred, ddG)
            pr, _ = pearsonr(ddG_pred, ddG)
            rmse = np.sqrt(np.mean((ddG - ddG_pred) ** 2))
            sp_array.append(sp)
            pr_array.append(pr)
            rmse_array.append(rmse)

        except Exception as e:
            print(e)
            print(name)
            continue

    return pr_array, sp_array, rmse_array

def t_test(df1, df2, one_sided=True):
    df1_names = df1['#Pdb'].value_counts()[df1['#Pdb'].value_counts() >= THRESHOLD].index.tolist()
    df2_names = df2['#Pdb'].value_counts()[df2['#Pdb'].value_counts() >= THRESHOLD].index.tolist()
    both = set([x for x in df1_names if x in df2_names])
    df1 = df1[df1['#Pdb'].isin(both)]
    df2 = df2[df2['#Pdb'].isin(both)]
    pr1, sp1, rmse1 = struct_metrics(df1)
    pr2, sp2, rmse2 = struct_metrics(df2)
    
    if one_sided:
        _, pr_pval = ttest_rel(pr1, pr2, alternative='greater')
        _, sp_pval = ttest_rel(sp1, sp2, alternative='greater')
        _, rmse_pval = ttest_rel(rmse1, rmse2, alternative='less')
    else:
        _, pr_pval = ttest_rel(pr1, pr2)
        _, sp_pval = ttest_rel(sp1, sp2)
        _, rmse_pval = ttest_rel(rmse1, rmse2)
        
    return pr_pval, sp_pval, rmse_pval

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate binding ddG predictions")
    parser.add_argument("--csv", type=str, required=True, help="Path to prediction csv")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(compute_metrics(df))