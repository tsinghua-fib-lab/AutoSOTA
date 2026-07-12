from utils import *
import pandas as pd
import numpy as np
import warnings
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")
# read the gene data
scores = load_single_cell_line('CRISPR_gene_effect.csv', cell_line_index=5)
scores.index = [x.split(' ')[0] for x in scores.index]
# label the data
df_labeled = build_labeled_dataframe(scores, FILE_CEG, FILE_NEG)
    
scores = df_labeled['Score']
df_labels = df_labeled[['Label']]
nseeds = 100
alpha = 0.2
# The test_size in the paper is set 3/4. One can tune the hyperparameter.
test_size = 3/4
all_results_records = []
alphalist = [0.05, 0.1, 0.2, 0.3]
for alpha in alphalist:
    for seed in range(1, 1+nseeds):
        np.random.seed(seed)
        p_values, e_values, test_labels = sample_split_testing(scores, df_labels,test_size, seed)
        
        rej_bh = BH(p_values, alpha)
        if len(rej_bh) > 0:
            max_p_local_idx = np.argmax(p_values[rej_bh])
            bh_boundary_index = rej_bh[max_p_local_idx]
            if test_labels[bh_boundary_index] == 0:
                is_bh_boundary = 1
            else:
                is_bh_boundary = 0
        else:
            bh_boundary_index = []
        
        rej_dominop, dominop_boundary_index, _ = harmonicpbfdr(p_values, alpha)
        # 2,3-domino_pbfdr
        
        reject_indices_2domino, boundary_indices_2domino = bfdr_k_domino(p_values, k=2, alpha=alpha)
        reject_indices_3domino, boundary_indices_3domino = bfdr_k_domino(p_values, k=3, alpha=alpha)
        
        # domino_ebfdr
        rej_dominope, dominoe_boundary_index = bfdr_k_eclosure(e_values, k=1, alpha = alpha)
        rej_2dominope, domino2e_boundary_index = bfdr_k_eclosure(e_values, k=2, alpha = alpha)
        rej_3dominope, domino3e_boundary_index = bfdr_k_eclosure(e_values, k=3, alpha = alpha)
        
        # sl method
        rej_sl, sl_boundary_index, _ = sl_procedure(p_values, alpha)
        
        # is_bh_boundary = kbfdr_evaluate(test_labels, rej_bh, k=1)
        is_boundary_false_2domino = kbfdr_evaluate(test_labels, boundary_indices_2domino, k=2)
        is_boundary_false_3domino = kbfdr_evaluate(test_labels, boundary_indices_3domino, k=3)
        is_dominop_boundary = kbfdr_evaluate(test_labels, dominop_boundary_index, k=1)
        is_sl_boundary = kbfdr_evaluate(test_labels, sl_boundary_index, k=1)
        is_dominoe_boundary = kbfdr_evaluate(test_labels, dominoe_boundary_index, k=1)
        is_domino2_boundary = kbfdr_evaluate(test_labels, domino2e_boundary_index, k=2)
        is_domino3_boundary = kbfdr_evaluate(test_labels, domino3e_boundary_index, k=3)

        # fdr and power
        bh_fdr, bh_power = evaluate_fdrpower(rej_bh, test_labels)
        dominop_fdr, dominop_power = evaluate_fdrpower(rej_dominop, test_labels)
        dominope_fdr, dominope_power = evaluate_fdrpower(rej_dominope, test_labels)
        sl_fdr, sl_power = evaluate_fdrpower(rej_sl, test_labels)
        domino2_fdr, domino2_power = evaluate_fdrpower(reject_indices_2domino, test_labels)
        domino3_fdr, domino3_power = evaluate_fdrpower(reject_indices_3domino, test_labels)
        domino2e_fdr, domino2e_power = evaluate_fdrpower(rej_2dominope, test_labels)
        domino3e_fdr, domino3e_power = evaluate_fdrpower(rej_3dominope, test_labels)

        all_results_records.append({
            'Seed': seed,
            'alpha': alpha,
            'BH_FDR': 1-bh_fdr, 'BH_Power': bh_power, 
            'BH_Boundary_Error': is_bh_boundary,
            'DominoP_FDR': 1-dominop_fdr, 'DominoP_Power': dominop_power, 'DominoP_Boundary_Error': is_dominop_boundary,
            'SL_FDR': 1-sl_fdr, 'SL_Power': sl_power, 'SL_Boundary_Error': is_sl_boundary,
            'DominoE_FDR': 1-dominope_fdr, 'DominoE_Power': dominope_power, 'DominoE_Boundary_Error': is_dominoe_boundary,
            'Domino2_FDR': 1-domino2_fdr, 'Domino2_Power': domino2_power, 'Domino2_Boundary_Error': is_boundary_false_2domino,
            'Domino3_FDR': 1-domino3_fdr, 'Domino3_Power': domino3_power, 'Domino3_Boundary_Error': is_boundary_false_3domino,
            'Domino2E_FDR': 1-domino2e_fdr, 'Domino2E_Power': domino2e_power, 'Domino2E_Boundary_Error': is_domino2_boundary,
            'Domino3E_FDR': 1-domino3e_fdr, 'Domino3E_Power': domino3e_power, 'Domino3E_Boundary_Error': is_domino3_boundary
        })
df_raw = pd.DataFrame(all_results_records)
df_summary = df_raw.groupby('alpha').agg({
    'BH_FDR': 'mean', 'BH_Power': 'mean', 
    'BH_Boundary_Error': 'mean',
    'DominoP_FDR': 'mean', 'DominoP_Power': 'mean', 'DominoP_Boundary_Error': 'mean',
    'SL_FDR': 'mean', 'SL_Power': 'mean', 'SL_Boundary_Error': 'mean',
    'DominoE_FDR': 'mean', 'DominoE_Power': 'mean', 'DominoE_Boundary_Error': 'mean',
    'Domino2_FDR': 'mean', 'Domino2_Power': 'mean', 'Domino2_Boundary_Error': 'mean',
    'Domino3_FDR': 'mean', 'Domino3_Power': 'mean', 'Domino3_Boundary_Error': 'mean',
    'Domino2E_FDR': 'mean', 'Domino2E_Power': 'mean', 'Domino2E_Boundary_Error': 'mean',
    'Domino3E_FDR': 'mean', 'Domino3E_Power': 'mean', 'Domino3E_Boundary_Error': 'mean'
}).reset_index()


# top-50 rejections evaluation
seed = 13
scores = df_labeled['Score']
df_labels = df_labeled[['Label']]
np.random.seed(seed)
test_size = 3/4
p_vals_labeled, e_values, true_labels = sample_split_testing(scores, df_labels, test_size, seed)
reject_indices_harmonic, boundary_indices_harmonic, _ = harmonicpbfdr(p_vals_labeled, alpha=0.1)
reject_indices_bh = BH(p_vals_labeled, q=0.1)
reject_indices_2domino, boundary_indices_2domino = bfdr_k_domino(p_vals_labeled, k=2, alpha=0.1)
reject_indices_sl, boundary_indices_sl, _ = sl_procedure(p_vals_labeled, alpha=0.1)
is_boundary_false_harmonic = 0
        
n_rej_harmonic, prop_harmonic, n_fd_harmonic,_ = evaluate_procedure(reject_indices_harmonic, p_vals_labeled, true_labels)
n_rej_bh, prop_bh, n_fd_bh,_ = evaluate_procedure(reject_indices_bh, p_vals_labeled, true_labels)
n_rej_2domino, prop_2domino, n_fd_2domino,_ = evaluate_procedure(reject_indices_2domino, p_vals_labeled, true_labels)
n_rej_sl, prop_sl, n_fd_sl,_ = evaluate_procedure(reject_indices_sl, p_vals_labeled, true_labels)
k_values = [1, 5, 10, 20, 35, 50]
results_list = []

for k_eval in range(1, 51):
    error_bfdr = evaluate_boundary_k_errors(reject_indices_harmonic, p_vals_labeled, true_labels, k=k_eval)
    error_bh = evaluate_boundary_k_errors(reject_indices_bh, p_vals_labeled, true_labels, k=k_eval)
    error_2domino = evaluate_boundary_k_errors(reject_indices_2domino, p_vals_labeled, true_labels, k=k_eval)
    error_sl = evaluate_boundary_k_errors(reject_indices_sl, p_vals_labeled, true_labels, k=k_eval)
    results_list.append({
        'k': k_eval,
        
        'Harmonic_Err': error_bfdr[0],
        'Harmonic_Rate': error_bfdr[0] / k_eval,
        
        'BH_Err': error_bh[0],
        'BH_Rate': error_bh[0] / k_eval,
        
        '2Domino_Err': error_2domino[0],
        '2Domino_Rate': error_2domino[0] / k_eval,
        
        'SL_Err': error_sl[0],
        'SL_Rate': error_sl[0] / k_eval
    })
df_results = pd.DataFrame(results_list)
