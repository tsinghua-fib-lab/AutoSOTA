from utils import *
import numpy as np
import scipy.stats as stats
import pandas as pd
import random
import matplotlib.pyplot as plt
from scipy.stats import norm, truncnorm
import seaborn as sns
from scipy.linalg import toeplitz

n_seeds = 100
results_n_rej = []
results_prop = []
results_n_fd = []
n_boundaryfalse = []
# Utilizers can modify the hyperparameters here for different simulation settings
alpha_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
all_results_records_pi1 = []
for alpha in alpha_list:
    print(f"\n>>> Running for alpha = {alpha}")
    for seed in range(1, 1+n_seeds):
        np.random.seed(seed)
        p_values_null , p_values_alt, Sigma = generate_correlated_data(m=100, pi1=0.2, muc=3, sigma=1, rho=0.25, structure='cs', seed=seed)
        e_values_null , e_values_alt = generate_correlated_data_evalues(m=100, pi1=0.2, muc=3, sigma=1, rho=0.25, seed=seed)
        p_values = np.concatenate([p_values_null, p_values_alt])
        e_values = np.concatenate([e_values_null, e_values_alt])
        n_null = len(p_values_null)
        n_alt = len(p_values_alt)
        ground_truth = np.concatenate([np.zeros(n_null), np.ones(n_alt)])
        
        
        # simes_dominop
        
        reject_indices_simes, boundary_indices_simes,_ = simesdomino(p_values,  alpha=alpha)
        is_boundary_false_simes = kbfdr_evaluate(ground_truth, boundary_indices_simes, k=1)
        
        # slmethod
        reject_indices_sl, boundary_indices_sl ,_ = sl_procedure(p_values,  alpha=alpha)
        is_boundary_false_sl = kbfdr_evaluate(ground_truth, boundary_indices_sl, k=1)
   
                
        # domino_ebfdr
        reject_indices_ebfdr, boundary_indices_ebfdr = bfdr_k_eclosure(e_values, k=1, alpha=alpha)
        is_boundary_false_ebfdr = kbfdr_evaluate(ground_truth, boundary_indices_ebfdr, k=1)
        
        #2domino_pbfdr
        reject_indices_2dominop, boundary_indices_2dominop = bfdr_k_domino(p_values, k=2, alpha=alpha)
        is_boundary_false_2dominop = kbfdr_evaluate(ground_truth, boundary_indices_2dominop, k=2)
        
        #2edomino_ebfdr
        reject_indices_2edominoe, boundary_indices_2edominoe = bfdr_k_eclosure(e_values, k=2, alpha=alpha)
        is_boundary_false_2edominoe = kbfdr_evaluate(ground_truth, boundary_indices_2edominoe, k=2)
        
        # 3-domino_pbfdr
        reject_indices_3dominop, boundary_indices_3dominop = bfdr_k_domino(p_values, k=3, alpha=alpha)
        is_boundary_false_3dominop = kbfdr_evaluate(ground_truth, boundary_indices_3dominop, k=3)
        
        # 3edomino_ebfdr
        reject_indices_3edominoe, boundary_indices_3edominoe = bfdr_k_eclosure(e_values, k=3, alpha=alpha)
        is_boundary_false_3edominoe = kbfdr_evaluate(ground_truth, boundary_indices_3edominoe, k=3)
        
        n_rej_ebfdr, prop_ebfdr, n_fd_ebfdr, power_ebfdr = evaluate_procedure(reject_indices_ebfdr, p_values, ground_truth)
        n_rej_simes, prop_simes, n_fd_simes, power_simes = evaluate_procedure(reject_indices_simes, p_values, ground_truth)
        n_rej_sl, prop_sl, n_fd_sl, power_sl = evaluate_procedure(reject_indices_sl, p_values, ground_truth)
        n_rej_2domino, prop_2domino, n_fd_2domino, power_2domino = evaluate_procedure(reject_indices_2dominop, p_values, ground_truth)
        n_rej_2ebfdr, prop_2ebfdr, n_fd_2ebfdr, power_2ebfdr = evaluate_procedure(reject_indices_2edominoe, p_values, ground_truth)
        n_rej_3domino, prop_3domino, n_fd_3domino, power_3domino = evaluate_procedure(reject_indices_3dominop, p_values, ground_truth)
        n_rej_3ebfdr, prop_3ebfdr, n_fd_3ebfdr, power_3ebfdr = evaluate_procedure(reject_indices_3edominoe, p_values, ground_truth)
        record = {
            "alpha": alpha,                          
            "n_rej_simes": n_rej_simes,            
            "rejectionprop_simes": prop_simes,             
            "n_fd_simes": 1-n_fd_simes,              
            "boundary_error_simes": is_boundary_false_simes,
            "power_simes": power_simes,
            "n_rej_sl": n_rej_sl,
            "rejectionprop_sl": prop_sl,
            "n_fd_sl": 1-n_fd_sl,
            "boundary_error_sl": is_boundary_false_sl,
            "power_sl": power_sl,
            "n_rej_ebfdr": n_rej_ebfdr,
            "rejectionprop_ebfdr": prop_ebfdr,
            "n_fd_ebfdr": 1-n_fd_ebfdr,
            "boundary_error_ebfdr": is_boundary_false_ebfdr,
            "power_ebfdr": power_ebfdr,
            "n_rej_2domino": n_rej_2domino,
            "rejectionprop_2domino": prop_2domino,
            "n_fd_2domino": 1-n_fd_2domino,
            "boundary_error_2domino": is_boundary_false_2dominop,
            "power_2domino": power_2domino,
            "n_rej_2ebfdr": n_rej_2ebfdr,
            "rejectionprop_2ebfdr": prop_2ebfdr,
            "n_fd_2ebfdr": 1-n_fd_2ebfdr,
            "boundary_error_2ebfdr": is_boundary_false_2edominoe,
            "power_2ebfdr": power_2ebfdr,
            "n_rej_3domino": n_rej_3domino,
            "rejectionprop_3domino": prop_3domino,
            "n_fd_3domino": 1-n_fd_3domino,
            "boundary_error_3domino": is_boundary_false_3dominop,
            "power_3domino": power_3domino,
            "n_rej_3ebfdr": n_rej_3ebfdr,
            "rejectionprop_3ebfdr": prop_3ebfdr,
            "n_fd_3ebfdr": 1-n_fd_3ebfdr,
            "boundary_error_3ebfdr": is_boundary_false_3edominoe,
            "power_3ebfdr": power_3ebfdr,
        }
        all_results_records_pi1.append(record)
df_raw = pd.DataFrame(all_results_records_pi1)
        
df_summary = df_raw.groupby("alpha").agg(
avg_n_rej_simes=('n_rej_simes', 'mean'),
avg_td_simes=('n_fd_simes', 'mean'),
bfdr_simes=('boundary_error_simes', 'mean'),
power_simes=('power_simes', 'mean'),
avg_n_rej_sl=('n_rej_sl', 'mean'),
avg_td_sl=('n_fd_sl', 'mean'),
bfdr_sl=('boundary_error_sl', 'mean'),
power_sl=('power_sl', 'mean'),
avg_n_rej_ebfdr=('n_rej_ebfdr', 'mean'),
avg_td_ebfdr=('n_fd_ebfdr', 'mean'),
bfdr_ebfdr=('boundary_error_ebfdr', 'mean'),
power_ebfdr=('power_ebfdr', 'mean'),
avg_n_rej_2domino=('n_rej_2domino', 'mean'),
avg_td_2domino=('n_fd_2domino', 'mean'),
bfdr_2domino=('boundary_error_2domino', 'mean'),
power_2domino=('power_2domino', 'mean'),
avg_n_rej_2ebfdr=('n_rej_2ebfdr', 'mean'),
avg_td_2ebfdr=('n_fd_2ebfdr', 'mean'),
bfdr_2ebfdr=('boundary_error_2ebfdr', 'mean'),
power_2ebfdr=('power_2ebfdr', 'mean'),
avg_n_rej_3domino=('n_rej_3domino', 'mean'),
avg_td_3domino=('n_fd_3domino', 'mean'),
bfdr_3domino=('boundary_error_3domino', 'mean'),
power_3domino=('power_3domino', 'mean'),
avg_n_rej_3ebfdr=('n_rej_3ebfdr', 'mean'),
avg_td_3ebfdr=('n_fd_3ebfdr', 'mean'),
bfdr_3ebfdr=('boundary_error_3ebfdr', 'mean'),
power_3ebfdr=('power_3ebfdr', 'mean'),
).reset_index()
