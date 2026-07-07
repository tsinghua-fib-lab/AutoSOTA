"""
Reproduce Section 5.4 mixed-quality judge experiment.

Paper settings:
- 8 judges: 6 low-quality + 2 high-quality (moonshot-v1-128k, kimi-k2-thinking-turbo)
- 45 candidate models
- Pearson correlation: 0.9394 (weighted) vs 0.8992 (unweighted)
- Spearman correlation: 0.9212 (weighted) vs 0.8316 (unweighted)

Reference scores come from fitting the judge-aware weighted model on ALL 18-judge data.
"""
import json
import os
import sys
from pathlib import Path
from collections import Counter, defaultdict, deque
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# ---- Copy all utility functions from main.py ----

def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x))
    )

def loglik(s, alpha, Omega, n_ijk, ybar_ijk):
    ll = 0.0
    for (i, j, k) in Omega:
        gamma_k = np.exp(alpha[k])
        z = gamma_k * (s[i] - s[j])
        p = sigmoid(z)
        ybar = ybar_ijk[(i, j, k)]
        n = n_ijk[(i, j, k)]
        ll += n * (ybar * np.log(p + 1e-12) + (1.0 - ybar) * np.log(1.0 - p + 1e-12))
    return ll

def mle_adam(N, K, Omega, n_ijk, ybar_ijk, lr_s=1e-2, lr_a=1e-3,
             beta1=0.9, beta2=0.999, eps=1e-6, max_iter=100000, tol=1e-5,
             verbose=True, s_init=None, alpha_init=None):
    if s_init is None:
        s = np.zeros(N, float)
    else:
        s = np.array(s_init, float)
    s -= s.mean()
    if alpha_init is None:
        alpha = np.zeros(K, float)
    else:
        alpha = np.array(alpha_init, float)
    alpha -= alpha.mean()

    m_s = np.zeros_like(s); v_s = np.zeros_like(s)
    m_a = np.zeros_like(alpha); v_a = np.zeros_like(alpha)

    for t in range(1, max_iter + 1):
        g_s = np.zeros_like(s); g_a = np.zeros_like(alpha)
        for (i, j, k) in Omega:
            n = n_ijk[(i, j, k)]
            ybar = ybar_ijk[(i, j, k)]
            gamma_k = np.exp(alpha[k])
            z = gamma_k * (s[i] - s[j])
            p = sigmoid(z)
            diff = ybar - p
            g_s[i] += n * gamma_k * diff
            g_s[j] -= n * gamma_k * diff
            g_a[k] += n * gamma_k * diff * (s[i] - s[j])

        grad_norm = max(np.linalg.norm(g_s), np.linalg.norm(g_a))
        m_s = beta1 * m_s + (1.0 - beta1) * g_s
        v_s = beta2 * v_s + (1.0 - beta2) * (g_s ** 2)
        m_s_hat = m_s / (1.0 - beta1 ** t)
        v_s_hat = v_s / (1.0 - beta2 ** t)
        s_new = s + lr_s * m_s_hat / (np.sqrt(v_s_hat) + eps)
        m_a = beta1 * m_a + (1.0 - beta1) * g_a
        v_a = beta2 * v_a + (1.0 - beta2) * (g_a ** 2)
        m_a_hat = m_a / (1.0 - beta1 ** t)
        v_a_hat = v_a / (1.0 - beta2 ** t)
        alpha_new = alpha + lr_a * m_a_hat / (np.sqrt(v_a_hat) + eps)
        s_new -= s_new.mean()
        alpha_new -= alpha_new.mean()
        diff_norm = max(np.linalg.norm(s_new - s), np.linalg.norm(alpha_new - alpha))
        s, alpha = s_new, alpha_new

        if verbose and (t % 500 == 0 or t == max_iter):
            ll = loglik(s, alpha, Omega, n_ijk, ybar_ijk)
            print(f"  iter {t}: ll={ll:.4f}, diff={diff_norm:.3e}, grad={grad_norm:.3e}")

        if diff_norm < tol:
            print(f"  Converged at iter {t}, diff={diff_norm:.3e}")
            break

    gamma_hat = np.exp(alpha)
    return s, gamma_hat

def aggregate_over_judges(i, j, k, p, y):
    counts = {}; sums = {}
    for ii, jj, yy in zip(i, j, y):
        key = (ii, jj)
        counts[key] = counts.get(key, 0) + 1
        sums[key] = sums.get(key, 0.0) + yy
    Omega_ij = list(counts.keys())
    n_ij = {key: counts[key] for key in Omega_ij}
    ybar_ij = {key: sums[key] / counts[key] for key in Omega_ij}
    return Omega_ij, n_ij, ybar_ij

def loglik_unweighted(s, Omega_ij, n_ij, ybar_ij):
    ll = 0.0
    for (i, j) in Omega_ij:
        z = s[i] - s[j]; p = sigmoid(z)
        ybar = ybar_ij[(i, j)]; n = n_ij[(i, j)]
        ll += n * (ybar * np.log(p + 1e-12) + (1.0 - ybar) * np.log(1.0 - p + 1e-12))
    return ll

def mle_adam_unweighted(N, Omega_ij, n_ij, ybar_ij, lr=0.003,
                        beta1=0.9, beta2=0.999, eps=1e-8,
                        max_iter=8000, tol=1e-6, verbose=True, s_init=None):
    if s_init is None:
        s = np.zeros(N, float)
    else:
        s = np.array(s_init, float)
    s -= s.mean()
    m_s = np.zeros_like(s); v_s = np.zeros_like(s)
    for t in range(1, max_iter + 1):
        g_s = np.zeros_like(s)
        for (i, j) in Omega_ij:
            n = n_ij[(i, j)]; ybar = ybar_ij[(i, j)]
            z = s[i] - s[j]; p = sigmoid(z)
            diff = ybar - p
            g_s[i] += n * diff; g_s[j] -= n * diff
        grad_norm = np.linalg.norm(g_s)
        m_s = beta1 * m_s + (1.0 - beta1) * g_s
        v_s = beta2 * v_s + (1.0 - beta2) * (g_s ** 2)
        m_hat = m_s / (1.0 - beta1 ** t)
        v_hat = v_s / (1.0 - beta2 ** t)
        s_new = s + lr * m_hat / (np.sqrt(v_hat) + eps)
        s_new -= s_new.mean()
        diff_norm = np.linalg.norm(s_new - s)
        s = s_new
        if verbose and (t % 500 == 0 or t == max_iter):
            ll = loglik_unweighted(s, Omega_ij, n_ij, ybar_ij)
            print(f"  iter {t}: ll={ll:.4f}, diff={diff_norm:.3e}, grad={grad_norm:.3e}")
        if diff_norm < tol and grad_norm < tol:
            print(f"  Converged at iter {t}, diff={diff_norm:.3e}")
            break
    return s

def check_connectivity(i, j, N):
    graph = defaultdict(set)
    for ii, jj in zip(i, j):
        graph[ii].add(jj); graph[jj].add(ii)
    visited = set([0]); queue = deque([0])
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor); queue.append(neighbor)
    return len(visited) == N

# ---- Main reproduction ----

def main():
    np.random.seed(42)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    data_path = os.path.join(parent_dir, "data", "in_house_data.json")
    output_dir = os.path.join(parent_dir, "results", "section54")
    os.makedirs(output_dir, exist_ok=True)

    # --- Load all data ---
    print("=" * 60)
    print("Loading in-house data...")
    with open(data_path, 'r') as f:
        records = json.load(f)
    print(f"Total records: {len(records)}")

    # Build mappings for ALL data
    model2id = {}; judge2id = {}; qid2id = {}
    i_list, j_list, k_list, p_list, y_list = [], [], [], [], []

    for rec in records:
        pref = rec.get("judge_preferred_model")
        if pref is None or pref == "unknown":
            continue
        ma, mb, jm = rec["model_a"], rec["model_b"], rec["judge_model"]
        if ma not in model2id: model2id[ma] = len(model2id)
        if mb not in model2id: model2id[mb] = len(model2id)
        i_tmp, j_tmp = model2id[ma], model2id[mb]
        if jm not in judge2id: judge2id[jm] = len(judge2id)
        k_tmp = judge2id[jm]
        qid_tuple = tuple(rec["question_id"])
        if qid_tuple not in qid2id: qid2id[qid_tuple] = len(qid2id)
        p_tmp = qid2id[qid_tuple]
        if pref == "a": y_tmp = 1.0
        elif pref == "b": y_tmp = 0.0
        elif pref == "c": y_tmp = 0.5
        else: continue
        if i_tmp < j_tmp:
            i_list.append(i_tmp); j_list.append(j_tmp)
        else:
            i_list.append(j_tmp); j_list.append(i_tmp)
            y_tmp = 1.0 - y_tmp
        k_list.append(k_tmp); p_list.append(p_tmp); y_list.append(y_tmp)

    i_all = np.array(i_list, int); j_all = np.array(j_list, int)
    k_all = np.array(k_list, int); p_all = np.array(p_list, int)
    y_all = np.array(y_list, float)

    N = len(model2id); K = len(judge2id)
    print(f"Models: {N}, Judges: {K}, Valid comparisons: {len(y_all)}")

    # Check connectivity of full graph
    is_conn = check_connectivity(i_all, j_all, N)
    print(f"Full graph connectivity: {'Connected' if is_conn else 'NOT connected'}")

    # --- Step 1: Fit full reference model on ALL data ---
    print("\n" + "=" * 60)
    print("STEP 1: Fitting FULL reference model (all 18 judges, all data)")
    print("=" * 60)

    counts_ijk = defaultdict(int); sums_ijk = defaultdict(float)
    for ii, jj, kk, yy in zip(i_all, j_all, k_all, y_all):
        key = (ii, jj, kk)
        counts_ijk[key] += 1; sums_ijk[key] += yy
    Omega_full = list(counts_ijk.keys())
    n_ijk_full = {key: counts_ijk[key] for key in Omega_full}
    ybar_ijk_full = {key: sums_ijk[key] / counts_ijk[key] for key in Omega_full}
    print(f"|Omega_full| = {len(Omega_full)}")

    print("\nFitting weighted model on full data...")
    s_ref, gamma_ref = mle_adam(N, K, Omega_full, n_ijk_full, ybar_ijk_full,
                                lr_s=1e-2, lr_a=1e-3, max_iter=100000, tol=1e-5, verbose=True)

    # --- Step 2: Select the 8 specific judges for mixed-quality experiment ---
    print("\n" + "=" * 60)
    print("STEP 2: Mixed-quality judge experiment (8 specific judges)")
    print("=" * 60)

    # The 8 judges from Table 2: 6 low-quality + 2 high-quality
    target_judges = [
        "Qwen/Qwen2.5-7B-Instruct-Turbo",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "moonshot-v1-128k",          # HIGH quality
        "kimi-k2-thinking-turbo",     # HIGH quality
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "google/gemma-3n-E4B-it",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
    ]

    print("Target judges:")
    for j_name in target_judges:
        in_data = j_name in judge2id
        n_comp = sum(1 for r in records if r.get('judge_model') == j_name)
        print(f"  {'✓' if in_data else '✗'} {j_name}: {n_comp} comparisons")

    # Filter to records with these 8 judges
    target_judge_set = set(target_judges)
    subset_records = [r for r in records if r.get('judge_model') in target_judge_set]
    print(f"\nRecords with target judges: {len(subset_records)}")

    # Build mappings for the subset
    model2id_sub = {}; judge2id_sub = {}; qid2id_sub = {}
    i_sub, j_sub, k_sub, p_sub, y_sub = [], [], [], [], []

    for rec in subset_records:
        pref = rec.get("judge_preferred_model")
        if pref is None or pref == "unknown": continue
        ma, mb, jm = rec["model_a"], rec["model_b"], rec["judge_model"]
        if ma not in model2id_sub: model2id_sub[ma] = len(model2id_sub)
        if mb not in model2id_sub: model2id_sub[mb] = len(model2id_sub)
        i_tmp, j_tmp = model2id_sub[ma], model2id_sub[mb]
        if jm not in judge2id_sub: judge2id_sub[jm] = len(judge2id_sub)
        k_tmp = judge2id_sub[jm]
        qid_tuple = tuple(rec["question_id"])
        if qid_tuple not in qid2id_sub: qid2id_sub[qid_tuple] = len(qid2id_sub)
        p_tmp = qid2id_sub[qid_tuple]
        if pref == "a": y_tmp = 1.0
        elif pref == "b": y_tmp = 0.0
        elif pref == "c": y_tmp = 0.5
        else: continue
        if i_tmp < j_tmp:
            i_sub.append(i_tmp); j_sub.append(j_tmp)
        else:
            i_sub.append(j_tmp); j_sub.append(i_tmp)
            y_tmp = 1.0 - y_tmp
        k_sub.append(k_tmp); p_sub.append(p_tmp); y_sub.append(y_tmp)

    i_sub = np.array(i_sub, int); j_sub = np.array(j_sub, int)
    k_sub = np.array(k_sub, int); p_sub = np.array(p_sub, int)
    y_sub = np.array(y_sub, float)

    N_sub = len(model2id_sub); K_sub = len(judge2id_sub)
    print(f"Subset: Models={N_sub}, Judges={K_sub}, Comparisons={len(y_sub)}")

    # Check connectivity
    is_conn = check_connectivity(i_sub, j_sub, N_sub)
    print(f"Subset graph connectivity: {'Connected' if is_conn else 'NOT connected'}")

    # Aggregate subset
    counts_ijk_sub = defaultdict(int); sums_ijk_sub = defaultdict(float)
    for ii, jj, kk, yy in zip(i_sub, j_sub, k_sub, y_sub):
        key = (ii, jj, kk)
        counts_ijk_sub[key] += 1; sums_ijk_sub[key] += yy
    Omega_sub = list(counts_ijk_sub.keys())
    n_ijk_sub = {key: counts_ijk_sub[key] for key in Omega_sub}
    ybar_ijk_sub = {key: sums_ijk_sub[key] / counts_ijk_sub[key] for key in Omega_sub}
    print(f"|Omega_sub| = {len(Omega_sub)}")

    # --- Step 3: Fit weighted model on the 8-judge subset ---
    print("\n--- Fitting WEIGHTED model on mixed-quality subset ---")
    s_w, gamma_w = mle_adam(N_sub, K_sub, Omega_sub, n_ijk_sub, ybar_ijk_sub,
                            lr_s=1e-2, lr_a=1e-3, max_iter=100000, tol=1e-5, verbose=True)

    print("\nJudge discrimination parameters (gamma):")
    id2judge_sub = {v: k for k, v in judge2id_sub.items()}
    for idx in range(K_sub):
        print(f"  {id2judge_sub[idx]}: gamma={gamma_w[idx]:.4f}")

    # --- Step 4: Fit unweighted model on the 8-judge subset ---
    print("\n--- Fitting UNWEIGHTED model on mixed-quality subset ---")
    Omega_ij_sub, n_ij_sub, ybar_ij_sub = aggregate_over_judges(i_sub, j_sub, k_sub, p_sub, y_sub)
    s_uw = mle_adam_unweighted(N_sub, Omega_ij_sub, n_ij_sub, ybar_ij_sub,
                               lr=0.003, max_iter=8000, tol=1e-6, verbose=True)

    # --- Step 5: Align subset scores to full model scores ---
    # Map subset model IDs back to full model IDs
    id2model_full = {v: k for k, v in model2id.items()}
    id2model_sub_rev = {v: k for k, v in model2id_sub.items()}

    # Build aligned score arrays (size N_full)
    s_w_aligned = np.full(N, np.nan)
    s_uw_aligned = np.full(N, np.nan)
    s_ref_subset_aligned = np.full(N, np.nan)

    for idx_sub, model_name in id2model_sub_rev.items():
        idx_full = model2id[model_name]
        s_w_aligned[idx_full] = s_w[idx_sub]
        s_uw_aligned[idx_full] = s_uw[idx_sub]
        s_ref_subset_aligned[idx_full] = s_ref[idx_full]

    # Remove NaN entries (any model not in subset? should be none since all 45 models are present)
    mask = ~np.isnan(s_w_aligned)
    s_w_final = s_w_aligned[mask]
    s_uw_final = s_uw_aligned[mask]
    s_ref_final = s_ref_subset_aligned[mask]
    print(f"\nModels in correlation: {np.sum(mask)}")

    # --- Step 6: Compute correlations ---
    print("\n" + "=" * 60)
    print("RESULTS: Mixed-Quality Judge Experiment")
    print("=" * 60)

    pearson_w, _ = pearsonr(s_ref_final, s_w_final)
    spearman_w, _ = spearmanr(s_ref_final, s_w_final)
    pearson_uw, _ = pearsonr(s_ref_final, s_uw_final)
    spearman_uw, _ = spearmanr(s_ref_final, s_uw_final)

    print(f"\nPearson Correlation:")
    print(f"  Weighted (judge-aware):   {pearson_w:.4f}")
    print(f"  Unweighted (baseline):    {pearson_uw:.4f}")
    print(f"  Difference:               {pearson_w - pearson_uw:+.4f}")
    print(f"  Paper weighted:           0.9394")
    print(f"  Paper unweighted:         0.8992")

    print(f"\nSpearman Correlation:")
    print(f"  Weighted (judge-aware):   {spearman_w:.4f}")
    print(f"  Unweighted (baseline):    {spearman_uw:.4f}")
    print(f"  Difference:               {spearman_w - spearman_uw:+.4f}")
    print(f"  Paper weighted:           0.9212")
    print(f"  Paper unweighted:         0.8316")

    # Save results
    results = {
        'pearson_weighted': float(pearson_w),
        'pearson_unweighted': float(pearson_uw),
        'spearman_weighted': float(spearman_w),
        'spearman_unweighted': float(spearman_uw),
        'paper_pearson_weighted': 0.9394,
        'paper_pearson_unweighted': 0.8992,
        'paper_spearman_weighted': 0.9212,
        'paper_spearman_unweighted': 0.8316,
        'n_models': int(N_sub),
        'n_judges': int(K_sub),
        'n_comparisons': int(len(y_sub)),
        'judges': target_judges,
    }

    with open(os.path.join(output_dir, 'section54_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Save detailed scores
    df_scores = pd.DataFrame({
        'model': [id2model_full[i] for i in range(N)],
        's_ref': s_ref,
        's_weighted_8judge': s_w_aligned,
        's_unweighted_8judge': s_uw_aligned,
    })
    df_scores.to_csv(os.path.join(output_dir, 'model_scores.csv'), index=False)

    # Save judge gammas
    df_judges = pd.DataFrame({
        'judge': [id2judge_sub[i] for i in range(K_sub)],
        'gamma': gamma_w,
    })
    df_judges.to_csv(os.path.join(output_dir, 'judge_gammas.csv'), index=False)

    print(f"\nResults saved to {output_dir}")
    print("Done!")

if __name__ == "__main__":
    main()
