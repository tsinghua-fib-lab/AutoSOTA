"""
Optimized reproduction of Section 5.4 with vectorized gradient computation.
"""
import json, os, time
import numpy as np
from scipy.stats import pearsonr, spearmanr

def sigmoid(x):
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))

def mle_adam_fast(N, K, i_arr, j_arr, k_arr, n_arr, ybar_arr, 
                  lr_s=1e-2, lr_a=1e-3, max_iter=3000, tol=1e-5, verbose=True):
    """Vectorized MLE for judge-aware BT model."""
    s = np.zeros(N); alpha = np.zeros(K)
    m_s = np.zeros(N); v_s = np.zeros(N)
    m_a = np.zeros(K); v_a = np.zeros(K)
    
    for t in range(1, max_iter + 1):
        gamma_k = np.exp(alpha[k_arr])
        z = gamma_k * (s[i_arr] - s[j_arr])
        p = sigmoid(z)
        diff = ybar_arr - p
        
        contrib = n_arr * gamma_k * diff
        g_s = np.bincount(i_arr, weights=contrib, minlength=N).astype(float)
        g_s -= np.bincount(j_arr, weights=contrib, minlength=N).astype(float)
        
        contrib_a = n_arr * gamma_k * diff * (s[i_arr] - s[j_arr])
        g_a = np.bincount(k_arr, weights=contrib_a, minlength=K).astype(float)
        
        grad_norm = max(np.linalg.norm(g_s), np.linalg.norm(g_a))
        
        m_s = 0.9 * m_s + 0.1 * g_s
        v_s = 0.999 * v_s + 0.001 * (g_s ** 2)
        m_s_hat = m_s / (1.0 - 0.9**t)
        v_s_hat = v_s / (1.0 - 0.999**t)
        s_new = s + lr_s * m_s_hat / (np.sqrt(v_s_hat) + 1e-6)
        
        m_a = 0.9 * m_a + 0.1 * g_a
        v_a = 0.999 * v_a + 0.001 * (g_a ** 2)
        m_a_hat = m_a / (1.0 - 0.9**t)
        v_a_hat = v_a / (1.0 - 0.999**t)
        alpha_new = alpha + lr_a * m_a_hat / (np.sqrt(v_a_hat) + 1e-6)
        
        s_new -= s_new.mean()
        alpha_new -= alpha_new.mean()
        
        diff_norm = max(np.linalg.norm(s_new - s), np.linalg.norm(alpha_new - alpha))
        s, alpha = s_new, alpha_new
        
        if verbose and (t % 100 == 0 or t == 1):
            gamma_k = np.exp(alpha[k_arr])
            z = gamma_k * (s[i_arr] - s[j_arr])
            p = sigmoid(z)
            ll = np.sum(n_arr * (ybar_arr * np.log(p + 1e-12) + (1.0 - ybar_arr) * np.log(1.0 - p + 1e-12)))
            print(f"  iter {t}: ll={ll:.4f}, diff={diff_norm:.3e}, grad={grad_norm:.3e}")
        
        if diff_norm < tol:
            print(f"  Converged at iter {t}")
            break
    
    return s, np.exp(alpha)

def mle_unweighted_fast(N, i_arr, j_arr, n_arr, ybar_arr, max_iter=3000, tol=1e-6, verbose=True):
    """Vectorized MLE for standard BT model."""
    s = np.zeros(N)
    m_s = np.zeros(N); v_s = np.zeros(N)
    
    for t in range(1, max_iter + 1):
        z = s[i_arr] - s[j_arr]
        p = sigmoid(z)
        diff = ybar_arr - p
        
        contrib = n_arr * diff
        g_s = np.bincount(i_arr, weights=contrib, minlength=N).astype(float)
        g_s -= np.bincount(j_arr, weights=contrib, minlength=N).astype(float)
        
        grad_norm = np.linalg.norm(g_s)
        
        m_s = 0.9 * m_s + 0.1 * g_s
        v_s = 0.999 * v_s + 0.001 * (g_s ** 2)
        m_hat = m_s / (1.0 - 0.9**t)
        v_hat = v_s / (1.0 - 0.999**t)
        s_new = s + 0.003 * m_hat / (np.sqrt(v_hat) + 1e-8)
        s_new -= s_new.mean()
        
        diff_norm = np.linalg.norm(s_new - s)
        s = s_new
        
        if verbose and (t % 100 == 0 or t == 1):
            z = s[i_arr] - s[j_arr]
            p = sigmoid(z)
            ll = np.sum(n_arr * (ybar_arr * np.log(p + 1e-12) + (1.0 - ybar_arr) * np.log(1.0 - p + 1e-12)))
            print(f"  iter {t}: ll={ll:.4f}, diff={diff_norm:.3e}, grad={grad_norm:.3e}")
        
        if diff_norm < tol:
            print(f"  Converged at iter {t}")
            break
    
    return s


def main():
    t0 = time.time()
    np.random.seed(42)
    data_path = "/repo/data/in_house_data.json"
    output_dir = "/repo/results/section54"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Loading in-house data...")
    with open(data_path, 'r') as f:
        records = json.load(f)
    print(f"Total records: {len(records)}")

    model2id = {}; judge2id = {}
    i_list, j_list, k_list, y_list = [], [], [], []

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
        if pref == "a": y_tmp = 1.0
        elif pref == "b": y_tmp = 0.0
        elif pref == "c": y_tmp = 0.5
        else: continue
        if i_tmp < j_tmp:
            i_list.append(i_tmp); j_list.append(j_tmp)
        else:
            i_list.append(j_tmp); j_list.append(i_tmp)
            y_tmp = 1.0 - y_tmp
        k_list.append(k_tmp); y_list.append(y_tmp)

    i_all = np.array(i_list, int); j_all = np.array(j_list, int)
    k_all = np.array(k_list, int); y_all = np.array(y_list, float)
    N = len(model2id); K = len(judge2id)
    print(f"Models: {N}, Judges: {K}, Valid comparisons: {len(y_all)}")

    # Aggregate (i,j,k) level
    n_ijk = {}; sum_ijk = {}
    for ii, jj, kk, yy in zip(i_all, j_all, k_all, y_all):
        key = (ii, jj, kk)
        n_ijk[key] = n_ijk.get(key, 0) + 1
        sum_ijk[key] = sum_ijk.get(key, 0.0) + yy

    Omega = list(n_ijk.keys())
    i_arr = np.array([o[0] for o in Omega], int)
    j_arr = np.array([o[1] for o in Omega], int)
    k_arr = np.array([o[2] for o in Omega], int)
    n_arr = np.array([n_ijk[o] for o in Omega], float)
    ybar_arr = np.array([sum_ijk[o] / n_ijk[o] for o in Omega], float)
    print(f"|Omega_full| = {len(Omega)}")

    # STEP 1: Full reference model
    print("\n" + "=" * 60)
    print("STEP 1: Fitting FULL reference model (all 18 judges)")
    print("=" * 60)
    s_ref, gamma_ref = mle_adam_fast(N, K, i_arr, j_arr, k_arr, n_arr, ybar_arr,
                                     max_iter=2000, tol=1e-5, verbose=True)
    
    print("\nFull model gamma values:")
    id2judge_full = {v: k for k, v in judge2id.items()}
    for idx in range(K):
        print(f"  {id2judge_full[idx]}: gamma={gamma_ref[idx]:.4f}")
    
    # STEP 2: Mixed-quality 8-judge subset
    print("\n" + "=" * 60)
    print("STEP 2: Mixed-quality judge experiment (8 judges)")
    print("=" * 60)
    
    target_judges = [
        "Qwen/Qwen2.5-7B-Instruct-Turbo",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "moonshot-v1-128k",
        "kimi-k2-thinking-turbo",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "google/gemma-3n-E4B-it",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
    ]
    
    target_judge_set = set(target_judges)
    
    model2id_sub = {}; judge2id_sub = {}
    i_sub_l, j_sub_l, k_sub_l, y_sub_l = [], [], [], []
    
    for rec in records:
        pref = rec.get("judge_preferred_model")
        if pref is None or pref == "unknown": continue
        jm = rec["judge_model"]
        if jm not in target_judge_set: continue
        ma, mb = rec["model_a"], rec["model_b"]
        if ma not in model2id_sub: model2id_sub[ma] = len(model2id_sub)
        if mb not in model2id_sub: model2id_sub[mb] = len(model2id_sub)
        i_tmp, j_tmp = model2id_sub[ma], model2id_sub[mb]
        if jm not in judge2id_sub: judge2id_sub[jm] = len(judge2id_sub)
        k_tmp = judge2id_sub[jm]
        if pref == "a": y_tmp = 1.0
        elif pref == "b": y_tmp = 0.0
        elif pref == "c": y_tmp = 0.5
        else: continue
        if i_tmp < j_tmp:
            i_sub_l.append(i_tmp); j_sub_l.append(j_tmp)
        else:
            i_sub_l.append(j_tmp); j_sub_l.append(i_tmp)
            y_tmp = 1.0 - y_tmp
        k_sub_l.append(k_tmp); y_sub_l.append(y_tmp)
    
    i_sub = np.array(i_sub_l, int); j_sub = np.array(j_sub_l, int)
    k_sub = np.array(k_sub_l, int); y_sub = np.array(y_sub_l, float)
    N_sub = len(model2id_sub); K_sub = len(judge2id_sub)
    print(f"Subset: Models={N_sub}, Judges={K_sub}, Comparisons={len(y_sub)}")
    
    # Aggregate subset
    n_ijk_sub = {}; sum_ijk_sub = {}
    for ii, jj, kk, yy in zip(i_sub, j_sub, k_sub, y_sub):
        key = (ii, jj, kk)
        n_ijk_sub[key] = n_ijk_sub.get(key, 0) + 1
        sum_ijk_sub[key] = sum_ijk_sub.get(key, 0.0) + yy
    
    Omega_sub = list(n_ijk_sub.keys())
    i_arr_sub = np.array([o[0] for o in Omega_sub], int)
    j_arr_sub = np.array([o[1] for o in Omega_sub], int)
    k_arr_sub = np.array([o[2] for o in Omega_sub], int)
    n_arr_sub = np.array([n_ijk_sub[o] for o in Omega_sub], float)
    ybar_arr_sub = np.array([sum_ijk_sub[o] / n_ijk_sub[o] for o in Omega_sub], float)
    print(f"|Omega_sub| = {len(Omega_sub)}")
    
    # STEP 3: Weighted model on subset
    print("\n--- Fitting WEIGHTED model on 8-judge subset ---")
    s_w, gamma_w = mle_adam_fast(N_sub, K_sub, i_arr_sub, j_arr_sub, k_arr_sub,
                                  n_arr_sub, ybar_arr_sub, max_iter=3000, tol=1e-5, verbose=True)
    
    id2judge_sub = {v: k for k, v in judge2id_sub.items()}
    print("\nJudge discrimination parameters (gamma):")
    for idx in range(K_sub):
        print(f"  {id2judge_sub[idx]}: gamma={gamma_w[idx]:.4f}")
    
    # STEP 4: Unweighted model on subset
    print("\n--- Fitting UNWEIGHTED model on 8-judge subset ---")
    n_ij_sub = {}; sum_ij_sub = {}
    for ii, jj, yy in zip(i_sub, j_sub, y_sub):
        key = (ii, jj)
        n_ij_sub[key] = n_ij_sub.get(key, 0) + 1
        sum_ij_sub[key] = sum_ij_sub.get(key, 0.0) + yy
    Omega_ij_sub = list(n_ij_sub.keys())
    i_arr_uw = np.array([o[0] for o in Omega_ij_sub], int)
    j_arr_uw = np.array([o[1] for o in Omega_ij_sub], int)
    n_arr_uw = np.array([n_ij_sub[o] for o in Omega_ij_sub], float)
    ybar_arr_uw = np.array([sum_ij_sub[o] / n_ij_sub[o] for o in Omega_ij_sub], float)
    
    s_uw = mle_unweighted_fast(N_sub, i_arr_uw, j_arr_uw, n_arr_uw, ybar_arr_uw,
                               max_iter=3000, tol=1e-6, verbose=True)
    
    # STEP 5: Align and compute correlations
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    id2model_full = {v: k for k, v in model2id.items()}
    id2model_sub_rev = {v: k for k, v in model2id_sub.items()}
    
    s_w_aligned = np.full(N, np.nan)
    s_uw_aligned = np.full(N, np.nan)
    s_ref_aligned = np.full(N, np.nan)
    
    for idx_sub, model_name in id2model_sub_rev.items():
        idx_full = model2id[model_name]
        s_w_aligned[idx_full] = s_w[idx_sub]
        s_uw_aligned[idx_full] = s_uw[idx_sub]
        s_ref_aligned[idx_full] = s_ref[idx_full]
    
    mask = ~np.isnan(s_w_aligned)
    s_w_final = s_w_aligned[mask]
    s_uw_final = s_uw_aligned[mask]
    s_ref_final = s_ref_aligned[mask]
    print(f"Models in correlation: {np.sum(mask)}")
    
    pearson_w, _ = pearsonr(s_ref_final, s_w_final)
    spearman_w, _ = spearmanr(s_ref_final, s_w_final)
    pearson_uw, _ = pearsonr(s_ref_final, s_uw_final)
    spearman_uw, _ = spearmanr(s_ref_final, s_uw_final)
    
    print(f"\nPearson Correlation:")
    print(f"  Weighted (judge-aware):   {pearson_w:.4f}  (paper: 0.9394)")
    print(f"  Unweighted (baseline):    {pearson_uw:.4f}  (paper: 0.8992)")
    print(f"  Difference:               {pearson_w - pearson_uw:+.4f}")
    
    print(f"\nSpearman Correlation:")
    print(f"  Weighted (judge-aware):   {spearman_w:.4f}  (paper: 0.9212)")
    print(f"  Unweighted (baseline):    {spearman_uw:.4f}  (paper: 0.8316)")
    print(f"  Difference:               {spearman_w - spearman_uw:+.4f}")
    
    # Save
    results = {
        'pearson_weighted': float(pearson_w),
        'pearson_unweighted': float(pearson_uw),
        'spearman_weighted': float(spearman_w),
        'spearman_unweighted': float(spearman_uw),
        'paper_pearson_weighted': 0.9394,
        'paper_pearson_unweighted': 0.8992,
        'paper_spearman_weighted': 0.9212,
        'paper_spearman_unweighted': 0.8316,
    }
    with open(os.path.join(output_dir, 'section54_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Results saved to {output_dir}")
    print("Done!")

if __name__ == "__main__":
    main()
