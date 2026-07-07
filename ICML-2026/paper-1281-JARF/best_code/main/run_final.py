"""
Exact reproduction of Section 5.4 mixed-quality experiment.
Samples p=1000 comparisons from 8 specific judges, repeated 20 times.
"""
import json, os, time
import numpy as np
from scipy.stats import pearsonr, spearmanr

def sigmoid(x):
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))

def mle_adam_fast(N, K, i_arr, j_arr, k_arr, n_arr, ybar_arr, 
                  lr_s=1e-2, lr_a=1e-3, max_iter=5000, tol=1e-5, verbose=False):
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
        
        if diff_norm < tol:
            break
    
    return s, np.exp(alpha)

def mle_unweighted_fast(N, i_arr, j_arr, n_arr, ybar_arr, max_iter=5000, tol=1e-6, verbose=False):
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
        
        if diff_norm < tol:
            break
    
    return s

def build_arrays(records_subset, model2id_ref, judge2id_ref):
    """Build mapping and return arrays for a subset of records."""
    model2id = {}; judge2id = {}
    i_l, j_l, k_l, y_l = [], [], [], []
    
    for rec in records_subset:
        pref = rec.get("judge_preferred_model")
        if pref is None or pref == "unknown": continue
        ma, mb, jm = rec["model_a"], rec["model_b"], rec["judge_model"]
        if ma not in model2id: model2id[ma] = len(model2id)
        if mb not in model2id: model2id[mb] = len(model2id)
        i_tmp, j_tmp = model2id[ma], model2id[mb]
        if jm not in judge2id: judge2id[jm] = len(judge2id)
        k_tmp = judge2id[jm]
        if pref == "a": y = 1.0
        elif pref == "b": y = 0.0
        elif pref == "c": y = 0.5
        else: continue
        if i_tmp < j_tmp:
            i_l.append(i_tmp); j_l.append(j_tmp)
        else:
            i_l.append(j_tmp); j_l.append(i_tmp)
            y = 1.0 - y
        k_l.append(k_tmp); y_l.append(y)
    
    i_arr = np.array(i_l, int); j_arr = np.array(j_l, int)
    k_arr = np.array(k_l, int); y_arr = np.array(y_l, float)
    return model2id, judge2id, i_arr, j_arr, k_arr, y_arr

def aggregate_ijk(i_arr, j_arr, k_arr, y_arr):
    n_ijk = {}; sum_ijk = {}
    for ii, jj, kk, yy in zip(i_arr, j_arr, k_arr, y_arr):
        key = (ii, jj, kk)
        n_ijk[key] = n_ijk.get(key, 0) + 1
        sum_ijk[key] = sum_ijk.get(key, 0.0) + yy
    Omega = list(n_ijk.keys())
    i_o = np.array([o[0] for o in Omega], int)
    j_o = np.array([o[1] for o in Omega], int)
    k_o = np.array([o[2] for o in Omega], int)
    n_o = np.array([n_ijk[o] for o in Omega], float)
    ybar_o = np.array([sum_ijk[o] / n_ijk[o] for o in Omega], float)
    return Omega, i_o, j_o, k_o, n_o, ybar_o

def aggregate_ij(i_arr, j_arr, y_arr):
    n_ij = {}; sum_ij = {}
    for ii, jj, yy in zip(i_arr, j_arr, y_arr):
        key = (ii, jj)
        n_ij[key] = n_ij.get(key, 0) + 1
        sum_ij[key] = sum_ij.get(key, 0.0) + yy
    Omega = list(n_ij.keys())
    i_o = np.array([o[0] for o in Omega], int)
    j_o = np.array([o[1] for o in Omega], int)
    n_o = np.array([n_ij[o] for o in Omega], float)
    ybar_o = np.array([sum_ij[o] / n_ij[o] for o in Omega], float)
    return i_o, j_o, n_o, ybar_o

def main():
    t0 = time.time()
    np.random.seed(42)
    data_path = "/repo/data/in_house_data.json"
    output_dir = "/repo/results/section54"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Section 5.4 Mixed-Quality Judge Reproduction")
    print("=" * 60)

    with open(data_path, 'r') as f:
        records = json.load(f)
    print(f"Total records: {len(records)}")

    # Full data mappings
    model2id = {}; judge2id = {}
    i_list, j_list, k_list, y_list = [], [], [], []
    for rec in records:
        pref = rec.get("judge_preferred_model")
        if pref is None or pref == "unknown": continue
        ma, mb, jm = rec["model_a"], rec["model_b"], rec["judge_model"]
        if ma not in model2id: model2id[ma] = len(model2id)
        if mb not in model2id: model2id[mb] = len(model2id)
        i_tmp, j_tmp = model2id[ma], model2id[mb]
        if jm not in judge2id: judge2id[jm] = len(judge2id)
        k_tmp = judge2id[jm]
        if pref == "a": y = 1.0
        elif pref == "b": y = 0.0
        elif pref == "c": y = 0.5
        else: continue
        if i_tmp < j_tmp:
            i_list.append(i_tmp); j_list.append(j_tmp)
        else:
            i_list.append(j_tmp); j_list.append(i_tmp)
            y = 1.0 - y
        k_list.append(k_tmp); y_list.append(y)

    i_all = np.array(i_list, int); j_all = np.array(j_list, int)
    k_all = np.array(k_list, int); y_all = np.array(y_list, float)
    N = len(model2id); K = len(judge2id)
    print(f"Full data: Models={N}, Judges={K}, Comparisons={len(y_all)}")

    # Fit full reference model
    print("\nFitting FULL reference model (all 18 judges, all data)...")
    _, i_ref, j_ref, k_ref, n_ref, ybar_ref = aggregate_ijk(i_all, j_all, k_all, y_all)
    s_ref, gamma_ref = mle_adam_fast(N, K, i_ref, j_ref, k_ref, n_ref, ybar_ref,
                                     max_iter=5000, tol=1e-5, verbose=True)
    
    print("\nFull model gamma values (top and bottom):")
    id2judge_full = {v: k for k, v in judge2id.items()}
    sorted_judges = sorted(range(K), key=lambda x: -gamma_ref[x])
    for idx in sorted_judges[:5]:
        print(f"  {id2judge_full[idx]}: gamma={gamma_ref[idx]:.4f}")
    print("  ...")
    for idx in sorted_judges[-3:]:
        print(f"  {id2judge_full[idx]}: gamma={gamma_ref[idx]:.4f}")

    # 8 specific judges for mixed-quality experiment
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
    target_judge_set = set(target_judges)

    # Filter to 8-judge records
    subset_records = [r for r in records if r.get('judge_model') in target_judge_set]
    print(f"\n8-judge subset records: {len(subset_records)}")

    # Experiment: for each sample size p, run num_repeats random samples
    p = 1000
    num_repeats = 20
    print(f"\nRunning experiment: k=8 judges, p={p} comparisons, {num_repeats} repeats")
    print("=" * 60)

    pearson_w_list, pearson_uw_list = [], []
    spearman_w_list, spearman_uw_list = [], []

    for run in range(num_repeats):
        # Sample p comparisons from subset
        if len(subset_records) <= p:
            sampled = subset_records
        else:
            indices = np.random.choice(len(subset_records), p, replace=False)
            sampled = [subset_records[i] for i in indices]

        # Build and fit weighted model
        m2id_sub, j2id_sub, i_sub, j_sub, k_sub, y_sub = build_arrays(sampled, model2id, judge2id)
        N_sub = len(m2id_sub)
        K_sub = len(j2id_sub)

        _, i_w, j_w, k_w, n_w, ybar_w = aggregate_ijk(i_sub, j_sub, k_sub, y_sub)
        s_w, _ = mle_adam_fast(N_sub, K_sub, i_w, j_w, k_w, n_w, ybar_w, max_iter=5000, tol=1e-5)

        # Build and fit unweighted model
        i_uw, j_uw, n_uw, ybar_uw = aggregate_ij(i_sub, j_sub, y_sub)
        s_uw = mle_unweighted_fast(N_sub, i_uw, j_uw, n_uw, ybar_uw, max_iter=5000, tol=1e-6)

        # Align subset scores to full model IDs
        id2model_sub = {v: k for k, v in m2id_sub.items()}
        s_w_aligned = np.full(N, np.nan)
        s_uw_aligned = np.full(N, np.nan)
        s_ref_aligned = np.full(N, np.nan)

        for idx_sub, model_name in id2model_sub.items():
            idx_full = model2id[model_name]
            s_w_aligned[idx_full] = s_w[idx_sub]
            s_uw_aligned[idx_full] = s_uw[idx_sub]
            s_ref_aligned[idx_full] = s_ref[idx_full]

        mask = ~np.isnan(s_w_aligned)
        pw, _ = pearsonr(s_ref_aligned[mask], s_w_aligned[mask])
        sw, _ = spearmanr(s_ref_aligned[mask], s_w_aligned[mask])
        puw, _ = pearsonr(s_ref_aligned[mask], s_uw_aligned[mask])
        suw, _ = spearmanr(s_ref_aligned[mask], s_uw_aligned[mask])

        pearson_w_list.append(pw)
        pearson_uw_list.append(puw)
        spearman_w_list.append(sw)
        spearman_uw_list.append(suw)

        print(f"  Run {run+1:2d}/{num_repeats}: Pearson W={pw:.4f} UW={puw:.4f} | Spearman W={sw:.4f} UW={suw:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS: Mixed-Quality Judge Experiment (k=8, p=1000)")
    print("=" * 60)
    
    avg_pw = np.mean(pearson_w_list); std_pw = np.std(pearson_w_list)
    avg_puw = np.mean(pearson_uw_list); std_puw = np.std(pearson_uw_list)
    avg_sw = np.mean(spearman_w_list); std_sw = np.std(spearman_w_list)
    avg_suw = np.mean(spearman_uw_list); std_suw = np.std(spearman_uw_list)

    print(f"\nPearson Correlation:")
    print(f"  Weighted (judge-aware):   {avg_pw:.4f} ± {std_pw:.4f}  (paper: 0.9394)")
    print(f"  Unweighted (baseline):    {avg_puw:.4f} ± {std_puw:.4f}  (paper: 0.8992)")
    print(f"  Difference:               {avg_pw - avg_puw:+.4f}")

    print(f"\nSpearman Correlation:")
    print(f"  Weighted (judge-aware):   {avg_sw:.4f} ± {std_sw:.4f}  (paper: 0.9212)")
    print(f"  Unweighted (baseline):    {avg_suw:.4f} ± {std_suw:.4f}  (paper: 0.8316)")
    print(f"  Difference:               {avg_sw - avg_suw:+.4f}")

    results = {
        'p': p, 'num_repeats': num_repeats,
        'pearson_weighted_mean': float(avg_pw), 'pearson_weighted_std': float(std_pw),
        'pearson_unweighted_mean': float(avg_puw), 'pearson_unweighted_std': float(std_puw),
        'spearman_weighted_mean': float(avg_sw), 'spearman_weighted_std': float(std_sw),
        'spearman_unweighted_mean': float(avg_suw), 'spearman_unweighted_std': float(std_suw),
        'paper_pearson_weighted': 0.9394, 'paper_pearson_unweighted': 0.8992,
        'paper_spearman_weighted': 0.9212, 'paper_spearman_unweighted': 0.8316,
    }
    with open(os.path.join(output_dir, 'section54_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Results saved to {output_dir}/section54_results.json")
    print("Done!")

if __name__ == "__main__":
    main()
