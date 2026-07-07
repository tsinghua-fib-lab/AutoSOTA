"""
Exact reproduction of Section 5.4 with p=5000 comparisons.
"""
import json, os, time
import numpy as np
from scipy.stats import pearsonr, spearmanr

def sigmoid(x):
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))

def mle_adam_fast(N, K, i_arr, j_arr, k_arr, n_arr, ybar_arr, 
                  lr_s=1e-2, lr_a=1e-3, max_iter=10000, tol=1e-5, verbose=False):
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
        m_s = 0.9*m_s + 0.1*g_s; v_s = 0.999*v_s + 0.001*(g_s**2)
        m_a = 0.9*m_a + 0.1*g_a; v_a = 0.999*v_a + 0.001*(g_a**2)
        s_new = s + lr_s * (m_s/(1-0.9**t)) / (np.sqrt(v_s/(1-0.999**t)) + 1e-6)
        alpha_new = alpha + lr_a * (m_a/(1-0.9**t)) / (np.sqrt(v_a/(1-0.999**t)) + 1e-6)
        s_new -= s_new.mean(); alpha_new -= alpha_new.mean()
        diff_norm = max(np.linalg.norm(s_new-s), np.linalg.norm(alpha_new-alpha))
        s, alpha = s_new, alpha_new
        if diff_norm < tol: break
    return s, np.exp(alpha)

def mle_unweighted_fast(N, i_arr, j_arr, n_arr, ybar_arr, max_iter=5000, tol=1e-6, verbose=False):
    s = np.zeros(N); m_s = np.zeros(N); v_s = np.zeros(N)
    for t in range(1, max_iter + 1):
        z = s[i_arr] - s[j_arr]; p = sigmoid(z); diff = ybar_arr - p
        contrib = n_arr * diff
        g_s = np.bincount(i_arr, weights=contrib, minlength=N).astype(float)
        g_s -= np.bincount(j_arr, weights=contrib, minlength=N).astype(float)
        m_s = 0.9*m_s + 0.1*g_s; v_s = 0.999*v_s + 0.001*(g_s**2)
        s_new = s + 0.003*(m_s/(1-0.9**t))/(np.sqrt(v_s/(1-0.999**t))+1e-8)
        s_new -= s_new.mean()
        if np.linalg.norm(s_new-s) < tol and np.linalg.norm(g_s) < tol: s = s_new; break
        s = s_new
    return s

def build_arrays(records_subset):
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
        if i_tmp < j_tmp: i_l.append(i_tmp); j_l.append(j_tmp)
        else: i_l.append(j_tmp); j_l.append(i_tmp); y = 1.0 - y
        k_l.append(k_tmp); y_l.append(y)
    return model2id, judge2id, np.array(i_l,int), np.array(j_l,int), np.array(k_l,int), np.array(y_l,float)

def aggregate_ijk(i_arr, j_arr, k_arr, y_arr):
    n_ijk = {}; sum_ijk = {}
    for ii, jj, kk, yy in zip(i_arr, j_arr, k_arr, y_arr):
        key = (ii, jj, kk); n_ijk[key]=n_ijk.get(key,0)+1; sum_ijk[key]=sum_ijk.get(key,0.0)+yy
    Omega = list(n_ijk.keys())
    return (np.array([o[0] for o in Omega],int), np.array([o[1] for o in Omega],int),
            np.array([o[2] for o in Omega],int), np.array([n_ijk[o] for o in Omega],float),
            np.array([sum_ijk[o]/n_ijk[o] for o in Omega],float))

def aggregate_ij(i_arr, j_arr, y_arr):
    n_ij = {}; sum_ij = {}
    for ii, jj, yy in zip(i_arr, j_arr, y_arr):
        key=(ii,jj); n_ij[key]=n_ij.get(key,0)+1; sum_ij[key]=sum_ij.get(key,0.0)+yy
    Omega = list(n_ij.keys())
    return (np.array([o[0] for o in Omega],int), np.array([o[1] for o in Omega],int),
            np.array([n_ij[o] for o in Omega],float), np.array([sum_ij[o]/n_ij[o] for o in Omega],float))

def main():
    t0 = time.time()
    np.random.seed(42)
    data_path = "/repo/data/in_house_data.json"
    output_dir = "/repo/results/section54"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Section 5.4: k=8 mixed-quality, p=5000")
    print("=" * 60)

    with open(data_path, 'r') as f: records = json.load(f)
    print(f"Total records: {len(records)}")

    # Full data
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
        if i_tmp < j_tmp: i_list.append(i_tmp); j_list.append(j_tmp)
        else: i_list.append(j_tmp); j_list.append(i_tmp); y = 1.0 - y
        k_list.append(k_tmp); y_list.append(y)

    i_all = np.array(i_list,int); j_all = np.array(j_list,int)
    k_all = np.array(k_list,int); y_all = np.array(y_list,float)
    N = len(model2id); K = len(judge2id)
    print(f"Full data: {N} models, {K} judges, {len(y_all)} comparisons")

    # Full reference model
    print("\nFitting full reference model (all 18 judges)...")
    i_r, j_r, k_r, n_r, ybar_r = aggregate_ijk(i_all, j_all, k_all, y_all)
    s_ref, gamma_ref = mle_adam_fast(N, K, i_r, j_r, k_r, n_r, ybar_r, max_iter=20000, tol=1e-5, verbose=True)

    # 8 judges
    target_judges = [
        "Qwen/Qwen2.5-7B-Instruct-Turbo",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "moonshot-v1-128k", "kimi-k2-thinking-turbo",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "google/gemma-3n-E4B-it",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
    ]
    target_set = set(target_judges)
    subset_records = [r for r in records if r.get('judge_model') in target_set]
    print(f"8-judge subset records: {len(subset_records)}")

    # Experiment
    p = 5000
    num_repeats = 20
    print(f"\nk=8, p={p}, {num_repeats} repeats")

    pearson_w_l, pearson_uw_l = [], []
    spearman_w_l, spearman_uw_l = [], []

    for run in range(num_repeats):
        indices = np.random.choice(len(subset_records), min(p, len(subset_records)), replace=False)
        sampled = [subset_records[i] for i in indices]
        m2id, j2id, i_s, j_s, k_s, y_s = build_arrays(sampled)
        Ns = len(m2id); Ks = len(j2id)

        i_w, j_w, k_w, n_w, ybar_w = aggregate_ijk(i_s, j_s, k_s, y_s)
        s_w, _ = mle_adam_fast(Ns, Ks, i_w, j_w, k_w, n_w, ybar_w, max_iter=10000, tol=1e-5)

        i_uw, j_uw, n_uw, ybar_uw = aggregate_ij(i_s, j_s, y_s)
        s_uw = mle_unweighted_fast(Ns, i_uw, j_uw, n_uw, ybar_uw, max_iter=5000, tol=1e-6)

        # Align
        id2m_sub = {v:k for k,v in m2id.items()}
        sw_a = np.full(N, np.nan); suw_a = np.full(N, np.nan); sr_a = np.full(N, np.nan)
        for isub, mn in id2m_sub.items():
            ifull = model2id[mn]; sw_a[ifull] = s_w[isub]; suw_a[ifull] = s_uw[isub]; sr_a[ifull] = s_ref[ifull]
        mask = ~np.isnan(sw_a)

        pw, _ = pearsonr(sr_a[mask], sw_a[mask])
        sw_r, _ = spearmanr(sr_a[mask], sw_a[mask])
        puw, _ = pearsonr(sr_a[mask], suw_a[mask])
        suw_r, _ = spearmanr(sr_a[mask], suw_a[mask])
        pearson_w_l.append(pw); pearson_uw_l.append(puw)
        spearman_w_l.append(sw_r); spearman_uw_l.append(suw_r)
        print(f"  Run {run+1:2d}/{num_repeats}: PW={pw:.4f} PUW={puw:.4f} SW={sw_r:.4f} SUW={suw_r:.4f}")

    avg_pw=np.mean(pearson_w_l); std_pw=np.std(pearson_w_l)
    avg_puw=np.mean(pearson_uw_l); std_puw=np.std(pearson_uw_l)
    avg_sw=np.mean(spearman_w_l); std_sw=np.std(spearman_w_l)
    avg_suw=np.mean(spearman_uw_l); std_suw=np.std(spearman_uw_l)

    print("\n"+"="*60)
    print("FINAL RESULTS (k=8, p=5000, 20 repeats)")
    print("="*60)
    print(f"\nPearson: W={avg_pw:.4f}±{std_pw:.4f} (paper:0.9394) | UW={avg_puw:.4f}±{std_puw:.4f} (paper:0.8992)")
    print(f"Spearman: W={avg_sw:.4f}±{std_sw:.4f} (paper:0.9212) | UW={avg_suw:.4f}±{std_suw:.4f} (paper:0.8316)")
    print(f"Diff: Pearson +{avg_pw-avg_puw:+.4f}, Spearman +{avg_sw-avg_suw:+.4f}")

    results = {'p':p,'num_repeats':num_repeats,
        'pearson_weighted_mean':float(avg_pw),'pearson_weighted_std':float(std_pw),
        'pearson_unweighted_mean':float(avg_puw),'pearson_unweighted_std':float(std_puw),
        'spearman_weighted_mean':float(avg_sw),'spearman_weighted_std':float(std_sw),
        'spearman_unweighted_mean':float(avg_suw),'spearman_unweighted_std':float(std_suw),
        'paper_pearson_weighted':0.9394,'paper_pearson_unweighted':0.8992,
        'paper_spearman_weighted':0.9212,'paper_spearman_unweighted':0.8316}
    with open(os.path.join(output_dir,'section54_results.json'),'w') as f: json.dump(results,f,indent=2)
    print(f"\nTotal: {time.time()-t0:.1f}s | {output_dir}/section54_results.json")
    print("Done!")

if __name__=="__main__": main()
