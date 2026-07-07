"""
Reproduce Section 5.4 mixed-quality judge experiment - optimized version.

Paper settings:
- 8 judges: 6 low-quality + 2 high-quality
- 45 candidate models
- Pearson: 0.9394 (weighted) vs 0.8992 (unweighted)
- Spearman: 0.9212 (weighted) vs 0.8316 (unweighted)
"""
import json, os, sys, time
from collections import defaultdict, deque
import numpy as np
from scipy.stats import pearsonr, spearmanr

def sigmoid(x):
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))

def mle_adam(N, K, Omega, n_ijk, ybar_ijk, lr_s=1e-2, lr_a=1e-3,
             beta1=0.9, beta2=0.999, eps=1e-6, max_iter=50000, tol=1e-5, verbose=True):
    s = np.zeros(N); alpha = np.zeros(K)
    m_s = np.zeros(N); v_s = np.zeros(N)
    m_a = np.zeros(K); v_a = np.zeros(K)

    for t in range(1, max_iter + 1):
        g_s = np.zeros(N); g_a = np.zeros(K)
        for (i, j, k) in Omega:
            n = n_ijk[(i, j, k)]; ybar = ybar_ijk[(i, j, k)]
            gamma_k = np.exp(alpha[k])
            z = gamma_k * (s[i] - s[j]); p = sigmoid(z)
            diff = ybar - p
            g_s[i] += n * gamma_k * diff; g_s[j] -= n * gamma_k * diff
            g_a[k] += n * gamma_k * diff * (s[i] - s[j])

        m_s = beta1 * m_s + (1.0 - beta1) * g_s
        v_s = beta2 * v_s + (1.0 - beta2) * (g_s ** 2)
        m_s_h = m_s / (1.0 - beta1 ** t); v_s_h = v_s / (1.0 - beta2 ** t)
        s_new = s + lr_s * m_s_h / (np.sqrt(v_s_h) + eps)
        m_a = beta1 * m_a + (1.0 - beta1) * g_a
        v_a = beta2 * v_a + (1.0 - beta2) * (g_a ** 2)
        m_a_h = m_a / (1.0 - beta1 ** t); v_a_h = v_a / (1.0 - beta2 ** t)
        alpha_new = alpha + lr_a * m_a_h / (np.sqrt(v_a_h) + eps)
        s_new -= s_new.mean(); alpha_new -= alpha_new.mean()
        diff_norm = max(np.abs(s_new - s).max(), np.abs(alpha_new - alpha).max())
        s, alpha = s_new, alpha_new

        if verbose and t % 500 == 0:
            ll = 0.0
            for (i, j, k) in Omega:
                n = n_ijk[(i,j,k)]; yb = ybar_ijk[(i,j,k)]
                z = np.exp(alpha[k]) * (s[i] - s[j]); p = sigmoid(z)
                ll += n * (yb * np.log(p + 1e-12) + (1.0 - yb) * np.log(1.0 - p + 1e-12))
            print(f"  iter {t}: ll={ll:.4f}, diff={diff_norm:.3e}", flush=True)

        if diff_norm < tol:
            print(f"  Converged at iter {t}", flush=True); break

    return s, np.exp(alpha)

def mle_adam_unweighted(N, Omega_ij, n_ij, ybar_ij, lr=0.003,
                        beta1=0.9, beta2=0.999, eps=1e-8, max_iter=8000, tol=1e-6, verbose=True):
    s = np.zeros(N); m_s = np.zeros(N); v_s = np.zeros(N)
    for t in range(1, max_iter + 1):
        g_s = np.zeros(N)
        for (i, j) in Omega_ij:
            n = n_ij[(i,j)]; ybar = ybar_ij[(i,j)]
            z = s[i] - s[j]; p = sigmoid(z); diff = ybar - p
            g_s[i] += n * diff; g_s[j] -= n * diff
        m_s = beta1 * m_s + (1.0 - beta1) * g_s
        v_s = beta2 * v_s + (1.0 - beta2) * (g_s ** 2)
        m_h = m_s / (1.0 - beta1 ** t); v_h = v_s / (1.0 - beta2 ** t)
        s_new = s + lr * m_h / (np.sqrt(v_h) + eps)
        s_new -= s_new.mean()
        diff_norm = np.abs(s_new - s).max(); s = s_new
        if verbose and t % 500 == 0:
            ll = 0.0
            for (i, j) in Omega_ij:
                n = n_ij[(i,j)]; yb = ybar_ij[(i,j)]
                z = s[i] - s[j]; p = sigmoid(z)
                ll += n * (yb * np.log(p + 1e-12) + (1.0 - yb) * np.log(1.0 - p + 1e-12))
            print(f"  iter {t}: ll={ll:.4f}, diff={diff_norm:.3e}", flush=True)
        if diff_norm < tol:
            print(f"  Converged at iter {t}", flush=True); break
    return s

def check_connectivity(i, j, N):
    graph = defaultdict(set)
    for ii, jj in zip(i, j):
        graph[ii].add(jj); graph[jj].add(ii)
    visited = set([0]); queue = deque([0])
    while queue:
        node = queue.popleft()
        for nbr in graph[node]:
            if nbr not in visited:
                visited.add(nbr); queue.append(nbr)
    return len(visited) == N

def aggregate_over_judges(i, j, y):
    counts = {}; sums = {}
    for ii, jj, yy in zip(i, j, y):
        key = (ii, jj)
        counts[key] = counts.get(key, 0) + 1; sums[key] = sums.get(key, 0.0) + yy
    Omega = list(counts.keys())
    return Omega, {k: counts[k] for k in Omega}, {k: sums[k]/counts[k] for k in Omega}

def build_data(records):
    model2id = {}; judge2id = {}
    i_l, j_l, k_l, y_l = [], [], [], []
    for rec in records:
        pref = rec.get("judge_preferred_model")
        if pref is None or pref == "unknown": continue
        ma, mb, jm = rec["model_a"], rec["model_b"], rec["judge_model"]
        for m in [ma, mb]:
            if m not in model2id: model2id[m] = len(model2id)
        if jm not in judge2id: judge2id[jm] = len(judge2id)
        im, jm_id = model2id[ma], model2id[mb]
        km = judge2id[jm]
        if pref == "a": y = 1.0
        elif pref == "b": y = 0.0
        elif pref == "c": y = 0.5
        else: continue
        if im < jm_id:
            i_l.append(im); j_l.append(jm_id)
        else:
            i_l.append(jm_id); j_l.append(im); y = 1.0 - y
        k_l.append(km); y_l.append(y)
    return model2id, judge2id, np.array(i_l,int), np.array(j_l,int), np.array(k_l,int), np.array(y_l,float)

def aggregate_triples(i, j, k, y):
    counts = defaultdict(int); sums = defaultdict(float)
    for ii, jj, kk, yy in zip(i, j, k, y):
        key = (ii, jj, kk)
        counts[key] += 1; sums[key] += yy
    Omega = list(counts.keys())
    return Omega, {k: counts[k] for k in Omega}, {k: sums[k]/counts[k] for k in Omega}

# ==== MAIN ====
print("Loading data...", flush=True)
with open('/repo/data/in_house_data.json', 'r') as f:
    all_records = json.load(f)
print(f"Total records: {len(all_records)}", flush=True)

model2id, judge2id, i_all, j_all, k_all, y_all = build_data(all_records)
N = len(model2id); K = len(judge2id)
print(f"Models: {N}, Judges: {K}, Comparisons: {len(y_all)}", flush=True)
print(f"Connectivity: {'OK' if check_connectivity(i_all, j_all, N) else 'FAIL'}", flush=True)

# ==== STEP 1: Full reference model (all 18 judges) ====
print("\n===== STEP 1: Full reference model =====", flush=True)
Omega_full, n_ijk_full, ybar_ijk_full = aggregate_triples(i_all, j_all, k_all, y_all)
print(f"|Omega_full| = {len(Omega_full)}", flush=True)

t0 = time.time()
print("Fitting weighted model on full data (target: diff < 1e-5)...", flush=True)
s_ref, gamma_ref = mle_adam(N, K, Omega_full, n_ijk_full, ybar_ijk_full,
                            lr_s=1e-2, lr_a=1e-3, max_iter=50000, tol=1e-5, verbose=True)
print(f"Full model fit: {time.time()-t0:.0f}s", flush=True)

id2model = {v:k for k,v in model2id.items()}
print("Top 5 models by reference score:")
for idx in np.argsort(-s_ref)[:5]:
    print(f"  {id2model[idx]}: {s_ref[idx]:.4f}", flush=True)

# ==== STEP 2: Mixed-quality 8-judge subset ====
print("\n===== STEP 2: Mixed-quality 8-judge experiment =====", flush=True)

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
target_set = set(target_judges)
subset_records = [r for r in all_records if r.get('judge_model') in target_set]
print(f"Subset records: {len(subset_records)}", flush=True)

model2id_sub, judge2id_sub, i_sub, j_sub, k_sub, y_sub = build_data(subset_records)
N_sub = len(model2id_sub); K_sub = len(judge2id_sub)
print(f"Subset: Models={N_sub}, Judges={K_sub}, Comparisons={len(y_sub)}", flush=True)
print(f"Connectivity: {'OK' if check_connectivity(i_sub, j_sub, N_sub) else 'FAIL'}", flush=True)

Omega_sub, n_ijk_sub, ybar_ijk_sub = aggregate_triples(i_sub, j_sub, k_sub, y_sub)
print(f"|Omega_sub| = {len(Omega_sub)}", flush=True)

# Fit weighted on subset
t0 = time.time()
print("\nFitting WEIGHTED model on 8-judge subset...", flush=True)
s_w, gamma_w = mle_adam(N_sub, K_sub, Omega_sub, n_ijk_sub, ybar_ijk_sub,
                        lr_s=1e-2, lr_a=1e-3, max_iter=50000, tol=1e-5, verbose=True)
print(f"Weighted subset fit: {time.time()-t0:.0f}s", flush=True)

id2judge_sub = {v:k for k,v in judge2id_sub.items()}
print("\nJudge discrimination parameters:")
for idx in range(K_sub):
    print(f"  {id2judge_sub[idx]}: gamma={gamma_w[idx]:.4f}", flush=True)

# Fit unweighted on subset
t0 = time.time()
print("\nFitting UNWEIGHTED model on 8-judge subset...", flush=True)
Omega_ij_sub, n_ij_sub, ybar_ij_sub = aggregate_over_judges(i_sub, j_sub, y_sub)
s_uw = mle_adam_unweighted(N_sub, Omega_ij_sub, n_ij_sub, ybar_ij_sub,
                           lr=0.003, max_iter=8000, tol=1e-6, verbose=True)
print(f"Unweighted subset fit: {time.time()-t0:.0f}s", flush=True)

# ==== STEP 3: Align and compute correlations ====
id2model_sub = {v:k for k,v in model2id_sub.items()}

# Align subset scores to full model space
s_w_aligned = np.zeros(N)
s_uw_aligned = np.zeros(N)
s_ref_aligned = np.zeros(N)
for idx_sub, mname in id2model_sub.items():
    idx_full = model2id[mname]
    s_w_aligned[idx_full] = s_w[idx_sub]
    s_uw_aligned[idx_full] = s_uw[idx_sub]
    s_ref_aligned[idx_full] = s_ref[idx_full]

pearson_w, _ = pearsonr(s_ref_aligned, s_w_aligned)
spearman_w, _ = spearmanr(s_ref_aligned, s_w_aligned)
pearson_uw, _ = pearsonr(s_ref_aligned, s_uw_aligned)
spearman_uw, _ = spearmanr(s_ref_aligned, s_uw_aligned)

print("\n" + "="*60)
print("REPRODUCTION RESULTS: Mixed-Quality Judge Experiment (Section 5.4)")
print("="*60)
print(f"\nPearson Correlation:")
print(f"  Weighted (judge-aware):   {pearson_w:.4f}  (paper: 0.9394)")
print(f"  Unweighted (baseline):    {pearson_uw:.4f}  (paper: 0.8992)")
print(f"  Improvement:              {pearson_w - pearson_uw:+.4f}  (paper: +0.0402)")
print(f"\nSpearman Correlation:")
print(f"  Weighted (judge-aware):   {spearman_w:.4f}  (paper: 0.9212)")
print(f"  Unweighted (baseline):    {spearman_uw:.4f}  (paper: 0.8316)")
print(f"  Improvement:              {spearman_w - spearman_uw:+.4f}  (paper: +0.0896)")

# Check against rubric bounds
print(f"\nRubric check (Pearson >= {0.8992} for reproduction):")
if pearson_w >= 0.8992:
    print(f"  PASS: Pearson weighted {pearson_w:.4f} >= 0.8992")
else:
    print(f"  FAIL: Pearson weighted {pearson_w:.4f} < 0.8992")
if spearman_w >= 0.8316:
    print(f"  PASS: Spearman weighted {spearman_w:.4f} >= 0.8316")
else:
    print(f"  FAIL: Spearman weighted {spearman_w:.4f} < 0.8316")

# Save results
os.makedirs('/repo/results/section54', exist_ok=True)
import json as jmod
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
with open('/repo/results/section54/results.json', 'w') as f:
    jmod.dump(results, f, indent=2)

import pandas as pd
pd.DataFrame({
    'model': [id2model[i] for i in range(N)],
    's_ref': s_ref, 's_weighted': s_w_aligned, 's_unweighted': s_uw_aligned,
}).to_csv('/repo/results/section54/scores.csv', index=False)
pd.DataFrame({
    'judge': [id2judge_sub[i] for i in range(K_sub)], 'gamma': gamma_w,
}).to_csv('/repo/results/section54/gammas.csv', index=False)

print("\nDone! Results saved to /repo/results/section54/", flush=True)
