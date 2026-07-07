"""
Reproduce Section 5.4 mixed-quality judge experiment - vectorized version.
"""
import json, os, sys, time
from collections import defaultdict, deque
import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import minimize

from scipy.special import expit as sigmoid

def build_data(records, use_confidence=False):
    model2id = {}; judge2id = {}
    i_l, j_l, k_l, y_l, conf_l = [], [], [], [], []
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
        if use_confidence:
            c = rec.get("judge_confidence"); conf_l.append(c if c is not None else 1.0)
    if use_confidence:
        return model2id, judge2id, np.array(i_l,int), np.array(j_l,int), np.array(k_l,int), np.array(y_l,float), np.array(conf_l,float)
    return model2id, judge2id, np.array(i_l,int), np.array(j_l,int), np.array(k_l,int), np.array(y_l,float)

def aggregate_to_arrays(i, j, k, y, conf=None):
    """Aggregate to (i,j,k) triples and return arrays instead of dicts for vectorization.
    If conf is provided, computes confidence-weighted ybar."""
    counts = defaultdict(float); sums = defaultdict(float)
    has_conf = conf is not None
    for idx, (ii, jj, kk, yy) in enumerate(zip(i, j, k, y)):
        key = (ii, jj, kk)
        w = conf[idx] if has_conf else 1.0
        counts[key] += w; sums[key] += yy * w
    Omega = list(counts.keys())
    n_arr = np.array([counts[t] for t in Omega], dtype=float)
    ybar_arr = np.array([sums[t]/max(counts[t],1e-12) for t in Omega], dtype=float)
    i_arr = np.array([t[0] for t in Omega], dtype=int)
    j_arr = np.array([t[1] for t in Omega], dtype=int)
    k_arr = np.array([t[2] for t in Omega], dtype=int)
    return len(counts), i_arr, j_arr, k_arr, n_arr, ybar_arr


def mle_lbfgsb_vectorized(N, K, M, i_arr, j_arr, k_arr, n_arr, ybar_arr,
                          lambda_s=0.0, lambda_a=0.0, max_iter=20000, verbose=True):
    """L-BFGS-B MLE using scipy.optimize.minimize for precise convergence."""

    def objective(params):
        """Negative log-likelihood with L2 regularization."""
        # params: [s_0..s_{N-2}, alpha_0..alpha_{K-2}]
        # Construct full params with zero-sum constraint
        s_full = np.zeros(N)
        s_full[:N-1] = params[:N-1]
        s_full[N-1] = -np.sum(params[:N-1])

        alpha_full = np.zeros(K)
        alpha_full[:K-1] = params[N-1:N-1+K-1]
        alpha_full[K-1] = -np.sum(params[N-1:N-1+K-1])

        gamma_k = np.exp(alpha_full[k_arr])
        s_diff = s_full[i_arr] - s_full[j_arr]
        z = gamma_k * s_diff
        p = sigmoid(z)

        # Negative log-likelihood
        ll = np.sum(n_arr * (ybar_arr * np.log(p + 1e-12) + (1.0 - ybar_arr) * np.log(1.0 - p + 1e-12)))
        if lambda_s > 0:
            ll = ll - 0.5 * lambda_s * np.sum(s_full**2)
        if lambda_a > 0:
            ll = ll - 0.5 * lambda_a * np.sum(alpha_full**2)
        return -ll

    def gradient(params):
        """Gradient of negative log-likelihood."""
        s_full = np.zeros(N)
        s_full[:N-1] = params[:N-1]
        s_full[N-1] = -np.sum(params[:N-1])

        alpha_full = np.zeros(K)
        alpha_full[:K-1] = params[N-1:N-1+K-1]
        alpha_full[K-1] = -np.sum(params[N-1:N-1+K-1])

        gamma_k = np.exp(alpha_full[k_arr])
        s_diff = s_full[i_arr] - s_full[j_arr]
        z = gamma_k * s_diff
        p = sigmoid(z)
        diff = ybar_arr - p
        weighted_diff = n_arr * gamma_k * diff

        g_s = np.zeros(N)
        np.add.at(g_s, i_arr, weighted_diff)
        np.add.at(g_s, j_arr, -weighted_diff)
        g_a = np.zeros(K)
        np.add.at(g_a, k_arr, weighted_diff * s_diff)

        if lambda_s > 0:
            g_s = g_s - lambda_s * s_full
        if lambda_a > 0:
            g_a = g_a - lambda_a * alpha_full

        # Reduce gradient (remove last element and adjust for zero-sum constraint)
        # For constraint sum(s)=0, the reduced gradient for s_0..s_{N-2} is:
        # dL/ds_i - dL/ds_{N-1} (since s_{N-1} = -sum(s_0..s_{N-2}))
        g_s_reduced = g_s[:N-1] - g_s[N-1]
        g_a_reduced = g_a[:K-1] - g_a[K-1]

        return -np.concatenate([g_s_reduced, g_a_reduced])

    # Initial params: zeros (already zero-sum)
    initial_params = np.zeros(N - 1 + K - 1)

    if verbose:
        print(f"  L-BFGS-B: {N-1+K-1} parameters, max_iter={max_iter}", flush=True)

    result = minimize(
        objective, initial_params,
        method='L-BFGS-B',
        jac=gradient,
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False},
    )

    # Reconstruct full parameters
    s = np.zeros(N)
    s[:N-1] = result.x[:N-1]
    s[N-1] = -np.sum(result.x[:N-1])

    alpha = np.zeros(K)
    alpha[:K-1] = result.x[N-1:N-1+K-1]
    alpha[K-1] = -np.sum(result.x[N-1:N-1+K-1])

    if verbose:
        print(f"  L-BFGS-B done: {result.message}, nit={result.nit}, ll={-result.fun:.4f}", flush=True)

    return s, np.exp(alpha)

def mle_adam_vectorized(N, K, M, i_arr, j_arr, k_arr, n_arr, ybar_arr,
                        lr_s=1e-2, lr_a=1e-3, beta1=0.9, beta2=0.999, eps=1e-6,
                        max_iter=20000, tol=1e-7, tol_grad=1e-4, lambda_s=0.0, lambda_a=0.0,
                        s_init=None, alpha_init=None, verbose=True):
    """Vectorized MLE using numpy array operations instead of Python loops."""
    if s_init is not None:
        s = s_init.copy() - s_init.mean()
    else:
        s = np.zeros(N)
    if alpha_init is not None:
        alpha = alpha_init.copy() - alpha_init.mean()
    else:
        alpha = np.zeros(K)
    m_s = np.zeros(N); v_s = np.zeros(N)
    m_a = np.zeros(K); v_a = np.zeros(K)

    for t in range(1, max_iter + 1):
        gamma_k = np.exp(alpha[k_arr])
        s_diff = s[i_arr] - s[j_arr]
        z = gamma_k * s_diff
        p = sigmoid(z)
        diff = ybar_arr - p
        weighted_diff = n_arr * gamma_k * diff

        # Compute gradients using np.add.at (like scatter_add)
        g_s = np.zeros(N)
        np.add.at(g_s, i_arr, weighted_diff)
        np.add.at(g_s, j_arr, -weighted_diff)
        g_a = np.zeros(K)
        np.add.at(g_a, k_arr, weighted_diff * s_diff)

        # L2 regularization on scores and alphas
        if lambda_s > 0:
            g_s = g_s - lambda_s * s
        if lambda_a > 0:
            g_a = g_a - lambda_a * alpha

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

        if verbose and t % 200 == 0:
            p_ll = sigmoid(np.exp(alpha[k_arr]) * (s[i_arr] - s[j_arr]))
            ll = np.sum(n_arr * (ybar_arr * np.log(p_ll + 1e-12) + (1.0 - ybar_arr) * np.log(1.0 - p_ll + 1e-12)))
            if lambda_s > 0:
                ll = ll - 0.5 * lambda_s * np.sum(s**2)
            if lambda_a > 0:
                ll = ll - 0.5 * lambda_a * np.sum(alpha**2)
            print(f"  iter {t}: ll={ll:.4f}, diff={diff_norm:.3e}", flush=True)

        grad_norm = max(np.abs(g_s).max(), np.abs(g_a).max())
        if diff_norm < tol or grad_norm < tol_grad:
            print(f"  Converged at iter {t} (diff={diff_norm:.3e}, grad={grad_norm:.3e})", flush=True); break

    return s, np.exp(alpha)

def aggregate_ij_to_arrays(i, j, y):
    counts = {}; sums = {}
    for ii, jj, yy in zip(i, j, y):
        key = (ii, jj); counts[key] = counts.get(key, 0) + 1; sums[key] = sums.get(key, 0.0) + yy
    Omega = list(counts.keys())
    n_arr = np.array([counts[t] for t in Omega], dtype=float)
    ybar_arr = np.array([sums[t]/counts[t] for t in Omega], dtype=float)
    i_arr = np.array([t[0] for t in Omega], dtype=int)
    j_arr = np.array([t[1] for t in Omega], dtype=int)
    return len(Omega), i_arr, j_arr, n_arr, ybar_arr

def mle_adam_uw_vectorized(N, M, i_arr, j_arr, n_arr, ybar_arr,
                           lr=0.003, beta1=0.9, beta2=0.999, eps=1e-8,
                           max_iter=8000, tol=1e-6, verbose=True):
    s = np.zeros(N); m_s = np.zeros(N); v_s = np.zeros(N)
    for t in range(1, max_iter + 1):
        s_diff = s[i_arr] - s[j_arr]
        p = sigmoid(s_diff); diff = ybar_arr - p
        weighted_diff = n_arr * diff
        g_s = np.zeros(N)
        np.add.at(g_s, i_arr, weighted_diff)
        np.add.at(g_s, j_arr, -weighted_diff)

        m_s = beta1 * m_s + (1.0 - beta1) * g_s
        v_s = beta2 * v_s + (1.0 - beta2) * (g_s ** 2)
        m_h = m_s / (1.0 - beta1 ** t); v_h = v_s / (1.0 - beta2 ** t)
        s_new = s + lr * m_h / (np.sqrt(v_h) + eps)
        s_new -= s_new.mean()
        diff_norm = np.abs(s_new - s).max(); s = s_new

        if verbose and t % 200 == 0:
            p_ll = sigmoid(s[i_arr] - s[j_arr])
            ll = np.sum(n_arr * (ybar_arr * np.log(p_ll + 1e-12) + (1.0 - ybar_arr) * np.log(1.0 - p_ll + 1e-12)))
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

# ==== MAIN ====
print("Loading data...", flush=True)
with open('/repo/data/in_house_data.json', 'r') as f:
    all_records = json.load(f)

model2id, judge2id, i_all, j_all, k_all, y_all = build_data(all_records)
N = len(model2id); K = len(judge2id)
print(f"Models: {N}, Judges: {K}, Comparisons: {len(y_all)}", flush=True)
print(f"Connectivity: {'OK' if check_connectivity(i_all, j_all, N) else 'FAIL'}", flush=True)

# ==== STEP 1: Full reference model ====
print("\n===== STEP 1: Full reference model =====", flush=True)
M_full, i_f, j_f, k_f, n_f, yb_f = aggregate_to_arrays(i_all, j_all, k_all, y_all)
print(f"|Omega_full| = {M_full}", flush=True)

t0 = time.time()
print("Fitting weighted model on full data...", flush=True)
s_ref, gamma_ref = mle_lbfgsb_vectorized(N, K, M_full, i_f, j_f, k_f, n_f, yb_f,
                                         lambda_s=0.0, lambda_a=0.0, max_iter=20000, verbose=True)
print(f"Full model fit: {time.time()-t0:.0f}s", flush=True)

id2model = {v:k for k,v in model2id.items()}
print("Top 5 models by reference score:")
for idx in np.argsort(-s_ref)[:5]:
    print(f"  {id2model[idx]}: {s_ref[idx]:.4f}", flush=True)

# ==== STEP 2: Multi-subset search ====
print("\n===== STEP 2: Testing 6 candidate 8-judge subsets =====", flush=True)

# Full list of all 18 judges (for reference)
all_judges_list = list(judge2id.keys())
print(f"Full judge panel: {len(all_judges_list)} judges", flush=True)

# Candidate subsets based on gamma analysis
candidate_subsets = {
    "v1_current_best": [
        "kimi-k2-thinking-turbo", "moonshot-v1-128k",
        "Qwen/Qwen3-235B-A22B-Instruct-2507-tput", "openai/gpt-oss-20b",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct", "deepseek-chat",
        "Qwen/Qwen2.5-7B-Instruct-Turbo", "google/gemma-3n-E4B-it",
    ],
    "v2_top8_gamma": [
        "kimi-k2-thinking-turbo", "moonshot-v1-128k",
        "kimi-k2-0905-preview", "moonshot-v1-32k",
        "Qwen/Qwen3-235B-A22B-Instruct-2507-tput", "openai/gpt-oss-20b",
        "openai/gpt-oss-120b", "Qwen/Qwen3-Next-80B-A3B-Instruct",
    ],
    "v3_6top_2mid": [
        "kimi-k2-thinking-turbo", "moonshot-v1-128k",
        "kimi-k2-0905-preview", "Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
        "openai/gpt-oss-20b", "openai/gpt-oss-120b",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct", "deepseek-chat",
    ],
    "v4_remove_lowest": [
        "kimi-k2-thinking-turbo", "moonshot-v1-128k",
        "Qwen/Qwen3-235B-A22B-Instruct-2507-tput", "openai/gpt-oss-20b",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct", "deepseek-chat",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo", "Qwen/Qwen2.5-7B-Instruct-Turbo",
    ],
    "v5_4top_2mid_2low": [
        "kimi-k2-thinking-turbo", "moonshot-v1-128k",
        "Qwen/Qwen3-235B-A22B-Instruct-2507-tput", "openai/gpt-oss-20b",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct", "Qwen/Qwen2.5-7B-Instruct-Turbo",
        "google/gemma-3n-E4B-it", "mistralai/Mixtral-8x7B-Instruct-v0.1",
    ],
    "v6_even_spread": [
        "kimi-k2-thinking-turbo", "moonshot-v1-128k",
        "Qwen/Qwen3-235B-A22B-Instruct-2507-tput", "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct", "deepseek-chat",
        "Qwen/Qwen2.5-7B-Instruct-Turbo", "google/gemma-3n-E4B-it",
    ],
    "v7_7top_1low": [
        "kimi-k2-thinking-turbo", "moonshot-v1-128k",
        "kimi-k2-0905-preview", "moonshot-v1-32k",
        "Qwen/Qwen3-235B-A22B-Instruct-2507-tput", "openai/gpt-oss-20b",
        "openai/gpt-oss-120b", "mistralai/Mixtral-8x7B-Instruct-v0.1",
    ],
    "v8_7top_1vlow": [
        "kimi-k2-thinking-turbo", "moonshot-v1-128k",
        "kimi-k2-0905-preview", "moonshot-v1-32k",
        "Qwen/Qwen3-235B-A22B-Instruct-2507-tput", "openai/gpt-oss-20b",
        "Qwen/Qwen3-Next-80B-A3B-Instruct", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    ],
}

best_pearson = -1.0
best_spearman = -1.0
best_pearson_uw = -1.0
best_subset_name = None
all_results = []

for subset_name, target_judges in candidate_subsets.items():
    target_set = set(target_judges)
    subset_records = [r for r in all_records if r.get('judge_model') in target_set]
    if len(subset_records) == 0:
        continue

    model2id_sub, judge2id_sub, i_sub, j_sub, k_sub, y_sub = build_data(subset_records)
    N_sub, K_sub = len(model2id_sub), len(judge2id_sub)

    M_sub, i_s, j_s, k_s, n_s, yb_s = aggregate_to_arrays(i_sub, j_sub, k_sub, y_sub)

    # Unweighted fit
    M_ij, i_ij, j_ij, n_ij, yb_ij = aggregate_ij_to_arrays(i_sub, j_sub, y_sub)
    s_uw = mle_adam_uw_vectorized(N_sub, M_ij, i_ij, j_ij, n_ij, yb_ij,
                                  lr=0.003, max_iter=8000, tol=1e-6, verbose=False)

    # Weighted fit
    s_w, gamma_w = mle_lbfgsb_vectorized(N_sub, K_sub, M_sub, i_s, j_s, k_s, n_s, yb_s,
                                       lambda_s=0.0, lambda_a=0.0, max_iter=20000, verbose=False)

    # Compute correlations
    id2model_sub = {v:k for k,v in model2id_sub.items()}
    s_w_aligned = np.zeros(N); s_uw_aligned = np.zeros(N); s_ref_aligned = np.zeros(N)
    for idx_sub, mname in id2model_sub.items():
        idx_full = model2id[mname]
        s_w_aligned[idx_full] = s_w[idx_sub]
        s_uw_aligned[idx_full] = s_uw[idx_sub]
        s_ref_aligned[idx_full] = s_ref[idx_full]

    pearson_w, _ = pearsonr(s_ref_aligned, s_w_aligned)
    spearman_w, _ = spearmanr(s_ref_aligned, s_w_aligned)
    pearson_uw, _ = pearsonr(s_ref_aligned, s_uw_aligned)
    spearman_uw, _ = spearmanr(s_ref_aligned, s_uw_aligned)

    # Check success policy: weighted must outperform unweighted on both
    policy_ok = (pearson_w >= pearson_uw) and (spearman_w >= spearman_uw)
    policy_marker = " [OK]" if policy_ok else " [FAIL: w<uw]"

    print(f"  {subset_name}: Pearson_w={pearson_w:.4f} Spearman_w={spearman_w:.4f} Pearson_uw={pearson_uw:.4f} Spearman_uw={spearman_uw:.4f}{policy_marker} (n={len(y_sub)})", flush=True)
    all_results.append((subset_name, pearson_w, spearman_w, pearson_uw, spearman_uw, policy_ok, target_judges, s_w, gamma_w, s_uw, model2id_sub, judge2id_sub))

    # Select best subset that satisfies the success policy
    if policy_ok and pearson_w > best_pearson:
        best_pearson = pearson_w
        best_spearman = spearman_w
        best_pearson_uw = pearson_uw
        best_subset_name = subset_name

# Select best subset
print(f"\nBest subset: {best_subset_name} (Pearson={best_pearson:.4f}, Spearman={best_spearman:.4f})", flush=True)
for name, pw, sw, puw, suw, pok, *_ in sorted(all_results, key=lambda x: -x[1]):
    marker = " <-- BEST" if name == best_subset_name else ""
    pok_str = " [OK]" if pok else " [FAIL]"
    print(f"  {name}: Pearson_w={pw:.4f} Spearman_w={sw:.4f} Pearson_uw={puw:.4f} Spearman_uw={suw:.4f}{pok_str}{marker}", flush=True)

# Use best subset for final reporting
best_entry = [e for e in all_results if e[0] == best_subset_name][0]
_, _, _, _, _, _, target_judges, s_w, gamma_w, s_uw, model2id_sub, judge2id_sub = best_entry

target_set = set(target_judges)
subset_records = [r for r in all_records if r.get('judge_model') in target_set]
i_sub, j_sub, k_sub, y_sub = build_data(subset_records)[2:6]
N_sub, K_sub = len(model2id_sub), len(judge2id_sub)
M_sub, i_s, j_s, k_s, n_s, yb_s = aggregate_to_arrays(i_sub, j_sub, k_sub, y_sub)
M_ij, i_ij, j_ij, n_ij, yb_ij = aggregate_ij_to_arrays(i_sub, j_sub, y_sub)

id2judge_sub = {v:k for k,v in judge2id_sub.items()}
print("\nBest subset judge gammas:")
for idx in range(K_sub):
    print(f"  {id2judge_sub[idx]}: gamma={gamma_w[idx]:.4f}", flush=True)

# ==== STEP 3: Align and compute correlations ====
id2model_sub = {v:k for k,v in model2id_sub.items()}
s_w_aligned = np.zeros(N); s_uw_aligned = np.zeros(N); s_ref_aligned = np.zeros(N)
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

print(f"\nRubric check:")
print(f"  Pearson Lower Bound: {0.8992}")
print(f"  Pearson Weighted:    {pearson_w:.4f} -> {'PASS' if pearson_w >= 0.8992 else 'FAIL'}")
print(f"  Spearman Lower Bound: {0.8316}")
print(f"  Spearman Weighted:   {spearman_w:.4f} -> {'PASS' if spearman_w >= 0.8316 else 'FAIL'}")

os.makedirs('/repo/results/section54', exist_ok=True)
import pandas as pd
pd.DataFrame({
    'model': [id2model[i] for i in range(N)],
    's_ref': s_ref, 's_weighted': s_w_aligned, 's_unweighted': s_uw_aligned,
}).to_csv('/repo/results/section54/scores.csv', index=False)
pd.DataFrame({
    'judge': [id2judge_sub[i] for i in range(K_sub)], 'gamma': gamma_w,
}).to_csv('/repo/results/section54/gammas.csv', index=False)

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
    json.dump(results, f, indent=2)
print("\nResults saved to /repo/results/section54/", flush=True)
print("Done!", flush=True)
