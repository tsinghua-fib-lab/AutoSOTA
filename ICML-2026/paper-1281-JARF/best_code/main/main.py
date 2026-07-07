"""
Bradley-Terry model with judge-specific scales.

Pipeline:
1. Load pairwise comparison data from JSON.
2. Map models / judges / prompts to integer IDs.
3. Aggregate comparisons to (i, j, k) and (i, j) level.
4. Fit:
   - weighted MLE with judge-specific judge scales via Adam (mle_adam)
   - unweighted Bradley-Terry via Adam (mle_adam_unweighted)
5. Summarize fitted scores for models and judges as pandas DataFrames.
"""
import json
from pathlib import Path
from collections import Counter, defaultdict, deque
from typing import Optional, Tuple, Dict, List
import os
import random

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from typing import Optional, Tuple, Dict, List
from collections import Counter
from itertools import combinations
from scipy.stats import pearsonr, spearmanr

def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        records = json.load(f)
    print("Total records:", len(records))
    print("First record example:")
    print(records[0])

    model2id: Dict[str, int] = {}
    judge2id: Dict[str, int] = {}
    qid2id: Dict[tuple, int] = {}

    i_list, j_list, k_list, p_list, y_list = [], [], [], [], []

    for rec in records:
        pref = rec.get("judge_preferred_model")
        if pref is None or pref == "unknown":
            continue

        ma = rec["model_a"]
        mb = rec["model_b"]
        jm = rec["judge_model"]

        if ma not in model2id:
            model2id[ma] = len(model2id)
        if mb not in model2id:
            model2id[mb] = len(model2id)
        i_tmp = model2id[ma]
        j_tmp = model2id[mb]

        if jm not in judge2id:
            judge2id[jm] = len(judge2id)
        k_tmp = judge2id[jm]

        qid_tuple = tuple(rec["question_id"])
        if qid_tuple not in qid2id:
            qid2id[qid_tuple] = len(qid2id)
        p_tmp = qid2id[qid_tuple]

        if pref == "a":
            y_tmp = 1.0
        elif pref == "b":
            y_tmp = 0.0
        elif pref == "c":
            y_tmp = 0.5
        else:
            continue

        if i_tmp < j_tmp:
            i_list.append(i_tmp)
            j_list.append(j_tmp)
            k_list.append(k_tmp)
            p_list.append(p_tmp)
            y_list.append(y_tmp)
        else:
            i_list.append(j_tmp)
            j_list.append(i_tmp)
            k_list.append(k_tmp)
            p_list.append(p_tmp)
            y_list.append(1.0 - y_tmp)

    i = np.array(i_list, int)
    j = np.array(j_list, int)
    k = np.array(k_list, int)
    p = np.array(p_list, int)
    y = np.array(y_list, float)

    print("models:", len(model2id))
    print("judges:", len(judge2id))
    print("prompts:", len(qid2id))
    print("samples after dropping unknown:", len(y))

    return records, model2id, judge2id, qid2id, i, j, k, p, y

def fit_full_model(N, K, Omega, n_ijk, ybar_ijk, i, j, k, p, y, model2id, judge2id, base_name):
    is_connected = check_connectivity(i, j, N)
    print(f"\nComparison graph connectivity: {'✓ Connected' if is_connected else '✗ Not connected'}")
    if not is_connected:
        print("Warning: Some models cannot be compared with others, which may cause estimation issues!")

    print("\n=== Fitting weighted MLE with judge scales ===")
    s_hat, gamma_hat = mle_adam(
        N,
        K,
        Omega,
        n_ijk,
        ybar_ijk,
        lr_s=1e-2,
        lr_a=1e-3,
        beta1=0.9,
        beta2=0.999,
        max_iter=10000,
        tol=1e-5,
        verbose=True,
    )

    df_models = make_model_df(model2id, i, j, s_hat)
    df_judges = make_judge_df(judge2id, k, gamma_hat)

    print("\nTop models by #comparisons:")
    print(df_models)

    print("\nJudges summary:")
    print(df_judges)

    print("\n=== Fitting unweighted Bradley-Terry MLE ===")
    Omega_ij, n_ij, ybar_ij = aggregate_over_judges(i, j, k, p, y)
    print("|Omega_ij| =", len(Omega_ij))

    s_hat_unweighted = mle_adam_unweighted(
        N,
        Omega_ij,
        n_ij,
        ybar_ij,
        lr=0.003,
        max_iter=8000,
        verbose=True,
    )

    df_models_unweighted = make_model_df(model2id, i, j, s_hat_unweighted)
    print("\nUnweighted BT model summary:")
    print(df_models_unweighted)

    Sigma_hat, A_s, A_alpha = Sigma_vartheta(
        s_hat,
        gamma_hat,
        n_ijk,
    )

    Sigma_s_alpha_hat, Sigma_s_gamma_hat = Sigma_vartheta_to_Sigma_theta(
        Sigma_hat,
        A_s,
        A_alpha,
        gamma_hat,
    )

    T = float(sum(n_ijk.values()))
    
    Sigma_u_hat, A_s_u = Sigma_u(s_hat_unweighted, n_ij)
    Sigma_s_hat = Sigma_u_to_Sigma_s(Sigma_u_hat, A_s_u)
    T_u = sum(n_ij.values())
    unweighted_s_cis = [ci_for_s(s_hat_unweighted, Sigma_s_hat, T_u, idx=i)[1] for i in range(N)]
    
    s_cis = []
    for idx in range(N):
        s_val, (lower, upper) = ci_for_s_or_gamma(s_hat, gamma_hat, Sigma_s_gamma_hat, T, which="s", idx=idx)
        s_cis.append((lower, upper))
    
    gamma_cis = []
    for idx in range(K):
        gamma_val, (lower, upper) = ci_for_s_or_gamma(s_hat, gamma_hat, Sigma_s_gamma_hat, T, which="gamma", idx=idx)
        gamma_cis.append((lower, upper))
    
    id2model = {v: k for k, v in model2id.items()}
    id2judge = {v: k for k, v in judge2id.items()}
    
    df_models_full = pd.DataFrame({
        'model': [id2model[i] for i in range(N)],
        's_hat_weighted': s_hat,
        's_hat_unweighted': s_hat_unweighted,
        'weighted_s_hat_CI_lower': [ci[0] if ci else None for ci in s_cis],
        'weighted_s_hat_CI_upper': [ci[1] if ci else None for ci in s_cis],
        'unweighted_s_hat_CI_lower': [ci[0] if ci else None for ci in unweighted_s_cis],
        'unweighted_s_hat_CI_upper': [ci[1] if ci else None for ci in unweighted_s_cis],
    })
    
    df_judges_full = pd.DataFrame({
        'judge': [id2judge[k] for k in range(K)],
        'gamma_hat': gamma_hat,
        'gamma_hat_CI_lower': [ci[0] if ci else None for ci in gamma_cis],
        'gamma_hat_CI_upper': [ci[1] if ci else None for ci in gamma_cis],
    })
    
    full_model_excel_path = f'{base_name}/full_model_results.xlsx'
    with pd.ExcelWriter(full_model_excel_path, engine='openpyxl') as writer:
        df_models_full.to_excel(writer, sheet_name='models', index=False)
        df_judges_full.to_excel(writer, sheet_name='judges', index=False)
    
    print(f"Results saved to {os.path.abspath(full_model_excel_path)}")

    return s_hat, gamma_hat, s_hat_unweighted, Sigma_s_alpha_hat, Sigma_s_gamma_hat, T, id2model, id2judge

def process_subdataset(sampled_records, id2model, id2judge, N, K, base_name, k, p, iter_j, writer):
    model2id_sub = {}
    judge2id_sub = {}
    qid2id_sub = {}
    
    i_list_sub, j_list_sub, k_list_sub, p_list_sub, y_list_sub = [], [], [], [], []
    
    for rec in sampled_records:
        pref = rec.get("judge_preferred_model")
        if pref is None or pref == "unknown":
            continue
        
        ma = rec["model_a"]
        mb = rec["model_b"]
        jm = rec["judge_model"]
        
        if ma not in model2id_sub:
            model2id_sub[ma] = len(model2id_sub)
        if mb not in model2id_sub:
            model2id_sub[mb] = len(model2id_sub)
        i_tmp = model2id_sub[ma]
        j_tmp = model2id_sub[mb]
        
        if jm not in judge2id_sub:
            judge2id_sub[jm] = len(judge2id_sub)
        k_tmp = judge2id_sub[jm]
        
        qid_tuple = tuple(rec["question_id"])
        if qid_tuple not in qid2id_sub:
            qid2id_sub[qid_tuple] = len(qid2id_sub)
        p_tmp = qid2id_sub[qid_tuple]
        
        if pref == "a":
            y_tmp = 1.0
        elif pref == "b":
            y_tmp = 0.0
        elif pref == "c":
            y_tmp = 0.5
        else:
            continue
        
        if i_tmp < j_tmp:
            i_list_sub.append(i_tmp)
            j_list_sub.append(j_tmp)
            k_list_sub.append(k_tmp)
            p_list_sub.append(p_tmp)
            y_list_sub.append(y_tmp)
        else:
            i_list_sub.append(j_tmp)
            j_list_sub.append(i_tmp)
            k_list_sub.append(k_tmp)
            p_list_sub.append(p_tmp)
            y_list_sub.append(1.0 - y_tmp)
    
    i_sub = np.array(i_list_sub, int)
    j_sub = np.array(j_list_sub, int)
    k_sub = np.array(k_list_sub, int)
    p_sub = np.array(p_list_sub, int)
    y_sub = np.array(y_list_sub, float)
    
    N_sub = len(model2id_sub)
    K_sub = len(judge2id_sub)
    
    Omega_sub = list(set(zip(i_sub, j_sub, k_sub)))
    n_ijk_sub = defaultdict(int)
    ybar_ijk_sub = defaultdict(float)
    for ii, jj, kk, yy in zip(i_sub, j_sub, k_sub, y_sub):
        n_ijk_sub[(ii, jj, kk)] += 1
        ybar_ijk_sub[(ii, jj, kk)] += yy
    for key in ybar_ijk_sub:
        ybar_ijk_sub[key] /= n_ijk_sub[key]
    
    s_hat_sub, gamma_hat_sub = mle_adam(N_sub, K_sub, Omega_sub, n_ijk_sub, ybar_ijk_sub)
    
    Sigma_hat_sub, A_s_sub, A_alpha_sub = Sigma_vartheta(s_hat_sub, gamma_hat_sub, n_ijk_sub)
    Sigma_s_alpha_hat_sub, Sigma_s_gamma_hat_sub = Sigma_vartheta_to_Sigma_theta(Sigma_hat_sub, A_s_sub, A_alpha_sub, gamma_hat_sub)
    T_sub = float(sum(n_ijk_sub.values()))
    
    s_cis_sub = []
    for idx in range(N_sub):
        s_val, (lower, upper) = ci_for_s_or_gamma(s_hat_sub, gamma_hat_sub, Sigma_s_gamma_hat_sub, T_sub, which="s", idx=idx)
        s_cis_sub.append((lower, upper))
    
    gamma_cis_sub = []
    for idx in range(K_sub):
        gamma_val, (lower, upper) = ci_for_s_or_gamma(s_hat_sub, gamma_hat_sub, Sigma_s_gamma_hat_sub, T_sub, which="gamma", idx=idx)
        gamma_cis_sub.append((lower, upper))
    
    Omega_ij_sub, n_ij_sub, ybar_ij_sub = aggregate_over_judges(i_sub, j_sub, k_sub, p_sub, y_sub)
    s_hat_unweighted_sub = mle_adam_unweighted(N_sub, Omega_ij_sub, n_ij_sub, ybar_ij_sub)
    
    Sigma_u_hat, A_s_u = Sigma_u(s_hat_unweighted_sub, n_ij_sub)
    Sigma_s_hat = Sigma_u_to_Sigma_s(Sigma_u_hat, A_s_u)
    T_u = sum(n_ij_sub.values())
    unweighted_s_cis_sub = []
    for idx in range(N_sub):
        s_val, (lower, upper) = ci_for_s(s_hat_unweighted_sub, Sigma_s_hat, T_u, idx=idx)
        unweighted_s_cis_sub.append((lower, upper))
    
    id2model_sub = {v: k for k, v in model2id_sub.items()}
    common_models = set(id2model.values()) & set(id2model_sub.values())
    if len(common_models) != N_sub:
        print(f"Model mismatch: Subset has {N_sub} models, but common models only {len(common_models)}")
        return None, None, None, None, None
    
    s_sub_weighted_aligned = np.zeros(N)
    s_sub_unweighted_aligned = np.zeros(N)
    for idx_sub, model in enumerate(id2model_sub.values()):
        idx_gt = [i for i, m in enumerate(id2model.values()) if m == model][0]
        s_sub_weighted_aligned[idx_gt] = s_hat_sub[idx_sub]
        s_sub_unweighted_aligned[idx_gt] = s_hat_unweighted_sub[idx_sub]
    
    s_cis_aligned = [None] * N
    unweighted_s_cis_aligned = [None] * N
    for idx_sub, model in enumerate(id2model_sub.values()):
        idx_gt = [i for i, m in enumerate(id2model.values()) if m == model][0]
        s_cis_aligned[idx_gt] = s_cis_sub[idx_sub]
        unweighted_s_cis_aligned[idx_gt] = unweighted_s_cis_sub[idx_sub]
    
    gamma_hat_aligned = np.zeros(K)
    gamma_cis_aligned = [None] * K
    id2judge_sub = {v: k for k, v in judge2id_sub.items()}
    for idx_sub, judge in enumerate(id2judge_sub.values()):
        idx_gt = [i for i, j in enumerate(id2judge.values()) if j == judge][0]
        gamma_hat_aligned[idx_gt] = gamma_hat_sub[idx_sub]
        gamma_cis_aligned[idx_gt] = gamma_cis_sub[idx_sub]
    
    df_detailed = pd.DataFrame({
        'model': list(id2model.values()),
        'weighted_s_hat': s_sub_weighted_aligned,
        'unweighted_s_hat': s_sub_unweighted_aligned,
        'judge': list(id2judge.values()) + [None] * (N - K),
        'gamma_hat': list(gamma_hat_aligned) + [None] * (N - K),
    })
    df_detailed['weighted_s_hat_CI_lower'] = [ci[0] if ci else None for ci in s_cis_aligned]
    df_detailed['weighted_s_hat_CI_upper'] = [ci[1] if ci else None for ci in s_cis_aligned]
    df_detailed['unweighted_s_hat_CI_lower'] = [ci[0] if ci else None for ci in unweighted_s_cis_aligned]
    df_detailed['unweighted_s_hat_CI_upper'] = [ci[1] if ci else None for ci in unweighted_s_cis_aligned]
    df_detailed['gamma_hat_CI_lower'] = [ci[0] if ci else None for ci in gamma_cis_aligned] + [None] * (N - K)
    df_detailed['gamma_hat_CI_upper'] = [ci[1] if ci else None for ci in gamma_cis_aligned] + [None] * (N - K)
    
    sheet_name = f"k_{k}_p_{p}_iter_{iter_j}"
    df_detailed.to_excel(writer, sheet_name=sheet_name, index=False)
    
    return s_sub_weighted_aligned, s_sub_unweighted_aligned, gamma_hat_aligned, s_cis_aligned, unweighted_s_cis_aligned, gamma_cis_aligned

def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x))
    )


def loglik(
    s: np.ndarray,
    alpha: np.ndarray,
    Omega,
    n_ijk: Dict[tuple, int],
    ybar_ijk: Dict[tuple, float],
) -> float:
    ll = 0.0
    for (i, j, k) in Omega:
        gamma_k = np.exp(alpha[k])
        z = gamma_k * (s[i] - s[j])
        p = sigmoid(z)
        ybar = ybar_ijk[(i, j, k)]
        n = n_ijk[(i, j, k)]
        ll += n * (
            ybar * np.log(p + 1e-12)
            + (1.0 - ybar) * np.log(1.0 - p + 1e-12)
        )
    return ll


def mle_adam(
    N: int,
    K: int,
    Omega,
    n_ijk: Dict[tuple, int],
    ybar_ijk: Dict[tuple, float],
    lr_s: float = 1e-2,
    lr_a: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-6,
    max_iter: int = 100000,
    tol: float = 1e-5,
    verbose: bool = True,
    s_init: Optional[np.ndarray] = None,
    alpha_init: Optional[np.ndarray] = None,
):
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

    m_s = np.zeros_like(s)
    v_s = np.zeros_like(s)
    m_a = np.zeros_like(alpha)
    v_a = np.zeros_like(alpha)

    for t in range(1, max_iter + 1):
        g_s = np.zeros_like(s)
        g_a = np.zeros_like(alpha)

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

        diff_norm = max(
            np.linalg.norm(s_new - s),
            np.linalg.norm(alpha_new - alpha),
        )

        s, alpha = s_new, alpha_new

        if t % 1000 == 0 or t == max_iter:
            np.savez("mle_last.npz", s=s, gamma=np.exp(alpha))

        if verbose and (t % 200 == 0 or t == max_iter):
            ll = loglik(s, alpha, Omega, n_ijk, ybar_ijk)
            print(
                f"  iter {t}: ll={ll:.4f}, "
                f"diff={diff_norm:.3e}, grad={grad_norm:.3e}"
            )

        if diff_norm < tol:
            break

    gamma_hat = np.exp(alpha)
    return s, gamma_hat


def check_connectivity(i: np.ndarray, j: np.ndarray, N: int) -> bool:
    graph = defaultdict(set)
    for ii, jj in zip(i, j):
        graph[ii].add(jj)
        graph[jj].add(ii)

    visited = set([0])
    queue = deque([0])
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return len(visited) == N


def make_model_df(
    model2id: Dict[str, int],
    i: np.ndarray,
    j: np.ndarray,
    s_hat: np.ndarray,
) -> pd.DataFrame:
    counts = Counter()
    for ii, jj in zip(i, j):
        counts[ii] += 1
        counts[jj] += 1

    id2model = {v: k for k, v in model2id.items()}

    rows = []
    N = len(s_hat)
    for idx in range(N):
        rows.append(
            {
                "model": id2model.get(idx, f"model_{idx}"),
                "s_hat": float(s_hat[idx]),
                "n_comp": counts.get(idx, 0),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("n_comp", ascending=False).reset_index(drop=True)
    return df


def make_judge_df(
    judge2id: Dict[str, int],
    k: np.ndarray,
    gamma_hat: np.ndarray,
) -> pd.DataFrame:
    counts = Counter(k.tolist())
    id2judge = {v: name for name, v in judge2id.items()}

    rows = []
    K = len(gamma_hat)
    for idx in range(K):
        rows.append(
            {
                "judge": id2judge.get(idx, f"judge_{idx}"),
                "gamma_hat": float(gamma_hat[idx]),
                "n_comp": counts.get(idx, 0),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("n_comp", ascending=False).reset_index(drop=True)
    return df


def aggregate_over_judges(
    i: np.ndarray,
    j: np.ndarray,
    k: np.ndarray,
    p: np.ndarray,
    y: np.ndarray,
):
    counts: Dict[tuple, int] = {}
    sums: Dict[tuple, float] = {}

    for ii, jj, yy in zip(i, j, y):
        key = (ii, jj)
        counts[key] = counts.get(key, 0) + 1
        sums[key] = sums.get(key, 0.0) + yy

    Omega_ij = list(counts.keys())
    n_ij = {key: counts[key] for key in Omega_ij}
    ybar_ij = {key: sums[key] / counts[key] for key in Omega_ij}
    return Omega_ij, n_ij, ybar_ij


def loglik_unweighted(
    s: np.ndarray,
    Omega_ij,
    n_ij: Dict[tuple, int],
    ybar_ij: Dict[tuple, float],
) -> float:
    ll = 0.0
    for (i, j) in Omega_ij:
        z = s[i] - s[j]
        p = sigmoid(z)
        ybar = ybar_ij[(i, j)]
        n = n_ij[(i, j)]
        ll += n * (
            ybar * np.log(p + 1e-12)
            + (1.0 - ybar) * np.log(1.0 - p + 1e-12)
        )
    return ll


def mle_adam_unweighted(
    N: int,
    Omega_ij,
    n_ij: Dict[tuple, int],
    ybar_ij: Dict[tuple, float],
    lr: float = 0.003,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    max_iter: int = 8000,
    tol: float = 1e-6,
    verbose: bool = True,
    s_init: Optional[np.ndarray] = None,
) -> np.ndarray:
    if s_init is None:
        s = np.zeros(N, float)
    else:
        s = np.array(s_init, float)
    s -= s.mean()

    m_s = np.zeros_like(s)
    v_s = np.zeros_like(s)

    for t in range(1, max_iter + 1):
        g_s = np.zeros_like(s)

        for (i, j) in Omega_ij:
            n = n_ij[(i, j)]
            ybar = ybar_ij[(i, j)]
            z = s[i] - s[j]
            p = sigmoid(z)
            diff = ybar - p
            g_s[i] += n * diff
            g_s[j] -= n * diff

        grad_norm = np.linalg.norm(g_s)

        m_s = beta1 * m_s + (1.0 - beta1) * g_s
        v_s = beta2 * v_s + (1.0 - beta2) * (g_s ** 2)
        m_hat = m_s / (1.0 - beta1 ** t)
        v_hat = v_s / (1.0 - beta2 ** t)
        s_new = s + lr * m_hat / (np.sqrt(v_hat) + eps)

        s_new -= s_new.mean()

        diff_norm = np.linalg.norm(s_new - s)
        s = s_new

        if verbose and (t % 200 == 0 or t == max_iter):
            ll = loglik_unweighted(s, Omega_ij, n_ij, ybar_ij)
            print(
                f"iter {t}: ll={ll:.4f}, "
                f"diff={diff_norm:.3e}, grad={grad_norm:.3e}"
            )

        if diff_norm < tol and grad_norm < tol:
            break

    return s


def orthonormal_zero_sum_basis(N: int) -> np.ndarray:
    B = np.zeros((N, N-1), dtype=float)
    B[:N-1, :N-1] = np.eye(N-1)
    B[N-1, :] = -1.0

    Q, _ = np.linalg.qr(B)

    A = Q[:, :N-1]
    return A


def compute_I_vartheta_from_dict(
    s_hat: np.ndarray,
    gamma_hat: np.ndarray,
    n_ijk_aggr: dict,
) -> np.ndarray:

    s_hat = np.asarray(s_hat, dtype=float)
    gamma_hat = np.asarray(gamma_hat, dtype=float)

    N = s_hat.shape[0]
    K = gamma_hat.shape[0]

    if N < 2 or K < 1:
        raise ValueError("At least N>=2 models and K>=1 judges are required.")

    d = (N - 1) + (K - 1)

    A_s = orthonormal_zero_sum_basis(N)
    A_alpha = orthonormal_zero_sum_basis(K)

    T = float(sum(n_ijk_aggr.values()))
    if T <= 0:
        raise ValueError("Total comparison count T must be positive.")

    I_hat = np.zeros((d, d), dtype=float)

    for (i_idx, j_idx, k_idx), n_ijk in n_ijk_aggr.items():
        n_ijk = float(n_ijk)
        if n_ijk <= 0:
            continue

        if not (0 <= i_idx < N and 0 <= j_idx < N and 0 <= k_idx < K):
            raise IndexError(
                f"Index out of bounds: i={i_idx}, j={j_idx}, k={k_idx}, "
                f"but N={N}, K={K}"
            )

        gamma_k = gamma_hat[k_idx]
        s_i = s_hat[i_idx]
        s_j = s_hat[j_idx]

        z = gamma_k * (s_i - s_j)

        p = sigmoid(z)

        w = p * (1.0 - p)

        if w == 0.0:
            continue

        g_u = gamma_k * (A_s[i_idx, :] - A_s[j_idx, :])

        g_v = gamma_k * (s_i - s_j) * A_alpha[k_idx, :]

        g = np.concatenate([g_u, g_v])

        weight = (n_ijk / T) * w
        I_hat += weight * np.outer(g, g)

    return I_hat, A_s, A_alpha


def Sigma_vartheta(
    s_hat: np.ndarray,
    gamma_hat: np.ndarray,
    n_ijk_aggr: dict,
    ridge: float = 0.0,
) -> np.ndarray:

    I_hat, A_s, A_alpha = compute_I_vartheta_from_dict(s_hat, gamma_hat, n_ijk_aggr)
    I_hat = 0.5 * (I_hat + I_hat.T)
    if ridge > 0:
        I_hat = I_hat + ridge * np.eye(I_hat.shape[0])


    Sigma_hat = np.linalg.inv(I_hat)
    return Sigma_hat, A_s, A_alpha


def Sigma_vartheta_to_Sigma_theta(
    Sigma_vartheta_hat: np.ndarray,
    A_s: np.ndarray,
    A_alpha: np.ndarray,
    gamma_hat: np.ndarray,
):

    Sigma_vartheta_hat = np.asarray(Sigma_vartheta_hat, dtype=float)
    A_s = np.asarray(A_s, dtype=float)
    A_alpha = np.asarray(A_alpha, dtype=float)
    gamma_hat = np.asarray(gamma_hat, dtype=float)

    N, Ns1 = A_s.shape
    K, Ks1 = A_alpha.shape
    d = Sigma_vartheta_hat.shape[0]

    assert d == (Ns1 + Ks1), "Σ̂_vartheta dimension must equal (N-1)+(K-1)"

    J = np.block([
        [A_s, np.zeros((N, Ks1))],
        [np.zeros((K, Ns1)), A_alpha ],
    ])

    Sigma_s_alpha_hat = J @ Sigma_vartheta_hat @ J.T

    Sigma_ss_hat      = Sigma_s_alpha_hat[:N, :N]
    Sigma_salpha_hat  = Sigma_s_alpha_hat[:N, N:]
    Sigma_alphas_hat  = Sigma_s_alpha_hat[N:, :N]
    Sigma_alphaalpha_hat = Sigma_s_alpha_hat[N:, N:]

    G = np.diag(gamma_hat)

    upper_right = Sigma_salpha_hat @ G
    lower_left  = G @ Sigma_alphas_hat
    lower_right = G @ Sigma_alphaalpha_hat @ G

    Sigma_s_gamma_hat = np.block([
        [Sigma_ss_hat,  upper_right],
        [lower_left,    lower_right],
    ])

    return Sigma_s_alpha_hat, Sigma_s_gamma_hat


def ci_for_s_or_gamma(
    s_hat: np.ndarray,
    gamma_hat: np.ndarray,
    Sigma_s_gamma_hat: np.ndarray,
    T: float,
    alpha_level: float = 0.05,
    which: str = "s",
    idx: int = 0,
):
    from scipy.stats import norm

    s_hat = np.asarray(s_hat, dtype=float)
    gamma_hat = np.asarray(gamma_hat, dtype=float)

    N = s_hat.shape[0]
    K = gamma_hat.shape[0]

    theta_hat = np.concatenate([s_hat, gamma_hat])

    if which == "s":
        r = idx
    elif which == "gamma":
        r = N + idx
    else:
        raise ValueError("which must be 's' or 'gamma'")

    Sigma_rr = Sigma_s_gamma_hat[r, r]

    var_hat = Sigma_rr / T
    if var_hat < 0:
        if var_hat > -1e-10:
            var_hat = 0.0
        else:
            raise RuntimeError(f"Negative var_hat={var_hat} at r={r}")
    se_hat = np.sqrt(var_hat)

    z = norm.ppf(1 - alpha_level / 2)

    lower = theta_hat[r] - z * se_hat
    upper = theta_hat[r] + z * se_hat

    return theta_hat[r], (lower, upper)


def compute_I_u_from_dict(
    s_hat: np.ndarray,
    n_ij_aggr: dict,
) -> np.ndarray:

    s_hat = np.asarray(s_hat, dtype=float)

    N = s_hat.shape[0]

    if N < 2:
        raise ValueError("At least N>=2 models are required.")

    d = (N - 1)

    A_s = orthonormal_zero_sum_basis(N)

    T = float(sum(n_ij_aggr.values()))
    if T <= 0:
        raise ValueError("Total comparison count T must be positive.")

    I_hat = np.zeros((d, d), dtype=float)

    for (i_idx, j_idx), n_ij in n_ij_aggr.items():
        n_ij = float(n_ij)
        if n_ij <= 0:
            continue

        if not (0 <= i_idx < N and 0 <= j_idx < N):
            raise IndexError(
                f"Index out of bounds: i={i_idx}, j={j_idx}, "
                f"but N={N}"
            )

        s_i = s_hat[i_idx]
        s_j = s_hat[j_idx]

        z = s_i - s_j

        p = sigmoid(z)

        w = p * (1.0 - p)

        if w == 0.0:
            continue

        g_u = (A_s[i_idx, :] - A_s[j_idx, :])

        weight = (n_ij / T) * w
        I_hat += weight * np.outer(g_u, g_u)

    return I_hat, A_s


def Sigma_u(
    s_hat: np.ndarray,
    n_ij_aggr: dict,
    ridge: float = 0.0,
) -> np.ndarray:

    I_hat, A_s = compute_I_u_from_dict(s_hat, n_ij_aggr)
    I_hat = 0.5 * (I_hat + I_hat.T)
    if ridge > 0:
        I_hat = I_hat + ridge * np.eye(I_hat.shape[0])


    Sigma_hat = np.linalg.inv(I_hat)
    return Sigma_hat, A_s


def Sigma_u_to_Sigma_s(
    Sigma_u_hat: np.ndarray,
    A_s: np.ndarray,
):

    Sigma_u_hat = np.asarray(Sigma_u_hat, dtype=float)
    A_s = np.asarray(A_s, dtype=float)

    N, Ns1 = A_s.shape
    d = Sigma_u_hat.shape[0]

    assert d == Ns1, "Σ̂_vartheta dimension must equal (N-1)"

    Sigma_s_hat = A_s @ Sigma_u_hat @ A_s.T

    return Sigma_s_hat


def ci_for_s(
    s_hat: np.ndarray,
    Sigma_s_hat: np.ndarray,
    T: float,
    alpha_level: float = 0.05,
    idx: int = 0,
):
    from scipy.stats import norm

    s_hat = np.asarray(s_hat, dtype=float)

    N = s_hat.shape[0]

    r = idx

    Sigma_rr = Sigma_s_hat[r, r]

    var_hat = Sigma_rr / T
    if var_hat < 0:
        if var_hat > -1e-10:
            var_hat = 0.0
        else:
            raise RuntimeError(f"Negative var_hat={var_hat} at r={r}")
    se_hat = np.sqrt(var_hat)

    z = norm.ppf(1 - alpha_level / 2)

    lower = s_hat[r] - z * se_hat
    upper = s_hat[r] + z * se_hat

    return s_hat[r], (lower, upper)


def main():
    np.random.seed(42)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    parent_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(parent_dir, "data")
    results_dir = os.path.join(parent_dir, "results")
    
    path = os.path.join(data_dir, "in_house_data.json")
    base_name = os.path.join(results_dir, "experiment1")
    os.makedirs(base_name, exist_ok=True)
    
    records, model2id, judge2id, qid2id, i, j, k, p, y = load_data(path)
    
    counts_ijk = defaultdict(int)
    sums_ijk = defaultdict(float)

    for ii, jj, kk, yy in zip(i, j, k, y):
        key = (ii, jj, kk)
        counts_ijk[key] += 1
        sums_ijk[key] += yy

    Omega = list(counts_ijk.keys())
    n_ijk = {key: counts_ijk[key] for key in Omega}
    ybar_ijk = {key: sums_ijk[key] / counts_ijk[key] for key in Omega}

    print("|Omega| =", len(Omega))

    tie_count = np.sum(y == 0.5)
    print(f"Tie proportion: {tie_count / len(y) * 100:.2f}%")

    model_counts = Counter()
    for ii, jj in zip(i, j):
        model_counts[ii] += 1
        model_counts[jj] += 1

    print("Model comparison count statistics:")
    print(f"  Min: {min(model_counts.values())}")
    print(f"  Max: {max(model_counts.values())}")
    print(f"  Average: {np.mean(list(model_counts.values())):.1f}")

    judge_counts = Counter(k)
    print("\nJudge comparison count statistics:")
    for jid, c in sorted(judge_counts.items(), key=lambda x: -x[1]):
        judge_name = [name for name, idx in judge2id.items() if idx == jid][0]
        print(f"  {judge_name}: {c}")

    N = len(model2id)
    K = len(judge2id)
    s_hat, gamma_hat, s_hat_unweighted, Sigma_s_alpha_hat, Sigma_s_gamma_hat, T, id2model, id2judge = fit_full_model(N, K, Omega, n_ijk, ybar_ijk, i, j, k, p, y, model2id, judge2id, base_name)

    full_s_weighted = s_hat.copy()
    full_s_unweighted = s_hat_unweighted.copy()
    full_rank_weighted = np.argsort(-full_s_weighted)
    full_rank_unweighted = np.argsort(-full_s_unweighted)

    # key is the number of judges(k)，value us the dict {'p_list': [p1, p2], 'num_combos': i} where p is the number of pair-wise comparisons
    experiment_dict = {
        1: {'p_list': [1000], 'num_combos': 2},
        8: {'p_list': [1000], 'num_combos': 5},
    }

    results = []
    excel_path = os.path.join(base_name, "detailed_results.xlsx")
    writer = pd.ExcelWriter(excel_path, engine='openpyxl')

    all_judges = list(judge2id.keys())

    for k, config in experiment_dict.items():
        p_list = config['p_list']
        num_combos = config['num_combos']
        for p in p_list:
            print(f"Starting run k={k}, p={p}")
            corrs_s_weighted = []
            corrs_rank_weighted = []
            corrs_s_unweighted = []
            corrs_rank_unweighted = []
            
            iter_j = 0
            combos = list(combinations(all_judges, k))
            selected_combos = random.sample(combos, min(num_combos, len(combos)))
            for judge_combo in selected_combos:
                iter_j += 1
                judge_set = set(judge_combo)
                
                filtered_records = [rec for rec in records if rec['judge_model'] in judge_set]
                if len(filtered_records) < p:
                    print(f"Warning: Combo {judge_combo} has only {len(filtered_records)} records, less than p={p}, using all available records")
                    actual_p = len(filtered_records)
                else:
                    actual_p = p

                if actual_p == len(filtered_records):
                    sampled_records = filtered_records
                else:
                    sampled_records = random.sample(filtered_records, actual_p)
                
                result = process_subdataset(sampled_records, id2model, id2judge, N, K, base_name, k, p, iter_j, writer)
                if result is None:
                    continue
                s_sub_weighted_aligned, s_sub_unweighted_aligned, gamma_hat_aligned, s_cis_aligned, unweighted_s_cis_aligned, gamma_cis_aligned = result
                
                corr_s_w = pearsonr(full_s_weighted, s_sub_weighted_aligned)[0]
                rank_sub_w = np.argsort(-s_sub_weighted_aligned)
                corr_rank_w = spearmanr(full_rank_weighted, rank_sub_w)[0]
                
                corr_s_uw = pearsonr(full_s_unweighted, s_sub_unweighted_aligned)[0]
                rank_sub_uw = np.argsort(-s_sub_unweighted_aligned)
                corr_rank_uw = spearmanr(full_rank_unweighted, rank_sub_uw)[0]
                
                corrs_s_weighted.append(corr_s_w)
                corrs_rank_weighted.append(corr_rank_w)
                corrs_s_unweighted.append(corr_s_uw)
                corrs_rank_unweighted.append(corr_rank_uw)
            
            avg_corr_s_w = np.mean(corrs_s_weighted) if corrs_s_weighted else np.nan
            avg_corr_rank_w = np.mean(corrs_rank_weighted) if corrs_rank_weighted else np.nan
            avg_corr_s_uw = np.mean(corrs_s_unweighted) if corrs_s_unweighted else np.nan
            avg_corr_rank_uw = np.mean(corrs_rank_unweighted) if corrs_rank_unweighted else np.nan
            sd_corr_s_w = np.std(corrs_s_weighted) if corrs_s_weighted else np.nan
            sd_corr_rank_w = np.std(corrs_rank_weighted) if corrs_rank_weighted else np.nan
            sd_corr_s_uw = np.std(corrs_s_unweighted) if corrs_s_unweighted else np.nan
            sd_corr_rank_uw = np.std(corrs_rank_unweighted) if corrs_rank_unweighted else np.nan
            
            results.append({
                'sample_size_p': p,
                'judges_k': k,
                'avg_corr_s_weighted': avg_corr_s_w,
                'avg_corr_rank_weighted': avg_corr_rank_w,
                'avg_corr_s_unweighted': avg_corr_s_uw,
                'avg_corr_rank_unweighted': avg_corr_rank_uw,
                'sd_corr_s_weighted': sd_corr_s_w,
                'sd_corr_rank_weighted': sd_corr_rank_w,
                'sd_corr_s_unweighted': sd_corr_s_uw,
                'sd_corr_rank_unweighted': sd_corr_rank_uw,
            })
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(f'{base_name}/average_correlations.csv', index=False)
    
    writer.close()
    
    print(f"Experiment completed. Average correlations saved to {os.path.abspath(f'{base_name}/average_correlations.csv')}, detailed results saved to {os.path.abspath(excel_path)}")


if __name__ == "__main__":
    main()