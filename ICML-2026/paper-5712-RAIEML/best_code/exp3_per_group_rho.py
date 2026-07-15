#!/usr/bin/env python3
"""Experiment 3: Per-group rho (IDEA-5712-03)"""
import numpy as np
import pandas as pd
import cvxpy as cp
from datasets import load_dataset
import time
import os
import json
import sys

os.environ.pop("HF_ENDPOINT", None)

MODELS = [
    "chatgpt-4o-latest-20250326", "claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022",
    "claude-3-7-sonnet-20250219", "claude-3-7-sonnet-20250219-thinking-32k", "claude-opus-4-20250514",
    "claude-opus-4-20250514-thinking-16k", "claude-sonnet-4-20250514", "claude-sonnet-4-20250514-thinking-32k",
    "deepseek-r1-0528", "deepseek-v3-0324", "gemini-2.0-flash-001", "gemini-2.5-flash", "gemini-2.5-pro",
    "gemma-3-27b-it", "gpt-4.1-mini-2025-04-14", "llama-4-maverick-03-26-experimental",
    "llama-4-maverick-17b-128e-instruct", "mistral-medium-2505", "o3-2025-04-16", "o4-mini-2025-04-16",
    "qwen3-235b-a22b-no-thinking", "qwen3-30b-a3b"
]

N_BOOT = 200
TRAIN_FRAC = 0.8
ALPHA = 1.0
SEED = 42

# IDEA-5712-03: Per-group rho disparity parameter (0 = disabled = scalar rho)
GAMMA_PARAM = 0.5

print(f"Models: {len(MODELS)}", flush=True)
print(f"n_bootstrap={N_BOOT}, train_frac={TRAIN_FRAC}, alpha={ALPHA}, seed={SEED}", flush=True)
print(f"gamma_param={GAMMA_PARAM} (per-group rho)", flush=True)


def load_df():
    ds = load_dataset("lmarena-ai/arena-human-preference-140k", split="train")
    df = ds.to_pandas()
    keep = ["model_a", "model_b", "winner", "language", "timestamp", "is_code", "category_tag"]
    return df[keep].copy()


def top_languages(df, k=4):
    vc = df["language"].value_counts(dropna=False)
    vc_filtered = vc[vc.index != "und"]
    langs = vc_filtered.head(k).index.tolist()
    return langs, vc


def build_margins(df, languages, models, alpha=1.0):
    m = len(models)
    K = len(languages)
    idx = {name: i for i, name in enumerate(models)}
    df = df[df["language"].isin(languages)].copy()
    df = df[df["model_a"].isin(models) & df["model_b"].isin(models)].copy()
    counts = df["language"].value_counts().reindex(languages).fillna(0).astype(int)
    tot = int(counts.sum())
    if tot == 0:
        raise ValueError("No rows after filtering; check languages/models.")
    w0 = (counts / tot).to_numpy(float)
    df = df.copy()
    df["idx_a"] = df["model_a"].map(idx)
    df["idx_b"] = df["model_b"].map(idx)
    df = df.dropna(subset=["idx_a", "idx_b"]).copy()
    M_list = []
    for lang in languages:
        rows = df[df["language"] == lang]
        if len(rows) == 0:
            M_list.append(np.zeros((m, m), float))
            continue
        win = np.zeros((m, m), float)
        idx_a = rows["idx_a"].values.astype(int)
        idx_b = rows["idx_b"].values.astype(int)
        winner = rows["winner"].values
        mask_a = (winner == "model_a")
        np.add.at(win, (idx_a[mask_a], idx_b[mask_a]), 1.0)
        mask_b = (winner == "model_b")
        np.add.at(win, (idx_b[mask_b], idx_a[mask_b]), 1.0)
        M = np.zeros((m, m), float)
        for i in range(m):
            for j in range(i + 1, m):
                tot_ij = win[i, j] + win[j, i]
                if tot_ij > 0:
                    mij = (win[i, j] - win[j, i]) / (tot_ij + 2.0 * alpha)
                else:
                    mij = 0.0
                M[i, j] = mij
                M[j, i] = -mij
        M_list.append(M)
    return M_list, w0, counts, df


def solve_drml_tv(M_list, w0, rho, solvers=("GUROBI", "MOSEK", "GLPK", "ECOS"), gamma_param=0.0):
    """
    IDEA-5712-03: Per-group rho when gamma_param > 0.
    rho_k = rho * (min(w0) / w0[k])^gamma_param
    """
    K = len(M_list)
    m = M_list[0].shape[0]

    # Per-group rho computation
    if gamma_param > 0 and K > 1:
        w0_arr = np.array(w0, dtype=float)
        w0_min = float(np.min(w0_arr[w0_arr > 0])) if np.any(w0_arr > 0) else 1.0 / K
        rho_k = np.array([rho * (w0_min / max(w0_arr[k], 1e-10)) ** gamma_param for k in range(K)], dtype=float)
        rho_k = np.clip(rho_k, 0.0, 2.0)
    else:
        rho_k = np.full(K, rho, dtype=float)

    p = cp.Variable(m, nonneg=True)
    t = cp.Variable()
    mu = cp.Variable(m)
    lam = cp.Variable(m, nonneg=True)
    gamma = cp.Variable((m, K))
    cons = [cp.sum(p) == 1]
    for a in range(m):
        cons += [t <= mu[a] - 2.0 * cp.sum(cp.multiply(rho_k, lam[a])) + w0 @ gamma[a, :]]
        for k in range(K):
            Mk = M_list[k]
            cons += [mu[a] + gamma[a, k] <= p @ Mk[:, a],
                     gamma[a, k] <= lam[a],
                     gamma[a, k] >= -lam[a]]
    prob = cp.Problem(cp.Maximize(t), cons)
    last = None
    for s in solvers:
        try:
            prob.solve(solver=getattr(cp, s), verbose=False)
            if prob.status in ("optimal", "optimal_inaccurate"):
                break
        except Exception as e:
            last = e
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError("LP not solved. status={}, last={}".format(prob.status, last))
    pval = np.array(p.value).reshape(-1)
    pval[pval < 0] = 0
    if pval.sum() > 0:
        pval /= pval.sum()
    return pval, float(t.value)


def per_group_winrate_vectorized(p, M_stack):
    pM = np.einsum("i,kij->kj", p, M_stack)
    margins = np.min(pM, axis=1)
    wr = 0.5 * (1.0 + margins)
    return wr


# Load data
print("\n=== Loading dataset ===", flush=True)
df = load_df()
print("Loaded {} rows".format(len(df)), flush=True)

languages, _ = top_languages(df, k=4)
print("Top languages: {}".format(languages), flush=True)
lang_counts = df["language"].value_counts().reindex(languages)
print("Language proportions: {}".format(
    {l: f"{lang_counts[l]/lang_counts.sum()*100:.1f}%" for l in languages}
), flush=True)

df_filtered = df[df["language"].isin(languages)].copy()
df_filtered = df_filtered[df_filtered["model_a"].isin(MODELS) & df_filtered["model_b"].isin(MODELS)].copy()
print("Filtered rows: {}".format(len(df_filtered)), flush=True)

print("\n=== Generating {} bootstrap train/test splits (80/20) ===".format(N_BOOT), flush=True)
t0 = time.time()
splits = []
for b in range(N_BOOT):
    rng = np.random.default_rng(SEED + 100000 * b)
    n_total = len(df_filtered)
    boot_idx = rng.integers(0, n_total, size=n_total)
    df_boot = df_filtered.iloc[boot_idx].reset_index(drop=True)
    df_train_list = []
    df_test_list = []
    for lang in languages:
        lang_df = df_boot[df_boot["language"] == lang]
        n = len(lang_df)
        if n == 0:
            continue
        n_train = int(TRAIN_FRAC * n)
        idx = np.arange(n)
        rng.shuffle(idx)
        train_idx = idx[:n_train]
        test_idx = idx[n_train:]
        df_train_list.append(lang_df.iloc[train_idx])
        df_test_list.append(lang_df.iloc[test_idx])
    if len(df_train_list) == 0:
        continue
    df_train = pd.concat(df_train_list, ignore_index=True)
    df_test = pd.concat(df_test_list, ignore_index=True)
    splits.append((df_train, df_test))
    if (b + 1) % max(1, N_BOOT // 10) == 0:
        print("  split {}/{}".format(b + 1, N_BOOT), flush=True)
print("Split generation took {:.1f}s".format(time.time() - t0), flush=True)

rhos = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

print("\n=== Computing generalization results ===", flush=True)
t0 = time.time()
boot_wr_test = {(rho, "overall"): [] for rho in rhos}
boot_wr_train = {(rho, "overall"): [] for rho in rhos}

for split_idx, (df_train, df_test) in enumerate(splits):
    M_train, w0_train, _, _ = build_margins(df_train, languages, MODELS, alpha=ALPHA)
    M_test, w0_test, _, _ = build_margins(df_test, languages, MODELS, alpha=ALPHA)
    M_test_stack = np.stack(M_test, axis=0)
    M_train_stack = np.stack(M_train, axis=0)
    w0_train_arr = np.array(w0_train, float)
    w0_test_arr = np.array(w0_test, float)
    for rho in rhos:
        p_rho, _ = solve_drml_tv(M_train, w0_train, rho, gamma_param=GAMMA_PARAM)
        M_pooled_test = np.einsum("k,kij->ij", w0_test_arr, M_test_stack)
        v_test_overall = float(np.min(p_rho @ M_pooled_test))
        wr_test_overall = 0.5 * (1 + v_test_overall)
        boot_wr_test[(rho, "overall")].append(float(wr_test_overall))
        M_pooled_train = np.einsum("k,kij->ij", w0_train_arr, M_train_stack)
        v_train_overall = float(np.min(p_rho @ M_pooled_train))
        wr_train_overall = 0.5 * (1 + v_train_overall)
        boot_wr_train[(rho, "overall")].append(float(wr_train_overall))
    if (split_idx + 1) % max(1, N_BOOT // 10) == 0:
        print("  compute {}/{}".format(split_idx + 1, N_BOOT), flush=True)
print("Computation took {:.1f}s".format(time.time() - t0), flush=True)

print("\n=== Win Rate Results (held-out, %) ===", flush=True)
print("{:>6}  {:>12}  {:>10}  {:>12}  {:>10}".format("rho", "test_mean%", "test_se%", "train_mean%", "train_se%"))
for rho in rhos:
    arr_test = np.array(boot_wr_test[(rho, "overall")], float)
    arr_train = np.array(boot_wr_train[(rho, "overall")], float)
    test_mean = float(arr_test.mean()) * 100.0
    test_se = float(arr_test.std() / np.sqrt(len(arr_test))) * 100.0
    train_mean = float(arr_train.mean()) * 100.0
    train_se = float(arr_train.std() / np.sqrt(len(arr_train))) * 100.0
    print("{:6.1f}  {:12.4f}  {:10.4f}  {:12.4f}  {:10.4f}".format(rho, test_mean, test_se, train_mean, train_se))

rho_04_test = np.array(boot_wr_test[(0.4, "overall")], float)
best_rho = max(rhos, key=lambda r: float(np.mean(boot_wr_test[(r, "overall")])))
best_test = np.array(boot_wr_test[(best_rho, "overall")], float)

results = {
    "best_rho": float(best_rho),
    "best_rho_win_rate_mean_pct": float(best_test.mean()) * 100,
    "best_rho_win_rate_se_pct": float(best_test.std() / np.sqrt(len(best_test))) * 100,
    "robust_lottery_rho_0.4": {
        "win_rate_mean_pct": float(rho_04_test.mean()) * 100,
        "win_rate_se_pct": float(rho_04_test.std() / np.sqrt(len(rho_04_test))) * 100
    }
}

print("\n=== EXPERIMENT METRICS ===", flush=True)
print(json.dumps(results, indent=2), flush=True)
print("\nBest rho={:.2f}: {:.2f} +/- {:.2f}".format(
    best_rho,
    results["best_rho_win_rate_mean_pct"],
    results["best_rho_win_rate_se_pct"]))
print("DONE.")
