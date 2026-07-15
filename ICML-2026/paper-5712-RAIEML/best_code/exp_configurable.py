#!/usr/bin/env python3
"""Configurable experiment runner for parameter sweeps."""
import numpy as np
import pandas as pd
import cvxpy as cp
from datasets import load_dataset
from scipy.stats import binomtest
import time
import os
import json
import sys
import argparse

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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sig-level", type=float, default=0.0, help="0=disabled")
    p.add_argument("--adaptive-alpha", type=int, default=0, help="1=enabled")
    p.add_argument("--base-alpha", type=float, default=1.0)
    p.add_argument("--fine-rho", type=int, default=0, help="1=use fine rho sweep")
    p.add_argument("--output", type=str, default="/tmp/exp_result.json")
    p.add_argument("--label", type=str, default="experiment")
    return p.parse_args()


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


def build_margins(df, languages, models, alpha=1.0, sig_level=0.0, adaptive_alpha=False, base_alpha=1.0):
    m = len(models)
    K = len(languages)
    idx = {name: i for i, name in enumerate(models)}
    df = df[df["language"].isin(languages)].copy()
    df = df[df["model_a"].isin(models) & df["model_b"].isin(models)].copy()
    counts = df["language"].value_counts().reindex(languages).fillna(0).astype(int)
    tot = int(counts.sum())
    if tot == 0:
        raise ValueError("No rows after filtering")
    w0 = (counts / tot).to_numpy(float)
    df = df.copy()
    df["idx_a"] = df["model_a"].map(idx)
    df["idx_b"] = df["model_b"].map(idx)
    df = df.dropna(subset=["idx_a", "idx_b"]).copy()

    if adaptive_alpha:
        all_totals = []
        for lang in languages:
            rows = df[df["language"] == lang]
            if len(rows) == 0:
                continue
            win_tmp = np.zeros((m, m), float)
            idx_a_tmp = rows["idx_a"].values.astype(int)
            idx_b_tmp = rows["idx_b"].values.astype(int)
            winner_tmp = rows["winner"].values
            mask_a_tmp = (winner_tmp == "model_a")
            np.add.at(win_tmp, (idx_a_tmp[mask_a_tmp], idx_b_tmp[mask_a_tmp]), 1.0)
            mask_b_tmp = (winner_tmp == "model_b")
            np.add.at(win_tmp, (idx_b_tmp[mask_b_tmp], idx_a_tmp[mask_b_tmp]), 1.0)
            for i in range(m):
                for j in range(i + 1, m):
                    t = win_tmp[i, j] + win_tmp[j, i]
                    if t > 0:
                        all_totals.append(t)
        c_ref = float(np.median(all_totals)) if all_totals else 1.0
    else:
        c_ref = 1.0

    M_list = []
    n_filtered = 0
    n_total_pairs = 0
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
                n_total_pairs += 1
                if tot_ij > 0:
                    if sig_level > 0:
                        n_wins_i = int(win[i, j])
                        result = binomtest(n_wins_i, n=int(tot_ij), p=0.5, alternative='two-sided')
                        if result.pvalue > sig_level:
                            M[i, j] = 0.0
                            M[j, i] = 0.0
                            n_filtered += 1
                            continue
                    if adaptive_alpha:
                        alpha_ij = base_alpha * np.sqrt(c_ref / (c_ref + tot_ij))
                    else:
                        alpha_ij = alpha
                    mij = (win[i, j] - win[j, i]) / (tot_ij + 2.0 * alpha_ij)
                else:
                    mij = 0.0
                M[i, j] = mij
                M[j, i] = -mij
        M_list.append(M)
    return M_list, w0, counts, df, n_filtered, n_total_pairs


def solve_drml_tv(M_list, w0, rho, solvers=("GUROBI", "MOSEK", "GLPK", "ECOS")):
    K = len(M_list)
    m = M_list[0].shape[0]
    p = cp.Variable(m, nonneg=True)
    t = cp.Variable()
    mu = cp.Variable(m)
    lam = cp.Variable(m, nonneg=True)
    gamma = cp.Variable((m, K))
    cons = [cp.sum(p) == 1]
    for a in range(m):
        cons += [t <= mu[a] - 2.0 * rho * lam[a] + w0 @ gamma[a, :]]
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


def main():
    args = parse_args()
    label = args.label

    print(f"=== {label} ===", flush=True)
    print(f"sig_level={args.sig_level}, adaptive_alpha={bool(args.adaptive_alpha)}, base_alpha={args.base_alpha}", flush=True)
    print(f"n_boot={args.n_boot}, fine_rho={bool(args.fine_rho)}", flush=True)

    print("\n=== Loading dataset ===", flush=True)
    df = load_df()
    print(f"Loaded {len(df)} rows", flush=True)

    languages, _ = top_languages(df, k=4)
    print(f"Languages: {languages}", flush=True)

    df_filtered = df[df["language"].isin(languages)].copy()
    df_filtered = df_filtered[df_filtered["model_a"].isin(MODELS) & df_filtered["model_b"].isin(MODELS)].copy()
    print(f"Filtered rows: {len(df_filtered)}", flush=True)

    print(f"\n=== Generating {args.n_boot} bootstrap splits ===", flush=True)
    t0 = time.time()
    splits = []
    for b in range(args.n_boot):
        rng = np.random.default_rng(args.seed + 100000 * b)
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
            n_train = int(args.train_frac * n)
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
        if (b + 1) % max(1, args.n_boot // 10) == 0:
            print(f"  split {b+1}/{args.n_boot}", flush=True)
    print(f"Split generation took {time.time()-t0:.1f}s", flush=True)

    if args.fine_rho:
        rhos = sorted(set(
            list(np.linspace(0.0, 1.0, 11))
            + list(np.arange(0.20, 0.55, 0.02))
        ))
    else:
        rhos = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    prev_filtered = None
    print(f"\n=== Computing results ({len(rhos)} rho values) ===", flush=True)
    t0 = time.time()
    boot_wr_test = {(rho, "overall"): [] for rho in rhos}
    boot_wr_train = {(rho, "overall"): [] for rho in rhos}

    for split_idx, (df_train, df_test) in enumerate(splits):
        M_train, w0_train, _, _, nf_train, nt_train = build_margins(
            df_train, languages, MODELS, alpha=args.alpha,
            sig_level=args.sig_level, adaptive_alpha=bool(args.adaptive_alpha),
            base_alpha=args.base_alpha
        )
        M_test, w0_test, _, _, nf_test, nt_test = build_margins(
            df_test, languages, MODELS, alpha=args.alpha,
            sig_level=args.sig_level, adaptive_alpha=bool(args.adaptive_alpha),
            base_alpha=args.base_alpha
        )
        curr_filtered = (nf_train, nt_train)
        if curr_filtered != prev_filtered and split_idx < 3:
            print(f"  split {split_idx}: filtered {nf_train}/{nt_train} train pairs", flush=True)
            prev_filtered = curr_filtered

        M_test_stack = np.stack(M_test, axis=0)
        w0_test_arr = np.array(w0_test, float)
        w0_train_arr = np.array(w0_train, float)
        for rho in rhos:
            p_rho, _ = solve_drml_tv(M_train, w0_train, rho)
            M_pooled_test = np.einsum("k,kij->ij", w0_test_arr, M_test_stack)
            v_test_overall = float(np.min(p_rho @ M_pooled_test))
            wr_test_overall = 0.5 * (1 + v_test_overall)
            boot_wr_test[(rho, "overall")].append(float(wr_test_overall))
            M_pooled_train = np.einsum("k,kij->ij", w0_train_arr, np.stack(M_train, axis=0))
            v_train_overall = float(np.min(p_rho @ M_pooled_train))
            wr_train_overall = 0.5 * (1 + v_train_overall)
            boot_wr_train[(rho, "overall")].append(float(wr_train_overall))
        if (split_idx + 1) % max(1, args.n_boot // 10) == 0:
            print(f"  compute {split_idx+1}/{args.n_boot}", flush=True)
    elapsed = time.time() - t0
    print(f"Computation took {elapsed:.1f}s", flush=True)

    # Find best rho
    best_rho = max(rhos, key=lambda r: float(np.mean(boot_wr_test[(r, "overall")])))
    best_test = np.array(boot_wr_test[(best_rho, "overall")], float)
    rho_04_test = np.array(boot_wr_test[(0.4, "overall")], float)

    result = {
        "label": label,
        "sig_level": args.sig_level,
        "adaptive_alpha": bool(args.adaptive_alpha),
        "base_alpha": args.base_alpha,
        "fine_rho": bool(args.fine_rho),
        "best_rho": float(best_rho),
        "best_win_rate_pct": float(best_test.mean()) * 100,
        "best_win_rate_se_pct": float(best_test.std() / np.sqrt(len(best_test))) * 100,
        "rho_0.4_win_rate_pct": float(rho_04_test.mean()) * 100,
        "rho_0.4_win_rate_se_pct": float(rho_04_test.std() / np.sqrt(len(rho_04_test))) * 100,
        "runtime_seconds": elapsed,
    }

    print(f"\n=== RESULTS: {label} ===", flush=True)
    print(json.dumps(result, indent=2), flush=True)

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {args.output}", flush=True)
    print("DONE.")


if __name__ == "__main__":
    main()
