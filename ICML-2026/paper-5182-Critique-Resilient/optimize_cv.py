#!/usr/bin/env python3
"""
Optimization wrapper for CRB predictive validity CV (Paper 5182).
Adds sigma tuning, Adam optimizer, L2 regularization, confidence weighting, Platt scaling.
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse game collection from the original CV script
from cv_predictive_validity import (
    build_itemized_dataset,
    collect_games,
)


def fit_itemized_bt_adam(
    dataset, sigmas, max_iter=3000, lr=0.01, tol=1e-6,
    l2_lambda=0.0, warmup_steps=100, beta1=0.9, beta2=0.999, eps=1e-8,
):
    """Fit itemized BT with Adam optimizer and optional L2 regularization."""
    num_answerers = len(dataset["answerers"])
    num_questioners = len(dataset["questioners"])
    num_questions = len(dataset["question_keys"])

    beta_free = np.zeros(max(0, num_answerers - 1), dtype=float)
    alpha = np.zeros(num_questioners, dtype=float)
    delta = np.zeros(num_questions, dtype=float)

    # Adam state
    m_beta = np.zeros_like(beta_free)
    v_beta = np.zeros_like(beta_free)
    m_alpha = np.zeros_like(alpha)
    v_alpha = np.zeros_like(alpha)
    m_delta = np.zeros_like(delta)
    v_delta = np.zeros_like(delta)

    inv_beta_var = 1.0 / (sigmas["sigma_beta"] ** 2)
    inv_alpha_var = 1.0 / (sigmas["sigma_alpha"] ** 2)
    inv_delta_var = 1.0 / (sigmas["sigma_delta"] ** 2)

    edges_arr = np.asarray(dataset["edges"], dtype=np.int64)
    if edges_arr.size == 0:
        return {"beta": [0.0] if num_answerers == 0 else [0.0] * num_answerers,
                "alpha": alpha.tolist(), "delta": delta.tolist()}

    q_idx = edges_arr[:, 0]
    b_idx = edges_arr[:, 1]
    outcome = edges_arr[:, 2].astype(float)
    questioner_by_question = np.asarray(dataset["questioner_by_question"], dtype=np.int64)
    a_idx = questioner_by_question[q_idx]

    last_obj = None
    for step_idx in range(max_iter):
        if beta_free.size == 0:
            beta_full = np.array([0.0])
        else:
            beta_full = np.concatenate([beta_free, [-beta_free.sum()]])
        beta_last = beta_full[-1]

        logit = beta_full[b_idx] - alpha[a_idx] - delta[q_idx]
        x_arr = np.asarray(logit, dtype=float)
        p = np.empty_like(x_arr)
        pos_mask = x_arr >= 0
        p[pos_mask] = 1.0 / (1.0 + np.exp(-x_arr[pos_mask]))
        exp_x = np.exp(x_arr[~pos_mask])
        p[~pos_mask] = exp_x / (1.0 + exp_x)

        ll = float(np.dot(outcome, logit) - np.sum(np.logaddexp(0.0, logit)))
        err = outcome - p

        grad_beta = np.zeros_like(beta_free)
        grad_alpha = np.zeros_like(alpha)
        grad_delta = np.zeros_like(delta)

        if grad_beta.size:
            mask = b_idx < grad_beta.size
            np.add.at(grad_beta, b_idx[mask], err[mask])
            if np.any(~mask):
                grad_beta -= err[~mask].sum()
        np.add.at(grad_alpha, a_idx, -err)
        np.add.at(grad_delta, q_idx, -err)

        # L2 prior
        if grad_beta.size:
            grad_beta -= inv_beta_var * (beta_free - beta_last)
        grad_alpha -= inv_alpha_var * alpha
        grad_delta -= inv_delta_var * delta

        # Explicit L2 regularization
        if l2_lambda > 0:
            grad_beta -= 2.0 * l2_lambda * beta_free
            grad_alpha -= 2.0 * l2_lambda * alpha
            grad_delta -= 2.0 * l2_lambda * delta

        prior = -0.5 * inv_beta_var * float(np.sum(beta_full * beta_full))
        prior -= 0.5 * inv_alpha_var * float(np.sum(alpha * alpha))
        prior -= 0.5 * inv_delta_var * float(np.sum(delta * delta))
        obj = ll + prior

        # Adam update with warmup
        t = step_idx + 1
        if t <= warmup_steps:
            effective_lr = lr * t / warmup_steps
        else:
            effective_lr = lr

        max_step = 0.0
        for param, grad, m, v in [
            (beta_free, grad_beta, m_beta, v_beta),
            (alpha, grad_alpha, m_alpha, v_alpha),
            (delta, grad_delta, m_delta, v_delta),
        ]:
            if param.size == 0:
                continue
            m[:] = beta1 * m + (1 - beta1) * grad
            v[:] = beta2 * v + (1 - beta2) * (grad * grad)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            step = effective_lr * m_hat / (np.sqrt(v_hat) + eps)
            param[:] += step
            max_step = max(max_step, float(np.max(np.abs(step))))

        if last_obj is not None and abs(obj - last_obj) < tol and max_step < tol:
            break
        last_obj = obj

    beta_full = (beta_free.tolist() + [-float(beta_free.sum())]) if beta_free.size else [0.0]
    return {"beta": beta_full, "alpha": alpha.tolist(), "delta": delta.tolist()}


def tune_sigmas(dataset, sigma_grid, max_iter=3000, lr=0.05):
    """Grid search over sigma values using Laplace-approximated marginal likelihood."""
    from analysis.bt_utils import (
        evaluate_sigmas,
        Sigmas,
        ItemizedDataset,
    )

    # Convert dict dataset to ItemizedDataset
    itemized = ItemizedDataset(
        answerers=dataset["answerers"],
        questioners=dataset["questioners"],
        question_keys=dataset["question_keys"],
        questioner_by_question=dataset["questioner_by_question"],
        edges=dataset["edges"],
    )

    best_score = -float("inf")
    best_sigmas = None

    for sb in sigma_grid.get("sigma_beta", [5.0]):
        for sa in sigma_grid.get("sigma_alpha", [5.0]):
            for sd in sigma_grid.get("sigma_delta", [1.0]):
                sigmas = Sigmas(sigma_beta=sb, sigma_alpha=sa, sigma_delta=sd)
                try:
                    score, params = evaluate_sigmas(
                        itemized, sigmas, max_iter=max_iter, lr=lr, tol=1e-6
                    )
                    if score > best_score:
                        best_score = score
                        best_sigmas = (sb, sa, sd)
                    print(f"  sigma=({sb},{sa},{sd}) -> log_marginal={score:.2f}")
                except Exception as e:
                    print(f"  sigma=({sb},{sa},{sd}) -> ERROR: {e}")
                    continue

    if best_sigmas is not None:
        print(f"  Best: sigma={best_sigmas} log_marginal={best_score:.2f}")
    return best_sigmas, best_score


def apply_platt_scaling(train_logits, train_outcomes, test_logits):
    """Fit Platt scaler on training data and calibrate test predictions."""
    from sklearn.linear_model import LogisticRegression

    if len(np.unique(train_outcomes)) < 2:
        return 1.0 / (1.0 + np.exp(-test_logits))

    X = train_logits.reshape(-1, 1)
    y = train_outcomes.astype(int)
    calib = LogisticRegression(penalty=None, solver="lbfgs")
    calib.fit(X, y)

    calibrated_logits = calib.decision_function(test_logits.reshape(-1, 1))
    return 1.0 / (1.0 + np.exp(-calibrated_logits))


def run_optimized_cv(
    games, n_folds=5, seed=42,
    sigmas=None, use_adam=False, adam_lr=0.01,
    l2_lambda=0.0, max_iter=3000, tol=1e-6,
    use_platt=False,
):
    """Run optimized predictive validity CV."""
    dataset = build_itemized_dataset(games)
    question_keys = dataset["question_keys"]
    edges = np.asarray(dataset["edges"], dtype=np.int64)
    questioner_by_question = np.asarray(dataset["questioner_by_question"], dtype=np.int64)
    answerers = dataset["answerers"]
    questioners = dataset["questioners"]

    if edges.size == 0:
        print("No edges to evaluate.")
        return None

    q_idx = edges[:, 0]
    b_idx = edges[:, 1]
    outcomes = edges[:, 2].astype(float)
    a_idx = questioner_by_question[q_idx]

    baserate_prob = float(np.mean(outcomes))
    baserate_accuracy = max(baserate_prob, 1 - baserate_prob)
    baserate_logloss = -float(np.mean(
        outcomes * math.log(max(baserate_prob, 1e-15)) +
        (1 - outcomes) * math.log(max(1 - baserate_prob, 1e-15))
    ))
    baserate_brier = float(np.mean((outcomes - baserate_prob) ** 2))

    if sigmas is None:
        sigmas = {"sigma_beta": 5.0, "sigma_alpha": 5.0, "sigma_delta": 1.0}

    np.random.seed(seed)
    unique_questions = np.unique(q_idx)
    np.random.shuffle(unique_questions)
    fold_size = len(unique_questions) // n_folds

    all_probs = np.zeros(len(outcomes))

    print(f"\n{n_folds}-fold CV, sigmas={sigmas}, adam={use_adam}, l2={l2_lambda}, platt={use_platt}")

    for fold in range(n_folds):
        start = fold * fold_size
        end = start + fold_size if fold < n_folds - 1 else len(unique_questions)
        test_q = set(unique_questions[start:end])
        train_q = set(unique_questions) - test_q

        train_mask = np.isin(q_idx, list(train_q))
        test_mask = np.isin(q_idx, list(test_q))

        train_edges = edges[train_mask].tolist()
        train_dataset = {
            "answerers": answerers,
            "questioners": questioners,
            "question_keys": question_keys,
            "questioner_by_question": questioner_by_question.tolist(),
            "edges": train_edges,
        }

        if use_adam:
            params = fit_itemized_bt_adam(
                train_dataset, sigmas, max_iter=max_iter, lr=adam_lr,
                tol=tol, l2_lambda=l2_lambda,
            )
        else:
            from cv_predictive_validity import fit_itemized_bt
            params = fit_itemized_bt(train_dataset, sigmas, max_iter=max_iter, lr=0.05, tol=tol)

        beta = np.asarray(params["beta"], dtype=float)
        alpha = np.asarray(params["alpha"], dtype=float)
        delta = np.asarray(params["delta"], dtype=float)

        test_q_idx = q_idx[test_mask]
        test_b_idx = b_idx[test_mask]
        test_a_idx = a_idx[test_mask]
        test_outcomes = outcomes[test_mask]

        logit = beta[test_b_idx] - alpha[test_a_idx] - delta[test_q_idx]

        if use_platt:
            train_q_idx = q_idx[train_mask]
            train_b_idx = b_idx[train_mask]
            train_a_idx = a_idx[train_mask]
            train_outcomes = outcomes[train_mask]
            train_logit = beta[train_b_idx] - alpha[train_a_idx] - delta[train_q_idx]
            probs = apply_platt_scaling(train_logit, train_outcomes, logit)
        else:
            probs = 1.0 / (1.0 + np.exp(-logit))

        all_probs[test_mask] = probs

        fold_acc = np.mean((probs >= 0.5) == test_outcomes)
        fold_n = len(test_outcomes)
        print(f"  Fold {fold+1}: {fold_n} test interactions, accuracy={fold_acc:.4f}")

    pred_outcomes = (all_probs >= 0.5).astype(float)
    accuracy = float(np.mean(pred_outcomes == outcomes))
    eps = 1e-15
    logloss = -float(np.mean(
        outcomes * np.log(np.clip(all_probs, eps, 1 - eps)) +
        (1 - outcomes) * np.log(np.clip(1 - all_probs, eps, 1 - eps))
    ))
    brier = float(np.mean((all_probs - outcomes) ** 2))

    print(f"\n{'='*60}")
    print(f"Optimized Results (Itemized BT, {n_folds}-fold CV)")
    print(f"{'='*60}")
    print(f"{'Metric':<15} {'Baserate':>10} {'Model':>10} {'Delta':>10}")
    print(f"{'-'*45}")
    print(f"{'Accuracy':<15} {baserate_accuracy:>10.4f} {accuracy:>10.4f} {accuracy - baserate_accuracy:>+10.4f}")
    print(f"{'Log-loss':<15} {baserate_logloss:>10.4f} {logloss:>10.4f} {logloss - baserate_logloss:>+10.4f}")
    print(f"{'Brier':<15} {baserate_brier:>10.4f} {brier:>10.4f} {brier - baserate_brier:>+10.4f}")
    print(f"{'='*60}")

    return {
        "accuracy": accuracy,
        "logloss": logloss,
        "brier": brier,
        "baserate_accuracy": baserate_accuracy,
        "baserate_logloss": baserate_logloss,
        "baserate_brier": baserate_brier,
    }


def main():
    parser = argparse.ArgumentParser(description="Optimized CV for CRB Paper 5182")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma-beta", type=float, default=5.0)
    parser.add_argument("--sigma-alpha", type=float, default=5.0)
    parser.add_argument("--sigma-delta", type=float, default=1.0)
    parser.add_argument("--tune-sigmas", action="store_true", help="Grid search sigmas before CV")
    parser.add_argument("--use-adam", action="store_true")
    parser.add_argument("--adam-lr", type=float, default=0.01)
    parser.add_argument("--l2-lambda", type=float, default=0.0)
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--use-platt", action="store_true")
    args = parser.parse_args()

    print("Collecting games...")
    games, skips = collect_games(
        Path("benchmarks"), Path("answers"), Path("critiques"),
        Path("automated_evaluations"), Path("configs/models.json"),
        "evaluator", "contradictor", False, Path("evaluations"),
        log_automated_disagreements=False,
    )

    total_games = len(games)
    wins = sum(1 for g in games if g[3] == 1)
    print(f"Games: {total_games} (wins={wins}, losses={total_games - wins})")

    if not games:
        print("No valid games found.")
        return 1

    sigmas = {"sigma_beta": args.sigma_beta, "sigma_alpha": args.sigma_alpha, "sigma_delta": args.sigma_delta}

    if args.tune_sigmas:
        full_dataset = build_itemized_dataset(games)
        sigma_grid = {
            "sigma_beta": [1.0, 2.0, 5.0, 10.0, 20.0],
            "sigma_alpha": [1.0, 2.0, 5.0, 10.0, 20.0],
            "sigma_delta": [0.5, 1.0, 2.0, 5.0, 10.0],
        }
        print("\nTuning sigmas via marginal likelihood...")
        best, score = tune_sigmas(full_dataset, sigma_grid, max_iter=args.max_iter, lr=0.05)
        if best is not None:
            sigmas = {"sigma_beta": best[0], "sigma_alpha": best[1], "sigma_delta": best[2]}
            print(f"Using tuned sigmas: {sigmas}")

    result = run_optimized_cv(
        games, n_folds=args.n_folds, seed=args.seed,
        sigmas=sigmas, use_adam=args.use_adam, adam_lr=args.adam_lr,
        l2_lambda=args.l2_lambda, max_iter=args.max_iter, tol=args.tol,
        use_platt=args.use_platt,
    )

    if result:
        print(f"\nFINAL: accuracy={result['accuracy']:.4f} logloss={result['logloss']:.4f} brier={result['brier']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
