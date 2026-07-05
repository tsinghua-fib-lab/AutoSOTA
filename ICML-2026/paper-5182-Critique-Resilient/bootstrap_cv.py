#!/usr/bin/env python3
"""Bootstrap ensemble BT for CRB Paper 5182 (Idea 6)."""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cv_predictive_validity import (
    build_itemized_dataset,
    collect_games,
    fit_itemized_bt,
)


def bootstrap_cv(games, n_folds=5, seed=42, n_bootstrap=25):
    """Run bootstrap-ensemble CV."""
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

    sigmas = {"sigma_beta": 5.0, "sigma_alpha": 5.0, "sigma_delta": 1.0}

    np.random.seed(seed)
    unique_questions = np.unique(q_idx)
    np.random.shuffle(unique_questions)
    fold_size = len(unique_questions) // n_folds

    all_probs = np.zeros(len(outcomes))

    print(f"\n{n_folds}-fold Bootstrap CV (B={n_bootstrap}), seed={seed}")

    for fold in range(n_folds):
        start = fold * fold_size
        end = start + fold_size if fold < n_folds - 1 else len(unique_questions)
        test_q = set(unique_questions[start:end])
        train_q = set(unique_questions) - test_q

        train_mask = np.isin(q_idx, list(train_q))
        test_mask = np.isin(q_idx, list(test_q))

        train_edges = edges[train_mask]
        n_train = len(train_edges)

        # Bootstrap ensemble over training edges
        ensemble_logits = np.zeros(len(outcomes[test_mask]))

        for b in range(n_bootstrap):
            # Sample with replacement
            bootstrap_indices = np.random.choice(n_train, size=n_train, replace=True)
            bt_edges = train_edges[bootstrap_indices].tolist()

            bt_dataset = {
                "answerers": answerers,
                "questioners": questioners,
                "question_keys": question_keys,
                "questioner_by_question": questioner_by_question.tolist(),
                "edges": bt_edges,
            }

            params = fit_itemized_bt(bt_dataset, sigmas, max_iter=3000, lr=0.05, tol=1e-6)
            beta = np.asarray(params["beta"], dtype=float)
            alpha = np.asarray(params["alpha"], dtype=float)
            delta = np.asarray(params["delta"], dtype=float)

            test_q_idx = q_idx[test_mask]
            test_b_idx = b_idx[test_mask]
            test_a_idx = a_idx[test_mask]

            logit = beta[test_b_idx] - alpha[test_a_idx] - delta[test_q_idx]
            ensemble_logits += logit

        ensemble_logits /= n_bootstrap
        probs = 1.0 / (1.0 + np.exp(-ensemble_logits))
        all_probs[test_mask] = probs

        test_outcomes = outcomes[test_mask]
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
    print(f"Bootstrap Ensemble Results (B={n_bootstrap}, {n_folds}-fold CV)")
    print(f"{'='*60}")
    print(f"{'Metric':<15} {'Baserate':>10} {'Model':>10} {'Delta':>10}")
    print(f"{'-'*45}")
    print(f"{'Accuracy':<15} {baserate_accuracy:>10.4f} {accuracy:>10.4f} {accuracy - baserate_accuracy:>+10.4f}")
    print(f"{'Log-loss':<15} {baserate_logloss:>10.4f} {logloss:>10.4f} {logloss - baserate_logloss:>+10.4f}")
    print(f"{'Brier':<15} {baserate_brier:>10.4f} {brier:>10.4f} {brier - baserate_brier:>+10.4f}")
    print(f"{'='*60}")

    return {"accuracy": accuracy, "logloss": logloss, "brier": brier}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstrap", type=int, default=25)
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

    result = bootstrap_cv(games, n_folds=args.n_folds, seed=args.seed,
                          n_bootstrap=args.n_bootstrap)

    if result:
        print(f"\nFINAL: accuracy={result['accuracy']:.4f} logloss={result['logloss']:.4f} brier={result['brier']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
