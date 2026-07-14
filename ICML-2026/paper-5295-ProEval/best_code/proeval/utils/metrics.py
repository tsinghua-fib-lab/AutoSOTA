# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Diversity and performance metrics for active evaluation.

Provides:

- :func:`topic_entropy` — Shannon entropy of topic distribution
- :func:`embedding_coverage` — DPP-based log-determinant diversity
- :func:`overall_diversity` — Weighted combination
- :func:`failure_rate` — Percentage of failures
- :func:`compute_all_metrics` — Compute all metrics for a list of records
- :func:`compute_samples_to_threshold` — Samples needed to reach error thresholds
- :func:`print_results_table` — Print comprehensive BQ experiment results
"""

import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests


OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
EMBEDDING_MODEL = "openai/text-embedding-3-small"


def get_question_embeddings(questions: List[str], batch_size: int = 50) -> np.ndarray:
    """Compute embeddings via the OpenRouter API.

    Returns array of shape ``(n_questions, embedding_dim)``.
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("Set OPENROUTER_API_KEY to compute embeddings.")

    all_embeddings: List[np.ndarray] = []
    for i in range(0, len(questions), batch_size):
        batch = questions[i : i + batch_size]
        resp = requests.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"model": EMBEDDING_MODEL, "input": batch},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        for item in sorted(data["data"], key=lambda x: x["index"]):
            all_embeddings.append(np.array(item["embedding"]))
    return np.array(all_embeddings)


def topic_entropy(topics: List[str], normalize: bool = True) -> float:
    """Shannon entropy of the topic distribution.

    Higher entropy → more diverse topic coverage.
    Returns 0–100 % if *normalize* is True, else raw entropy.
    """
    if not topics:
        return 0.0
    counts = Counter(topics)
    n = len(topics)
    probs = np.array([c / n for c in counts.values()])
    
    # The 1e-12 smoothing term avoids log2(0) but makes a single-class
    # distribution (true entropy == 0) compute as a tiny negative number
    # (e.g. -1.44e-12). Clamp to 0 so entropy is never negative.
    entropy = max(0.0, float(-np.sum(probs * np.log2(probs + 1e-12))))
    
    if normalize:
        max_ent = np.log2(len(counts)) if len(counts) > 1 else 1.0
        return (entropy / max_ent) * 100.0 if max_ent > 0 else 0.0
    return entropy


def embedding_coverage(
    embeddings: np.ndarray,
    regularization: float = 1e-6,
    normalize_to_01: bool = True,
    fixed_n: int = 100,
) -> float:
    """Log-determinant diversity measure (DPP-inspired).

    Higher → more diverse coverage in embedding space.
    """
    if embeddings.ndim != 2 or len(embeddings) < 2:
        return 0.0
    n = min(len(embeddings), fixed_n)
    if len(embeddings) > n:
        idx = np.random.choice(len(embeddings), n, replace=False)
        emb = embeddings[idx]
    else:
        emb = embeddings

    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normed = emb / norms
    K = normed @ normed.T + regularization * np.eye(len(normed))
    sign, logdet = np.linalg.slogdet(K)
    logdet_val = float(logdet) if sign > 0 else 0.0

    if normalize_to_01:
        max_logdet = n * np.log(1 + regularization)
        return logdet_val / max_logdet if max_logdet > 0 else 0.0
    return logdet_val


def overall_diversity(
    topic_entropy_score: float,
    embedding_coverage_score: float,
    weight_entropy: float = 0.5,
    weight_coverage: float = 0.5,
) -> float:
    """Weighted overall diversity score (0–100)."""
    return (
        weight_entropy * topic_entropy_score
        + weight_coverage * (embedding_coverage_score * 100.0)
    )


def failure_rate(scores: List[float]) -> float:
    """Percentage of failures (score == 1.0) in a list of error scores."""
    if not scores:
        return 0.0
    return (sum(1 for s in scores if s == 1.0) / len(scores)) * 100.0


def compute_all_metrics(
    records: List[Dict],
    topics: Optional[List[str]] = None,
    pool_embeddings: Optional[np.ndarray] = None,
    use_generated_embeddings: bool = True,
) -> Dict[str, float]:
    """Compute all metrics for a list of experiment records.

    Each record should have keys like ``"topic"``, ``"question"``, ``"score"``
    (or ``"seed_idx"`` for pool-based methods).

    Returns dict with ``topic_entropy``, ``embedding_coverage``,
    ``overall_diversity``, ``failure_rate``.
    """
    result: Dict[str, float] = {}

    # Topic entropy
    rec_topics = topics or [r.get("topic", "unknown") for r in records]
    result["topic_entropy"] = topic_entropy(rec_topics)

    # Embedding coverage
    if pool_embeddings is not None:
        idx = [r.get("seed_idx") for r in records if r.get("seed_idx") is not None]
        if idx:
            emb = pool_embeddings[idx]
            result["embedding_coverage"] = embedding_coverage(emb)
        else:
            result["embedding_coverage"] = 0.0
    elif use_generated_embeddings:
        questions = [r.get("question", "") for r in records if r.get("question")]
        if questions:
            try:
                emb = get_question_embeddings(questions)
                result["embedding_coverage"] = embedding_coverage(emb)
            except Exception:
                result["embedding_coverage"] = 0.0
        else:
            result["embedding_coverage"] = 0.0
    else:
        result["embedding_coverage"] = 0.0

    result["overall_diversity"] = overall_diversity(
        result["topic_entropy"], result["embedding_coverage"]
    )

    scores = [r.get("score", r.get("correct", 1.0)) for r in records]
    result["failure_rate"] = failure_rate(scores)

    return result


# BQ Experiment Metrics


def compute_samples_to_threshold(
    results: Dict[str, Dict],
    true_mean: float,
    thresholds: List[float] = [0.05, 0.02, 0.01],
) -> Tuple[Dict[str, Dict[float, float]], Dict[str, Dict[float, float]]]:
    """Compute average number of samples needed to reach each error threshold.

    Args:
        results: Dictionary of method results with ``'estimates'`` lists.
        true_mean: True mean value.
        thresholds: List of MAE thresholds (e.g., 0.05 = 5% error).

    Returns:
        ``(means_dict, stds_dict)`` where each maps
        ``method -> {threshold: value}``.
        If threshold is never reached, returns budget (max samples).
    """
    samples_mean: Dict[str, Dict[float, float]] = {}
    samples_std: Dict[str, Dict[float, float]] = {}

    for method, data in results.items():
        if not data.get("estimates"):
            continue

        samples_mean[method] = {}
        samples_std[method] = {}
        estimates_list = data["estimates"]

        for threshold in thresholds:
            samples_per_run = []

            for estimates in estimates_list:
                estimates = np.array(estimates)
                mae = np.abs(estimates - true_mean)

                below_threshold = np.where(mae <= threshold)[0]
                if len(below_threshold) > 0:
                    samples_per_run.append(below_threshold[0] + 1)
                else:
                    samples_per_run.append(len(estimates))

            samples_mean[method][threshold] = np.mean(samples_per_run)
            n_runs = len(samples_per_run)
            samples_std[method][threshold] = (
                np.std(samples_per_run) / np.sqrt(n_runs) if n_runs > 1 else 0
            )

    return samples_mean, samples_std


def print_results_table(
    results: Dict[str, Dict],
    true_mean: float,
    budget: int,
    n_runs: int,
    test_y: np.ndarray = None,
) -> None:
    """Print comprehensive BQ experiment results table.

    Prints MAE, AUC-MAE, accuracy, failure rate for all methods, and
    surrogate model metrics (RMSE, NLL, AUROC) for BQ methods with posteriors.
    """
    methods = [
        m
        for m in results.keys()
        if results[m].get("estimates") and len(results[m]["estimates"]) > 0
    ]
    method_names = {
        "bq_posterior": "BQ mean(u_t)",
        "bq_sample": "BQ mean(train_y)",
        "bq_rounded": "BQ mean(round(u_t))",
        "bq_vanilla": "BQ (Vanilla)",
        "random_bq": "Random+BQ",
        "random": "Random",
        "lr_lure": "LR+LURE",
        "rf_lure": "RF+LURE",
        "lr_is": "LR+IS",
        "rf_is": "RF+IS",
    }

    mae_all = {}
    accuracy_all = {}
    failure_rate_all = {}
    for method in methods:
        estimates = results[method]["estimates"]
        mae_all[method] = [np.abs(est - true_mean) for est in estimates]
        accuracy_all[method] = [est for est in estimates]
        failure_rate_all[method] = [1 - est for est in estimates]

    mae_mean = {m: np.mean(mae_all[m], axis=0) for m in methods}
    mae_std = {m: np.std(mae_all[m], axis=0) for m in methods}
    accuracy_mean = {m: np.mean(accuracy_all[m], axis=0) for m in methods}
    accuracy_std = {m: np.std(accuracy_all[m], axis=0) for m in methods}
    failure_rate_mean = {m: np.mean(failure_rate_all[m], axis=0) for m in methods}
    failure_rate_std = {m: np.std(failure_rate_all[m], axis=0) for m in methods}

    # AUC-MAE
    auc_mae_all = {}
    for method in methods:
        auc_mae_all[method] = [np.sum(mae) for mae in mae_all[method]]
    auc_mae_mean = {m: np.mean(auc_mae_all[m]) for m in methods}
    auc_mae_std = {m: np.std(auc_mae_all[m]) for m in methods}

    # BQ integral variance (if available)
    has_int_var = {}
    int_var_mean = {}
    int_var_std = {}
    for method in methods:
        iv_list = results[method].get("integral_variance", [])
        if iv_list:
            final_ivs = [float(iv[-1]) for iv in iv_list]
            has_int_var[method] = True
            int_var_mean[method] = np.mean(final_ivs)
            int_var_std[method] = np.std(final_ivs)
        else:
            has_int_var[method] = False
            int_var_mean[method] = float("nan")
            int_var_std[method] = float("nan")

    true_accuracy = true_mean
    true_failure_rate = 1 - true_mean

    print("\n" + "=" * 150)
    print("RESULTS TABLE")
    print("=" * 150)
    print(
        f"True Accuracy: {true_accuracy:.4f} ({true_accuracy*100:.2f}%) | "
        f"True Failure Rate: {true_failure_rate:.4f} ({true_failure_rate*100:.2f}%) | "
        f"Budget: {budget} | Runs: {n_runs}"
    )
    print("=" * 150)

    print(
        f"\n{'Method':<20} {'MAE@'+str(budget):>18} {'AUC-MAE':>18} "
        f"{'Est. Acc@'+str(budget):>20} {'True Acc':>12} "
        f"{'Var@'+str(budget):>16} {'Abstain':>8}"
    )
    print("-" * 120)

    abstain_threshold = 0.05
    for method in methods:
        mae_final = f"{mae_mean[method][-1]:.6f} ± {mae_std[method][-1]:.4f}"
        auc_mae_str = f"{auc_mae_mean[method]:.4f} ± {auc_mae_std[method]:.4f}"
        accuracy_final = (
            f"{accuracy_mean[method][-1]*100:.2f}% ± {accuracy_std[method][-1]*100:.2f}%"
        )
        true_acc_str = f"{true_accuracy*100:.2f}%"

        if has_int_var.get(method, False):
            var_str = f"{int_var_mean[method]:.6f} ± {int_var_std[method]:.4f}"
            final_std = np.sqrt(max(int_var_mean[method], 0.0))
            abstain_str = "YES" if final_std > abstain_threshold else "no"
        else:
            var_str = "N/A"
            abstain_str = "N/A"

        print(
            f"{method_names.get(method, method):<20} {mae_final:>18} "
            f"{auc_mae_str:>18} {accuracy_final:>20} {true_acc_str:>12} "
            f"{var_str:>16} {abstain_str:>8}"
        )

    print("-" * 120)

    # Surrogate Model Metrics (BQ methods only)
    bq_methods = [
        m
        for m in methods
        if "posteriors" in results[m] and results[m]["posteriors"]
    ]

    if bq_methods and test_y is not None:
        print("\n" + "=" * 100)
        print("SURROGATE MODEL METRICS (BQ Methods)")
        print("=" * 100)
        print("Evaluating posterior predictions on ALL samples (labeled + unlabeled)")
        print("-" * 100)

        surrogate_metrics = {}
        for method in bq_methods:
            posteriors = results[method]["posteriors"]
            rmse_list, nll_list, auroc_list = [], [], []

            for final_u_t, final_s_t in posteriors:
                residuals = final_u_t - test_y
                rmse = np.sqrt(np.mean(residuals ** 2))
                rmse_list.append(rmse)

                sigma_sq = np.clip(final_s_t, 1e-6, None)
                nll_per_point = 0.5 * np.log(2 * np.pi * sigma_sq) + (
                    residuals ** 2
                ) / (2 * sigma_sq)
                nll_list.append(np.mean(nll_per_point))

                if len(np.unique(test_y)) > 1:
                    try:
                        from sklearn.metrics import roc_auc_score

                        auroc_list.append(roc_auc_score(test_y, final_u_t))
                    except Exception:
                        auroc_list.append(np.nan)
                else:
                    auroc_list.append(np.nan)

            surrogate_metrics[method] = {
                "rmse": (np.mean(rmse_list), np.std(rmse_list)),
                "nll": (np.mean(nll_list), np.std(nll_list)),
                "auroc": (np.nanmean(auroc_list), np.nanstd(auroc_list)),
            }

        print(f"\n{'Method':<20} {'RMSE':>20} {'NLL':>20} {'AUROC':>20}")
        print("-" * 85)

        for method in bq_methods:
            rmse_mean, rmse_std = surrogate_metrics[method]["rmse"]
            nll_mean, nll_std = surrogate_metrics[method]["nll"]
            auroc_mean, auroc_std = surrogate_metrics[method]["auroc"]

            rmse_str = f"{rmse_mean:.6f} ± {rmse_std:.4f}"
            nll_str = f"{nll_mean:.4f} ± {nll_std:.4f}"
            auroc_str = (
                "N/A"
                if np.isnan(auroc_mean)
                else f"{auroc_mean:.4f} ± {auroc_std:.4f}"
            )

            print(
                f"{method_names.get(method, method):<20} {rmse_str:>20} "
                f"{nll_str:>20} {auroc_str:>20}"
            )

        print("-" * 85)
        print("RMSE: Lower is better. Measures prediction error of posterior mean.")
        print("NLL:  Lower is better. Measures calibration of posterior uncertainty.")
        print(
            "AUROC: Higher is better. Measures discrimination ability "
            "(0.5=random, 1.0=perfect)."
        )