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

"""
Failure Discovery Experiment — Unified Online GP with SS Acquisition.

Supports both seed-pool sampling (SS, Rand) using pre-computed labels
and LLM-based generation (SS-Gen, TSS, Rand-T-Gen, Rand-Gen) across
any dataset in DATASET_CONFIGS (GSM8K, StrategyQA, etc.).

Feature Types:
  RPF — Raw Prompt Features:   Matérn 2.5 kernel on raw text embeddings, neutral prior (0.5)
  TPF — Tuned Prompt Features: Matérn 2.5 kernel on encoder phi embeddings, encoder-derived prior

Methods:
  RPF Methods (raw text embeddings):
  - SS-RPF      : Superlevel Set acquisition from seed pool (Matérn kernel, neutral prior)
  - SS-Gen-RPF  : SS anchor selection + LLM generation (Matérn kernel, neutral prior)
  - TSS-RPF     : Topic-aware SS — UCB topic + SS anchors + generation (Matérn kernel)

  TPF Methods (encoder features, requires --encoder-path):
  - SS-TPF      : Superlevel Set acquisition from seed pool (encoder Matérn prior)
  - SS-Gen-TPF  : SS anchor selection + LLM generation (encoder prior)
  - TSS-TPF     : Topic-aware SS + generation (encoder prior)

  Baselines (feature-agnostic):
  - Rand        : Random sampling from seed pool
  - Rand-T-Gen  : Random topic + LLM generation
  - Rand-Gen    : Pure random LLM generation

Usage:
  # Run all RPF methods on GSM8K
  python -m experiment.exp_failure_discovery --dataset gsm8k --model gemma3_27b --runall

  # Run TPF methods (auto-detects encoder from data/checkpoints/)
  python -m experiment.exp_failure_discovery --dataset gsm8k --model gemma3_27b --methods SS-TPF SS-Gen-TPF TSS-TPF

  # Explicit encoder path
  python -m experiment.exp_failure_discovery --dataset gsm8k --model gemma3_27b --methods SS-TPF \
      --encoder-path data/checkpoints/encoder_holdout_gsm8k_setting_new_pair.pth

  # Failure mode: stop after 20 failures
  python -m experiment.exp_failure_discovery --dataset gsm8k --model gemma3_27b --failure 20

  # Multi-run for variance estimation
  python -m experiment.exp_failure_discovery --dataset gsm8k --model gemma3_4b --runall --runs 3
"""

import argparse
import csv
import os
import random
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Resolve paths
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from proeval.sampler.bq import _get_posterior_matern
from proeval.sampler.data import (
    extract_model_predictions,
    load_embeddings,
    load_predictions,
)
from proeval.generator.core import (
    TopicAwareGenerator,
    extract_topics_bertopic,
    get_posterior_embedding,
    setup_encoder_prior,
    ss_acquisition,
)
from proeval.evaluator import (
    DATASET_CONFIGS,
    LLMPredictor,
)
from proeval.evaluator.client import resolve_model_name


# All supported methods (order matters for output)
ALL_METHODS = [
    "SS-RPF", "SS-Gen-RPF", "TSS-RPF",
    "SS-TPF", "SS-Gen-TPF", "TSS-TPF",
    "Rand", "Rand-T-Gen", "Rand-Gen",
]

# Methods that require LLM generation (the rest use the seed pool directly)
GENERATION_METHODS = {
    "SS-Gen-RPF", "TSS-RPF",
    "SS-Gen-TPF", "TSS-TPF",
    "Rand-T-Gen", "Rand-Gen",
}

# Methods that require a trained encoder (TPF = Tuned Prompt Features)
TPF_METHODS = {"SS-TPF", "SS-Gen-TPF", "TSS-TPF"}

# Map method name → TopicAwareGenerator strategy
METHOD_TO_STRATEGY = {
    "SS-Gen-RPF": "ss_gen",
    "TSS-RPF": "tss",
    "SS-Gen-TPF": "ss_gen",
    "TSS-TPF": "tss",
    "Rand-T-Gen": "random_topic",
    "Rand-Gen": "pure_random",
}


def _feature_type(method: str) -> str:
    """Return ``'tpf'`` or ``'rpf'`` for a method name."""
    return "tpf" if method in TPF_METHODS else "rpf"


def run_online_gp_experiment(
    df: pd.DataFrame,
    dataset: str,
    target_model: str,
    test_y: np.ndarray,
    method: str,
    n_iterations: int,
    target_failures: Optional[int] = None,
    noise_variance: float = 0.3,
    ss_threshold: float = 0.0,
    ss_beta: float = 1.96,
    top_k: int = 5,
    generator: Optional[TopicAwareGenerator] = None,
    evaluator: Optional[LLMPredictor] = None,
    quiet: bool = False,
    verbose: bool = False,
    # RPF parameters (raw text embeddings + Matérn kernel)
    rpf_embeddings: Optional[np.ndarray] = None,
    # TPF parameters (encoder phi embeddings + Matérn kernel)
    encoder=None,
    phi_embeddings: Optional[np.ndarray] = None,
    tpf_prior_u: Optional[np.ndarray] = None,
) -> Tuple[List[int], List[Dict], int]:
    """Run a single online GP experiment for a given method.

    For sampling methods (SS-RPF, SS-TPF, Rand), problems are drawn from the
    seed pool and scored using pre-computed labels.

    For generation methods (SS-Gen-RPF/TPF, TSS-RPF/TPF, Rand-T-Gen, Rand-Gen),
    problems are generated via TopicAwareGenerator and evaluated with
    LLMPredictor.

    Both RPF and TPF use Matérn 2.5 kernels:
    - RPF: raw text embeddings, neutral prior (0.5)
    - TPF: encoder phi embeddings, encoder-derived prior

    Args:
        df: Source DataFrame with ``question`` and ``ground_truth`` columns.
        dataset: Dataset name (e.g., ``"gsm8k"``, ``"strategyqa"``).
        test_y: Ground truth error labels ``(n_samples,)``
            (0 = correct, 1 = error/failure).
        method: One of ``ALL_METHODS``.
        n_iterations: Maximum iterations (used when ``target_failures`` is None).
        target_failures: Stop when this many failures found (if set).
        noise_variance: GP noise variance.
        ss_threshold: SS acquisition threshold λ.
        ss_beta: SS acquisition β.
        top_k: Number of hard anchors per generation step.
        generator: TopicAwareGenerator instance (required for generation methods).
        evaluator: LLMPredictor instance (required for generation methods).
        quiet: Suppress per-iteration output.
        verbose: Print detailed per-iteration output (question, ground truth,
            model answer, failure label).
        rpf_embeddings: Normalized raw text embeddings ``(n_samples, d)``
            (required for RPF sampling methods).
        encoder: Pre-trained neural encoder (required for TPF methods).
        phi_embeddings: Encoder phi features ``(n_samples, hidden_dim)``
            (required for TPF sampling methods).
        tpf_prior_u: Encoder-derived prior mean ``(n_samples,)``
            (required for TPF sampling methods).

    Returns:
        ``(cumulative_failures, records, total_iterations)``
    """
    is_generation = method in GENERATION_METHODS
    is_tpf = method in TPF_METHODS
    is_rpf = method in {"SS-RPF"} and not is_generation

    if is_generation and (generator is None or evaluator is None):
        raise ValueError(
            f"Method {method!r} requires both 'generator' and 'evaluator'."
        )

    if is_rpf and rpf_embeddings is None:
        raise ValueError(
            f"Method {method!r} requires 'rpf_embeddings'."
        )

    if is_tpf and not is_generation:
        if encoder is None or phi_embeddings is None or tpf_prior_u is None:
            raise ValueError(
                f"Method {method!r} requires 'encoder', 'phi_embeddings', "
                f"and 'tpf_prior_u'."
            )

    n_samples = len(test_y)

    # Determine stopping criteria
    if target_failures is not None:
        max_iterations = n_samples * 2  # safety limit
    else:
        max_iterations = n_iterations

    # Initialize GP posterior with prior (both RPF and TPF use Matérn)
    if is_tpf and not is_generation:
        u_prior = tpf_prior_u
        u_t = tpf_prior_u.copy()
        s_t = np.ones(n_samples)  # Matérn self-kernel ≈ 1
        _device = next(encoder.parameters()).device
    elif is_rpf:
        u_prior = np.ones(n_samples) * 0.5  # neutral prior for RPF
        u_t = u_prior.copy()
        s_t = np.ones(n_samples)  # Matérn self-kernel ≈ 1
    else:
        # Rand or generation methods (generation methods manage their own GP)
        u_prior = np.ones(n_samples) * 0.5
        u_t = u_prior.copy()
        s_t = np.ones(n_samples)

    labeled_indices: List[int] = []
    unlabeled_indices = list(range(n_samples))

    failures_found = 0
    history_failures: List[int] = []
    records: List[Dict] = []
    iteration = 0
    dataset_config = DATASET_CONFIGS[dataset]

    if not quiet:
        feat_tag = f" [{_feature_type(method).upper()}]" if method != "Rand" else ""
        mode_desc = f"target={target_failures} failures" if target_failures else f"{n_iterations} iterations"
        print(f"\n>>> {method}{feat_tag} — {mode_desc} <<<")

    while True:
        # Check stopping criteria
        if target_failures is not None:
            if failures_found >= target_failures:
                if not quiet:
                    print(f"  Reached {target_failures} failures after {iteration} iterations.")
                break
        else:
            if iteration >= n_iterations:
                break

        if iteration >= max_iterations:
            if not quiet:
                print(f"  Hit max iterations limit ({max_iterations}).")
            break

        iteration += 1

        if not is_generation:
            if not unlabeled_indices:
                if not quiet:
                    print("  No more unlabeled samples.")
                break

            if method == "Rand":
                # Random sampling from unlabeled pool
                seed_idx = int(np.random.choice(unlabeled_indices))
            else:
                # SS acquisition from GP posterior (works for both RPF and TPF)
                seed_idx = ss_acquisition(
                    u_t, s_t, unlabeled_indices,
                    threshold=ss_threshold,
                    beta=ss_beta,
                )

            # Use pre-computed label (test_y: 0 = correct, 1 = error/failure)
            score = float(test_y[seed_idx])  # already an error indicator

            labeled_indices.append(seed_idx)
            unlabeled_indices.remove(seed_idx)

            if score >= 0.5:
                failures_found += 1

            history_failures.append(failures_found)

            # Update GP posterior — branch on feature type
            if labeled_indices:
                if is_tpf:
                    # TPF: Matérn kernel GP posterior via encoder
                    phi_train = phi_embeddings[labeled_indices]
                    u_t, s_t = get_posterior_embedding(
                        phi_train, test_y[labeled_indices],
                        phi_embeddings, noise_variance,
                        labeled_indices, tpf_prior_u,
                        encoder, _device,
                    )
                elif is_rpf:
                    # RPF: Matérn kernel GP posterior on raw embeddings
                    u_t, s_t = _get_posterior_matern(
                        rpf_embeddings[labeled_indices],
                        test_y[labeled_indices],
                        rpf_embeddings, noise_variance,
                        labeled_indices, u_prior,
                    )

            # Look up pre-computed prediction from CSV
            pred_col = f"prediction_{target_model}"
            pred_val = (
                str(df.iloc[seed_idx].get(pred_col, ""))
                if pred_col in df.columns else ""
            )

            records.append({
                "method": method,
                "iteration": iteration,
                "seed_idx": seed_idx,
                "question": df.iloc[seed_idx].get("question", ""),
                "ground_truth": str(df.iloc[seed_idx].get("ground_truth", "")),
                "prediction": pred_val,
                "score": score,
                "cumulative_failures": failures_found,
                "prior_mean": float(u_prior[seed_idx]),
                "topic": "N/A",
            })

            if verbose:
                q_text = str(df.iloc[seed_idx].get("question", ""))[:120]
                gt_raw = str(df.iloc[seed_idx].get("ground_truth", ""))[:60]
                gt_cleaned = str(dataset_config.extract_ground_truth(df.iloc[seed_idx].get("ground_truth", "")))[:60]
                # Look up pre-computed prediction from CSV
                pred_col = f"prediction_{target_model}"
                pred_val = str(df.iloc[seed_idx].get(pred_col, "N/A"))[:60] if pred_col in df.columns else "N/A"
                label = "FAIL" if score >= 0.5 else "OK"
                print(f"  [{method}] iter={iteration} label={label} "
                      f"cum_fail={failures_found}")
                print(f"    Q:    {q_text}")
                print(f"    GT:   {gt_cleaned} (raw: {gt_raw})")
                print(f"    Pred: {pred_val}")
            elif not quiet and iteration % 10 == 0:
                print(f"  [{method}] iter={iteration} failures={failures_found} last_prior={u_prior[seed_idx]:.3f}")

        # ── Generation Methods ─────────────────────────────────────────
        else:
            strategy = METHOD_TO_STRATEGY[method]
            k = top_k if method in ("SS-Gen-RPF", "TSS-RPF", "SS-Gen-TPF", "TSS-TPF") else 0

            # Generate via TopicAwareGenerator
            case = generator.generate(strategy=strategy, k_examples=k)

            question = case.get("question", "")
            ground_truth = case.get("ground_truth", "")

            # Evaluate via LLMPredictor
            raw_resp, prediction, score = evaluator.evaluate(
                question, ground_truth, dataset_config,
            )

            # score from LLMPredictor: 0.0 = correct, 1.0 = error (error indicator)
            if score is None:
                score = 1.0  # treat parse errors as failures

            # Update generator's GP posterior
            generator.update(score)

            if score >= 0.5:
                failures_found += 1

            history_failures.append(failures_found)

            records.append({
                "method": method,
                "iteration": iteration,
                "question": question[:200],
                "ground_truth": str(ground_truth),
                "prediction": str(prediction) if prediction else "",
                "score": score,
                "cumulative_failures": failures_found,
                "topic": case.get("topic", "N/A"),
                "anchor_indices": case.get("anchor_indices", []),
            })

            if verbose:
                q_short = question[:120]
                gt_raw = str(ground_truth)[:60]
                gt_cleaned = str(dataset_config.extract_ground_truth(ground_truth))[:60]
                pred_short = str(prediction)[:60] if prediction else "N/A"
                label = "FAIL" if score >= 0.5 else "OK"
                topic_short = str(case.get("topic", ""))[:30]
                print(f"  [{method}] iter={iteration} label={label} "
                      f"topic={topic_short} cum_fail={failures_found}")
                print(f"    Q:    {q_short}")
                print(f"    GT:   {gt_cleaned} (raw: {gt_raw})")
                print(f"    Pred: {pred_short}")
            elif not quiet:
                topic_short = str(case.get("topic", ""))[:30]
                print(
                    f"  [{method}] iter={iteration} "
                    f"topic={topic_short} "
                    f"score={'FAIL' if score >= 0.5 else 'OK'} "
                    f"cum_fail={failures_found}"
                )

    return history_failures, records, iteration


def print_results_summary(
    all_runs: Dict[str, List[List[int]]],
    method_names: List[str],
    n_runs: int,
    n_iterations: int,
    dataset: str,
    target_model: str,
) -> None:
    """Print a formatted results table to the terminal."""
    print("\n" + "=" * 78)
    print(f"FAILURE DISCOVERY RESULTS — {dataset.upper()} | Model: {target_model}")
    print(f"Runs: {n_runs} | Budget: {n_iterations}")
    print("=" * 78)

    header = f"{'Method':<15} | {'Failures':>20} | {'Failure Rate':>14} | {'Iterations':>10}"
    print(header)
    print("-" * 78)

    for name in method_names:
        runs = all_runs.get(name, [])
        if not runs:
            print(f"{name:<15} | {'N/A':>20} | {'N/A':>14} | {'N/A':>10}")
            continue

        # Get final failure count from each run
        final_failures = [r[-1] if r else 0 for r in runs]
        n_iters = [len(r) for r in runs]

        mean_fail = np.mean(final_failures)
        std_fail = np.std(final_failures) if n_runs > 1 else 0.0
        mean_iters = np.mean(n_iters)
        mean_rate = mean_fail / mean_iters * 100 if mean_iters > 0 else 0.0

        if n_runs > 1:
            fail_str = f"{mean_fail:.1f} ± {std_fail:.1f}"
        else:
            fail_str = f"{mean_fail:.0f}"

        print(f"{name:<15} | {fail_str:>20} | {mean_rate:>12.1f}% | {mean_iters:>10.0f}")

    print("=" * 78)


def print_failure_mode_summary(
    results: Dict[str, Dict],
    target_failures: int,
) -> None:
    """Print failure-mode results (samples-to-failure)."""
    print("\n" + "=" * 78)
    print(f"FAILURE MODE — Target: {target_failures} failures")
    print("=" * 78)

    header = f"{'Method':<15} | {'Iterations to Target':>22} | {'Speedup vs Rand':>16}"
    print(header)
    print("-" * 78)

    rand_iters = results.get("Rand", {}).get("iterations", None)

    for name, data in results.items():
        iters = data.get("iterations", 0)
        if rand_iters and rand_iters > 0 and iters > 0:
            speedup = rand_iters / iters
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "—"
        print(f"{name:<15} | {iters:>22} | {speedup_str:>16}")

    print("=" * 78)


def export_records_csv(
    all_records: Dict[str, List[Dict]],
    output_path: str,
) -> None:
    """Export all records to a CSV file."""
    columns = [
        "Timestamp", "Method", "Run", "Iteration", "Question",
        "GroundTruth", "Prediction", "Score", "CumulativeFailures",
        "Topic", "SeedIdx", "PriorMean", "AnchorIndices",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)

        for method_name, records in all_records.items():
            for record in records:
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    method_name,
                    record.get("run_id", 1),
                    record.get("iteration", -1),
                    str(record.get("question", ""))[:500],
                    record.get("ground_truth", ""),
                    record.get("prediction", ""),
                    record.get("score", 0),
                    record.get("cumulative_failures", ""),
                    record.get("topic", ""),
                    record.get("seed_idx", ""),
                    record.get("prior_mean", ""),
                    str(record.get("anchor_indices", ""))[:200],
                ])

    total = sum(len(r) for r in all_records.values())
    print(f"\nExported {total} records to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Failure Discovery — Online GP with SS Acquisition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # All RPF methods on GSM8K
  python -m experiment.exp_failure_discovery --dataset gsm8k --model gemma3_27b --runall

  # TPF methods (auto-detects encoder)
  python -m experiment.exp_failure_discovery --dataset gsm8k --model gemma3_27b --methods SS-TPF TSS-TPF

  # Failure mode: stop after 20 failures
  python -m experiment.exp_failure_discovery --dataset gsm8k --model gemma3_27b --failure 20
""",
    )

    # Dataset and model
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        help="Dataset name (e.g., gsm8k, strategyqa, svamp)")
    parser.add_argument("--model", type=str, default="gemma3_27b",
                        help="Target model to evaluate (CSV column name)")
    parser.add_argument("--generator-model", type=str, default="gemini3_flash",
                        help="Generator LLM for generation methods (friendly name "
                             "or full OpenRouter identifier, e.g. gemini3_pro)")

    # Experiment control
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of iterations per method")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of runs for variance estimation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-iteration details (question, GT, answer, label)")

    # Method selection
    parser.add_argument("--runall", action="store_true",
                        help="Run all 9 methods and compare")
    parser.add_argument("--methods", nargs="+", type=str, default=None,
                        choices=ALL_METHODS,
                        help="Specific methods to run (default: SS-RPF + Rand)")
    parser.add_argument("--failure", type=int, default=None,
                        help="Failure mode: stop when N failures found")

    # GP / SS config
    parser.add_argument("--noise-variance", type=float, default=0.3,
                        help="GP noise variance")
    parser.add_argument("--ss-beta", type=float, default=1.96,
                        help="SS acquisition β parameter")
    parser.add_argument("--ss-threshold", type=float, default=0.0,
                        help="SS acquisition threshold λ")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of hard anchors for generation methods")

    # Topics
    parser.add_argument("--n-topics", type=int, default=11,
                        help="Number of topics for BERTopic")

    # TPF / Encoder config
    parser.add_argument("--encoder-path", type=str, default=None,
                        help="Path to trained encoder .pth for TPF methods. "
                             "If not provided, auto-detects from data/checkpoints/.")
    parser.add_argument("--embeddings-path", type=str, default=None,
                        help="Path to embeddings .npy for TPF methods. "
                             "Auto-detects if not provided.")
    parser.add_argument("--tpf-prior-mode", type=str, default="encoder",
                        choices=["encoder", "neutral"],
                        help="TPF prior mean: 'encoder' uses learned psi prior, "
                             "'neutral' uses constant 0.5.")
    parser.add_argument("--setting", type=str, default="new_pair",
                        choices=["new_pair", "new_benchmark"],
                        help="Encoder checkpoint setting for auto-detection.")
    parser.add_argument("--rpf-pca-dim", type=int, default=16,
                        help="PCA dim for RPF embeddings (0 to skip PCA).")

    # Data paths
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Data directory (default: data/)")

    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = os.path.join(_PROJECT_ROOT, "data")

    if args.runall:
        method_names = list(ALL_METHODS)
    elif args.methods:
        method_names = list(args.methods)
    elif args.failure is not None:
        # Failure mode — compare SS-RPF vs Rand by default
        method_names = ["SS-RPF", "Rand"]
    else:
        # Default: sampling methods only (no API key needed)
        method_names = ["SS-RPF", "Rand"]

    needs_generation = any(m in GENERATION_METHODS for m in method_names)
    needs_tpf = any(m in TPF_METHODS for m in method_names)

    print("=" * 70)
    print(f"Failure Discovery — {args.dataset.upper()} | Model: {args.model}")
    print(f"Methods: {', '.join(method_names)} | Iterations: {args.iterations} | Runs: {args.runs}")
    print("=" * 70)

    df = load_predictions(args.dataset, data_dir=args.data_dir)
    pred_matrix, model_names = extract_model_predictions(df, args.dataset)
    print(f"Loaded {len(df)} samples | Models: {model_names}")

    if args.model not in model_names:
        print(f"\nError: Model '{args.model}' not found. Available: {model_names}")
        return

    # Extract target labels (binary: 0 = correct, 1 = error/failure)
    target_idx = model_names.index(args.model)
    test_y = pred_matrix[:, target_idx]
    error_rate = np.mean(test_y)
    print(f"True error rate: {error_rate:.4f} ({error_rate:.1%} failures)")

    # Load raw text embeddings for RPF methods
    needs_rpf = any(
        m in {"SS-RPF", "SS-Gen-RPF", "TSS-RPF"} for m in method_names
    )
    rpf_embeddings = None

    if needs_rpf:
        raw_emb_path = os.path.join(
            args.data_dir,
            f"{args.dataset}_embeddings_text_embedding_3_large.npy",
        )
        raw_emb = load_embeddings(args.dataset, data_dir=args.data_dir)
        print(f"[RPF] Loaded raw embeddings: {raw_emb.shape}")

        # PCA dimensionality reduction (high-dim embeddings + Matérn can be slow)
        if args.rpf_pca_dim > 0 and raw_emb.shape[1] > args.rpf_pca_dim:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=args.rpf_pca_dim)
            raw_emb = pca.fit_transform(raw_emb)
            print(f"[RPF] PCA reduced to {raw_emb.shape[1]} dims "
                  f"({pca.explained_variance_ratio_.sum():.1%} variance)")

        # Normalize for Matérn kernel
        rpf_embeddings = raw_emb / (
            np.linalg.norm(raw_emb, axis=1, keepdims=True) + 1e-10
        )
        print(f"[RPF] Features shape: {rpf_embeddings.shape} | Prior: neutral 0.5")

    # Load encoder for TPF methods (if needed)
    encoder = None
    phi_embeddings = None
    tpf_prior_u = None

    if needs_tpf:
        # Resolve encoder path: explicit > auto-detect
        encoder_path = args.encoder_path
        if encoder_path is None:
            auto_path = os.path.join(
                args.data_dir, "checkpoints",
                f"encoder_holdout_{args.dataset}_setting_{args.setting}.pth",
            )
            if os.path.exists(auto_path):
                encoder_path = auto_path
                print(f"[TPF] Auto-detected encoder: {encoder_path}")
            else:
                print(f"[TPF] Error: No encoder found at {auto_path}")
                print("[TPF] Provide --encoder-path or train an encoder first.")
                print("[TPF] TPF methods will be skipped.")
                method_names = [m for m in method_names if m not in TPF_METHODS]
                needs_tpf = False

        if needs_tpf:
            # Resolve embeddings path
            emb_path = args.embeddings_path or os.path.join(
                args.data_dir,
                f"{args.dataset}_embeddings_text_embedding_3_large.npy",
            )

            try:
                encoder_obj, phi_emb, enc_u, enc_S, enc_var = setup_encoder_prior(
                    encoder_path, emb_path,
                )
                encoder = encoder_obj
                phi_embeddings = phi_emb

                if args.tpf_prior_mode == "neutral":
                    tpf_prior_u = np.ones(len(df)) * 0.5
                    print(f"[TPF] Prior overridden to neutral 0.5")
                else:
                    tpf_prior_u = enc_u
                    print(f"[TPF] Using encoder prior. Range: [{enc_u.min():.3f}, {enc_u.max():.3f}]")

                print(f"[TPF] Phi shape: {phi_embeddings.shape}")
            except Exception as e:
                print(f"[TPF] Error loading encoder: {e}")
                print("[TPF] TPF methods will be skipped.")
                method_names = [m for m in method_names if m not in TPF_METHODS]
                needs_tpf = False

    # Recalculate generation needs after possible TPF skip
    needs_generation = any(m in GENERATION_METHODS for m in method_names)

    # Topic extraction (if any generation methods are selected)
    topic_labels, topic_keywords, topic_assignments = None, None, None
    if needs_generation:
        questions = df["question"].tolist()
        try:
            topic_labels, topic_keywords, topic_assignments = extract_topics_bertopic(
                questions, n_topics=args.n_topics,
            )
            print(f"Extracted {len(topic_labels)} topics via BERTopic")
        except ImportError:
            print("Warning: BERTopic not installed. Generation methods will use fallback topics.")
            topic_labels = [f"Topic {i}" for i in range(args.n_topics)]
            topic_keywords = {}
            topic_assignments = [i % args.n_topics for i in range(len(df))]

    all_runs: Dict[str, List[List[int]]] = {name: [] for name in method_names}
    all_records: Dict[str, List[Dict]] = {name: [] for name in method_names}

    for run_id in range(args.runs):
        run_seed = args.seed + run_id
        np.random.seed(run_seed)
        random.seed(run_seed)

        if args.runs > 1:
            print(f"\n{'─'*50}")
            print(f"RUN {run_id + 1}/{args.runs} (seed={run_seed})")
            print(f"{'─'*50}")

        # Create generators and evaluator for generation methods (per run)
        generator_rpf = None
        generator_tpf = None
        evaluator_llm = None

        # RPF generator (for SS-Gen-RPF, TSS-RPF)
        needs_rpf_gen = any(
            m in {"SS-Gen-RPF", "TSS-RPF"} for m in method_names
        )
        # TPF generator (for SS-Gen-TPF, TSS-TPF)
        needs_tpf_gen = any(
            m in {"SS-Gen-TPF", "TSS-TPF"} for m in method_names
        )
        # Baseline generator (for Rand-T-Gen, Rand-Gen)
        needs_baseline_gen = any(
            m in {"Rand-T-Gen", "Rand-Gen"} for m in method_names
        )

        gen_model = resolve_model_name(args.generator_model)

        if needs_rpf_gen or needs_baseline_gen:
            generator_rpf = TopicAwareGenerator(
                df=df,
                dataset=args.dataset,
                model=gen_model,
                n_topics=args.n_topics,
                rpf_embeddings=rpf_embeddings,
                noise_variance=args.noise_variance,
                ss_threshold=args.ss_threshold,
                ss_beta=args.ss_beta,
            )

        if needs_tpf_gen and needs_tpf:
            emb_path = args.embeddings_path or os.path.join(
                args.data_dir,
                f"{args.dataset}_embeddings_text_embedding_3_large.npy",
            )
            enc_path = args.encoder_path or os.path.join(
                args.data_dir, "checkpoints",
                f"encoder_holdout_{args.dataset}_setting_{args.setting}.pth",
            )
            generator_tpf = TopicAwareGenerator(
                df=df,
                dataset=args.dataset,
                model=gen_model,
                n_topics=args.n_topics,
                encoder_path=enc_path,
                embeddings_path=emb_path,
                ss_threshold=args.ss_threshold,
                ss_beta=args.ss_beta,
            )

        if needs_generation:
            openrouter_model = resolve_model_name(args.model)
            evaluator_llm = LLMPredictor(model=openrouter_model)

        for method in method_names:
            # Select the right generator for this method
            if method in {"SS-Gen-TPF", "TSS-TPF"}:
                gen = generator_tpf
            elif method in {"SS-Gen-RPF", "TSS-RPF", "Rand-T-Gen", "Rand-Gen"}:
                gen = generator_rpf
            else:
                gen = None  # sampling methods don't need a generator

            history, records, total_iters = run_online_gp_experiment(
                df=df,
                dataset=args.dataset,
                target_model=args.model,
                test_y=test_y,
                method=method,
                n_iterations=args.iterations,
                target_failures=args.failure,
                noise_variance=args.noise_variance,
                ss_threshold=args.ss_threshold,
                ss_beta=args.ss_beta,
                top_k=args.top_k,
                generator=gen,
                evaluator=evaluator_llm,
                quiet=(args.runs > 1),
                verbose=args.verbose,
                # RPF parameters
                rpf_embeddings=rpf_embeddings,
                # TPF parameters
                encoder=encoder,
                phi_embeddings=phi_embeddings,
                tpf_prior_u=tpf_prior_u,
            )

            # Tag records with run ID
            for r in records:
                r["run_id"] = run_id + 1

            all_runs[method].append(history)
            all_records[method].extend(records)

    if args.failure is not None:
        # Failure mode summary
        failure_results = {}
        for name in method_names:
            runs = all_runs[name]
            total_iters = [len(r) for r in runs]
            failure_results[name] = {
                "iterations": int(np.mean(total_iters)),
            }
        print_failure_mode_summary(failure_results, args.failure)
    else:
        # Standard summary
        print_results_summary(
            all_runs, method_names, args.runs, args.iterations,
            args.dataset, args.model,
        )

    results_dir = os.path.join(_PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(
        results_dir,
        f"failure_discovery_{args.dataset}_{args.model}_{timestamp}.csv",
    )
    export_records_csv(all_records, csv_path)

    print("\nDone.")

if __name__ == "__main__":
    main()
