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
Unified Performance Estimation Experiment.

Runs all 14 methods in a single invocation:

  Baselines (5) — all use RPF (raw prompt embeddings):
    1. Random            — Random sample selection, simple mean estimate
    2. RF+IS             — Random Forest + Importance Sampling
    3. LR+IS             — Logistic Regression + IS
    4. RF+LURE           — Random Forest + LURE
    5. LR+LURE           — Logistic Regression + LURE

  Random Selection + BQ (3):
    6. BQ-SF Rand        — Random selection + BQ posterior (linear kernel, score features)
    7. BQ-RPF Rand       — Random selection + BQ posterior (Matérn kernel, raw prompt features)
    8. BQ-TPF Rand       — Random selection + BQ posterior (Matérn kernel, tuned prompt features)

  Active Selection + BQ (6):
    9.  BQ-SF            — Active BQ (linear kernel, score features)
    10. BQ-RPF           — Active BQ (Matérn kernel, raw prompt features)
    11. BQ-TPF           — Active BQ (Matérn kernel, tuned prompt features)
    12. BQ-SF Rounded    — Active BQ-SF with rounded posterior mean
    13. BQ-RPF Rounded   — Active BQ-RPF with rounded posterior mean
    14. BQ-TPF Rounded   — Active BQ-TPF with rounded posterior mean

Feature Setups:
  SF  — Score Features:        linear kernel, score-derived prior (u, S)
  RPF — Raw Prompt Features:   Matérn 2.5 kernel, informative prior (pretrain mean), no training
  TPF — Tuned Prompt Features: Matérn 2.5 kernel, encoder-derived or neutral prior,
                                 trained encoder (--tpf-prior-mode=encoder|neutral)

Supported Problem Settings:
  new_pair      : Exclude only the target model-benchmark pair from training.
  new_benchmark : Exclude the ENTIRE target benchmark from training.
  new_model     : Exclude the target model predictions from ALL benchmarks.

Usage:
  python -m experiment.exp_performance_estimation \\
      --dataset svamp --target-model gemini25_flash \\
      --encoder-path data/checkpoints/encoder_holdout_svamp.pth \\
      --n-runs 5
"""

import argparse
import os
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

from proeval.sampler import (
    BQEncoderSampler, SamplingResult,
    load_predictions, load_embeddings,
    extract_model_predictions, setup_train_test_split
)
from proeval.sampler.bq import (
    _bq_active_sampling, _bq_random_sampling,
    _bq_matern_active_sampling, _bq_matern_random_sampling,
    _bq_encoder_sampling, _bq_encoder_random_sampling,
)
from proeval.sampler.baselines import (
    random_sampling,
    run_lr_is_evaluation, run_rf_is_evaluation,
    run_lr_lure_evaluation, run_rf_lure_evaluation
)
from experiment.helper import save_experiment_results


# Feature Preparation Helpers

def prepare_score_features(
    pred_matrix: np.ndarray,
    target_idx: int,
    pretrain_indices: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare SF (Score Features) for linear kernel BQ.

    Returns:
        (test_x, u, S) — centered/normalized score features, prior mean, prior cov.
    """
    pretrain_matrix = pred_matrix[:, pretrain_indices]
    u = np.mean(pretrain_matrix, axis=1)
    S = np.cov(pretrain_matrix)
    if S.ndim == 0:
        S = np.array([[S]])

    test_x = pretrain_matrix.T
    n_pretrain = test_x.shape[0]
    if n_pretrain > 1:
        test_x = (test_x - u) / np.sqrt(n_pretrain - 1)
    else:
        test_x = test_x - u

    return test_x, u, S


def augment_pairwise_interactions(
    test_x: np.ndarray,
    pretrain_matrix: np.ndarray,
    top_k: int = 5,
    verbose: bool = True,
) -> np.ndarray:
    """Augment SF features with top-K pairwise interaction terms (ALGO-04).

    Selects the K most correlated pretrain model pairs and adds their
    element-wise products as additional features.  The linear kernel
    ``K = X_aug @ X_aug^T`` then approximates a degree-2 polynomial
    kernel, capturing agreement/disagreement patterns between models.

    Args:
        test_x: SF feature matrix (d, n_samples) — pretrain model predictions.
        pretrain_matrix: (n_samples, d) — raw pretrain predictions.
        top_k: Number of interaction pairs to add.
        verbose: Print selected pairs.

    Returns:
        Augmented feature matrix (d + top_k, n_samples).
    """
    d = pretrain_matrix.shape[1]
    if d < 2 or top_k < 1:
        return test_x

    # Compute pairwise correlation of pretrain model predictions
    corr = np.corrcoef(pretrain_matrix.T)
    np.fill_diagonal(corr, 0.0)  # exclude self-pairs

    # Select top-K pairs by absolute correlation
    upper = np.triu_indices(d, k=1)
    scores = np.abs(corr[upper])
    top_local = np.argsort(scores)[-top_k:]  # highest K

    selected_pairs = [(upper[0][i], upper[1][i]) for i in top_local]

    if verbose:
        print(f"\n[ALGO-04] Adding {len(selected_pairs)} pairwise interaction features:")
        for i, (a, b) in enumerate(selected_pairs):
            print(f"  Pair {i+1}: models {a}×{b} (corr={corr[a,b]:.4f})")

    # Build interaction features from test_x (which is pretrain_matrix.T)
    # test_x has shape (d, n_samples), so test_x[a] is model a's predictions
    interactions = np.zeros((top_k, test_x.shape[1]))
    for i, (a, b) in enumerate(selected_pairs):
        interactions[i] = test_x[a] * test_x[b]

    return np.vstack([test_x, interactions])


def optimize_noise_variance_lopo(
    pred_matrix: np.ndarray,
    pretrain_indices: List[int],
    budget: int = 13,
    seed: int = 42,
    noise_grid: Optional[List[float]] = None,
    verbose: bool = True,
) -> Tuple[float, Dict[float, float]]:
    """Optimize noise_variance via Leave-One-Pretrain-Out CV.

    For each candidate noise_variance, treats each pretrain model in turn
    as a pseudo-target and runs BQ active sampling using the remaining
    pretrain models as features.  Selects the noise_variance with the
    lowest average MAE across all held-out pretrain models.

    Args:
        pred_matrix: Full prediction matrix (n_samples, n_models).
        pretrain_indices: Indices of pretrain models to use as features.
        budget: Sampling budget for BQ (default 13).
        seed: Random seed for reproducibility.
        noise_grid: Candidate noise_variance values to try.
        verbose: Print progress.

    Returns:
        (best_noise_variance, noise_to_mae_dict)
    """
    if noise_grid is None:
        noise_grid = [0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0]

    # Build pretrain matrix once
    pretrain_matrix = pred_matrix[:, pretrain_indices]
    n_pretrain = pretrain_matrix.shape[1]

    if n_pretrain < 3:
        if verbose:
            print("[ALGO-01] Too few pretrain models for CV, using default noise_variance=0.3")
        return 0.3, {}

    noise_to_mae = {}
    best_noise = 0.3
    best_mae = float("inf")

    for noise_var in noise_grid:
        np.random.seed(seed)
        maes = []

        for pseudo_target_j in range(n_pretrain):
            # Hold out pretrain model j as pseudo-target
            pseudo_y = pretrain_matrix[:, pseudo_target_j].copy()
            feature_indices = [k for k in range(n_pretrain) if k != pseudo_target_j]

            # Features from remaining pretrain models
            feature_matrix = pretrain_matrix[:, feature_indices]
            u_j = np.mean(feature_matrix, axis=1)
            S_j = np.cov(feature_matrix)
            if S_j.ndim == 0:
                S_j = np.array([[S_j]])

            test_x_j = feature_matrix.T
            n_feat = test_x_j.shape[0]
            if n_feat > 1:
                test_x_j = (test_x_j - u_j) / np.sqrt(n_feat - 1)
            else:
                test_x_j = test_x_j - u_j

            # Run BQ active sampling with this noise_variance
            result = _bq_active_sampling(
                test_x_j, pseudo_y, u_j, S_j,
                budget=budget, n_init=0,
                noise_variance=noise_var,
            )
            maes.append(result.mae(float(np.mean(pseudo_y))))

        avg_mae = float(np.mean(maes))
        std_mae = float(np.std(maes))
        noise_to_mae[noise_var] = avg_mae

        if verbose:
            print(f"  [ALGO-01] noise_var={noise_var:.2f}  LOPO-MAE={avg_mae:.6f} ± {std_mae:.6f}")

        if avg_mae < best_mae:
            best_mae = avg_mae
            best_noise = noise_var

    if verbose:
        print(f"  [ALGO-01] Best noise_variance = {best_noise:.2f} (LOPO-MAE={best_mae:.6f})")

    # Safety check: if best noise is extreme, fall back to default
    if best_noise < 0.01 or best_noise > 1.5:
        if verbose:
            print(f"  [ALGO-01] Best noise {best_noise:.2f} out of safe range, "
                  f"falling back to 0.3")
        best_noise = 0.3

    return best_noise, noise_to_mae


def compute_pretrain_model_weights(
    target_benchmark: str,
    target_model: str,
    pretrain_names: List[str],
    data_dir: str,
    temperature: float = 0.1,
    verbose: bool = True,
) -> np.ndarray:
    """Compute soft-min weights for pretrain models based on reference-benchmark similarity.

    Uses the GMM feature space from reference benchmarks to measure each pretrain
    model's similarity to the target model. Returns normalized weights that sum to 1.

    Args:
        target_benchmark: Name of the target benchmark.
        target_model: Name of the target model.
        pretrain_names: Names of pretrain models (in order matching pretrain_indices).
        data_dir: Data directory with prediction CSVs.
        temperature: Softmax temperature (lower = more peaked weights).
        verbose: Print weight details.

    Returns:
        Normalized weight array of shape (len(pretrain_names),).
    """
    from proeval.sampler.pretrain_selector import _build_features, get_reference_benchmarks

    ref_benchmarks = get_reference_benchmarks(target_benchmark, data_dir=data_dir)
    # Filter to benchmarks where the target model exists
    from proeval.sampler.pretrain_selector import _load_benchmark_predictions
    filtered_refs = []
    for bench in ref_benchmarks:
        try:
            _, bench_models = _load_benchmark_predictions(bench, data_dir)
            if target_model in bench_models:
                filtered_refs.append(bench)
        except Exception:
            continue

    if len(filtered_refs) < 2:
        if verbose:
            print("[ALGO-02] Too few reference benchmarks for weighting, using uniform weights")
        return np.ones(len(pretrain_names)) / len(pretrain_names)

    # Build features
    features, model_names = _build_features(filtered_refs, data_dir)

    if target_model not in model_names:
        if verbose:
            print(f"[ALGO-02] Target model {target_model} not in reference data, using uniform weights")
        return np.ones(len(pretrain_names)) / len(pretrain_names)

    target_idx = model_names.index(target_model)
    target_feat = features[target_idx]

    # Compute distances and weights
    weights = np.zeros(len(pretrain_names))
    for j, pname in enumerate(pretrain_names):
        if pname in model_names:
            p_idx = model_names.index(pname)
            dist_sq = np.sum((features[p_idx] - target_feat) ** 2)
            weights[j] = np.exp(-dist_sq / temperature)
        else:
            weights[j] = 1e-6  # near-zero weight for unknown models

    w_sum = np.sum(weights)
    if w_sum < 1e-10:
        return np.ones(len(pretrain_names)) / len(pretrain_names)
    weights = weights / w_sum

    if verbose:
        print(f"\n[ALGO-02] Pretrain model weights (temperature={temperature}):")
        for j, (pname, w) in enumerate(zip(pretrain_names, weights)):
            print(f"  {j+1:2d}. {pname:<20} weight={w:.4f}")
        # Entropy check
        ent = -np.sum(weights * np.log(weights + 1e-10))
        max_ent = np.log(len(weights))
        print(f"  Weight entropy: {ent:.3f} / {max_ent:.3f} "
              f"({100*ent/max_ent:.1f}% of uniform)")

    return weights


def _collect_result(methods_results, methods_variances, name, estimates, variance=None, true_mean=None, threshold=None):
    """Helper to collect a method's results across runs."""
    if threshold is not None and true_mean is not None:
        maes = np.abs(estimates - true_mean)
        reached = np.where(maes <= threshold)[0]
        if len(reached) > 0:
            stop_idx = reached[0]
            estimates = estimates[:stop_idx+1]
            if variance is not None:
                variance = variance[:stop_idx+1]
            # print(f"  [{name}] Reached threshold at step {stop_idx+1}")
            
    if name not in methods_results:
        methods_results[name] = []
    methods_results[name].append(estimates)
    if variance is not None:
        if name not in methods_variances:
            methods_variances[name] = []
        methods_variances[name].append(variance)


def _save_comprehensive_csv(
    df: pd.DataFrame,
    dataset: str,
    target_model: str,
    setting: str,
    test_y: np.ndarray,
    true_mean: float,
    methods_results: Dict[str, List[np.ndarray]],
    methods_variances: Dict[str, List[np.ndarray]],
    methods_indices: Dict[str, List[List[int]]],
    results_dir: str,
    budget: int,
) -> None:
    """Save a comprehensive per-problem CSV with all available information.

    For each method, run, and step, records the sampled problem's index,
    question text, ground truth, target model prediction/label, the BQ
    estimate at that step, MAE, and integral variance.

    The output filename includes ``failure_discovery`` and a timestamp.
    """
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Resolve target model's prediction and ground_truth columns
    pred_col = f"prediction_{target_model}"
    has_pred_col = pred_col in df.columns
    has_question = "question" in df.columns
    has_gt = "ground_truth" in df.columns

    records = []
    for method_name, runs_estimates in methods_results.items():
        run_indices_list = methods_indices.get(method_name, [])

        for run_idx, est_arr in enumerate(runs_estimates):
            # Sampled indices for this run (may not exist for some methods)
            sampled = (
                run_indices_list[run_idx]
                if run_idx < len(run_indices_list)
                else []
            )
            var_arr = None
            if method_name in methods_variances:
                if run_idx < len(methods_variances[method_name]):
                    var_arr = methods_variances[method_name][run_idx]

            n_steps = len(est_arr)
            for step_idx in range(n_steps):
                est = float(est_arr[step_idx])
                mae = abs(est - true_mean)

                # Problem index sampled at this step
                prob_idx = (
                    int(sampled[step_idx])
                    if step_idx < len(sampled)
                    else None
                )

                record = {
                    "dataset": dataset,
                    "target_model": target_model,
                    "setting": setting,
                    "method": method_name,
                    "run": run_idx,
                    "step": step_idx + 1,
                    "problem_index": prob_idx,
                    "question": (
                        str(df.iloc[prob_idx].get("question", ""))[:500]
                        if prob_idx is not None and has_question
                        else ""
                    ),
                    "ground_truth": (
                        str(df.iloc[prob_idx].get("ground_truth", ""))
                        if prob_idx is not None and has_gt
                        else ""
                    ),
                    "target_model_prediction": (
                        str(df.iloc[prob_idx].get(pred_col, ""))
                        if prob_idx is not None and has_pred_col
                        else ""
                    ),
                    "target_model_label": (
                        float(test_y[prob_idx])
                        if prob_idx is not None
                        else None
                    ),
                    "estimate": est,
                    "mae": mae,
                    "true_mean": float(true_mean),
                    "integral_variance": (
                        float(var_arr[step_idx])
                        if var_arr is not None and step_idx < len(var_arr)
                        else None
                    ),
                    "budget": budget,
                }
                records.append(record)

    df_out = pd.DataFrame(records)
    csv_path = os.path.join(
        results_dir,
        f"failure_discovery_{dataset}_{target_model}_{setting}_{timestamp}.csv",
    )
    df_out.to_csv(csv_path, index=False)
    print(f"\nSaved comprehensive per-problem CSV ({len(df_out)} rows) to {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified Performance Estimation (all 14 methods)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, default="svamp")
    parser.add_argument("--target-model", type=str, default="gemini25_flash")
    parser.add_argument("--setting", type=str, default="new_pair",
                        choices=["new_pair", "new_benchmark", "new_model"])
    parser.add_argument("--data-selection", type=str, default="gmm",
                        choices=["all", "gmm"])
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--n-runs", type=int, default=1)
    parser.add_argument("--n-init", type=int, default=0,
                        help="Random warm-start samples for active learning")
    parser.add_argument("--noise-variance", type=float, default=0.3)
    parser.add_argument("--threshold", type=float, default=None,
                        help="MAE threshold for early stopping simulation.")
    parser.add_argument("--dynamic-budget", action="store_true",
                        help="Compute budget dynamically as max(50, 10% of dataset size).")
    parser.add_argument("--encoder-path", type=str, default=None,
                        help="Path to trained encoder .pth for TPF methods. "
                             "If not provided, TPF methods are skipped.")
    parser.add_argument("--embeddings-path", type=str, default=None)
    parser.add_argument("--tpf-prior-mode", type=str, default="encoder",
                        choices=["encoder", "neutral"],
                        help="TPF prior mean: 'encoder' uses the learned psi prior, "
                             "'neutral' uses constant 0.5.")
    parser.add_argument("--rpf-pca-dim", type=int, default=16,
                        help="Reduce RPF embeddings to this dimension via PCA (set to 0 to disable)")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = os.path.join(_PROJECT_ROOT, "data")

    print("=" * 70)
    print(f"Unified Performance Estimation — Dataset: {args.dataset.upper()} | Model: {args.target_model}")
    print(f"Setting: {args.setting} | Budget: {args.budget} | Runs: {args.n_runs}")
    print(f"Data Selection: {args.data_selection}")
    print(f"Encoder: {args.encoder_path or 'Not provided (TPF skipped)'}")
    print(f"TPF Prior Mode: {args.tpf_prior_mode}")
    print("=" * 70)

    # Load common data (once, outside run loop)
    df = load_predictions(args.dataset, data_dir=args.data_dir)
    pred_matrix, model_names = extract_model_predictions(df, args.dataset)

    if args.target_model not in model_names:
        raise ValueError(
            f"Model {args.target_model} not found. Available: {model_names}"
        )
    target_idx = model_names.index(args.target_model)
    test_y = pred_matrix[:, target_idx]
    true_mean = np.mean(test_y)
    n_samples = len(test_y)
    
    if args.dynamic_budget:
        budget = max(50, int(0.10 * n_samples))
        print(f"Dynamic budget set to {budget} (10% of {n_samples} dataset size)")
    else:
        budget = args.budget
        
    print(f"\nTrue Error Rate: {true_mean:.6f} | Samples: {n_samples} | Budget: {budget}")

    # GMM model selection (for SF pretrain + TPF encoder training)
    is_gmm_abstain = False
    if args.data_selection == "gmm" and args.setting != "new_model":
        from proeval.sampler.pretrain_selector import select_pretrain_models_gmm
        pretrain_indices, pretrain_names = select_pretrain_models_gmm(
            target_benchmark=args.dataset,
            target_model=args.target_model,
            data_dir=args.data_dir,
            verbose=True,
        )
        is_gmm_abstain = len(pretrain_indices) < 3
        print(f"[GMM] Abstention: {is_gmm_abstain} "
              f"(Found {len(pretrain_indices)} models, requires >= 3)")
    else:
        pretrain_indices = [i for i in range(len(model_names)) if i != target_idx]
        pretrain_names = [model_names[i] for i in pretrain_indices]

    # Prepare SF features (linear kernel, shared across runs)
    test_x_sf, u_sf, S_sf = prepare_score_features(
        pred_matrix, target_idx, pretrain_indices
    )
    print(f"\n[SF] Score features shape: {test_x_sf.shape}")

    # ALGO-01: Optimize noise_variance via LOPO-CV on pretrain data
    print("\n[ALGO-01] Optimizing noise_variance via Leave-One-Pretrain-Out CV...")
    optimized_noise, noise_mae_map = optimize_noise_variance_lopo(
        pred_matrix, pretrain_indices,
        budget=budget, seed=args.seed, verbose=True,
    )
    if optimized_noise != args.noise_variance and args.noise_variance == 0.3:
        # Only override when using default noise (not explicitly set by user)
        print(f"[ALGO-01] Overriding noise_variance: {args.noise_variance} -> {optimized_noise:.2f}")
        args.noise_variance = optimized_noise
    elif optimized_noise != args.noise_variance:
        print(f"[ALGO-01] Optimized noise = {optimized_noise:.2f}, but keeping explicit --noise-variance {args.noise_variance}")
    else:
        print(f"[ALGO-01] Optimized noise_variance = {optimized_noise:.2f} (same as default)")

    # ALGO-02: Weighted prior NOT applied (uniform prior works better with GMM selection)
    # The weighted prior degraded MAE from 0.0087 to 0.0122. Using uniform prior
    # as computed by prepare_score_features() above.

    # ALGO-04: Augment SF features with pairwise interaction terms
    pretrain_matrix_sf = pred_matrix[:, pretrain_indices]
    test_x_sf = augment_pairwise_interactions(
        test_x_sf, pretrain_matrix_sf, top_k=5, verbose=True,
    )
    print(f"[ALGO-04] Augmented SF features shape: {test_x_sf.shape}")

    # Prepare RPF features (Matérn kernel, raw prompt embeddings)
    embeddings = load_embeddings(args.dataset, data_dir=args.data_dir)
    if args.rpf_pca_dim > 0:
        from sklearn.decomposition import PCA
        print(f"[RPF] Reducing embeddings to {args.rpf_pca_dim}d via PCA...")
        pca = PCA(n_components=args.rpf_pca_dim)
        embeddings = pca.fit_transform(embeddings)
    # Informative prior: mean of pretrain model predictions per sample
    u_rpf = np.mean(pred_matrix[:, pretrain_indices], axis=1)
    print(f"[RPF] Embeddings shape: {embeddings.shape} | Prior: informative "
          f"(pretrain mean, range [{u_rpf.min():.3f}, {u_rpf.max():.3f}], "
          f"mean {u_rpf.mean():.3f})")

    # Prepare TPF features (Matérn kernel, encoder-based)
    sampler_tpf = None
    if args.encoder_path:
        emb_path = args.embeddings_path or os.path.join(
            args.data_dir,
            f"{args.dataset}_embeddings_text_embedding_3_large.npy"
        )
        try:
            sampler_tpf = BQEncoderSampler(
                encoder_path=args.encoder_path,
                embeddings_path=emb_path,
                noise_variance=args.noise_variance,
            )
            # Optionally override encoder prior with neutral 0.5
            if args.tpf_prior_mode == "neutral":
                sampler_tpf.prior_u = np.ones(n_samples) * 0.5
                print(f"[TPF] Prior overridden to neutral 0.5 (--tpf-prior-mode=neutral)")
            else:
                print(f"[TPF] Using encoder prior. Range: [{sampler_tpf.prior_u.min():.3f}, {sampler_tpf.prior_u.max():.3f}]")
            print(f"[TPF] Encoder loaded. Phi shape: {sampler_tpf.phi_embeddings.shape}")
        except Exception as e:
            print(f"[TPF] Warning: Failed to load encoder: {e}")
            print("[TPF] TPF methods will be skipped.")

    # Run loop
    methods_results: Dict[str, List[np.ndarray]] = {}
    methods_variances: Dict[str, List[np.ndarray]] = {}
    methods_indices: Dict[str, List[List[int]]] = {}

    for run_id in range(args.n_runs):
        run_seed = args.seed + run_id
        np.random.seed(run_seed)

        print(f"\n{'─' * 60}")
        print(f"Run {run_id + 1}/{args.n_runs} (Seed: {run_seed})")
        print(f"{'─' * 60}")

        # Baselines (all use RPF = raw prompt embeddings)

        # 1. Random
        res_rand, rand_indices = random_sampling(test_y, budget=budget)
        _collect_result(methods_results, methods_variances, "Random", res_rand, true_mean=true_mean, threshold=args.threshold)
        methods_indices.setdefault("Random", []).append(list(rand_indices))

        # 2-5. Incremental baselines
        seed_size = min(8, budget // 2) if budget > 2 else 1
        try:
            res_rf_is, rf_is_idx = run_rf_is_evaluation(
                test_y, embeddings, steps=budget, seed_size=seed_size
            )
            _collect_result(methods_results, methods_variances, "RF+IS", res_rf_is, true_mean=true_mean, threshold=args.threshold)
            methods_indices.setdefault("RF+IS", []).append(list(rf_is_idx))

            res_lr_is, lr_is_idx = run_lr_is_evaluation(
                test_y, embeddings, steps=budget, seed_size=seed_size
            )
            _collect_result(methods_results, methods_variances, "LR+IS", res_lr_is, true_mean=true_mean, threshold=args.threshold)
            methods_indices.setdefault("LR+IS", []).append(list(lr_is_idx))

            res_rf_lure, rf_lure_idx = run_rf_lure_evaluation(
                test_y, embeddings, steps=budget, seed_size=seed_size
            )
            _collect_result(methods_results, methods_variances, "RF+LURE", res_rf_lure, true_mean=true_mean, threshold=args.threshold)
            methods_indices.setdefault("RF+LURE", []).append(list(rf_lure_idx))

            res_lr_lure, lr_lure_idx = run_lr_lure_evaluation(
                test_y, embeddings, steps=budget, seed_size=seed_size
            )
            _collect_result(methods_results, methods_variances, "LR+LURE", res_lr_lure, true_mean=true_mean, threshold=args.threshold)
            methods_indices.setdefault("LR+LURE", []).append(list(lr_lure_idx))
        except Exception as e:
            print(f"  Warning: baseline methods failed: {e}")

        # BQ-SF (linear kernel, score features)

        # Active
        res_sf = _bq_active_sampling(
            test_x_sf, test_y, u_sf, S_sf,
            budget=budget, n_init=args.n_init,
            noise_variance=args.noise_variance,
        )
        _collect_result(
            methods_results, methods_variances,
            "BQ-SF", res_sf.estimates, res_sf.integral_variance,
            true_mean=true_mean, threshold=args.threshold
        )
        _collect_result(
            methods_results, methods_variances,
            "BQ-SF Rounded", res_sf.rounded_estimates, res_sf.integral_variance,
            true_mean=true_mean, threshold=args.threshold
        )
        methods_indices.setdefault("BQ-SF", []).append(list(res_sf.selected_indices))
        methods_indices.setdefault("BQ-SF Rounded", []).append(list(res_sf.selected_indices))

        # Random + BQ
        res_sf_rand = _bq_random_sampling(
            test_x_sf, test_y, u_sf, S_sf,
            budget=budget, noise_variance=args.noise_variance,
        )
        _collect_result(
            methods_results, methods_variances,
            "BQ-SF Rand", res_sf_rand.estimates, res_sf_rand.integral_variance,
            true_mean=true_mean, threshold=args.threshold
        )
        methods_indices.setdefault("BQ-SF Rand", []).append(list(res_sf_rand.selected_indices))

        # BQ-RPF (Matérn kernel, raw prompt features, neutral prior)

        # Active
        res_rpf = _bq_matern_active_sampling(
            embeddings, test_y, u_rpf,
            budget=budget, n_init=0,  # stratified init hurts RPF (Matérn kernel)
            noise_variance=args.noise_variance,
        )
        _collect_result(
            methods_results, methods_variances,
            "BQ-RPF", res_rpf.estimates, res_rpf.integral_variance,
            true_mean=true_mean, threshold=args.threshold
        )
        _collect_result(
            methods_results, methods_variances,
            "BQ-RPF Rounded", res_rpf.rounded_estimates, res_rpf.integral_variance,
            true_mean=true_mean, threshold=args.threshold
        )
        methods_indices.setdefault("BQ-RPF", []).append(list(res_rpf.selected_indices))
        methods_indices.setdefault("BQ-RPF Rounded", []).append(list(res_rpf.selected_indices))

        # Random + BQ
        res_rpf_rand = _bq_matern_random_sampling(
            embeddings, test_y, u_rpf,
            budget=budget, noise_variance=args.noise_variance,
        )
        _collect_result(
            methods_results, methods_variances,
            "BQ-RPF Rand", res_rpf_rand.estimates, res_rpf_rand.integral_variance,
            true_mean=true_mean, threshold=args.threshold
        )
        methods_indices.setdefault("BQ-RPF Rand", []).append(list(res_rpf_rand.selected_indices))

        # BQ-TPF (Matérn kernel, tuned prompt features, encoder prior)
        if sampler_tpf is not None:
            # Active (via encoder sampling)
            res_tpf = _bq_encoder_sampling(
                sampler_tpf.phi_embeddings, test_y,
                sampler_tpf.prior_u,
                sampler_tpf.noise_variance,
                sampler_tpf.encoder,
                budget=budget, n_init=args.n_init,
            )
            _collect_result(
                methods_results, methods_variances,
                "BQ-TPF", res_tpf.estimates, res_tpf.integral_variance,
                true_mean=true_mean, threshold=args.threshold
            )
            _collect_result(
                methods_results, methods_variances,
                "BQ-TPF Rounded", res_tpf.rounded_estimates, res_tpf.integral_variance,
                true_mean=true_mean, threshold=args.threshold
            )
            methods_indices.setdefault("BQ-TPF", []).append(list(res_tpf.selected_indices))
            methods_indices.setdefault("BQ-TPF Rounded", []).append(list(res_tpf.selected_indices))

            # Random + encoder BQ
            res_tpf_rand = _bq_encoder_random_sampling(
                sampler_tpf.phi_embeddings, test_y,
                sampler_tpf.prior_u,
                sampler_tpf.noise_variance,
                sampler_tpf.encoder,
                budget=budget,
            )
            _collect_result(
                methods_results, methods_variances,
                "BQ-TPF Rand", res_tpf_rand.estimates, res_tpf_rand.integral_variance,
                true_mean=true_mean, threshold=args.threshold
            )
            methods_indices.setdefault("BQ-TPF Rand", []).append(list(res_tpf_rand.selected_indices))

    # Aggregate & Print Results
    print("\n" + "=" * 70)
    print(f"AGGREGATE RESULTS ({args.n_runs} runs)")
    print("=" * 70)
    print(f"True Error Rate: {true_mean:.6f}")
    print("-" * 70)

    final_mae = {}
    for name, run_estimates in methods_results.items():
        all_mae = [np.abs(est[-1] - true_mean) for est in run_estimates]
        all_preds = [est[-1] for est in run_estimates]
        mean_mae = np.mean(all_mae)
        median_mae = np.median(all_mae)
        mean_pred = np.mean(all_preds)
        std_mae = np.std(all_mae) if len(all_mae) > 1 else 0.0

        mean_final_var = None
        bq_abstain_rate = None
        if name in methods_variances:
            final_vars = [var_arr[-1] for var_arr in methods_variances[name]]
            mean_final_var = np.mean(final_vars)
            abstained_runs = sum(1 for v in final_vars if np.sqrt(v) > 0.05)
            bq_abstain_rate = abstained_runs / len(final_vars)

        final_mae[name] = {
            "mean": mean_mae, "median": median_mae,
            "std": std_mae, "var": mean_final_var,
            "abstain_rate": bq_abstain_rate,
        }
        print(f"{name:<20}: Pred = {mean_pred:.6f} | MAE = {mean_mae:.6f} ± {std_mae:.6f}")

    # Build summary table
    df_table = pd.DataFrame([
        {
            "Method": name,
            "Mean MAE": stats["mean"],
            "Median MAE": stats["median"],
            "Std MAE": stats["std"],
            "Mean Final Var": stats["var"],
            "BQ_Abstain_Rate": stats["abstain_rate"],
        }
        for name, stats in final_mae.items()
    ])

    results_dir = os.path.join(_PROJECT_ROOT, "results")
    save_experiment_results(
        dataset=args.dataset,
        target_model=args.target_model,
        setting=args.setting,
        methods_results=methods_results,
        true_mean=true_mean,
        df_table=df_table,
        results_dir=results_dir,
        gmm_abstained=is_gmm_abstain,
        methods_variances=methods_variances,
        budget=budget,
    )

    # Save comprehensive per-problem CSV (failure_discovery format).
    # Contains all available information: problem text, ground truth,
    # model predictions, sampling order, estimates, etc.
    _save_comprehensive_csv(
        df=df,
        dataset=args.dataset,
        target_model=args.target_model,
        setting=args.setting,
        test_y=test_y,
        true_mean=true_mean,
        methods_results=methods_results,
        methods_variances=methods_variances,
        methods_indices=methods_indices,
        results_dir=results_dir,
        budget=budget,
    )

    # Save threshold summary CSV (when --threshold is used).
    # Records the number of samples each method needed per run.
    if args.threshold is not None:
        threshold_records = []
        for method_name, runs_estimates in methods_results.items():
            for run_idx, est_arr in enumerate(runs_estimates):
                maes = np.abs(np.array(est_arr) - true_mean)
                crossed = np.where(maes <= args.threshold)[0]
                if len(crossed) > 0:
                    n_samples_needed = int(crossed[0]) + 1  # 1-indexed
                    reached = True
                else:
                    n_samples_needed = budget
                    reached = False

                threshold_records.append({
                    "dataset": args.dataset,
                    "method": method_name,
                    "run": run_idx,
                    "threshold": args.threshold,
                    "max_budget": budget,
                    "samples_to_threshold": n_samples_needed,
                    "reached": reached,
                    "true_mean": float(true_mean),
                })

        df_threshold = pd.DataFrame(threshold_records)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        thr_path = os.path.join(
            results_dir,
            f"threshold_{args.dataset}_{args.target_model}_thr{args.threshold}_{args.setting}_{timestamp}.csv"
        )
        df_threshold.to_csv(thr_path, index=False)
        print(f"\nSaved threshold summary to {thr_path}")

        # Print threshold summary
        print(f"\n{'─' * 60}")
        print(f"THRESHOLD SUMMARY — MAE <= {args.threshold}")
        print(f"{'─' * 60}")
        for method_name in methods_results:
            method_rows = df_threshold[df_threshold["method"] == method_name]
            avg_samples = method_rows["samples_to_threshold"].mean()
            sem = method_rows["samples_to_threshold"].std() / np.sqrt(len(method_rows)) if len(method_rows) > 1 else 0
            n_reached = method_rows["reached"].sum()
            n_total = len(method_rows)
            print(f"  {method_name:<20}: {avg_samples:>6.1f} ± {sem:.1f} samples (reached: {n_reached}/{n_total})")
        print(f"{'─' * 60}")


if __name__ == "__main__":
    main()
