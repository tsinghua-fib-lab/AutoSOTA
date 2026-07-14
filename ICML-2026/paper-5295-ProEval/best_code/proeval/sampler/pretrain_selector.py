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

"""Pretrain model selection for BQ active sampling.

Provides automatic selection of pretrain models using GMM clustering on
reference benchmark features. The key idea: models that perform similarly
on reference benchmarks are likely to be similar on the target benchmark,
making them good pretrain sources — without needing the target model's
actual evaluation results.

Example::

    from proeval.sampler.pretrain_selector import select_pretrain_models_gmm

    indices, names = select_pretrain_models_gmm(
        target_benchmark="svamp",
        target_model="gemini25_flash",
    )

"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from proeval.sampler.data import _default_data_dir


# Reference benchmark discovery
def get_reference_benchmarks(
    target_benchmark: str, data_dir: Optional[str] = None
) -> List[str]:
    """Return all available benchmarks as references, excluding *target_benchmark*.

    Scans *data_dir* for ``*_predictions.csv`` files and returns every
    benchmark name found except the target itself.

    Args:
        target_benchmark: Benchmark being evaluated (will be excluded).
        data_dir: Directory containing prediction CSVs.  Defaults to ``data/``.

    Returns:
        Sorted list of benchmark names.
    """
    if data_dir is None:
        data_dir = _default_data_dir()

    benchmarks: List[str] = []
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith("_predictions.csv"):
            name = fname.replace("_predictions.csv", "")
            if name != target_benchmark:
                benchmarks.append(name)
    return benchmarks


# Feature extraction from reference benchmark predictions
def _load_benchmark_predictions(
    benchmark: str, data_dir: str
) -> Tuple[np.ndarray, List[str]]:
    """Load binary predictions for all models on a single benchmark.

    Returns:
        ``(prediction_matrix, model_names)`` where ``prediction_matrix`` has
        shape ``(n_questions, n_models)``.
    """
    csv_path = os.path.join(data_dir, f"{benchmark}_predictions.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Prediction file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    model_cols = [c for c in df.columns if c.startswith("label_")]
    if not model_cols:
        raise ValueError(f"No label_ columns found in {csv_path}")

    prediction_matrix = df[model_cols].values
    model_names = [c.replace("label_", "") for c in model_cols]
    return prediction_matrix, model_names


def _build_features(
    reference_benchmarks: List[str], data_dir: str
) -> Tuple[np.ndarray, List[str]]:
    """Build per-model feature vectors from reference benchmark predictions.

    For each model, the feature vector is ``[mean_acc_b1, std_b1, mean_acc_b2,
    std_b2, ...]`` across the reference benchmarks.  Models absent from a
    benchmark get that benchmark's features imputed with column means so that
    we retain as many models as possible.

    Returns:
        ``(features, model_names)`` where ``features`` has shape
        ``(n_models, 2 * n_benchmarks)``.
    """
    benchmark_data: Dict[str, Dict] = {}
    all_models: set = set()

    for bench in reference_benchmarks:
        preds, models = _load_benchmark_predictions(bench, data_dir)
        benchmark_data[bench] = {"predictions": preds, "models": models}
        all_models.update(models)

    all_models_sorted = sorted(all_models)
    if not all_models_sorted:
        raise ValueError(
            f"No models found across reference benchmarks: {reference_benchmarks}"
        )

    # Build feature matrix with NaN for missing entries
    n_models = len(all_models_sorted)
    n_features = 2 * len(reference_benchmarks)
    features = np.full((n_models, n_features), np.nan)

    for b_idx, bench in enumerate(reference_benchmarks):
        data = benchmark_data[bench]
        for model in data["models"]:
            m_idx = all_models_sorted.index(model)
            pred_idx = data["models"].index(model)
            preds = data["predictions"][:, pred_idx]
            features[m_idx, 2 * b_idx] = float(np.mean(preds))
            features[m_idx, 2 * b_idx + 1] = float(np.std(preds))

    # Impute NaN with column means
    col_means = np.nanmean(features, axis=0)
    for col in range(n_features):
        mask = np.isnan(features[:, col])
        features[mask, col] = col_means[col]

    return features, all_models_sorted


# GMM clustering
def _find_optimal_clusters(
    features: np.ndarray,
    max_clusters: int = 10,
    random_state: int = 42,
) -> int:
    """Select optimal GMM cluster count using BIC."""
    n_samples = features.shape[0]
    max_k = min(max_clusters, n_samples - 1)
    if max_k < 2:
        return 2

    bics: List[Tuple[int, float]] = []
    for k in range(2, max_k + 1):
        try:
            gmm = GaussianMixture(
                n_components=k, random_state=random_state,
                covariance_type="full", reg_covar=1e-3,
            )
            gmm.fit(features)
            bics.append((k, gmm.bic(features)))
        except Exception:  # noqa: BLE001
            # Fallback to diag if full covariance fails
            try:
                gmm = GaussianMixture(
                    n_components=k, random_state=random_state,
                    covariance_type="diag", reg_covar=1e-4,
                )
                gmm.fit(features)
                bics.append((k, gmm.bic(features)))
            except Exception:
                continue

    if not bics:
        return 2
    return min(bics, key=lambda x: x[1])[0]


# Public API
def select_pretrain_models_gmm(
    target_benchmark: str,
    target_model: str,
    data_dir: Optional[str] = None,
    reference_benchmarks: Optional[List[str]] = None,
    n_clusters: Optional[int] = None,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[List[int], List[str]]:
    """Select pretrain models via GMM clustering on reference benchmark features.

    Fits a GMM to per-model feature vectors derived from *reference_benchmarks*
    and returns the models in the same cluster as the target model.  This
    **does not** require knowing the target model's eval results on the target
    benchmark.

    Args:
        target_benchmark: Benchmark being evaluated (used to auto-select
            reference benchmarks if ``reference_benchmarks`` is ``None``).
        target_model: Name of the target model.
        data_dir: Directory containing prediction CSVs.  Defaults to
            ``data/``.
        reference_benchmarks: Benchmarks for clustering.  If ``None``,
            auto-selected by category.
        n_clusters: Number of GMM clusters.  ``None`` → auto-select via BIC.
        random_state: Seed for GMM.
        verbose: Print selection details.

    Returns:
        ``(pretrain_indices, pretrain_names)`` — indices into the model list
        of the *target benchmark's* prediction CSV, and the corresponding
        model names.  These indices can be passed directly as
        ``pretrain_indices`` to :meth:`BQPriorSampler.sample`.
    """
    if data_dir is None:
        data_dir = _default_data_dir()

    # Auto-select reference benchmarks if needed
    if reference_benchmarks is None:
        reference_benchmarks = get_reference_benchmarks(target_benchmark, data_dir)

    # Filter to only benchmarks that actually have the target model's predictions
    filtered_refs: List[str] = []
    for bench in reference_benchmarks:
        try:
            _, bench_models = _load_benchmark_predictions(bench, data_dir)
            if target_model in bench_models:
                filtered_refs.append(bench)
        except FileNotFoundError:
            continue
    if not filtered_refs:
        raise ValueError(
            f"Target model '{target_model}' not found in any reference benchmark."
        )
    reference_benchmarks = filtered_refs

    if verbose:
        print(f"\n{'='*60}")
        print("Pretrain Model Selection (GMM)")
        print(f"{'='*60}")
        print(f"Target benchmark:     {target_benchmark}")
        print(f"Target model:        {target_model}")
        print(f"Reference benchmarks: {reference_benchmarks}")

    # Build features from reference benchmarks
    features, ref_model_names = _build_features(reference_benchmarks, data_dir)

    if verbose:
        print(f"Models in references: {len(ref_model_names)}")
        print(f"Feature dimensions:   {features.shape[1]}")

    # Verify target model is present
    if target_model not in ref_model_names:
        raise ValueError(
            f"Target model '{target_model}' not found in reference data. "
            f"Available: {ref_model_names}"
        )
    target_ref_idx = ref_model_names.index(target_model)

    # Determine cluster count
    if n_clusters is None:
        n_clusters = _find_optimal_clusters(features, random_state=random_state)
    else:
        n_clusters = min(n_clusters, len(ref_model_names))
    if verbose:
        print(f"GMM clusters:         {n_clusters}")

    # Fit GMM (CODE-06: full covariance captures benchmark correlations)
    try:
        gmm = GaussianMixture(
            n_components=n_clusters, random_state=random_state,
            covariance_type="full", reg_covar=1e-3,
        )
        labels = gmm.fit_predict(features)
    except Exception:
        if verbose:
            print("  [CODE-06] Full covariance failed, falling back to diag")
        gmm = GaussianMixture(
            n_components=n_clusters, random_state=random_state,
            covariance_type="diag", reg_covar=1e-4,
        )
        labels = gmm.fit_predict(features)
    target_cluster = labels[target_ref_idx]

    # Models in same cluster (excluding target)
    selected_ref_names = [
        ref_model_names[i]
        for i in range(len(ref_model_names))
        if labels[i] == target_cluster and i != target_ref_idx
    ]

    if verbose:
        print(f"\nSelected {len(selected_ref_names)} pretrain models (cluster {target_cluster}):")
        for i, name in enumerate(selected_ref_names, 1):
            dist = float(np.linalg.norm(
                features[ref_model_names.index(name)] - features[target_ref_idx]
            ))
            print(f"  {i}. {name} (dist: {dist:.4f})")

    # Map back to target benchmark's model indices
    # Need to load the target benchmark's model list to resolve indices
    target_csv = os.path.join(data_dir, f"{target_benchmark}_predictions.csv")
    if not os.path.exists(target_csv):
        raise FileNotFoundError(f"Target predictions not found: {target_csv}")
    target_df = pd.read_csv(target_csv, nrows=0)  # header only
    target_model_cols = [c for c in target_df.columns if c.startswith("label_")]
    target_model_names = [c.replace("label_", "") for c in target_model_cols]

    pretrain_indices: List[int] = []
    pretrain_names: List[str] = []
    for name in selected_ref_names:
        if name in target_model_names:
            pretrain_indices.append(target_model_names.index(name))
            pretrain_names.append(name)
        elif verbose:
            print(f"  Warning: '{name}' not in target benchmark, skipping.")

    if not pretrain_indices:
        if verbose:
            print("Warning: GMM selected no valid pretrain models. Falling back to all.")
        # Fallback: use all models except target
        fallback_target_idx = (
            target_model_names.index(target_model)
            if target_model in target_model_names
            else 0
        )
        pretrain_indices = [
            i for i in range(len(target_model_names)) if i != fallback_target_idx
        ]
        pretrain_names = [target_model_names[i] for i in pretrain_indices]

    if verbose:
        print(f"\nFinal pretrain set ({len(pretrain_names)} models): {pretrain_names}")
        print(f"{'='*60}")

    return pretrain_indices, pretrain_names
