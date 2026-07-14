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

"""Baseline sampling methods for BQ active evaluation.

Provides BQ active sampling, vanilla BQ, random sampling, IS, and LURE
baseline methods along with results reporting utilities.
"""

import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# Config Classes


@dataclass
class SurrogateConfig:
    """Configuration for surrogate model in IS/LURE methods."""
    use_rf: bool = False  # If True, use RandomForest; else LogisticRegression
    # Logistic Regression params
    lr_solver: str = 'liblinear'
    lr_class_weight: str = 'balanced'
    lr_max_iter: int = 100
    # Random Forest params
    rf_n_estimators: int = 50
    rf_max_depth: int = 5
    rf_class_weight: str = 'balanced'
    rf_random_state: int = 42

@dataclass
class AcquisitionConfig:
    """Configuration for acquisition function in IS/LURE methods."""
    smoothing_factor: float = 0.01  # Mix with uniform distribution (0.1 = 10% uniform)
    target_class: int = 1  # Which class probability to use for sampling (0 = failure/unsafe)

@dataclass
class ISEstimatorConfig:
    """Configuration for Simplified IS estimator."""
    normalize_weights: bool = True  # Self-normalized importance sampling

@dataclass
class LUREEstimatorConfig:
    """Configuration for LURE estimator (full formula from paper)."""
    normalize_weights: bool = False  # LURE uses (1/M) normalization by default

# Path Configuration

# Resolve the project root directory (3 levels up from this file)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_DATA_DIR = os.path.join(_PROJECT_ROOT, 'data')

# Data Loading

def load_predictions_data(dataset_name: str, data_dir: str = None) -> pd.DataFrame:
    """Load predictions CSV for a dataset."""
    if data_dir is None:
        data_dir = _DEFAULT_DATA_DIR
    csv_path = os.path.join(data_dir, f'{dataset_name}_predictions.csv')
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Returning empty DF for import check.")
        return pd.DataFrame()
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} examples from {csv_path}")
    return df


def load_text_embeddings(dataset_name: str, data_dir: str = None, 
                          embedding_model: str = 'text_embedding_3_large') -> np.ndarray:
    """Load pre-computed text embeddings for a dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'gsm8k', 'strategyqa')
        data_dir: Directory containing embedding files (default: data/)
        embedding_model: Embedding model name (default: 'text_embedding_3_large')
        
    Returns:
        Embeddings array of shape (n_samples, n_features)
    """
    if data_dir is None:
        data_dir = _DEFAULT_DATA_DIR
    # Special case: GQA uses a different embedding filename
    if dataset_name == 'gqa':
        embedding_path = os.path.join(data_dir, 'gqa_embeddings.npy')
    else:
        embedding_path = os.path.join(data_dir, f'{dataset_name}_embeddings_{embedding_model}.npy')
    
    if not os.path.exists(embedding_path):
        raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
    
    embeddings = np.load(embedding_path)
    print(f"Loaded text embeddings: {embeddings.shape} from {embedding_path}")
    return embeddings




def extract_model_predictions(df: pd.DataFrame, dataset_name: str = None) -> Tuple[np.ndarray, List[str]]:
    """Extract model predictions from dataframe.
    
    For DICES dataset, ratings are normalized to 0-1.25 range (from 1-5 scale).
    We binarize using threshold 0.75 (rating 4 on 1-5 scale = 1.0 normalized).
    """
    model_columns = [col for col in df.columns if col.startswith('label_')]
    model_names = [col.replace('label_', '') for col in model_columns]
    
    model_data = {}
    for model_name in model_names:
        label_col = f'label_{model_name}'
        y_labels = df[label_col].values
        
        # Use raw labels: 1=error, 0=correct.
        # For DICES/DICES_T2I: continuous ratings, binarise at 0.5,
        # then invert so 1=unsafe/poor and 0=safe/good.
        # For other datasets: raw labels are already 1=error, 0=correct.
        if dataset_name in ['dices', 'dices_t2i']:
            y_error = (y_labels < 0.5).astype(float)
        else:
            y_error = y_labels
        
        model_data[model_name] = y_error
    
    prediction_matrix = np.column_stack([model_data[name] for name in model_names])
    return prediction_matrix, model_names


def setup_train_test_split(prediction_matrix: np.ndarray, target_model: Union[int, str],
                           pretrain_indices: Optional[List[int]] = None,
                           model_names: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split prediction matrix into pre-training data and test target.
    
    Args:
        prediction_matrix: Full prediction matrix (n_samples, n_models)
        target_model: Name or index of the model to target for testing
        pretrain_indices: Optional list of model indices to use for pre-training.
                         If None, uses all models except the target.
        model_names: List of model names (required when target_model is a string).
    """
    # Resolve target model to index
    if isinstance(target_model, str):
        if model_names is None:
            raise ValueError(
                "model_names must be provided when target_model is a string"
            )
        if target_model not in model_names:
            raise ValueError(
                f"Model {target_model!r} not found. Available: {model_names}"
            )
        target_model_index = model_names.index(target_model)
    else:
        target_model_index = int(target_model)

    test_y = prediction_matrix[:, target_model_index]
    
    if pretrain_indices is not None:
        # Use only specified pretrain models
        pretrain_matrix = prediction_matrix[:, pretrain_indices]
    else:
        # Use all models except the target
        pretrain_matrix = np.delete(prediction_matrix, slice(target_model_index, target_model_index+1), axis=1)
    
    u = np.mean(pretrain_matrix, axis=1)
    S = np.cov(pretrain_matrix)
    
    # Handle edge case: single pretrain model returns scalar covariance
    if S.ndim == 0:
        S = np.array([[S]])
    
    test_x = pretrain_matrix.T
    n_pretrain = test_x.shape[0]
    if n_pretrain > 1:
        test_x = (test_x - u) / np.sqrt(n_pretrain - 1)
    else:
        # Single pretrain model: no normalization needed, just center
        test_x = test_x - u
    
    return pretrain_matrix, test_x, test_y, u, S


# BQ Implementation
def get_posterior(train_x, train_y, test_x, noise_variance, train_x_inds, u, full_cov=False):
    k_t_inv = np.linalg.inv(np.dot(train_x, train_x.T)/noise_variance + np.eye(train_x.shape[0]))
    u_t = u + 1 / noise_variance * np.dot(np.dot(np.dot(test_x.T, k_t_inv), train_x), (train_y - u[train_x_inds]))
    if full_cov:
        s_t = np.dot(np.dot(test_x.T, k_t_inv), test_x)
    else:
        s_t = np.sum(np.dot(test_x.T, k_t_inv) * test_x.T, axis=1)
    return u_t, s_t


def find_best_i_and_update(A, K, Q, e):
    s = A.sum(axis=0)
    v = s @ K
    numerator_roots = Q @ v
    numerators = e * (numerator_roots ** 2)
    QK = Q @ K
    quad_forms = np.einsum('ij,ij->i', Q, QK)
    denominators = 1 + e * quad_forms
    scores = numerators / denominators
    best_i = np.argmax(scores)
    
    Q_best = Q[best_i]
    KQ_best = QK[best_i] 
    denom_best = denominators[best_i]
    B_best = e * np.outer(KQ_best, KQ_best) / denom_best
    result_matrix = K - B_best
    
    return best_i, result_matrix


def variance_improvement(train_x, k_t_inv, noise_variance, unlabeled_indices, test_x):
    if train_x.shape[1] == 0:
        k_t_inv = np.eye(test_x.shape[0])
    A = test_x.T
    K = k_t_inv
    Q = test_x[:, unlabeled_indices].T
    e = noise_variance
    best_local_idx, k_t_inv = find_best_i_and_update(A, K, Q, e)
    return unlabeled_indices[best_local_idx], k_t_inv


def bq_active_sampling(test_x: np.ndarray, test_y: np.ndarray, u: np.ndarray, S: np.ndarray,
                       budget: int, n_init: int = 0, noise_variance: float = 0.3,
                       use_bq_var: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], np.ndarray, np.ndarray]:
    n_samples = test_x.shape[1]
    
    good_indices = [index for index in range(len(u)) if u[index] > 0.2 and u[index] < 0.6]
    if len(good_indices) < n_init:
        good_indices = list(range(n_samples))
    
    labeled_indices = list(np.random.choice(len(good_indices), min(n_init, len(good_indices)), replace=False))
    labeled_indices = [good_indices[i] for i in labeled_indices] if n_init > 0 else []
    unlabeled_indices = [i for i in range(n_samples) if i not in labeled_indices]
    
    estimates = np.ones(budget) * np.mean(u)
    estimates2 = np.ones(budget) * np.mean(u)
    estimates3 = np.ones(budget) * np.mean(u)
    integral_variance = np.ones(budget) * np.mean(np.diag(S))
    
    k_t_inv = None
    
    for t in range(n_init, budget):
        train_x = test_x[:, labeled_indices]
        train_y = test_y[labeled_indices]
        
        if t == 0:
            u_t, s_t = u, np.diag(S)
        else:
            u_t, s_t = get_posterior(train_x, train_y, test_x, noise_variance, labeled_indices, u)
        
        uu = np.round(u_t)
        estimates[t] = np.mean(uu)
        estimates2[t] = np.mean(u_t)
        estimates3[t] = np.mean(train_y) if len(train_y) > 0 else np.mean(u)
        # Integral variance = mean of posterior marginal variances
        integral_variance[t] = np.mean(np.maximum(s_t, 0.0))
        
        if t < budget - 1:
            if use_bq_var:
                best_global_idx, k_t_inv = variance_improvement(train_x, k_t_inv, noise_variance, unlabeled_indices, test_x)
            else:
                best_local_idx = np.argmax(s_t[unlabeled_indices])
                best_global_idx = unlabeled_indices[best_local_idx]
            
            labeled_indices.append(best_global_idx)
            unlabeled_indices.remove(best_global_idx)
    
    for t in range(n_init):
        indices = labeled_indices[:t+1]
        train_x = test_x[:, indices]
        train_y = test_y[indices]
        u_t, s_t = get_posterior(train_x, train_y, test_x, noise_variance, indices, u)
        estimates[t] = np.mean(np.round(u_t))
        estimates2[t] = np.mean(u_t)
        estimates3[t] = np.mean(train_y)
        integral_variance[t] = np.mean(np.maximum(s_t, 0.0))
    
    # Compute final posterior for surrogate model metrics
    # This is the posterior after all budget samples have been collected
    final_train_x = test_x[:, labeled_indices]
    final_train_y = test_y[labeled_indices]
    if len(labeled_indices) > 0:
        final_u_t, final_s_t = get_posterior(final_train_x, final_train_y, test_x, noise_variance, labeled_indices, u)
    else:
        final_u_t, final_s_t = u, np.diag(S)
    
    return estimates, estimates2, estimates3, labeled_indices, final_u_t, final_s_t


def random_sampling(test_y: np.ndarray, budget: int) -> Tuple[np.ndarray, List[int]]:
    n_samples = len(test_y)
    labeled_indices = list(np.random.choice(n_samples, budget, replace=False))
    estimates = np.zeros(budget)
    for t in range(budget):
        sampled_indices = labeled_indices[:t+1]
        estimates[t] = np.mean(test_y[sampled_indices])
    return estimates, labeled_indices


# Vanilla BQ Implementation (Text Embeddings + Linear Kernel, No Prior)
def get_posterior_vanilla(train_x: np.ndarray, train_y: np.ndarray, 
                           test_x: np.ndarray, noise_variance: float,
                           full_cov: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get posterior for vanilla BQ with zero mean prior (no learned prior).
    
    Uses linear kernel: K(X, X') = X @ X'.T
    Prior mean: 0 (no informative prior)
    
    Args:
        train_x: Labeled sample embeddings (n_labeled, n_features)
        train_y: Labels of labeled samples (n_labeled,)
        test_x: All sample embeddings (n_samples, n_features)
        noise_variance: Observation noise variance
        full_cov: Whether to return full covariance matrix
        
    Returns:
        u_t: Posterior mean (n_samples,)
        s_t: Posterior variance (n_samples,) or covariance matrix
    """
    n_train = train_x.shape[0]
    
    # Linear kernel: K = X @ X.T
    K_train = train_x @ train_x.T  # (n_train, n_train)
    K_train_test = train_x @ test_x.T  # (n_train, n_test)
    
    # Regularize for numerical stability
    K_train_reg = K_train + noise_variance * np.eye(n_train)
    K_train_inv = np.linalg.inv(K_train_reg)
    
    # Posterior mean: K_test_train @ K_train^{-1} @ y
    u_t = K_train_test.T @ K_train_inv @ train_y
    
    # Posterior variance
    if full_cov:
        K_test = test_x @ test_x.T
        s_t = K_test - K_train_test.T @ K_train_inv @ K_train_test
    else:
        # Diagonal only (more efficient)
        K_test_diag = np.sum(test_x * test_x, axis=1)  # Diagonal of K_test
        s_t = K_test_diag - np.sum((K_train_test.T @ K_train_inv) * K_train_test.T, axis=1)
    
    return u_t, np.maximum(s_t, 1e-10)  # Ensure non-negative variance


def variance_improvement_vanilla(embeddings: np.ndarray, labeled_indices: List[int], 
                                   noise_variance: float, unlabeled_indices: List[int]) -> int:
    """
    Find the next best sample using variance improvement criterion for vanilla BQ.
    Selects the point that maximizes predictive variance reduction.
    """
    if len(labeled_indices) == 0:
        # No labeled samples yet, pick randomly
        return unlabeled_indices[np.random.randint(len(unlabeled_indices))]
    
    train_x = embeddings[labeled_indices]
    
    # Precompute K_train inverse
    K_train = train_x @ train_x.T
    K_train_reg = K_train + noise_variance * np.eye(len(labeled_indices))
    K_train_inv = np.linalg.inv(K_train_reg)
    
    # For each unlabeled point, compute predictive variance
    best_idx = 0
    best_var = -np.inf
    
    for i, ul_idx in enumerate(unlabeled_indices):
        x_new = embeddings[ul_idx]  # (n_features,)
        k_new = train_x @ x_new  # (n_train,)
        k_new_new = x_new @ x_new  # scalar (self-kernel)
        
        # Predictive variance for this point
        pred_var = k_new_new - k_new @ K_train_inv @ k_new
        
        if pred_var > best_var:
            best_var = pred_var
            best_idx = i
    
    return unlabeled_indices[best_idx]


def bq_vanilla_sampling(embeddings: np.ndarray, test_y: np.ndarray, 
                         budget: int, n_init: int = 0, 
                         noise_variance: float = 0.3) -> Tuple[np.ndarray, List[int], np.ndarray, np.ndarray]:
    """
    Vanilla BQ active sampling with text embeddings and linear kernel (no prior).
    
    This is a "pure" BQ method that:
    - Uses text embeddings as features
    - Uses linear kernel: k(x, x') = x^T x'
    - Has no informative prior (zero mean)
    
    Args:
        embeddings: Text embeddings (n_samples, n_features)
        test_y: True labels (n_samples,)
        budget: Sampling budget
        n_init: Number of random initial samples
        noise_variance: GP noise variance
        
    Returns:
        estimates: Posterior mean estimates at each step (budget,)
        labeled_indices: Selected sample indices
        final_u_t: Final posterior mean (n_samples,)
        final_s_t: Final posterior variance (n_samples,)
    """
    n_samples = embeddings.shape[0]
    
    # Normalize embeddings for numerical stability
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
    
    # Initialize with random samples if n_init > 0
    if n_init > 0:
        labeled_indices = list(np.random.choice(n_samples, min(n_init, n_samples), replace=False))
    else:
        labeled_indices = []
    unlabeled_indices = [i for i in range(n_samples) if i not in labeled_indices]
    
    estimates = np.zeros(budget)
    
    for t in range(budget):
        if t < n_init:
            # Already have init samples, just compute estimate
            pass
        else:
            # Select next sample using variance improvement
            if len(unlabeled_indices) > 0:
                best_idx = variance_improvement_vanilla(
                    embeddings_norm, labeled_indices, noise_variance, unlabeled_indices
                )
                labeled_indices.append(best_idx)
                unlabeled_indices.remove(best_idx)
        
        # Compute posterior and estimate
        if len(labeled_indices) > 0:
            train_x = embeddings_norm[labeled_indices]
            train_y = test_y[labeled_indices]
            u_t, s_t = get_posterior_vanilla(train_x, train_y, embeddings_norm, noise_variance)
            estimates[t] = np.mean(u_t)
        else:
            estimates[t] = 0.5  # No prior, default to 0.5
    
    # Compute final posterior
    if len(labeled_indices) > 0:
        final_u_t, final_s_t = get_posterior_vanilla(
            embeddings_norm[labeled_indices], test_y[labeled_indices], 
            embeddings_norm, noise_variance
        )
    else:
        final_u_t = np.zeros(n_samples)
        final_s_t = np.ones(n_samples)
    
    return estimates, labeled_indices, final_u_t, final_s_t


# Baseline Methods
def get_is_estimate(y: np.ndarray, embeddings: np.ndarray, n: int, 
                    seed_size: int = 20, 
                    surrogate_config: SurrogateConfig = None,
                    acquisition_config: AcquisitionConfig = None,
                    estimator_config: ISEstimatorConfig = None,
                    return_indices: bool = False):
    """
    Simplified Importance Sampling estimate (matching SimplifiedLUREEstimator).
    
    Uses surrogate model to compute acquisition probabilities, then applies
    self-normalized importance sampling weights.
    
    Args:
        y: True labels (n_samples,)
        embeddings: Feature embeddings (n_samples, n_features)
        n: Total budget (seed + active samples)
        seed_size: Number of initial uniform samples
        surrogate_config: Configuration for surrogate model
        acquisition_config: Configuration for acquisition function
        estimator_config: Configuration for IS estimator
        return_indices: If True, also return selected indices
        
    Returns:
        estimate: Estimated mean
        indices (optional): Selected sample indices
    """
    # Use default configs if not provided
    if surrogate_config is None:
        surrogate_config = SurrogateConfig()
    if acquisition_config is None:
        acquisition_config = AcquisitionConfig()
    if estimator_config is None:
        estimator_config = ISEstimatorConfig()
    
    n_total = len(y)
    all_indices = np.arange(n_total)
    
    # Handle small budget case
    if n <= seed_size:
        idx = np.random.choice(all_indices, size=n, replace=False)
        if return_indices:
            return y[idx].mean(), idx
        return y[idx].mean()
    
    # Step 1: Uniform seed sampling
    seed_indices = np.random.choice(all_indices, size=seed_size, replace=False)
    X_train = embeddings[seed_indices]
    y_train = y[seed_indices]
    
    # Handle degenerate case (single class in seed)
    if len(np.unique(y_train)) < 2:
        remaining = list(set(all_indices) - set(seed_indices))
        fill = np.random.choice(remaining, size=n-seed_size, replace=False)
        combined = np.concatenate([seed_indices, fill])
        if return_indices:
            return y[combined].mean(), combined
        return y[combined].mean()
    
    # Step 2: Train surrogate model
    if surrogate_config.use_rf:
        clf = RandomForestClassifier(
            n_estimators=surrogate_config.rf_n_estimators, 
            max_depth=surrogate_config.rf_max_depth, 
            random_state=surrogate_config.rf_random_state, 
            class_weight=surrogate_config.rf_class_weight
        )
    else:
        clf = LogisticRegression(
            solver=surrogate_config.lr_solver, 
            class_weight=surrogate_config.lr_class_weight,
            max_iter=surrogate_config.lr_max_iter
        )
    clf.fit(X_train, y_train)
    
    # Step 3: Compute acquisition probabilities (matching SurrogateProbAcquisition)
    remaining_pool = np.array(list(set(all_indices) - set(seed_indices)))
    X_pool = embeddings[remaining_pool]
    probs = clf.predict_proba(X_pool)
    raw_probs = probs[:, acquisition_config.target_class]
    
    # Smoothing: mix with uniform distribution
    smoothing = acquisition_config.smoothing_factor
    n_pool = len(remaining_pool)
    q = (1 - smoothing) * raw_probs + smoothing * (1.0 / n_pool)
    q = q / q.sum()
    
    # Step 4: Active sampling
    n_active = n - seed_size
    active_indices_local = np.random.choice(n_pool, size=n_active, replace=False, p=q)
    active_indices_global = remaining_pool[active_indices_local]
    
    # Step 5: Compute importance weights (matching SimplifiedLUREEstimator)
    # Seed samples have weight 1.0
    weights = [1.0] * seed_size
    
    # Active samples have importance weights: w = 1 / (n_pool * q)
    sampled_q = q[active_indices_local]
    imp_weights = 1.0 / (n_pool * sampled_q)
    weights.extend(imp_weights)
    
    weights = np.array(weights)
    
    # Self-normalized importance sampling
    if estimator_config.normalize_weights:
        weights = weights / weights.mean()
    
    # Step 6: Compute weighted estimate
    all_selected = np.concatenate([seed_indices, active_indices_global])
    all_labels = y[all_selected]
    
    estimate = np.sum(weights * all_labels) / np.sum(weights)
    
    if return_indices:
        return estimate, all_selected
    return estimate


def get_lure_estimate(y: np.ndarray, embeddings: np.ndarray, n: int, 
                      seed_size: int = 20,
                      surrogate_config: SurrogateConfig = None,
                      acquisition_config: AcquisitionConfig = None,
                      estimator_config: LUREEstimatorConfig = None,
                      return_indices: bool = False):
    """
    LURE importance sampling estimate (matching LUREEstimator).
    
    Uses the full LURE formula from the paper:
    R_LURE = (1/M) * Σ_{m=1}^M v_m * L(f(x_{i_m}), y_{i_m})
    
    where:
    - M = total number of labeled samples
    - N = total pool size
    - v_m = 1 + (N-M)/(N-m) * (1/((N-m+1)*q(i_m)) - 1)
    - q(i_m) = acquisition probability for sample i_m at step m
    
    Reference: https://arxiv.org/pdf/2103.05331
    
    Args:
        y: True labels (n_samples,)
        embeddings: Feature embeddings (n_samples, n_features)
        n: Total budget (seed + active samples)
        seed_size: Number of initial uniform samples
        surrogate_config: Configuration for surrogate model
        acquisition_config: Configuration for acquisition function
        estimator_config: Configuration for LURE estimator
        return_indices: If True, also return selected indices
        
    Returns:
        estimate: Estimated mean
        indices (optional): Selected sample indices
    """
    # Use default configs if not provided
    if surrogate_config is None:
        surrogate_config = SurrogateConfig()
    if acquisition_config is None:
        acquisition_config = AcquisitionConfig()
    if estimator_config is None:
        estimator_config = LUREEstimatorConfig()
    
    n_total = len(y)
    N = n_total  # Total pool size
    M = n  # Total number of samples we'll collect
    all_indices = np.arange(n_total)
    
    # Handle small budget case
    if n <= seed_size:
        idx = np.random.choice(all_indices, size=n, replace=False)
        if return_indices:
            return y[idx].mean(), idx
        return y[idx].mean()
    
    # Step 1: Uniform seed sampling
    seed_indices = list(np.random.choice(all_indices, size=seed_size, replace=False))
    X_train = embeddings[seed_indices]
    y_train = y[seed_indices]
    
    # Handle degenerate case (single class in seed)
    if len(np.unique(y_train)) < 2:
        remaining = list(set(all_indices) - set(seed_indices))
        fill = np.random.choice(remaining, size=n-seed_size, replace=False)
        combined = np.concatenate([seed_indices, fill])
        if return_indices:
            return y[combined].mean(), combined
        return y[combined].mean()
    
    # Step 2: Train surrogate model
    if surrogate_config.use_rf:
        clf = RandomForestClassifier(
            n_estimators=surrogate_config.rf_n_estimators, 
            max_depth=surrogate_config.rf_max_depth, 
            random_state=surrogate_config.rf_random_state, 
            class_weight=surrogate_config.rf_class_weight
        )
    else:
        clf = LogisticRegression(
            solver=surrogate_config.lr_solver, 
            class_weight=surrogate_config.lr_class_weight,
            max_iter=surrogate_config.lr_max_iter
        )
    clf.fit(X_train, y_train)
    
    # Step 3: Compute acquisition probabilities
    remaining_pool = np.array(list(set(all_indices) - set(seed_indices)))
    X_pool = embeddings[remaining_pool]
    probs = clf.predict_proba(X_pool)
    raw_probs = probs[:, acquisition_config.target_class]
    
    # Smoothing: mix with uniform distribution
    smoothing = acquisition_config.smoothing_factor
    n_pool = len(remaining_pool)
    q = (1 - smoothing) * raw_probs + smoothing * (1.0 / n_pool)
    q = q / q.sum()
    
    # Step 4: Active sampling
    n_active = n - seed_size
    active_indices_local = np.random.choice(n_pool, size=n_active, replace=False, p=q)
    active_indices_global = remaining_pool[active_indices_local]
    
    # Step 5: Compute LURE importance weights (full formula from paper)
    weights = []
    
    # For initial uniform samples: q(i_m) = 1/(N - m + 1)
    for m in range(1, seed_size + 1):
        q_i_m = 1.0 / (N - m + 1)
        # v_m = 1 + (N-M)/(N-m) * (1/((N-m+1)*q) - 1)
        v_m = 1 + (N - M) / (N - m) * (1 / ((N - m + 1) * q_i_m) - 1)
        weights.append(v_m)
    
    # For actively sampled points: use actual acquisition probabilities
    sampled_q = q[active_indices_local]
    for step_idx in range(n_active):
        m = seed_size + step_idx + 1  # 1-indexed step number
        q_i_m = sampled_q[step_idx]
        # v_m = 1 + (N-M)/(N-m) * (1/((N-m+1)*q) - 1)
        v_m = 1 + (N - M) / (N - m) * (1 / ((N - m + 1) * q_i_m) - 1)
        weights.append(v_m)
    
    weights = np.array(weights)
    
    # Step 6: Compute LURE estimate: R_LURE = (1/M) * Σ v_m * y_m
    all_selected = np.concatenate([seed_indices, active_indices_global])
    all_labels = y[all_selected]
    
    estimate = np.sum(weights * all_labels) / M
    
    if return_indices:
        return estimate, all_selected
    return estimate


def run_incremental_is_evaluation(y_true: np.ndarray, embeddings: np.ndarray,
                                   steps: int, seed_size: int = 8, 
                                   surrogate_config: SurrogateConfig = None,
                                   acquisition_config: AcquisitionConfig = None,
                                   estimator_config: ISEstimatorConfig = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Incremental Importance Sampling evaluation (matching SimplifiedLUREEstimator).
    
    Step-by-step active sampling that:
    1. Trains surrogate on labeled samples
    2. Uses stochastic uncertainty sampling
    3. Estimates mean using importance-weighted samples
    
    Args:
        y_true: True labels (n_samples,)
        embeddings: Feature embeddings (n_samples, n_features)
        steps: Total budget
        seed_size: Number of initial uniform samples
        surrogate_config: Configuration for surrogate model
        acquisition_config: Configuration for acquisition function
        estimator_config: Configuration for IS estimator
        
    Returns:
        estimations: Array of estimates at each step
        selected_indices: Array of selected sample indices
    """
    # Use default configs if not provided
    if surrogate_config is None:
        surrogate_config = SurrogateConfig()
    if acquisition_config is None:
        acquisition_config = AcquisitionConfig()
    if estimator_config is None:
        estimator_config = ISEstimatorConfig()
    
    n_total = len(y_true)
    all_indices = np.arange(n_total)
    
    # Phase 1: Random seed
    seed_size = min(seed_size, steps)
    seed_indices = list(np.random.choice(all_indices, size=seed_size, replace=False))
    
    # Initialize tracking
    selected_indices = list(seed_indices)
    acquisition_probs_history = []  # Track (probs, unlabeled_indices) for each step
    
    estimations = []
    
    # Record estimates for seed phase (simple mean)
    for k in range(1, seed_size + 1):
        current_labels = y_true[selected_indices[:k]]
        est = np.mean(current_labels)
        estimations.append(est)
    
    if steps <= seed_size:
        return np.array(estimations), np.array(selected_indices)
    
    # Phase 2: Train surrogate
    X_train = embeddings[seed_indices]
    y_train = y_true[seed_indices]
    
    # Fallback if seed is pure
    if len(np.unique(y_train)) < 2:
        remaining = list(set(all_indices) - set(selected_indices))
        np.random.shuffle(remaining)
        for k in range(seed_size + 1, steps + 1):
            if remaining:
                new_idx = remaining.pop()
                selected_indices.append(new_idx)
            est = np.mean(y_true[selected_indices])
            estimations.append(est)
        return np.array(estimations), np.array(selected_indices)
    
    # Train classifier
    if surrogate_config.use_rf:
        clf = RandomForestClassifier(
            n_estimators=surrogate_config.rf_n_estimators, 
            max_depth=surrogate_config.rf_max_depth, 
            random_state=surrogate_config.rf_random_state, 
            class_weight=surrogate_config.rf_class_weight
        )
    else:
        clf = LogisticRegression(
            solver=surrogate_config.lr_solver, 
            class_weight=surrogate_config.lr_class_weight,
            max_iter=surrogate_config.lr_max_iter
        )
    clf.fit(X_train, y_train)
    
    # Phase 3: Incremental model-assisted sampling
    remaining_pool = list(set(all_indices) - set(selected_indices))
    
    for k in range(seed_size + 1, steps + 1):
        if not remaining_pool:
            break
        
        # Retrain classifier periodically (every 10 samples)
        if (k - seed_size) % 10 == 0 and len(selected_indices) >= seed_size:
            X_train = embeddings[selected_indices]
            y_train = y_true[selected_indices]
            if len(np.unique(y_train)) >= 2:
                clf.fit(X_train, y_train)
        
        # Compute acquisition probabilities (matching SurrogateProbAcquisition)
        X_pool = embeddings[remaining_pool]
        probs = clf.predict_proba(X_pool)
        raw_probs = probs[:, acquisition_config.target_class]
        
        # Smoothing: mix with uniform distribution
        smoothing = acquisition_config.smoothing_factor
        n_pool = len(remaining_pool)
        q = (1 - smoothing) * raw_probs + smoothing * (1.0 / n_pool)
        q = q / q.sum()
        
        # Store acquisition probs for importance weighting
        acquisition_probs_history.append((q.copy(), list(remaining_pool)))
        
        # Stochastic selection
        local_idx = np.random.choice(n_pool, p=q)
        global_idx = remaining_pool[local_idx]
        
        # Add to selected
        selected_indices.append(global_idx)
        remaining_pool.remove(global_idx)
        
        # Compute IS estimate with importance weights
        weights = [1.0] * seed_size  # Seed samples have weight 1.0
        
        for step_idx, (acq_probs, unlabeled_indices) in enumerate(acquisition_probs_history):
            sampled_global_idx = selected_indices[seed_size + step_idx]
            try:
                local_idx_hist = unlabeled_indices.index(sampled_global_idx)
                prob_q = acq_probs[local_idx_hist]
            except ValueError:
                prob_q = acq_probs.max()
            
            n_pool_hist = len(unlabeled_indices)
            w = 1.0 / (n_pool_hist * prob_q)
            weights.append(w)
        
        weights = np.array(weights)
        
        # Self-normalized importance sampling
        if estimator_config.normalize_weights:
            weights = weights / weights.mean()
        
        # Weighted estimate
        current_labels = y_true[selected_indices]
        est = np.sum(weights * current_labels) / np.sum(weights)
        estimations.append(est)
    
    return np.array(estimations), np.array(selected_indices)


def run_incremental_lure_evaluation(y_true: np.ndarray, embeddings: np.ndarray,
                                    steps: int, seed_size: int = 8,
                                    surrogate_config: SurrogateConfig = None,
                                    acquisition_config: AcquisitionConfig = None,
                                    estimator_config: LUREEstimatorConfig = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Incremental LURE evaluation (matching LUREEstimator).
    
    Uses the full LURE formula from the paper for importance weighting.
    
    Args:
        y_true: True labels (n_samples,)
        embeddings: Feature embeddings (n_samples, n_features)
        steps: Total budget
        seed_size: Number of initial uniform samples
        surrogate_config: Configuration for surrogate model
        acquisition_config: Configuration for acquisition function
        estimator_config: Configuration for LURE estimator
        
    Returns:
        estimations: Array of estimates at each step
        selected_indices: Array of selected sample indices
    """
    # Use default configs if not provided
    if surrogate_config is None:
        surrogate_config = SurrogateConfig()
    if acquisition_config is None:
        acquisition_config = AcquisitionConfig()
    if estimator_config is None:
        estimator_config = LUREEstimatorConfig()
    
    n_total = len(y_true)
    N = n_total  # Total pool size
    all_indices = np.arange(n_total)
    
    # Phase 1: Random seed
    seed_size = min(seed_size, steps)
    seed_indices = list(np.random.choice(all_indices, size=seed_size, replace=False))
    
    # Initialize tracking
    selected_indices = list(seed_indices)
    acquisition_probs_history = []
    
    estimations = []
    
    # Record estimates for seed phase (simple mean)
    for k in range(1, seed_size + 1):
        current_labels = y_true[selected_indices[:k]]
        est = np.mean(current_labels)
        estimations.append(est)
    
    if steps <= seed_size:
        return np.array(estimations), np.array(selected_indices)
    
    # Phase 2: Train surrogate
    X_train = embeddings[seed_indices]
    y_train = y_true[seed_indices]
    
    # Fallback if seed is pure
    if len(np.unique(y_train)) < 2:
        remaining = list(set(all_indices) - set(selected_indices))
        np.random.shuffle(remaining)
        for k in range(seed_size + 1, steps + 1):
            if remaining:
                new_idx = remaining.pop()
                selected_indices.append(new_idx)
            est = np.mean(y_true[selected_indices])
            estimations.append(est)
        return np.array(estimations), np.array(selected_indices)
    
    # Train classifier
    if surrogate_config.use_rf:
        clf = RandomForestClassifier(
            n_estimators=surrogate_config.rf_n_estimators, 
            max_depth=surrogate_config.rf_max_depth, 
            random_state=surrogate_config.rf_random_state, 
            class_weight=surrogate_config.rf_class_weight
        )
    else:
        clf = LogisticRegression(
            solver=surrogate_config.lr_solver, 
            class_weight=surrogate_config.lr_class_weight,
            max_iter=surrogate_config.lr_max_iter
        )
    clf.fit(X_train, y_train)
    
    # Phase 3: Incremental LURE sampling
    remaining_pool = list(set(all_indices) - set(selected_indices))
    
    for k in range(seed_size + 1, steps + 1):
        if not remaining_pool:
            break
        
        M = k  # Current total samples
        
        # Retrain classifier periodically
        if (k - seed_size) % 10 == 0 and len(selected_indices) >= seed_size:
            X_train = embeddings[selected_indices]
            y_train = y_true[selected_indices]
            if len(np.unique(y_train)) >= 2:
                clf.fit(X_train, y_train)
        
        # Compute acquisition probabilities
        X_pool = embeddings[remaining_pool]
        probs = clf.predict_proba(X_pool)
        raw_probs = probs[:, acquisition_config.target_class]
        
        smoothing = acquisition_config.smoothing_factor
        n_pool = len(remaining_pool)
        q = (1 - smoothing) * raw_probs + smoothing * (1.0 / n_pool)
        q = q / q.sum()
        
        acquisition_probs_history.append((q.copy(), list(remaining_pool)))
        
        # Stochastic selection
        local_idx = np.random.choice(n_pool, p=q)
        global_idx = remaining_pool[local_idx]
        
        selected_indices.append(global_idx)
        remaining_pool.remove(global_idx)
        
        # Compute LURE weights (full formula from paper)
        weights = []
        
        # For initial uniform samples
        for m in range(1, seed_size + 1):
            q_i_m = 1.0 / (N - m + 1)
            v_m = 1 + (N - M) / (N - m) * (1 / ((N - m + 1) * q_i_m) - 1)
            weights.append(v_m)
        
        # For actively sampled points
        for step_idx, (acq_probs, unlabeled_indices) in enumerate(acquisition_probs_history):
            m = seed_size + step_idx + 1
            sampled_global_idx = selected_indices[m - 1]
            try:
                local_idx_hist = unlabeled_indices.index(sampled_global_idx)
                q_i_m = acq_probs[local_idx_hist]
            except ValueError:
                q_i_m = acq_probs.max()
            
            v_m = 1 + (N - M) / (N - m) * (1 / ((N - m + 1) * q_i_m) - 1)
            weights.append(v_m)
        
        weights = np.array(weights)
        
        # LURE estimate: R_LURE = (1/M) * Σ v_m * y_m
        current_labels = y_true[selected_indices]
        est = np.sum(weights * current_labels) / M
        estimations.append(est)
    
    return np.array(estimations), np.array(selected_indices)


def run_lr_is_evaluation(y_true: np.ndarray, embeddings: np.ndarray, 
                         steps: int, seed_size: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """LR + Importance Sampling (incremental)."""
    surrogate_config = SurrogateConfig(use_rf=False)
    return run_incremental_is_evaluation(y_true, embeddings, steps, seed_size, 
                                         surrogate_config=surrogate_config)


def run_rf_is_evaluation(y_true: np.ndarray, embeddings: np.ndarray,
                         steps: int, seed_size: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """RF + Importance Sampling (incremental)."""
    surrogate_config = SurrogateConfig(use_rf=True)
    return run_incremental_is_evaluation(y_true, embeddings, steps, seed_size,
                                         surrogate_config=surrogate_config)


def run_lr_lure_evaluation(y_true: np.ndarray, embeddings: np.ndarray,
                           steps: int, seed_size: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """LR + LURE (incremental)."""
    surrogate_config = SurrogateConfig(use_rf=False)
    return run_incremental_lure_evaluation(y_true, embeddings, steps, seed_size,
                                           surrogate_config=surrogate_config)


def run_rf_lure_evaluation(y_true: np.ndarray, embeddings: np.ndarray,
                           steps: int, seed_size: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """RF + LURE (incremental)."""
    surrogate_config = SurrogateConfig(use_rf=True)
    return run_incremental_lure_evaluation(y_true, embeddings, steps, seed_size,
                                           surrogate_config=surrogate_config)

