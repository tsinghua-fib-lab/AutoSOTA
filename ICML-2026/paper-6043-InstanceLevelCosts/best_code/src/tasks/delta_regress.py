"""
Delta regression task runner for cost-sensitive learning experiments.

This module provides a unified interface for running regression experiments
that predict signed delta (Δ) values, with evaluation on both regression
quality and classification accuracy (via sign thresholding).

Supports:
- Multiple model types (TF-IDF, embeddings, tabular)
- Weighting strategies: unweighted, |delta|-weighted, alpha-balanced
- Train/val/test splits with reproducible seeding
- Comprehensive metrics: mae, rmse, sign_accuracy (weighted and unweighted)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd

from models.base import BaseModel
from core.metrics import regression_metrics
from core.weights import make_absdelta_weights, make_alpha_balanced_weights
from core.seed import set_seed


@dataclass
class RegressResult:
    """
    Results from a delta regression experiment.

    Attributes:
        model_name: Name/identifier of the model
        weighting: Weighting strategy used ('none', 'absdelta', 'alpha_balanced')
        seed: Random seed used
        train_metrics: Metrics computed on training set
        val_metrics: Metrics computed on validation set (if provided)
        test_metrics: Metrics computed on test set
        config: Full configuration dict for reproducibility
    """
    model_name: str
    weighting: str
    seed: int
    train_metrics: Dict[str, float]
    val_metrics: Optional[Dict[str, float]]
    test_metrics: Dict[str, float]
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary for easy DataFrame creation."""
        result = {
            'model_name': self.model_name,
            'weighting': self.weighting,
            'seed': self.seed,
        }

        # Add train metrics with prefix
        for k, v in self.train_metrics.items():
            result[f'train_{k}'] = v

        # Add val metrics with prefix (if present)
        if self.val_metrics:
            for k, v in self.val_metrics.items():
                result[f'val_{k}'] = v

        # Add test metrics with prefix
        for k, v in self.test_metrics.items():
            result[f'test_{k}'] = v

        return result


def run_regression(
    model: BaseModel,
    X_train: Union[np.ndarray, List[str], pd.DataFrame],
    y_train: np.ndarray,
    X_test: Union[np.ndarray, List[str], pd.DataFrame],
    y_test: np.ndarray,
    delta_train: np.ndarray,
    delta_test: np.ndarray,
    X_val: Optional[Union[np.ndarray, List[str], pd.DataFrame]] = None,
    y_val: Optional[np.ndarray] = None,
    delta_val: Optional[np.ndarray] = None,
    weighting: str = 'none',
    seed: int = 42,
    model_name: Optional[str] = None,
) -> RegressResult:
    """
    Run a single delta regression experiment.

    The model is trained to predict signed delta values. Evaluation includes
    both regression metrics (MAE, RMSE) and classification metrics derived
    from thresholding predictions at 0 (sign_accuracy).

    Args:
        model: A BaseModel instance (must have task='regression')
        X_train: Training features (texts, embeddings, or DataFrame)
        y_train: Training binary labels (for sign_accuracy evaluation)
        X_test: Test features
        y_test: Test binary labels
        delta_train: Signed delta values for training (target for regression)
        delta_test: Signed delta values for test (ground truth for MAE/RMSE)
        X_val: Optional validation features
        y_val: Optional validation binary labels
        delta_val: Optional signed delta values for validation
        weighting: Weighting strategy - 'none', 'absdelta', or 'alpha_balanced'
        seed: Random seed for reproducibility
        model_name: Optional name for the model (defaults to class name)

    Returns:
        RegressResult with metrics on train/val/test sets

    Raises:
        ValueError: If model task is not 'regression' or invalid weighting
    """
    if model.task != 'regression':
        raise ValueError(f"Expected regression model, got task='{model.task}'")

    if weighting not in ('none', 'absdelta', 'alpha_balanced'):
        raise ValueError(f"Invalid weighting: {weighting}. Must be 'none', 'absdelta', or 'alpha_balanced'")

    # Set seed for reproducibility
    set_seed(seed)

    # Compute training weights based on strategy
    if weighting == 'none':
        sample_weight = None
    elif weighting == 'absdelta':
        # Use |delta| as weights
        sample_weight = np.abs(delta_train).astype(np.float32)
        # Add small epsilon to avoid zero weights
        sample_weight = np.clip(sample_weight, 1e-6, None)
    elif weighting == 'alpha_balanced':
        # Create DataFrame for weight computation
        df_temp = pd.DataFrame({'delta_signed': delta_train})
        # normalize=True to stabilize sklearn solvers with extreme weight ranges
        sample_weight = make_alpha_balanced_weights(df_temp, delta_col='delta_signed', normalize=True)

    # Fit model on signed delta values
    model.fit(X_train, delta_train, sample_weight=sample_weight)

    # Compute predictions (predicted signed delta)
    delta_train_pred = model.predict(X_train)
    delta_test_pred = model.predict(X_test)

    # Compute metrics
    # For evaluation, use raw |delta| as weights (no epsilon, no normalization)
    abs_delta_train = np.abs(delta_train).astype(np.float32)
    abs_delta_test = np.abs(delta_test).astype(np.float32)

    train_metrics = regression_metrics(
        y_true=y_train,
        y_pred=delta_train_pred,
        delta_true=delta_train,
        weights=abs_delta_train,
    )
    test_metrics = regression_metrics(
        y_true=y_test,
        y_pred=delta_test_pred,
        delta_true=delta_test,
        weights=abs_delta_test,
    )

    # Validation metrics (if provided)
    val_metrics = None
    if X_val is not None and y_val is not None and delta_val is not None:
        delta_val_pred = model.predict(X_val)
        abs_delta_val = np.abs(delta_val).astype(np.float32)
        val_metrics = regression_metrics(
            y_true=y_val,
            y_pred=delta_val_pred,
            delta_true=delta_val,
            weights=abs_delta_val,
        )

    # Build result
    if model_name is None:
        model_name = model.__class__.__name__

    config = {
        'model_params': model.get_params(),
        'weighting': weighting,
        'seed': seed,
        'n_train': len(y_train),
        'n_test': len(y_test),
        'n_val': len(y_val) if y_val is not None else 0,
    }

    return RegressResult(
        model_name=model_name,
        weighting=weighting,
        seed=seed,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        config=config,
    )


def run_regression_sweep(
    model_factory,
    X_train: Union[np.ndarray, List[str], pd.DataFrame],
    y_train: np.ndarray,
    X_test: Union[np.ndarray, List[str], pd.DataFrame],
    y_test: np.ndarray,
    delta_train: np.ndarray,
    delta_test: np.ndarray,
    X_val: Optional[Union[np.ndarray, List[str], pd.DataFrame]] = None,
    y_val: Optional[np.ndarray] = None,
    delta_val: Optional[np.ndarray] = None,
    weightings: List[str] = ['none', 'absdelta', 'alpha_balanced'],
    seeds: List[int] = [0, 1, 2],
    model_name: Optional[str] = None,
) -> List[RegressResult]:
    """
    Run regression experiments across multiple seeds and weighting strategies.

    Args:
        model_factory: Callable that returns a fresh BaseModel instance
        X_train, y_train, X_test, y_test: Train/test data
        delta_train, delta_test: Signed delta values
        X_val, y_val, delta_val: Optional validation data
        weightings: List of weighting strategies to try
        seeds: List of random seeds to try
        model_name: Optional name for results

    Returns:
        List of RegressResult, one per (weighting, seed) combination
    """
    results = []

    for weighting in weightings:
        for seed in seeds:
            # Create fresh model instance
            model = model_factory()

            result = run_regression(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                delta_train=delta_train,
                delta_test=delta_test,
                X_val=X_val,
                y_val=y_val,
                delta_val=delta_val,
                weighting=weighting,
                seed=seed,
                model_name=model_name,
            )
            results.append(result)

    return results


def results_to_dataframe(results: List[RegressResult]) -> pd.DataFrame:
    """
    Convert list of RegressResult to a pandas DataFrame.

    Args:
        results: List of RegressResult objects

    Returns:
        DataFrame with one row per result, columns for all metrics
    """
    rows = [r.to_dict() for r in results]
    return pd.DataFrame(rows)


def summarize_results(
    results: List[RegressResult],
    group_by: List[str] = ['model_name', 'weighting'],
    metrics: List[str] = ['test_mae', 'test_rmse', 'test_sign_accuracy',
                          'test_weighted_mae', 'test_weighted_sign_accuracy'],
) -> pd.DataFrame:
    """
    Summarize results across seeds, computing mean and std for each metric.

    Args:
        results: List of RegressResult objects
        group_by: Columns to group by (default: model_name, weighting)
        metrics: Metrics to summarize

    Returns:
        DataFrame with mean and std for each metric, grouped as specified
    """
    df = results_to_dataframe(results)

    # Filter to requested metrics that exist
    available_metrics = [m for m in metrics if m in df.columns]

    # Group and aggregate
    agg_dict = {}
    for m in available_metrics:
        agg_dict[f'{m}_mean'] = (m, 'mean')
        agg_dict[f'{m}_std'] = (m, 'std')
        agg_dict[f'{m}_n'] = (m, 'count')

    summary = df.groupby(group_by).agg(**agg_dict).reset_index()
    return summary
