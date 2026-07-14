import numpy as np
import torch
from typing import Any

METRIC_ALIASES = {
    'svm': 'svm_accuracy',
    'svm_accuracy': 'svm_accuracy',
    'logistic': 'logistic_linear_accuracy',
    'logreg': 'logistic_linear_accuracy',
    'logistic_linear_accuracy': 'logistic_linear_accuracy',
    'kmeans': 'kmeans_accuracy',
    'kmeans_accuracy': 'kmeans_accuracy',
    'spectral': 'spectral_clustering_accuracy',
    'spectral_clustering_accuracy': 'spectral_clustering_accuracy',
    'knn': 'knn_accuracy',
    'knn_accuracy': 'knn_accuracy',
}

def metric_to_str(
    metric_key: str, 
    metric_val: Any, 
    *, 
    proportion_check: bool = True,
    round_digits: int = 6
) -> str:
    """
    Return a nicely formatted string for a metric value.

    Handles scalar floats, torch tensors, NumPy arrays and lists.  Vector
    metrics are printed as a comma-separated list in scientific notation.

    If *proportion_check* is True, scalar values in [0, 1] are formatted with
    four decimal places instead of scientific notation - useful for
    accuracy/F1 etc.
    """
    # Convert to NumPy for unified handling
    if torch.is_tensor(metric_val):
        metric_val = metric_val.detach().cpu().numpy()
    if isinstance(metric_val, list):
        metric_val = np.asarray(metric_val)

    # Vector metric -> list representation
    if hasattr(metric_val, "size") \
    and getattr(metric_val, "size") > 1:
        vals = metric_val.tolist()
        vals_str = ", ".join(f"{v:.{round_digits}e}" for v in vals)
        return f"{metric_key.upper()}: [{vals_str}]"

    # Scalar value
    try:
        scalar_val = float(metric_val)
    except Exception:
        scalar_val = metric_val  # fallback – best effort

    if proportion_check \
    and isinstance(scalar_val, float) \
    and 0.0 <= scalar_val <= 1.0:
        return f"{metric_key.upper()}: {scalar_val:.{round_digits}f}"
    else:
        return f"{metric_key.upper()}: {scalar_val:.{round_digits}e}" 
    
