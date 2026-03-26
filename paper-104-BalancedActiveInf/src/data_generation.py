"""
Data Generation Module

This module provides functions for generating synthetic datasets used in
active sampling experiments. The primary function generates data following
the Friedman nonlinear regression model.

Reference:
    Friedman, J. H. (1991). Multivariate adaptive regression splines.
    The Annals of Statistics, 19(1), 1-67.
"""

import numpy as np
from typing import Tuple


def generate_friedman_data(
    n_samples: int,
    n_features: int,
    noise_std: float = 0.3,
    random_state: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data following the Friedman nonlinear regression model.
    
    The response variable is generated as:
        Y = 10 * sin(π * X₁ * X₂) + 20 * (X₃ - 0.5)² + 10 * X₄ + 5 * X₅ + ε
    
    where X₁, ..., X₅ ~ Uniform(0, 1), ε ~ N(0, noise_std²), and additional
    features X₆, ..., Xₚ are noise variables.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Total number of features (must be >= 5)
        noise_std: Standard deviation of Gaussian noise (default: 0.3)
        random_state: Random seed for reproducibility (default: None)
    
    Returns:
        X: Feature matrix of shape (n_samples, n_features)
        y: Response vector of shape (n_samples,)
    
    Raises:
        ValueError: If n_features < 5
    
    Example:
        >>> X, y = generate_friedman_data(n_samples=1000, n_features=10)
        >>> X.shape, y.shape
        ((1000, 10), (1000,))
    """
    if n_features < 5:
        raise ValueError("n_features must be at least 5 for Friedman function")
    
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate uniform random features in [0, 1]
    X = np.random.uniform(low=0.0, high=1.0, size=(n_samples, n_features))
    
    # Compute nonlinear response using first 5 features
    y = (
        10.0 * np.sin(np.pi * X[:, 0] * X[:, 1]) +
        20.0 * (X[:, 2] - 0.5) ** 2 +
        10.0 * X[:, 3] +
        5.0 * X[:, 4]
    )
    
    # Add Gaussian noise
    noise = np.random.normal(loc=0.0, scale=noise_std, size=n_samples)
    y += noise
    
    return X, y


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.5,
    random_state: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and test sets.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Response vector of shape (n_samples,)
        test_size: Proportion of data for test set (default: 0.5)
        random_state: Random seed for reproducibility (default: None)
    
    Returns:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
    
    Example:
        >>> X, y = generate_friedman_data(1000, 10)
        >>> X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.5)
    """
    from sklearn.model_selection import train_test_split
    
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )
