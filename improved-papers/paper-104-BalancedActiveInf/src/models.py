"""
Predictive Modeling Module

This module implements the two-stage modeling approach for active inference:
    1. Label Model: Predicts the response variable
    2. Error Model: Estimates prediction uncertainty

Both models use gradient boosting (XGBoost) for flexible nonlinear modeling.
"""

import numpy as np
from xgboost import XGBRegressor
from typing import Tuple, Dict, Any


class ActiveInferenceModels:
    """
    Two-stage model for active inference with uncertainty quantification.
    
    The framework consists of:
        - Label model: Learns the mapping from features to labels
        - Error model: Learns to predict prediction errors (uncertainty)
    
    Attributes:
        label_model: XGBoost regressor for label prediction
        error_model: XGBoost regressor for uncertainty estimation
        label_params: Hyperparameters for label model
        error_params: Hyperparameters for error model
    """
    
    def __init__(
        self,
        label_params: Dict[str, Any] = None,
        error_params: Dict[str, Any] = None
    ):
        """
        Initialize the active inference models.
        
        Args:
            label_params: Hyperparameters for label model (default: None)
            error_params: Hyperparameters for error model (default: None)
        """
        # Default hyperparameters for label model
        default_label_params = {
            'n_estimators': 1000,
            'learning_rate': 0.001,
            'max_depth': 7,
            'random_state': 0
        }
        
        # Default hyperparameters for error model
        default_error_params = {
            'n_estimators': 1000,
            'learning_rate': 0.001,
            'max_depth': 7,
            'random_state': 0
        }
        
        # Update with user-provided parameters
        self.label_params = default_label_params
        if label_params is not None:
            self.label_params.update(label_params)
        
        self.error_params = default_error_params
        if error_params is not None:
            self.error_params.update(error_params)
        
        # Override with optimized hyperparameters (iter-1: lr=0.1, n=300, depth=5)
        optimized_params = {
            'n_estimators': 300,
            'learning_rate': 0.1,
            'max_depth': 5,
        }
        self.label_params.update(optimized_params)
        self.error_params.update(optimized_params)
        
        # Initialize models
        self.label_model = XGBRegressor(**self.label_params)
        self.error_model = XGBRegressor(**self.error_params)
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> 'ActiveInferenceModels':
        """
        Train both label and error models.
        
        Training procedure:
            1. Train label model on (X_train, y_train)
            2. Compute prediction errors on training data
            3. Train error model to predict |errors|
        
        Args:
            X_train: Training features of shape (n_train, n_features)
            y_train: Training labels of shape (n_train,)
        
        Returns:
            self: Fitted model instance
        
        Example:
            >>> models = ActiveInferenceModels()
            >>> models.fit(X_train, y_train)
        """
        # Step 1: Train label model
        self.label_model.fit(X_train, y_train)
        
        # Step 2: Compute training errors for error model
        y_pred_train = self.label_model.predict(X_train)
        errors_train = y_pred_train - y_train
        
        # Step 3: Train error model to predict prediction errors
        self.error_model.fit(X_train, errors_train)
        
        return self
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions and uncertainty estimates.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
        
        Returns:
            predictions: Predicted labels of shape (n_samples,)
            uncertainty: Estimated absolute errors of shape (n_samples,)
        
        Example:
            >>> predictions, uncertainty = models.predict(X_test)
        """
        predictions = self.label_model.predict(X)
        error_predictions = self.error_model.predict(X)
        uncertainty = np.abs(error_predictions)
        
        return predictions, uncertainty
    
    def predict_label(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels only (without uncertainty).
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
        
        Returns:
            predictions: Predicted labels of shape (n_samples,)
        """
        return self.label_model.predict(X)
    
    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """
        Predict uncertainty only.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
        
        Returns:
            uncertainty: Estimated absolute errors of shape (n_samples,)
        """
        error_predictions = self.error_model.predict(X)
        return np.abs(error_predictions)


def train_predictive_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    label_params: Dict[str, Any] = None,
    error_params: Dict[str, Any] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function to train models and get test predictions.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        label_params: Hyperparameters for label model (default: None)
        error_params: Hyperparameters for error model (default: None)
    
    Returns:
        y_pred: Predictions on test set
        uncertainty: Uncertainty estimates on test set
        error_pred: Raw error predictions (before absolute value)
    
    Example:
        >>> y_pred, uncertainty, error_pred = train_predictive_models(
        ...     X_train, y_train, X_test
        ... )
    """
    models = ActiveInferenceModels(label_params, error_params)
    models.fit(X_train, y_train)
    
    y_pred = models.predict_label(X_test)
    error_pred = models.error_model.predict(X_test)
    uncertainty = np.abs(error_pred)
    
    return y_pred, uncertainty, error_pred
