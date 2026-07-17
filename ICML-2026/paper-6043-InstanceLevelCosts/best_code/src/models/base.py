"""
Base model abstraction for cost-sensitive learning.

Defines the interface that all model implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Union, Optional, Dict, Any
import numpy as np


class BaseModel(ABC):
    """
    Abstract base class for all models.
    
    All models support:
    - Both classification (y_star) and regression (delta_signed) tasks
    - Optional sample weighting during training
    - Fit/predict interface similar to scikit-learn
    
    Subclasses must implement:
    - fit(): Train the model
    - predict(): Make predictions
    - predict_proba(): Get class probabilities (classification only)
    """
    
    def __init__(
        self,
        task: str = 'classification',
        **kwargs
    ):
        """
        Initialize base model.
        
        Args:
            task: Either 'classification' or 'regression'
            **kwargs: Model-specific hyperparameters
        """
        if task not in ['classification', 'regression']:
            raise ValueError(f"task must be 'classification' or 'regression', got '{task}'")
        
        self.task = task
        self.is_fitted_ = False
        self.kwargs = kwargs
    
    @abstractmethod
    def fit(
        self,
        X: Union[np.ndarray, list],
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> 'BaseModel':
        """
        Train the model.
        
        Args:
            X: Training features (format depends on model type)
            y: Training targets (binary labels for classification, signed delta for regression)
            sample_weight: Optional per-example weights (e.g., |delta|)
            
        Returns:
            self (fitted model)
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        X: Union[np.ndarray, list],
    ) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Test features (same format as fit)
            
        Returns:
            Predictions (binary labels for classification, continuous for regression)
        """
        pass
    
    def predict_proba(
        self,
        X: Union[np.ndarray, list],
    ) -> np.ndarray:
        """
        Get predicted class probabilities (classification only).
        
        Args:
            X: Test features
            
        Returns:
            Probabilities for each class, shape (N, n_classes)
            
        Raises:
            NotImplementedError: If called on regression model or not supported
        """
        if self.task != 'classification':
            raise NotImplementedError("predict_proba only available for classification tasks")
        raise NotImplementedError("predict_proba not implemented by this model")
    
    def _check_fitted(self):
        """Verify that model has been fitted."""
        if not self.is_fitted_:
            raise RuntimeError(f"{self.__class__.__name__} must be fitted before making predictions")
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns:
            Dictionary of model hyperparameters
        """
        return {
            'task': self.task,
            **self.kwargs
        }
    
    def __repr__(self) -> str:
        """String representation of the model."""
        params = ', '.join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{self.__class__.__name__}({params})"
