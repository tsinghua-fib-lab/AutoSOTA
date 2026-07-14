from typing import Optional, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted

class LimiXWrapperClassifier(BaseEstimator, ClassifierMixin):
    """
    A scikit-learn compatible wrapper for the LimiX classifier.
    
    This wrapper handles the in-context learning nature of LimiX by storing 
    the training data during `fit` and passing it to `predict` along with the query data.
    """
    
    def __init__(self, predictor):
        """
        Initialize the wrapper.
        
        Args:
            predictor: An initialized LimiXPredictor instance.
                       The instance must implement a `predict` method with the signature:
                       `predict(x_train, y_train, x_test, task_type='Classification') -> np.ndarray`
        """
        self.predictor = predictor

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]):
        """
        Store the training data for in-context inference.
        """
        # LimiX expects numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        self.classes_ = np.unique(y)
        self.X_train_ = X
        self.y_train_ = y
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class labels for X.
        """
        check_is_fitted(self, ['X_train_', 'y_train_'])
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities for X.
        """
        check_is_fitted(self, ['X_train_', 'y_train_'])
        X = np.array(X)
        
        # Call the underlying predictor
        result = self.predictor.predict(
            self.X_train_, 
            self.y_train_, 
            X,
            task_type="Classification"
        )
            
        return np.asarray(result)


class LimiXWrapperRegressor(BaseEstimator, RegressorMixin):
    """
    A scikit-learn compatible wrapper for the LimiX regressor.
    """
    
    def __init__(self, predictor):
        """
        Initialize the wrapper.
        
        Args:
            predictor: An initialized LimiXPredictor instance.
        """
        self.predictor = predictor

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]):
        """
        Store the training data.
        """
        X = np.array(X)
        y = np.array(y)
        
        self.X_train_ = X
        self.y_train_ = y
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict target values for X.
        """
        check_is_fitted(self, ['X_train_', 'y_train_'])
        X = np.array(X)
        
        result = self.predictor.predict(
            self.X_train_, 
            self.y_train_, 
            X,
            task_type="Regression"
        )
        
        return np.asarray(result)
