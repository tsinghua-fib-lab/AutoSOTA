"""
Tabular model for classification and regression.

Uses HistGradientBoosting with sklearn ColumnTransformer for mixed
numeric/categorical features. Designed for NHANES and similar datasets.
"""

from typing import Union, Optional, List
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from models.base import BaseModel


class TabularModel(BaseModel):
    """
    HistGradientBoosting model for tabular data with mixed feature types.

    Handles numeric and categorical features separately with appropriate
    preprocessing (imputation, scaling, encoding).

    Example:
        >>> model = TabularModel(
        ...     task='classification',
        ...     num_features=['age', 'bmi', 'sbp'],
        ...     cat_features=['gender', 'race']
        ... )
        >>> model.fit(df_train, y_train, sample_weight=weights)
        >>> predictions = model.predict(df_test)
    """

    def __init__(
        self,
        task: str = 'classification',
        num_features: Optional[List[str]] = None,
        cat_features: Optional[List[str]] = None,
        # HistGradientBoosting params
        max_iter: int = 100,
        max_depth: Optional[int] = None,
        learning_rate: float = 0.1,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.0,
        random_state: int = 0,
        **kwargs
    ):
        """
        Initialize tabular model.

        Args:
            task: 'classification' or 'regression'
            num_features: List of numeric feature column names
            cat_features: List of categorical feature column names
            max_iter: Maximum number of boosting iterations
            max_depth: Maximum depth of trees (None = unlimited)
            learning_rate: Learning rate for boosting
            min_samples_leaf: Minimum samples per leaf
            l2_regularization: L2 regularization strength
            random_state: Random seed for reproducibility
        """
        super().__init__(task=task)

        self.num_features = num_features or []
        self.cat_features = cat_features or []
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.random_state = random_state

        # Will be initialized in fit()
        self.preprocessor_ = None
        self.model_ = None

    def _build_preprocessor(self) -> ColumnTransformer:
        """Build ColumnTransformer for mixed feature types."""
        transformers = []

        # Numeric pipeline: impute missing -> standardize
        if self.num_features:
            numeric_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', numeric_pipeline, self.num_features))

        # Categorical pipeline: impute missing -> one-hot encode
        if self.cat_features:
            categorical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            transformers.append(('cat', categorical_pipeline, self.cat_features))

        return ColumnTransformer(
            transformers=transformers,
            remainder='drop'  # Drop columns not in num_features or cat_features
        )

    def _build_model(self):
        """Build HistGradientBoosting model based on task."""
        if self.task == 'classification':
            return HistGradientBoostingClassifier(
                max_iter=self.max_iter,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                min_samples_leaf=self.min_samples_leaf,
                l2_regularization=self.l2_regularization,
                random_state=self.random_state,
            )
        else:  # regression
            return HistGradientBoostingRegressor(
                max_iter=self.max_iter,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                min_samples_leaf=self.min_samples_leaf,
                l2_regularization=self.l2_regularization,
                random_state=self.random_state,
            )

    def _prepare_input(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Convert input to DataFrame with expected columns."""
        if isinstance(X, np.ndarray):
            # Assume columns are in order: num_features + cat_features
            all_features = self.num_features + self.cat_features
            if X.shape[1] != len(all_features):
                raise ValueError(
                    f"Expected {len(all_features)} columns, got {X.shape[1]}. "
                    f"When passing numpy array, columns must match num_features + cat_features."
                )
            X = pd.DataFrame(X, columns=all_features)
        return X

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> 'TabularModel':
        """
        Fit tabular model.

        Args:
            X: Feature data (DataFrame with column names, or array matching
               num_features + cat_features order)
            y: Target labels (classification) or values (regression)
            sample_weight: Optional per-example weights

        Returns:
            self (fitted model)
        """
        X = self._prepare_input(X)
        y = np.asarray(y)

        # Build and fit preprocessor
        self.preprocessor_ = self._build_preprocessor()
        X_transformed = self.preprocessor_.fit_transform(X)

        # Build and fit model
        self.model_ = self._build_model()

        if sample_weight is not None:
            self.model_.fit(X_transformed, y, sample_weight=sample_weight)
        else:
            self.model_.fit(X_transformed, y)

        self.is_fitted_ = True
        return self

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """
        Make predictions on tabular data.

        Args:
            X: Feature data (same format as fit)

        Returns:
            Predictions (binary labels for classification, continuous for regression)
        """
        self._check_fitted()

        X = self._prepare_input(X)
        X_transformed = self.preprocessor_.transform(X)
        return self.model_.predict(X_transformed)

    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """
        Get predicted class probabilities (classification only).

        Args:
            X: Feature data

        Returns:
            Class probabilities, shape (N, n_classes)
        """
        if self.task != 'classification':
            raise NotImplementedError("predict_proba only available for classification")

        self._check_fitted()

        X = self._prepare_input(X)
        X_transformed = self.preprocessor_.transform(X)
        return self.model_.predict_proba(X_transformed)

    def get_params(self) -> dict:
        """Get model hyperparameters."""
        params = super().get_params()
        params.update({
            'num_features': self.num_features,
            'cat_features': self.cat_features,
            'max_iter': self.max_iter,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'min_samples_leaf': self.min_samples_leaf,
            'l2_regularization': self.l2_regularization,
            'random_state': self.random_state,
        })
        return params

    def get_feature_names(self) -> Optional[List[str]]:
        """
        Get transformed feature names (requires fitted model).

        Returns:
            List of feature names after preprocessing, or None if not fitted
        """
        if not self.is_fitted_:
            return None
        return self.preprocessor_.get_feature_names_out().tolist()

    def get_n_features(self) -> Optional[int]:
        """
        Get number of features after preprocessing.

        Returns:
            Number of transformed features, or None if not fitted
        """
        if not self.is_fitted_:
            return None
        return len(self.get_feature_names())
