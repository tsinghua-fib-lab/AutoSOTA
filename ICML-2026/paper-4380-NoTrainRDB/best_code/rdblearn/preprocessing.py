from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from loguru import logger
from .config import TemporalDiffConfig


class TypeCastTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to cast boolean and nullable Int64 columns to float32.
    Handles <NA> values by converting them to np.nan.
    """
    def fit(self, X: pd.DataFrame, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        for col in X.columns:
            # Check for boolean types
            is_bool = pd.api.types.is_bool_dtype(X[col])
            # Check for nullable Int64 (often comes from mixed sources)
            is_nullable_int = str(X[col].dtype) == 'Int64'

            if is_bool or is_nullable_int:
                # astype('float32') handles mapping <NA>/None/False->0/True->1 correctly
                # and converts <NA> to IEEE 754 NaN
                X[col] = X[col].astype('float32')
        return X


class TemporalDiffTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, config: TemporalDiffConfig, cutoff_time_col: Optional[str] = None):
        self.config = config
        self.cutoff_time_col = cutoff_time_col
        self.timestamp_columns_: List[str] = []

    def _sanitize_column_name(self, col: str) -> str:
        sanitized = col.replace('(', '_').replace(')', '').replace('.', '_')
        while '__' in sanitized:
            sanitized = sanitized.replace('__', '_')
        sanitized = sanitized.strip('_')
        return sanitized

    def fit(self, X: pd.DataFrame, y=None):
        # Find all epochtime columns
        all_epochtime_cols = [
            col for col in X.columns
            if '_epochtime' in col and col not in self.config.exclude_columns
        ]
        
        # Drop epochtime columns containing 'std' — std of timestamps is meaningless.
        self.columns_to_drop_ = [
            col for col in all_epochtime_cols if 'std' in col.lower()
        ]

        # Transform all remaining epochtime columns
        self.timestamp_columns_ = [
            col for col in all_epochtime_cols if col not in self.columns_to_drop_
        ]

        if self.timestamp_columns_:
            logger.info(f"TemporalDiffTransformer: Found {len(self.timestamp_columns_)} timestamp columns for transformation.")
        if self.columns_to_drop_:
            logger.info(f"TemporalDiffTransformer: Will drop {len(self.columns_to_drop_)} epochtime columns containing 'std'.")
        
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        
        # Drop unwanted epochtime columns first
        if self.columns_to_drop_:
            cols_present = [col for col in self.columns_to_drop_ if col in X.columns]
            if cols_present:
                X = X.drop(columns=cols_present)
        
        if not self.timestamp_columns_:
            return X

        if self.cutoff_time_col is None or self.cutoff_time_col not in X.columns:
            return X

        cutoff_series = X[self.cutoff_time_col]
        cutoff_nano = (cutoff_series.astype('datetime64[ns]') - np.array(0).astype('datetime64[ns]')).astype('int64')

        for col in self.timestamp_columns_:
            if col not in X.columns:
                continue

            timestamp_nano = X[col].values
            time_diff = (cutoff_nano - timestamp_nano).astype('float64')

            sanitized_name = self._sanitize_column_name(col)
            feature_name = f"{sanitized_name}_diff"

            X[feature_name] = time_diff
            X = X.drop(columns=[col])

        logger.info(f"TemporalDiffTransformer: Generated {len(self.timestamp_columns_)} temporal difference features.")
        return X


class SafeLabelEncoderTransformer(BaseEstimator, TransformerMixin):
    """
    Wraps standard LabelEncoder logic but handles unseen labels dynamically by expanding classes,
    matching original implementation behavior.
    """
    def __init__(self):
        self.label_encoders_ = {}
        self.cat_columns_ = []

    def fit(self, X: pd.DataFrame, y=None):
        # Identify object/category columns
        self.cat_columns_ = X.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in self.cat_columns_:
            le = LabelEncoder()
            # Convert to string to handle mixed types/NaNs consistently
            le.fit(X[col].astype(str))
            self.label_encoders_[col] = le
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        for col in self.cat_columns_:
            if col in X.columns:
                le = self.label_encoders_[col]
                vals = X[col].astype(str)
                
                # Check for unseen labels and expand if necessary
                unseen_mask = ~vals.isin(le.classes_)
                if unseen_mask.any():
                    # Extend classes with new unseen labels
                    new_classes = vals[unseen_mask].unique()
                    le.classes_ = np.concatenate([le.classes_, new_classes])
                    # Re-sort if strictly necessary for LabelEncoder, but usually unnecessary for just transform mapping
                    le.classes_ = np.sort(le.classes_)
                
                X[col] = le.transform(vals)
        return X


class AutoGluonTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn wrapper for AutoGluon's AutoMLPipelineFeatureGenerator.
    """
    def __init__(self, ag_config: Optional[Dict[str, Any]] = None):
        self.ag_config = ag_config
        self.feature_generator_ = None
        
        self._default_ag_config = {
            "enable_datetime_features": True,
            "enable_raw_text_features": False,
            "enable_text_special_features": False,
            "enable_text_ngram_features": False,
        }
        if self.ag_config:
            self._default_ag_config.update(self.ag_config)

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_generator_ = AutoMLPipelineFeatureGenerator(**self._default_ag_config)
        # AutoGluon generator fits and needs the data
        self.feature_generator_.fit(X=X)
        return self

    def transform(self, X: pd.DataFrame):
        if self.feature_generator_ is None:
            raise RuntimeError("AutoGluonTransformer not fitted.")
        return self.feature_generator_.transform(X)


class TabularPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self, 
        ag_config: Optional[Dict[str, Any]] = None,
        temporal_diff_config: Optional[TemporalDiffConfig] = None,
        cutoff_time: Optional[str] = None
    ):
        self.ag_config = ag_config
        self.temporal_diff_config = temporal_diff_config
        self.cutoff_time = cutoff_time
        self.pipeline = None
        self.is_fitted_ = False

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the preprocessor pipeline on training data."""
        steps = []

        # 1. Type Casting (Must be first to fix legacy types)
        steps.append(('type_cast', TypeCastTransformer()))

        # 2. Temporal Features
        if self.temporal_diff_config and self.temporal_diff_config.enabled:
            # We pass the column name for cutoff time so the transformer can find it in X
            steps.append(('temporal', TemporalDiffTransformer(
                config=self.temporal_diff_config, 
                cutoff_time_col=self.cutoff_time
            )))
        
        # 3. Categorical Encoding
        steps.append(('label_encoder', SafeLabelEncoderTransformer()))

        # 4. AutoGluon Features
        steps.append(('autogluon', AutoGluonTransformer(ag_config=self.ag_config)))

        self.pipeline = Pipeline(steps)
        logger.debug(f"Preprocessor pipeline: {self.pipeline}")

        self.pipeline.fit(X, y)
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using the fitted pipeline."""
        if self.pipeline is None:
            raise RuntimeError("Preprocessor not fitted.")
        
        return self.pipeline.transform(X)