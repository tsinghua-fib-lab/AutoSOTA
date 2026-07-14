from typing import Optional, Dict, Union, List, Any
import pandas as pd
import numpy as np
from loguru import logger
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.preprocessing import LabelEncoder
import fastdfs
from fastdfs import RDB, DFSConfig
from fastdfs.utils.type_utils import safe_convert_to_string
from fastdfs.transform import (
    RDBTransformWrapper, RDBTransformPipeline, HandleDummyTable, 
    FeaturizeDatetime, FillMissingPrimaryKey, 
    FilterColumn, CanonicalizeTypes, EncodeCategoryColumns,
)
from fastdfs.transform.encode_categorical import encode_series_with_label_encoder

from .config import RDBLearnConfig
from .preprocessing import TabularPreprocessor
from .constants import RDBLEARN_DEFAULT_CONFIG, TARGET_HISTORY_TABLE_NAME
from .base10_hierarchical import Base10Decomposer, reconstruct_class_proba_from_digit_probs

class RDBLearnEstimator(BaseEstimator):
    def __init__(
        self, 
        base_estimator, 
        config: Optional[Union[RDBLearnConfig, dict]] = None
    ):
        self.base_estimator = base_estimator

        if isinstance(config, RDBLearnConfig):
            self.config = config
        else:
            # Start with defaults
            config_dict = RDBLEARN_DEFAULT_CONFIG.copy()
            # Update with user provided dict if any
            if isinstance(config, dict):
                config_dict.update(config)
            
            self.config = RDBLearnConfig(**config_dict)
            
        self.rdb_ = None
        self.preprocessor_ = None
        self.key_mappings_ = None
        self.cutoff_time_column_ = None
        
        self.history_df_ = None
        self.target_history_fks_ = None
        self.train_cutoff_time_column_ = None
        self.rdb_category_encoders_ = None
        self.task_category_encoders_ = None

    def _ensure_keys_are_strings(self, X: pd.DataFrame, key_mappings: Dict[str, str]) -> None:
        """Modifies X in place, using safe_convert_to_string for consistency with RDB."""
        for col in key_mappings.keys():
            if col in X.columns:
                X[col] = safe_convert_to_string(X[col])

    def _task_categorical_columns(
        self,
        X: pd.DataFrame,
        key_mappings: Dict[str, str],
        cutoff_time_column: Optional[str],
    ) -> List[str]:
        exclude = set(key_mappings.keys())
        if cutoff_time_column is not None:
            exclude.add(cutoff_time_column)
        cols: List[str] = []
        for col in X.columns:
            if col in exclude:
                continue
            dtype = X[col].dtype
            if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
                cols.append(col)
        return cols

    def _fit_task_categorical_encoders(
        self,
        X: pd.DataFrame,
        key_mappings: Dict[str, str],
        cutoff_time_column: Optional[str],
    ) -> None:
        cols = self._task_categorical_columns(X, key_mappings, cutoff_time_column)
        self.task_category_encoders_ = {}
        for col in cols:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.task_category_encoders_[col] = le
            logger.debug(
                f"Task categorical encoder fitted for {col} ({len(le.classes_)} classes)"
            )

    def _apply_task_categorical_encoders(self, X: pd.DataFrame) -> pd.DataFrame:
        encoders = self.task_category_encoders_
        if not encoders:
            return X
        out = X.copy()
        for col, le in encoders.items():
            if col not in out.columns:
                continue
            out[col] = encode_series_with_label_encoder(out[col], le)
        return out

    def _downsample(
        self,
        data: pd.DataFrame,
        target_column: str,
        task_type: str,
        max_samples: int,
        stratified_sampling: bool = False,
        balanced_sampling: bool = False
    ) -> pd.DataFrame:
        """Downsample data to max_samples.
        
        Supports three modes:
        - default: keep one per class + random fill
        - stratified_sampling: equal samples from each class
        - balanced_sampling: keep ALL minority examples, fill remaining with majority
        """
        if len(data) <= max_samples:
            return data

        logger.info(f"Downsampling training set from {len(data)} to {max_samples} samples.")
        
        X = data.drop(columns=[target_column])
        y = data[target_column].values

        if task_type == "regression":
            idx = np.random.choice(len(X), max_samples, replace=False)
            return data.iloc[idx].reset_index(drop=True)

        # Classification: Balanced minority-all sampling
        if balanced_sampling:
            unique_labels, label_counts = np.unique(y, return_counts=True)
            # Find minority class
            minority_label = unique_labels[np.argmin(label_counts)]
            majority_label = unique_labels[np.argmax(label_counts)]
            
            minority_indices = np.where(y == minority_label)[0]
            majority_indices = np.where(y == majority_label)[0]
            
            # Keep all minority examples, but cap at half the budget
            n_minority = min(len(minority_indices), max_samples // 2)
            n_majority = max_samples - n_minority
            
            selected_minority = np.random.choice(minority_indices, n_minority, replace=False)
            selected_majority = np.random.choice(majority_indices, min(n_majority, len(majority_indices)), replace=False)
            
            selected_indices = np.concatenate([selected_minority, selected_majority])
            np.random.shuffle(selected_indices)
            logger.info(f"Balanced sampling: {n_minority} minority + {n_majority} majority = {max_samples} total")
            return data.iloc[selected_indices].reset_index(drop=True)
        
        if not stratified_sampling:
            unique_labels = np.unique(y)
            selected_indices = []
            for label in unique_labels:
                class_indices = np.where(y == label)[0]
                if len(class_indices) > 0:
                    selected_idx = np.random.choice(class_indices, 1)[0]
                    selected_indices.append(selected_idx)

            remaining_samples = max_samples - len(selected_indices)
            if remaining_samples > 0:
                mask = np.ones(len(X), dtype=bool)
                mask[selected_indices] = False
                eligible_indices = np.where(mask)[0]

                if len(eligible_indices) > 0:
                    additional_indices = np.random.choice(
                        eligible_indices,
                        min(remaining_samples, len(eligible_indices)),
                        replace=False
                    )
                    selected_indices.extend(additional_indices)

            np.random.shuffle(selected_indices)
            idx = np.array(selected_indices)
            return data.iloc[idx].reset_index(drop=True)

        else:
            unique_labels, label_counts = np.unique(y, return_counts=True)
            n_classes = len(unique_labels)
            samples_per_class = max(1, max_samples // n_classes)

            balanced_indices = []
            remaining_indices = []

            for label in unique_labels:
                class_indices = np.where(y == label)[0]
                if len(class_indices) == 0:
                    continue

                if len(class_indices) <= samples_per_class:
                    balanced_indices.extend(class_indices)
                else:
                    sampled_indices = np.random.choice(class_indices, samples_per_class, replace=False)
                    balanced_indices.extend(sampled_indices)
                    mask = np.ones(len(class_indices), dtype=bool)
                    mask[np.isin(class_indices, sampled_indices)] = False
                    remaining_indices.extend(class_indices[mask])

            samples_needed = max_samples - len(balanced_indices)
            if samples_needed > 0 and len(remaining_indices) > 0:
                additional_samples = np.random.choice(
                    remaining_indices,
                    min(samples_needed, len(remaining_indices)),
                    replace=False
                )
                balanced_indices.extend(additional_samples)

            np.random.shuffle(balanced_indices)
            balanced_indices = balanced_indices[:max_samples]
            idx = np.array(balanced_indices)
            return data.iloc[idx].reset_index(drop=True)

    def _prepare_rdb(self, rdb: RDB) -> RDB:
        # Augment with target history if enabled and available
        if (
            self.config.enable_target_augmentation 
            and self.history_df_ is not None 
            and self.target_history_fks_ is not None
            and self.train_cutoff_time_column_ is not None
        ):
            logger.info(f"Augmenting RDB with {TARGET_HISTORY_TABLE_NAME} table.")
                
            rdb = rdb.add_table(
                dataframe=self.history_df_,
                name=TARGET_HISTORY_TABLE_NAME,
                time_column=self.train_cutoff_time_column_,
                foreign_keys=self.target_history_fks_
            )
            rdb = rdb.canonicalize_key_types()
            rdb.validate_key_consistency()

        logger.info("Preparing RDB with transformation pipeline.")
        steps = [
            HandleDummyTable(),
            FillMissingPrimaryKey(),
            RDBTransformWrapper(FeaturizeDatetime(features=["epochtime"])),
            RDBTransformWrapper(FilterColumn(drop_dtypes=["text"])),
        ]
        encode_transform = None
        if self.config.encode_categorical_as_float:
            encode_transform = EncodeCategoryColumns(encoders=self.rdb_category_encoders_)
            steps.append(RDBTransformWrapper(encode_transform))
        steps.append(RDBTransformWrapper(CanonicalizeTypes()))
        rdb = RDBTransformPipeline(steps)(rdb)
        if encode_transform is not None and self.rdb_category_encoders_ is None:
            self.rdb_category_encoders_ = encode_transform.encoders
        return rdb

    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        rdb: RDB,
        key_mappings: Dict[str, str],
        cutoff_time_column: Optional[str] = None,
        **kwargs
    ):
        # 0. Copy and ensure keys are string
        X = X.copy()
        self._ensure_keys_are_strings(X, key_mappings)
        self.key_mappings_ = key_mappings
        self.cutoff_time_column_ = cutoff_time_column
        
        # 1. Setup Target History Augmentation (Using FULL X, y)
        if self.config.enable_target_augmentation:
            if cutoff_time_column is None:
                logger.debug("enable_target_augmentation is True but cutoff_time_column is None. Skipping augmentation to prevent leakage.")
            else:
                logger.info("Storing target history for augmentation.")
                
                # Create history dataframe (using current X which is the full train set)
                self.history_df_ = X.copy()
                target_col = y.name or "_RDBL_target"
                self.history_df_[target_col] = y.copy()
                
                self.train_cutoff_time_column_ = cutoff_time_column
                
                # Construct foreign keys for the history table
                self.target_history_fks_ = []
                for x_col, rdb_ref in key_mappings.items():
                    if "." in rdb_ref:
                        rdb_table, rdb_col = rdb_ref.split(".", 1)
                        # (this_table, this_col, other_table, other_col)
                        self.target_history_fks_.append((x_col, rdb_table, rdb_col))

        # 2. RDB Transformation (Augments RDB using stored history)
        self.rdb_ = self._prepare_rdb(rdb)

        # 3. Downsampling (Modifies X and y for training)
        if len(X) > self.config.max_train_samples:
            data = X
            target_col = y.name or "_RDBL_target"
            data[target_col] = y
            
            task_type = "regression" if isinstance(self, RegressorMixin) else "classification"
            
            downsampled_data = self._downsample(
                data, target_col, task_type, 
                self.config.max_train_samples, 
                self.config.stratified_sampling,
                self.config.balanced_sampling
            )
            X = downsampled_data.drop(columns=[target_col])
            y = downsampled_data[target_col]

        if self.config.encode_categorical_as_float:
            self._fit_task_categorical_encoders(X, key_mappings, cutoff_time_column)
            X = self._apply_task_categorical_encoders(X)

        # 4. Feature Augmentation
        logger.info("Computing DFS features...")
        dfs_config = self.config.dfs or DFSConfig()
        
        X_dfs = fastdfs.compute_dfs_features(
            self.rdb_,
            X,
            key_mappings=key_mappings,
            cutoff_time_column=cutoff_time_column,
            config=dfs_config
        )
        logger.debug(f"DFS features: {X_dfs.columns.tolist()}")

        # 5. Preprocessing
        logger.info("Preprocessing augmented features ...")
        self.preprocessor_ = TabularPreprocessor(
            ag_config=self.config.ag_config,
            temporal_diff_config=self.config.temporal_diff,
            cutoff_time=cutoff_time_column
        )
        X_transformed = self.preprocessor_.fit(X_dfs).transform(X_dfs)
        self.downstream_feature_columns_ = list(X_transformed.columns)

        return self._fit_model(X_transformed, y, **kwargs)

    def _fit_model(self, X_transformed: pd.DataFrame, y: pd.Series, **kwargs):
        """Train the base estimator on preprocessed features (regression default)."""
        logger.info("Fitting base estimator ...")
        self.base_estimator.fit(X_transformed, y, **kwargs)
        return self

    def _transform_for_prediction(self, X: pd.DataFrame, rdb: Optional[RDB]) -> pd.DataFrame:
        """DFS + preprocessor transform (same as predict path, without calling the base model)."""
        X = X.copy()
        if self.key_mappings_:
            self._ensure_keys_are_strings(X, self.key_mappings_)

        if rdb is None:
            selected_rdb = self.rdb_
        else:
            selected_rdb = self._prepare_rdb(rdb)

        if self.config.encode_categorical_as_float:
            X = self._apply_task_categorical_encoders(X)

        logger.info("Computing DFS features...")
        dfs_config = self.config.dfs or DFSConfig()
        X_dfs = fastdfs.compute_dfs_features(
            selected_rdb,
            X,
            key_mappings=self.key_mappings_,
            cutoff_time_column=self.cutoff_time_column_,
            config=dfs_config,
        )
        logger.info("Preprocessing augmented features ...")
        return self.preprocessor_.transform(X_dfs)

    def _predict_common(self, X: pd.DataFrame, rdb: Optional[RDB], method: str, **kwargs):
        # 0. Copy and ensure keys are string
        X = X.copy()
        if self.key_mappings_:
            self._ensure_keys_are_strings(X, self.key_mappings_)

        # 2. RDB Selection
        if rdb is None:
            selected_rdb = self.rdb_
        else:
            # Augment new RDB with stored training history!
            selected_rdb = self._prepare_rdb(rdb)

        if self.config.encode_categorical_as_float:
            X = self._apply_task_categorical_encoders(X)

        # 3. Feature Augmentation
        logger.info("Computing DFS features...")
        
        dfs_config = self.config.dfs or DFSConfig()
        
        X_dfs = fastdfs.compute_dfs_features(
            selected_rdb, 
            X, 
            key_mappings=self.key_mappings_, 
            cutoff_time_column=self.cutoff_time_column_, 
            config=dfs_config
        )


        # 4. Preprocessing
        logger.info("Preprocessing augmented features ...")
        X_transformed = self.preprocessor_.transform(X_dfs)
        
        # 5. Prediction
        logger.info("Making predictions ...")
        predict_func = getattr(self.base_estimator, method)
        
        if self.config.predict_batch_size and len(X_transformed) > self.config.predict_batch_size:
            results = []
            for i in range(0, len(X_transformed), self.config.predict_batch_size):
                batch = X_transformed.iloc[i:i+self.config.predict_batch_size]
                results.append(predict_func(batch, **kwargs))
             
            if isinstance(results[0], dict):
                # Aggregate dictionary results
                aggregated = {}
                for key in results[0].keys():
                    key_results = [r[key] for r in results]
                    if isinstance(key_results[0], np.ndarray):
                        aggregated[key] = np.concatenate(key_results)
                    elif isinstance(key_results[0], (pd.Series, pd.DataFrame)):
                        aggregated[key] = pd.concat(key_results, axis=0)
                    else:
                        print(f"Warning: Unexpected type of key_results: {type(key_results[0])} when aggregating results for key {key}, skipping this key")
                return aggregated
            elif isinstance(results[0], np.ndarray):
                return np.concatenate(results)
            elif isinstance(results[0], (pd.Series, pd.DataFrame)):
                return pd.concat(results, axis=0)
            else:
                return np.concatenate(results)
        else:
            return predict_func(X_transformed, **kwargs)

class RDBLearnClassifier(RDBLearnEstimator, ClassifierMixin):
    def _fit_model(self, X_transformed: pd.DataFrame, y: pd.Series, **kwargs):
        y_series = y if isinstance(y, pd.Series) else pd.Series(y, name="target")
        le = LabelEncoder()
        y_enc = le.fit_transform(y_series)
        C = int(len(le.classes_))

        self.label_encoder_ = le
        self.classes_ = np.asarray(le.classes_)
        self.n_classes_ = C

        self.base10_hierarchical_ = False
        self.base10_decomposer_ = None
        self.digit_estimators_ = None

        if C <= 10:
            logger.info(f"Fitting base estimator (C={C}, single head) ...")
            self.base_estimator.fit(X_transformed, y_series, **kwargs)
            return self

        self.base10_hierarchical_ = True

        decomposer = Base10Decomposer(C)
        self.base10_decomposer_ = decomposer
        train_digits = decomposer.digits_for_array(y_enc)
        digit_estimators: List[Any] = []
        for i in range(decomposer.D):
            est = clone(self.base_estimator)
            y_digit = pd.Series(
                train_digits[i], index=X_transformed.index, name=y_series.name
            )
            est.fit(X_transformed, y_digit, **kwargs)
            digit_estimators.append(est)
        self.digit_estimators_ = digit_estimators
        logger.info(
            f"Fitted base-10-hierarchical classifier (C={C}, D={decomposer.D} digit heads)."
        )
        return self

    def _batched_predict_proba_subestimator(
        self, estimator: Any, X_transformed: pd.DataFrame, **kwargs
    ) -> np.ndarray:
        fn = getattr(estimator, "predict_proba")
        bs = self.config.predict_batch_size
        if bs and len(X_transformed) > bs:
            parts = []
            for i in range(0, len(X_transformed), bs):
                batch = X_transformed.iloc[i : i + bs]
                parts.append(np.asarray(fn(batch, **kwargs)))
            return np.concatenate(parts, axis=0)
        return np.asarray(fn(X_transformed, **kwargs))

    def _predict_proba_base10_hierarchical(
        self, X_transformed: pd.DataFrame, **kwargs
    ) -> np.ndarray:
        digit_probs = [
            self._batched_predict_proba_subestimator(est, X_transformed, **kwargs)
            for est in self.digit_estimators_
        ]
        return reconstruct_class_proba_from_digit_probs(
            self.base10_decomposer_, digit_probs, self.n_classes_
        )

    def predict(self, X: pd.DataFrame, rdb: Optional[RDB] = None, **kwargs):
        if getattr(self, "base10_hierarchical_", False):
            proba = self.predict_proba(X, rdb, **kwargs)
            idx = np.argmax(proba, axis=1)
            return self.classes_.take(idx.astype(np.intp, copy=False))
        return self._predict_common(X, rdb, method="predict", **kwargs)

    def predict_proba(self, X: pd.DataFrame, rdb: Optional[RDB] = None, **kwargs):
        if getattr(self, "base10_hierarchical_", False):
            from sklearn.utils.validation import check_is_fitted

            check_is_fitted(
                self, attributes=["digit_estimators_", "base10_decomposer_"]
            )
            X_transformed = self._transform_for_prediction(X, rdb)
            logger.info("Making predictions (base-10-hierarchical) ...")
            return self._predict_proba_base10_hierarchical(X_transformed, **kwargs)
        return self._predict_common(X, rdb, method="predict_proba", **kwargs)

class RDBLearnRegressor(RDBLearnEstimator, RegressorMixin):
    def predict(self, X: pd.DataFrame, rdb: Optional[RDB] = None, **kwargs):
        return self._predict_common(X, rdb, method="predict", **kwargs)
