import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from rdblearn.estimator import RDBLearnClassifier, RDBLearnRegressor
from rdblearn.config import RDBLearnConfig, TemporalDiffConfig
from rdblearn.constants import RDBLEARN_DEFAULT_CONFIG, TARGET_HISTORY_TABLE_NAME
import fastdfs

from fastdfs import RDB, DFSConfig
from fastdfs.api import create_rdb

from loguru import logger
logger.enable("rdblearn")

class MockTabPFNClassifier:
    def __init__(self, seed=42):
        self.seed = seed
        self.classes_ = np.array([0, 1])

    def get_params(self, deep=True):
        return {"seed": self.seed}

    def set_params(self, seed=42, **params):
        self.seed = seed
        return self

    def fit(self, X, y, **kwargs):
        self.X_fit = X
        self.y_fit = y
        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)
        return self

    def predict(self, X, **kwargs):
        n = len(self.classes_)
        return np.random.choice(self.classes_, size=len(X))

    def predict_proba(self, X, **kwargs):
        n_classes = len(self.classes_)
        proba = np.random.rand(len(X), n_classes)
        proba /= proba.sum(axis=1, keepdims=True)
        return proba

class MockTabPFNRegressor:
    def __init__(self, seed=42):
        self.seed = seed

    def get_params(self, deep=True):
        return {"seed": self.seed}

    def set_params(self, seed=42, **params):
        self.seed = seed
        return self

    def fit(self, X, y, **kwargs):
        self.X_fit = X
        self.y_fit = y
        return self

    def predict(self, X, **kwargs):
        return np.random.rand(len(X))

class TestRDBLearnEstimator(unittest.TestCase):
    def setUp(self):
        # Create synthetic RDB
        # Users table
        self.users_df = pd.DataFrame({
            'user_id': [str(i) for i in range(100)],
            'age': np.random.randint(18, 80, 100),
            'city': ['New York', 'London', 'Paris', 'Tokyo'] * 25
        })
        
        # Transactions table
        self.tx_df = pd.DataFrame({
            'tx_id': range(500),
            'user_id': [str(i) for i in np.random.randint(0, 100, 500)],
            'amount': np.random.rand(500) * 100,
            'timestamp': pd.date_range('2023-01-01', periods=500, freq='h')
        })
        
        self.rdb = create_rdb(
            name="test_db",
            tables={
                "users": self.users_df,
                "transactions": self.tx_df
            },
            primary_keys={
                "users": "user_id",
                "transactions": "tx_id"
            },
            foreign_keys=[
                ("transactions", "user_id", "users", "user_id")
            ],
            time_columns={
                "transactions": "timestamp"
            }
        )
        
        # Create target dataframe (X)
        # We want to predict something for users at a specific time
        self.X_train = pd.DataFrame({
            'user_id': [str(i) for i in range(100)],
            'cutoff_time': pd.date_range('2023-01-10', periods=100),
            'cat_col': ['a', 'b'] * 50 # Extra column in X
        })

        self.y_train_cls = pd.Series(np.random.randint(0, 2, 100), name='target')
        self.y_train_reg = pd.Series(np.random.rand(100), name='target')
        
        self.X_test = pd.DataFrame({
            'user_id': [str(i) for i in range(20)],
            'cutoff_time': pd.date_range('2023-02-01', periods=20),
            'cat_col': ['a', 'c'] * 10
        })

        self.key_mappings = {'user_id': 'users.user_id'}
        self.cutoff_time_column = 'cutoff_time'

    def test_classifier_fit_predict(self):
        base_model = MockTabPFNClassifier()
        clf = RDBLearnClassifier(base_estimator=base_model)
        
        clf.fit(
            self.X_train, 
            self.y_train_cls, 
            rdb=self.rdb, 
            key_mappings=self.key_mappings, 
            cutoff_time_column=self.cutoff_time_column
        )
        
        # Check if base model was fitted
        self.assertTrue(hasattr(base_model, 'X_fit'))
        self.assertEqual(len(base_model.X_fit), 100)
        # Check if features were generated (more columns than original X)
        # Original X has 3 columns (user_id, cutoff_time, cat_col)
        # DFS should add features from transactions
        self.assertGreater(base_model.X_fit.shape[1], 3)
        
        # Predict
        preds = clf.predict(self.X_test)
        self.assertEqual(len(preds), 20)
        
        # Predict proba
        proba = clf.predict_proba(self.X_test)
        self.assertEqual(len(proba), 20)
        self.assertEqual(proba.shape[1], 2)

    def test_regressor_fit_predict(self):
        base_model = MockTabPFNRegressor()
        reg = RDBLearnRegressor(base_estimator=base_model)
        
        reg.fit(
            self.X_train, 
            self.y_train_reg, 
            rdb=self.rdb, 
            key_mappings=self.key_mappings, 
            cutoff_time_column=self.cutoff_time_column
        )
        
        # Check if base model was fitted
        self.assertTrue(hasattr(base_model, 'X_fit'))
        self.assertGreater(base_model.X_fit.shape[1], 3)
        
        # Predict
        preds = reg.predict(self.X_test)
        self.assertEqual(len(preds), 20)

    def test_downsampling(self):
        base_model = MockTabPFNClassifier()
        config = RDBLearnConfig(max_train_samples=50)
        clf = RDBLearnClassifier(base_estimator=base_model, config=config)
        
        clf.fit(
            self.X_train, 
            self.y_train_cls, 
            rdb=self.rdb, 
            key_mappings=self.key_mappings, 
            cutoff_time_column=self.cutoff_time_column
        )
        
        # Check if base model was fitted with downsampled data
        self.assertEqual(len(base_model.X_fit), 50)
        self.assertEqual(len(base_model.y_fit), 50)

    def test_config_passing(self):
        base_model = MockTabPFNClassifier()
        config_dict = {
            "dfs": {"max_depth": 1},
            "max_train_samples": 80
        }
        clf = RDBLearnClassifier(base_estimator=base_model, config=config_dict)
        
        self.assertIsInstance(clf.config, RDBLearnConfig)
        self.assertEqual(clf.config.max_train_samples, 80)
        self.assertIsNotNone(clf.config.dfs)

    def test_config_override(self):
        # Test 1: Default config
        base_model = MockTabPFNClassifier()
        clf = RDBLearnClassifier(base_estimator=base_model)
        self.assertEqual(clf.config.max_train_samples, RDBLEARN_DEFAULT_CONFIG["max_train_samples"])
        self.assertEqual(clf.config.dfs.max_depth, RDBLEARN_DEFAULT_CONFIG["dfs"]["max_depth"])

        # Test 2: Partial override
        override_config = {"max_train_samples": 500}
        clf = RDBLearnClassifier(base_estimator=base_model, config=override_config)
        
        # Check overridden value
        self.assertEqual(clf.config.max_train_samples, 500)
        
        # Check preserved default value (nested)
        self.assertEqual(clf.config.dfs.max_depth, RDBLEARN_DEFAULT_CONFIG["dfs"]["max_depth"])


    def test_classifier_with_temporal_diff(self):
        """Test classifier with temporal diff feature enabled."""
        base_model = MockTabPFNClassifier()
        temporal_config = TemporalDiffConfig(enabled=True, exclude_columns=[])
        config = RDBLearnConfig(
            temporal_diff=temporal_config,
            dfs=DFSConfig(
                agg_primitives=["max", "min", "mean", "std", "count"],
                max_depth=2
            )
        )
        clf = RDBLearnClassifier(base_estimator=base_model, config=config)

        clf.fit(
            self.X_train,
            self.y_train_cls,
            rdb=self.rdb,
            key_mappings=self.key_mappings,
            cutoff_time_column=self.cutoff_time_column
        )

        self.assertIsNotNone(clf.preprocessor_.temporal_transformer)
        self.assertTrue(clf.preprocessor_.temporal_transformer.is_fitted_)

        # Verify that expected diff columns are present after fitting
        x_fit_cols = clf.base_estimator.X_fit.columns
        self.assertTrue(any(c.endswith('_diff') for c in x_fit_cols))
        self.assertFalse(any('timestamp_epochtime' in c and 'diff' not in c for c in x_fit_cols))

        preds = clf.predict(self.X_test)
        self.assertEqual(len(preds), 20)

    def test_fit_with_target_augmentation(self):
        # Enable target augmentation and ensure it works
        config = RDBLearnConfig(
            enable_target_augmentation=True,
            max_train_samples=200 # No downsampling for this test to keep it simple, or make X larger
        )
        
        clf = RDBLearnClassifier(
            base_estimator=MockTabPFNClassifier(),
            config=config
        )
        
        # Use a fresh RDB copy as fit transforms it
        rdb_copy = self.rdb
        
        clf.fit(
            X=self.X_train,
            y=self.y_train_cls,
            rdb=rdb_copy,
            key_mappings=self.key_mappings,
            cutoff_time_column=self.cutoff_time_column
        )
        
        # Verify 'target_history' table exists in the prepared rdb
        self.assertIn(TARGET_HISTORY_TABLE_NAME, clf.rdb_.table_names)
        
        # Verify the content of target_history
        history_table = clf.rdb_.get_table(TARGET_HISTORY_TABLE_NAME)
        # Should have original columns + target + time
        expected_cols = set(self.X_train.columns) | {self.y_train_cls.name or "target"}
        self.assertTrue(expected_cols.issubset(history_table.columns))
        
        # Verify relationships
        relationships = clf.rdb_.get_relationships()
        found_rel = False
        for rel in relationships:
            # Check if rel is tuple or object
            # Depending on fastdfs implem, it might return tuples (child_table, child_col, parent_table, parent_col)
            # or Relationship objects.
            c_name, p_name = None, None
            
            if isinstance(rel, tuple):
                 c_table, c_col, p_table, p_col = rel
                 c_name = c_table if isinstance(c_table, str) else c_table.name
                 p_name = p_table if isinstance(p_table, str) else p_table.name
            else:
                 c_name = rel.child_table.name
                 p_name = rel.parent_table.name

            if c_name == TARGET_HISTORY_TABLE_NAME and p_name == "users":
                found_rel = True
                break
        self.assertTrue(found_rel, f"Relationship between {TARGET_HISTORY_TABLE_NAME} and users not found")

    def test_target_augmentation_no_cutoff(self):
        # Should skip augmentation if cutoff_time is missing
        config = RDBLearnConfig(enable_target_augmentation=True)
        clf = RDBLearnClassifier(base_estimator=MockTabPFNClassifier(), config=config)
        
        rdb_copy = self.rdb
        
        # Call fit without cutoff_time_column
        clf.fit(
            X=self.X_train.drop(columns=['cutoff_time']),
            y=self.y_train_cls,
            rdb=rdb_copy,
            key_mappings=self.key_mappings,
            cutoff_time_column=None
        )
        
        # Verify 'target_history' table does NOT exist
        self.assertNotIn(TARGET_HISTORY_TABLE_NAME, clf.rdb_.table_names)

    def test_encode_categorical_as_float_config(self):
        config = RDBLearnConfig(encode_categorical_as_float=True)
        clf = RDBLearnClassifier(
            base_estimator=MockTabPFNClassifier(),
            config=config,
        )
        self.assertTrue(clf.config.encode_categorical_as_float)

    def test_encode_categorical_as_float_fit_predict(self):
        config = RDBLearnConfig(
            encode_categorical_as_float=True,
            enable_target_augmentation=False,
        )
        clf = RDBLearnClassifier(
            base_estimator=MockTabPFNClassifier(),
            config=config,
        )
        clf.fit(
            self.X_train,
            self.y_train_cls,
            rdb=self.rdb,
            key_mappings=self.key_mappings,
            cutoff_time_column=self.cutoff_time_column,
        )
        self.assertIsNotNone(clf.rdb_category_encoders_)
        self.assertIn("cat_col", clf.task_category_encoders_)
        preds = clf.predict(self.X_test)
        self.assertEqual(len(preds), 20)
        # Unseen category 'c' in test should not break predict
        proba = clf.predict_proba(self.X_test)
        self.assertEqual(proba.shape[0], 20)

    def test_encode_categorical_as_float_disabled(self):
        config = RDBLearnConfig(encode_categorical_as_float=False)
        clf = RDBLearnClassifier(
            base_estimator=MockTabPFNClassifier(),
            config=config,
        )
        clf.fit(
            self.X_train,
            self.y_train_cls,
            rdb=self.rdb,
            key_mappings=self.key_mappings,
            cutoff_time_column=self.cutoff_time_column,
        )
        self.assertIsNone(clf.rdb_category_encoders_)
        self.assertIsNone(clf.task_category_encoders_)

    def test_classifier_base10_hierarchical_when_c_gt_10(self):
        base_model = MockTabPFNClassifier()
        clf = RDBLearnClassifier(base_estimator=base_model)
        y_multi = pd.Series(np.random.randint(0, 12, 100), name="target")
        clf.fit(
            self.X_train,
            y_multi,
            rdb=self.rdb,
            key_mappings=self.key_mappings,
            cutoff_time_column=self.cutoff_time_column,
        )
        self.assertTrue(clf.base10_hierarchical_)
        self.assertEqual(clf.n_classes_, 12)
        self.assertEqual(clf.base10_decomposer_.D, 2)
        self.assertEqual(len(clf.digit_estimators_), 2)
        proba = clf.predict_proba(self.X_test)
        self.assertEqual(proba.shape, (20, 12))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)
        preds = clf.predict(self.X_test)
        self.assertEqual(len(preds), 20)

    def test_classifier_single_head_when_c_le_10(self):
        base_model = MockTabPFNClassifier()
        clf = RDBLearnClassifier(base_estimator=base_model)
        clf.fit(
            self.X_train,
            self.y_train_cls,
            rdb=self.rdb,
            key_mappings=self.key_mappings,
            cutoff_time_column=self.cutoff_time_column,
        )
        self.assertFalse(getattr(clf, "base10_hierarchical_", True))
        proba = clf.predict_proba(self.X_test)
        self.assertEqual(proba.shape[1], 2)


if __name__ == '__main__':
    unittest.main()
