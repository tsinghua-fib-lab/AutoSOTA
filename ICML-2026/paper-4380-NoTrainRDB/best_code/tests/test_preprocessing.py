import unittest
import sys
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

# Mock autogluon if not present to allow importing rdblearn.preprocessing
try:
    import autogluon.features.generators
except ImportError:
    mock_ag = MagicMock()
    sys.modules["autogluon"] = mock_ag
    sys.modules["autogluon.features"] = mock_ag
    sys.modules["autogluon.features.generators"] = mock_ag

from rdblearn.preprocessing import (
    TemporalDiffTransformer, 
    TypeCastTransformer, 
    SafeLabelEncoderTransformer, 
    TabularPreprocessor,
    AutoGluonTransformer
)
from rdblearn.config import TemporalDiffConfig

from loguru import logger
logger.enable("rdblearn")


class TestTypeCastTransformer(unittest.TestCase):
    def test_transform_converts_types(self):
        transformer = TypeCastTransformer()
        df = pd.DataFrame({
            'bool_col': [True, False, None],
            'int64_col': pd.Series([1, 2, None], dtype="Int64"),
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c']
        })
        
        # Ensure input types are as expected for the test
        df['bool_col'] = df['bool_col'].astype('boolean')
        
        res = transformer.fit(df).transform(df)
        
        # Check bool -> float32
        self.assertEqual(res['bool_col'].dtype, np.float32)
        np.testing.assert_array_equal(res['bool_col'].values[:2], [1.0, 0.0])
        self.assertTrue(np.isnan(res['bool_col'].values[2]))

        # Check Int64 -> float32
        self.assertEqual(res['int64_col'].dtype, np.float32)
        np.testing.assert_array_equal(res['int64_col'].values[:2], [1.0, 2.0])
        self.assertTrue(np.isnan(res['int64_col'].values[2]))

        # Check others untouched
        self.assertEqual(res['float_col'].dtype, np.float64)
        self.assertEqual(res['str_col'].dtype, object)


class TestTemporalDiffTransformer(unittest.TestCase):
    """Tests for TemporalDiffTransformer class."""

    def test_fit_detects_epochtime_columns(self):
        """Test that fit() correctly identifies epochtime columns with max/min/median/mean suffixes."""
        config = TemporalDiffConfig(enabled=True)
        transformer = TemporalDiffTransformer(config)

        df = pd.DataFrame({
            'feature_epochtime_max': [1000, 2000],
            'feature_epochtime_min': [500, 1500],
            'feature_epochtime_std': [100, 200],
            'regular_col': [1, 2]
        })

        transformer.fit(df)

        # Only max and min should be in timestamp_columns_ (for transformation)
        self.assertEqual(len(transformer.timestamp_columns_), 2)
        self.assertIn('feature_epochtime_max', transformer.timestamp_columns_)
        self.assertIn('feature_epochtime_min', transformer.timestamp_columns_)
        # std should be in columns_to_drop_
        self.assertIn('feature_epochtime_std', transformer.columns_to_drop_)
        # regular_col should be in neither
        self.assertNotIn('regular_col', transformer.timestamp_columns_)
        self.assertNotIn('regular_col', transformer.columns_to_drop_)

    def test_fit_detects_all_suffix_variants(self):
        """Test that fit() picks up max, min, median, and mean suffixes and drops std."""
        config = TemporalDiffConfig(enabled=True)
        transformer = TemporalDiffTransformer(config)

        df = pd.DataFrame({
            'ts_epochtime_max': [1000],
            'ts_epochtime_min': [500],
            'ts_epochtime_median': [750],
            'ts_epochtime_mean': [700],
            'ts_epochtime_std': [10],
        })

        transformer.fit(df)

        self.assertEqual(len(transformer.timestamp_columns_), 4)
        self.assertEqual(len(transformer.columns_to_drop_), 1)
        self.assertIn('ts_epochtime_std', transformer.columns_to_drop_)

    def test_transform_computes_time_diff(self):
        """Test that transform correctly computes time differences and drops originals."""
        config = TemporalDiffConfig(enabled=True)
        transformer = TemporalDiffTransformer(config, cutoff_time_col='cutoff_time')

        t0 = pd.Timestamp('2023-01-01')
        cutoff_0 = pd.Timestamp('2023-01-03') # diff = 2 days

        t2 = pd.Timestamp('2023-01-10')
        cutoff_1 = pd.Timestamp('2023-01-11') # diff = 1 day

        df = pd.DataFrame({
            'tx_epochtime_max': [t0.value, t2.value], # int64 nano
            'cutoff_time': [cutoff_0, cutoff_1]
        })

        transformer.fit(df)
        result = transformer.transform(df)

        # Original timestamp column should be DROPPED after transformation
        self.assertNotIn('tx_epochtime_max', result.columns)
        # New feature column should exist
        self.assertIn('tx_epochtime_max_diff', result.columns)

        # Check computed values
        expected_diff_0 = float(cutoff_0.value - t0.value)
        expected_diff_1 = float(cutoff_1.value - t2.value)

        np.testing.assert_array_almost_equal(
            result['tx_epochtime_max_diff'].values, [expected_diff_0, expected_diff_1]
        )


    def test_transform_missing_cutoff_col(self):
        """Test that transform drops non-matching epochtime cols even without cutoff."""
        config = TemporalDiffConfig(enabled=True)
        transformer = TemporalDiffTransformer(config, cutoff_time_col='missing_col')
        df = pd.DataFrame({
            'tx_epochtime_max': [1000],
            'tx_epochtime_std': [100],
            'regular_col': [42],
        })
        transformer.fit(df)
        res = transformer.transform(df)
        # Non-matching epochtime col should still be dropped
        self.assertNotIn('tx_epochtime_std', res.columns)
        # Matching epochtime col should remain (no diff computed since cutoff missing)
        # but still present because cutoff is missing so transform returns early
        self.assertIn('tx_epochtime_max', res.columns)
        self.assertIn('regular_col', res.columns)


class TestSafeLabelEncoderTransformer(unittest.TestCase):
    def test_encodes_and_handles_unseen(self):
        transformer = SafeLabelEncoderTransformer()
        df_train = pd.DataFrame({'cat': ['a', 'b', 'b']})
        df_test = pd.DataFrame({'cat': ['b', 'c', 'a']}) # 'c' is unseen
        
        transformer.fit(df_train)
        
        # Transform train
        res_train = transformer.transform(df_train)
        # a->0, b->1
        np.testing.assert_array_equal(res_train['cat'], [0, 1, 1])
        
        # Transform test
        res_test = transformer.transform(df_test)
        # b->1, a->0. 'c' should be assigned new code, likely 2.
        # Check that it runs without error and returns numeric
        self.assertTrue(np.issubdtype(res_test['cat'].dtype, np.number))
        self.assertEqual(len(res_test), 3)


class TestTabularPreprocessor(unittest.TestCase):
    @patch('rdblearn.preprocessing.AutoMLPipelineFeatureGenerator')
    def test_pipeline_composition(self, mock_ag_class):
        # Mock instance
        mock_ag_instance = MagicMock()
        mock_ag_class.return_value = mock_ag_instance
        # Mock fit/transform
        mock_ag_instance.transform.side_effect = lambda X: X
        
        config = TemporalDiffConfig(enabled=True)
        pp = TabularPreprocessor(
            temporal_diff_config=config, 
            cutoff_time='cutoff'
        )
        
        t0 = pd.Timestamp('2023-01-01')
        t1 = pd.Timestamp('2023-01-02')
        df = pd.DataFrame({
            'bool_col': [True, False],
            'tx_epochtime_max': [t0.value, t1.value],
            'cat_col': ['x', 'y'],
            'cutoff': pd.to_datetime(['2023-01-03', '2023-01-04'])
        })

        # Fit
        pp.fit(df)

        # Check pipeline steps
        step_names = [s[0] for s in pp.pipeline.steps]
        self.assertEqual(step_names, ['type_cast', 'temporal', 'label_encoder', 'autogluon'])

        # Transform
        res = pp.transform(df)

        # Verify transformations occurred (indirectly via result checks)
        # 1. Type cast: bool -> float
        self.assertEqual(res['bool_col'].dtype, np.float32)
        # 2. Temporal: diff created, original dropped
        self.assertIn('tx_epochtime_max_diff', res.columns)
        self.assertNotIn('tx_epochtime_max', res.columns)
        # 3. Label Encoder: cat -> number
        self.assertTrue(np.issubdtype(res['cat_col'].dtype, np.number))
        
        # Verify AG was called
        mock_ag_instance.fit.assert_called_once()
        mock_ag_instance.transform.assert_called_once()

if __name__ == '__main__':
    unittest.main()
