"""
Tests for src/tasks/delta_regress.py delta regression task runner.
"""

import numpy as np
import pandas as pd
import pytest

from tasks.delta_regress import (
    RegressResult,
    run_regression,
    run_regression_sweep,
    results_to_dataframe,
    summarize_results,
)
from models.tfidf import TfidfModel


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def text_regression_data():
    """Create synthetic text regression data with deltas."""
    texts_train = [
        "This is positive and good",
        "Another positive example here",
        "This is negative and bad",
        "More negative content here",
        "Positive vibes only",
        "Bad negative terrible",
        "Good positive excellent",
        "Negative horrible awful",
    ]
    texts_test = [
        "Positive happy good content",
        "Negative sad bad content",
        "More positive things",
        "More negative things",
    ]

    # Binary labels (y_star)
    y_train = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    y_test = np.array([1, 0, 1, 0])

    # Signed deltas (positive = y_star=1, negative = y_star=0)
    delta_train = np.array([0.8, 0.6, -0.7, -0.5, 0.9, -0.8, 0.7, -0.6])
    delta_test = np.array([0.75, -0.65, 0.55, -0.45])

    return {
        'X_train': texts_train,
        'y_train': y_train,
        'X_test': texts_test,
        'y_test': y_test,
        'delta_train': delta_train,
        'delta_test': delta_test,
    }


@pytest.fixture
def text_data_with_val(text_regression_data):
    """Extend text data with validation set."""
    data = text_regression_data.copy()
    data['X_val'] = ["Validation positive text", "Validation negative text"]
    data['y_val'] = np.array([1, 0])
    data['delta_val'] = np.array([0.5, -0.5])
    return data


# =============================================================================
# RegressResult Tests
# =============================================================================

class TestRegressResult:
    """Tests for RegressResult dataclass."""

    def test_to_dict_basic(self):
        """Test conversion to dictionary."""
        result = RegressResult(
            model_name='TestModel',
            weighting='none',
            seed=42,
            train_metrics={'mae': 0.1, 'rmse': 0.15, 'sign_accuracy': 0.9},
            val_metrics=None,
            test_metrics={'mae': 0.2, 'rmse': 0.25, 'sign_accuracy': 0.8},
        )
        d = result.to_dict()

        assert d['model_name'] == 'TestModel'
        assert d['weighting'] == 'none'
        assert d['seed'] == 42
        assert d['train_mae'] == 0.1
        assert d['train_rmse'] == 0.15
        assert d['train_sign_accuracy'] == 0.9
        assert d['test_mae'] == 0.2
        assert d['test_rmse'] == 0.25
        assert d['test_sign_accuracy'] == 0.8
        assert 'val_mae' not in d

    def test_to_dict_with_val(self):
        """Test conversion with validation metrics."""
        result = RegressResult(
            model_name='TestModel',
            weighting='absdelta',
            seed=0,
            train_metrics={'mae': 0.1},
            val_metrics={'mae': 0.15},
            test_metrics={'mae': 0.2},
        )
        d = result.to_dict()

        assert d['val_mae'] == 0.15


# =============================================================================
# run_regression Tests
# =============================================================================

class TestRunRegression:
    """Tests for run_regression function."""

    def test_basic_unweighted(self, text_regression_data):
        """Test basic unweighted regression."""
        model = TfidfModel(task='regression')
        result = run_regression(
            model=model,
            weighting='none',
            seed=42,
            **text_regression_data,
        )

        assert result.model_name == 'TfidfModel'
        assert result.weighting == 'none'
        assert result.seed == 42
        assert 'mae' in result.train_metrics
        assert 'rmse' in result.train_metrics
        assert 'sign_accuracy' in result.train_metrics
        assert 'mae' in result.test_metrics
        assert result.val_metrics is None

    def test_absdelta_weighting(self, text_regression_data):
        """Test |delta| weighted regression."""
        model = TfidfModel(task='regression')
        result = run_regression(
            model=model,
            weighting='absdelta',
            seed=42,
            **text_regression_data,
        )

        assert result.weighting == 'absdelta'
        assert 'weighted_mae' in result.test_metrics
        assert 'weighted_rmse' in result.test_metrics
        assert 'weighted_sign_accuracy' in result.test_metrics

    def test_alpha_balanced_weighting(self, text_regression_data):
        """Test alpha-balanced weighted regression."""
        model = TfidfModel(task='regression')
        result = run_regression(
            model=model,
            weighting='alpha_balanced',
            seed=42,
            **text_regression_data,
        )

        assert result.weighting == 'alpha_balanced'
        assert 'weighted_sign_accuracy' in result.test_metrics

    def test_with_validation_set(self, text_data_with_val):
        """Test regression with validation set."""
        model = TfidfModel(task='regression')
        result = run_regression(
            model=model,
            weighting='none',
            seed=42,
            **text_data_with_val,
        )

        assert result.val_metrics is not None
        assert 'mae' in result.val_metrics
        assert 'sign_accuracy' in result.val_metrics

    def test_custom_model_name(self, text_regression_data):
        """Test custom model name."""
        model = TfidfModel(task='regression')
        result = run_regression(
            model=model,
            weighting='none',
            seed=42,
            model_name='MyRegressionModel',
            **text_regression_data,
        )

        assert result.model_name == 'MyRegressionModel'

    def test_config_stored(self, text_regression_data):
        """Test that config is stored in result."""
        model = TfidfModel(task='regression', ridge_alpha=2.0)
        result = run_regression(
            model=model,
            weighting='absdelta',
            seed=123,
            **text_regression_data,
        )

        assert result.config['weighting'] == 'absdelta'
        assert result.config['seed'] == 123
        assert result.config['n_train'] == len(text_regression_data['y_train'])
        assert result.config['n_test'] == len(text_regression_data['y_test'])
        assert 'model_params' in result.config

    def test_invalid_weighting_raises(self, text_regression_data):
        """Test that invalid weighting raises ValueError."""
        model = TfidfModel(task='regression')
        with pytest.raises(ValueError, match="Invalid weighting"):
            run_regression(
                model=model,
                weighting='invalid_strategy',
                seed=42,
                **text_regression_data,
            )

    def test_classification_model_raises(self, text_regression_data):
        """Test that classification model raises ValueError."""
        model = TfidfModel(task='classification')
        with pytest.raises(ValueError, match="Expected regression model"):
            run_regression(
                model=model,
                weighting='none',
                seed=42,
                **text_regression_data,
            )

    def test_reproducibility(self, text_regression_data):
        """Test that same seed gives same results."""
        model1 = TfidfModel(task='regression')
        result1 = run_regression(
            model=model1,
            weighting='none',
            seed=42,
            **text_regression_data,
        )

        model2 = TfidfModel(task='regression')
        result2 = run_regression(
            model=model2,
            weighting='none',
            seed=42,
            **text_regression_data,
        )

        assert result1.test_metrics['mae'] == result2.test_metrics['mae']
        assert result1.test_metrics['sign_accuracy'] == result2.test_metrics['sign_accuracy']

    def test_sign_accuracy_threshold_at_zero(self, text_regression_data):
        """Test that sign accuracy correctly thresholds predictions at 0."""
        model = TfidfModel(task='regression')
        result = run_regression(
            model=model,
            weighting='none',
            seed=42,
            **text_regression_data,
        )

        # Sign accuracy should be between 0 and 1
        assert 0.0 <= result.test_metrics['sign_accuracy'] <= 1.0

        # Manually verify: get predictions and check
        delta_pred = model.predict(text_regression_data['X_test'])
        y_pred_from_sign = (delta_pred >= 0).astype(int)
        y_test = text_regression_data['y_test']
        expected_sign_acc = float((y_pred_from_sign == y_test).mean())

        assert abs(result.test_metrics['sign_accuracy'] - expected_sign_acc) < 1e-6

    def test_sign_consistency_with_delta(self, text_regression_data):
        """Test that sign of delta matches y_star in test data."""
        # Verify our test data is consistent: sign(delta) should match y_star
        delta_test = text_regression_data['delta_test']
        y_test = text_regression_data['y_test']

        # y_star=1 should have positive delta, y_star=0 should have negative
        for delta, y in zip(delta_test, y_test):
            if y == 1:
                assert delta > 0, f"y_star=1 but delta={delta}"
            else:
                assert delta < 0, f"y_star=0 but delta={delta}"


# =============================================================================
# run_regression_sweep Tests
# =============================================================================

class TestRunRegressionSweep:
    """Tests for run_regression_sweep function."""

    def test_sweep_multiple_weightings(self, text_regression_data):
        """Test sweep across multiple weighting strategies."""
        def model_factory():
            return TfidfModel(task='regression')

        results = run_regression_sweep(
            model_factory=model_factory,
            weightings=['none', 'absdelta'],
            seeds=[0],
            **text_regression_data,
        )

        assert len(results) == 2
        assert results[0].weighting == 'none'
        assert results[1].weighting == 'absdelta'

    def test_sweep_multiple_seeds(self, text_regression_data):
        """Test sweep across multiple seeds."""
        def model_factory():
            return TfidfModel(task='regression')

        results = run_regression_sweep(
            model_factory=model_factory,
            weightings=['none'],
            seeds=[0, 1, 2],
            **text_regression_data,
        )

        assert len(results) == 3
        assert [r.seed for r in results] == [0, 1, 2]

    def test_sweep_full_grid(self, text_regression_data):
        """Test full grid sweep."""
        def model_factory():
            return TfidfModel(task='regression')

        results = run_regression_sweep(
            model_factory=model_factory,
            weightings=['none', 'absdelta', 'alpha_balanced'],
            seeds=[0, 1],
            **text_regression_data,
        )

        # 3 weightings x 2 seeds = 6 results
        assert len(results) == 6

    def test_sweep_all_results_have_regression_metrics(self, text_regression_data):
        """Test that all sweep results have expected regression metrics."""
        def model_factory():
            return TfidfModel(task='regression')

        results = run_regression_sweep(
            model_factory=model_factory,
            weightings=['none', 'absdelta', 'alpha_balanced'],
            seeds=[0],
            **text_regression_data,
        )

        for result in results:
            assert 'mae' in result.test_metrics
            assert 'rmse' in result.test_metrics
            assert 'sign_accuracy' in result.test_metrics


# =============================================================================
# results_to_dataframe Tests
# =============================================================================

class TestResultsToDataframe:
    """Tests for results_to_dataframe function."""

    def test_basic_conversion(self, text_regression_data):
        """Test basic conversion to DataFrame."""
        def model_factory():
            return TfidfModel(task='regression')

        results = run_regression_sweep(
            model_factory=model_factory,
            weightings=['none', 'absdelta'],
            seeds=[0],
            **text_regression_data,
        )

        df = results_to_dataframe(results)

        assert len(df) == 2
        assert 'model_name' in df.columns
        assert 'weighting' in df.columns
        assert 'seed' in df.columns
        assert 'test_mae' in df.columns
        assert 'test_rmse' in df.columns
        assert 'test_sign_accuracy' in df.columns

    def test_weighted_metrics_in_dataframe(self, text_regression_data):
        """Test that weighted metrics appear in DataFrame."""
        def model_factory():
            return TfidfModel(task='regression')

        results = run_regression_sweep(
            model_factory=model_factory,
            weightings=['absdelta'],
            seeds=[0],
            **text_regression_data,
        )

        df = results_to_dataframe(results)

        assert 'test_weighted_mae' in df.columns
        assert 'test_weighted_rmse' in df.columns
        assert 'test_weighted_sign_accuracy' in df.columns


# =============================================================================
# summarize_results Tests
# =============================================================================

class TestSummarizeResults:
    """Tests for summarize_results function."""

    def test_summarize_by_weighting(self, text_regression_data):
        """Test summarization by weighting strategy."""
        def model_factory():
            return TfidfModel(task='regression')

        results = run_regression_sweep(
            model_factory=model_factory,
            weightings=['none', 'absdelta'],
            seeds=[0, 1, 2],
            **text_regression_data,
        )

        summary = summarize_results(
            results,
            group_by=['weighting'],
            metrics=['test_mae', 'test_sign_accuracy'],
        )

        assert len(summary) == 2
        assert 'test_mae_mean' in summary.columns
        assert 'test_mae_std' in summary.columns
        assert 'test_mae_n' in summary.columns
        assert 'test_sign_accuracy_mean' in summary.columns
        assert (summary['test_mae_n'] == 3).all()

    def test_summarize_regression_specific_metrics(self, text_regression_data):
        """Test summarization with regression-specific metrics."""
        def model_factory():
            return TfidfModel(task='regression')

        results = run_regression_sweep(
            model_factory=model_factory,
            weightings=['absdelta'],
            seeds=[0, 1],
            **text_regression_data,
        )

        summary = summarize_results(
            results,
            group_by=['weighting'],
            metrics=['test_mae', 'test_rmse', 'test_weighted_mae', 'test_weighted_sign_accuracy'],
        )

        assert len(summary) == 1
        assert 'test_mae_mean' in summary.columns
        assert 'test_rmse_mean' in summary.columns
        assert 'test_weighted_mae_mean' in summary.columns
        assert 'test_weighted_sign_accuracy_mean' in summary.columns
