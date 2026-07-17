"""
Tests for src/tasks/classify.py classification task runner.
"""

import numpy as np
import pandas as pd
import pytest

from tasks.classify import (
    ClassifyResult,
    run_classification,
    run_classification_sweep,
    results_to_dataframe,
    summarize_results,
)
from models.tfidf import TfidfModel


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def text_classification_data():
    """Create synthetic text classification data with deltas."""
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
def text_data_with_val(text_classification_data):
    """Extend text data with validation set."""
    data = text_classification_data.copy()
    data['X_val'] = ["Validation positive text", "Validation negative text"]
    data['y_val'] = np.array([1, 0])
    data['delta_val'] = np.array([0.5, -0.5])
    return data


# =============================================================================
# ClassifyResult Tests
# =============================================================================

class TestClassifyResult:
    """Tests for ClassifyResult dataclass."""

    def test_to_dict_basic(self):
        """Test conversion to dictionary."""
        result = ClassifyResult(
            model_name='TestModel',
            weighting='none',
            seed=42,
            train_metrics={'accuracy': 0.9, 'weighted_accuracy': 0.85},
            val_metrics=None,
            test_metrics={'accuracy': 0.8, 'weighted_accuracy': 0.75},
        )
        d = result.to_dict()

        assert d['model_name'] == 'TestModel'
        assert d['weighting'] == 'none'
        assert d['seed'] == 42
        assert d['train_accuracy'] == 0.9
        assert d['train_weighted_accuracy'] == 0.85
        assert d['test_accuracy'] == 0.8
        assert d['test_weighted_accuracy'] == 0.75
        assert 'val_accuracy' not in d

    def test_to_dict_with_val(self):
        """Test conversion with validation metrics."""
        result = ClassifyResult(
            model_name='TestModel',
            weighting='absdelta',
            seed=0,
            train_metrics={'accuracy': 0.9},
            val_metrics={'accuracy': 0.85},
            test_metrics={'accuracy': 0.8},
        )
        d = result.to_dict()

        assert d['val_accuracy'] == 0.85


# =============================================================================
# run_classification Tests
# =============================================================================

class TestRunClassification:
    """Tests for run_classification function."""

    def test_basic_unweighted(self, text_classification_data):
        """Test basic unweighted classification."""
        model = TfidfModel(task='classification')
        result = run_classification(
            model=model,
            weighting='none',
            seed=42,
            **text_classification_data,
        )

        assert result.model_name == 'TfidfModel'
        assert result.weighting == 'none'
        assert result.seed == 42
        assert 'accuracy' in result.train_metrics
        assert 'accuracy' in result.test_metrics
        assert result.val_metrics is None

    def test_absdelta_weighting(self, text_classification_data):
        """Test |delta| weighted classification."""
        model = TfidfModel(task='classification')
        result = run_classification(
            model=model,
            weighting='absdelta',
            seed=42,
            **text_classification_data,
        )

        assert result.weighting == 'absdelta'
        assert 'weighted_accuracy' in result.test_metrics
        assert 'expected_cost' in result.test_metrics

    def test_alpha_balanced_weighting(self, text_classification_data):
        """Test alpha-balanced weighted classification."""
        model = TfidfModel(task='classification')
        result = run_classification(
            model=model,
            weighting='alpha_balanced',
            seed=42,
            **text_classification_data,
        )

        assert result.weighting == 'alpha_balanced'
        assert 'weighted_accuracy' in result.test_metrics

    def test_with_validation_set(self, text_data_with_val):
        """Test classification with validation set."""
        model = TfidfModel(task='classification')
        result = run_classification(
            model=model,
            weighting='none',
            seed=42,
            **text_data_with_val,
        )

        assert result.val_metrics is not None
        assert 'accuracy' in result.val_metrics

    def test_custom_model_name(self, text_classification_data):
        """Test custom model name."""
        model = TfidfModel(task='classification')
        result = run_classification(
            model=model,
            weighting='none',
            seed=42,
            model_name='MyCustomModel',
            **text_classification_data,
        )

        assert result.model_name == 'MyCustomModel'

    def test_config_stored(self, text_classification_data):
        """Test that config is stored in result."""
        model = TfidfModel(task='classification', max_features=1000)
        result = run_classification(
            model=model,
            weighting='absdelta',
            seed=123,
            **text_classification_data,
        )

        assert result.config['weighting'] == 'absdelta'
        assert result.config['seed'] == 123
        assert result.config['n_train'] == len(text_classification_data['y_train'])
        assert result.config['n_test'] == len(text_classification_data['y_test'])
        assert 'model_params' in result.config

    def test_invalid_weighting_raises(self, text_classification_data):
        """Test that invalid weighting raises ValueError."""
        model = TfidfModel(task='classification')
        with pytest.raises(ValueError, match="Invalid weighting"):
            run_classification(
                model=model,
                weighting='invalid_strategy',
                seed=42,
                **text_classification_data,
            )

    def test_regression_model_raises(self, text_classification_data):
        """Test that regression model raises ValueError."""
        model = TfidfModel(task='regression')
        with pytest.raises(ValueError, match="Expected classification model"):
            run_classification(
                model=model,
                weighting='none',
                seed=42,
                **text_classification_data,
            )

    def test_reproducibility(self, text_classification_data):
        """Test that same seed gives same results."""
        model1 = TfidfModel(task='classification')
        result1 = run_classification(
            model=model1,
            weighting='none',
            seed=42,
            **text_classification_data,
        )

        model2 = TfidfModel(task='classification')
        result2 = run_classification(
            model=model2,
            weighting='none',
            seed=42,
            **text_classification_data,
        )

        assert result1.test_metrics['accuracy'] == result2.test_metrics['accuracy']


# =============================================================================
# run_classification_sweep Tests
# =============================================================================

class TestRunClassificationSweep:
    """Tests for run_classification_sweep function."""

    def test_sweep_multiple_weightings(self, text_classification_data):
        """Test sweep across multiple weighting strategies."""
        def model_factory():
            return TfidfModel(task='classification')

        results = run_classification_sweep(
            model_factory=model_factory,
            weightings=['none', 'absdelta'],
            seeds=[0],
            **text_classification_data,
        )

        assert len(results) == 2
        assert results[0].weighting == 'none'
        assert results[1].weighting == 'absdelta'

    def test_sweep_multiple_seeds(self, text_classification_data):
        """Test sweep across multiple seeds."""
        def model_factory():
            return TfidfModel(task='classification')

        results = run_classification_sweep(
            model_factory=model_factory,
            weightings=['none'],
            seeds=[0, 1, 2],
            **text_classification_data,
        )

        assert len(results) == 3
        assert [r.seed for r in results] == [0, 1, 2]

    def test_sweep_full_grid(self, text_classification_data):
        """Test full grid sweep."""
        def model_factory():
            return TfidfModel(task='classification')

        results = run_classification_sweep(
            model_factory=model_factory,
            weightings=['none', 'absdelta', 'alpha_balanced'],
            seeds=[0, 1],
            **text_classification_data,
        )

        # 3 weightings x 2 seeds = 6 results
        assert len(results) == 6


# =============================================================================
# results_to_dataframe Tests
# =============================================================================

class TestResultsToDataframe:
    """Tests for results_to_dataframe function."""

    def test_basic_conversion(self, text_classification_data):
        """Test basic conversion to DataFrame."""
        def model_factory():
            return TfidfModel(task='classification')

        results = run_classification_sweep(
            model_factory=model_factory,
            weightings=['none', 'absdelta'],
            seeds=[0],
            **text_classification_data,
        )

        df = results_to_dataframe(results)

        assert len(df) == 2
        assert 'model_name' in df.columns
        assert 'weighting' in df.columns
        assert 'seed' in df.columns
        assert 'test_accuracy' in df.columns


# =============================================================================
# summarize_results Tests
# =============================================================================

class TestSummarizeResults:
    """Tests for summarize_results function."""

    def test_summarize_by_weighting(self, text_classification_data):
        """Test summarization by weighting strategy."""
        def model_factory():
            return TfidfModel(task='classification')

        results = run_classification_sweep(
            model_factory=model_factory,
            weightings=['none', 'absdelta'],
            seeds=[0, 1, 2],
            **text_classification_data,
        )

        summary = summarize_results(
            results,
            group_by=['weighting'],
            metrics=['test_accuracy'],
        )

        assert len(summary) == 2
        assert 'test_accuracy_mean' in summary.columns
        assert 'test_accuracy_std' in summary.columns
        assert 'test_accuracy_n' in summary.columns
        assert (summary['test_accuracy_n'] == 3).all()
