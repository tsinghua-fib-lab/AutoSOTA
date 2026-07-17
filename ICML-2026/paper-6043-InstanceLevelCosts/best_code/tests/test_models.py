"""
Tests for src/models/ model classes.

Validates that all models follow the BaseModel interface:
- fit(X, y, sample_weight=None) -> self
- predict(X) -> np.ndarray
- predict_proba(X) -> np.ndarray (classification only)
- get_params() -> dict
"""

import numpy as np
import pandas as pd
import pytest

from models.base import BaseModel
from models.tfidf import TfidfModel
from models.tabular import TabularModel

# Optional imports (may require GPU or additional dependencies)
try:
    from models.text_embed import TextEmbedModel
    TEXT_EMBED_AVAILABLE = True
except ImportError:
    TEXT_EMBED_AVAILABLE = False

try:
    from models.image_embed import ImageEmbedModel
    IMAGE_EMBED_AVAILABLE = True
except ImportError:
    IMAGE_EMBED_AVAILABLE = False


# =============================================================================
# Fixtures - Synthetic test data
# =============================================================================

@pytest.fixture
def text_data():
    """Simple text data for TfidfModel and TextEmbedModel."""
    texts = [
        "This is a positive example with good words",
        "Another positive sentence here",
        "This is negative and bad content",
        "More negative stuff in this text",
        "Positive vibes only in this message",
        "Bad negative terrible awful text",
        "Good positive excellent wonderful",
        "Negative bad horrible content here",
    ]
    labels = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    values = np.array([0.8, 0.6, -0.7, -0.5, 0.9, -0.8, 0.7, -0.6])
    weights = np.abs(values)
    return texts, labels, values, weights


@pytest.fixture
def tabular_data():
    """Simple tabular data for TabularModel."""
    df = pd.DataFrame({
        'age': [25, 35, 45, 55, 30, 40, 50, 60],
        'bmi': [22.0, 28.5, 31.2, 24.8, 26.1, 29.3, 27.5, 23.9],
        'sbp': [120, 135, 145, 130, 125, 140, 138, 128],
        'gender': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'],
        'smoker': ['no', 'yes', 'yes', 'no', 'no', 'yes', 'no', 'yes'],
    })
    labels = np.array([0, 1, 1, 0, 0, 1, 1, 0])
    values = np.array([-0.5, 0.7, 0.8, -0.3, -0.4, 0.6, 0.5, -0.2])
    weights = np.abs(values)
    num_features = ['age', 'bmi', 'sbp']
    cat_features = ['gender', 'smoker']
    return df, labels, values, weights, num_features, cat_features


# =============================================================================
# TfidfModel Tests
# =============================================================================

class TestTfidfModel:
    """Tests for TfidfModel."""

    def test_classification_fit_predict(self, text_data):
        """Test basic classification fit and predict."""
        texts, labels, _, _ = text_data
        model = TfidfModel(task='classification')
        model.fit(texts, labels)

        assert model.is_fitted_
        preds = model.predict(texts)
        assert preds.shape == (len(texts),)
        assert set(preds).issubset({0, 1})

    def test_regression_fit_predict(self, text_data):
        """Test basic regression fit and predict."""
        texts, _, values, _ = text_data
        model = TfidfModel(task='regression')
        model.fit(texts, values)

        assert model.is_fitted_
        preds = model.predict(texts)
        assert preds.shape == (len(texts),)
        assert preds.dtype in [np.float64, np.float32]

    def test_sample_weight_classification(self, text_data):
        """Test that sample_weight is accepted for classification."""
        texts, labels, _, weights = text_data
        model = TfidfModel(task='classification')
        model.fit(texts, labels, sample_weight=weights)

        assert model.is_fitted_
        preds = model.predict(texts)
        assert preds.shape == (len(texts),)

    def test_sample_weight_regression(self, text_data):
        """Test that sample_weight is accepted for regression."""
        texts, _, values, weights = text_data
        model = TfidfModel(task='regression')
        model.fit(texts, values, sample_weight=weights)

        assert model.is_fitted_
        preds = model.predict(texts)
        assert preds.shape == (len(texts),)

    def test_predict_proba_classification(self, text_data):
        """Test predict_proba for classification."""
        texts, labels, _, _ = text_data
        model = TfidfModel(task='classification')
        model.fit(texts, labels)

        probs = model.predict_proba(texts)
        assert probs.shape == (len(texts), 2)
        assert np.allclose(probs.sum(axis=1), 1.0)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_predict_proba_regression_raises(self, text_data):
        """Test that predict_proba raises for regression."""
        texts, _, values, _ = text_data
        model = TfidfModel(task='regression')
        model.fit(texts, values)

        with pytest.raises(NotImplementedError):
            model.predict_proba(texts)

    def test_predict_before_fit_raises(self, text_data):
        """Test that predict before fit raises error."""
        texts, _, _, _ = text_data
        model = TfidfModel(task='classification')

        with pytest.raises(RuntimeError):
            model.predict(texts)

    def test_get_params(self, text_data):
        """Test get_params returns expected keys."""
        model = TfidfModel(task='classification', max_features=1000)
        params = model.get_params()

        assert 'task' in params
        assert params['task'] == 'classification'
        assert 'max_features' in params
        assert params['max_features'] == 1000

    def test_invalid_task_raises(self):
        """Test that invalid task raises ValueError."""
        with pytest.raises(ValueError):
            TfidfModel(task='invalid')

    def test_numpy_array_input(self, text_data):
        """Test that numpy array input works."""
        texts, labels, _, _ = text_data
        texts_array = np.array(texts)

        model = TfidfModel(task='classification')
        model.fit(texts_array, labels)
        preds = model.predict(texts_array)

        assert preds.shape == (len(texts),)


# =============================================================================
# TabularModel Tests
# =============================================================================

class TestTabularModel:
    """Tests for TabularModel."""

    def test_classification_fit_predict(self, tabular_data):
        """Test basic classification fit and predict."""
        df, labels, _, _, num_features, cat_features = tabular_data
        model = TabularModel(
            task='classification',
            num_features=num_features,
            cat_features=cat_features
        )
        model.fit(df, labels)

        assert model.is_fitted_
        preds = model.predict(df)
        assert preds.shape == (len(df),)
        assert set(preds).issubset({0, 1})

    def test_regression_fit_predict(self, tabular_data):
        """Test basic regression fit and predict."""
        df, _, values, _, num_features, cat_features = tabular_data
        model = TabularModel(
            task='regression',
            num_features=num_features,
            cat_features=cat_features
        )
        model.fit(df, values)

        assert model.is_fitted_
        preds = model.predict(df)
        assert preds.shape == (len(df),)
        assert preds.dtype in [np.float64, np.float32]

    def test_sample_weight_classification(self, tabular_data):
        """Test that sample_weight is accepted for classification."""
        df, labels, _, weights, num_features, cat_features = tabular_data
        model = TabularModel(
            task='classification',
            num_features=num_features,
            cat_features=cat_features
        )
        model.fit(df, labels, sample_weight=weights)

        assert model.is_fitted_
        preds = model.predict(df)
        assert preds.shape == (len(df),)

    def test_sample_weight_regression(self, tabular_data):
        """Test that sample_weight is accepted for regression."""
        df, _, values, weights, num_features, cat_features = tabular_data
        model = TabularModel(
            task='regression',
            num_features=num_features,
            cat_features=cat_features
        )
        model.fit(df, values, sample_weight=weights)

        assert model.is_fitted_
        preds = model.predict(df)
        assert preds.shape == (len(df),)

    def test_predict_proba_classification(self, tabular_data):
        """Test predict_proba for classification."""
        df, labels, _, _, num_features, cat_features = tabular_data
        model = TabularModel(
            task='classification',
            num_features=num_features,
            cat_features=cat_features
        )
        model.fit(df, labels)

        probs = model.predict_proba(df)
        assert probs.shape == (len(df), 2)
        assert np.allclose(probs.sum(axis=1), 1.0)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_predict_proba_regression_raises(self, tabular_data):
        """Test that predict_proba raises for regression."""
        df, _, values, _, num_features, cat_features = tabular_data
        model = TabularModel(
            task='regression',
            num_features=num_features,
            cat_features=cat_features
        )
        model.fit(df, values)

        with pytest.raises(NotImplementedError):
            model.predict_proba(df)

    def test_predict_before_fit_raises(self, tabular_data):
        """Test that predict before fit raises error."""
        df, _, _, _, num_features, cat_features = tabular_data
        model = TabularModel(
            task='classification',
            num_features=num_features,
            cat_features=cat_features
        )

        with pytest.raises(RuntimeError):
            model.predict(df)

    def test_get_params(self, tabular_data):
        """Test get_params returns expected keys."""
        _, _, _, _, num_features, cat_features = tabular_data
        model = TabularModel(
            task='classification',
            num_features=num_features,
            cat_features=cat_features,
            max_iter=50
        )
        params = model.get_params()

        assert 'task' in params
        assert params['task'] == 'classification'
        assert 'num_features' in params
        assert params['max_iter'] == 50

    def test_get_feature_names(self, tabular_data):
        """Test get_feature_names after fitting."""
        df, labels, _, _, num_features, cat_features = tabular_data
        model = TabularModel(
            task='classification',
            num_features=num_features,
            cat_features=cat_features
        )
        model.fit(df, labels)

        feature_names = model.get_feature_names()
        assert feature_names is not None
        assert len(feature_names) > 0
        # Should have numeric features + one-hot encoded categorical features
        assert len(feature_names) >= len(num_features)

    def test_numeric_only(self, tabular_data):
        """Test with only numeric features."""
        df, labels, _, _, num_features, _ = tabular_data
        model = TabularModel(
            task='classification',
            num_features=num_features,
            cat_features=[]
        )
        model.fit(df, labels)

        preds = model.predict(df)
        assert preds.shape == (len(df),)

    def test_categorical_only(self, tabular_data):
        """Test with only categorical features."""
        df, labels, _, _, _, cat_features = tabular_data
        model = TabularModel(
            task='classification',
            num_features=[],
            cat_features=cat_features
        )
        model.fit(df, labels)

        preds = model.predict(df)
        assert preds.shape == (len(df),)

    def test_handles_missing_values(self):
        """Test that missing values are handled correctly."""
        df = pd.DataFrame({
            'age': [25, np.nan, 45, 55],
            'bmi': [22.0, 28.5, np.nan, 24.8],
            'gender': ['M', 'F', None, 'F'],
        })
        labels = np.array([0, 1, 1, 0])

        model = TabularModel(
            task='classification',
            num_features=['age', 'bmi'],
            cat_features=['gender']
        )
        # Should not raise - imputers handle missing values
        model.fit(df, labels)
        preds = model.predict(df)
        assert preds.shape == (len(df),)


# =============================================================================
# TextEmbedModel Tests (Skip if not available)
# =============================================================================

@pytest.mark.skipif(not TEXT_EMBED_AVAILABLE, reason="TextEmbedModel not available")
class TestTextEmbedModel:
    """Tests for TextEmbedModel (requires transformers)."""

    @pytest.fixture
    def small_model(self):
        """Use a small model for faster tests."""
        # prajjwal1/bert-tiny is very small (~17MB)
        return 'prajjwal1/bert-tiny'

    def test_classification_fit_predict(self, text_data, small_model):
        """Test basic classification fit and predict."""
        texts, labels, _, _ = text_data
        model = TextEmbedModel(
            task='classification',
            hf_model=small_model,
            batch_size=4
        )
        model.fit(texts, labels)

        assert model.is_fitted_
        preds = model.predict(texts)
        assert preds.shape == (len(texts),)
        assert set(preds).issubset({0, 1})

    def test_regression_fit_predict(self, text_data, small_model):
        """Test basic regression fit and predict."""
        texts, _, values, _ = text_data
        model = TextEmbedModel(
            task='regression',
            hf_model=small_model,
            batch_size=4
        )
        model.fit(texts, values)

        assert model.is_fitted_
        preds = model.predict(texts)
        assert preds.shape == (len(texts),)

    def test_sample_weight(self, text_data, small_model):
        """Test that sample_weight is accepted."""
        texts, labels, _, weights = text_data
        model = TextEmbedModel(
            task='classification',
            hf_model=small_model,
            batch_size=4
        )
        model.fit(texts, labels, sample_weight=weights)
        assert model.is_fitted_

    def test_predict_proba_classification(self, text_data, small_model):
        """Test predict_proba for classification."""
        texts, labels, _, _ = text_data
        model = TextEmbedModel(
            task='classification',
            hf_model=small_model,
            batch_size=4
        )
        model.fit(texts, labels)

        probs = model.predict_proba(texts)
        assert probs.shape == (len(texts), 2)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_pooling_options(self, text_data, small_model):
        """Test both pooling strategies."""
        texts, labels, _, _ = text_data

        for pooling in ['mean', 'cls']:
            model = TextEmbedModel(
                task='classification',
                hf_model=small_model,
                pooling=pooling,
                batch_size=4
            )
            model.fit(texts, labels)
            preds = model.predict(texts)
            assert preds.shape == (len(texts),)


# =============================================================================
# ImageEmbedModel Tests (Skip if not available or no test images)
# =============================================================================

@pytest.mark.skipif(not IMAGE_EMBED_AVAILABLE, reason="ImageEmbedModel not available")
class TestImageEmbedModel:
    """Tests for ImageEmbedModel (requires torchvision)."""

    @pytest.fixture
    def temp_images(self, tmp_path):
        """Create temporary test images."""
        from PIL import Image

        image_paths = []
        for i in range(8):
            # Create simple colored images
            color = (255, 0, 0) if i % 2 == 0 else (0, 0, 255)
            img = Image.new('RGB', (64, 64), color=color)
            path = tmp_path / f"test_image_{i}.png"
            img.save(path)
            image_paths.append(path)

        labels = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        values = np.array([0.8, -0.6, 0.7, -0.5, 0.9, -0.7, 0.6, -0.8])
        weights = np.abs(values)
        return image_paths, labels, values, weights

    def test_classification_fit_predict(self, temp_images):
        """Test basic classification fit and predict."""
        image_paths, labels, _, _ = temp_images
        model = ImageEmbedModel(
            task='classification',
            batch_size=4
        )
        model.fit(image_paths, labels)

        assert model.is_fitted_
        preds = model.predict(image_paths)
        assert preds.shape == (len(image_paths),)
        assert set(preds).issubset({0, 1})

    def test_regression_fit_predict(self, temp_images):
        """Test basic regression fit and predict."""
        image_paths, _, values, _ = temp_images
        model = ImageEmbedModel(
            task='regression',
            batch_size=4
        )
        model.fit(image_paths, values)

        assert model.is_fitted_
        preds = model.predict(image_paths)
        assert preds.shape == (len(image_paths),)

    def test_sample_weight(self, temp_images):
        """Test that sample_weight is accepted."""
        image_paths, labels, _, weights = temp_images
        model = ImageEmbedModel(
            task='classification',
            batch_size=4
        )
        model.fit(image_paths, labels, sample_weight=weights)
        assert model.is_fitted_

    def test_predict_proba_classification(self, temp_images):
        """Test predict_proba for classification."""
        image_paths, labels, _, _ = temp_images
        model = ImageEmbedModel(
            task='classification',
            batch_size=4
        )
        model.fit(image_paths, labels)

        probs = model.predict_proba(image_paths)
        assert probs.shape == (len(image_paths), 2)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_get_feature_dim(self, temp_images):
        """Test get_feature_dim returns 2048 for ResNet50."""
        image_paths, labels, _, _ = temp_images
        model = ImageEmbedModel(task='classification')

        assert model.get_feature_dim() == 2048
