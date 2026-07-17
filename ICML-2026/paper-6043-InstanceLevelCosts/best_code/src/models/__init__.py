"""
Model abstractions for cost-sensitive learning experiments.

This package provides reusable model classes that support both
classification and regression tasks with optional sample weighting.
"""

from typing import Dict, Any, List, Callable

from models.base import BaseModel
from models.tfidf import TfidfModel
from models.tabular import TabularModel

__all__ = ['BaseModel', 'TfidfModel', 'TabularModel', 'get_model', 'list_models']


def _make_tfidf(task: str, **kwargs) -> BaseModel:
    """Create TfidfModel."""
    return TfidfModel(task=task, **kwargs)


def _make_roberta(task: str, **kwargs) -> BaseModel:
    """Create TextEmbedModel with RoBERTa."""
    from models.text_embed import TextEmbedModel
    return TextEmbedModel(
        task=task,
        hf_model='roberta-base',
        **kwargs
    )


def _make_resnet50(task: str, **kwargs) -> BaseModel:
    """Create ImageEmbedModel with ResNet50."""
    from models.image_embed import ImageEmbedModel
    return ImageEmbedModel(task=task, **kwargs)


def _make_histgbm(task: str, **kwargs) -> BaseModel:
    """Create TabularModel with HistGradientBoosting."""
    return TabularModel(task=task, **kwargs)


# Model registry: name -> factory function
_MODEL_FACTORIES: Dict[str, Callable[..., BaseModel]] = {
    'tfidf': _make_tfidf,
    'roberta': _make_roberta,
    'resnet50': _make_resnet50,
    'histgbm': _make_histgbm,
}

# Model -> compatible feature types
_MODEL_FEATURE_TYPES: Dict[str, List[str]] = {
    'tfidf': ['text'],
    'roberta': ['text'],
    'resnet50': ['image'],
    'histgbm': ['tabular'],
}


def get_model(
    name: str,
    task: str = 'classification',
    **kwargs,
) -> BaseModel:
    """
    Get a model instance by name.

    Args:
        name: Model name ('tfidf', 'roberta', 'resnet50', 'histgbm')
        task: 'classification' or 'regression'
        **kwargs: Additional model-specific arguments

    Returns:
        Configured BaseModel instance

    Raises:
        ValueError: If model name is not recognized
    """
    if name not in _MODEL_FACTORIES:
        available = ', '.join(_MODEL_FACTORIES.keys())
        raise ValueError(f"Unknown model: {name}. Available: {available}")

    factory = _MODEL_FACTORIES[name]
    return factory(task=task, **kwargs)


def list_models() -> List[str]:
    """Return list of available model names."""
    return list(_MODEL_FACTORIES.keys())


def get_model_feature_types(name: str) -> List[str]:
    """Return compatible feature types for a model."""
    if name not in _MODEL_FEATURE_TYPES:
        raise ValueError(f"Unknown model: {name}")
    return _MODEL_FEATURE_TYPES[name]
