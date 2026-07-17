"""
Tests for src/core/seed.py

Validates seed-setting functionality for reproducibility.
"""

import numpy as np
import random
import pytest
from core.seed import set_seed, get_rng


def test_set_seed_numpy_reproducibility():
    """Test that set_seed produces reproducible numpy results."""
    set_seed(42)
    result1 = np.random.rand(5)
    
    set_seed(42)
    result2 = np.random.rand(5)
    
    assert np.allclose(result1, result2)


def test_set_seed_python_random_reproducibility():
    """Test that set_seed produces reproducible Python random results."""
    set_seed(42)
    result1 = [random.random() for _ in range(5)]
    
    set_seed(42)
    result2 = [random.random() for _ in range(5)]
    
    assert result1 == result2


def test_set_seed_different_seeds_different_results():
    """Test that different seeds produce different results."""
    set_seed(42)
    result1 = np.random.rand(5)
    
    set_seed(123)
    result2 = np.random.rand(5)
    
    assert not np.allclose(result1, result2)


def test_get_rng_reproducibility():
    """Test that get_rng produces reproducible results."""
    rng1 = get_rng(42)
    result1 = rng1.random(5)
    
    rng2 = get_rng(42)
    result2 = rng2.random(5)
    
    assert np.allclose(result1, result2)


def test_get_rng_different_seeds():
    """Test that different RNG seeds produce different results."""
    rng1 = get_rng(42)
    result1 = rng1.random(5)
    
    rng2 = get_rng(123)
    result2 = rng2.random(5)
    
    assert not np.allclose(result1, result2)


def test_get_rng_none_seed_different_results():
    """Test that get_rng(None) produces different results each call."""
    rng1 = get_rng(None)
    result1 = rng1.random(5)

    rng2 = get_rng(None)
    result2 = rng2.random(5)

    # With None seed, should use system entropy and be different
    assert not np.allclose(result1, result2)


def test_pythonhashseed_set():
    """Test that PYTHONHASHSEED is set in environment."""
    import os

    set_seed(42)
    assert os.environ.get("PYTHONHASHSEED") == "42"

    set_seed(123)
    assert os.environ.get("PYTHONHASHSEED") == "123"
