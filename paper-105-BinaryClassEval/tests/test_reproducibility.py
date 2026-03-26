"""
Tests for reproducibility guarantees in the randomization system.
These tests verify that the seed management system works as expected.
"""
import numpy as np
import pytest

from reproducibility.seed_manager import (
    get_default_seed,
    get_random_generator,
    get_derived_seed,
    set_global_seed,
    with_seed
)


def test_default_seed_consistent():
    """Test that the default seed is consistent."""
    seed1 = get_default_seed()
    seed2 = get_default_seed()
    assert seed1 == seed2
    assert seed1 == 42  # Default value if not overridden by environment


def test_random_generator_reproducibility():
    """Test that random generators with the same seed produce identical results."""
    rng1 = get_random_generator(seed=123)
    rng2 = get_random_generator(seed=123)
    
    # Generate random values and compare
    values1 = rng1.integers(0, 1000, size=100)
    values2 = rng2.integers(0, 1000, size=100)
    assert np.array_equal(values1, values2)


def test_different_seeds_different_outputs():
    """Test that different seeds produce different outputs."""
    rng1 = get_random_generator(seed=123)
    rng2 = get_random_generator(seed=456)
    
    # Generate random values and verify they're different
    values1 = rng1.integers(0, 1000, size=100)
    values2 = rng2.integers(0, 1000, size=100)
    assert not np.array_equal(values1, values2)


def test_derived_seeds_deterministic():
    """Test that derived seeds are deterministic based on inputs."""
    seed1 = get_derived_seed(base_seed=100, component_name="test1")
    seed2 = get_derived_seed(base_seed=100, component_name="test1")
    assert seed1 == seed2
    
    # Different component should produce different derived seed
    seed3 = get_derived_seed(base_seed=100, component_name="test2")
    assert seed1 != seed3


def test_component_name_affects_generator():
    """Test that specifying component_name creates different but reproducible generators."""
    rng1 = get_random_generator(seed=123, component_name="component1")
    rng2 = get_random_generator(seed=123, component_name="component2")
    rng1b = get_random_generator(seed=123, component_name="component1")
    
    # Different component names should produce different results
    values1 = rng1.integers(0, 1000, size=100)
    values2 = rng2.integers(0, 1000, size=100)
    assert not np.array_equal(values1, values2)
    
    # Same component name should be reproducible
    values1b = rng1b.integers(0, 1000, size=100)
    assert np.array_equal(values1, values1b)


def test_with_seed_context_manager():
    """Test that the with_seed context manager works correctly."""
    # Set a known global seed
    original_state = np.random.get_state()
    
    # Generate a value before the context
    set_global_seed(111)
    before = np.random.rand()
    
    # Generate a value within a context using a different seed
    with with_seed(222):
        within_context = np.random.rand()
        
        # Nested context
        with with_seed(333):
            nested_context = np.random.rand()
        
        # Back to original context seed
        after_nested = np.random.rand()
    
    # Generate a value after the context
    after_context = np.random.rand()
    
    # Reset global state
    np.random.set_state(original_state)
    
    # Verify that the context manager restores the previous state
    set_global_seed(111)
    repeat_before = np.random.rand()
    assert before == repeat_before
    
    # Within the context with seed 222
    set_global_seed(222)
    repeat_within = np.random.rand()
    assert within_context == repeat_within
    
    # Nested context with seed 333
    set_global_seed(333)
    repeat_nested = np.random.rand()
    assert nested_context == repeat_nested
    
    # Back to context with seed 222
    set_global_seed(222)
    np.random.rand()  # Skip the first call that was used for within_context
    repeat_after_nested = np.random.rand()
    assert after_nested == repeat_after_nested
    
    # After exiting context, should be back to seed 111
    set_global_seed(111)
    np.random.rand()  # Skip the first call used for before
    repeat_after = np.random.rand()
    assert after_context == repeat_after
