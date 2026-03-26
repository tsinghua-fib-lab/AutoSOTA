"""Integration tests for reproducibility framework."""
import numpy as np

# Import reproducibility modules
from reproducibility.seed_manager import (
    get_default_seed,
    get_random_generator,
    set_global_seed,
    with_seed
)
from reproducibility.config import get_config

# Test functions using our reproducibility framework
def test_consistent_random_generation():
    """Test that random generation with the same seed is consistent."""
    # First generation with seed 123
    with with_seed(123):
        values1 = np.random.rand(10)
    
    # Second generation with the same seed should produce identical values
    with with_seed(123):
        values2 = np.random.rand(10)
    
    # Verify values are identical
    assert np.array_equal(values1, values2)


def test_config_and_seed_management_integration():
    """Test that configuration and seed management work together."""
    # Set a specific seed in the configuration
    config = get_config()
    config.seed = 456
    
    # Generate random values using the config's seed
    rng1 = get_random_generator(config.seed)
    values1 = rng1.random(10)
    
    # Set the same seed directly and verify results are the same
    rng2 = get_random_generator(456)
    values2 = rng2.random(10)
    
    assert np.array_equal(values1, values2)
