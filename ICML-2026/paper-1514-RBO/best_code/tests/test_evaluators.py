"""Unit tests for evaluator functionality and stacking."""

import pytest
import torch
from bo_framework.base.evaluation_result import EvaluationResult
from bo_framework.evaluators.synthetic import ForresterEvaluator
from bo_framework.evaluators.noisy import NoisyEvaluator
from bo_framework.evaluators.corrupted import CorruptedEvaluator


class FixedCorruptor:
    """Simple corruptor for testing that adds a fixed amount."""
    
    def __init__(self, corruption: float):
        self.corruption = corruption
    
    def corrupt(self, context):
        """Add fixed corruption to current observed value."""
        return context.y_observed + self.corruption, abs(self.corruption)
    
    def reset(self):
        pass


class CountingEvaluator:
    """Evaluator that counts how many times it's called."""
    
    def __init__(self, base_evaluator):
        self.base_evaluator = base_evaluator
        self.call_count = 0
    
    def evaluate(self, params):
        self.call_count += 1
        return self.base_evaluator.evaluate(params)
    
    @property
    def is_deterministic(self):
        return self.base_evaluator.is_deterministic


class TestEvaluationResult:
    """Test EvaluationResult dataclass functionality."""
    
    def test_from_true_value(self):
        """Test creating result from true value."""
        x = {'x0': 0.5}
        result = EvaluationResult.from_true_value(x, 10.0)
        assert result.x == x
        assert result.y_true == 10.0
        assert result.y_noisy == 10.0
        assert result.y_observed == 10.0
        assert result.noise == 0.0
        assert result.corruption == 0.0
    
    def test_with_noise_accumulation(self):
        """Test that noise accumulates correctly."""
        x = {'x0': 0.5}
        base = EvaluationResult.from_true_value(x, 10.0)
        
        # Add first noise
        noisy1 = base.with_noise(2.0)
        assert noisy1.y_true == 10.0
        assert noisy1.y_noisy == 12.0
        assert noisy1.y_observed == 12.0
        assert noisy1.noise == 2.0
        assert noisy1.corruption == 0.0
        
        # Add second noise (should accumulate)
        noisy2 = noisy1.with_noise(1.5)
        assert noisy2.y_true == 10.0
        assert noisy2.y_noisy == 13.5
        assert noisy2.y_observed == 13.5
        assert noisy2.noise == 3.5
        assert noisy2.corruption == 0.0
    
    def test_with_corruption_accumulation(self):
        """Test that corruption accumulates correctly."""
        x = {'x0': 0.5}
        base = EvaluationResult.from_true_value(x, 10.0)
        
        # Add first corruption
        corrupt1 = base.with_corruption(5.0)
        assert corrupt1.y_true == 10.0
        assert corrupt1.y_noisy == 10.0
        assert corrupt1.y_observed == 15.0
        assert corrupt1.noise == 0.0
        assert corrupt1.corruption == 5.0
        
        # Add second corruption (should accumulate)
        corrupt2 = corrupt1.with_corruption(-3.0)
        assert corrupt2.y_true == 10.0
        assert corrupt2.y_noisy == 10.0
        assert corrupt2.y_observed == 12.0
        assert corrupt2.noise == 0.0
        assert corrupt2.corruption == 2.0
    
    def test_noise_then_corruption(self):
        """Test adding noise then corruption."""
        x = {'x0': 0.5}
        base = EvaluationResult.from_true_value(x, 10.0)
        noisy = base.with_noise(2.0)
        corrupt = noisy.with_corruption(3.0)
        
        assert corrupt.y_true == 10.0
        assert corrupt.y_noisy == 12.0
        assert corrupt.y_observed == 15.0
        assert corrupt.noise == 2.0
        assert corrupt.corruption == 3.0


class TestBasicEvaluators:
    """Test individual evaluator functionality."""
    
    def test_forrester_evaluator(self):
        """Test ForresterEvaluator returns correct format."""
        evaluator = ForresterEvaluator()
        result = evaluator.evaluate({'x0': 0.5})
        
        assert isinstance(result, EvaluationResult)
        assert result.y_true == result.y_noisy == result.y_observed
        assert result.noise == 0.0
        assert result.corruption == 0.0
        assert evaluator.is_deterministic
    
    def test_noisy_evaluator(self):
        """Test NoisyEvaluator adds noise correctly."""
        base = ForresterEvaluator()
        noisy = NoisyEvaluator(base, noise_std=1.0, seed=42)
        
        result = noisy.evaluate({'x0': 0.5})
        
        assert isinstance(result, EvaluationResult)
        assert result.y_true == base.evaluate({'x0': 0.5}).y_true
        assert result.y_noisy != result.y_true  # Should have noise
        assert result.y_observed == result.y_noisy  # No corruption yet
        assert result.noise != 0.0
        assert result.corruption == 0.0
        assert not noisy.is_deterministic
    
    def test_corrupted_evaluator(self):
        """Test CorruptedEvaluator adds corruption correctly."""
        base = ForresterEvaluator()
        corruptor = FixedCorruptor(corruption=5.0)
        corrupted = CorruptedEvaluator(base, corruptor)
        
        result = corrupted.evaluate({'x0': 0.5})
        
        assert isinstance(result, EvaluationResult)
        base_result = base.evaluate({'x0': 0.5})
        assert result.y_true == base_result.y_true
        assert result.y_noisy == base_result.y_noisy  # Should be same as base
        assert result.y_observed == base_result.y_observed + 5.0
        assert result.noise == 0.0
        assert result.corruption == 5.0
        assert not corrupted.is_deterministic


class TestEvaluatorStacking:
    """Test stacking multiple evaluators."""
    
    def test_double_noise(self):
        """Test stacking two noisy evaluators."""
        base = ForresterEvaluator()
        noise1 = NoisyEvaluator(base, noise_std=1.0, seed=42)
        noise2 = NoisyEvaluator(noise1, noise_std=0.5, seed=123)
        
        base_result = base.evaluate({'x0': 0.5})
        result = noise2.evaluate({'x0': 0.5})
        
        assert result.y_true == base_result.y_true
        assert result.y_noisy != base_result.y_noisy
        assert result.y_observed == result.y_noisy
        assert result.noise != 0.0  # Should have accumulated noise
        assert result.corruption == 0.0
        
        # Verify relationship: y_noisy = y_true + noise
        assert abs(result.y_noisy - (result.y_true + result.noise)) < 1e-10
    
    def test_double_corruption(self):
        """Test stacking two corrupted evaluators."""
        base = ForresterEvaluator()
        corrupt1 = CorruptedEvaluator(base, FixedCorruptor(2.0))
        corrupt2 = CorruptedEvaluator(corrupt1, FixedCorruptor(-0.5))
        
        base_result = base.evaluate({'x0': 0.5})
        result = corrupt2.evaluate({'x0': 0.5})
        
        assert result.y_true == base_result.y_true
        assert result.y_noisy == base_result.y_noisy
        assert result.corruption == 1.5  # 2.0 + (-0.5)
        assert result.y_observed == result.y_noisy + result.corruption
        assert result.noise == 0.0
    
    def test_noise_then_corruption(self):
        """Test noise evaluator wrapped by corrupted evaluator."""
        base = ForresterEvaluator()
        noisy = NoisyEvaluator(base, noise_std=1.0, seed=42)
        corrupted = CorruptedEvaluator(noisy, FixedCorruptor(3.0))
        
        base_result = base.evaluate({'x0': 0.5})
        result = corrupted.evaluate({'x0': 0.5})
        
        assert result.y_true == base_result.y_true
        assert result.noise != 0.0  # Should preserve noise
        assert result.corruption == 3.0
        
        # Verify relationships
        assert abs(result.y_noisy - (result.y_true + result.noise)) < 1e-10
        assert abs(result.y_observed - (result.y_noisy + result.corruption)) < 1e-10
    
    def test_corruption_then_noise(self):
        """Test corrupted evaluator wrapped by noise evaluator."""
        base = ForresterEvaluator()
        corrupted = CorruptedEvaluator(base, FixedCorruptor(2.0))
        noisy = NoisyEvaluator(corrupted, noise_std=0.5, seed=42)
        
        base_result = base.evaluate({'x0': 0.5})
        result = noisy.evaluate({'x0': 0.5})
        
        assert result.y_true == base_result.y_true
        assert result.noise != 0.0
        assert result.corruption == 2.0  # Should preserve corruption
        
        # Verify relationships
        assert abs(result.y_noisy - (result.y_true + result.noise)) < 1e-10
        assert abs(result.y_observed - (result.y_noisy + result.corruption)) < 1e-10
    
    def test_complex_stacking(self):
        """Test complex stacking: Base → Noise1 → Corrupt1 → Noise2 → Corrupt2."""
        base = ForresterEvaluator()
        noise1 = NoisyEvaluator(base, noise_std=1.0, seed=42)
        corrupt1 = CorruptedEvaluator(noise1, FixedCorruptor(1.5))
        noise2 = NoisyEvaluator(corrupt1, noise_std=0.3, seed=123)
        corrupt2 = CorruptedEvaluator(noise2, FixedCorruptor(-0.8))
        
        base_result = base.evaluate({'x0': 0.5})
        result = corrupt2.evaluate({'x0': 0.5})
        
        assert result.y_true == base_result.y_true
        assert result.noise != 0.0  # Should have accumulated noise
        assert abs(result.corruption - 0.7) < 1e-10  # 1.5 + (-0.8)
        
        # Verify final relationships hold
        assert abs(result.y_noisy - (result.y_true + result.noise)) < 1e-10
        assert abs(result.y_observed - (result.y_noisy + result.corruption)) < 1e-10


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_zero_noise(self):
        """Test noisy evaluator with zero noise."""
        base = ForresterEvaluator()
        noisy = NoisyEvaluator(base, noise_std=0.0, seed=42)
        
        base_result = base.evaluate({'x0': 0.5})
        result = noisy.evaluate({'x0': 0.5})
        
        # Should be identical to base when noise_std=0
        assert result.y_true == base_result.y_true
        assert result.y_noisy == base_result.y_noisy
        assert result.y_observed == base_result.y_observed
        assert result.noise == 0.0
        assert result.corruption == 0.0
    
    def test_zero_corruption(self):
        """Test corrupted evaluator with zero corruption."""
        base = ForresterEvaluator()
        corrupted = CorruptedEvaluator(base, FixedCorruptor(0.0))
        
        base_result = base.evaluate({'x0': 0.5})
        result = corrupted.evaluate({'x0': 0.5})
        
        # Should be identical to base when corruption=0
        assert result.y_true == base_result.y_true
        assert result.y_noisy == base_result.y_noisy
        assert result.y_observed == base_result.y_observed
        assert result.noise == 0.0
        assert result.corruption == 0.0
    
    def test_negative_corruption(self):
        """Test corrupted evaluator with negative corruption."""
        base = ForresterEvaluator()
        corrupted = CorruptedEvaluator(base, FixedCorruptor(-3.0))
        
        base_result = base.evaluate({'x0': 0.5})
        result = corrupted.evaluate({'x0': 0.5})
        
        assert result.y_true == base_result.y_true
        assert result.y_observed == base_result.y_observed - 3.0
        assert result.corruption == -3.0
    
    def test_reproducibility_with_seeds(self):
        """Test that same seeds produce same results."""
        base = ForresterEvaluator()
        noisy1 = NoisyEvaluator(base, noise_std=1.0, seed=42)
        noisy2 = NoisyEvaluator(base, noise_std=1.0, seed=42)
        
        result1 = noisy1.evaluate({'x0': 0.5})
        result2 = noisy2.evaluate({'x0': 0.5})
        
        assert result1.y_true == result2.y_true
        assert result1.y_noisy == result2.y_noisy
        assert result1.y_observed == result2.y_observed
        assert result1.noise == result2.noise
    
    def test_no_re_evaluation_in_stacking(self):
        """Test that stacking evaluators doesn't cause re-evaluation."""
        base = CountingEvaluator(ForresterEvaluator())
        noisy = NoisyEvaluator(base, noise_std=1.0, seed=42)
        corrupted = CorruptedEvaluator(noisy, FixedCorruptor(2.0))
        
        # Single evaluation should only call base evaluator once
        result = corrupted.evaluate({'x0': 0.5})
        
        assert base.call_count == 1
        assert isinstance(result, EvaluationResult)
        
        # Multiple evaluations should increment count properly
        corrupted.evaluate({'x0': 0.3})
        corrupted.evaluate({'x0': 0.7})
        
        assert base.call_count == 3
    
    def test_evaluation_result_includes_input(self):
        """Test that EvaluationResult properly includes input parameters."""
        evaluator = ForresterEvaluator()
        
        # Test with dict input
        params_dict = {'x0': 0.5}
        result = evaluator.evaluate(params_dict)
        assert result.x == params_dict
        assert isinstance(result.x, dict)
        
        # Test with tensor input  
        import torch
        params_tensor = torch.tensor([0.3])
        result2 = evaluator.evaluate(params_tensor)
        assert torch.equal(result2.x, params_tensor)
        assert isinstance(result2.x, torch.Tensor)


if __name__ == "__main__":
    pytest.main([__file__])