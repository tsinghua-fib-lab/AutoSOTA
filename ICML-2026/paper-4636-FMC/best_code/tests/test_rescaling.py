"""Unit tests for rescaling infrastructure.

Tests cover:
- NoOpRescaler: pass-through behavior
- ZScoreRescaler: standardization and inverse
- WhitenRescaler: whitening transformation
- Factory function: create_rescaler
- State dict serialization and loading
- Error handling and validation
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile

from utils.rescaling import (
    DataRescaler,
    NoOpRescaler,
    ZScoreRescaler,
    WhitenRescaler,
    create_rescaler,
    validate_rescaler_compatibility,
    DistributionShiftWarning,
)


class TestNoOpRescaler:
    """Test NoOpRescaler (no rescaling)."""

    def test_passthrough(self):
        """Test that NoOp rescaler passes data through unchanged."""
        rescaler = NoOpRescaler()

        # Create random data
        data = torch.randn(100, 5)

        # Transform should return identical data
        transformed = rescaler.transform(data)
        assert torch.allclose(transformed, data)

        # Inverse transform should also return identical data
        inversed = rescaler.inverse_transform(data)
        assert torch.allclose(inversed, data)

    def test_no_fit_required(self):
        """Test that NoOp rescaler works without fitting."""
        rescaler = NoOpRescaler()
        data = torch.randn(50, 3)

        # Should work without calling fit()
        transformed = rescaler.transform(data)
        assert torch.allclose(transformed, data)

    def test_mode(self):
        """Test that mode is correctly set."""
        rescaler = NoOpRescaler()
        assert rescaler.mode == "none"
        assert rescaler._is_fitted == True

    def test_config(self):
        """Test get_config method."""
        rescaler = NoOpRescaler()
        config = rescaler.get_config()

        assert config["mode"] == "none"
        assert config["is_fitted"] == True


class TestZScoreRescaler:
    """Test ZScoreRescaler (standardization)."""

    def test_fit(self):
        """Test fitting computes correct statistics."""
        # Create data with known mean and std
        data = torch.randn(1000, 10) * 3.0 + 5.0

        rescaler = ZScoreRescaler()
        rescaler.fit(data)

        # Check mean is approximately 5.0
        assert torch.abs(rescaler.mean - 5.0) < 0.2

        # Check std is approximately 3.0
        assert torch.abs(rescaler.std - 3.0) < 0.2

    def test_transform(self):
        """Test transformation standardizes data."""
        data = torch.randn(1000, 5) * 2.0 + 10.0

        rescaler = ZScoreRescaler()
        rescaler.fit(data)

        transformed = rescaler.transform(data)

        # After standardization, mean should be ~0, std should be ~1
        assert torch.abs(transformed.mean()) < 0.1
        assert torch.abs(transformed.std(unbiased=True) - 1.0) < 0.1

    def test_inverse_transform(self):
        """Test inverse transformation restores original data."""
        data = torch.randn(100, 3) * 5.0 - 2.0

        rescaler = ZScoreRescaler()
        rescaler.fit(data)

        transformed = rescaler.transform(data)
        restored = rescaler.inverse_transform(transformed)

        # Should recover original data (within numerical precision)
        assert torch.allclose(restored, data, atol=1e-5)

    def test_roundtrip(self):
        """Test transform -> inverse_transform roundtrip."""
        data = torch.randn(50, 7)

        rescaler = ZScoreRescaler()
        rescaler.fit(data)

        # Roundtrip should preserve data
        roundtrip = rescaler.inverse_transform(rescaler.transform(data))
        assert torch.allclose(roundtrip, data, atol=1e-5)

    def test_error_not_fitted(self):
        """Test error when transform called before fit."""
        rescaler = ZScoreRescaler()
        data = torch.randn(10, 2)

        with pytest.raises(RuntimeError, match="must be fitted"):
            rescaler.transform(data)

        with pytest.raises(RuntimeError, match="must be fitted"):
            rescaler.inverse_transform(data)

    def test_error_empty_data(self):
        """Test error when fitting on empty data."""
        rescaler = ZScoreRescaler()
        empty_data = torch.tensor([])

        with pytest.raises(ValueError, match="empty data"):
            rescaler.fit(empty_data)

    def test_epsilon_prevents_zero_division(self):
        """Test that epsilon prevents division by zero."""
        # Create constant data (std = 0)
        data = torch.ones(100, 3) * 5.0

        rescaler = ZScoreRescaler(epsilon=1e-8)
        rescaler.fit(data)

        # Std should be clamped to epsilon
        assert rescaler.std >= 1e-8

        # Transform should not raise error
        transformed = rescaler.transform(data)
        assert not torch.isnan(transformed).any()
        assert not torch.isinf(transformed).any()

    def test_mode_and_config(self):
        """Test mode and config."""
        rescaler = ZScoreRescaler(epsilon=1e-6)
        assert rescaler.mode == "z_score"

        # Before fitting
        config = rescaler.get_config()
        assert config["mode"] == "z_score"
        assert config["is_fitted"] == False

        # After fitting
        data = torch.randn(50, 2)
        rescaler.fit(data)
        config = rescaler.get_config()
        assert config["is_fitted"] == True
        assert "mean" in config
        assert "std" in config
        assert config["epsilon"] == 1e-6

    def test_state_dict(self):
        """Test state_dict save and load."""
        data = torch.randn(100, 4) * 2.0 + 3.0

        # Fit original rescaler
        rescaler1 = ZScoreRescaler()
        rescaler1.fit(data)

        # Save state
        state = rescaler1.state_dict()

        # Create new rescaler and load state
        rescaler2 = ZScoreRescaler()
        rescaler2.load_state_dict(state)

        # Both should produce same results
        test_data = torch.randn(20, 4)
        result1 = rescaler1.transform(test_data)
        result2 = rescaler2.transform(test_data)

        assert torch.allclose(result1, result2)
        assert torch.allclose(rescaler1.mean, rescaler2.mean)
        assert torch.allclose(rescaler1.std, rescaler2.std)


class TestWhitenRescaler:
    """Test WhitenRescaler (whitening transformation)."""

    def test_fit_zca(self):
        """Test fitting ZCA whitening."""
        # Create correlated data
        data = torch.randn(500, 10)

        rescaler = WhitenRescaler(method="zca")
        rescaler.fit(data)

        # Whiten matrix should be stored
        assert rescaler.whiten_matrix.shape == (10, 10)
        assert rescaler.dewhiten_matrix.shape == (10, 10)

    def test_fit_pca(self):
        """Test fitting PCA whitening."""
        data = torch.randn(500, 8)

        rescaler = WhitenRescaler(method="pca")
        rescaler.fit(data)

        assert rescaler.whiten_matrix.shape == (8, 8)

    def test_transform_whitens_data(self):
        """Test that transformation whitens data (identity covariance)."""
        # Create correlated data
        torch.manual_seed(42)
        data = torch.randn(1000, 5)
        # Add correlation
        data[:, 1] = data[:, 0] * 0.8 + data[:, 1] * 0.2

        rescaler = WhitenRescaler(method="zca")
        rescaler.fit(data)

        whitened = rescaler.transform(data)

        # Compute covariance of whitened data
        cov = torch.cov(whitened.T)

        # Should be approximately identity matrix
        identity = torch.eye(5)
        assert torch.allclose(cov, identity, atol=0.2)

    def test_inverse_transform(self):
        """Test inverse transformation restores data."""
        data = torch.randn(200, 6)

        rescaler = WhitenRescaler()
        rescaler.fit(data)

        whitened = rescaler.transform(data)
        restored = rescaler.inverse_transform(whitened)

        assert torch.allclose(restored, data, atol=1e-4)

    def test_roundtrip(self):
        """Test transform -> inverse_transform roundtrip."""
        data = torch.randn(100, 4)

        rescaler = WhitenRescaler()
        rescaler.fit(data)

        roundtrip = rescaler.inverse_transform(rescaler.transform(data))
        assert torch.allclose(roundtrip, data, atol=1e-4)

    def test_error_invalid_method(self):
        """Test error on invalid whitening method."""
        with pytest.raises(ValueError, match="must be 'zca' or 'pca'"):
            WhitenRescaler(method="invalid")

    def test_mode_and_config(self):
        """Test mode and config."""
        rescaler = WhitenRescaler(epsilon=1e-6, method="pca")
        assert rescaler.mode == "whiten"

        config = rescaler.get_config()
        assert config["mode"] == "whiten"
        assert config["epsilon"] == 1e-6
        assert config["method"] == "pca"


class TestCreateRescaler:
    """Test create_rescaler factory function."""

    def test_create_noop(self):
        """Test creating NoOp rescaler."""
        rescaler = create_rescaler("none")
        assert isinstance(rescaler, NoOpRescaler)
        assert rescaler.mode == "none"

    def test_create_zscore(self):
        """Test creating ZScore rescaler."""
        rescaler = create_rescaler("z_score")
        assert isinstance(rescaler, ZScoreRescaler)
        assert rescaler.mode == "z_score"

    def test_create_whiten(self):
        """Test creating Whiten rescaler."""
        rescaler = create_rescaler("whiten")
        assert isinstance(rescaler, WhitenRescaler)
        assert rescaler.mode == "whiten"

    def test_create_with_kwargs(self):
        """Test creating rescaler with kwargs."""
        rescaler = create_rescaler("z_score", epsilon=1e-7)
        assert rescaler.epsilon == 1e-7

        rescaler = create_rescaler("whiten", method="pca", epsilon=1e-6)
        assert rescaler.method == "pca"
        assert rescaler.epsilon == 1e-6

    def test_case_insensitive(self):
        """Test that mode is case insensitive."""
        rescaler1 = create_rescaler("Z_SCORE")
        rescaler2 = create_rescaler("z_score")

        assert type(rescaler1) == type(rescaler2)

    def test_none_defaults_to_noop(self):
        """Test that None mode defaults to NoOp."""
        rescaler = create_rescaler(None)
        assert isinstance(rescaler, NoOpRescaler)

    def test_error_invalid_mode(self):
        """Test error on invalid mode."""
        with pytest.raises(ValueError, match="Unknown rescaling mode"):
            create_rescaler("invalid_mode")


class TestValidateRescalerCompatibility:
    """Test validate_rescaler_compatibility function."""

    def test_compatible_rescalers(self):
        """Test that compatible rescalers pass validation."""
        rescaler1 = create_rescaler("z_score")
        rescaler2 = create_rescaler("z_score")

        # Should not raise error
        validate_rescaler_compatibility(rescaler1, rescaler2)

    def test_incompatible_rescalers(self):
        """Test that incompatible rescalers raise error."""
        rescaler1 = create_rescaler("z_score")
        rescaler2 = create_rescaler("whiten")

        with pytest.raises(ValueError, match="Rescaler mode mismatch"):
            validate_rescaler_compatibility(rescaler1, rescaler2)

    def test_custom_names(self):
        """Test validation with custom names."""
        rescaler1 = create_rescaler("z_score")
        rescaler2 = create_rescaler("none")

        with pytest.raises(ValueError, match="data_rescaler.*cond_rescaler"):
            validate_rescaler_compatibility(
                rescaler1, rescaler2,
                name1="data_rescaler",
                name2="cond_rescaler"
            )


class TestDeviceHandling:
    """Test that rescalers handle device transfers correctly."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_transfer(self):
        """Test rescaler works after moving to CUDA."""
        data = torch.randn(100, 5)

        rescaler = ZScoreRescaler()
        rescaler.fit(data)

        # Move rescaler to CUDA
        rescaler.to('cuda')

        # Transform data on CUDA
        data_cuda = data.to('cuda')
        transformed = rescaler.transform(data_cuda)

        assert transformed.device.type == 'cuda'

        # Move back to CPU and verify
        rescaler.to('cpu')
        transformed_cpu = rescaler.transform(data)

        assert torch.allclose(transformed.cpu(), transformed_cpu, atol=1e-5)

    def test_cpu_gpu_consistency(self):
        """Test that rescaler produces same results on CPU and GPU."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        data = torch.randn(50, 3)

        # Create and fit rescaler on CPU
        rescaler = ZScoreRescaler()
        rescaler.fit(data)

        # Transform on CPU
        result_cpu = rescaler.transform(data)

        # Move to GPU and transform
        rescaler.to('cuda')
        data_gpu = data.to('cuda')
        result_gpu = rescaler.transform(data_gpu)

        # Results should be the same (within numerical precision)
        assert torch.allclose(result_cpu, result_gpu.cpu(), atol=1e-5)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_sample(self):
        """Test rescaler with single sample."""
        data = torch.randn(1, 5)

        rescaler = ZScoreRescaler()
        # Should work but std will be 0 (handled by epsilon)
        rescaler.fit(data)

        transformed = rescaler.transform(data)
        assert not torch.isnan(transformed).any()

    def test_single_feature(self):
        """Test rescaler with single feature."""
        data = torch.randn(100, 1)

        rescaler = ZScoreRescaler()
        rescaler.fit(data)

        transformed = rescaler.transform(data)
        restored = rescaler.inverse_transform(transformed)

        assert torch.allclose(restored, data, atol=1e-5)

    def test_different_shapes(self):
        """Test that rescaler handles different input shapes."""
        # Fit on 2D data
        train_data = torch.randn(100, 5)
        rescaler = ZScoreRescaler()
        rescaler.fit(train_data)

        # Transform 2D data
        test_2d = torch.randn(50, 5)
        result_2d = rescaler.transform(test_2d)
        assert result_2d.shape == (50, 5)

        # Transform 1D data (single sample)
        test_1d = torch.randn(5)
        result_1d = rescaler.transform(test_1d)
        assert result_1d.shape == (5,)

    def test_batch_processing(self):
        """Test rescaler on batched data."""
        # Fit on all data
        all_data = torch.randn(1000, 10)
        rescaler = ZScoreRescaler()
        rescaler.fit(all_data)

        # Transform in batches
        batch_size = 100
        results = []
        for i in range(0, 1000, batch_size):
            batch = all_data[i:i+batch_size]
            results.append(rescaler.transform(batch))

        batched_result = torch.cat(results, dim=0)

        # Transform all at once
        full_result = rescaler.transform(all_data)

        # Should be identical
        assert torch.allclose(batched_result, full_result)


class TestIntegrationWithModels:
    """Test integration with PyTorch models."""

    def test_rescaler_in_module(self):
        """Test using rescaler as part of a PyTorch module."""
        import torch.nn as nn

        class ModelWithRescaler(nn.Module):
            def __init__(self):
                super().__init__()
                self.rescaler = ZScoreRescaler()
                self.linear = nn.Linear(5, 3)

            def forward(self, x):
                x = self.rescaler.transform(x)
                return self.linear(x)

        # Create and fit model
        model = ModelWithRescaler()
        data = torch.randn(100, 5)
        model.rescaler.fit(data)

        # Forward pass
        output = model(data)
        assert output.shape == (100, 3)

        # Save and load state dict
        state = model.state_dict()

        model2 = ModelWithRescaler()
        # Use strict=False because rescaler adds _is_fitted to state_dict
        model2.load_state_dict(state, strict=False)

        # Should produce same output
        output2 = model2(data)
        assert torch.allclose(output, output2)


class TestStateDictPersistence:
    """Test state_dict save/load with files."""

    def test_save_load_rescaler(self):
        """Test saving and loading rescaler to file."""
        data = torch.randn(100, 8) * 5.0 + 10.0

        # Create and fit rescaler
        rescaler1 = ZScoreRescaler()
        rescaler1.fit(data)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            temp_path = f.name
            torch.save({
                'config': rescaler1.get_config(),
                'state_dict': rescaler1.state_dict()
            }, temp_path)

        try:
            # Load from file
            checkpoint = torch.load(temp_path, weights_only=True)

            # Create new rescaler and load
            rescaler2 = create_rescaler(checkpoint['config']['mode'])
            rescaler2.load_state_dict(checkpoint['state_dict'])

            # Test they produce same results
            test_data = torch.randn(20, 8)
            result1 = rescaler1.transform(test_data)
            result2 = rescaler2.transform(test_data)

            assert torch.allclose(result1, result2)
        finally:
            # Clean up
            Path(temp_path).unlink()


class TestDistributionShiftDetection:
    """Test distribution shift detection functionality."""

    def test_zscore_no_warning_same_distribution(self):
        """Test that no warning is raised for data from same distribution."""
        import warnings

        # Fit on training data
        train_data = torch.randn(1000, 5) * 2.0 + 5.0
        rescaler = ZScoreRescaler()
        rescaler.fit(train_data)

        # Test on similar data (should not warn)
        eval_data = torch.randn(100, 5) * 2.0 + 5.0
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            rescaler.validate_distribution(eval_data)

        # No DistributionShiftWarning should be raised
        distribution_shift_warnings = [
            w for w in warning_list
            if issubclass(w.category, DistributionShiftWarning)
        ]
        assert len(distribution_shift_warnings) == 0

    def test_zscore_warns_on_mean_shift(self):
        """Test that warning is raised when eval data has different mean."""
        # Fit on training data
        train_data = torch.randn(1000, 5) * 2.0 + 5.0
        rescaler = ZScoreRescaler()
        rescaler.fit(train_data)

        # Test on data with very different mean (shifted by 10 std devs)
        eval_data = torch.randn(100, 5) * 2.0 + 25.0  # Mean ~25 instead of ~5
        with pytest.warns(DistributionShiftWarning, match="mean.*differs"):
            rescaler.validate_distribution(eval_data, std_threshold=3.0)

    def test_zscore_warns_on_std_shift(self):
        """Test that warning is raised when eval data has different std."""
        # Fit on training data
        train_data = torch.randn(1000, 5) * 2.0 + 5.0
        rescaler = ZScoreRescaler()
        rescaler.fit(train_data)

        # Test on data with very different std (10x larger)
        eval_data = torch.randn(100, 5) * 20.0 + 5.0  # Std ~20 instead of ~2
        with pytest.warns(DistributionShiftWarning, match="std.*differs"):
            rescaler.validate_distribution(eval_data, std_threshold=3.0)

    def test_validation_can_be_disabled(self):
        """Test that validation can be disabled via enable_validation flag."""
        # Fit on training data
        train_data = torch.randn(1000, 5) * 2.0 + 5.0
        rescaler = ZScoreRescaler()
        rescaler.fit(train_data)

        # Disable validation
        rescaler.enable_validation = False

        # Test on data with very different distribution (should not warn)
        eval_data = torch.randn(100, 5) * 20.0 + 25.0
        with pytest.warns(None) as warning_list:
            rescaler.validate_distribution(eval_data)

        # No warnings should be raised
        distribution_shift_warnings = [
            w for w in warning_list
            if issubclass(w.category, DistributionShiftWarning)
        ]
        assert len(distribution_shift_warnings) == 0

    def test_whiten_warns_on_mean_shift(self):
        """Test that whitening rescaler warns on mean shift."""
        # Fit on training data
        train_data = torch.randn(1000, 5) * 2.0 + 5.0
        rescaler = WhitenRescaler(method="zca")
        rescaler.fit(train_data)

        # Test on data with very different mean
        eval_data = torch.randn(100, 5) * 2.0 + 50.0
        with pytest.warns(DistributionShiftWarning, match="mean"):
            rescaler.validate_distribution(eval_data)

    def test_whiten_warns_on_covariance_shift(self):
        """Test that whitening rescaler warns on covariance shift."""
        # Create training data with specific covariance structure
        train_data = torch.randn(1000, 5)
        train_data[:, 0] *= 5.0  # Make first dimension have large variance

        rescaler = WhitenRescaler(method="zca")
        rescaler.fit(train_data)

        # Test on data with very different covariance structure
        eval_data = torch.randn(100, 5)
        eval_data[:, 0] *= 0.1  # Make first dimension have small variance
        eval_data[:, 4] *= 10.0  # Make last dimension have large variance

        with pytest.warns(DistributionShiftWarning, match="covariance"):
            rescaler.validate_distribution(eval_data)

    def test_noop_rescaler_no_validation(self):
        """Test that NoOpRescaler doesn't perform validation."""
        rescaler = NoOpRescaler()

        # Should not raise any warnings regardless of data
        eval_data = torch.randn(100, 5) * 100.0 + 1000.0
        with pytest.warns(None) as warning_list:
            rescaler.validate_distribution(eval_data)

        distribution_shift_warnings = [
            w for w in warning_list
            if issubclass(w.category, DistributionShiftWarning)
        ]
        assert len(distribution_shift_warnings) == 0

    def test_validation_not_called_if_not_fitted(self):
        """Test that validation is skipped if rescaler not fitted."""
        rescaler = ZScoreRescaler()
        # Don't call fit()

        eval_data = torch.randn(100, 5)
        with pytest.warns(None) as warning_list:
            rescaler.validate_distribution(eval_data)

        distribution_shift_warnings = [
            w for w in warning_list
            if issubclass(w.category, DistributionShiftWarning)
        ]
        assert len(distribution_shift_warnings) == 0

    def test_validation_custom_threshold(self):
        """Test that custom thresholds work correctly."""
        # Fit on training data
        train_data = torch.randn(1000, 5) * 2.0 + 5.0
        rescaler = ZScoreRescaler()
        rescaler.fit(train_data)

        # Test on data with moderate shift (3-5 std devs)
        eval_data = torch.randn(100, 5) * 2.0 + 13.0  # ~4 std devs away

        # With high threshold (5.0), should not warn
        with pytest.warns(None) as warning_list:
            rescaler.validate_distribution(eval_data, std_threshold=5.0)

        distribution_shift_warnings = [
            w for w in warning_list
            if issubclass(w.category, DistributionShiftWarning)
        ]
        assert len(distribution_shift_warnings) == 0

        # With low threshold (2.0), should warn
        with pytest.warns(DistributionShiftWarning):
            rescaler.validate_distribution(eval_data, std_threshold=2.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
