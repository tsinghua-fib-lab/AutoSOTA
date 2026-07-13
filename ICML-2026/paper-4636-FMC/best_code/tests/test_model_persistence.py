"""Integration tests for model persistence with rescaling.

Tests that rescaling parameters are correctly saved and loaded with model checkpoints,
ensuring reproducibility across save/load cycles.
"""

import pytest
import torch
import tempfile
from pathlib import Path

from posteriors import Estimator
from flow_matching.torch_flow import FlowMatching
from utils.rescaling import ZScoreRescaler, WhitenRescaler


@pytest.fixture
def dummy_data():
    """Create dummy training data."""
    torch.manual_seed(42)
    theta = torch.randn(100, 5)
    x = torch.randn(100, 10)
    y = torch.randn(100, 8)
    return theta, x, y


@pytest.fixture
def temp_model_path():
    """Create temporary directory for saving models."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestEstimatorPersistence:
    """Test save/load for Estimator (NPE) models."""

    def test_save_load_preserves_rescaling_zscore(self, dummy_data, temp_model_path):
        """Test that z-score rescaling persists through save/load."""
        theta, x, _ = dummy_data

        # Create and fit model
        model = Estimator(
            "gaussian",
            theta.shape[1:],
            x.shape[1:],
            density_estimator="maf",
            npe_params={"hidden_features": [16]},
            embedding_net={"type": "identity"},
        )
        model.set_scales(theta, x, "z_score")

        # Save model
        model_path = temp_model_path / "estimator.pth"
        torch.save(model.state_dict(), model_path)

        # Load into new model (must use same rescale_mode so rescaler architecture matches)
        model2 = Estimator(
            "gaussian",
            theta.shape[1:],
            x.shape[1:],
            density_estimator="maf",
            npe_params={"hidden_features": [16]},
            embedding_net={"type": "identity"},
            rescale_mode="z_score",
        )
        model2.load_state_dict(torch.load(model_path, weights_only=True))

        # Verify rescaling is identical
        test_theta = torch.randn(10, 5)
        test_x = torch.randn(10, 10)

        theta_rescaled_1, x_rescaled_1 = model.rescale(test_theta, test_x)
        theta_rescaled_2, x_rescaled_2 = model2.rescale(test_theta, test_x)

        assert torch.allclose(theta_rescaled_1, theta_rescaled_2), \
            "Theta rescaling differs after load"
        assert torch.allclose(x_rescaled_1, x_rescaled_2), \
            "X rescaling differs after load"

    def test_save_load_preserves_rescaling_whiten(self, dummy_data, temp_model_path):
        """Test that whitening rescaling persists through save/load."""
        theta, x, _ = dummy_data

        # Create and fit model
        model = Estimator(
            "gaussian",
            theta.shape[1:],
            x.shape[1:],
            density_estimator="maf",
            npe_params={"hidden_features": [16]},
            embedding_net={"type": "identity"},
        )
        model.set_scales(theta, x, "whiten")

        # Save model
        model_path = temp_model_path / "estimator_whiten.pth"
        torch.save(model.state_dict(), model_path)

        # Load into new model (fit rescalers first so WhitenRescaler buffer shapes match)
        model2 = Estimator(
            "gaussian",
            theta.shape[1:],
            x.shape[1:],
            density_estimator="maf",
            npe_params={"hidden_features": [16]},
            embedding_net={"type": "identity"},
            rescale_mode="whiten",
        )
        model2.set_scales(theta, x, "whiten")
        model2.load_state_dict(torch.load(model_path, weights_only=True))

        # Verify rescaling is identical
        test_theta = torch.randn(10, 5)
        test_x = torch.randn(10, 10)

        theta_rescaled_1, x_rescaled_1 = model.rescale(test_theta, test_x)
        theta_rescaled_2, x_rescaled_2 = model2.rescale(test_theta, test_x)

        assert torch.allclose(theta_rescaled_1, theta_rescaled_2, atol=1e-5), \
            "Theta rescaling differs after load"
        assert torch.allclose(x_rescaled_1, x_rescaled_2, atol=1e-5), \
            "X rescaling differs after load"

    def test_rescaler_mode_persists(self, dummy_data, temp_model_path):
        """Test that rescaler mode is correctly saved and loaded via Estimator.save/load."""
        theta, x, _ = dummy_data

        # Create and fit model with z_score
        model = Estimator(
            "gaussian",
            theta.shape[1:],
            x.shape[1:],
            density_estimator="maf",
            npe_params={"hidden_features": [16]},
            embedding_net={"type": "identity"},
        )
        model.set_scales(theta, x, "z_score")

        # Save using Estimator.save (persists config + weights)
        model_dir = temp_model_path / "estimator_mode"
        model.save(model_dir)

        # Load using Estimator.load (restores config including rescale_mode)
        model2 = Estimator.load(model_dir)

        # Check mode
        assert model2.data_rescaler.mode == "z_score", \
            "Data rescaler mode not persisted"
        assert model2.cond_rescaler.mode == "z_score", \
            "Cond rescaler mode not persisted"

    def test_inverse_transform_consistency(self, dummy_data, temp_model_path):
        """Test that inverse transform works correctly after load."""
        theta, x, _ = dummy_data

        # Create and fit model
        model = Estimator(
            "gaussian",
            theta.shape[1:],
            x.shape[1:],
            density_estimator="maf",
            npe_params={"hidden_features": [16]},
            embedding_net={"type": "identity"},
        )
        model.set_scales(theta, x, "z_score")

        # Save and load (must use same rescale_mode so rescaler architecture matches)
        model_path = temp_model_path / "estimator_inverse.pth"
        torch.save(model.state_dict(), model_path)

        model2 = Estimator(
            "gaussian",
            theta.shape[1:],
            x.shape[1:],
            density_estimator="maf",
            npe_params={"hidden_features": [16]},
            embedding_net={"type": "identity"},
            rescale_mode="z_score",
        )
        model2.load_state_dict(torch.load(model_path, weights_only=True))

        # Test roundtrip
        test_theta = torch.randn(10, 5)
        test_x = torch.randn(10, 10)

        theta_rescaled, x_rescaled = model2.rescale(test_theta, test_x)
        theta_back, x_back = model2.scale(theta_rescaled, x_rescaled)

        assert torch.allclose(test_theta, theta_back, atol=1e-5), \
            "Theta roundtrip failed"
        assert torch.allclose(test_x, x_back, atol=1e-5), \
            "X roundtrip failed"


class TestFlowMatchingPersistence:
    """Test save/load for FlowMatching models."""

    def test_save_load_preserves_rescaling(self, dummy_data, temp_model_path):
        """Test that FlowMatching rescaling persists through save/load."""
        theta, _, y = dummy_data

        # Create and fit model
        model = FlowMatching(
            conditional=True,
            probability_path="ot",
            prior="uniform",
            base_dist="gaussian",
            dim=theta.shape[1:],
            cond_dim=y.shape[1:],
            drift={"type": "mlp", "hidden_dims": [16, 16]},
        )
        model.set_scales(theta, y, "z_score")

        # Save
        model_path = temp_model_path / "flow_matching.pth"
        torch.save(model.state_dict(), model_path)

        # Load (must use same rescale_mode so rescaler architecture matches)
        model2 = FlowMatching(
            conditional=True,
            probability_path="ot",
            prior="uniform",
            base_dist="gaussian",
            dim=theta.shape[1:],
            cond_dim=y.shape[1:],
            drift={"type": "mlp", "hidden_dims": [16, 16]},
            rescale_mode="z_score",
        )
        model2.load_state_dict(torch.load(model_path, weights_only=True))

        # Verify rescaling is identical
        test_theta = torch.randn(10, 5)
        test_y = torch.randn(10, 8)

        theta_rescaled_1, y_rescaled_1 = model.rescale(test_theta, test_y)
        theta_rescaled_2, y_rescaled_2 = model2.rescale(test_theta, test_y)

        assert torch.allclose(theta_rescaled_1, theta_rescaled_2), \
            "Theta rescaling differs after load"
        assert torch.allclose(y_rescaled_1, y_rescaled_2), \
            "Y rescaling differs after load"

    def test_save_load_no_rescaling(self, dummy_data, temp_model_path):
        """Test that models with no rescaling work correctly."""
        theta, _, y = dummy_data

        # Create model without rescaling
        model = FlowMatching(
            conditional=True,
            probability_path="ot",
            prior="uniform",
            base_dist="gaussian",
            dim=theta.shape[1:],
            cond_dim=y.shape[1:],
            drift={"type": "mlp", "hidden_dims": [16, 16]},
        )
        model.set_scales(theta, y, "none")

        # Save and load
        model_path = temp_model_path / "flow_matching_none.pth"
        torch.save(model.state_dict(), model_path)

        model2 = FlowMatching(
            conditional=True,
            probability_path="ot",
            prior="uniform",
            base_dist="gaussian",
            dim=theta.shape[1:],
            cond_dim=y.shape[1:],
            drift={"type": "mlp", "hidden_dims": [16, 16]},
        )
        model2.load_state_dict(torch.load(model_path, weights_only=True))

        # Verify no rescaling is applied
        test_theta = torch.randn(10, 5)
        test_y = torch.randn(10, 8)

        theta_rescaled, y_rescaled = model2.rescale(test_theta, test_y)

        assert torch.allclose(theta_rescaled, test_theta), \
            "NoOp rescaler modified theta"
        assert torch.allclose(y_rescaled, test_y), \
            "NoOp rescaler modified y"


class TestRescalerStatePersistence:
    """Test that individual rescaler state persists correctly."""

    def test_zscore_rescaler_state_dict(self, dummy_data):
        """Test ZScoreRescaler state_dict includes all necessary info."""
        theta, _, _ = dummy_data

        rescaler = ZScoreRescaler()
        rescaler.fit(theta)

        state = rescaler.state_dict()

        # Check all required keys are present
        assert "mean" in state, "Missing mean in state_dict"
        assert "std" in state, "Missing std in state_dict"
        assert "_is_fitted_tensor" in state, "Missing _is_fitted_tensor flag"

        # Load into new rescaler
        rescaler2 = ZScoreRescaler()
        rescaler2.load_state_dict(state)

        # Verify functionality
        test_data = torch.randn(10, 5)
        rescaled_1 = rescaler.transform(test_data)
        rescaled_2 = rescaler2.transform(test_data)

        assert torch.allclose(rescaled_1, rescaled_2), \
            "Rescaling differs after state_dict load"

    def test_whiten_rescaler_state_dict(self, dummy_data):
        """Test WhitenRescaler state_dict includes all necessary info."""
        theta, _, _ = dummy_data

        rescaler = WhitenRescaler(method="zca")
        rescaler.fit(theta)

        state = rescaler.state_dict()

        # Check all required keys are present
        assert "mean" in state, "Missing mean in state_dict"
        assert "whiten_matrix" in state, "Missing whiten_matrix"
        assert "dewhiten_matrix" in state, "Missing dewhiten_matrix"
        assert "_is_fitted_tensor" in state, "Missing _is_fitted_tensor flag"

        # Load into new rescaler (fit first so buffer shapes match)
        rescaler2 = WhitenRescaler(method="zca")
        rescaler2.fit(torch.randn(10, theta.shape[1]))
        rescaler2.load_state_dict(state)

        # Verify functionality
        test_data = torch.randn(10, 5)
        rescaled_1 = rescaler.transform(test_data)
        rescaled_2 = rescaler2.transform(test_data)

        assert torch.allclose(rescaled_1, rescaled_2, atol=1e-5), \
            "Rescaling differs after state_dict load"
