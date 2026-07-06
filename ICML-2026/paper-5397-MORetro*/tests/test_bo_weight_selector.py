"""
Unit tests for BOWeightSelector class in bo_weight_selector.py
"""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from moretro.utils.bo_weight_selector import BOWeightSelector


class TestBOWeightSelector:
    """Test cases for BOWeightSelector class"""

    @pytest.fixture
    def bo_selector(self):
        """Create a BOWeightSelector instance for testing"""
        return BOWeightSelector(
            n_obj=3,
            seed=42,
            kappa=2.0,
            n_warmup=5,
            decay_factor=0.5,
            max_age=2,
        )

    def test_init(self, bo_selector):
        """Test initialization"""
        assert bo_selector.n_obj == 3
        assert bo_selector.n_warmup == 5
        assert bo_selector.decay_factor == 0.5
        assert bo_selector.max_age == 2
        assert len(bo_selector.weights_history) == 0
        assert len(bo_selector.utilities_history) == 0
        assert len(bo_selector.batch_ids) == 0
        assert bo_selector.current_batch_id == 0

    def test_add_batch_warmup(self, bo_selector):
        """Test add_batch during warmup phase"""
        weights = np.array([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]])
        bo_selector.add_batch(weights)

        assert len(bo_selector.weights_history) == 2
        assert len(bo_selector.utilities_history) == 2
        assert len(bo_selector.batch_ids) == 2
        # In warmup, batch_id should be 0
        assert all(bid == 0 for bid in bo_selector.batch_ids)
        # Weights are inserted at the beginning (reversed input)
        np.testing.assert_array_equal(bo_selector.weights_history[0], weights[0])
        np.testing.assert_array_equal(bo_selector.weights_history[1], weights[1])

    def test_add_batch_post_warmup(self, bo_selector):
        """Test add_batch after warmup phase"""
        # Fill warmup
        warmup_weights = np.zeros((5, 3))
        bo_selector.add_batch(warmup_weights)
        assert bo_selector.current_batch_id == 0

        # Add new batch
        new_weights = np.array([[0.5, 0.5, 0.0]])
        bo_selector.add_batch(new_weights)

        assert bo_selector.current_batch_id == 1
        assert bo_selector.batch_ids[0] == 1
        assert len(bo_selector.weights_history) == 6

    def test_process_pareto_update(self, bo_selector):
        """Test processing of pareto updates and utility assignment"""
        # Setup history
        weights = np.array([[0.5, 0.5, 0.0]])
        bo_selector.add_batch(weights)

        # Mock compute_hypervolume to return controlled values
        with patch.object(bo_selector, "compute_hypervolume") as mock_hv:
            mock_hv.side_effect = [10.0, 12.0]  # old_hv, new_hv

            old_pareto = np.array([[1.0, 1.0, 1.0]])
            new_pareto = np.array([[0.5, 0.5, 0.5]])
            contributing_indices = {
                0
            }  # Index 0 in history (which is the weight we just added)

            bo_selector.process_pareto_update(
                old_pareto, new_pareto, contributing_indices
            )

            # Delta HV should be 2.0
            # Share per index = 2.0 / 1 = 2.0
            assert bo_selector.utilities_history[0] == 2.0

        # Add Batch 1
        # Fill up warmup first to trigger batch increment
        bo_selector.n_warmup = 1  # Set low warmup for test
        bo_selector.add_batch(np.zeros((1, 3)))  # This will be batch 1

        assert bo_selector.current_batch_id == 1
        bo_selector.utilities_history[0] = 5.0  # Newest weight is at index 0

    def test_prepare_gp_data_decay(self, bo_selector):
        """Test GP data preparation with time decay"""
        bo_selector.n_warmup = 0  # Disable warmup logic for batch ids
        bo_selector.max_age = 2
        bo_selector.decay_factor = 0.5

        # Add 4 batches
        for i in range(4):
            bo_selector.add_batch(np.array([[float(i)] * 3]))

        # History: [Batch 3, Batch 2, Batch 1, Batch 0]
        # Current batch ID is 3
        # Ages: [0, 1, 2, 3]

        # Set utilities
        bo_selector.utilities_history = [10.0, 10.0, 10.0, 10.0]

        train_X, train_Y = bo_selector._prepare_gp_data()

        y_np = train_Y.numpy().flatten()
        # Expected decay:
        # Age 0: 10.0 * 0.5^0 = 10.0
        # Age 1: 10.0 * 0.5^1 = 5.0
        # Age 2: 10.0 * 0.5^2 = 2.5
        # Age 3: 0.0 (since > max_age=2)

        # Note: _prepare_gp_data applies log1p at the end
        expected_raw = np.array([10.0, 5.0, 2.5, 0.0])
        expected_log = np.log1p(expected_raw)

        np.testing.assert_allclose(y_np, expected_log, atol=1e-5)

    @patch("moretro.utils.bo_weight_selector.SingleTaskGP")
    @patch("moretro.utils.bo_weight_selector.ExactMarginalLogLikelihood")
    @patch("moretro.utils.bo_weight_selector.fit_gpytorch_mll")
    @patch("moretro.utils.bo_weight_selector.optimize_acqf_discrete")
    def test_select_next_weights(
        self, mock_opt, mock_fit, mock_mll, mock_gp, bo_selector
    ):
        """Test selection of next weights"""
        # Setup state
        bo_selector.weights_history = [
            np.array([0.1, 0.1, 0.8])
        ] * 10  # Enough for warmup
        bo_selector.utilities_history = [1.0] * 10
        bo_selector.batch_ids = [0] * 10

        weights_open = np.array([[0.2, 0.2, 0.6], [0.3, 0.3, 0.4]])
        k = 1

        # Configure GP mock to avoid AttributeError
        mock_model = MagicMock()
        mock_model.likelihood = MagicMock()
        mock_model.covar_module = MagicMock()  # For lengthscale check
        mock_model.num_outputs = 1  # For GIBBON check
        # Mock train_inputs for GIBBON check - needs to be a tensor for torch.cat
        # Shape should match candidate_set feature dim (3)
        mock_train_input = torch.tensor([[0.1, 0.1, 0.8]], dtype=torch.double)
        mock_model.train_inputs = [mock_train_input]
        mock_gp.return_value = mock_model

        # Mock optimization result
        # optimize_acqf_discrete returns (candidates, acq_values)
        mock_opt.return_value = (
            torch.tensor([[0.2, 0.2, 0.6]], dtype=torch.double),
            None,
        )

        selected, remaining = bo_selector.select_next_weights(weights_open, k)

        assert len(selected) == 1
        assert len(remaining) == 1
        np.testing.assert_array_equal(selected[0], weights_open[0])

        # Verify GP was called
        mock_gp.assert_called_once()
        mock_fit.assert_called_once()

    @patch("moretro.utils.bo_weight_selector.SingleTaskGP")
    @patch("moretro.utils.bo_weight_selector.ExactMarginalLogLikelihood")
    @patch("moretro.utils.bo_weight_selector.fit_gpytorch_mll")
    def test_get_max_ucb(self, mock_fit, mock_mll, mock_gp, bo_selector):
        """Test max UCB calculation"""
        # Setup state
        bo_selector.weights_history = [np.array([0.1, 0.1, 0.8])] * 10
        bo_selector.utilities_history = [1.0] * 10
        bo_selector.batch_ids = [0] * 10

        weights_open = np.array([[0.2, 0.2, 0.6]])

        # Mock GP posterior
        mock_model = MagicMock()
        mock_model.likelihood = MagicMock()
        mock_model.covar_module = MagicMock()

        mock_posterior = MagicMock()
        mock_posterior.mean = torch.tensor([1.0])
        mock_posterior.variance = torch.tensor([0.04])  # std = 0.2
        mock_model.posterior.return_value = mock_posterior
        mock_gp.return_value = mock_model

        # Expected UCB = mean + kappa * std = 1.0 + 2.0 * 0.2 = 1.4

        max_ucb = bo_selector.get_max_ucb(weights_open)

        assert max_ucb == pytest.approx(1.4)
