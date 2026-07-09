import sys

import numpy as np
import pytest
from omegaconf import OmegaConf

import pandas as pd

from cpc_llm.calibrate.grid import prepare_grid
from cpc_llm.calibrate.normalization import (
    importance_weighted_monte_carlo_integration,
    iwmci_overlap_est,
)
from cpc_llm.calibrate.process_likelihoods import (
    check_col_names,
    constrain_likelihoods,
    mixture_pdf_from_densities_mat,
)
from cpc_llm.data_contracts import PARTICLE, SCORE, con_lik_col, lik_col


class TestPrepareGrid:
    @pytest.fixture
    def cfg(self):
        return OmegaConf.create({})

    def test_sorting_and_coarsening(self, cfg):
        V = np.array([5.0, 1.0, 3.0, 2.0, 4.0, 1.0, 3.0])
        grid = prepare_grid(cfg, V, n_grid=3, proposal="unconstrained")

        # Should be sorted and unique (before inf)
        assert np.all(grid[:-1] <= grid[1:])
        # Should be coarsened (fewer elements than unique inputs)
        assert len(grid) <= len(np.unique(V)) + 1

    @pytest.mark.parametrize(
        "proposal,check_first,check_last",
        [
            ("unconstrained", False, True),  # ends with inf
            ("safe", True, False),  # starts with float_min
            ("mixed", True, True),  # both
        ],
    )
    def test_proposal_endpoints(self, cfg, proposal, check_first, check_last):
        V = np.array([1.0, 2.0, 3.0])
        grid = prepare_grid(cfg, V, n_grid=50, proposal=proposal)

        if check_first:
            assert grid[0] == sys.float_info.min
        if check_last:
            assert grid[-1] == np.inf

    def test_invalid_proposal_raises(self, cfg):
        with pytest.raises(ValueError, match="unrecognized proposal"):
            prepare_grid(cfg, np.array([1.0]), proposal="invalid")


class TestIWMCI:
    def test_safe_proposal(self):
        # With safe proposal: mean(min(LR, beta))
        LRs = np.array([0.5, 1.5, 2.0, 3.0])
        beta = 1.0
        result = importance_weighted_monte_carlo_integration(LRs, beta, "safe")
        expected = np.mean(np.minimum(LRs, beta))
        np.testing.assert_almost_equal(result, expected)

    def test_unconstrained_proposal(self):
        # With unconstrained proposal: mean(min(beta/LR, 1))
        LRs = np.array([0.5, 1.5, 2.0, 3.0])
        beta = 2.0
        result = importance_weighted_monte_carlo_integration(LRs, beta, "unconstrained")
        expected = np.mean(np.minimum(beta / LRs, 1))
        np.testing.assert_almost_equal(result, expected)

    def test_beta_one_safe_bounds_output(self):
        # With beta=1 and safe proposal, result should be <= 1
        LRs = np.array([0.1, 0.5, 1.0, 5.0, 10.0])
        result = importance_weighted_monte_carlo_integration(LRs, 1.0, "safe")
        assert 0 < result <= 1.0


class TestConstrainLikelihoods:
    def test_init_mode_two_models(self, cpc_cfg_init):
        # 4 proposals, 2 models (safe + unconstrained)
        liks = np.array([[1.0, 0.5], [1.0, 2.0], [1.0, 1.0], [1.0, 3.0]])
        betas = np.array([0.0, 2.0])  # beta[0] unused, beta[1] is the bound
        psis = np.array([1.0, 1.0])

        result = constrain_likelihoods(cpc_cfg_init, liks, betas, psis)

        # First column should be unchanged (safe model)
        np.testing.assert_array_equal(result[:, 0], liks[:, 0])
        # Second column: where ratio < beta, use lik/psi; else use safe * beta/psi
        for i in range(4):
            ratio = liks[i, 1] / result[i, 0]
            if ratio < betas[1]:
                assert result[i, 1] == liks[i, 1] / psis[1]
            else:
                assert result[i, 1] == result[i, 0] * (betas[1] / psis[1])

    def test_sequential_mode_three_models(self, cpc_cfg_sequential):
        # 3 proposals, 3 models
        liks = np.array([[1.0, 2.0, 4.0], [1.0, 0.5, 0.3]])
        betas = np.array([0.0, 3.0, 3.0])
        psis = np.array([1.0, 1.0, 1.0])

        result = constrain_likelihoods(cpc_cfg_sequential, liks, betas, psis)

        # First column unchanged
        np.testing.assert_array_equal(result[:, 0], liks[:, 0])
        # Each subsequent column constrains against the previous
        assert result.shape == liks.shape

    def test_init_mode_rejects_three_models(self, cpc_cfg_init):
        liks = np.array([[1.0, 2.0, 3.0]])
        with pytest.raises(ValueError, match="only constrain likelihoods"):
            constrain_likelihoods(cpc_cfg_init, liks, [0, 1, 1], [1, 1, 1])


class TestMixturePdf:
    def test_uniform_weights(self):
        densities = np.array([[1.0, 3.0], [2.0, 4.0]])
        weights = np.array([1.0, 1.0])

        result = mixture_pdf_from_densities_mat(densities, weights)

        # Uniform weights → simple average of columns
        np.testing.assert_array_almost_equal(result, [2.0, 3.0])

    def test_single_weight(self):
        densities = np.array([[1.0, 3.0], [2.0, 4.0]])
        weights = np.array([0.0, 5.0])

        result = mixture_pdf_from_densities_mat(densities, weights)

        # All weight on second column
        np.testing.assert_array_almost_equal(result, [3.0, 4.0])


class TestCheckColNames:
    def test_valid_lik_columns(self):
        df = pd.DataFrame(columns=[PARTICLE, SCORE, lik_col(0), lik_col(1), lik_col(2)])
        check_col_names(df)  # should not raise

    def test_valid_constrained_lik_columns(self):
        df = pd.DataFrame(
            columns=[PARTICLE, SCORE, con_lik_col(0), con_lik_col(1), con_lik_col(2)]
        )
        check_col_names(df)  # should not raise

    def test_ignores_non_likelihood_columns(self):
        # Columns like "chosen" or "count_r0" should not be treated as likelihood columns
        df = pd.DataFrame(columns=["chosen", "count_r0", lik_col(0), lik_col(1)])
        check_col_names(df)  # should not raise

    def test_non_sequential_raises(self):
        df = pd.DataFrame(columns=[PARTICLE, SCORE, lik_col(0), lik_col(2)])
        with pytest.raises(ValueError, match="col indices not increasing"):
            check_col_names(df)


class TestIWMCIOverlapEst:
    def test_safe_proposal_exact(self):
        LRs = np.array([0.5, 1.5, 2.0, 3.0])
        unconstrained = np.array([1.0, 2.0, 3.0, 4.0])
        safe = np.array([2.0, 1.0, 1.5, 1.0])
        result = iwmci_overlap_est(
            LRs, unconstrained, safe, beta_t=2.0, psi_t=1.0, proposal="safe"
        )
        # constrained_density = min(safe * 2, unconstrained) = [1, 2, 3, 2]
        # ratio = min([1,2,3,2], [2,1,1.5,1]) / [2,1,1.5,1] = [0.5, 1, 1, 1]
        # mean = 0.875
        np.testing.assert_almost_equal(result, 0.875)

    def test_unconstrained_proposal_exact(self):
        LRs = np.array([0.5, 1.5, 2.0, 3.0])
        unconstrained = np.array([1.0, 2.0, 3.0, 4.0])
        safe = np.array([2.0, 1.0, 1.5, 1.0])
        result = iwmci_overlap_est(
            LRs, unconstrained, safe, beta_t=2.0, psi_t=1.0, proposal="unconstrained"
        )
        # constrained_density = min(safe * 2, unconstrained) = [1, 2, 3, 2]
        # ratio = min([1,2,3,2], [1,2,3,4]) / [1,2,3,4] = [1, 1, 1, 0.5]
        # mean = 0.875
        np.testing.assert_almost_equal(result, 0.875)

    def test_invalid_proposal_raises(self):
        LRs = np.array([1.0])
        with pytest.raises(ValueError, match="proposal name not recognized"):
            iwmci_overlap_est(LRs, LRs, LRs, 1.0, 1.0, proposal="invalid")
