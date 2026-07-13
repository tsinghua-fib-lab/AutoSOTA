"""Integration tests for the benchmark pipeline.

Tests cover:
- Posterior class imports from posteriors/
- Trainer registry completeness
- Simulation trainers (NPE, FMPE) on tiny data
- Baseline trainers (DPE) on calibration data
- Calibration trainers (fm_post_transform_plug_y) with pretrained NPE
- Posterior sampling shapes
"""

import pytest
import torch

from posteriors import (
    DirectFlowMatchingPosterior,
    DirectPosteriorEstimator,
    DualFlowPosteriorEstimator,
    Estimator,
    EstimatorConfig,
    FlowMatchingPosterior,
    NPEPFNPosterior,
    Posterior,
    RopePosterior,
    UMNNFlow,
    get_build_fn,
)
from training.base import BaseTrainer, TrainerConfig
from training.registry import TrainerRegistry

# Ensure all trainers are registered by importing the modules
import training.trainers.simulation  # noqa: F401
import training.trainers.baselines  # noqa: F401
import training.trainers.calibration  # noqa: F401


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def dummy_sim_data():
    """Simulation data: (theta, x) pairs."""
    torch.manual_seed(42)
    theta = torch.randn(80, 2)
    x = torch.randn(80, 3)
    return theta, x


@pytest.fixture(scope="module")
def dummy_cal_data():
    """Calibration data: (theta, x, y) triples."""
    torch.manual_seed(42)
    theta = torch.randn(80, 2)
    x = torch.randn(80, 3)
    y = torch.randn(80, 3)
    return theta, x, y


@pytest.fixture
def temp_model_path(tmp_path):
    """Temporary directory for model saving."""
    return tmp_path


@pytest.fixture
def minimal_task_config():
    """Minimal task config dict with tiny architectures."""
    return {
        "name": "gaussian",
        "npe": {
            "params": {
                "density_estimator": "maf",
                "npe_params": {"hidden_features": [16]},
                "embedding_net": {"type": "identity"},
            },
        },
        "fmpe": {
            "config": {
                "probability_path": "ot2",
                "prior": "power",
                "base_dist": "gaussian",
                "params": {
                    "drift": {"type": "mlp", "hidden_dims": [16, 16]},
                },
            },
        },
        "fm_post_transform": {
            "config": {
                "npe": "fmpe",
                "flow_theta": {
                    "probability_path": "ot2",
                    "prior": "power",
                    "base_dist": "gaussian",
                    "params": {
                        "drift": {"type": "mlp", "hidden_dims": [16, 16]},
                    },
                },
                "flow_x": {
                    "probability_path": "ot2",
                    "prior": "power",
                    "base_dist": "gaussian",
                    "params": {
                        "drift": {"type": "mlp", "hidden_dims": [16, 16]},
                    },
                },
            },
        },
    }


@pytest.fixture
def minimal_trainer_config():
    """Minimal TrainerConfig for fast training."""
    return TrainerConfig(
        epochs=3,
        lr=1e-3,
        batch_size=32,
        train_size=0.8,
        max_patience=None,
        rescale="z_score",
        save=False,
        load=False,
    )


# =============================================================================
# Test 1: Posteriors Imports
# =============================================================================


class TestPosteriorsImports:
    """Verify all public exports from posteriors/."""

    def test_posterior_base_class(self):
        assert Posterior is not None
        assert isinstance(Posterior, type)

    def test_estimator(self):
        assert Estimator is not None
        assert isinstance(Estimator, type)

    def test_estimator_config(self):
        assert EstimatorConfig is not None

    def test_direct_posterior_estimator(self):
        assert DirectPosteriorEstimator is not None
        assert issubclass(DirectPosteriorEstimator, Posterior)

    def test_direct_flow_matching_posterior(self):
        assert DirectFlowMatchingPosterior is not None
        assert issubclass(DirectFlowMatchingPosterior, Posterior)

    def test_dual_flow_posterior_estimator(self):
        assert DualFlowPosteriorEstimator is not None
        assert issubclass(DualFlowPosteriorEstimator, Posterior)

    def test_flow_matching_posterior(self):
        assert FlowMatchingPosterior is not None
        assert issubclass(FlowMatchingPosterior, Posterior)

    def test_rope_posterior(self):
        assert RopePosterior is not None
        assert issubclass(RopePosterior, Posterior)

    def test_npe_pfn_posterior(self):
        assert NPEPFNPosterior is not None
        assert issubclass(NPEPFNPosterior, Posterior)

    def test_umnn_flow(self):
        assert UMNNFlow is not None

    def test_get_build_fn(self):
        assert get_build_fn is not None
        assert callable(get_build_fn)


# =============================================================================
# Test 2: Trainer Registry
# =============================================================================


class TestTrainerRegistry:
    """Verify all registered trainers are discoverable."""

    @pytest.mark.parametrize(
        "name",
        [
            "npe",
            "fmpe",
            "rope",
            "fm_post_transform",
            "fm_post_transform_only_ut",
            "fm_post_transform_plug_y",
            "fmcpe_fm",
            "dpe",
            "mf_npe",
            "fmpe_cal",
            "npe_rs",
        ],
    )
    def test_trainer_registered(self, name):
        trainer_cls = TrainerRegistry.get(name)
        assert trainer_cls is not None, f"Trainer '{name}' not found in registry"
        assert issubclass(trainer_cls, BaseTrainer)

    def test_simulation_category(self):
        sim_trainers = TrainerRegistry.list_trainers(category="simulation")
        assert "npe" in sim_trainers
        assert "fmpe" in sim_trainers

    def test_calibration_category(self):
        cal_trainers = TrainerRegistry.list_trainers(category="calibration")
        for name in ["fm_post_transform", "fm_post_transform_only_ut",
                      "fm_post_transform_plug_y", "fmcpe_fm"]:
            assert name in cal_trainers, f"'{name}' not in calibration category"

    def test_baseline_category(self):
        base_trainers = TrainerRegistry.list_trainers(category="baseline")
        for name in ["dpe", "mf_npe", "fmpe_cal", "npe_rs", "rope"]:
            assert name in base_trainers, f"'{name}' not in baseline category"


# =============================================================================
# Test 3: Simulation Trainers
# =============================================================================


class TestSimulationTrainers:
    """NPE and FMPE train on tiny data, return correct posterior types."""

    def test_npe_trainer(
        self, dummy_sim_data, temp_model_path, minimal_task_config, minimal_trainer_config
    ):
        from training.trainers.simulation import NPETrainer

        trainer = NPETrainer(
            minimal_trainer_config, temp_model_path, minimal_task_config
        )
        theta, x = dummy_sim_data
        posterior = trainer.train((theta, x), device=torch.device("cpu"))

        assert isinstance(posterior, DirectPosteriorEstimator)

    def test_fmpe_trainer(
        self, dummy_sim_data, temp_model_path, minimal_task_config, minimal_trainer_config
    ):
        from training.trainers.simulation import FMPETrainer

        trainer = FMPETrainer(
            minimal_trainer_config, temp_model_path, minimal_task_config
        )
        theta, x = dummy_sim_data
        posterior = trainer.train((theta, x), device=torch.device("cpu"))

        assert isinstance(posterior, DirectFlowMatchingPosterior)


# =============================================================================
# Test 4: Baseline Trainers
# =============================================================================


class TestBaselineTrainers:
    """DPE trains on calibration triples."""

    def test_dpe_trainer(
        self, dummy_cal_data, temp_model_path, minimal_task_config, minimal_trainer_config
    ):
        from training.trainers.baselines import DPETrainer

        trainer = DPETrainer(
            minimal_trainer_config, temp_model_path, minimal_task_config
        )
        posterior = trainer.train(dummy_cal_data, device=torch.device("cpu"))

        assert isinstance(posterior, DirectPosteriorEstimator)


# =============================================================================
# Test 5: Calibration Trainers
# =============================================================================


class TestCalibrationTrainers:
    """fm_post_transform_plug_y trains with pre-trained NPE."""

    def test_fm_post_transform_plug_y(
        self, dummy_cal_data, temp_model_path, minimal_task_config, minimal_trainer_config
    ):
        from training.trainers.calibration import FMPostTransformPlugYTrainer
        from training.trainers.simulation import FMPETrainer

        theta, x, y = dummy_cal_data

        # Train an FMPE to serve as base NPE (using the real trainer pipeline)
        fmpe_config = TrainerConfig(
            epochs=2, lr=1e-3, batch_size=32, train_size=0.8,
            max_patience=None, rescale="z_score", save=True, load=False,
        )
        fmpe_trainer = FMPETrainer(fmpe_config, temp_model_path, minimal_task_config)
        fmpe_trainer.train((theta, x), device=torch.device("cpu"))

        # The FMPETrainer saves as {name}.pth -> fmpe.pth; rename to npe_fmpe.pth
        import shutil
        shutil.copy(temp_model_path / "fmpe.pth", temp_model_path / "npe_fmpe.pth")

        # Save the FMPE config as JSON
        import json
        fmpe_cfg = minimal_task_config["fmpe"]["config"]
        with open(temp_model_path / "npe_fmpe.json", "w") as f:
            json.dump(fmpe_cfg, f)

        trainer = FMPostTransformPlugYTrainer(
            minimal_trainer_config, temp_model_path, minimal_task_config
        )
        posterior = trainer.train((theta, x, y), device=torch.device("cpu"))

        assert isinstance(posterior, DualFlowPosteriorEstimator)


# =============================================================================
# Test 6: Posterior Sampling
# =============================================================================


class TestPosteriorSampling:
    """Verify sample output shapes from trained posteriors."""

    def test_direct_posterior_estimator_sample_shape(self, dummy_sim_data):
        theta, x = dummy_sim_data
        model = Estimator(
            "gaussian",
            theta.shape[1:],
            x.shape[1:],
            density_estimator="maf",
            npe_params={"hidden_features": [16]},
            embedding_net={"type": "identity"},
        )
        model.set_scales(theta, x, "z_score")

        posterior = DirectPosteriorEstimator("test", model)
        nsamples = 5
        n_obs = 10
        test_x = x[:n_obs]

        samples = posterior.sample(test_x, nsamples, torch.device("cpu"))
        assert samples.shape == (nsamples, n_obs, theta.shape[1])

    def test_direct_flow_matching_posterior_sample_shape(self, dummy_sim_data):
        from flow_matching.torch_flow import FlowMatching

        theta, x = dummy_sim_data
        flow = FlowMatching(
            "ot2", "power", "gaussian",
            theta.shape[1:],
            cond_dim=x.shape[1:],
            drift={"type": "mlp", "hidden_dims": [16, 16]},
        )
        flow.set_scales(theta, x, "z_score")

        posterior = DirectFlowMatchingPosterior("test", flow)
        nsamples = 5
        n_obs = 10
        test_x = x[:n_obs]

        samples = posterior.sample(test_x, nsamples, torch.device("cpu"))
        assert samples.shape == (nsamples, n_obs, theta.shape[1])
