"""
Tests for Gulf of Mexico ocean currents trajectory inference module.

Run with: pytest tests/test_gulfofmexico.py -v

Author(s): Raghav Kansal
"""

from collections import OrderedDict

import numpy as np
import pytest
import torch

from experiments.gulfofmexico import GoMTrainer, plotting
from experiments.gulfofmexico import data as dataset

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def device():
    """Get device for tests."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def synthetic_gom_data():
    """Create synthetic GoM data for tests."""
    # Generate simple 2D Gaussian data for each time point
    n_samples = 100
    n_times = 10
    marginals = []
    for t in range(n_times):
        # Create data with slight drift over time
        center = np.array([t * 0.1, t * 0.05])
        data = np.random.randn(n_samples, 2).astype(np.float32) + center
        marginals.append(data)
    return marginals


@pytest.fixture(scope="module")
def gom_marginals_dict(synthetic_gom_data):
    """Convert synthetic data to dictionary format."""
    return {i: torch.tensor(m, dtype=torch.float32) for i, m in enumerate(synthetic_gom_data)}


@pytest.fixture(scope="module")
def simple_model(device):
    """Create a simple OTPFM model for testing."""
    from otpfm import OTPFM
    from otpfm.potentials import W2InfPotential as IndependentPotential

    dim = 2
    tks = [0.5]
    potentials = OrderedDict(
        [(tk, IndependentPotential(tk=tk, strength=5.0, width=0.1)) for tk in tks]
    )

    model = OTPFM(
        d=dim,
        tks=tks,
        potentials=potentials,
        flownet_args={
            "x_emb_dim": 16,
            "t_emb_dim": 16,
            "num_hidden_layers": 1,
            "hidden_dim": 32,
        },
        ema_decay=0.999,
        euler_steps=2,
    ).to(device)

    return model


# ============================================================================
# Dataset Tests
# ============================================================================


class TestDataset:
    """Tests for GoM dataset loading and preprocessing."""

    def test_gom_multi_marginal_dataset(self, synthetic_gom_data):
        """Test GoMMultiMarginalDataset creation."""
        ds = dataset.GoMMultiMarginalDataset(
            synthetic_gom_data,
            holdout_times=[1, 3, 5, 7],
        )

        # Check train times (with 10 times and holdout [1,3,5,7], train is [0,2,4,6,8,9])
        assert ds.train_times == [0, 2, 4, 6, 8, 9]
        assert ds.holdout_times == [1, 3, 5, 7]

        # Check dataset length
        assert len(ds) > 0

        # Check sample format
        sample = ds[0]
        assert len(sample) == 6  # 6 training times
        assert all(s.shape == (2,) for s in sample)

    def test_gom_multi_marginal_dataset_iteration(self, synthetic_gom_data):
        """Test iterating through dataset."""
        ds = dataset.GoMMultiMarginalDataset(
            synthetic_gom_data,
            holdout_times=[1, 3, 5, 7],
        )

        # Iterate through a few samples
        for i, sample in enumerate(ds):
            if i >= 5:
                break
            assert len(sample) == 6  # 6 training times

    def test_create_gom_dataloaders(self, synthetic_gom_data):
        """Test dataloader creation."""
        train_loader, val_loader = dataset.create_gom_dataloaders(
            synthetic_gom_data,
            holdout_times=[1, 3, 5, 7],
            batch_size=16,
            val_split=0.2,
        )

        assert len(train_loader) > 0
        assert len(val_loader) > 0

        # Check batch format
        batch = next(iter(train_loader))
        assert len(batch) == 6  # 6 training times (0,2,4,6,8,9)
        assert batch[0].shape[1] == 2  # 2D

    def test_compute_gom_ot_alignments(self, synthetic_gom_data):
        """Test computing OT alignments for all consecutive pairs."""
        train_times = [0, 2, 4, 6, 8]
        alignments = dataset.compute_gom_ot_alignments(
            synthetic_gom_data,
            train_times=train_times,
            method="emd",
        )

        # Should have alignment for each consecutive pair
        assert len(alignments) == len(train_times) - 1

        # Check keys
        expected_keys = [(0, 2), (2, 4), (4, 6), (6, 8)]
        assert set(alignments.keys()) == set(expected_keys)

        # Check alignment format
        for (t_src, t_tgt), mapping in alignments.items():
            assert isinstance(mapping, np.ndarray)
            assert mapping.dtype == np.int64

    def test_dataset_with_ot_coupling(self, synthetic_gom_data):
        """Test dataset with OT coupling enabled."""
        # With holdout [1,3,5,7], train times are [0,2,4,6,8,9]
        train_times = [0, 2, 4, 6, 8, 9]
        alignments = dataset.compute_gom_ot_alignments(
            synthetic_gom_data,
            train_times=train_times,
        )

        ds = dataset.GoMMultiMarginalDataset(
            synthetic_gom_data,
            holdout_times=[1, 3, 5, 7],
            ot_alignments=alignments,
        )

        assert ds.use_ot
        assert len(ds) > 0

        # Check sample
        sample = ds[0]
        assert len(sample) == 6  # 6 training times

    def test_dataset_reshuffle(self, synthetic_gom_data):
        """Test dataset reshuffling."""
        ds = dataset.GoMMultiMarginalDataset(
            synthetic_gom_data,
            holdout_times=[1, 3, 5, 7],
        )

        # Get sample before reshuffle
        sample1 = [s.clone() for s in ds[0]]

        # Reshuffle
        ds.reshuffle()

        # Sample might be different (probabilistic)
        # Just check it doesn't crash
        sample2 = ds[0]
        assert len(sample2) == len(sample1)


# ============================================================================
# Plotting Tests
# ============================================================================


class TestPlotting:
    """Tests for GoM plotting functions."""

    def test_plot_trajectories(self, gom_marginals_dict, tmp_path):
        """Test 2D trajectory plot creation."""
        # Create dummy trajectories
        n_steps = 20
        n_samples = 50
        trajectories = torch.randn(n_steps, n_samples, 2)
        t_eval = np.linspace(0, 1, n_steps)

        save_path = tmp_path / "trajectories.pdf"

        plotting.plot_trajectories(
            trajectories=trajectories,
            t_eval=t_eval,
            ground_truth_marginals=gom_marginals_dict,
            num_trajectories=20,
            save_path=save_path,
            show=False,
        )

        assert save_path.exists()


# ============================================================================
# Trainer Tests
# ============================================================================


class TestTrainer:
    """Tests for GoMTrainer."""

    def test_trainer_creation(
        self, simple_model, synthetic_gom_data, gom_marginals_dict, tmp_path, device
    ):
        """Test trainer initialization."""
        train_loader, val_loader = dataset.create_gom_dataloaders(
            synthetic_gom_data,
            holdout_times=[1, 3, 5, 7],
            batch_size=16,
        )

        trainer = GoMTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            save_dir=tmp_path,
            lr=1e-3,
            epochs=2,
            marginals=gom_marginals_dict,
            train_times=[0, 2, 4, 6, 8, 9],
            holdout_times=[1, 3, 5, 7],
            device=device,
        )

        assert trainer is not None
        assert trainer.train_times == [0, 2, 4, 6, 8, 9]
        assert trainer.holdout_times == [1, 3, 5, 7]

    def test_trainer_short_training(
        self, simple_model, synthetic_gom_data, gom_marginals_dict, tmp_path, device
    ):
        """Test running a few training steps."""
        train_loader, val_loader = dataset.create_gom_dataloaders(
            synthetic_gom_data,
            holdout_times=[1, 3, 5, 7],
            batch_size=16,
        )

        trainer = GoMTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            save_dir=tmp_path,
            lr=1e-3,
            epochs=1,
            marginals=gom_marginals_dict,
            train_times=[0, 2, 4, 6, 8, 9],
            holdout_times=[1, 3, 5, 7],
            eval_num_samples=50,
            device=device,
        )

        losses, _ = trainer.train()

        assert "train_loss" in losses
        assert len(losses["train_loss"]) > 0
