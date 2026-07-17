"""
Tests for Beijing Air Quality PM2.5 trajectory inference module.

Run with: pytest tests/test_beijingair.py -v

Author(s): Raghav Kansal
"""

from collections import OrderedDict

import numpy as np
import pytest
import torch

from experiments.beijingair import BeijingTrainer, plotting
from experiments.beijingair import data as dataset

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def device():
    """Get device for tests."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def synthetic_beijing_data():
    """Create synthetic Beijing PM2.5 data for tests."""
    np.random.seed(42)
    n_quarters = 13

    marginals = []
    for q in range(n_quarters):
        # Simulate seasonal variation
        base_mean = 50 + 20 * np.sin(2 * np.pi * q / 4)
        n_samples = np.random.randint(200, 400)
        data = np.random.exponential(base_mean, (n_samples, 1)).astype(np.float32)
        marginals.append(data)

    return marginals


@pytest.fixture(scope="module")
def beijing_marginals_dict(synthetic_beijing_data):
    """Convert synthetic data to dictionary format."""
    return {i: torch.tensor(m, dtype=torch.float32) for i, m in enumerate(synthetic_beijing_data)}


@pytest.fixture(scope="module")
def simple_model(device):
    """Create a simple OTPFM model for testing."""
    from otpfm import OTPFM
    from otpfm.potentials import W2InfPotential as IndependentPotential

    dim = 1
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
    """Tests for Beijing dataset loading and preprocessing."""

    def test_default_times_constants(self):
        """Test default train/holdout time constants."""
        # Check no overlap between train and holdout
        assert len(set(dataset.DEFAULT_TRAIN_TIMES) & set(dataset.DEFAULT_HOLDOUT_TIMES)) == 0

        # Check all times covered
        all_covered = set(dataset.DEFAULT_TRAIN_TIMES) | set(dataset.DEFAULT_HOLDOUT_TIMES)
        assert all_covered == set(dataset.ALL_TIMES)

        # Check expected holdout pattern
        assert dataset.DEFAULT_HOLDOUT_TIMES == [2, 5, 8, 11]

    def test_beijing_multi_marginal_dataset(self, synthetic_beijing_data):
        """Test BeijingMultiMarginalDataset creation."""
        train_times = dataset.DEFAULT_TRAIN_TIMES

        ds = dataset.BeijingMultiMarginalDataset(
            synthetic_beijing_data,
            train_times=train_times,
        )

        # Check dataset length
        assert len(ds) > 0

        # Check sample format
        sample = ds[0]
        assert len(sample) == len(train_times)
        assert all(s.shape == (1,) for s in sample)  # 1D PM2.5

    def test_beijing_multi_marginal_dataset_iteration(self, synthetic_beijing_data):
        """Test iterating through dataset."""
        train_times = dataset.DEFAULT_TRAIN_TIMES

        ds = dataset.BeijingMultiMarginalDataset(
            synthetic_beijing_data,
            train_times=train_times,
        )

        # Iterate through a few samples
        for i, sample in enumerate(ds):
            if i >= 5:
                break
            assert len(sample) == len(train_times)

    def test_create_beijing_dataloaders(self, synthetic_beijing_data):
        """Test dataloader creation."""
        train_times = dataset.DEFAULT_TRAIN_TIMES

        train_loader, val_loader = dataset.create_beijing_dataloaders(
            synthetic_beijing_data,
            holdout_times=dataset.DEFAULT_HOLDOUT_TIMES,
            batch_size=32,
            val_split=0.2,
        )

        assert len(train_loader) > 0
        assert len(val_loader) > 0

        # Check batch format
        batch = next(iter(train_loader))
        assert len(batch) == len(train_times)
        assert batch[0].shape[1] == 1  # 1D

    def test_compute_beijing_ot_alignments(self, synthetic_beijing_data):
        """Test computing OT alignments for consecutive training pairs."""
        train_times = dataset.DEFAULT_TRAIN_TIMES

        alignments = dataset.compute_beijing_ot_alignments(
            synthetic_beijing_data,
            train_times=train_times,
            method="emd",
        )

        # Should have alignment for each consecutive pair
        assert len(alignments) == len(train_times) - 1

        # Check alignment format
        for (t_src, t_tgt), mapping in alignments.items():
            assert isinstance(mapping, np.ndarray)
            assert mapping.dtype == np.int64

    def test_dataset_with_ot_coupling(self, synthetic_beijing_data):
        """Test dataset with OT coupling enabled."""
        train_times = dataset.DEFAULT_TRAIN_TIMES

        alignments = dataset.compute_beijing_ot_alignments(
            synthetic_beijing_data,
            train_times=train_times,
        )

        ds = dataset.BeijingMultiMarginalDataset(
            synthetic_beijing_data,
            train_times=train_times,
            ot_alignments=alignments,
        )

        assert ds.use_ot
        assert len(ds) > 0

        # Check sample
        sample = ds[0]
        assert len(sample) == len(train_times)

    def test_dataset_reshuffle(self, synthetic_beijing_data):
        """Test dataset reshuffling."""
        train_times = dataset.DEFAULT_TRAIN_TIMES

        ds = dataset.BeijingMultiMarginalDataset(
            synthetic_beijing_data,
            train_times=train_times,
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
    """Tests for Beijing plotting functions."""

    def test_plot_trajectories(self, beijing_marginals_dict, tmp_path):
        """Test trajectory plotting."""
        # Create dummy trajectories
        n_steps = 20
        n_samples = 50
        trajectories = torch.randn(n_steps, n_samples, 1)
        t_eval = np.linspace(0, 1, n_steps)

        save_path = tmp_path / "trajectories.pdf"

        plotting.plot_trajectories(
            trajectories=trajectories,
            t_eval=t_eval,
            ground_truth_marginals=beijing_marginals_dict,
            plot_times=dataset.DEFAULT_TRAIN_TIMES,
            num_trajectories=20,
            save_path=save_path,
            show=False,
        )

        assert save_path.exists()

    def test_plot_scatter(self, beijing_marginals_dict, tmp_path):
        """Test PM2.5 scatter plot."""
        save_path = tmp_path / "scatter.pdf"

        plotting.plot_scatter(
            marginals=beijing_marginals_dict,
            times=dataset.DEFAULT_TRAIN_TIMES,
            save_path=save_path,
            show=False,
        )

        assert save_path.exists()


# ============================================================================
# Trainer Tests
# ============================================================================


class TestTrainer:
    """Tests for BeijingTrainer."""

    def test_trainer_creation(
        self, simple_model, synthetic_beijing_data, beijing_marginals_dict, tmp_path, device
    ):
        """Test trainer initialization."""
        train_times = dataset.DEFAULT_TRAIN_TIMES
        holdout_times = dataset.DEFAULT_HOLDOUT_TIMES

        train_loader, val_loader = dataset.create_beijing_dataloaders(
            synthetic_beijing_data,
            holdout_times=holdout_times,
            batch_size=32,
        )

        trainer = BeijingTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            save_dir=tmp_path,
            lr=1e-3,
            epochs=2,
            marginals=beijing_marginals_dict,
            train_times=train_times,
            holdout_times=holdout_times,
            device=device,
        )

        assert trainer is not None
        assert trainer.train_times == train_times
        assert trainer.holdout_times == holdout_times

    def test_trainer_short_training(
        self, simple_model, synthetic_beijing_data, beijing_marginals_dict, tmp_path, device
    ):
        """Test running a few training steps."""
        train_times = dataset.DEFAULT_TRAIN_TIMES
        holdout_times = dataset.DEFAULT_HOLDOUT_TIMES

        train_loader, val_loader = dataset.create_beijing_dataloaders(
            synthetic_beijing_data,
            holdout_times=holdout_times,
            batch_size=32,
        )

        trainer = BeijingTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            save_dir=tmp_path,
            lr=1e-3,
            epochs=1,
            marginals=beijing_marginals_dict,
            train_times=train_times,
            holdout_times=holdout_times,
            eval_num_samples=50,
            device=device,
        )

        losses, _ = trainer.train()

        assert "train_loss" in losses
        assert len(losses["train_loss"]) > 0
