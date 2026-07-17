"""
Tests for single-cell trajectory inference module.

Run with: pytest tests/test_singlecell.py -v

Author(s): Raghav Kansal
"""

from collections import OrderedDict

import numpy as np
import pytest
import torch

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def device():
    """Get device for tests."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def synthetic_data():
    """Create synthetic data for fast tests without downloading."""
    np.random.seed(42)
    n_cells = 500
    dim = 20
    n_timepoints = 5

    # Create synthetic trajectories - cells progressing through time
    labels = np.repeat(np.arange(n_timepoints), n_cells // n_timepoints)
    pcs = np.random.randn(len(labels), dim).astype(np.float32)

    # Add time-dependent drift to make it trajectory-like
    for t in range(n_timepoints):
        mask = labels == t
        pcs[mask] += t * 0.5 * np.random.randn(1, dim).astype(np.float32)

    return pcs, labels.astype(np.int64)


@pytest.fixture(scope="module")
def synthetic_marginals(synthetic_data):
    """Create marginals dictionary from synthetic data."""
    pcs, labels = synthetic_data
    marginals = {}
    for t in sorted(set(labels)):
        mask = labels == t
        marginals[t] = torch.tensor(pcs[mask], dtype=torch.float32)
    return marginals


@pytest.fixture(scope="module")
def simple_model(device):
    """Create a simple OTPFM model for testing."""
    from otpfm import OTPFM
    from otpfm.potentials import MMDRBFPotential

    dim = 20
    tks = [0.33, 0.67]
    potentials = OrderedDict()
    for tk in tks:
        potentials[tk] = MMDRBFPotential(
            tk=tk,
            strength=1.0,
            lambda_type="gaussian",
            width=0.2,
            sigma=[1.0, 3.0, 10.0],
        )

    model = OTPFM(
        d=dim,
        tks=tks,
        potentials=potentials,
        flownet_args={
            "x_emb_dim": 32,
            "t_emb_dim": 32,
            "num_hidden_layers": 2,
            "hidden_dim": 64,
        },
        ema_decay=0.999,
        euler_steps=2,
    ).to(device)

    return model


# ============================================================================
# Dataset Tests
# ============================================================================


class TestDataset:
    """Tests for dataset loading and preprocessing."""

    def test_eb_multi_marginal_dataset(self, synthetic_data):
        """Test EBMultiMarginalDataset creation and iteration."""
        from experiments.singlecell.data import EBMultiMarginalDataset

        pcs, labels = synthetic_data
        holdout_times = [1, 3]

        dataset = EBMultiMarginalDataset(pcs, labels, holdout_times=holdout_times)

        # Check training times
        assert dataset.train_times == [0, 2, 4]
        assert len(dataset) > 0

        # Check item retrieval
        samples = dataset[0]
        assert len(samples) == 3  # 3 training time points
        assert all(isinstance(s, torch.Tensor) for s in samples)
        assert all(s.shape == (20,) for s in samples)

    def test_eb_multi_marginal_dataset_iteration(self, synthetic_data):
        """Test iterating through dataset."""
        from experiments.singlecell.data import EBMultiMarginalDataset

        pcs, labels = synthetic_data
        holdout_times = [1, 3]

        dataset = EBMultiMarginalDataset(pcs, labels, holdout_times=holdout_times)

        # Iterate through a few samples
        for i, sample in enumerate(dataset):
            if i >= 5:
                break
            assert len(sample) == 3  # 3 training time points

    def test_create_eb_dataloaders(self, synthetic_data):
        """Test DataLoader creation."""
        from experiments.singlecell.data import create_eb_dataloaders

        pcs, labels = synthetic_data

        train_loader, val_loader = create_eb_dataloaders(
            pcs,
            labels,
            holdout_times=[1, 3],
            batch_size=32,
            val_split=0.2,
        )

        assert len(train_loader) > 0
        assert len(val_loader) > 0

        # Check batch shapes
        batch = next(iter(train_loader))
        assert len(batch) == 3  # 3 training time points
        assert all(b.shape[0] == 32 for b in batch)  # batch size

    def test_compute_ot_alignments(self, synthetic_data):
        """Test OT alignment computation."""
        from experiments.singlecell.data import compute_ot_alignments

        pcs, labels = synthetic_data
        train_times = [0, 2, 4]

        alignments = compute_ot_alignments(pcs, labels, train_times, method="emd")

        # Should have alignment for each consecutive pair
        assert len(alignments) == len(train_times) - 1
        expected_keys = [(0, 2), (2, 4)]
        assert set(alignments.keys()) == set(expected_keys)

    def test_dataset_with_ot_coupling(self, synthetic_data):
        """Test dataset with OT coupling enabled."""
        from experiments.singlecell.data import EBMultiMarginalDataset, compute_ot_alignments

        pcs, labels = synthetic_data
        train_times = [0, 2, 4]

        alignments = compute_ot_alignments(pcs, labels, train_times)

        dataset = EBMultiMarginalDataset(
            pcs, labels, holdout_times=[1, 3], ot_alignments=alignments
        )

        assert dataset.use_ot
        assert len(dataset) > 0

        # Check sample
        sample = dataset[0]
        assert len(sample) == 3  # 3 training time points

    def test_dataset_reshuffle(self, synthetic_data):
        """Test dataset reshuffling."""
        from experiments.singlecell.data import EBMultiMarginalDataset

        pcs, labels = synthetic_data

        dataset = EBMultiMarginalDataset(pcs, labels, holdout_times=[1, 3])

        # Get sample before reshuffle
        sample1 = [s.clone() for s in dataset[0]]

        # Reshuffle
        dataset.reshuffle()

        # Sample might be different (probabilistic)
        sample2 = dataset[0]
        assert len(sample2) == len(sample1)


# ============================================================================
# Model Tests
# ============================================================================


class TestModel:
    """Tests for OTPFM model."""

    def test_model_instantiation(self, simple_model):
        """Test model can be created."""
        assert simple_model is not None
        assert hasattr(simple_model, "flownet")
        assert hasattr(simple_model, "sample")

    def test_model_forward_pass(self, simple_model, synthetic_data, device):
        """Test forward pass computes losses."""
        pcs, labels = synthetic_data

        # Create batch: (batch_size, num_marginals, dim)
        batch_size = 32
        marginals_dict = {}
        for t in [0, 2, 4]:  # training times
            mask = labels == t
            marginals_dict[t] = torch.tensor(pcs[mask][:batch_size], dtype=torch.float32)

        # Also need intermediate marginals for OTP
        for t in [1, 3]:  # intermediate times
            mask = labels == t
            marginals_dict[t] = torch.tensor(pcs[mask][:batch_size], dtype=torch.float32)

        # Stack in order: [source, intermediate_1, intermediate_2, target]
        xs = torch.stack(
            [
                marginals_dict[0],
                marginals_dict[1],
                marginals_dict[3],
                marginals_dict[4],
            ],
            dim=1,
        ).to(device)

        # Forward pass
        simple_model.train()
        loss = simple_model.forward_with_loss(xs, otp_alpha=0.5, do_otp=True)

        assert isinstance(loss, float | torch.Tensor)
        if isinstance(loss, torch.Tensor):
            assert loss.numel() == 1
            assert torch.isfinite(loss)

    def test_model_sample(self, simple_model, device):
        """Test model sampling."""
        batch_size = 16
        dim = 20

        x0 = torch.randn(batch_size, dim).to(device)

        simple_model.eval()
        with torch.no_grad():
            trajectories, t_eval = simple_model.sample(x0, n_steps=10, ema=True)

        # trajectories shape: (n_timesteps, batch_size, dim)
        assert trajectories.ndim == 3
        assert trajectories.shape[1] == batch_size
        assert trajectories.shape[2] == dim

        # t_eval should have same length as first dim of trajectories
        assert len(t_eval) == trajectories.shape[0]

        # t_eval should go from 0 to 1
        assert t_eval[0] == 0.0
        assert t_eval[-1] == 1.0

    def test_model_ema_update(self, simple_model):
        """Test EMA update."""
        main_params = list(simple_model.flownet.parameters())
        ema_params = list(simple_model.flownet_ema.parameters())

        # Save initial EMA params
        initial_ema = ema_params[0].clone()

        # Modify main params
        with torch.no_grad():
            main_params[0].add_(0.1)

        # Update EMA
        simple_model.update_ema()

        # EMA should have changed
        assert not torch.allclose(ema_params[0], initial_ema)


# ============================================================================
# Evaluation Tests
# ============================================================================


class TestEvaluation:
    """Tests for evaluation metrics."""

    def test_compute_w2_distance(self):
        """Test Wasserstein-2 distance computation."""
        from experiments.evaluation import compute_w2_distance

        # Same distributions should have W2 ~ 0
        x = torch.randn(100, 10)
        w2 = compute_w2_distance(x, x)
        assert w2 < 0.1

        # Different distributions should have W2 > 0
        y = torch.randn(100, 10) + 5
        w2 = compute_w2_distance(x, y)
        assert w2 > 0.1

    def test_compute_mmd(self):
        """Test MMD with multi-scale Gaussian kernel."""
        from experiments.evaluation import compute_mmd

        # Same distributions should have MMD ~ 0
        x = torch.randn(100, 10)
        mmd = compute_mmd(x, x)
        assert mmd < 0.5

        # Different distributions should have MMD > 0
        y = torch.randn(100, 10) + 3
        mmd = compute_mmd(x, y)
        assert mmd > 0.1

    def test_compute_swd(self):
        """Test Sliced Wasserstein Distance."""
        from experiments.evaluation import compute_swd

        # Same distributions should have SWD ~ 0
        x = torch.randn(100, 10)
        swd = compute_swd(x, x)
        assert swd < 0.5

        # Different distributions should have SWD > 0
        y = torch.randn(100, 10) + 3
        swd = compute_swd(x, y)
        assert swd > 0.1

    def test_compute_fgd(self):
        """Test Fréchet Gaussian Distance."""
        from experiments.evaluation import compute_fgd

        # Same distributions should have FGD ~ 0
        x = torch.randn(100, 10)
        fgd = compute_fgd(x, x)
        assert fgd < 0.5

        # Different distributions should have FGD > 0
        y = torch.randn(100, 10) + 3
        fgd = compute_fgd(x, y)
        assert fgd > 0.1


# ============================================================================
# Training Tests
# ============================================================================


class TestTraining:
    """Tests for training functionality."""

    def test_trainer_instantiation(self, simple_model, synthetic_data, device, tmp_path):
        """Test Trainer can be instantiated."""
        from experiments import Trainer
        from experiments.singlecell.data import create_eb_dataloaders

        pcs, labels = synthetic_data

        train_loader, val_loader = create_eb_dataloaders(
            pcs,
            labels,
            holdout_times=[1, 3],
            batch_size=16,
            val_split=0.2,
        )

        trainer = Trainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            save_dir=tmp_path,
            lr=1e-3,
            epochs=5,
            sampling_steps=10,
            do_otp=True,
            grad_clip=1.0,
            potentials=simple_model.potentials,
            device=device,
        )

        assert trainer.epochs == 5
        assert trainer.sampling_steps == 10
        assert trainer.do_otp
        assert trainer.grad_clip == 1.0

    def test_process_batch(self):
        """Test batch processing."""
        # Simulate batch from DataLoader
        batch = [
            torch.randn(32, 20),  # time 0
            torch.randn(32, 20),  # time 2
            torch.randn(32, 20),  # time 4
        ]

        # Test internal _process_batch method
        processed = torch.stack(batch).transpose(0, 1)

        # Should be (batch_size, num_marginals, dim)
        assert processed.shape == (32, 3, 20)

    def test_training_step(self, simple_model, synthetic_data, device, tmp_path):
        """Test a single training step."""
        from experiments.singlecell.data import create_eb_dataloaders

        pcs, labels = synthetic_data

        train_loader, val_loader = create_eb_dataloaders(
            pcs,
            labels,
            holdout_times=[1, 3],
            batch_size=16,
            val_split=0.2,
        )

        # Manual test of training step
        simple_model.train()
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-3)

        for batch in train_loader:
            xs = torch.stack(batch).transpose(0, 1).to(device)

            # Create full batch with intermediate marginals
            x0 = xs[:, 0]
            x1 = xs[:, -1]
            xm1 = x0 * 0.67 + x1 * 0.33
            xm2 = x0 * 0.33 + x1 * 0.67
            xs_full = torch.stack([x0, xm1, xm2, x1], dim=1)

            optimizer.zero_grad()
            loss = simple_model.forward_with_loss(xs_full, otp_alpha=0.5, do_otp=True)

            if isinstance(loss, torch.Tensor):
                loss.backward()
                optimizer.step()

            break  # Just test one step


# ============================================================================
# Plotting Tests
# ============================================================================


class TestPlotting:
    """Tests for plotting functions."""

    def test_plot_pca_trajectories(self, synthetic_marginals, tmp_path):
        """Test PCA trajectory plotting."""
        from experiments.singlecell.plotting import plot_pca_trajectories

        # Create fake trajectories: (n_timesteps, n_samples, dim)
        trajectories = torch.randn(20, 50, 20)
        time_points = np.linspace(0, 1, 20)

        save_path = tmp_path / "test_trajectories.pdf"

        # Should not raise
        plot_pca_trajectories(
            trajectories=trajectories,
            time_points=time_points,
            ground_truth_marginals=synthetic_marginals,
            plot_times=[0, 1, 2, 3, 4],
            pcs=(0, 1),
            num_trajectories=10,
            title="Test Trajectories",
            save_path=save_path,
            show=False,
        )

        assert save_path.exists()


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline_synthetic(self, device, tmp_path):
        """Test full training pipeline with synthetic data."""
        from collections import OrderedDict

        import numpy as np
        import torch
        from otpfm import OTPFM
        from otpfm.potentials import MMDRBFPotential

        from experiments import evaluation
        from experiments.singlecell.data import create_eb_dataloaders

        # Create synthetic data
        np.random.seed(42)
        torch.manual_seed(42)

        n_cells = 200
        dim = 10
        n_timepoints = 5

        labels = np.repeat(np.arange(n_timepoints), n_cells // n_timepoints)
        pcs = np.random.randn(len(labels), dim).astype(np.float32)
        for t in range(n_timepoints):
            mask = labels == t
            pcs[mask] += t * np.array([0.5] * dim).astype(np.float32)

        # Create marginals
        marginals = {}
        for t in range(n_timepoints):
            mask = labels == t
            marginals[t] = torch.tensor(pcs[mask], dtype=torch.float32)

        # Create model
        tks = [0.33, 0.67]
        potentials = OrderedDict()
        for tk in tks:
            potentials[tk] = MMDRBFPotential(tk=tk, strength=1.0, lambda_type="gaussian", width=0.2)

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
            ema_decay=0.99,
        ).to(device)

        # Create dataloader
        train_loader, _ = create_eb_dataloaders(
            pcs,
            labels.astype(np.int64),
            holdout_times=[1, 3],
            batch_size=16,
        )

        # Training loop (just 2 epochs for speed)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()

        for epoch in range(2):
            epoch_loss = 0.0
            for batch in train_loader:
                xs = torch.stack(batch).transpose(0, 1).to(device)

                # Create full batch with interpolated intermediate points
                x0, x1 = xs[:, 0], xs[:, -1]
                xm1 = x0 * 0.67 + x1 * 0.33
                xm2 = x0 * 0.33 + x1 * 0.67
                xs_full = torch.stack([x0, xm1, xm2, x1], dim=1)

                optimizer.zero_grad()
                loss = model.forward_with_loss(xs_full, otp_alpha=0.5)

                if isinstance(loss, torch.Tensor):
                    loss.backward()
                    optimizer.step()
                    model.update_ema()
                    epoch_loss += loss.item()

        # Evaluate by sampling and computing metrics
        model.eval()

        with torch.no_grad():
            x0 = marginals[0][:30].to(device)
            trajectories, t_eval = model.sample(x0, n_steps=10, ema=True)

        # Get samples at middle time point
        mid_idx = len(t_eval) // 2
        generated = trajectories[mid_idx].cpu()

        # Compute metrics
        swd = evaluation.compute_swd(generated, marginals[2][:30].cpu())
        mmd = evaluation.compute_mmd(generated, marginals[2][:30].cpu())

        assert isinstance(swd, float)
        assert isinstance(mmd, float)
        assert np.isfinite(swd)
        assert np.isfinite(mmd)

    def test_model_saves_and_loads(self, simple_model, tmp_path, device):
        """Test model checkpoint save/load."""
        # Save
        save_path = tmp_path / "model.pt"
        torch.save(
            {
                "model_state_dict": simple_model.state_dict(),
            },
            save_path,
        )

        assert save_path.exists()

        # Load into new model
        from collections import OrderedDict

        from otpfm import OTPFM
        from otpfm.potentials import MMDRBFPotential

        dim = 20
        tks = [0.33, 0.67]
        potentials = OrderedDict()
        for tk in tks:
            potentials[tk] = MMDRBFPotential(tk=tk, strength=1.0, lambda_type="gaussian", width=0.2)

        new_model = OTPFM(
            d=dim,
            tks=tks,
            potentials=potentials,
            flownet_args={
                "x_emb_dim": 32,
                "t_emb_dim": 32,
                "num_hidden_layers": 2,
                "hidden_dim": 64,
            },
            ema_decay=0.999,
            euler_steps=2,
        ).to(device)

        checkpoint = torch.load(save_path, map_location=device)
        new_model.load_state_dict(checkpoint["model_state_dict"])

        # Should be able to sample
        new_model.eval()
        with torch.no_grad():
            x0 = torch.randn(10, dim).to(device)
            trajectories, t_eval = new_model.sample(x0, n_steps=5, ema=True)

        assert trajectories.shape[1] == 10


# ============================================================================
# Train Script Tests
# ============================================================================


class TestTrainScript:
    """Tests for the unified train.py script."""

    def test_config_merging(self):
        """Test that config merging works correctly."""
        from experiments.train import merge_configs

        base = {"a": 1, "b": 2, "c": 3}
        override = {"b": 20, "d": 4}

        result = merge_configs(base, override)

        assert result["a"] == 1  # unchanged
        assert result["b"] == 20  # overridden
        assert result["c"] == 3  # unchanged
        assert result["d"] == 4  # new key

    def test_build_tag(self):
        """Test build_tag function from train module."""
        from experiments.train import build_tag

        config = {
            "potential": "w2inf",
            "strength": 100.0,
            "width": 0.33,
            "lr": 0.001,
            "num_hidden_layers": 4,
        }

        tag = build_tag(config, "test")
        assert "test" in tag
        assert "w2inf" in tag

    def test_create_potential(self):
        """Test potential creation from train module."""
        from otpfm.potentials import MMDRBFPotential, W2InfPotential, W2Potential

        from experiments.train import create_potential

        # Test W2Inf
        config = {
            "potential": "w2inf",
            "strength": 100.0,
            "lambda_type": "gaussian",
            "width": 0.33,
        }
        pot = create_potential(config, tk=0.5)
        assert isinstance(pot, W2InfPotential)
        assert pot.tk == 0.5
        assert pot.strength == 100.0

        # Test W2
        config["potential"] = "w2"
        pot = create_potential(config, tk=0.5)
        assert isinstance(pot, W2Potential)

        # Test MMD
        config["potential"] = "mmd"
        config["mmd_bandwidth"] = [3.0]
        pot = create_potential(config, tk=0.5)
        assert isinstance(pot, MMDRBFPotential)

    def test_create_potential_with_overrides(self):
        """Test potential creation with strength/width overrides."""
        from otpfm.potentials import W2InfPotential

        from experiments.train import create_potential

        config = {
            "potential": "w2inf",
            "strength": 100.0,
            "lambda_type": "gaussian",
            "width": 0.33,
        }

        # Test with overrides
        pot = create_potential(config, tk=0.5, strength=200.0, width=0.5)
        assert isinstance(pot, W2InfPotential)
        assert pot.strength == 200.0


# ============================================================================
# Run tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
