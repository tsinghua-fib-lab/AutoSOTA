"""
Tests for Gaussian OTP-FM experiments.

Run with: pytest tests/test_gaussian.py -v

Author(s): Raghav Kansal
"""

from collections import OrderedDict

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def device():
    """Get device for tests."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def gaussian_data():
    """Create synthetic 1D Gaussian data for testing."""
    torch.manual_seed(42)
    np.random.seed(42)

    # Configuration: 3 Gaussians
    means = [0.0, 1.0, 0.5]
    stds = [0.3, 0.2, 0.4]
    n_samples = 1000
    dim = 1

    # Generate samples using shared random base
    rand = torch.randn(n_samples, dim)
    p_samples = [rand * std + mean for mean, std in zip(means, stds)]

    return {
        "p_samples": p_samples,
        "means": means,
        "stds": stds,
        "n_samples": n_samples,
        "dim": dim,
    }


@pytest.fixture(scope="module")
def normalized_gaussian_data(gaussian_data):
    """Normalize Gaussian data for model training."""
    p_samples = gaussian_data["p_samples"]

    # Concatenate and normalize
    all_samples = torch.cat(p_samples, dim=0)
    p_mean = torch.mean(all_samples, dim=0)
    p_std = torch.std(all_samples, dim=0)

    normalized = (all_samples - p_mean) / p_std
    n_samples = gaussian_data["n_samples"]
    num_p = len(p_samples)

    # Recover list structure
    normalized = normalized.view(num_p, n_samples, gaussian_data["dim"])
    normalized_list = [normalized[i] for i in range(num_p)]

    return {
        **gaussian_data,
        "normalized_samples": normalized_list,
        "p_mean": p_mean,
        "p_std": p_std,
    }


@pytest.fixture(scope="module")
def gaussian_dataloaders(normalized_gaussian_data):
    """Create train/val dataloaders from Gaussian data."""
    from sklearn.model_selection import train_test_split

    p_samples = normalized_gaussian_data["normalized_samples"]

    # Split into train/val
    ps_split = train_test_split(*p_samples, test_size=0.2, random_state=42)
    ps_train = ps_split[::2]
    ps_val = ps_split[1::2]

    train_dataset = TensorDataset(*ps_train)
    val_dataset = TensorDataset(*ps_val)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


@pytest.fixture(scope="module")
def simple_gaussian_model(device):
    """Create a simple OTPFM model for 1D Gaussian tests."""
    from otpfm import OTPFM
    from otpfm.potentials import W2InfPotential as IndependentPotential

    dim = 1
    tks = [0.5]
    potentials = OrderedDict()
    potentials[0.5] = IndependentPotential(
        tk=0.5,
        strength=10.0,
        lambda_type="gaussian",
        width=0.2,
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
        ema_decay=0.99,
        euler_steps=2,
    ).to(device)

    return model


# ============================================================================
# Data Normalization Tests
# ============================================================================


class TestDataNormalization:
    """Tests for data normalization utilities."""

    def test_normalize_data(self, gaussian_data):
        """Test normalize_data function."""

        def normalize_data(samples, mean=None, std=None):
            if mean is None:
                mean = torch.mean(samples, dim=0)
            if std is None:
                std = torch.std(samples, dim=0)
            return (samples - mean) / std, mean, std

        samples = torch.cat(gaussian_data["p_samples"], dim=0)
        normalized, mean, std = normalize_data(samples)

        # Check normalized statistics
        assert normalized.shape == samples.shape
        assert torch.allclose(normalized.mean(dim=0), torch.zeros(1), atol=1e-5)
        assert torch.allclose(normalized.std(dim=0), torch.ones(1), atol=1e-2)

    def test_unnormalize_data(self, gaussian_data):
        """Test unnormalize_data restores original distribution."""

        def normalize_data(samples, mean=None, std=None):
            if mean is None:
                mean = torch.mean(samples, dim=0)
            if std is None:
                std = torch.std(samples, dim=0)
            return (samples - mean) / std, mean, std

        def unnormalize_data(samples, mean, std):
            return (samples * std) + mean

        samples = torch.cat(gaussian_data["p_samples"], dim=0)
        normalized, mean, std = normalize_data(samples)
        unnormalized = unnormalize_data(normalized, mean, std)

        assert torch.allclose(unnormalized, samples, atol=1e-5)


# ============================================================================
# Potential Tests
# ============================================================================


class TestPotentials:
    """Tests for various potential types used in Gaussian experiments."""

    def test_independent_potential(self, device):
        """Test IndependentPotential gradient computation."""
        from otpfm.potentials import W2InfPotential as IndependentPotential

        potential = IndependentPotential(
            tk=0.5,
            strength=10.0,
            lambda_type="gaussian",
            width=0.2,
        )

        # Create test data
        x_target = torch.randn(100, 1, device=device)
        x_source = torch.randn(100, 1, device=device)

        # Compute gradient
        grad = potential.grad_gk(x_target, x_source)

        assert grad.shape == x_source.shape
        # Independent potential: grad should be x_source - x_target (direction toward target)
        expected = x_source - x_target
        assert torch.allclose(grad, expected, atol=1e-5)

    def test_w2_potential(self, device):
        """Test W2Potential (exact OT) gradient computation."""
        from otpfm.potentials import W2Potential

        potential = W2Potential(tk=0.5, strength=1.0)

        # Create sorted test data for easier OT verification
        x_target = torch.linspace(-2, 2, 50, device=device).unsqueeze(-1)
        x_source = torch.linspace(-1, 3, 50, device=device).unsqueeze(-1)

        grad = potential.grad_gk(x_target, x_source)

        assert grad.shape == x_source.shape
        # W2 grad should transport source toward target

    def test_mmd_rbf_potential(self, device):
        """Test MMDRBFPotential gradient computation."""
        from otpfm.potentials import MMDRBFPotential

        potential = MMDRBFPotential(
            tk=0.5,
            strength=10.0,
            lambda_type="gaussian",
            width=0.2,
            sigma=[1.0, 3.0],
        )

        x_target = torch.randn(100, 1, device=device)
        x_source = torch.randn(100, 1, device=device)

        grad = potential.grad_gk(x_target, x_source)

        assert grad.shape == x_source.shape
        assert torch.isfinite(grad).all()

    def test_mmd_poly_potential(self, device):
        """Test MMDPolyPotential gradient computation."""
        from otpfm.potentials import MMDPolyPotential

        potential = MMDPolyPotential(
            tk=0.5,
            strength=10.0,
            lambda_type="gaussian",
            width=0.2,
            degree=2,
            coef0=0.1,
        )

        x_target = torch.randn(100, 1, device=device)
        x_source = torch.randn(100, 1, device=device)

        grad = potential.grad_gk(x_target, x_source)

        assert grad.shape == x_source.shape
        assert torch.isfinite(grad).all()

    def test_entropic_ot_potential(self, device):
        """Test EntropicW2Potential gradient computation."""
        from otpfm.potentials import EntropicW2Potential

        potential = EntropicW2Potential(
            tk=0.5,
            strength=1.0,
            reg=0.1,
            numItermax=100,
        )

        x_target = torch.randn(50, 1, device=device)
        x_source = torch.randn(50, 1, device=device)

        grad = potential.grad_gk(x_target, x_source)

        assert grad.shape == x_source.shape
        assert torch.isfinite(grad).all()

    def test_kl_potential_sliced(self, device):
        """Test KLPotential with sliced method."""
        from otpfm.potentials import KLPotential

        potential = KLPotential(
            tk=0.5,
            strength=10.0,
            lambda_type="gaussian",
            width=0.2,
            rho_method="sliced",
            n_projections=50,
        )

        x_target = torch.randn(100, 2, device=device)
        x_source = torch.randn(100, 2, device=device)

        grad = potential.grad_gk(x_target, x_source)

        assert grad.shape == x_source.shape
        assert torch.isfinite(grad).all()

    def test_kl_potential_kde(self, device):
        """Test KLPotential with KDE method."""
        from otpfm.potentials import KLPotential

        potential = KLPotential(
            tk=0.5,
            strength=10.0,
            lambda_type="gaussian",
            width=0.2,
            rho_method="kde",
            bandwidth=1.0,
        )

        x_target = torch.randn(100, 1, device=device)
        x_source = torch.randn(100, 1, device=device)

        grad = potential.grad_gk(x_target, x_source)

        assert grad.shape == x_source.shape
        assert torch.isfinite(grad).all()


# ============================================================================
# Lambda Function Tests
# ============================================================================


class TestLambdaFunctions:
    """Tests for time-dependent lambda functions."""

    def test_gaussian_lambda(self):
        """Test GaussianLambda function."""
        from otpfm.lambdas import GaussianLambda

        tk = 0.5
        width = 0.1
        lambda_fn = GaussianLambda(tk, width)

        # Test at peak
        t_peak = torch.tensor([tk])
        val_peak = lambda_fn(t_peak)
        assert val_peak.item() > 0.99  # Should be close to 1 at peak

        # Test away from peak
        t_far = torch.tensor([0.0])
        val_far = lambda_fn(t_far)
        assert val_far.item() < val_peak.item()

    def test_triangle_lambda(self):
        """Test TriangleLambda function."""
        from otpfm.lambdas import TriangleLambda

        tk = 0.5
        width = 0.2
        lambda_fn = TriangleLambda(tk, width)

        # Test at peak - value is 1/width (scaled for integration)
        t_peak = torch.tensor([tk])
        val_peak = lambda_fn(t_peak)
        assert val_peak.item() > 0  # Peak should be positive

        # Test at edges of support - should be close to 0
        t_edge = torch.tensor([tk + width])
        val_edge = lambda_fn(t_edge)
        assert val_edge.item() < val_peak.item() * 0.01  # Much smaller than peak

    def test_box_lambda(self):
        """Test BoxLambda function."""
        from otpfm.lambdas import BoxLambda

        tk = 0.5
        width = 0.1
        lambda_fn = BoxLambda(tk, width)

        # Test inside box - value is 1/(2*width) (scaled for integration)
        t_inside = torch.tensor([tk])
        val_inside = lambda_fn(t_inside)
        assert val_inside.item() > 0  # Should be positive inside box

        # Test outside box - should be exactly 0
        t_outside = torch.tensor([0.0])
        val_outside = lambda_fn(t_outside)
        assert torch.isclose(val_outside, torch.tensor(0.0), atol=1e-5)


# ============================================================================
# Model Tests
# ============================================================================


class TestModel:
    """Tests for OTPFM model with Gaussian data."""

    def test_model_instantiation(self, simple_gaussian_model):
        """Test model can be instantiated."""
        assert simple_gaussian_model is not None
        assert hasattr(simple_gaussian_model, "flownet")
        assert hasattr(simple_gaussian_model, "potentials")

    def test_model_forward_pass(self, simple_gaussian_model, device):
        """Test model forward pass with batch data."""
        batch_size = 32
        num_marginals = 3  # x0, xm, x1
        dim = 1

        # Create batch: (batch_size, num_marginals, dim)
        batch = torch.randn(batch_size, num_marginals, dim, device=device)

        # Forward pass
        simple_gaussian_model.train()
        loss = simple_gaussian_model.forward_with_loss(batch, otp_alpha=0.5)

        assert loss.dim() == 0  # Scalar loss
        assert torch.isfinite(loss)

    def test_model_sample(self, simple_gaussian_model, device):
        """Test model sampling."""
        n_samples = 100
        dim = 1
        n_steps = 10

        x0 = torch.randn(n_samples, dim, device=device)

        simple_gaussian_model.eval()
        with torch.no_grad():
            xs, t_eval = simple_gaussian_model.sample(x0, n_steps=n_steps, ema=True)

        # xs shape: (n_steps * euler_steps + 1, n_samples, dim)
        # Model uses euler_steps=2, so total steps = n_steps * 2 + 1 = 21
        euler_steps = simple_gaussian_model.euler_steps
        expected_timesteps = n_steps * euler_steps + 1
        assert xs.shape[0] == expected_timesteps
        assert xs.shape[1] == n_samples
        assert xs.shape[2] == dim

        # t_eval shape should match xs
        assert t_eval.shape[0] == expected_timesteps
        assert t_eval[0] == 0.0
        assert t_eval[-1] == 1.0

    def test_model_ema_update(self, simple_gaussian_model):
        """Test EMA model updates correctly."""
        if simple_gaussian_model.flownet_ema is None:
            pytest.skip("Model has no EMA")

        # Get initial EMA params
        initial_ema_params = {
            k: v.clone() for k, v in simple_gaussian_model.flownet_ema.state_dict().items()
        }

        # Simulate a training step by modifying flownet params
        with torch.no_grad():
            for p in simple_gaussian_model.flownet.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        # Update EMA
        simple_gaussian_model.update_ema()

        # Check EMA params changed
        current_ema_params = simple_gaussian_model.flownet_ema.state_dict()
        changed = False
        for k in initial_ema_params:
            if not torch.equal(initial_ema_params[k], current_ema_params[k]):
                changed = True
                break

        assert changed, "EMA params should have changed after update"


# ============================================================================
# Training Tests
# ============================================================================


class TestTraining:
    """Tests for training components."""

    def test_trainer_instantiation(
        self, simple_gaussian_model, gaussian_dataloaders, device, tmp_path
    ):
        """Test Trainer can be instantiated with individual parameters."""
        from experiments import Trainer

        train_loader, val_loader = gaussian_dataloaders

        trainer = Trainer(
            model=simple_gaussian_model,
            train_loader=train_loader,
            val_loader=val_loader,
            save_dir=tmp_path,
            lr=1e-3,
            epochs=10,
            sampling_steps=50,
            do_otp=True,
            potentials=simple_gaussian_model.potentials,
            device=device,
        )

        assert trainer.epochs == 10
        assert trainer.sampling_steps == 50
        assert trainer.do_otp is True

        # Test curriculum function
        # At step 0, alpha should be low
        assert trainer.curriculum(0) < 0.5
        # At final step, alpha should be high
        assert trainer.curriculum(trainer.total_steps) > 0.5

    def test_process_batch(self):
        """Test process_batch utility."""
        batch_size = 32
        num_marginals = 3
        dim = 1

        # Simulate DataLoader output: list of tensors per marginal
        batch = [torch.randn(batch_size, dim) for _ in range(num_marginals)]

        # Process batch the same way Trainer does
        processed = torch.stack(batch).transpose(0, 1)

        # Should be (batch_size, num_marginals, dim)
        assert processed.shape == (batch_size, num_marginals, dim)

    def test_training_step(self, simple_gaussian_model, gaussian_dataloaders, device, tmp_path):
        """Test a single training step."""
        from experiments import Trainer

        train_loader, val_loader = gaussian_dataloaders

        trainer = Trainer(
            model=simple_gaussian_model,
            train_loader=train_loader,
            val_loader=val_loader,
            save_dir=tmp_path,
            lr=1e-3,
            epochs=1,
            sampling_steps=10,
            potentials=simple_gaussian_model.potentials,
            device=device,
        )

        # Get a batch
        batch = next(iter(train_loader))
        batch = torch.stack(batch).transpose(0, 1).to(device)

        # Forward pass
        simple_gaussian_model.train()
        loss = simple_gaussian_model.forward_with_loss(batch, otp_alpha=0.5)

        assert torch.isfinite(loss)

        # Backward pass
        loss.backward()

        # Check gradients exist
        for p in simple_gaussian_model.flownet.parameters():
            if p.requires_grad:
                assert p.grad is not None


class TestGaussianTrainer:
    """Tests for GaussianTrainer specific functionality."""

    def test_gaussian_trainer_instantiation(
        self,
        simple_gaussian_model,
        gaussian_dataloaders,
        normalized_gaussian_data,
        device,
        tmp_path,
    ):
        """Test GaussianTrainer can be instantiated."""
        from experiments.gaussian import GaussianTrainer

        train_loader, val_loader = gaussian_dataloaders

        # Create x0s for trajectories
        x0s = torch.randn(100, 1)

        # Create transform function
        means = normalized_gaussian_data["means"]
        stds = normalized_gaussian_data["stds"]
        p_mean = normalized_gaussian_data["p_mean"]
        p_std = normalized_gaussian_data["p_std"]

        def x0s_transform_fn(x0s):
            return (x0s * stds[0] + means[0] - p_mean) / p_std

        def unnormalize_fn(x):
            return (x * p_std) + p_mean

        trainer = GaussianTrainer(
            model=simple_gaussian_model,
            train_loader=train_loader,
            val_loader=val_loader,
            save_dir=tmp_path,
            lr=1e-3,
            epochs=1,
            sampling_steps=10,
            x0s_for_trajectories=x0s,
            unnormalize_fn=unnormalize_fn,
            x0s_transform_fn=x0s_transform_fn,
            plot_kwargs={"means": means, "stds": stds},
            potentials=simple_gaussian_model.potentials,
            device=device,
        )

        assert trainer is not None
        assert trainer.x0s_for_trajectories is not None
        assert trainer.unnormalize_fn is not None

    def test_save_trajectories(
        self,
        simple_gaussian_model,
        gaussian_dataloaders,
        normalized_gaussian_data,
        device,
        tmp_path,
    ):
        """Test trajectory saving."""
        from experiments.gaussian import GaussianTrainer

        train_loader, val_loader = gaussian_dataloaders

        x0s = torch.randn(50, 1, device=device)

        # Keep original floats for plotting (needs CPU/numpy)
        means_cpu = normalized_gaussian_data["means"]
        stds_cpu = normalized_gaussian_data["stds"]

        # Convert to tensors on device for transforms
        means = torch.tensor(means_cpu, device=device)
        stds = torch.tensor(stds_cpu, device=device)
        p_mean = normalized_gaussian_data["p_mean"].to(device)
        p_std = normalized_gaussian_data["p_std"].to(device)

        def x0s_transform_fn(x0s):
            return (x0s * stds[0] + means[0] - p_mean) / p_std

        def unnormalize_fn(x):
            return (x * p_std.to(x.device)) + p_mean.to(x.device)

        trainer = GaussianTrainer(
            model=simple_gaussian_model,
            train_loader=train_loader,
            val_loader=val_loader,
            save_dir=tmp_path,
            lr=1e-3,
            epochs=1,
            sampling_steps=10,
            x0s_for_trajectories=x0s,
            unnormalize_fn=unnormalize_fn,
            x0s_transform_fn=x0s_transform_fn,
            plot_kwargs={"means": means_cpu, "stds": stds_cpu},
            potentials=simple_gaussian_model.potentials,
            device=device,
        )

        # Save trajectories
        xs_vcorr, t_eval = trainer.save_trajectories(x0s, unnormalize_fn, epoch=0)

        assert xs_vcorr is not None
        assert t_eval is not None
        assert len(trainer.epoch_trajectories_vcorr) == 1


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for full Gaussian pipeline."""

    def test_full_training_pipeline(self, device, tmp_path):
        """Test full training pipeline with Gaussian data."""
        from otpfm import OTPFM
        from otpfm.potentials import W2InfPotential as IndependentPotential
        from sklearn.model_selection import train_test_split

        from experiments import Trainer

        # Generate data
        torch.manual_seed(42)
        means = [0.0, 1.0, 0.5]
        stds = [0.3, 0.2, 0.4]
        n_samples = 500
        dim = 1

        rand = torch.randn(n_samples, dim)
        p_samples = [rand * std + mean for mean, std in zip(means, stds)]

        # Normalize
        all_samples = torch.cat(p_samples, dim=0)
        p_mean = torch.mean(all_samples, dim=0)
        p_std = torch.std(all_samples, dim=0)
        normalized = (all_samples - p_mean) / p_std
        normalized = normalized.view(len(means), n_samples, dim)
        normalized_list = [normalized[i] for i in range(len(means))]

        # Create dataloaders
        ps_split = train_test_split(*normalized_list, test_size=0.2, random_state=42)
        ps_train = ps_split[::2]
        ps_val = ps_split[1::2]

        train_dataset = TensorDataset(*ps_train)
        val_dataset = TensorDataset(*ps_val)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        # Create model
        tks = [0.5]
        potentials = OrderedDict()
        potentials[0.5] = IndependentPotential(
            tk=0.5, strength=10.0, lambda_type="gaussian", width=0.2
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
            ema_decay=0.99,
        ).to(device)

        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            save_dir=tmp_path,
            lr=1e-3,
            epochs=2,
            sampling_steps=10,
            otp_alpha_type="1",
            potentials=potentials,
            device=device,
        )

        # Train
        losses, batch = trainer.train()

        # Check training completed
        assert "train_loss" in losses
        assert "val_loss" in losses
        assert len(losses["train_loss"]) == 2  # 2 epochs

        # Check model can sample
        model.eval()
        x0 = torch.randn(50, dim, device=device)
        with torch.no_grad():
            xs, t_eval = model.sample(x0, n_steps=10)

        # Model samples with internal euler_steps (default=2), so total = n_steps * euler_steps + 1
        assert xs.shape[0] >= 11  # At least 10 steps + initial
        assert torch.isfinite(xs).all()

    def test_multiple_potentials(self, device, tmp_path):
        """Test model with multiple intermediate potentials."""
        from otpfm import OTPFM
        from otpfm.potentials import W2InfPotential as IndependentPotential
        from sklearn.model_selection import train_test_split

        from experiments import Trainer

        # Generate data with 4 marginals (2 intermediate)
        torch.manual_seed(42)
        means = [0.0, 0.5, 1.5, 1.0]
        stds = [0.3, 0.2, 0.2, 0.4]
        n_samples = 500
        dim = 1

        rand = torch.randn(n_samples, dim)
        p_samples = [rand * std + mean for mean, std in zip(means, stds)]

        # Normalize
        all_samples = torch.cat(p_samples, dim=0)
        p_mean = torch.mean(all_samples, dim=0)
        p_std = torch.std(all_samples, dim=0)
        normalized = (all_samples - p_mean) / p_std
        normalized = normalized.view(len(means), n_samples, dim)
        normalized_list = [normalized[i] for i in range(len(means))]

        # Create dataloaders
        ps_split = train_test_split(*normalized_list, test_size=0.2, random_state=42)
        ps_train = ps_split[::2]
        ps_val = ps_split[1::2]

        train_dataset = TensorDataset(*ps_train)
        val_dataset = TensorDataset(*ps_val)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        # Create model with 2 potentials
        tks = [0.33, 0.67]
        potentials = OrderedDict()
        for tk in tks:
            potentials[tk] = IndependentPotential(
                tk=tk, strength=10.0, lambda_type="gaussian", width=0.15
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
            ema_decay=0.99,
        ).to(device)

        # Train briefly
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            save_dir=tmp_path,
            lr=1e-3,
            epochs=1,
            sampling_steps=10,
            otp_alpha_type="1",
            potentials=potentials,
            device=device,
        )

        losses, _ = trainer.train()

        assert len(losses["train_loss"]) == 1

    def test_model_checkpoint_save_load(
        self, simple_gaussian_model, gaussian_dataloaders, device, tmp_path
    ):
        """Test model checkpoint save and load."""
        from experiments import Trainer

        train_loader, val_loader = gaussian_dataloaders

        trainer = Trainer(
            model=simple_gaussian_model,
            train_loader=train_loader,
            val_loader=val_loader,
            save_dir=tmp_path,
            lr=1e-3,
            epochs=1,
            sampling_steps=10,
            potentials=simple_gaussian_model.potentials,
            device=device,
        )

        # Save checkpoint
        save_path = trainer.save_checkpoint("test_model.pt")
        assert save_path.exists()

        # Load checkpoint
        trainer.load_checkpoint(save_path)

        # Model should still work
        simple_gaussian_model.eval()
        x0 = torch.randn(10, 1, device=device)
        with torch.no_grad():
            xs, _ = simple_gaussian_model.sample(x0, n_steps=5)

        assert torch.isfinite(xs).all()


# ============================================================================
# FlowNet Architecture Tests
# ============================================================================


class TestFlowNetArchitectures:
    """Tests for different FlowNet architectures."""

    def test_flownet_mlp(self, device):
        """Test FlowNetMLP architecture."""
        from otpfm.networks import FlowNetMLP

        dim = 2
        flownet = FlowNetMLP(
            d=dim,
            x_emb_dim=32,
            t_emb_dim=32,
            num_hidden_layers=2,
            hidden_dim=64,
        ).to(device)

        batch_size = 16
        x = torch.randn(batch_size, dim, device=device)
        t = torch.rand(batch_size, 1, device=device)
        dt = torch.rand(batch_size, 1, device=device)

        v = flownet(x, t, dt)

        assert v.shape == (batch_size, dim)
        assert torch.isfinite(v).all()
