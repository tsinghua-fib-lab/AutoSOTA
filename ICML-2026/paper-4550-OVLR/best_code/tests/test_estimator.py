"""Unit tests for OVLRGradientEstimator."""

import pytest
import torch
import torch.nn as nn

from ovlr import (
    OVLRGradientEstimator,
    SymmetricGaussianNoise,
    AsymmetricGaussianNoise,
    SymmetricStudentTNoise,
    AsymmetricStudentTNoise,
    get_noise_fn,
)


class TestNoiseGenerators:
    """Test noise generator classes."""

    def test_symmetric_gaussian_noise_shape(self):
        """Test symmetric Gaussian noise produces correct shape and antithetic pairs."""
        noise_fn = SymmetricGaussianNoise(noise_scale=1.0)
        outputs = torch.randn(4, 10)  # batch=4, dim=10
        noise, epsilon = noise_fn.generate(outputs)

        assert noise.shape == outputs.shape
        assert epsilon.shape == outputs.shape

        # Antithetic: first half = -second half
        assert torch.allclose(epsilon[:2], -epsilon[2:], atol=1e-6)

    def test_asymmetric_gaussian_noise_shape(self):
        """Test asymmetric Gaussian noise produces correct shape."""
        noise_fn = AsymmetricGaussianNoise(noise_scale=1.0)
        outputs = torch.randn(4, 10)
        noise, epsilon = noise_fn.generate(outputs)

        assert noise.shape == outputs.shape
        assert epsilon.shape == outputs.shape

    def test_symmetric_student_t_noise_shape(self):
        """Test symmetric Student's t noise produces correct shape."""
        noise_fn = SymmetricStudentTNoise(df=5.0, noise_scale=1.0)
        outputs = torch.randn(4, 10)
        noise, epsilon = noise_fn.generate(outputs)

        assert noise.shape == outputs.shape
        assert epsilon.shape == outputs.shape
        assert torch.allclose(epsilon[:2], -epsilon[2:], atol=1e-6)

    def test_asymmetric_student_t_noise_shape(self):
        """Test asymmetric Student's t noise produces correct shape."""
        noise_fn = AsymmetricStudentTNoise(df=5.0, noise_scale=1.0)
        outputs = torch.randn(4, 10)
        noise, epsilon = noise_fn.generate(outputs)

        assert noise.shape == outputs.shape
        assert epsilon.shape == outputs.shape

    def test_get_noise_fn_factory(self):
        """Test get_noise_fn returns correct types."""
        assert isinstance(get_noise_fn("symmetric"), SymmetricGaussianNoise)
        assert isinstance(get_noise_fn("asymmetric"), AsymmetricGaussianNoise)
        assert isinstance(get_noise_fn("symmetric_student_t"), SymmetricStudentTNoise)
        assert isinstance(get_noise_fn("asymmetric_student_t"), AsymmetricStudentTNoise)

        with pytest.raises(ValueError):
            get_noise_fn("invalid_mode")


class TestOVLRGradientEstimator:
    """Test OVLRGradientEstimator."""

    def test_estimator_init(self):
        """Test estimator initialization."""
        noise_fn = SymmetricGaussianNoise(noise_scale=1.0)
        estimator = OVLRGradientEstimator(noise_fn, n_repeat=10)
        assert estimator.n_repeat == 10
        assert estimator.noise_fn is noise_fn

    def test_forward_noisy_outputs_shape(self):
        """Test forward_noisy_outputs produces correct shapes."""
        noise_fn = SymmetricGaussianNoise(noise_scale=1.0)
        estimator = OVLRGradientEstimator(noise_fn, n_repeat=4)

        outputs = torch.randn(2, 5)  # batch=2, dim=5
        outputs_rep, noisy_outputs, epsilon = estimator.forward_noisy_outputs(outputs)

        # After n_repeat=4: batch dimension should be 2 * 4 = 8
        assert outputs_rep.shape == (8, 5)
        assert noisy_outputs.shape == (8, 5)
        assert epsilon.shape == (8, 5)

    def test_gradient_produces_finite_norm(self):
        """Test gradient estimation produces finite non-zero gradients."""
        torch.manual_seed(42)

        # Simple linear model
        model = nn.Linear(3, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)  # w = [1, 1, 1]

        noise_fn = SymmetricGaussianNoise(noise_scale=0.5)
        estimator = OVLRGradientEstimator(noise_fn, n_repeat=200)

        x = torch.randn(1, 3)
        target = torch.tensor([5.0])

        def mse_loss(output, target):
            return ((output.squeeze(1) - target) ** 2)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer.zero_grad()
        out = model(x)
        loss = estimator(out, target, mse_loss, loss_fn_reduction='mean')

        # Gradient should be finite and non-zero
        grad_norm = model.weight.grad.norm().item()
        assert torch.isfinite(torch.tensor(grad_norm))
        assert grad_norm > 0.01

    def test_gradient_flow_through_model(self):
        """Test that gradients actually flow through model parameters."""
        torch.manual_seed(42)

        model = nn.Linear(5, 2)
        noise_fn = SymmetricGaussianNoise(noise_scale=0.5)
        estimator = OVLRGradientEstimator(noise_fn, n_repeat=100)

        x = torch.randn(3, 5)
        targets = torch.tensor([0, 1, 0])

        def cross_entropy_loss(outputs, targets):
            return nn.CrossEntropyLoss(reduction='none')(outputs, targets)

        initial_weight = model.weight.clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer.zero_grad()

        outputs = model(x)
        loss = estimator(outputs, targets, cross_entropy_loss, loss_fn_reduction='mean')
        optimizer.step()

        # Weights should have changed
        assert not torch.allclose(model.weight, initial_weight)

    def test_loss_reduction_modes_produce_different_grads(self):
        """Test 'mean' and 'sum' reduction modes produce different gradient scales."""
        torch.manual_seed(42)
        noise_fn = SymmetricGaussianNoise(noise_scale=1.0)
        estimator = OVLRGradientEstimator(noise_fn, n_repeat=1)

        model = nn.Linear(3, 1)
        with torch.no_grad():
            model.weight.fill_(1.0)

        x = torch.randn(4, 3)
        target = torch.ones(4)

        def l1_loss(output, target):
            return (output.squeeze(1) - target).abs()

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        # Mean reduction
        optimizer.zero_grad()
        out = model(x)
        estimator(out, target, l1_loss, loss_fn_reduction='mean')
        grad_mean = model.weight.grad.norm().item()

        # Sum reduction
        optimizer.zero_grad()
        out = model(x)
        estimator(out, target, l1_loss, loss_fn_reduction='sum')
        grad_sum = model.weight.grad.norm().item()

        # Sum reduction should produce different gradient scale than mean
        # Batch size = 4, so difference should be noticeable
        assert abs(grad_sum - grad_mean) > 0.1 * min(grad_sum, grad_mean)

    def test_invalid_reduction_raises(self):
        """Test invalid reduction mode raises ValueError."""
        noise_fn = SymmetricGaussianNoise(noise_scale=1.0)
        estimator = OVLRGradientEstimator(noise_fn, n_repeat=4)

        model = nn.Linear(3, 1)
        x = torch.randn(2, 3)
        target = torch.ones(2)

        def dummy_loss(output, target):
            return (output - target).abs().squeeze(1)

        with pytest.raises(ValueError):
            out = model(x)
            estimator(out, target, dummy_loss, loss_fn_reduction='invalid')

    def test_gradient_variance_reduction_with_more_samples(self):
        """Test that larger n_repeat tends to produce more stable gradient estimates."""
        torch.manual_seed(42)

        def l2_loss(output, target):
            return ((output - target) ** 2).sum(dim=-1).squeeze(-1)

        def estimate_grad_stdev(n_repeat, trials=30):
            grad_norms = []
            for _ in range(trials):
                model = nn.Linear(5, 2)
                noise_fn = SymmetricGaussianNoise(noise_scale=0.5)
                estimator = OVLRGradientEstimator(noise_fn, n_repeat=n_repeat)

                x = torch.randn(2, 5)
                target = torch.randn(2, 2)

                out = model(x)
                estimator(out, target, l2_loss, loss_fn_reduction='mean')
                grad_norms.append(model.weight.grad.norm().item())

            return torch.tensor(grad_norms).std().item()

        # More samples should generally reduce variance (statistical test)
        variances_low = []
        variances_high = []
        for seed in range(5):
            torch.manual_seed(seed)
            variances_low.append(estimate_grad_stdev(10))
            variances_high.append(estimate_grad_stdev(200))

        # On average, higher n_repeat should give lower variance
        avg_var_low = sum(variances_low) / len(variances_low)
        avg_var_high = sum(variances_high) / len(variances_high)
        assert avg_var_high < avg_var_low * 0.7


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
