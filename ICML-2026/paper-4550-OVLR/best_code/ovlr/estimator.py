"""
OVLR Gradient Estimators.

Core estimators:
- OVLRGradientEstimator: Base Likelihood Ratio estimator
- ScoreFunctionOVLRGradientEstimator: Score-function based (with neg_score)
- TwoPointSPSAOVLRGradientEstimator: Two-point SPSA for Rademacher noise
"""

import torch
import torch.nn as nn


class OVLRGradientEstimator(nn.Module):
    """
    Output-Level Variance-Reduced Likelihood Ratio (OVLR) Gradient Estimator.

    This estimator enables gradient-based optimization of non-differentiable
    loss functions by injecting noise at the output level and using the
    likelihood ratio method with variance reduction.

    Args:
        noise_fn: Noise generator function that creates noise for output perturbation
        n_repeat: Number of noisy samples to use for variance reduction (default: 1)

    Example:
        >>> estimator = OVLRGradientEstimator(noise_fn, n_repeat=100)
        >>> outputs = model(inputs)
        >>> loss = estimator(outputs, labels, hard_01_loss)
    """
    def __init__(self, noise_fn, n_repeat=1):
        super().__init__()
        self.noise_fn = noise_fn
        self.n_repeat = n_repeat

    def forward_noisy_outputs(self, outputs):
        """Generate noisy outputs for gradient estimation."""
        if self.n_repeat > 1:
            outputs = outputs.repeat(self.n_repeat, *([1] * (outputs.dim() - 1)))
        with torch.no_grad():
            noise, epsilon = self.noise_fn.generate(outputs)
            noisy_outputs = outputs + noise
        return outputs, noisy_outputs, epsilon

    def estimate_gradient_and_backward(self, outputs, noisy_outputs, epsilon, labels,
                                       loss_fn, loss_fn_reduction, retain_graph=False):
        """Estimate gradients using the likelihood ratio method and backpropagate."""
        batch_size = outputs.size(0)
        loss = loss_fn(noisy_outputs, labels)

        if loss_fn_reduction == "sum":
            batch_size = self.n_repeat
        elif loss_fn_reduction != "mean":
            raise ValueError(f"Unsupported reduction: {loss_fn_reduction}")

        while loss.dim() < epsilon.dim():
            loss = loss.unsqueeze(-1)

        vec = (loss * epsilon) / (batch_size * self.noise_fn.noise_scale)
        outputs.backward(vec, retain_graph=retain_graph)

        if loss_fn_reduction == "sum":
            loss = loss.sum()
        else:
            loss = loss.mean()
        return loss

    def forward(self, outputs, labels, loss_fn, loss_fn_reduction='mean', retain_graph=False):
        """
        Compute loss and estimate gradients via OVLR.

        Args:
            outputs: Model outputs (pre-softmax logits for classification)
            labels: Target labels
            loss_fn: Non-differentiable loss function to optimize
            loss_fn_reduction: 'mean' or 'sum' - how loss_fn reduces batch
            retain_graph: Passed to backward() for retaining computation graph

        Returns:
            Scalar loss value (averaged over noisy samples)
        """
        outputs, noisy_outputs, epsilon = self.forward_noisy_outputs(outputs)
        if labels is not None and self.n_repeat > 1:
            labels = labels.repeat(self.n_repeat, *([1] * (labels.dim() - 1)))
        loss = self.estimate_gradient_and_backward(
            outputs, noisy_outputs, epsilon, labels, loss_fn,
            loss_fn_reduction, retain_graph=retain_graph
        )
        return loss


class ScoreFunctionOVLRGradientEstimator(nn.Module):
    """
    Score-function based OVLR gradient estimator.

    Uses the negative score function of the noise distribution for gradient
    estimation: g ≈ E[loss(x + noise) * neg_score(epsilon) / noise_scale]

    This is the theoretically correct likelihood ratio gradient estimator
    where the score function -∇_epsilon log p(epsilon) is computed analytically
    for each noise distribution.

    Requires noise_fn to have a neg_score(epsilon) method.

    Args:
        noise_fn: Noise generator with neg_score() method
        n_repeat: Number of noisy samples for Monte Carlo estimation

    Reference:
        OVLR: Efficient, Scalable, and Robust Training via
        Output-Level Variance-Reduced Likelihood Ratio, ICML 2026
    """
    def __init__(self, noise_fn, n_repeat=1):
        super().__init__()
        self.noise_fn = noise_fn
        self.n_repeat = n_repeat
        if not hasattr(noise_fn, 'neg_score'):
            raise ValueError("ScoreFunctionOVLRGradientEstimator requires noise_fn with neg_score() method")

    def _repeat_tensor(self, tensor):
        if self.n_repeat <= 1:
            return tensor
        return tensor.repeat(self.n_repeat, *([1] * (tensor.dim() - 1)))

    def forward(self, outputs, labels, loss_fn, loss_fn_reduction='mean', retain_graph=False):
        if loss_fn_reduction != 'mean':
            raise ValueError('ScoreFunctionOVLRGradientEstimator currently supports only loss_fn_reduction="mean".')

        outputs_repeat = self._repeat_tensor(outputs)
        labels_repeat = self._repeat_tensor(labels) if labels is not None else labels

        with torch.no_grad():
            noise, epsilon = self.noise_fn.generate(outputs_repeat)
            noisy_outputs = outputs_repeat + noise

        loss = loss_fn(noisy_outputs, labels_repeat)
        neg_score = self.noise_fn.neg_score(epsilon) / self.noise_fn.noise_scale

        while loss.dim() < neg_score.dim():
            loss = loss.unsqueeze(-1)

        vec = (loss * neg_score) / outputs_repeat.size(0)
        outputs_repeat.backward(vec, retain_graph=retain_graph)
        return loss.mean()


class TwoPointSPSAOVLRGradientEstimator(nn.Module):
    """
    Two-Point SPSA OVLR gradient estimator.

    Uses symmetric finite differences for gradient estimation:
    g ≈ E[(loss(x + noise) - loss(x - noise)) * direction / (2 * noise_scale)]

    Suitable for Rademacher-type direction noise. This is equivalent to
    simultaneous perturbation stochastic approximation (SPSA) at the
    output level.

    Args:
        direction_noise: Direction noise generator (e.g., RademacherDirectionNoise)
        n_repeat: Number of noisy samples for Monte Carlo estimation

    Reference:
        OVLR: Efficient, Scalable, and Robust Training via
        Output-Level Variance-Reduced Likelihood Ratio, ICML 2026
    """
    def __init__(self, direction_noise, n_repeat=1):
        super().__init__()
        self.direction_noise = direction_noise
        self.n_repeat = n_repeat

    def _repeat_tensor(self, tensor):
        if self.n_repeat <= 1:
            return tensor
        return tensor.repeat(self.n_repeat, *([1] * (tensor.dim() - 1)))

    def forward(self, outputs, labels, loss_fn, loss_fn_reduction='mean', retain_graph=False):
        if loss_fn_reduction != 'mean':
            raise ValueError('TwoPointSPSAOVLRGradientEstimator currently supports only loss_fn_reduction="mean".')

        outputs_repeat = self._repeat_tensor(outputs)
        labels_repeat = self._repeat_tensor(labels) if labels is not None else labels

        with torch.no_grad():
            noise, direction = self.direction_noise.generate(outputs_repeat)
            loss_plus = loss_fn(outputs_repeat + noise, labels_repeat)
            loss_minus = loss_fn(outputs_repeat - noise, labels_repeat)

        coeff = (loss_plus - loss_minus) / (2.0 * self.direction_noise.noise_scale)
        while coeff.dim() < direction.dim():
            coeff = coeff.unsqueeze(-1)

        vec = (coeff * direction) / outputs_repeat.size(0)
        outputs_repeat.backward(vec, retain_graph=retain_graph)
        return 0.5 * (loss_plus.mean() + loss_minus.mean())
