"""
Noise generators for OVLR gradient estimation.

Core noise types used in the paper:
- Symmetric Gaussian: Default choice with antithetic variance reduction
- Symmetric Laplace: Heavy-tailed for robustness
- Symmetric Rademacher: SPSA-style discrete noise
- Asymmetric variants: For ablation studies

Score-function noise types (with neg_score method):
- GaussianScoreNoise, StudentTScoreNoise, LaplaceScoreNoise
"""

import math
import torch
import torch.distributions as D


# ==================== Symmetric (Antithetic) Noise ====================

class SymmetricGaussianNoise:
    """
    Symmetric Gaussian noise with antithetic pairing.

    Generates (epsilon, -epsilon) for half the batch to achieve variance
    reduction in score-function gradient estimates.
    """
    def __init__(self, noise_scale=1.0):
        self.noise_scale = noise_scale

    def generate(self, outputs):
        batch_size = outputs.size(0)
        half_batch = batch_size // 2
        feature_shape = outputs.shape[1:]

        epsilon = torch.zeros_like(outputs, device=outputs.device)
        noise_half = torch.randn((half_batch, *feature_shape), device=outputs.device)

        epsilon[:half_batch] = noise_half
        epsilon[half_batch:] = -noise_half

        noise = epsilon * self.noise_scale
        return noise, epsilon


class SymmetricLaplaceNoise:
    """
    Symmetric Laplace (double-exponential) noise with antithetic pairing.

    Heavy-tailed distribution providing robustness to outlier losses.
    """
    def __init__(self, noise_scale=1.0):
        self.noise_scale = noise_scale
        self.base_scale = 1.0 / math.sqrt(2.0)

    def generate(self, outputs):
        batch_size = outputs.size(0)
        half_batch = batch_size // 2
        feature_shape = outputs.shape[1:]

        epsilon = torch.zeros_like(outputs, device=outputs.device)
        location = torch.tensor(0.0, device=outputs.device, dtype=outputs.dtype)
        scale = torch.tensor(self.base_scale, device=outputs.device, dtype=outputs.dtype)
        noise_half = D.Laplace(location, scale).sample((half_batch, *feature_shape))

        epsilon[:half_batch] = noise_half
        epsilon[half_batch:] = -noise_half

        noise = epsilon * self.noise_scale
        return noise, epsilon


class SymmetricRademacherNoise:
    """
    Symmetric Rademacher (+/-1) noise with antithetic pairing.

    Discrete binary noise used in SPSA-style methods. Higher variance
    but sometimes more robust to gradient estimation.
    """
    def __init__(self, noise_scale=1.0):
        self.noise_scale = noise_scale

    def generate(self, outputs):
        batch_size = outputs.size(0)
        half_batch = batch_size // 2
        feature_shape = outputs.shape[1:]

        epsilon = torch.zeros_like(outputs, device=outputs.device)
        noise_half = torch.empty(
            (half_batch, *feature_shape),
            device=outputs.device,
            dtype=outputs.dtype,
        )
        noise_half.bernoulli_(0.5).mul_(2.0).sub_(1.0)

        epsilon[:half_batch] = noise_half
        epsilon[half_batch:] = -noise_half

        noise = epsilon * self.noise_scale
        return noise, epsilon


class SymmetricStudentTNoise:
    """
    Symmetric Student's T-distribution noise with antithetic pairing.

    Even heavier tails than Laplace. Good for extreme robustness.
    """
    def __init__(self, df=5.0, noise_scale=1.0):
        if df <= 2.0:
            raise ValueError('Student-t requires df > 2 for finite variance.')
        self.df = df
        self.noise_scale = noise_scale
        self.std = math.sqrt(df / (df - 2.0))

    def generate(self, outputs):
        batch_size = outputs.size(0)
        half_batch = batch_size // 2
        feature_shape = outputs.shape[1:]

        epsilon = torch.zeros_like(outputs, device=outputs.device)
        noise_half = D.StudentT(self.df).sample((half_batch, *feature_shape)).to(outputs.device)
        noise_half = noise_half / self.std

        epsilon[:half_batch] = noise_half
        epsilon[half_batch:] = -noise_half

        noise = epsilon * self.noise_scale
        return noise, epsilon


# ==================== Asymmetric Noise (for ablation) ====================

class AsymmetricGaussianNoise:
    """Standard Gaussian noise without antithetic pairing."""
    def __init__(self, noise_scale=1.0):
        self.noise_scale = noise_scale

    def generate(self, outputs):
        epsilon = torch.randn_like(outputs)
        noise = epsilon * self.noise_scale
        return noise, epsilon


# ==================== Score-Function Noise (with neg_score) ====================

class GaussianScoreNoise:
    """
    Gaussian noise with score-function support.

    Score: ∇_epsilon log p(epsilon) = -epsilon
    Negative score: -score = epsilon
    """
    def __init__(self, noise_scale=1.0):
        self.noise_scale = noise_scale

    def generate(self, outputs):
        epsilon = torch.randn_like(outputs)
        noise = epsilon * self.noise_scale
        return noise, epsilon

    def neg_score(self, epsilon):
        """Negative score function: -∇_epsilon log p(epsilon)"""
        return epsilon


class StudentTScoreNoise:
    """
    Student's T-distribution noise with score-function support.

    Score: ∇_epsilon log p(epsilon) = -(df + 1) * epsilon / (df + epsilon^2)
    Negative score: (df + 1) * epsilon / (df + epsilon^2)

    Note: epsilon is variance-normalized: raw_t / sqrt(df / (df - 2))
    """
    def __init__(self, df=5.0, noise_scale=1.0):
        if df <= 2.0:
            raise ValueError('Student-t score estimator requires df > 2 for finite variance.')
        self.df = df
        self.noise_scale = noise_scale
        self.std = math.sqrt(df / (df - 2.0))

    def generate(self, outputs):
        epsilon = D.StudentT(self.df).sample(outputs.shape).to(device=outputs.device, dtype=outputs.dtype)
        epsilon = epsilon / self.std
        noise = epsilon * self.noise_scale
        return noise, epsilon

    def neg_score(self, epsilon):
        """Negative score function for variance-normalized epsilon."""
        return ((self.df + 1.0) * epsilon) / (self.df - 2.0 + epsilon.pow(2))


class LaplaceScoreNoise:
    """
    Laplace (double-exponential) noise with score-function support.

    Score: ∇_epsilon log p(epsilon) = -sqrt(2) * sign(epsilon)
    Negative score: sqrt(2) * sign(epsilon)
    """
    def __init__(self, noise_scale=1.0):
        self.noise_scale = noise_scale
        self.base_scale = 1.0 / math.sqrt(2.0)

    def generate(self, outputs):
        location = torch.tensor(0.0, device=outputs.device, dtype=outputs.dtype)
        scale = torch.tensor(self.base_scale, device=outputs.device, dtype=outputs.dtype)
        epsilon = D.Laplace(location, scale).sample(outputs.shape)
        noise = epsilon * self.noise_scale
        return noise, epsilon

    def neg_score(self, epsilon):
        """Negative score function for Laplace noise."""
        return math.sqrt(2.0) * torch.sign(epsilon)


class RademacherDirectionNoise:
    """
    Rademacher (+/-1) direction noise for Two-Point SPSA estimation.

    Does not use score function - instead uses finite differences between
    two perturbed evaluations (f(x + noise) - f(x - noise)) / (2 * scale).
    """
    def __init__(self, noise_scale=1.0):
        self.noise_scale = noise_scale

    def generate(self, outputs):
        direction = torch.empty_like(outputs)
        direction.bernoulli_(0.5).mul_(2.0).sub_(1.0)
        noise = direction * self.noise_scale
        return noise, direction


# ==================== Factory Function ====================

def get_noise_fn(mode="symmetric", noise_scale=1.0, df=5.0):
    """
    Create noise generator by mode name.

    Args:
        mode: "symmetric" (default), "laplace", "rademacher", "studentt",
              "asymmetric", "gaussian_score", "studentt_score", "laplace_score",
              "rademacher_spsa"
        noise_scale: Noise scaling factor
        df: Degrees of freedom for Student-t

    Returns:
        Noise generator with generate(outputs) method
    """
    if mode in ["symmetric", "gaussian", "symmetric_gaussian"]:
        return SymmetricGaussianNoise(noise_scale=noise_scale)
    elif mode in ["laplace", "symmetric_laplace"]:
        return SymmetricLaplaceNoise(noise_scale=noise_scale)
    elif mode in ["rademacher", "symmetric_rademacher"]:
        return SymmetricRademacherNoise(noise_scale=noise_scale)
    elif mode in ["studentt", "symmetric_studentt"]:
        return SymmetricStudentTNoise(df=df, noise_scale=noise_scale)
    elif mode in ["asymmetric", "asymmetric_gaussian"]:
        return AsymmetricGaussianNoise(noise_scale=noise_scale)
    elif mode in ["gaussian_score"]:
        return GaussianScoreNoise(noise_scale=noise_scale)
    elif mode in ["studentt_score", "student_t_score"]:
        return StudentTScoreNoise(df=df, noise_scale=noise_scale)
    elif mode in ["laplace_score"]:
        return LaplaceScoreNoise(noise_scale=noise_scale)
    elif mode in ["rademacher_spsa"]:
        return RademacherDirectionNoise(noise_scale=noise_scale)
    else:
        raise ValueError(
            f"Unknown noise mode: {mode}. Available modes: "
            "symmetric (gaussian), laplace, rademacher, studentt, asymmetric, "
            "gaussian_score, studentt_score, laplace_score, rademacher_spsa"
        )
