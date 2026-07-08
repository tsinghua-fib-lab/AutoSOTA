"""
Adaptive PGD attack for CLIP models.

This attack follows the same L-inf 10-step PGD setting as the standard PGD
attack, but augments the objective with a frequency-domain regularizer that
directly penalizes absolute high-frequency perturbation energy. The low/high-
frequency split is controlled by a radial cutoff in the FFT domain.
"""

import torch
import torch.nn as nn


class AdaptivePGDAttack:
    """
    Adaptive PGD-Linf attack with absolute high-frequency energy regularization.

    Args:
        attacker: CLIPAttacker instance.
        epsilon: Maximum perturbation (L-inf norm).
        steps: Number of PGD iterations.
        alpha: Step size per iteration. If None, computed as (epsilon / steps) * alpha_scale.
        alpha_scale: Multiplier for auto-computed alpha.
        random_start: Whether to initialize with random noise.
        freq_radius: Radius separating low vs. mid/high frequencies in FFT space.
        hf_reg_weight: Weight applied to the high-frequency energy penalty term.
    """

    def __init__(
        self,
        attacker,
        epsilon: float = 4 / 255,
        steps: int = 10,
        alpha: float = None,
        alpha_scale: float = 2.5,
        random_start: bool = True,
        freq_radius: int = 40,
        hf_reg_weight: float = 1.0,
    ):
        self.attacker = attacker
        self.epsilon = epsilon
        self.steps = steps
        self.alpha = alpha if alpha is not None else (epsilon / steps) * alpha_scale
        self.random_start = random_start
        self.freq_radius = freq_radius
        self.hf_reg_weight = hf_reg_weight
        self.device = attacker.device

        self._cached_hf_mask = None
        self._cached_shape = None

    def _get_high_freq_mask(self, height: int, width: int) -> torch.Tensor:
        """Build a cached FFT-domain mask for frequencies outside the low-pass radius."""
        if self._cached_hf_mask is not None and self._cached_shape == (height, width):
            return self._cached_hf_mask

        cy, cx = height // 2, width // 2
        y = torch.arange(height, device=self.device) - cy
        x = torch.arange(width, device=self.device) - cx
        y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")
        dist_sq = y_grid ** 2 + x_grid ** 2
        high_freq_mask = (dist_sq > (self.freq_radius ** 2)).float().view(1, 1, height, width)

        self._cached_hf_mask = high_freq_mask
        self._cached_shape = (height, width)
        return high_freq_mask

    def _high_freq_energy(self, delta: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample absolute high-frequency perturbation energy.

        We normalize by the number of active high-frequency bins so the scale
        stays comparable across image sizes, but this remains an absolute
        penalty rather than a ratio against total perturbation energy.
        """
        _, _, height, width = delta.shape
        high_freq_mask = self._get_high_freq_mask(height, width)

        fft_delta = torch.fft.fftshift(torch.fft.fft2(delta), dim=(-2, -1))
        power = fft_delta.abs().pow(2)

        high_freq_energy = (power * high_freq_mask).sum(dim=(1, 2, 3))
        high_freq_bins = high_freq_mask.sum().clamp_min(1.0)
        return high_freq_energy / high_freq_bins

    @torch.enable_grad()
    def attack(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate adversarial examples.

        Args:
            images: [B, 3, H, W] clean images in [0, 1].
            labels: [B] ground-truth class indices.
            text_features: [num_classes, embed_dim] precomputed text embeddings.

        Returns:
            Adversarial images [B, 3, H, W] in [0, 1].
        """
        images = images.to(self.device)
        labels = labels.to(self.device)
        loss_fn = nn.CrossEntropyLoss()

        if self.random_start:
            delta = torch.zeros_like(images).uniform_(-self.epsilon, self.epsilon)
            delta = torch.clamp(images + delta, 0, 1) - images
        else:
            delta = torch.zeros_like(images)
        delta.requires_grad = True

        for _ in range(self.steps):
            adv_images = torch.clamp(images + delta, 0, 1)
            logits = self.attacker(adv_images, text_features)

            ce_loss = loss_fn(logits, labels)
            high_freq_energy = self._high_freq_energy(delta)
            loss = ce_loss - self.hf_reg_weight * high_freq_energy.mean()
            loss.backward()

            grad = delta.grad.detach()
            delta.data = torch.clamp(
                delta + self.alpha * grad.sign(),
                -self.epsilon,
                self.epsilon,
            )
            delta.data = torch.clamp(images + delta.data, 0, 1) - images
            delta.grad.zero_()

        adv_images = torch.clamp(images + delta, 0, 1).detach()
        return adv_images
