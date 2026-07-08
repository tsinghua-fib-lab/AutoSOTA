"""
Global color attack for CLIP models.

This attack learns image-wide affine color transforms:

    x_adv = clamp(M @ x + bias, 0, 1)

The transformation is optimized end-to-end to maximize CLIP zero-shot
classification loss while staying within bounded global color changes.
"""

import torch
import torch.nn as nn


class GlobalColorAttack:
    """
    Global color manipulation attack.

    Args:
        attacker: CLIPAttacker instance.
        steps: Number of optimization steps.
        lr: Adam learning rate.
        scale_delta: Maximum deviation from the identity color matrix.
        bias_delta: Maximum additive change.
        random_start: Whether to initialize with small random parameters.
    """

    def __init__(
        self,
        attacker,
        steps: int = 100,
        lr: float = 1e-2,
        scale_delta: float = 0.15,
        bias_delta: float = 0.08,
        gamma_delta: float = 0.0,
        random_start: bool = True,
    ):
        self.attacker = attacker
        self.steps = steps
        self.lr = lr
        self.scale_delta = scale_delta
        self.bias_delta = bias_delta
        self.gamma_delta = gamma_delta
        self.random_start = random_start
        self.device = attacker.device

    def _build_transform(
        self,
        raw_matrix: torch.Tensor,
        raw_bias: torch.Tensor,
        raw_gamma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eye = torch.eye(3, device=raw_matrix.device).view(1, 3, 3)
        scale = eye + self.scale_delta * torch.tanh(raw_matrix)
        bias = self.bias_delta * torch.tanh(raw_bias)
        gamma = torch.exp(self.gamma_delta * torch.tanh(raw_gamma))
        return scale, bias, gamma

    @torch.enable_grad()
    def attack(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate adversarial images with global color manipulations.
        """
        images = images.to(self.device)
        labels = labels.to(self.device)
        batch_size, channels, _, _ = images.shape
        loss_fn = nn.CrossEntropyLoss()

        raw_matrix = torch.zeros(batch_size, channels, channels, device=self.device)
        raw_bias = torch.zeros(batch_size, channels, 1, 1, device=self.device)
        raw_gamma = torch.zeros(batch_size, channels, 1, 1, device=self.device)
        if self.random_start:
            raw_matrix.uniform_(-0.1, 0.1)
            raw_bias.uniform_(-0.1, 0.1)
            raw_gamma.uniform_(-0.1, 0.1)

        raw_matrix.requires_grad = True
        raw_bias.requires_grad = True
        raw_gamma.requires_grad = True
        optimizer = torch.optim.Adam([raw_matrix, raw_bias, raw_gamma], lr=self.lr)

        best_imgs = images.detach().clone()
        best_loss = torch.full((batch_size,), -float("inf"), device=self.device)

        for _ in range(self.steps):
            scale, bias, gamma = self._build_transform(raw_matrix, raw_bias, raw_gamma)
            mixed = torch.einsum("bij,bjhw->bihw", scale, images)
            adv_images = torch.clamp(mixed + bias, 1e-4, 1.0)
            if self.gamma_delta > 0:
                adv_images = torch.pow(adv_images, gamma)
            logits = self.attacker(adv_images, text_features)
            per_sample_loss = nn.functional.cross_entropy(logits, labels, reduction="none")
            loss = per_sample_loss.mean()

            better_mask = per_sample_loss > best_loss
            if better_mask.any():
                best_loss[better_mask] = per_sample_loss[better_mask].detach()
                best_imgs[better_mask] = adv_images[better_mask].detach()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            scale, bias, gamma = self._build_transform(raw_matrix, raw_bias, raw_gamma)
            mixed = torch.einsum("bij,bjhw->bihw", scale, images)
            adv_images = torch.clamp(mixed + bias, 1e-4, 1.0)
            if self.gamma_delta > 0:
                adv_images = torch.pow(adv_images, gamma)
            logits = self.attacker(adv_images, text_features)
            per_sample_loss = nn.functional.cross_entropy(logits, labels, reduction="none")
            better_mask = per_sample_loss > best_loss
            if better_mask.any():
                best_imgs[better_mask] = adv_images[better_mask]

        return best_imgs.detach()
