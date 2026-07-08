"""
PGD (Projected Gradient Descent) attack for CLIP models.

Untargeted L-inf attack that maximizes cross-entropy loss to cause
misclassification in zero-shot CLIP inference.
"""

import torch
import torch.nn as nn


class PGDAttack:
    """
    PGD-Linf adversarial attack.
    
    Args:
        attacker: CLIPAttacker instance.
        epsilon: Maximum perturbation (L-inf norm).
        steps: Number of PGD iterations.
        alpha: Step size per iteration. If None, computed as (epsilon/steps)*alpha_scale.
        alpha_scale: Multiplier for auto-computed alpha.
        random_start: Whether to initialize with random noise.
    """

    def __init__(
        self,
        attacker,
        epsilon: float = 4 / 255,
        steps: int = 10,
        alpha: float = None,
        alpha_scale: float = 2.5,
        random_start: bool = True,
    ):
        self.attacker = attacker
        self.epsilon = epsilon
        self.steps = steps
        self.alpha = alpha if alpha is not None else (epsilon / steps) * alpha_scale
        self.random_start = random_start
        self.device = attacker.device

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

        # Random start
        if self.random_start:
            delta = torch.zeros_like(images).uniform_(-self.epsilon, self.epsilon)
            delta = torch.clamp(images + delta, 0, 1) - images
        else:
            delta = torch.zeros_like(images)
        delta.requires_grad = True

        for _ in range(self.steps):
            adv_images = torch.clamp(images + delta, 0, 1)
            logits = self.attacker(adv_images, text_features)

            # Maximize cross-entropy (untargeted)
            loss = loss_fn(logits, labels)
            loss.backward()

            grad = delta.grad.detach()
            delta.data = torch.clamp(
                delta + self.alpha * grad.sign(), -self.epsilon, self.epsilon
            )
            delta.grad.zero_()

        adv_images = torch.clamp(images + delta, 0, 1).detach()
        return adv_images
