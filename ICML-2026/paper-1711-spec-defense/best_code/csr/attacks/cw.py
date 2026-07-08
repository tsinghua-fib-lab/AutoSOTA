"""
CW (Carlini-Wagner) margin loss attack for CLIP models.

Uses the CW margin loss within a PGD framework (PGD-CW) for L-inf attacks.
"""

import torch
import torch.nn as nn


class CWAttack:
    """
    PGD-CW (CW margin loss) L-inf adversarial attack.

    The CW margin loss maximizes: max_other(logit) - logit_true
    subject to L-inf perturbation bound.

    Args:
        attacker: CLIPAttacker instance.
        epsilon: Maximum perturbation (L-inf norm).
        steps: Number of PGD iterations.
        alpha: Step size. If None, auto-computed.
        alpha_scale: Multiplier for auto alpha.
        kappa: Confidence margin. Higher values produce more transferable adversarial examples.
        random_start: Whether to use random initialization.
    """

    def __init__(
        self,
        attacker,
        epsilon: float = 4 / 255,
        steps: int = 30,
        alpha: float = None,
        alpha_scale: float = 2.5,
        kappa: float = 0.0,
        random_start: bool = True,
    ):
        self.attacker = attacker
        self.epsilon = epsilon
        self.steps = steps
        self.alpha = alpha if alpha is not None else (epsilon / steps) * alpha_scale
        self.kappa = kappa
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
        Generate adversarial examples using CW margin loss.

        Args:
            images: [B, 3, H, W] clean images in [0, 1].
            labels: [B] ground-truth class indices.
            text_features: [num_classes, embed_dim].

        Returns:
            Adversarial images [B, 3, H, W] in [0, 1].
        """
        images = images.to(self.device)
        labels = labels.to(self.device)

        if self.random_start:
            delta = torch.zeros_like(images).uniform_(-self.epsilon, self.epsilon)
            delta = torch.clamp(images + delta, 0, 1) - images
        else:
            delta = torch.zeros_like(images)
        delta.requires_grad = True

        for _ in range(self.steps):
            adv_images = torch.clamp(images + delta, 0, 1)
            logits = self.attacker(adv_images, text_features)

            # CW margin loss
            real_logits = torch.gather(logits, 1, labels.view(-1, 1)).squeeze(1)
            tmp_logits = logits.clone()
            tmp_logits.scatter_(1, labels.view(-1, 1), -float("inf"))
            max_other_logits, _ = torch.max(tmp_logits, dim=1)

            # Maximize: max_other - real (clamp at kappa)
            loss = torch.clamp(max_other_logits - real_logits, max=self.kappa).sum()

            if delta.grad is not None:
                delta.grad.zero_()
            loss.backward()

            grad = delta.grad.detach()
            delta.data = torch.clamp(
                delta + self.alpha * grad.sign(), -self.epsilon, self.epsilon
            )
            delta.grad.zero_()

        adv_images = torch.clamp(images + delta, 0, 1).detach()
        return adv_images
