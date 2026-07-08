"""
APGD (Auto-PGD) attack for CLIP models.

Implements APGD-CE (CrossEntropy loss) with adaptive step size scheduling
and best-sample tracking, following the AutoAttack framework.
"""

import torch
import torch.nn as nn


class APGDAttack:
    """
    Auto-PGD (APGD-CE) L-inf adversarial attack.

    Features:
      - Adaptive step-size with checkpoint-based halving
      - Best-sample tracking across iterations
      - Random start initialization

    Args:
        attacker: CLIPAttacker instance.
        epsilon: Maximum perturbation (L-inf norm).
        steps: Total number of PGD iterations.
    """

    def __init__(
        self,
        attacker,
        epsilon: float = 4 / 255,
        steps: int = 30,
    ):
        self.attacker = attacker
        self.epsilon = epsilon
        self.steps = steps
        self.device = attacker.device

    @torch.enable_grad()
    def attack(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate adversarial examples using APGD-CE.

        Args:
            images: [B, 3, H, W] clean images in [0, 1].
            labels: [B] ground-truth class indices.
            text_features: [num_classes, embed_dim].

        Returns:
            Adversarial images [B, 3, H, W] in [0, 1].
        """
        images = images.to(self.device)
        labels = labels.to(self.device)
        batch_size = images.shape[0]

        # Initialize best tracking
        loss_best = -1e9 * torch.ones(batch_size, device=self.device)
        x_best = images.clone()

        # Random start within epsilon ball
        x_adv = images + torch.zeros_like(images).uniform_(-self.epsilon, self.epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        x_adv.requires_grad = True

        # Adaptive step size (per-sample)
        step_size = torch.ones(batch_size, 1, 1, 1, device=self.device) * (2.0 * self.epsilon)

        # Checkpoints for step-size reduction (following AutoAttack paper)
        checkpoints = [
            max(int(0.22 * self.steps), 1),
            max(int(0.48 * self.steps), 1),
            max(int(0.78 * self.steps), 1),
        ]

        loss_fn = nn.CrossEntropyLoss(reduction="none")

        for i in range(self.steps):
            logits = self.attacker(x_adv, text_features)
            loss_indiv = loss_fn(logits, labels)
            loss_total = loss_indiv.sum()

            self.attacker.zero_grad()
            loss_total.backward()
            grad = x_adv.grad.data

            # Track best results
            with torch.no_grad():
                is_better = loss_indiv > loss_best
                loss_best[is_better] = loss_indiv[is_better]

                current_x = x_adv.detach()
                if i == 0:
                    x_best = current_x.clone()
                else:
                    x_best[is_better] = current_x[is_better]

            # Step-size halving at checkpoints
            if (i + 1) in checkpoints:
                step_size = step_size * 0.5

            # Gradient update
            with torch.no_grad():
                x_next = x_adv + step_size * grad.sign()
                delta = torch.clamp(x_next - images, -self.epsilon, self.epsilon)
                x_adv.data = torch.clamp(images + delta, 0.0, 1.0)
                x_adv.grad.zero_()

        return x_best.detach()
