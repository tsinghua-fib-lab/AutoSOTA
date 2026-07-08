"""
Localized patch attack for CLIP models.

The attack learns a square RGB patch inserted at a fixed random location
per image. Only patch pixels are optimized.
"""

import torch
import torch.nn as nn


class PatchAttack:
    """
    Localized square patch attack.

    Args:
        attacker: CLIPAttacker instance.
        patch_size: Patch side length in pixels.
        steps: Number of optimization steps.
        lr: Adam learning rate.
        random_start: Whether to initialize the patch randomly.
        seed: Random seed for patch locations.
    """

    def __init__(
        self,
        attacker,
        patch_size: int = 48,
        steps: int = 200,
        lr: float = 5e-2,
        random_start: bool = True,
        seed: int = 42,
    ):
        self.attacker = attacker
        self.patch_size = patch_size
        self.steps = steps
        self.lr = lr
        self.random_start = random_start
        self.seed = seed
        self.device = attacker.device

    def _sample_locations(self, batch_size: int, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
        max_y = height - self.patch_size
        max_x = width - self.patch_size
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        ys = torch.randint(0, max_y + 1, (batch_size,), generator=generator)
        xs = torch.randint(0, max_x + 1, (batch_size,), generator=generator)
        return ys.to(self.device), xs.to(self.device)

    def _apply_patch(
        self,
        images: torch.Tensor,
        patch: torch.Tensor,
        ys: torch.Tensor,
        xs: torch.Tensor,
    ) -> torch.Tensor:
        adv = images.clone()
        for i in range(images.shape[0]):
            y0 = int(ys[i].item())
            x0 = int(xs[i].item())
            adv[i, :, y0 : y0 + self.patch_size, x0 : x0 + self.patch_size] = patch[i]
        return adv

    @torch.enable_grad()
    def attack(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate adversarial images with learned localized patches.
        """
        images = images.to(self.device)
        labels = labels.to(self.device)
        batch_size, channels, height, width = images.shape
        if self.patch_size > min(height, width):
            raise ValueError(
                f"patch_size={self.patch_size} exceeds input size {(height, width)}"
            )

        ys, xs = self._sample_locations(batch_size, height, width)
        loss_fn = nn.CrossEntropyLoss()

        if self.random_start:
            patch_param = torch.rand(
                batch_size,
                channels,
                self.patch_size,
                self.patch_size,
                device=self.device,
            )
        else:
            patch_param = torch.zeros(
                batch_size,
                channels,
                self.patch_size,
                self.patch_size,
                device=self.device,
            )
        patch_param.requires_grad = True
        optimizer = torch.optim.Adam([patch_param], lr=self.lr)

        best_imgs = images.detach().clone()
        best_loss = torch.full((batch_size,), -float("inf"), device=self.device)

        for _ in range(self.steps):
            patch = torch.sigmoid(patch_param)
            adv_images = self._apply_patch(images, patch, ys, xs)
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
            patch = torch.sigmoid(patch_param)
            adv_images = self._apply_patch(images, patch, ys, xs)
            logits = self.attacker(adv_images, text_features)
            per_sample_loss = nn.functional.cross_entropy(logits, labels, reduction="none")
            better_mask = per_sample_loss > best_loss
            if better_mask.any():
                best_imgs[better_mask] = adv_images[better_mask]

        return best_imgs.detach()
