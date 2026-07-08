"""
FastCSR — Optimized CSR Defense

Performance-optimized variant of CSR with:
  - Cached frequency-domain masks (avoid recomputing per batch)
  - Combined batch forward (detect both original and LPF images in one pass)
  - Early exit in purification loop (skip last backward)
  - Feature reuse when no adversarial input is detected

Drop-in replacement for CSRDefense with identical API and improved throughput.
"""

import math
import torch
import torch.nn as nn
import torch.fft
import torchvision.transforms as T
from transformers import CLIPProcessor, CLIPModel
from typing import Dict, Union

from ..config import CSRConfig, CLIP_MEAN, CLIP_STD


class FastCSRDefense(nn.Module):
    """
    Performance-optimized CSR defense.
    
    Key optimizations over CSRDefense:
      - LPF mask caching (recomputed only on resolution change)
      - Batch concatenation for detect (1 forward instead of 2)
      - Early break in purification to skip unnecessary backward pass
      - Feature passthrough when batch is all clean

    Args:
        model_name_or_obj: HuggingFace model name/path or CLIPModel instance.
        config: CSRConfig dataclass.
        device: Computation device.
    """

    def __init__(
        self,
        model_name_or_obj: Union[str, CLIPModel] = "openai/clip-vit-base-patch32",
        config: CSRConfig = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = device
        self.config = config or CSRConfig()

        if isinstance(model_name_or_obj, str):
            self.base_model = CLIPModel.from_pretrained(model_name_or_obj).to(device).eval()
            self.processor = CLIPProcessor.from_pretrained(model_name_or_obj)
        else:
            self.base_model = model_name_or_obj.to(device).eval()
            self.processor = CLIPProcessor.from_pretrained(model_name_or_obj.config.name_or_path)

        # Freeze all base model parameters
        for p in self.base_model.parameters():
            p.requires_grad = False

        self.clip_norm = T.Normalize(CLIP_MEAN, CLIP_STD)
        self.register_buffer("_clip_mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer("_clip_std", torch.tensor(CLIP_STD).view(1, 3, 1, 1))
        self.enable_defense = True
        self.last_detection_mask = None

        # LPF mask cache
        self._cached_mask = None
        self._cached_shape = None

    # ===================== Internal =====================

    def _get_lpf_mask(self, H: int, W: int) -> torch.Tensor:
        """Get or compute cached frequency-domain LPF mask."""
        if self._cached_mask is not None and self._cached_shape == (H, W):
            return self._cached_mask

        radius = self.config.lpf_radius
        cy, cx = H // 2, W // 2
        y = torch.arange(H, device=self.device) - cy
        x = torch.arange(W, device=self.device) - cx
        y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")
        dist_sq = y_grid ** 2 + x_grid ** 2

        if self.config.filter_type == "gaussian":
            mask = torch.exp(-dist_sq / (2 * radius ** 2))
        elif self.config.filter_type == "butterworth":
            n = self.config.butterworth_order
            mask = 1.0 / (1.0 + (torch.sqrt(dist_sq) / radius) ** (2 * n))
        else:
            mask = (dist_sq <= radius ** 2).float()

        # Pre-shift mask for efficient application (avoid fftshift per call)
        mask_shifted = torch.fft.ifftshift(mask).view(1, 1, H, W)
        self._cached_mask = mask_shifted
        self._cached_shape = (H, W)
        return mask_shifted

    def _apply_lpf(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Apply cached low-pass filter."""
        mask = self._get_lpf_mask(img_tensor.shape[2], img_tensor.shape[3])
        fft = torch.fft.fft2(img_tensor)
        img_back = torch.fft.ifft2(fft * mask)
        return torch.clamp(img_back.real, 0.0, 1.0)

    def _get_feats(self, img: torch.Tensor) -> torch.Tensor:
        """Extract normalized CLIP image features."""
        normed = self.clip_norm(img)
        feats = self.base_model.get_image_features(normed)
        return feats / feats.norm(dim=-1, keepdim=True)

    def _prepare_pixel_values(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Accept either raw [0, 1] images or CLIP-normalized tensors.

        FastCSR's detection/purification logic is defined in raw image space.
        Some evaluation code may accidentally pass processor-normalized pixel
        values; in that case, invert the normalization before applying defense.
        """
        pixel_values = pixel_values.to(self.device)

        if pixel_values.numel() == 0:
            return pixel_values

        # Heuristic: CLIP-normalized tensors almost always leave the [0, 1]
        # range, while raw image tensors from PIL/processor(do_normalize=False)
        # stay inside it.
        if pixel_values.min() < 0.0 or pixel_values.max() > 1.0:
            clip_std = self._clip_std.to(pixel_values.device)
            clip_mean = self._clip_mean.to(pixel_values.device)
            pixel_values = pixel_values * clip_std + clip_mean
            pixel_values = torch.clamp(pixel_values, 0.0, 1.0)

        return pixel_values

    def _detect(self, pixel_values: torch.Tensor):
        """
        Optimized detection: single forward pass for both original and LPF images.
        
        Returns:
            (is_adv_mask, lpf_feats, orig_feats)
        """
        lpf_imgs = self._apply_lpf(pixel_values)

        # Batch both through model in one forward pass
        combined = torch.cat([pixel_values, lpf_imgs], dim=0)
        combined_feats = self._get_feats(combined)
        orig_feats, lpf_feats = torch.chunk(combined_feats, 2, dim=0)

        sims = (orig_feats * lpf_feats).sum(dim=1)
        is_adv = sims < self.config.detect_thresh
        return is_adv, lpf_feats, orig_feats

    @torch.enable_grad()
    def _purify(
        self,
        img_tensor: torch.Tensor,
        target_feats: torch.Tensor,
        orig_adv_feats: torch.Tensor,
    ) -> torch.Tensor:
        """
        Optimized PGD purification with early exit.
        
        Skips the backward pass on the last iteration since the updated
        image would not be evaluated.
        """
        img_in = img_tensor.detach().clone()
        img_in.requires_grad = True

        best_img = img_tensor.detach().clone()
        best_score = torch.full((img_tensor.shape[0],), -float("inf"), device=self.device)
        steps = self.config.purify_steps

        for i in range(steps):
            curr_feats = self._get_feats(img_in)
            sim_lpf = (curr_feats * target_feats).sum(dim=1)
            sim_adv = (curr_feats * orig_adv_feats).sum(dim=1)
            current_score = sim_lpf - sim_adv

            # Track best
            better = current_score > best_score
            if better.any():
                best_score[better] = current_score[better].detach()
                best_img[better] = img_in[better].detach()

            # Early exit: skip backward on last step
            if i == steps - 1:
                break

            # Flip consistency: encourages spatial invariance of purified image
            img_flipped = torch.flip(img_in, dims=[3])
            flipped_feats = self._get_feats(img_flipped)
            flip_sim = (curr_feats * flipped_feats).sum(dim=1)
            lambda_flip = 0.2
            loss = -(current_score + lambda_flip * flip_sim).sum()
            if img_in.grad is not None:
                img_in.grad.zero_()
            loss.backward()

            with torch.no_grad():
                grad = img_in.grad
                alpha_t = self.config.purify_alpha * 0.5 * (1 + math.cos(math.pi * i / steps))
                img_in -= alpha_t * grad.sign()
                eta = torch.clamp(
                    img_in - img_tensor, -self.config.purify_eps, self.config.purify_eps
                )
                img_in.copy_(torch.clamp(img_tensor + eta, 0.0, 1.0))
                img_in.grad = None

        return best_img.detach()

    # ===================== Image Processing =====================

    def process_images_and_feats(self, pixel_values: torch.Tensor):
        """
        Process images with defense. Returns (processed_images, precomputed_feats).
        
        If all images are clean, returns precomputed features to avoid 
        redundant forward passes.
        """
        pixel_values = self._prepare_pixel_values(pixel_values)

        if not self.enable_defense:
            self.last_detection_mask = torch.zeros(
                pixel_values.shape[0], dtype=torch.bool, device=self.device
            )
            return pixel_values, None

        is_adv, lpf_feats, orig_feats = self._detect(pixel_values)
        self.last_detection_mask = is_adv

        if not is_adv.any():
            # All clean — return precomputed features
            return pixel_values, orig_feats

        final_imgs = pixel_values.clone()
        idx = torch.where(is_adv)[0]
        purified = self._purify(pixel_values[idx], lpf_feats[idx], orig_feats[idx])
        final_imgs[idx] = purified
        return final_imgs, None

    def process_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Process images with defense (without feature reuse)."""
        imgs, _ = self.process_images_and_feats(pixel_values)
        return imgs

    # ===================== Public API =====================

    def detect(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Detect adversarial samples."""
        pixel_values = self._prepare_pixel_values(pixel_values)
        is_adv, _, _ = self._detect(pixel_values)
        return is_adv

    # ===================== CLIP-Compatible API =====================

    def forward(self, input_ids=None, pixel_values=None, **kwargs):
        """CLIP forward with automatic defense."""
        if pixel_values is not None:
            clean_pixels, _ = self.process_images_and_feats(pixel_values)
            pixel_values = self.clip_norm(clean_pixels)

        return self.base_model(input_ids=input_ids, pixel_values=pixel_values, **kwargs)

    def predict_zero_shot(self, image_tensor: torch.Tensor, text_labels: list):
        """Zero-shot classification with defense."""
        text_inputs = self.processor(
            text=text_labels, return_tensors="pt", padding=True
        ).to(self.device)

        image_features = self.get_image_features(pixel_values=image_tensor)
        text_features = self.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        logit_scale = self.base_model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        probs = logits_per_image.softmax(dim=1)
        return probs, logits_per_image, self.last_detection_mask

    def get_image_features(self, pixel_values=None, **kwargs):
        """Get defended image features with feature reuse."""
        clean_pixels, precomputed_feats = self.process_images_and_feats(pixel_values)
        if precomputed_feats is not None:
            return precomputed_feats
        normed = self.clip_norm(clean_pixels)
        return self.base_model.get_image_features(pixel_values=normed, **kwargs)

    def get_text_features(self, input_ids=None, **kwargs):
        """Pass-through to base model text encoder."""
        return self.base_model.get_text_features(input_ids=input_ids, **kwargs)
