"""
CSR (CLIP Spectral Robustness) — Core Defense Method

A frequency-domain adversarial defense for CLIP models that performs:
  1. **Detection**: Compare original vs. low-pass filtered image features.
     Adversarial perturbations are concentrated in high-frequency components,
     so a large feature discrepancy indicates an adversarial input.
  2. **Purification**: PGD-based optimization that pushes adversarial 
     features toward the low-frequency reference while moving away from 
     the original adversarial features, with best-step tracking.

This module wraps a standard HuggingFace CLIPModel and provides a 
drop-in replacement with the same API (forward, get_image_features, etc.).

Usage:
    from csr.defense import CSRDefense

    defense = CSRDefense("openai/clip-vit-base-patch32")
    
    # Zero-shot prediction with automatic defense
    probs, logits, is_attack = defense.predict_zero_shot(images, text_labels)
    
    # Or use as a drop-in CLIP replacement
    outputs = defense(input_ids=text_ids, pixel_values=images)
"""

import torch
import torch.nn as nn
import torch.fft
import torchvision.transforms as T
from transformers import CLIPProcessor, CLIPModel
from typing import Dict, Union

from ..config import CSRConfig, CLIP_MEAN, CLIP_STD


class CSRDefense(nn.Module):
    """
    CSR (CLIP Spectral Robustness) adversarial defense.

    Wraps a CLIPModel with automatic adversarial detection and purification.
    After calling forward() or get_image_features(), the detection result
    is available via `model.last_detection_mask`.

    Args:
        model_name_or_obj: HuggingFace model name/path or a CLIPModel instance.
        config: CSRConfig dataclass with defense hyperparameters.
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

        # Load base CLIP model
        if isinstance(model_name_or_obj, str):
            print(f"[CSR] Loading base model: {model_name_or_obj}...")
            self.base_model = CLIPModel.from_pretrained(model_name_or_obj).to(device).eval()
            self.processor = CLIPProcessor.from_pretrained(model_name_or_obj)
        else:
            self.base_model = model_name_or_obj.to(device).eval()
            self.processor = CLIPProcessor.from_pretrained(model_name_or_obj.config.name_or_path)

        self.clip_norm = T.Normalize(CLIP_MEAN, CLIP_STD)
        self.enable_defense = True

        # State: detection mask from last inference
        self.last_detection_mask = None

    # ===================== Internal Logic =====================

    def _apply_lpf(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply low-pass filter in frequency domain.
        
        Supports: gaussian, butterworth, ideal filters.
        """
        B, C, H, W = img_tensor.shape
        radius = self.config.lpf_radius

        fft = torch.fft.fft2(img_tensor)
        fft_shift = torch.fft.fftshift(fft)

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
        else:  # ideal
            mask = (dist_sq <= radius ** 2).float()

        f_filtered = fft_shift * mask.view(1, 1, H, W)
        img_back = torch.fft.ifft2(torch.fft.ifftshift(f_filtered))
        return torch.clamp(img_back.real, 0.0, 1.0)

    def _get_feats(self, img: torch.Tensor) -> torch.Tensor:
        """Extract normalized CLIP image features."""
        normed = self.clip_norm(img)
        feats = self.base_model.get_image_features(normed)
        return feats / feats.norm(dim=-1, keepdim=True)

    def _detect(self, pixel_values: torch.Tensor):
        """
        Detect adversarial samples via spectral consistency.
        
        Returns:
            (is_adv_mask, lpf_feats): Boolean mask and low-freq features.
        """
        orig_feats = self._get_feats(pixel_values)
        lpf_imgs = self._apply_lpf(pixel_values)
        lpf_feats = self._get_feats(lpf_imgs)

        sims = (orig_feats * lpf_feats).sum(dim=1)
        is_adv = sims < self.config.detect_thresh
        return is_adv, lpf_feats

    def _purify(self, img_tensor: torch.Tensor, target_feats: torch.Tensor) -> torch.Tensor:
        """
        PGD-based purification with best-step tracking.
        
        Optimizes: maximize sim(current, lpf_target) - sim(current, orig_adv)
        within an epsilon ball around the input image.
        """
        with torch.enable_grad():
            img_in = img_tensor.detach().clone()
            img_in.requires_grad = True

            orig_adv_feats = self._get_feats(img_in).detach()

            best_img = img_tensor.detach().clone()
            best_score = torch.full((img_tensor.shape[0],), -float("inf"), device=self.device)

            for _ in range(self.config.purify_steps):
                curr_feats = self._get_feats(img_in)

                sim_lpf = (curr_feats * target_feats).sum(dim=1)
                sim_adv = (curr_feats * orig_adv_feats).sum(dim=1)
                current_score = sim_lpf - sim_adv

                # Track best
                better_mask = current_score > best_score
                if better_mask.any():
                    best_score[better_mask] = current_score[better_mask].detach()
                    best_img[better_mask] = img_in[better_mask].detach()

                # Gradient step
                loss = -current_score.sum()
                self.base_model.zero_grad()
                loss.backward()

                grad = img_in.grad.data
                img_in.data = img_in.data - self.config.purify_alpha * grad.sign()
                eta = torch.clamp(
                    img_in.data - img_tensor.data,
                    -self.config.purify_eps,
                    self.config.purify_eps,
                )
                img_in.data = torch.clamp(img_tensor.data + eta, 0.0, 1.0)
                img_in.grad = None

            # Final check
            with torch.no_grad():
                final_feats = self._get_feats(img_in)
                sim_lpf = (final_feats * target_feats).sum(dim=1)
                sim_adv = (final_feats * orig_adv_feats).sum(dim=1)
                final_score = sim_lpf - sim_adv
                better = final_score > best_score
                if better.any():
                    best_img[better] = img_in[better]

        return best_img.detach()

    # ===================== Image Processing =====================

    def process_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Full defense pipeline: detect + purify adversarial samples.
        
        Updates `self.last_detection_mask` as side effect.
        
        Args:
            pixel_values: [B, 3, H, W] in range [0, 1].
            
        Returns:
            Processed (potentially purified) images [B, 3, H, W].
        """
        if not self.enable_defense:
            self.last_detection_mask = torch.zeros(
                pixel_values.shape[0], dtype=torch.bool, device=self.device
            )
            return pixel_values

        pixel_values = pixel_values.to(self.device)

        # Detection
        is_adv, lpf_feats = self._detect(pixel_values)
        self.last_detection_mask = is_adv

        # Purification (only for detected adversarial samples)
        if not is_adv.any():
            return pixel_values

        final_imgs = pixel_values.clone()
        idx = torch.where(is_adv)[0]
        purified = self._purify(pixel_values[idx], lpf_feats[idx])
        final_imgs[idx] = purified
        return final_imgs

    # ===================== Public API =====================

    def detect(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Detect adversarial samples (standalone, without purification).

        Args:
            pixel_values: [B, 3, H, W] in range [0, 1].

        Returns:
            Boolean Tensor [B] (True = adversarial, False = clean).
        """
        pixel_values = pixel_values.to(self.device)
        is_adv, _ = self._detect(pixel_values)
        return is_adv

    # ===================== CLIP-Compatible API =====================

    def forward(self, input_ids=None, pixel_values=None, **kwargs):
        """
        CLIP forward with automatic adversarial defense.
        
        After calling, check `model.last_detection_mask` for detection results.
        """
        if pixel_values is not None:
            pixel_values = self.process_images(pixel_values)
            pixel_values = self.clip_norm(pixel_values)

        return self.base_model(input_ids=input_ids, pixel_values=pixel_values, **kwargs)

    def predict_zero_shot(self, image_tensor: torch.Tensor, text_labels: list):
        """
        Zero-shot classification with adversarial defense.

        Args:
            image_tensor: [B, 3, H, W] in range [0, 1].
            text_labels: List of class name strings.

        Returns:
            (probs, logits, is_attack_mask) tuple.
        """
        text_inputs = self.processor(
            text=text_labels, return_tensors="pt", padding=True
        ).to(self.device)

        outputs = self.forward(
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
            pixel_values=image_tensor,
        )

        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        return probs, logits_per_image, self.last_detection_mask

    def get_image_features(self, pixel_values=None, **kwargs):
        """Get defended image features."""
        clean_pixels = self.process_images(pixel_values)
        normed_pixels = self.clip_norm(clean_pixels)
        return self.base_model.get_image_features(pixel_values=normed_pixels, **kwargs)

    def get_text_features(self, input_ids=None, **kwargs):
        """Pass-through to base model text encoder."""
        return self.base_model.get_text_features(input_ids=input_ids, **kwargs)
