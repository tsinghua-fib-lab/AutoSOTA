"""
TTC (Test-Time Correction) defense for HuggingFace CLIP models.

This module adapts the TTC notebook implementation into the same wrapper-style
API used by CSR/FastCSR so it can participate in the shared evaluation stack.
"""

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from typing import Union

from ..config import TTCConfig, CLIP_MEAN, CLIP_STD


class TTCDefense(nn.Module):
    """
    TTC defense wrapper for HuggingFace CLIP models.

    TTC operates in CLIP-normalized pixel space. For consistency with the rest
    of this project, the public API accepts either raw [0, 1] images or already
    normalized CLIP tensors and adapts as needed.
    """

    def __init__(
        self,
        model_name_or_obj: Union[str, CLIPModel] = "openai/clip-vit-base-patch32",
        config: TTCConfig = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = device
        self.config = config or TTCConfig()

        if isinstance(model_name_or_obj, str):
            self.base_model = CLIPModel.from_pretrained(model_name_or_obj).to(device).eval()
            self.processor = CLIPProcessor.from_pretrained(model_name_or_obj)
        else:
            self.base_model = model_name_or_obj.to(device).eval()
            self.processor = CLIPProcessor.from_pretrained(model_name_or_obj.config.name_or_path)

        for p in self.base_model.parameters():
            p.requires_grad = False

        self.register_buffer("_clip_mean", torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1))
        self.register_buffer("_clip_std", torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1))

        self.enable_defense = True
        self.last_detection_mask = None

    def _prepare_pixel_values(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Accept raw [0, 1] images or normalized CLIP tensors."""
        pixel_values = pixel_values.to(self.device)

        if pixel_values.numel() == 0:
            return pixel_values

        if pixel_values.min() < 0.0 or pixel_values.max() > 1.0:
            pixel_values = pixel_values * self._clip_std + self._clip_mean
            pixel_values = torch.clamp(pixel_values, 0.0, 1.0)

        return pixel_values

    def _normalize(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return (pixel_values - self._clip_mean) / self._clip_std

    def _denormalize(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return torch.clamp(pixel_values * self._clip_std + self._clip_mean, 0.0, 1.0)

    @torch.enable_grad()
    def _run_ttc(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        TTC in normalized CLIP space.

        Reproduces the notebook logic:
        1. Estimate a per-sample scheme sign from noisy feature consistency.
        2. Optimize a delta toward self-consistent features.
        3. Aggregate step deltas using TTC's weighted rule.
        """
        eps = self.config.eps
        alpha = self.config.alpha
        steps = self.config.steps
        tau_thres = self.config.tau
        beta = self.config.beta

        B = pixel_values.shape[0]
        delta = torch.empty_like(pixel_values).uniform_(-eps, eps)
        delta.requires_grad_(True)

        with torch.no_grad():
            clean_feats = self.base_model.get_image_features(pixel_values=pixel_values)
            clean_norm = torch.norm(clean_feats, dim=-1)

            noise = torch.randn_like(pixel_values) * eps
            noisy_feats = self.base_model.get_image_features(pixel_values=pixel_values + noise)

            diff_feat = noisy_feats - clean_feats
            diff_ratio = torch.norm(diff_feat, dim=-1) / (clean_norm + 1e-8)
            scheme_sign = (tau_thres - diff_ratio).sign()

        deltas_per_step = [delta.detach().clone()]

        for _ in range(steps):
            current_input = pixel_values + delta
            current_feats = self.base_model.get_image_features(pixel_values=current_input)
            l2_loss = ((current_feats - clean_feats) ** 2).sum()
            grad = torch.autograd.grad(l2_loss, delta, retain_graph=False, create_graph=False)[0]

            with torch.no_grad():
                delta.add_(alpha * grad.sign())
                delta.clamp_(-eps, eps)
                deltas_per_step.append(delta.detach().clone())

        delta_stack = torch.stack(deltas_per_step, dim=1)

        weights = torch.arange(steps + 1, dtype=torch.float32, device=pixel_values.device)
        weights = weights.unsqueeze(0).expand(B, -1)
        weights = torch.exp(scheme_sign.view(-1, 1) * weights * beta)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        weights_hard = torch.zeros_like(weights)
        weights_hard[:, 0] = 1.0
        final_weights = torch.where(scheme_sign.unsqueeze(1) > 0, weights, weights_hard)
        final_weights = final_weights.view(B, steps + 1, 1, 1, 1)

        delta_final = (final_weights * delta_stack).sum(dim=1)
        self.last_detection_mask = scheme_sign < 0
        return (pixel_values + delta_final).detach()

    def process_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Process raw images in [0, 1] through TTC and return raw images."""
        pixel_values = self._prepare_pixel_values(pixel_values)

        if not self.enable_defense:
            self.last_detection_mask = torch.zeros(
                pixel_values.shape[0], dtype=torch.bool, device=self.device
            )
            return pixel_values

        normed = self._normalize(pixel_values)
        purified = self._run_ttc(normed)
        return self._denormalize(purified)

    def detect(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return TTC's per-sample attack indicator."""
        pixel_values = self._prepare_pixel_values(pixel_values)
        normed = self._normalize(pixel_values)
        _ = self._run_ttc(normed)
        return self.last_detection_mask

    def forward(self, input_ids=None, pixel_values=None, **kwargs):
        if pixel_values is not None:
            pixel_values = self._prepare_pixel_values(pixel_values)
            normed_pixels = self._normalize(pixel_values)
            if self.enable_defense:
                normed_pixels = self._run_ttc(normed_pixels)
            else:
                self.last_detection_mask = torch.zeros(
                    pixel_values.shape[0], dtype=torch.bool, device=self.device
                )
            pixel_values = normed_pixels

        return self.base_model(input_ids=input_ids, pixel_values=pixel_values, **kwargs)

    def predict_zero_shot(self, image_tensor: torch.Tensor, text_labels: list):
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
        pixel_values = self._prepare_pixel_values(pixel_values)
        normed_pixels = self._normalize(pixel_values)
        if self.enable_defense:
            normed_pixels = self._run_ttc(normed_pixels)
        else:
            self.last_detection_mask = torch.zeros(
                pixel_values.shape[0], dtype=torch.bool, device=self.device
            )
        return self.base_model.get_image_features(pixel_values=normed_pixels, **kwargs)

    def get_text_features(self, input_ids=None, **kwargs):
        return self.base_model.get_text_features(input_ids=input_ids, **kwargs)
