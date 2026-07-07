"""Model wrapper integrating SAM2 with FerretNet for Ferret-SAM localization."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2
from sam2.modeling.backbones.image_encoder import ImageEncoder
from sam2.modeling.sam.mask_decoder import MaskDecoder
from sam2.modeling.sam.prompt_encoder import PromptEncoder
from sam2.modeling.sam2_base import SAM2Base
from sam2.utils.transforms import SAM2Transforms
from torchvision.ops import roi_align
from torchvision.transforms import v2

from model.adapters import NormGatedAdapter, SharedAdapter
from model.ferret_backbone import FerretBackbone
# from model.glp_module import GatedLPDPromptModule


logger = logging.getLogger(__name__)


class PromptCalibrator(nn.Module):
    """Sample-conditioned affine calibration for dense coarse prompts.

    The module is intentionally tiny and identity-initialized so existing
    checkpoints keep their original behavior at step zero.  It predicts one
    scale and one bias per sample from coarse-prompt statistics.
    """

    def __init__(
        self,
        hidden_dim: int = 16,
        identity_init: bool = True,
        max_delta_scale: float = 1.0,
        max_delta_bias: float = 2.0,
    ) -> None:
        super().__init__()
        self.max_delta_scale = float(max_delta_scale)
        self.max_delta_bias = float(max_delta_bias)
        self.net = nn.Sequential(
            nn.Linear(5, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 2),
        )
        if identity_init:
            final = self.net[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)

    def forward(self, prompt: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if prompt.ndim != 4:
            raise ValueError(f"PromptCalibrator expects [B,C,H,W], got shape={tuple(prompt.shape)}")
        dims = tuple(range(1, prompt.ndim))
        prompt_f = prompt.float()
        mean = prompt_f.mean(dim=dims)
        std = prompt_f.std(dim=dims, unbiased=False)
        min_v = prompt_f.amin(dim=dims)
        max_v = prompt_f.amax(dim=dims)
        prob_area = torch.sigmoid(prompt_f).mean(dim=dims)
        stats = torch.stack([mean, std, min_v, max_v, prob_area], dim=-1).to(dtype=prompt.dtype)
        delta = self.net(stats)
        delta_scale = torch.tanh(delta[:, 0]).view(-1, 1, 1, 1) * self.max_delta_scale
        delta_bias = torch.tanh(delta[:, 1]).view(-1, 1, 1, 1) * self.max_delta_bias
        scale = torch.exp(delta_scale).to(dtype=prompt.dtype)
        bias = delta_bias.to(dtype=prompt.dtype)
        calibrated = prompt * scale + bias
        diagnostics = {
            "scale": scale,
            "bias": bias,
            "delta_scale": delta_scale.to(dtype=prompt.dtype),
            "delta_bias": bias,
        }
        return calibrated, diagnostics


class ContextualPromptCalibrator(nn.Module):
    """Context-conditioned affine calibration for post-transform dense prompts.

    ``PromptCalibrator`` only sees the already centered/z-scored prompt.  In
    the SAM3+LAD runs this made it easy to suppress the prompt area but hard to
    know when the suppression should be relaxed.  This contextual variant keeps
    the same identity-initialized affine output, but conditions it on:

    * statistics of the transformed prompt sent to SAM,
    * statistics of the raw dense/coarse prompt before centering,
    * a compact summary of forensic features.

    The output remains one scale and one bias per sample so it is deliberately
    low capacity and safe to fine-tune from existing checkpoints.
    """

    def __init__(
        self,
        hidden_dim: int = 16,
        feature_channels: int = 384,
        identity_init: bool = True,
        max_delta_scale: float = 1.0,
        max_delta_bias: float = 2.0,
    ) -> None:
        super().__init__()
        # ``feature_channels`` is accepted for config/test compatibility.  We
        # summarize features with global scalar stats so the module can handle
        # either SAM2 or SAM3 forensic feature widths without shape-specific
        # checkpoint incompatibilities.
        self.feature_channels = int(feature_channels)
        self.max_delta_scale = float(max_delta_scale)
        self.max_delta_bias = float(max_delta_bias)
        self.net = nn.Sequential(
            nn.Linear(14, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 2),
        )
        if identity_init:
            final = self.net[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)

    @staticmethod
    def _stats5(tensor: torch.Tensor, batch_size: int, *, like: torch.Tensor) -> torch.Tensor:
        if tensor is None:
            return like.new_zeros(batch_size, 5)
        tensor_f = tensor.float()
        if tensor_f.shape[0] != batch_size:
            raise ValueError(
                "ContextualPromptCalibrator context batch mismatch: "
                f"expected {batch_size}, got {tensor_f.shape[0]}"
            )
        dims = tuple(range(1, tensor_f.ndim))
        mean = tensor_f.mean(dim=dims)
        std = tensor_f.std(dim=dims, unbiased=False)
        min_v = tensor_f.amin(dim=dims)
        max_v = tensor_f.amax(dim=dims)
        prob_area = torch.sigmoid(tensor_f).mean(dim=dims)
        return torch.stack([mean, std, min_v, max_v, prob_area], dim=-1).to(
            device=like.device,
            dtype=like.dtype,
        )

    @staticmethod
    def _feature_summary4(tensor: torch.Tensor, batch_size: int, *, like: torch.Tensor) -> torch.Tensor:
        if tensor is None:
            return like.new_zeros(batch_size, 4)
        tensor_f = tensor.float()
        if tensor_f.shape[0] != batch_size:
            raise ValueError(
                "ContextualPromptCalibrator feature batch mismatch: "
                f"expected {batch_size}, got {tensor_f.shape[0]}"
            )
        dims = tuple(range(1, tensor_f.ndim))
        mean = tensor_f.mean(dim=dims)
        std = tensor_f.std(dim=dims, unbiased=False)
        min_v = tensor_f.amin(dim=dims)
        max_v = tensor_f.amax(dim=dims)
        return torch.stack([mean, std, min_v, max_v], dim=-1).to(
            device=like.device,
            dtype=like.dtype,
        )

    def forward(
        self,
        prompt: torch.Tensor,
        context: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if prompt.ndim != 4:
            raise ValueError(
                f"ContextualPromptCalibrator expects [B,C,H,W], got shape={tuple(prompt.shape)}"
            )
        context = context or {}
        batch_size = int(prompt.shape[0])
        prompt_stats = self._stats5(prompt, batch_size, like=prompt)
        raw_stats = self._stats5(context.get("raw_prompt"), batch_size, like=prompt)
        feature_summary = self._feature_summary4(
            context.get("forensic_features"),
            batch_size,
            like=prompt,
        )
        context_stats = torch.cat([prompt_stats, raw_stats, feature_summary], dim=-1)
        delta = self.net(context_stats)
        delta_scale = torch.tanh(delta[:, 0]).view(-1, 1, 1, 1) * self.max_delta_scale
        delta_bias = torch.tanh(delta[:, 1]).view(-1, 1, 1, 1) * self.max_delta_bias
        scale = torch.exp(delta_scale).to(dtype=prompt.dtype)
        bias = delta_bias.to(dtype=prompt.dtype)
        calibrated = prompt * scale + bias
        diagnostics = {
            "scale": scale,
            "bias": bias,
            "delta_scale": delta_scale.to(dtype=prompt.dtype),
            "delta_bias": bias,
            "context_stats": context_stats,
            "feature_summary": feature_summary,
        }
        return calibrated, diagnostics


class FinalLogitCalibrator(nn.Module):
    """Sample-conditioned affine calibration for final mask logits.

    SAM3+LAD diagnostics showed that a single global threshold is a poor fit
    for the final logits: different samples benefit from different effective
    thresholds.  This module keeps the checkpoint-compatible identity behavior
    at initialization, but predicts one bounded scale and bias per sample from
    final-logit and prompt statistics.
    """

    def __init__(
        self,
        hidden_dim: int = 16,
        identity_init: bool = True,
        max_delta_scale: float = 0.0,
        max_delta_bias: float = 1.0,
    ) -> None:
        super().__init__()
        self.max_delta_scale = float(max_delta_scale)
        self.max_delta_bias = float(max_delta_bias)
        # final logits + coarse prompt + raw dense prompt + coarse mask + detection logit
        self.net = nn.Sequential(
            nn.Linear(21, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 2),
        )
        if identity_init:
            final = self.net[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)

    @staticmethod
    def _stats5(tensor: Optional[torch.Tensor], batch_size: int, *, like: torch.Tensor) -> torch.Tensor:
        if tensor is None:
            return like.new_zeros(batch_size, 5)
        tensor_f = tensor.float()
        if tensor_f.shape[0] != batch_size:
            raise ValueError(
                "FinalLogitCalibrator context batch mismatch: "
                f"expected {batch_size}, got {tensor_f.shape[0]}"
            )
        dims = tuple(range(1, tensor_f.ndim))
        return torch.stack(
            [
                tensor_f.mean(dim=dims),
                tensor_f.std(dim=dims, unbiased=False),
                tensor_f.amin(dim=dims),
                tensor_f.amax(dim=dims),
                torch.sigmoid(tensor_f).mean(dim=dims),
            ],
            dim=-1,
        ).to(device=like.device, dtype=like.dtype)

    @staticmethod
    def _scalar1(tensor: Optional[torch.Tensor], batch_size: int, *, like: torch.Tensor) -> torch.Tensor:
        if tensor is None:
            return like.new_zeros(batch_size, 1)
        tensor_f = tensor.float()
        if tensor_f.shape[0] != batch_size:
            raise ValueError(
                "FinalLogitCalibrator scalar context batch mismatch: "
                f"expected {batch_size}, got {tensor_f.shape[0]}"
            )
        return tensor_f.flatten(1).mean(dim=1, keepdim=True).to(device=like.device, dtype=like.dtype)

    def forward(
        self,
        logits: torch.Tensor,
        context: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if logits.ndim != 4:
            raise ValueError(f"FinalLogitCalibrator expects [B,C,H,W], got shape={tuple(logits.shape)}")
        context = context or {}
        batch_size = int(logits.shape[0])
        context_stats = torch.cat(
            [
                self._stats5(logits, batch_size, like=logits),
                self._stats5(context.get("coarse_prompt"), batch_size, like=logits),
                self._stats5(context.get("dense_prompt_mask"), batch_size, like=logits),
                self._stats5(context.get("coarse_mask"), batch_size, like=logits),
                self._scalar1(context.get("detection_logit"), batch_size, like=logits),
            ],
            dim=-1,
        )
        delta = self.net(context_stats)
        delta_scale = torch.tanh(delta[:, 0]).view(-1, 1, 1, 1) * self.max_delta_scale
        delta_bias = torch.tanh(delta[:, 1]).view(-1, 1, 1, 1) * self.max_delta_bias
        scale = torch.exp(delta_scale).to(dtype=logits.dtype)
        bias = delta_bias.to(dtype=logits.dtype)
        calibrated = logits * scale + bias
        diagnostics = {
            "pre_calibrator_logits": logits,
            "scale": scale,
            "bias": bias,
            "delta_scale": delta_scale.to(dtype=logits.dtype),
            "delta_bias": bias,
            "context_stats": context_stats,
        }
        return calibrated, diagnostics


class QuantileFinalLogitCalibrator(nn.Module):
    """Sample-conditioned affine final-logit calibrator with quantile features.

    Earlier final-logit calibrators only saw mean/std/min/max-like statistics
    and tended to become either no-ops or global shifts.  The R90 oracle shows
    the useful decision is effectively a per-sample threshold; quantiles expose
    the logit distribution shape needed to predict that threshold while keeping
    inference independent of ground truth.
    """

    _QUANTILES = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    def __init__(
        self,
        hidden_dim: int = 32,
        identity_init: bool = True,
        max_delta_scale: float = 0.0,
        max_delta_bias: float = 1.0,
    ) -> None:
        super().__init__()
        self.max_delta_scale = float(max_delta_scale)
        self.max_delta_bias = float(max_delta_bias)
        self.net = nn.Sequential(
            nn.Linear(57, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 2),
        )
        if identity_init:
            final = self.net[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)

    @staticmethod
    def _stats14(tensor: Optional[torch.Tensor], batch_size: int, *, like: torch.Tensor) -> torch.Tensor:
        if tensor is None:
            return like.new_zeros(batch_size, 14)
        tensor_f = tensor.detach().float() if not tensor.requires_grad else tensor.float()
        if tensor_f.shape[0] != batch_size:
            raise ValueError(
                "QuantileFinalLogitCalibrator context batch mismatch: "
                f"expected {batch_size}, got {tensor_f.shape[0]}"
            )
        flat = tensor_f.flatten(1)
        q = QuantileFinalLogitCalibrator._QUANTILES.to(device=flat.device, dtype=flat.dtype)
        quantiles = torch.quantile(flat, q, dim=1).transpose(0, 1)
        stats = torch.stack(
            [
                flat.mean(dim=1),
                flat.std(dim=1, unbiased=False),
                flat.amin(dim=1),
                flat.amax(dim=1),
                torch.sigmoid(flat).mean(dim=1),
            ],
            dim=-1,
        )
        return torch.cat([stats, quantiles], dim=-1).to(device=like.device, dtype=like.dtype)

    @staticmethod
    def _scalar1(tensor: Optional[torch.Tensor], batch_size: int, *, like: torch.Tensor) -> torch.Tensor:
        return FinalLogitCalibrator._scalar1(tensor, batch_size, like=like)

    def forward(
        self,
        logits: torch.Tensor,
        context: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if logits.ndim != 4:
            raise ValueError(f"QuantileFinalLogitCalibrator expects [B,C,H,W], got shape={tuple(logits.shape)}")
        context = context or {}
        batch_size = int(logits.shape[0])
        context_stats = torch.cat(
            [
                self._stats14(logits, batch_size, like=logits),
                self._stats14(context.get("coarse_prompt"), batch_size, like=logits),
                self._stats14(context.get("dense_prompt_mask"), batch_size, like=logits),
                self._stats14(context.get("coarse_mask"), batch_size, like=logits),
                self._scalar1(context.get("detection_logit"), batch_size, like=logits),
            ],
            dim=-1,
        )
        delta = self.net(context_stats)
        delta_scale = torch.tanh(delta[:, 0]).view(-1, 1, 1, 1) * self.max_delta_scale
        delta_bias = torch.tanh(delta[:, 1]).view(-1, 1, 1, 1) * self.max_delta_bias
        scale = torch.exp(delta_scale).to(dtype=logits.dtype)
        bias = delta_bias.to(dtype=logits.dtype)
        calibrated = logits * scale + bias
        diagnostics = {
            "pre_calibrator_logits": logits,
            "scale": scale,
            "bias": bias,
            "delta_scale": delta_scale.to(dtype=logits.dtype),
            "delta_bias": bias,
            "context_stats": context_stats,
        }
        return calibrated, diagnostics


class SemanticSpatialFinalLogitCalibrator(nn.Module):
    """Spatial final-logit residual calibrated by prompt, LAD, and SAM context.

    Scalar final-logit calibration had a large oracle gap but learned global
    expansion/suppression.  This module predicts a bounded per-pixel residual
    after the SAM decoder using the final logits, prompt maps, forensic feature
    summaries, and SAM decoder feature summaries.  Identity initialization keeps
    existing checkpoints unchanged at step zero.
    """

    def __init__(
        self,
        hidden_channels: int = 8,
        identity_init: bool = True,
        max_residual: float = 1.0,
        gate_init: float = 0.1,
        gate_max: float = 1.0,
    ) -> None:
        super().__init__()
        self.max_residual = float(max_residual)
        self.gate_max = float(gate_max)
        gate_init_clamped = min(max(float(gate_init), 1e-4), 1.0 - 1e-4)
        hidden = int(hidden_channels)
        in_channels = 12
        self.residual_net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=3, padding=1),
        )
        self.gate_net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        if identity_init:
            final = self.residual_net[-1]
            if isinstance(final, nn.Conv2d):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)
            gate_final = self.gate_net[-1]
            if isinstance(gate_final, nn.Conv2d):
                nn.init.zeros_(gate_final.weight)
                nn.init.constant_(
                    gate_final.bias,
                    math.log(gate_init_clamped / (1.0 - gate_init_clamped)),
                )

    @staticmethod
    def _resize_one_channel(
        tensor: Optional[torch.Tensor],
        *,
        size: Tuple[int, int],
        like: torch.Tensor,
    ) -> torch.Tensor:
        return ContextualPromptRefiner._resize_one_channel(tensor, size=size, like=like)

    @staticmethod
    def _feature_pair(
        tensor: Optional[torch.Tensor],
        *,
        size: Tuple[int, int],
        like: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return ContextualPromptRefiner._feature_maps(tensor, size=size, like=like)

    def forward(
        self,
        logits: torch.Tensor,
        context: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if logits.ndim != 4:
            raise ValueError(
                f"SemanticSpatialFinalLogitCalibrator expects [B,C,H,W], got shape={tuple(logits.shape)}"
            )
        context = context or {}
        size = (int(logits.shape[-2]), int(logits.shape[-1]))
        coarse_prompt = self._resize_one_channel(context.get("coarse_prompt"), size=size, like=logits)
        dense_prompt = self._resize_one_channel(context.get("dense_prompt_mask"), size=size, like=logits)
        coarse_mask = self._resize_one_channel(context.get("coarse_mask"), size=size, like=logits)
        forensic_mean, forensic_std = self._feature_pair(context.get("forensic_features"), size=size, like=logits)
        image_mean, image_std = self._feature_pair(context.get("image_embeddings"), size=size, like=logits)
        high_res = context.get("high_res_features")
        high0 = high_res[0] if isinstance(high_res, (list, tuple)) and len(high_res) > 0 else None
        high1 = high_res[1] if isinstance(high_res, (list, tuple)) and len(high_res) > 1 else None
        high0_mean, high0_std = self._feature_pair(high0, size=size, like=logits)
        high1_mean, high1_std = self._feature_pair(high1, size=size, like=logits)
        context_maps = torch.cat(
            [
                logits,
                coarse_prompt,
                dense_prompt,
                coarse_mask,
                forensic_mean,
                forensic_std,
                image_mean,
                image_std,
                high0_mean,
                high0_std,
                high1_mean,
                high1_std,
            ],
            dim=1,
        )
        residual_raw = self.residual_net(context_maps)
        residual = torch.tanh(residual_raw) * self.max_residual
        gate_logits = self.gate_net(context_maps)
        spatial_gate = torch.sigmoid(gate_logits) * self.gate_max
        delta = residual * spatial_gate
        diagnostics = {
            "pre_calibrator_logits": logits,
            "context_maps": context_maps,
            "residual": residual.to(dtype=logits.dtype),
            "spatial_gate": spatial_gate.to(dtype=logits.dtype),
            "delta_bias": delta.to(dtype=logits.dtype),
            "bias": delta.to(dtype=logits.dtype),
            "scale": torch.ones(
                logits.shape[0],
                1,
                1,
                1,
                device=logits.device,
                dtype=logits.dtype,
            ),
        }
        return logits + delta.to(dtype=logits.dtype), diagnostics


class DualPromptFusionGate(nn.Module):
    """Sample-conditioned fusion for SAM3 legacy/native prompt decoders."""

    def __init__(self, hidden_dim: int = 16, identity_init: bool = True) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )
        if identity_init:
            final = self.net[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)

    @staticmethod
    def _stats5(tensor: torch.Tensor, batch_size: int, *, like: torch.Tensor) -> torch.Tensor:
        if tensor is None:
            return like.new_zeros(batch_size, 5)
        tensor_f = tensor.float()
        dims = tuple(range(1, tensor_f.ndim))
        return torch.stack(
            [
                tensor_f.mean(dim=dims),
                tensor_f.std(dim=dims, unbiased=False),
                tensor_f.amin(dim=dims),
                tensor_f.amax(dim=dims),
                torch.sigmoid(tensor_f).mean(dim=dims),
            ],
            dim=-1,
        ).to(device=like.device, dtype=like.dtype)

    @staticmethod
    def _feature_summary4(tensor: torch.Tensor, batch_size: int, *, like: torch.Tensor) -> torch.Tensor:
        if tensor is None:
            return like.new_zeros(batch_size, 4)
        tensor_f = tensor.float()
        dims = tuple(range(1, tensor_f.ndim))
        return torch.stack(
            [
                tensor_f.mean(dim=dims),
                tensor_f.std(dim=dims, unbiased=False),
                tensor_f.amin(dim=dims),
                tensor_f.amax(dim=dims),
            ],
            dim=-1,
        ).to(device=like.device, dtype=like.dtype)

    def forward(
        self,
        *,
        prompt_source: torch.Tensor,
        forensic_features: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(prompt_source.shape[0])
        stats = torch.cat(
            [
                self._stats5(prompt_source, batch_size, like=prompt_source),
                self._feature_summary4(forensic_features, batch_size, like=prompt_source),
            ],
            dim=-1,
        )
        return torch.sigmoid(self.net(stats)).view(-1, 1, 1, 1)


class SpatialDualPromptFusionGate(nn.Module):
    """Pixel-conditioned fusion for SAM3 legacy/native prompt decoder logits.

    The dummy no-point SAM3 branch tends to expand masks: useful on
    under-covered edits, harmful on high-FP samples.  A scalar gate can only
    choose one global tradeoff per image.  This module predicts a spatial gate
    at decoder-logit resolution so training can borrow native/dummy expansion
    only where local evidence supports it.
    """

    def __init__(
        self,
        hidden_channels: int = 8,
        init_prob: float = 0.05,
        identity_init: bool = True,
    ) -> None:
        super().__init__()
        hidden = int(hidden_channels)
        self.net = nn.Sequential(
            nn.Conv2d(6, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        if identity_init:
            final = self.net[-1]
            if isinstance(final, nn.Conv2d):
                init = min(max(float(init_prob), 1e-4), 1.0 - 1e-4)
                nn.init.zeros_(final.weight)
                nn.init.constant_(final.bias, math.log(init / (1.0 - init)))

    @staticmethod
    def _resize_one_channel(
        tensor: Optional[torch.Tensor],
        *,
        size: Tuple[int, int],
        like: torch.Tensor,
    ) -> torch.Tensor:
        if tensor is None:
            return like.new_zeros((int(like.shape[0]), 1, int(size[0]), int(size[1])))
        if tensor.ndim == 2:
            tensor = tensor.view(tensor.shape[0], 1, 1, 1)
        elif tensor.ndim == 3:
            tensor = tensor.unsqueeze(1)
        if tensor.shape[1] != 1:
            tensor = tensor.mean(dim=1, keepdim=True)
        tensor = tensor.to(device=like.device, dtype=like.dtype)
        if tensor.shape[-2:] != size:
            tensor = F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
        return tensor

    @staticmethod
    def _feature_maps(
        tensor: Optional[torch.Tensor],
        *,
        size: Tuple[int, int],
        like: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if tensor is None:
            zero = like.new_zeros((int(like.shape[0]), 1, int(size[0]), int(size[1])))
            return zero, zero
        tensor_f = tensor.to(device=like.device, dtype=like.dtype)
        mean = tensor_f.mean(dim=1, keepdim=True)
        std = tensor_f.float().std(dim=1, keepdim=True, unbiased=False).to(dtype=like.dtype)
        if mean.shape[-2:] != size:
            mean = F.interpolate(mean, size=size, mode="bilinear", align_corners=False)
            std = F.interpolate(std, size=size, mode="bilinear", align_corners=False)
        return mean, std

    def forward(
        self,
        *,
        prompt_source: torch.Tensor,
        forensic_features: torch.Tensor,
        legacy_logits: torch.Tensor,
        native_logits: torch.Tensor,
    ) -> torch.Tensor:
        size = (int(legacy_logits.shape[-2]), int(legacy_logits.shape[-1]))
        prompt = self._resize_one_channel(prompt_source, size=size, like=legacy_logits)
        legacy = self._resize_one_channel(legacy_logits, size=size, like=legacy_logits)
        native = self._resize_one_channel(native_logits, size=size, like=legacy_logits)
        diff = native - legacy
        feat_mean, feat_std = self._feature_maps(forensic_features, size=size, like=legacy_logits)
        context = torch.cat([prompt, legacy, native, diff, feat_mean, feat_std], dim=1)
        return torch.sigmoid(self.net(context)).to(dtype=legacy_logits.dtype)


class PromptRefiner(nn.Module):
    """Identity-initialized spatial residual refiner for dense prompts."""

    def __init__(self, hidden_channels: int = 8, identity_init: bool = True) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, int(hidden_channels), kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(int(hidden_channels), 1, kernel_size=3, padding=1),
        )
        if identity_init:
            final = self.net[-1]
            if isinstance(final, nn.Conv2d):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)

    def forward(self, prompt: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        residual = self.net(prompt)
        return prompt + residual, {
            "residual": residual,
            "residual_abs_mean": residual.detach().abs().mean(),
        }


class ContextualPromptRefiner(nn.Module):
    """Spatial, context-conditioned residual refiner for dense SAM prompts.

    Unlike the affine calibrators, this module can make local corrections.  It
    consumes the already transformed prompt plus resized raw-prompt and
    forensic-feature summary maps, then predicts a bounded spatial residual.
    The final convolution is zero-initialized so existing checkpoints preserve
    their prompt behavior at step zero.
    """

    def __init__(
        self,
        hidden_channels: int = 8,
        identity_init: bool = True,
        max_residual: float = 0.5,
    ) -> None:
        super().__init__()
        self.max_residual = float(max_residual)
        self.net = nn.Sequential(
            nn.Conv2d(4, int(hidden_channels), kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(int(hidden_channels), int(hidden_channels), kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(int(hidden_channels), 1, kernel_size=3, padding=1),
        )
        if identity_init:
            final = self.net[-1]
            if isinstance(final, nn.Conv2d):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)

    @staticmethod
    def _resize_one_channel(
        tensor: Optional[torch.Tensor],
        *,
        size: Tuple[int, int],
        like: torch.Tensor,
    ) -> torch.Tensor:
        if tensor is None:
            return like.new_zeros(like.shape[0], 1, *size)
        tensor_f = tensor.float()
        if tensor_f.shape[0] != like.shape[0]:
            raise ValueError(
                "ContextualPromptRefiner context batch mismatch: "
                f"expected {like.shape[0]}, got {tensor_f.shape[0]}"
            )
        if tensor_f.shape[1] != 1:
            tensor_f = tensor_f.mean(dim=1, keepdim=True)
        if tensor_f.shape[-2:] != size:
            tensor_f = F.interpolate(
                tensor_f,
                size=size,
                mode="bilinear",
                align_corners=False,
            )
        return tensor_f.to(device=like.device, dtype=like.dtype)

    @staticmethod
    def _feature_maps(
        tensor: Optional[torch.Tensor],
        *,
        size: Tuple[int, int],
        like: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if tensor is None:
            zeros = like.new_zeros(like.shape[0], 1, *size)
            return zeros, zeros
        tensor_f = tensor.float()
        if tensor_f.shape[0] != like.shape[0]:
            raise ValueError(
                "ContextualPromptRefiner feature batch mismatch: "
                f"expected {like.shape[0]}, got {tensor_f.shape[0]}"
            )
        mean_map = tensor_f.mean(dim=1, keepdim=True)
        std_map = tensor_f.std(dim=1, keepdim=True, unbiased=False)
        if mean_map.shape[-2:] != size:
            mean_map = F.interpolate(mean_map, size=size, mode="bilinear", align_corners=False)
            std_map = F.interpolate(std_map, size=size, mode="bilinear", align_corners=False)
        return (
            mean_map.to(device=like.device, dtype=like.dtype),
            std_map.to(device=like.device, dtype=like.dtype),
        )

    def forward(
        self,
        prompt: torch.Tensor,
        context: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if prompt.ndim != 4:
            raise ValueError(f"ContextualPromptRefiner expects [B,C,H,W], got shape={tuple(prompt.shape)}")
        context = context or {}
        size = (int(prompt.shape[-2]), int(prompt.shape[-1]))
        raw_prompt = self._resize_one_channel(context.get("raw_prompt"), size=size, like=prompt)
        feat_mean, feat_std = self._feature_maps(context.get("forensic_features"), size=size, like=prompt)
        context_maps = torch.cat([prompt, raw_prompt, feat_mean, feat_std], dim=1)
        residual_unbounded = self.net(context_maps)
        residual = torch.tanh(residual_unbounded) * self.max_residual
        return prompt + residual.to(dtype=prompt.dtype), {
            "residual": residual.to(dtype=prompt.dtype),
            "residual_unbounded": residual_unbounded.to(dtype=prompt.dtype),
            "residual_abs_mean": residual.detach().abs().mean(),
            "context_maps": context_maps,
        }


class GatedContextualPromptRefiner(nn.Module):
    """Contextual prompt refiner with a learned per-sample residual gate.

    The plain contextual refiner can learn a broadly positive spatial residual,
    which helped CoCoGLIDE/AutoSplice/SID but increased MagicBrush false
    positives.  This variant keeps the same local residual path but scales it
    by a bounded image-level gate predicted from prompt/feature statistics.  It
    is still identity-initialized: the residual conv starts at zero, and the
    gate starts small, so fine-tuning can learn selective expansion rather than
    immediately changing every dataset in the same direction.
    """

    def __init__(
        self,
        hidden_channels: int = 8,
        identity_init: bool = True,
        max_residual: float = 0.5,
        gate_init: float = 0.2,
        gate_max: float = 1.0,
    ) -> None:
        super().__init__()
        self.max_residual = float(max_residual)
        self.gate_max = float(gate_max)
        gate_init_clamped = min(max(float(gate_init), 1e-4), 1.0 - 1e-4)
        self.net = nn.Sequential(
            nn.Conv2d(4, int(hidden_channels), kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(int(hidden_channels), int(hidden_channels), kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(int(hidden_channels), 1, kernel_size=3, padding=1),
        )
        self.gate_net = nn.Sequential(
            nn.Linear(20, int(hidden_channels)),
            nn.GELU(),
            nn.Linear(int(hidden_channels), 1),
        )
        if identity_init:
            final = self.net[-1]
            if isinstance(final, nn.Conv2d):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)
            gate_final = self.gate_net[-1]
            if isinstance(gate_final, nn.Linear):
                # The residual path is zero-initialized, so a tiny
                # sample-dependent gate does not change the prompt at step
                # zero.  Keeping this weight non-zero avoids the gate staying
                # globally constant until residuals have already drifted.
                nn.init.normal_(gate_final.weight, mean=0.0, std=1e-3)
                gate_bias = math.log(gate_init_clamped / (1.0 - gate_init_clamped))
                nn.init.constant_(gate_final.bias, gate_bias)

    @staticmethod
    def _resize_one_channel(
        tensor: Optional[torch.Tensor],
        *,
        size: Tuple[int, int],
        like: torch.Tensor,
    ) -> torch.Tensor:
        return ContextualPromptRefiner._resize_one_channel(tensor, size=size, like=like)

    @staticmethod
    def _feature_maps(
        tensor: Optional[torch.Tensor],
        *,
        size: Tuple[int, int],
        like: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return ContextualPromptRefiner._feature_maps(tensor, size=size, like=like)

    @staticmethod
    def _map_stats(tensor: torch.Tensor) -> torch.Tensor:
        tensor_f = tensor.float()
        dims = tuple(range(1, tensor_f.ndim))
        return torch.stack(
            [
                tensor_f.mean(dim=dims),
                tensor_f.std(dim=dims, unbiased=False),
                tensor_f.amin(dim=dims),
                tensor_f.amax(dim=dims),
                torch.sigmoid(tensor_f).mean(dim=dims),
            ],
            dim=-1,
        )

    def forward(
        self,
        prompt: torch.Tensor,
        context: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if prompt.ndim != 4:
            raise ValueError(f"GatedContextualPromptRefiner expects [B,C,H,W], got shape={tuple(prompt.shape)}")
        context = context or {}
        size = (int(prompt.shape[-2]), int(prompt.shape[-1]))
        raw_prompt = self._resize_one_channel(context.get("raw_prompt"), size=size, like=prompt)
        feat_mean, feat_std = self._feature_maps(context.get("forensic_features"), size=size, like=prompt)
        context_maps = torch.cat([prompt, raw_prompt, feat_mean, feat_std], dim=1)
        gate_stats = torch.cat(
            [
                self._map_stats(prompt),
                self._map_stats(raw_prompt),
                self._map_stats(feat_mean),
                self._map_stats(feat_std),
            ],
            dim=-1,
        ).to(device=prompt.device, dtype=prompt.dtype)
        gate_logits = self.gate_net(gate_stats)
        gate = torch.sigmoid(gate_logits).view(-1, 1, 1, 1) * self.gate_max
        residual_unbounded = self.net(context_maps)
        residual = torch.tanh(residual_unbounded) * self.max_residual * gate
        return prompt + residual.to(dtype=prompt.dtype), {
            "residual": residual.to(dtype=prompt.dtype),
            "residual_unbounded": residual_unbounded.to(dtype=prompt.dtype),
            "residual_abs_mean": residual.detach().abs().mean(),
            "context_maps": context_maps,
            "gate": gate.to(dtype=prompt.dtype),
            "gate_logits": gate_logits.to(dtype=prompt.dtype),
            "gate_stats": gate_stats,
        }


class SpatialContextPromptRefiner(nn.Module):
    """Pixel-gated contextual prompt refiner for SAM dense prompts.

    Earlier scalar calibrators and image-level gates could only expand or
    suppress the whole prompt at once.  This refiner keeps the same
    identity-preserving safety properties, but predicts a local residual and a
    local residual gate from richer prompt/forensic context.  The final
    residual convolution is zero-initialized, so loading it on top of existing
    R90/R101 checkpoints starts with exactly the old prompt map.
    """

    def __init__(
        self,
        hidden_channels: int = 8,
        identity_init: bool = True,
        max_residual: float = 0.5,
        gate_init: float = 0.2,
        gate_max: float = 1.0,
    ) -> None:
        super().__init__()
        self.max_residual = float(max_residual)
        self.gate_max = float(gate_max)
        gate_init_clamped = min(max(float(gate_init), 1e-4), 1.0 - 1e-4)
        hidden = int(hidden_channels)
        in_channels = 6
        self.residual_net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=3, padding=1),
        )
        self.gate_net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        if identity_init:
            residual_final = self.residual_net[-1]
            if isinstance(residual_final, nn.Conv2d):
                nn.init.zeros_(residual_final.weight)
                nn.init.zeros_(residual_final.bias)
            gate_final = self.gate_net[-1]
            if isinstance(gate_final, nn.Conv2d):
                nn.init.zeros_(gate_final.weight)
                gate_bias = math.log(gate_init_clamped / (1.0 - gate_init_clamped))
                nn.init.constant_(gate_final.bias, gate_bias)

    @staticmethod
    def _resize_one_channel(
        tensor: Optional[torch.Tensor],
        *,
        size: Tuple[int, int],
        like: torch.Tensor,
    ) -> torch.Tensor:
        return ContextualPromptRefiner._resize_one_channel(tensor, size=size, like=like)

    @staticmethod
    def _feature_maps(
        tensor: Optional[torch.Tensor],
        *,
        size: Tuple[int, int],
        like: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return ContextualPromptRefiner._feature_maps(tensor, size=size, like=like)

    def forward(
        self,
        prompt: torch.Tensor,
        context: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if prompt.ndim != 4:
            raise ValueError(f"SpatialContextPromptRefiner expects [B,C,H,W], got shape={tuple(prompt.shape)}")
        context = context or {}
        size = (int(prompt.shape[-2]), int(prompt.shape[-1]))
        raw_prompt = self._resize_one_channel(context.get("raw_prompt"), size=size, like=prompt)
        dense_prompt = self._resize_one_channel(context.get("dense_prompt_mask"), size=size, like=prompt)
        coarse_mask = self._resize_one_channel(context.get("coarse_mask"), size=size, like=prompt)
        feat_mean, feat_std = self._feature_maps(context.get("forensic_features"), size=size, like=prompt)
        context_maps = torch.cat(
            [prompt, raw_prompt, dense_prompt, coarse_mask, feat_mean, feat_std],
            dim=1,
        )
        residual_unbounded = self.residual_net(context_maps)
        gate_logits = self.gate_net(context_maps)
        spatial_gate = torch.sigmoid(gate_logits) * self.gate_max
        residual = torch.tanh(residual_unbounded) * self.max_residual * spatial_gate
        return prompt + residual.to(dtype=prompt.dtype), {
            "residual": residual.to(dtype=prompt.dtype),
            "residual_unbounded": residual_unbounded.to(dtype=prompt.dtype),
            "residual_abs_mean": residual.detach().abs().mean(),
            "context_maps": context_maps,
            "spatial_gate": spatial_gate.to(dtype=prompt.dtype),
            "gate": spatial_gate.to(dtype=prompt.dtype),
            "gate_logits": gate_logits.to(dtype=prompt.dtype),
        }


class RawBlendPromptComposer(nn.Module):
    """Locally blend the transformed SAM prompt back toward raw dense logits.

    The signed-tribranch diagnostics showed that the raw dense prompt can be
    much more precise than the post ``center+bias`` prompt, but replacing the
    transformed prompt with raw logits globally hurts recall.  This composer is
    therefore constrained to the stable direction:

    ``prompt + gate * (raw_prompt - prompt)``.

    A small spatial gate starts near identity and can learn where to trust the
    raw prompt for false-positive suppression while preserving the transformed
    prompt elsewhere for recall.
    """

    def __init__(
        self,
        hidden_channels: int = 8,
        identity_init: bool = True,
        gate_init: float = 0.05,
        gate_max: float = 0.5,
    ) -> None:
        super().__init__()
        self.gate_max = float(gate_max)
        gate_fraction = min(max(float(gate_init), 1e-4), 1.0 - 1e-4)
        hidden = int(hidden_channels)
        in_channels = 6
        self.gate_net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=3, padding=1),
        )
        if identity_init:
            final = self.gate_net[-1]
            if isinstance(final, nn.Conv2d):
                nn.init.zeros_(final.weight)
                final.bias.data.fill_(math.log(gate_fraction / (1.0 - gate_fraction)))

    @staticmethod
    def _resize_one_channel(
        tensor: Optional[torch.Tensor],
        *,
        size: Tuple[int, int],
        like: torch.Tensor,
        fallback: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if tensor is None:
            if fallback is not None:
                return fallback.to(device=like.device, dtype=like.dtype)
            return like.new_zeros((like.shape[0], 1, *size))
        tensor_f = tensor.to(device=like.device, dtype=like.dtype)
        if tensor_f.ndim == 3:
            tensor_f = tensor_f.unsqueeze(1)
        if tensor_f.shape[1] != 1:
            tensor_f = tensor_f.mean(dim=1, keepdim=True)
        if tensor_f.shape[-2:] != size:
            tensor_f = F.interpolate(tensor_f, size=size, mode="bilinear", align_corners=False)
        return tensor_f

    @staticmethod
    def _feature_maps(
        tensor: Optional[torch.Tensor],
        *,
        size: Tuple[int, int],
        like: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return SpatialContextPromptRefiner._feature_maps(tensor, size=size, like=like)

    def forward(
        self,
        prompt: torch.Tensor,
        context: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if prompt.ndim != 4:
            raise ValueError(f"RawBlendPromptComposer expects [B,C,H,W], got shape={tuple(prompt.shape)}")
        context = context or {}
        size = (int(prompt.shape[-2]), int(prompt.shape[-1]))
        raw_prompt = self._resize_one_channel(
            context.get("raw_prompt"),
            size=size,
            like=prompt,
            fallback=prompt,
        )
        dense_prompt = self._resize_one_channel(context.get("dense_prompt_mask"), size=size, like=prompt)
        coarse_mask = self._resize_one_channel(context.get("coarse_mask"), size=size, like=prompt)
        feat_mean, feat_std = self._feature_maps(context.get("forensic_features"), size=size, like=prompt)
        context_maps = torch.cat(
            [prompt, raw_prompt, raw_prompt - prompt, dense_prompt, coarse_mask, feat_mean + feat_std],
            dim=1,
        )
        gate_logits = self.gate_net(context_maps)
        blend_gate = torch.sigmoid(gate_logits) * self.gate_max
        blend_delta = (raw_prompt - prompt) * blend_gate
        refined = prompt + blend_delta.to(dtype=prompt.dtype)
        return refined, {
            "blend_gate": blend_gate.to(dtype=prompt.dtype),
            "blend_delta": blend_delta.to(dtype=prompt.dtype),
            "raw_prompt_resized": raw_prompt.to(dtype=prompt.dtype),
            "context_maps": context_maps,
            "gate_logits": gate_logits.to(dtype=prompt.dtype),
            "gate": blend_gate.to(dtype=prompt.dtype),
            "residual": blend_delta.to(dtype=prompt.dtype),
        }


class PrecisionRecallEndpointPromptComposer(nn.Module):
    """Choose local precision/recall prompt endpoints in SAM prompt-logit space.

    ``raw_blend_context_cnn`` exposed an important interface pitfall: after the
    standard ``center+bias`` transform, ``raw_prompt - transformed_prompt`` is a
    per-sample constant whenever ``scale=1``.  Blending to raw logits therefore
    acts like an ambiguous local threshold shift instead of a clear prompt
    policy.  This composer makes that policy explicit by constructing two
    endpoint prompts in the same centered coordinate system:

    - ``precision_prompt = raw_centered + precision_bias`` for FP suppression.
    - ``recall_prompt = raw_centered + recall_bias`` for FN expansion.

    Separate bounded gates decide where to move the current prompt toward each
    endpoint:

    ``prompt - bg_gate * relu(prompt - precision_prompt)
      + fg_gate * relu(recall_prompt - prompt)``.

    The default identity initialization uses very small gates, so resumed
    checkpoints start close to the incoming prompt while gradients can still
    teach local FP/FN routing.
    """

    def __init__(
        self,
        hidden_channels: int = 8,
        identity_init: bool = True,
        gate_init: float = 0.02,
        gate_max: float = 0.5,
        precision_bias: float = -0.10,
        recall_bias: float = 0.45,
    ) -> None:
        super().__init__()
        self.gate_max = float(gate_max)
        self.precision_bias = float(precision_bias)
        self.recall_bias = float(recall_bias)
        hidden = int(hidden_channels)
        in_channels = 9

        def gate_net() -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden, 1, kernel_size=3, padding=1),
            )

        self.fg_gate_net = gate_net()
        self.bg_gate_net = gate_net()

        if identity_init:
            actual_gate = min(max(float(gate_init), 1e-6), max(self.gate_max - 1e-6, 1e-6))
            gate_fraction = min(max(actual_gate / max(self.gate_max, 1e-6), 1e-6), 1.0 - 1e-6)
            gate_bias = math.log(gate_fraction / (1.0 - gate_fraction))
            for net in [self.fg_gate_net, self.bg_gate_net]:
                final = net[-1]
                if isinstance(final, nn.Conv2d):
                    nn.init.zeros_(final.weight)
                    nn.init.constant_(final.bias, gate_bias)

    @staticmethod
    def _resize_one_channel(
        tensor: Optional[torch.Tensor],
        *,
        size: Tuple[int, int],
        like: torch.Tensor,
        fallback: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return RawBlendPromptComposer._resize_one_channel(
            tensor,
            size=size,
            like=like,
            fallback=fallback,
        )

    @staticmethod
    def _feature_maps(
        tensor: Optional[torch.Tensor],
        *,
        size: Tuple[int, int],
        like: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return SpatialContextPromptRefiner._feature_maps(tensor, size=size, like=like)

    @staticmethod
    def _center(raw_prompt: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(1, raw_prompt.ndim))
        return raw_prompt - raw_prompt.mean(dim=dims, keepdim=True)

    def forward(
        self,
        prompt: torch.Tensor,
        context: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if prompt.ndim != 4:
            raise ValueError(
                f"PrecisionRecallEndpointPromptComposer expects [B,C,H,W], got shape={tuple(prompt.shape)}"
            )
        context = context or {}
        size = (int(prompt.shape[-2]), int(prompt.shape[-1]))
        raw_prompt = self._resize_one_channel(
            context.get("raw_prompt"),
            size=size,
            like=prompt,
            fallback=prompt,
        )
        dense_prompt = self._resize_one_channel(context.get("dense_prompt_mask"), size=size, like=prompt)
        coarse_mask = self._resize_one_channel(context.get("coarse_mask"), size=size, like=prompt)
        feat_mean, feat_std = self._feature_maps(context.get("forensic_features"), size=size, like=prompt)

        raw_centered = self._center(raw_prompt)
        precision_prompt = raw_centered + self.precision_bias
        recall_prompt = raw_centered + self.recall_bias
        bg_endpoint_delta = torch.relu(prompt - precision_prompt)
        fg_endpoint_delta = torch.relu(recall_prompt - prompt)
        endpoint_gap = torch.clamp(recall_prompt - precision_prompt, min=0.0)

        context_maps = torch.cat(
            [
                prompt,
                raw_centered,
                precision_prompt,
                recall_prompt,
                bg_endpoint_delta,
                fg_endpoint_delta,
                coarse_mask,
                dense_prompt,
                feat_mean + feat_std,
            ],
            dim=1,
        )
        fg_gate_logits = self.fg_gate_net(context_maps)
        bg_gate_logits = self.bg_gate_net(context_maps)
        fg_gate = torch.sigmoid(fg_gate_logits) * self.gate_max
        bg_gate = torch.sigmoid(bg_gate_logits) * self.gate_max
        fg_delta = fg_gate * fg_endpoint_delta
        bg_delta = bg_gate * bg_endpoint_delta
        residual = fg_delta - bg_delta
        refined = prompt + residual.to(dtype=prompt.dtype)
        return refined, {
            "residual": residual.to(dtype=prompt.dtype),
            "residual_abs_mean": residual.detach().abs().mean(),
            "context_maps": context_maps,
            "raw_prompt_resized": raw_prompt.to(dtype=prompt.dtype),
            "raw_centered": raw_centered.to(dtype=prompt.dtype),
            "precision_prompt": precision_prompt.to(dtype=prompt.dtype),
            "recall_prompt": recall_prompt.to(dtype=prompt.dtype),
            "endpoint_gap": endpoint_gap.to(dtype=prompt.dtype),
            "fg_endpoint_delta": fg_endpoint_delta.to(dtype=prompt.dtype),
            "bg_endpoint_delta": bg_endpoint_delta.to(dtype=prompt.dtype),
            "fg_gate": fg_gate.to(dtype=prompt.dtype),
            "bg_gate": bg_gate.to(dtype=prompt.dtype),
            "fg_gate_logits": fg_gate_logits.to(dtype=prompt.dtype),
            "bg_gate_logits": bg_gate_logits.to(dtype=prompt.dtype),
            "fg_residual": fg_delta.to(dtype=prompt.dtype),
            "bg_residual": bg_delta.to(dtype=prompt.dtype),
            "fg_delta": fg_delta.to(dtype=prompt.dtype),
            "bg_delta": bg_delta.to(dtype=prompt.dtype),
            "gate": torch.maximum(fg_gate, bg_gate).to(dtype=prompt.dtype),
        }


class TransformRouterPromptComposer(nn.Module):
    """Route the SAM3 dense prompt between default and conservative transforms.

    Diagnostics repeatedly showed that the raw LAD prompt can move in useful
    directions, but the fixed ``center + bias`` interface expands it into an
    over-large SAM3 mask prompt.  A single static conservative transform
    (e.g. ``scale=0.75, bias=0.15``) helps some datasets and hurts others, so
    this module exposes the transform choice as a bounded spatial gate:

    ``prompt + gate * (conservative_prompt - prompt)``.

    ``prompt`` is the existing/default transformed prompt, which preserves the
    R170 behavior when the gate is initialized to zero.  The conservative
    endpoint is recomputed from the pre-transform raw prompt so it is a real
    scale/bias alternative, not the degenerate raw-vs-centered constant shift.
    """

    def __init__(
        self,
        hidden_channels: int = 8,
        identity_init: bool = True,
        gate_init: float = 0.0,
        gate_max: float = 1.0,
        conservative_scale: float = 0.75,
        conservative_bias: float = 0.15,
    ) -> None:
        super().__init__()
        self.gate_max = float(gate_max)
        self.conservative_scale = float(conservative_scale)
        self.conservative_bias = float(conservative_bias)
        hidden = int(hidden_channels)
        in_channels = 8
        self.gate_net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=3, padding=1),
        )
        if identity_init:
            actual_gate = min(max(float(gate_init), 0.0), max(self.gate_max, 0.0))
            if actual_gate <= 0.0 or self.gate_max <= 0.0:
                gate_bias = -20.0
            else:
                gate_fraction = min(max(actual_gate / max(self.gate_max, 1e-6), 1e-6), 1.0 - 1e-6)
                gate_bias = math.log(gate_fraction / (1.0 - gate_fraction))
            final = self.gate_net[-1]
            if isinstance(final, nn.Conv2d):
                nn.init.zeros_(final.weight)
                nn.init.constant_(final.bias, gate_bias)

    @staticmethod
    def _resize_one_channel(
        tensor: Optional[torch.Tensor],
        *,
        size: Tuple[int, int],
        like: torch.Tensor,
        fallback: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return RawBlendPromptComposer._resize_one_channel(
            tensor,
            size=size,
            like=like,
            fallback=fallback,
        )

    @staticmethod
    def _feature_maps(
        tensor: Optional[torch.Tensor],
        *,
        size: Tuple[int, int],
        like: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return SpatialContextPromptRefiner._feature_maps(tensor, size=size, like=like)

    @staticmethod
    def _center(raw_prompt: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(1, raw_prompt.ndim))
        return raw_prompt - raw_prompt.mean(dim=dims, keepdim=True)

    def forward(
        self,
        prompt: torch.Tensor,
        context: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if prompt.ndim != 4:
            raise ValueError(f"TransformRouterPromptComposer expects [B,C,H,W], got shape={tuple(prompt.shape)}")
        context = context or {}
        size = (int(prompt.shape[-2]), int(prompt.shape[-1]))
        raw_prompt = self._resize_one_channel(
            context.get("raw_prompt"),
            size=size,
            like=prompt,
            fallback=prompt,
        )
        dense_prompt = self._resize_one_channel(context.get("dense_prompt_mask"), size=size, like=prompt)
        coarse_mask = self._resize_one_channel(context.get("coarse_mask"), size=size, like=prompt)
        feat_mean, feat_std = self._feature_maps(context.get("forensic_features"), size=size, like=prompt)

        raw_centered = self._center(raw_prompt)
        default_prompt = prompt
        conservative_prompt = raw_centered * self.conservative_scale + self.conservative_bias
        transform_delta = conservative_prompt - default_prompt
        context_maps = torch.cat(
            [
                default_prompt,
                raw_centered,
                conservative_prompt,
                transform_delta,
                dense_prompt,
                coarse_mask,
                feat_mean,
                feat_std,
            ],
            dim=1,
        )
        gate_logits = self.gate_net(context_maps)
        gate = torch.sigmoid(gate_logits) * self.gate_max
        delta = gate * transform_delta
        refined = default_prompt + delta.to(dtype=prompt.dtype)
        return refined, {
            "transform_router_gate": gate.to(dtype=prompt.dtype),
            "transform_router_gate_logits": gate_logits.to(dtype=prompt.dtype),
            "transform_router_default_prompt": default_prompt.to(dtype=prompt.dtype),
            "transform_router_conservative_prompt": conservative_prompt.to(dtype=prompt.dtype),
            "transform_router_delta": delta.to(dtype=prompt.dtype),
            "transform_router_raw_centered": raw_centered.to(dtype=prompt.dtype),
            "transform_router_context_maps": context_maps,
            "gate": gate.to(dtype=prompt.dtype),
            "residual": delta.to(dtype=prompt.dtype),
            "residual_abs_mean": delta.detach().abs().mean(),
        }


class LearnedPrecisionRecallEndpointPromptComposer(nn.Module):
    """Feature-conditioned precision/recall endpoints for SAM3 mask prompts.

    The fixed transform router can only interpolate between two hand-written
    scale/bias transforms.  This composer keeps that safe starting point but
    makes the endpoints themselves feature-conditioned:

    - precision endpoint: suppress false-positive prompt regions.
    - recall endpoint: expand false-negative prompt regions.

    The endpoint residuals and gates are identity/near-identity initialized so
    resumed R170-style checkpoints start from the existing prompt behavior, but
    training can learn richer local endpoint maps than a fixed conservative
    transform.
    """

    def __init__(
        self,
        hidden_channels: int = 8,
        identity_init: bool = True,
        gate_init: float = 0.02,
        gate_max: float = 0.5,
        endpoint_residual_max: float = 0.25,
        precision_scale: float = 0.75,
        precision_bias: float = 0.15,
        recall_scale: float = 1.0,
        recall_bias: float = 0.35,
    ) -> None:
        super().__init__()
        self.gate_max = float(gate_max)
        self.endpoint_residual_max = float(endpoint_residual_max)
        self.precision_scale = float(precision_scale)
        self.precision_bias = float(precision_bias)
        self.recall_scale = float(recall_scale)
        self.recall_bias = float(recall_bias)
        hidden = int(hidden_channels)
        in_channels = 10

        def map_net() -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden, 1, kernel_size=3, padding=1),
            )

        self.precision_endpoint_net = map_net()
        self.recall_endpoint_net = map_net()
        self.fg_gate_net = map_net()
        self.bg_gate_net = map_net()

        if identity_init:
            actual_gate = min(max(float(gate_init), 0.0), max(self.gate_max, 0.0))
            if actual_gate <= 0.0 or self.gate_max <= 0.0:
                gate_bias = -20.0
            else:
                gate_fraction = min(max(actual_gate / max(self.gate_max, 1e-6), 1e-6), 1.0 - 1e-6)
                gate_bias = math.log(gate_fraction / (1.0 - gate_fraction))
            for net in [self.precision_endpoint_net, self.recall_endpoint_net]:
                final = net[-1]
                if isinstance(final, nn.Conv2d):
                    nn.init.zeros_(final.weight)
                    nn.init.zeros_(final.bias)
            for net in [self.fg_gate_net, self.bg_gate_net]:
                final = net[-1]
                if isinstance(final, nn.Conv2d):
                    nn.init.zeros_(final.weight)
                    nn.init.constant_(final.bias, gate_bias)

    @staticmethod
    def _resize_one_channel(
        tensor: Optional[torch.Tensor],
        *,
        size: Tuple[int, int],
        like: torch.Tensor,
        fallback: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return RawBlendPromptComposer._resize_one_channel(
            tensor,
            size=size,
            like=like,
            fallback=fallback,
        )

    @staticmethod
    def _feature_maps(
        tensor: Optional[torch.Tensor],
        *,
        size: Tuple[int, int],
        like: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return SpatialContextPromptRefiner._feature_maps(tensor, size=size, like=like)

    @staticmethod
    def _center(raw_prompt: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(1, raw_prompt.ndim))
        return raw_prompt - raw_prompt.mean(dim=dims, keepdim=True)

    def forward(
        self,
        prompt: torch.Tensor,
        context: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if prompt.ndim != 4:
            raise ValueError(
                f"LearnedPrecisionRecallEndpointPromptComposer expects [B,C,H,W], got shape={tuple(prompt.shape)}"
            )
        context = context or {}
        size = (int(prompt.shape[-2]), int(prompt.shape[-1]))
        raw_prompt = self._resize_one_channel(
            context.get("raw_prompt"),
            size=size,
            like=prompt,
            fallback=prompt,
        )
        dense_prompt = self._resize_one_channel(context.get("dense_prompt_mask"), size=size, like=prompt)
        coarse_mask = self._resize_one_channel(context.get("coarse_mask"), size=size, like=prompt)
        feat_mean, feat_std = self._feature_maps(context.get("forensic_features"), size=size, like=prompt)

        raw_centered = self._center(raw_prompt)
        base_precision = raw_centered * self.precision_scale + self.precision_bias
        base_recall = raw_centered * self.recall_scale + self.recall_bias
        precision_gap = torch.relu(prompt - base_precision)
        recall_gap = torch.relu(base_recall - prompt)
        context_maps = torch.cat(
            [
                prompt,
                raw_centered,
                base_precision,
                base_recall,
                precision_gap,
                recall_gap,
                dense_prompt,
                coarse_mask,
                feat_mean,
                feat_std,
            ],
            dim=1,
        )

        precision_endpoint_delta = torch.tanh(self.precision_endpoint_net(context_maps)) * self.endpoint_residual_max
        recall_endpoint_delta = torch.tanh(self.recall_endpoint_net(context_maps)) * self.endpoint_residual_max
        precision_endpoint = base_precision + precision_endpoint_delta
        recall_endpoint = base_recall + recall_endpoint_delta

        bg_residual = torch.relu(prompt - precision_endpoint)
        fg_residual = torch.relu(recall_endpoint - prompt)
        fg_gate_logits = self.fg_gate_net(context_maps)
        bg_gate_logits = self.bg_gate_net(context_maps)
        fg_gate = torch.sigmoid(fg_gate_logits) * self.gate_max
        bg_gate = torch.sigmoid(bg_gate_logits) * self.gate_max
        fg_delta = fg_gate * fg_residual
        bg_delta = bg_gate * bg_residual
        residual = fg_delta - bg_delta
        refined = prompt + residual.to(dtype=prompt.dtype)
        return refined, {
            "residual": residual.to(dtype=prompt.dtype),
            "residual_abs_mean": residual.detach().abs().mean(),
            "context_maps": context_maps,
            "raw_prompt_resized": raw_prompt.to(dtype=prompt.dtype),
            "raw_centered": raw_centered.to(dtype=prompt.dtype),
            "base_precision_endpoint": base_precision.to(dtype=prompt.dtype),
            "base_recall_endpoint": base_recall.to(dtype=prompt.dtype),
            "precision_endpoint": precision_endpoint.to(dtype=prompt.dtype),
            "recall_endpoint": recall_endpoint.to(dtype=prompt.dtype),
            "precision_endpoint_delta": precision_endpoint_delta.to(dtype=prompt.dtype),
            "recall_endpoint_delta": recall_endpoint_delta.to(dtype=prompt.dtype),
            "fg_gate": fg_gate.to(dtype=prompt.dtype),
            "bg_gate": bg_gate.to(dtype=prompt.dtype),
            "fg_gate_logits": fg_gate_logits.to(dtype=prompt.dtype),
            "bg_gate_logits": bg_gate_logits.to(dtype=prompt.dtype),
            "fg_residual": fg_residual.to(dtype=prompt.dtype),
            "bg_residual": bg_residual.to(dtype=prompt.dtype),
            "fg_delta": fg_delta.to(dtype=prompt.dtype),
            "bg_delta": bg_delta.to(dtype=prompt.dtype),
            "gate": torch.maximum(fg_gate, bg_gate).to(dtype=prompt.dtype),
        }


class TeacherOracleEndpointPromptComposer(nn.Module):
    """Richer local precision/recall endpoint router for teacher/oracle losses.

    This module is intentionally close to
    :class:`LearnedPrecisionRecallEndpointPromptComposer`, but exposes the
    diagnostics expected by the dual-branch teacher/oracle losses and gives the
    route networks a slightly richer local state.  The training oracle is
    supplied outside this module; inference only uses prompt/image features.
    """

    def __init__(
        self,
        hidden_channels: int = 8,
        identity_init: bool = True,
        gate_init: float = 0.02,
        gate_max: float = 0.5,
        endpoint_residual_max: float = 0.25,
        precision_scale: float = 0.75,
        precision_bias: float = 0.15,
        recall_scale: float = 1.0,
        recall_bias: float = 0.35,
    ) -> None:
        super().__init__()
        self.gate_max = float(gate_max)
        self.endpoint_residual_max = float(endpoint_residual_max)
        self.precision_scale = float(precision_scale)
        self.precision_bias = float(precision_bias)
        self.recall_scale = float(recall_scale)
        self.recall_bias = float(recall_bias)
        hidden = int(hidden_channels)
        in_channels = 14

        def map_net() -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden, 1, kernel_size=3, padding=1),
            )

        self.precision_endpoint_net = map_net()
        self.recall_endpoint_net = map_net()
        self.fg_gate_net = map_net()
        self.bg_gate_net = map_net()

        if identity_init:
            actual_gate = min(max(float(gate_init), 0.0), max(self.gate_max, 0.0))
            if actual_gate <= 0.0 or self.gate_max <= 0.0:
                gate_bias = -20.0
            else:
                gate_fraction = min(max(actual_gate / max(self.gate_max, 1e-6), 1e-6), 1.0 - 1e-6)
                gate_bias = math.log(gate_fraction / (1.0 - gate_fraction))
            for net in [self.precision_endpoint_net, self.recall_endpoint_net]:
                final = net[-1]
                if isinstance(final, nn.Conv2d):
                    nn.init.zeros_(final.weight)
                    nn.init.zeros_(final.bias)
            for net in [self.fg_gate_net, self.bg_gate_net]:
                final = net[-1]
                if isinstance(final, nn.Conv2d):
                    nn.init.zeros_(final.weight)
                    nn.init.constant_(final.bias, gate_bias)

    @staticmethod
    def _resize_one_channel(
        tensor: Optional[torch.Tensor],
        *,
        size: Tuple[int, int],
        like: torch.Tensor,
        fallback: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return RawBlendPromptComposer._resize_one_channel(
            tensor,
            size=size,
            like=like,
            fallback=fallback,
        )

    @staticmethod
    def _feature_maps(
        tensor: Optional[torch.Tensor],
        *,
        size: Tuple[int, int],
        like: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return SpatialContextPromptRefiner._feature_maps(tensor, size=size, like=like)

    @staticmethod
    def _center(raw_prompt: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(1, raw_prompt.ndim))
        return raw_prompt - raw_prompt.mean(dim=dims, keepdim=True)

    def forward(
        self,
        prompt: torch.Tensor,
        context: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if prompt.ndim != 4:
            raise ValueError(
                f"TeacherOracleEndpointPromptComposer expects [B,C,H,W], got shape={tuple(prompt.shape)}"
            )
        context = context or {}
        size = (int(prompt.shape[-2]), int(prompt.shape[-1]))
        raw_prompt = self._resize_one_channel(
            context.get("raw_prompt"),
            size=size,
            like=prompt,
            fallback=prompt,
        )
        dense_prompt = self._resize_one_channel(context.get("dense_prompt_mask"), size=size, like=prompt)
        coarse_mask = self._resize_one_channel(context.get("coarse_mask"), size=size, like=prompt)
        feat_mean, feat_std = self._feature_maps(context.get("forensic_features"), size=size, like=prompt)

        raw_centered = self._center(raw_prompt)
        base_precision = raw_centered * self.precision_scale + self.precision_bias
        base_recall = raw_centered * self.recall_scale + self.recall_bias
        precision_gap = torch.relu(prompt - base_precision)
        recall_gap = torch.relu(base_recall - prompt)
        prompt_prob = torch.sigmoid(prompt)
        prompt_uncertainty = prompt_prob * (1.0 - prompt_prob)
        endpoint_gap = torch.clamp(base_recall - base_precision, min=0.0)
        context_maps = torch.cat(
            [
                prompt,
                raw_centered,
                base_precision,
                base_recall,
                precision_gap,
                recall_gap,
                dense_prompt,
                coarse_mask,
                feat_mean,
                feat_std,
                prompt_prob,
                prompt_uncertainty,
                raw_centered.abs(),
                endpoint_gap,
            ],
            dim=1,
        )

        precision_endpoint_delta = torch.tanh(self.precision_endpoint_net(context_maps)) * self.endpoint_residual_max
        recall_endpoint_delta = torch.tanh(self.recall_endpoint_net(context_maps)) * self.endpoint_residual_max
        precision_endpoint = base_precision + precision_endpoint_delta
        recall_endpoint = base_recall + recall_endpoint_delta

        bg_residual = torch.relu(prompt - precision_endpoint)
        fg_residual = torch.relu(recall_endpoint - prompt)
        fg_gate_logits = self.fg_gate_net(context_maps)
        bg_gate_logits = self.bg_gate_net(context_maps)
        fg_gate = torch.sigmoid(fg_gate_logits) * self.gate_max
        bg_gate = torch.sigmoid(bg_gate_logits) * self.gate_max
        fg_delta = fg_gate * fg_residual
        bg_delta = bg_gate * bg_residual
        residual = fg_delta - bg_delta
        refined = prompt + residual.to(dtype=prompt.dtype)

        return refined, {
            "residual": residual.to(dtype=prompt.dtype),
            "residual_abs_mean": residual.detach().abs().mean(),
            "teacher_oracle_context_maps": context_maps,
            "context_maps": context_maps,
            "raw_prompt_resized": raw_prompt.to(dtype=prompt.dtype),
            "raw_centered": raw_centered.to(dtype=prompt.dtype),
            "base_precision_endpoint": base_precision.to(dtype=prompt.dtype),
            "base_recall_endpoint": base_recall.to(dtype=prompt.dtype),
            "precision_endpoint": precision_endpoint.to(dtype=prompt.dtype),
            "recall_endpoint": recall_endpoint.to(dtype=prompt.dtype),
            "precision_endpoint_delta": precision_endpoint_delta.to(dtype=prompt.dtype),
            "recall_endpoint_delta": recall_endpoint_delta.to(dtype=prompt.dtype),
            "fg_gate": fg_gate.to(dtype=prompt.dtype),
            "bg_gate": bg_gate.to(dtype=prompt.dtype),
            "post_prompt_fg_gate": fg_gate.to(dtype=prompt.dtype),
            "post_prompt_bg_gate": bg_gate.to(dtype=prompt.dtype),
            "fg_gate_logits": fg_gate_logits.to(dtype=prompt.dtype),
            "bg_gate_logits": bg_gate_logits.to(dtype=prompt.dtype),
            "fg_residual": fg_residual.to(dtype=prompt.dtype),
            "bg_residual": bg_residual.to(dtype=prompt.dtype),
            "post_prompt_fg_residual": fg_residual.to(dtype=prompt.dtype),
            "post_prompt_bg_residual": bg_residual.to(dtype=prompt.dtype),
            "fg_delta": fg_delta.to(dtype=prompt.dtype),
            "bg_delta": bg_delta.to(dtype=prompt.dtype),
            "gate": torch.maximum(fg_gate, bg_gate).to(dtype=prompt.dtype),
        }


class PostTransformDualBranchPromptRefiner(nn.Module):
    """Post-transform foreground/background prompt composer.

    The dense prompt sent to SAM3 is produced after centering/scaling/biasing
    the raw LAD prompt.  Earlier dual-branch prompt heads act before that
    transform, so a positive post-transform bias can still leave the actual SAM
    prompt over-expanded.  This module operates directly on the transformed
    prompt and applies two bounded non-negative residual branches:

    ``prompt + fg_gate * fg_residual - bg_gate * bg_residual``.

    It is identity-initialized for safe checkpoint continuation, while
    straight-through non-negative residual magnitudes keep the zero-initialized
    branches trainable.
    """

    def __init__(
        self,
        hidden_channels: int = 8,
        identity_init: bool = True,
        max_residual: float = 0.5,
        gate_init: float = 0.2,
        gate_max: float = 1.0,
        feature_channels: Optional[int] = None,
        feature_proj_channels: int = 0,
        extra_context_channels: int = 0,
    ) -> None:
        super().__init__()
        self.max_residual = float(max_residual)
        self.gate_max = float(gate_max)
        gate_init_clamped = min(max(float(gate_init), 1e-4), 1.0 - 1e-4)
        hidden = int(hidden_channels)
        self.feature_proj_channels = int(feature_proj_channels)
        if self.feature_proj_channels > 0:
            if feature_channels is None:
                raise ValueError("feature_channels is required when feature_proj_channels > 0")
            self.feature_proj = nn.Sequential(
                nn.Conv2d(int(feature_channels), self.feature_proj_channels, kernel_size=1, bias=False),
                nn.InstanceNorm2d(self.feature_proj_channels, affine=True),
                nn.GELU(),
            )
        else:
            self.feature_proj = None
        self.extra_context_channels = int(extra_context_channels)
        in_channels = 6 + max(self.feature_proj_channels, 0) + self.extra_context_channels

        def residual_net() -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden, 1, kernel_size=3, padding=1),
            )

        def gate_net() -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden, 1, kernel_size=1),
            )

        self.fg_residual_net = residual_net()
        self.bg_residual_net = residual_net()
        self.fg_gate_net = gate_net()
        self.bg_gate_net = gate_net()

        if identity_init:
            for net in [self.fg_residual_net, self.bg_residual_net]:
                final = net[-1]
                if isinstance(final, nn.Conv2d):
                    nn.init.zeros_(final.weight)
                    nn.init.zeros_(final.bias)
            gate_bias = math.log(gate_init_clamped / (1.0 - gate_init_clamped))
            for net in [self.fg_gate_net, self.bg_gate_net]:
                final = net[-1]
                if isinstance(final, nn.Conv2d):
                    nn.init.zeros_(final.weight)
                    nn.init.constant_(final.bias, gate_bias)

    @staticmethod
    def _resize_one_channel(
        tensor: Optional[torch.Tensor],
        *,
        size: Tuple[int, int],
        like: torch.Tensor,
    ) -> torch.Tensor:
        return ContextualPromptRefiner._resize_one_channel(tensor, size=size, like=like)

    @staticmethod
    def _feature_maps(
        tensor: Optional[torch.Tensor],
        *,
        size: Tuple[int, int],
        like: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return ContextualPromptRefiner._feature_maps(tensor, size=size, like=like)

    @staticmethod
    def _nonnegative_magnitude(raw: torch.Tensor) -> torch.Tensor:
        positive = torch.relu(raw)
        return raw + (positive - raw).detach()

    def _feature_context(
        self,
        tensor: Optional[torch.Tensor],
        *,
        size: Tuple[int, int],
        like: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if self.feature_proj is None:
            return None
        if tensor is None:
            return like.new_zeros(like.shape[0], self.feature_proj_channels, *size)
        tensor_f = tensor.float()
        if tensor_f.shape[0] != like.shape[0]:
            raise ValueError(
                "PostTransformDualBranchPromptRefiner feature batch mismatch: "
                f"expected {like.shape[0]}, got {tensor_f.shape[0]}"
            )
        projected = self.feature_proj(tensor_f.to(device=like.device))
        if projected.shape[-2:] != size:
            projected = F.interpolate(projected, size=size, mode="bilinear", align_corners=False)
        return projected.to(device=like.device, dtype=like.dtype)

    def _extra_context_parts(
        self,
        context: Dict[str, torch.Tensor],
        *,
        size: Tuple[int, int],
        like: torch.Tensor,
    ) -> List[torch.Tensor]:
        return []

    def forward(
        self,
        prompt: torch.Tensor,
        context: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if prompt.ndim != 4:
            raise ValueError(
                f"PostTransformDualBranchPromptRefiner expects [B,C,H,W], got shape={tuple(prompt.shape)}"
            )
        context = context or {}
        size = (int(prompt.shape[-2]), int(prompt.shape[-1]))
        raw_prompt = self._resize_one_channel(context.get("raw_prompt"), size=size, like=prompt)
        dense_prompt = self._resize_one_channel(context.get("dense_prompt_mask"), size=size, like=prompt)
        coarse_mask = self._resize_one_channel(context.get("coarse_mask"), size=size, like=prompt)
        feat_mean, feat_std = self._feature_maps(context.get("forensic_features"), size=size, like=prompt)
        context_parts = [prompt, raw_prompt, dense_prompt, coarse_mask, feat_mean, feat_std]
        feature_context = self._feature_context(
            context.get("forensic_features"),
            size=size,
            like=prompt,
        )
        if feature_context is not None:
            context_parts.append(feature_context)
        context_parts.extend(self._extra_context_parts(context, size=size, like=prompt))
        context_maps = torch.cat(context_parts, dim=1)

        fg_residual_raw = self.fg_residual_net(context_maps)
        bg_residual_raw = self.bg_residual_net(context_maps)
        fg_residual = self._nonnegative_magnitude(torch.tanh(fg_residual_raw)) * self.max_residual
        bg_residual = self._nonnegative_magnitude(torch.tanh(bg_residual_raw)) * self.max_residual
        fg_gate_logits = self.fg_gate_net(context_maps)
        bg_gate_logits = self.bg_gate_net(context_maps)
        fg_gate = torch.sigmoid(fg_gate_logits) * self.gate_max
        bg_gate = torch.sigmoid(bg_gate_logits) * self.gate_max
        fg_delta = fg_gate * fg_residual
        bg_delta = bg_gate * bg_residual
        residual = fg_delta - bg_delta
        diagnostics = {
            "residual": residual.to(dtype=prompt.dtype),
            "residual_abs_mean": residual.detach().abs().mean(),
            "context_maps": context_maps,
            "fg_gate": fg_gate.to(dtype=prompt.dtype),
            "bg_gate": bg_gate.to(dtype=prompt.dtype),
            "fg_gate_logits": fg_gate_logits.to(dtype=prompt.dtype),
            "bg_gate_logits": bg_gate_logits.to(dtype=prompt.dtype),
            "fg_residual": fg_residual.to(dtype=prompt.dtype),
            "bg_residual": bg_residual.to(dtype=prompt.dtype),
            "fg_delta": fg_delta.to(dtype=prompt.dtype),
            "bg_delta": bg_delta.to(dtype=prompt.dtype),
            "fg_residual_raw": fg_residual_raw.to(dtype=prompt.dtype),
            "bg_residual_raw": bg_residual_raw.to(dtype=prompt.dtype),
        }
        if feature_context is not None:
            diagnostics["feature_context"] = feature_context
        return prompt + residual.to(dtype=prompt.dtype), diagnostics


class FeatureGuidedPostTransformDualBranchPromptRefiner(PostTransformDualBranchPromptRefiner):
    """Post-transform dual-branch composer with projected forensic features.

    The previous post-transform composer only used scalar feature summaries
    (mean/std), and in experiments it tended to learn global suppression.  This
    variant injects a small projected multichannel forensic feature map into the
    FG/BG residual and gate CNNs so the post-SAM prompt correction can depend on
    local forensic structure while preserving the same identity initialization.
    """

    def __init__(
        self,
        hidden_channels: int = 8,
        identity_init: bool = True,
        max_residual: float = 0.5,
        gate_init: float = 0.2,
        gate_max: float = 1.0,
        feature_channels: int = 384,
        feature_proj_channels: Optional[int] = None,
    ) -> None:
        hidden = int(hidden_channels)
        proj_channels = int(feature_proj_channels or max(hidden // 2, 4))
        super().__init__(
            hidden_channels=hidden,
            identity_init=identity_init,
            max_residual=max_residual,
            gate_init=gate_init,
            gate_max=gate_max,
            feature_channels=int(feature_channels),
            feature_proj_channels=proj_channels,
        )


class SemanticGuidedPostTransformDualBranchPromptRefiner(PostTransformDualBranchPromptRefiner):
    """Post-transform dual-branch composer with SAM feature context.

    The prompt sent to SAM3 should be compatible with the decoder features it
    conditions.  Earlier post-transform composers only saw LAD prompt maps and
    forensic summaries, which made them prone to dataset-global expansion or
    suppression.  This variant appends six semantic/adapted SAM context maps:
    mean and standard-deviation maps from image embeddings and the two
    high-resolution decoder feature scales.  The added context is
    channel-count agnostic, so it works for both real SAM3 features and test
    doubles while preserving identity initialization.
    """

    def __init__(
        self,
        hidden_channels: int = 8,
        identity_init: bool = True,
        max_residual: float = 0.5,
        gate_init: float = 0.2,
        gate_max: float = 1.0,
    ) -> None:
        super().__init__(
            hidden_channels=hidden_channels,
            identity_init=identity_init,
            max_residual=max_residual,
            gate_init=gate_init,
            gate_max=gate_max,
            extra_context_channels=6,
        )

    @staticmethod
    def _semantic_feature_pair(
        tensor: Optional[torch.Tensor],
        *,
        size: Tuple[int, int],
        like: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return ContextualPromptRefiner._feature_maps(tensor, size=size, like=like)

    def _extra_context_parts(
        self,
        context: Dict[str, torch.Tensor],
        *,
        size: Tuple[int, int],
        like: torch.Tensor,
    ) -> List[torch.Tensor]:
        image_mean, image_std = self._semantic_feature_pair(
            context.get("image_embeddings"),
            size=size,
            like=like,
        )
        high_res = context.get("high_res_features")
        high0 = high_res[0] if isinstance(high_res, (list, tuple)) and len(high_res) > 0 else None
        high1 = high_res[1] if isinstance(high_res, (list, tuple)) and len(high_res) > 1 else None
        high0_mean, high0_std = self._semantic_feature_pair(high0, size=size, like=like)
        high1_mean, high1_std = self._semantic_feature_pair(high1, size=size, like=like)
        return [image_mean, image_std, high0_mean, high0_std, high1_mean, high1_std]


class ForgeryLocalizer(nn.Module):
    """Detect forgery regions using LAD/Ferret features plus a SAM backend."""

    def __init__(
        self,
        sam_config: str,
        sam_checkpoint: str,
        prompt_dim: int = 256,
        output_resolution: Tuple[int, int] = (256, 256),
        global_image_size: int = 512,
        downscale: int = 4,
        train_sam_iou : bool = False,
        dropout_rate: float = 0.1,
        use_detection_probe: bool = True,
        ferret_dim: int = 96,
        lad_tau: float = 0.004,
        lad_multi_taus: Optional[Sequence[float]] = None,
        forensic_operator: str = "lad",
        sam_backend: str = "sam2",
        adapter_residual_scale: float = 1.0,
        adapter_type: str = "shared",
        adapter_active_scales: Optional[Sequence[int]] = None,
        adapter_gamma_init: float = 0.0,
        adapter_sample_gate: bool = False,
        adapter_sample_gate_scales: Optional[Sequence[int]] = None,
        adapter_sample_gate_max_delta: float = 0.5,
        adapter_forensic_source: str = "final",
        adapter_diagnostics: bool = False,
        sam3_prompt_mode: str = "legacy",
        coarse_prompt_transform: str = "none",
        coarse_prompt_scale: float = 1.0,
        coarse_prompt_bias: float = 0.0,
        coarse_prompt_eps: float = 1e-6,
        coarse_prompt_calibrator: str = "none",
        coarse_prompt_calibrator_hidden: int = 16,
        coarse_prompt_calibrator_max_delta_scale: float = 1.0,
        coarse_prompt_calibrator_max_delta_bias: float = 2.0,
        final_logit_calibrator: str = "none",
        final_logit_calibrator_hidden: int = 16,
        final_logit_calibrator_max_delta_scale: float = 0.0,
        final_logit_calibrator_max_delta_bias: float = 1.0,
        coarse_prompt_refiner: str = "none",
        coarse_prompt_refiner_hidden: int = 8,
        coarse_prompt_refiner_max_residual: float = 1.0,
        coarse_prompt_refiner_gate_init: float = 0.2,
        coarse_prompt_refiner_gate_max: float = 1.0,
        coarse_prompt_refiner_precision_bias: float = -0.10,
        coarse_prompt_refiner_recall_bias: float = 0.45,
        coarse_prompt_head: str = "mask_compressor",
        coarse_prompt_hidden: Optional[int] = None,
        coarse_prompt_dropout: float = 0.0,
        coarse_prompt_gate_init: float = 0.02,
        coarse_prompt_gate_max: float = 1.0,
        coarse_prompt_area_bias: bool = False,
        coarse_prompt_signed_residual_max_delta: float = 0.5,
        coarse_prompt_unet_gate_init: Optional[float] = None,
        coarse_prompt_unet_gate_max: Optional[float] = None,
        coarse_prompt_unet_signed_residual_max_delta: Optional[float] = None,
        mask_compressor_kernel_size: int = 3,
        mask_compressor_output: str = "logits",
        legacy_logit_head: bool = False,
    ) -> None:
        super().__init__()

        self.output_resolution = output_resolution
        self.global_image_size = int(global_image_size)
        self.use_detection_probe = use_detection_probe
        self.sam_backend = str(sam_backend)
        self.adapter_residual_scale = float(adapter_residual_scale)
        self.adapter_type = str(adapter_type)
        self.adapter_active_scales = None if adapter_active_scales is None else [int(s) for s in adapter_active_scales]
        self.adapter_gamma_init = float(adapter_gamma_init)
        self.adapter_sample_gate = bool(adapter_sample_gate)
        self.adapter_sample_gate_scales = (
            None if adapter_sample_gate_scales is None else [int(s) for s in adapter_sample_gate_scales]
        )
        self.adapter_sample_gate_max_delta = float(adapter_sample_gate_max_delta)
        self.adapter_forensic_source = str(adapter_forensic_source)
        self.adapter_diagnostics_enabled = bool(adapter_diagnostics)
        self.sam3_prompt_mode = str(sam3_prompt_mode)
        self.coarse_prompt_transform = str(coarse_prompt_transform)
        self.coarse_prompt_scale = float(coarse_prompt_scale)
        self.coarse_prompt_bias = float(coarse_prompt_bias)
        self.coarse_prompt_eps = float(coarse_prompt_eps)
        self.coarse_prompt_calibrator = str(coarse_prompt_calibrator)
        self.final_logit_calibrator_name = str(final_logit_calibrator)
        self.coarse_prompt_refiner = str(coarse_prompt_refiner)
        self.coarse_prompt_refiner_gate_init = float(coarse_prompt_refiner_gate_init)
        self.coarse_prompt_refiner_gate_max = float(coarse_prompt_refiner_gate_max)
        self.coarse_prompt_refiner_precision_bias = float(coarse_prompt_refiner_precision_bias)
        self.coarse_prompt_refiner_recall_bias = float(coarse_prompt_refiner_recall_bias)
        self.coarse_prompt_head = str(coarse_prompt_head)
        self.coarse_prompt_hidden = None if coarse_prompt_hidden is None else int(coarse_prompt_hidden)
        self.coarse_prompt_dropout = float(coarse_prompt_dropout)
        self.coarse_prompt_gate_init = float(coarse_prompt_gate_init)
        self.coarse_prompt_gate_max = float(coarse_prompt_gate_max)
        self.coarse_prompt_area_bias = bool(coarse_prompt_area_bias)
        self.coarse_prompt_signed_residual_max_delta = float(coarse_prompt_signed_residual_max_delta)
        self.coarse_prompt_unet_gate_init = (
            None if coarse_prompt_unet_gate_init is None else float(coarse_prompt_unet_gate_init)
        )
        self.coarse_prompt_unet_gate_max = (
            None if coarse_prompt_unet_gate_max is None else float(coarse_prompt_unet_gate_max)
        )
        self.coarse_prompt_unet_signed_residual_max_delta = (
            None
            if coarse_prompt_unet_signed_residual_max_delta is None
            else float(coarse_prompt_unet_signed_residual_max_delta)
        )
        self.mask_compressor_kernel_size = int(mask_compressor_kernel_size)
        self.mask_compressor_output = str(mask_compressor_output)
        self.legacy_logit_head = bool(legacy_logit_head)
        self.forensic_operator = str(forensic_operator)
        self.lad_multi_taus = None if lad_multi_taus is None else [float(tau) for tau in lad_multi_taus]
        self.sam3_model = None
        self.sam3_tracker = None
        self.sam3_image_transform = None

        if self.sam_backend == "sam2":
            self._init_sam2_backend(
                sam_config=sam_config,
                sam_checkpoint=sam_checkpoint,
                train_sam_iou=train_sam_iou,
            )
        elif self.sam_backend == "sam3_interactive":
            self._init_sam3_interactive_backend(
                sam_checkpoint=sam_checkpoint,
                train_sam_iou=train_sam_iou,
            )
        else:
            raise ValueError(f"Unsupported sam_backend={self.sam_backend!r}")

        # ------------------------------
        # Forensic Stream (Trainable)
        # ------------------------------        
        # FerretNet backbone (first 4 Ferret Blocks)
        self.ferret_backbone = FerretBackbone(
            dim=ferret_dim,
            lad_tau=lad_tau,
            lad_multi_taus=self.lad_multi_taus,
            forensic_operator=self.forensic_operator,
            coarse_prompt_head=self.coarse_prompt_head,
            coarse_prompt_hidden=self.coarse_prompt_hidden,
            coarse_prompt_dropout=self.coarse_prompt_dropout,
            coarse_prompt_gate_init=self.coarse_prompt_gate_init,
            coarse_prompt_gate_max=self.coarse_prompt_gate_max,
            coarse_prompt_area_bias=self.coarse_prompt_area_bias,
            coarse_prompt_signed_residual_max_delta=self.coarse_prompt_signed_residual_max_delta,
            coarse_prompt_unet_gate_init=self.coarse_prompt_unet_gate_init,
            coarse_prompt_unet_gate_max=self.coarse_prompt_unet_gate_max,
            coarse_prompt_unet_signed_residual_max_delta=self.coarse_prompt_unet_signed_residual_max_delta,
            mask_compressor_kernel_size=self.mask_compressor_kernel_size,
            mask_compressor_output=self.mask_compressor_output,
            legacy_logit_head=self.legacy_logit_head,
        )
        # Calculate output channels based on actual architecture
        # Block 1: dim → dim*2
        # Block 2: dim*2 → dim*2
        # Block 3: dim*2 → dim*4
        # Block 4: dim*4 → dim*4
        ferret_output_channels = ferret_dim * 4  # Output channels after 4 Ferret Blocks
        
        # ------------------------------
        # Cross-Modal Feature Adapters
        # ------------------------------        
        # Create cross-modal adapters for feature fusion
        adapter_kwargs = dict(
            hidden_dim=prompt_dim,
            in_channels_list=[256, 32, 64],
            forensic_channels=ferret_output_channels,
            dropout_rate=dropout_rate,
            active_scales=self.adapter_active_scales,
        )
        if self.adapter_type == "shared":
            self.adapters = SharedAdapter(
                **adapter_kwargs,
                residual_scale=self.adapter_residual_scale,
            )
        elif self.adapter_type == "norm_gated":
            self.adapters = NormGatedAdapter(
                **adapter_kwargs,
                gamma_init=self.adapter_gamma_init,
                residual_scale=self.adapter_residual_scale,
                sample_gate=self.adapter_sample_gate,
                sample_gate_scales=self.adapter_sample_gate_scales,
                sample_gate_max_delta=self.adapter_sample_gate_max_delta,
            )
        else:
            raise ValueError(f"Unsupported adapter_type={self.adapter_type!r}")

        if self.adapter_forensic_source == "final":
            self.pyramid_adapters = None
        elif self.adapter_forensic_source == "final_plus_pyramid":
            if self.adapter_type != "norm_gated":
                raise ValueError(
                    "adapter_forensic_source='final_plus_pyramid' currently requires "
                    "adapter_type='norm_gated'"
                )
            self.pyramid_adapters = NormGatedAdapter(
                hidden_dim=prompt_dim,
                in_channels_list=[256, 32, 64],
                forensic_channels=[ferret_output_channels, ferret_dim // 2, ferret_dim],
                dropout_rate=dropout_rate,
                gamma_init=0.0,
                residual_scale=self.adapter_residual_scale,
                active_scales=self.adapter_active_scales,
                sample_gate=self.adapter_sample_gate,
                sample_gate_scales=self.adapter_sample_gate_scales,
                sample_gate_max_delta=self.adapter_sample_gate_max_delta,
            )
        else:
            raise ValueError(f"Unsupported adapter_forensic_source={self.adapter_forensic_source!r}")

        if self.coarse_prompt_calibrator == "none":
            self.prompt_calibrator = None
        elif self.coarse_prompt_calibrator == "stats_mlp":
            self.prompt_calibrator = PromptCalibrator(
                hidden_dim=int(coarse_prompt_calibrator_hidden),
                identity_init=True,
                max_delta_scale=float(coarse_prompt_calibrator_max_delta_scale),
                max_delta_bias=float(coarse_prompt_calibrator_max_delta_bias),
            )
        elif self.coarse_prompt_calibrator == "context_stats_mlp":
            self.prompt_calibrator = ContextualPromptCalibrator(
                hidden_dim=int(coarse_prompt_calibrator_hidden),
                identity_init=True,
                max_delta_scale=float(coarse_prompt_calibrator_max_delta_scale),
                max_delta_bias=float(coarse_prompt_calibrator_max_delta_bias),
            )
        else:
            raise ValueError(f"Unsupported coarse_prompt_calibrator={self.coarse_prompt_calibrator!r}")
        if self.final_logit_calibrator_name == "none":
            self.final_logit_calibrator = None
        elif self.final_logit_calibrator_name in {"stats_mlp", "context_stats_mlp"}:
            self.final_logit_calibrator = FinalLogitCalibrator(
                hidden_dim=int(final_logit_calibrator_hidden),
                identity_init=True,
                max_delta_scale=float(final_logit_calibrator_max_delta_scale),
                max_delta_bias=float(final_logit_calibrator_max_delta_bias),
            )
        elif self.final_logit_calibrator_name == "quantile_mlp":
            self.final_logit_calibrator = QuantileFinalLogitCalibrator(
                hidden_dim=int(final_logit_calibrator_hidden),
                identity_init=True,
                max_delta_scale=float(final_logit_calibrator_max_delta_scale),
                max_delta_bias=float(final_logit_calibrator_max_delta_bias),
            )
        elif self.final_logit_calibrator_name == "semantic_spatial_cnn":
            self.final_logit_calibrator = SemanticSpatialFinalLogitCalibrator(
                hidden_channels=int(final_logit_calibrator_hidden),
                identity_init=True,
                max_residual=float(final_logit_calibrator_max_delta_bias),
                gate_init=0.1,
                gate_max=1.0,
            )
        else:
            raise ValueError(f"Unsupported final_logit_calibrator={self.final_logit_calibrator_name!r}")
        if self.sam3_prompt_mode == "dual_spatial_gated":
            self.dual_prompt_fusion_gate = SpatialDualPromptFusionGate(
                hidden_channels=8,
                init_prob=0.05,
                identity_init=True,
            )
        else:
            self.dual_prompt_fusion_gate = DualPromptFusionGate(hidden_dim=16, identity_init=True)
        if self.coarse_prompt_refiner == "none":
            self.prompt_refiner = None
        elif self.coarse_prompt_refiner == "residual_cnn":
            self.prompt_refiner = PromptRefiner(
                hidden_channels=int(coarse_prompt_refiner_hidden),
                identity_init=True,
            )
        elif self.coarse_prompt_refiner == "context_residual_cnn":
            self.prompt_refiner = ContextualPromptRefiner(
                hidden_channels=int(coarse_prompt_refiner_hidden),
                identity_init=True,
                max_residual=float(coarse_prompt_refiner_max_residual),
            )
        elif self.coarse_prompt_refiner == "gated_context_residual_cnn":
            self.prompt_refiner = GatedContextualPromptRefiner(
                hidden_channels=int(coarse_prompt_refiner_hidden),
                identity_init=True,
                max_residual=float(coarse_prompt_refiner_max_residual),
                gate_init=float(coarse_prompt_refiner_gate_init),
                gate_max=float(coarse_prompt_refiner_gate_max),
            )
        elif self.coarse_prompt_refiner == "spatial_context_residual_cnn":
            self.prompt_refiner = SpatialContextPromptRefiner(
                hidden_channels=int(coarse_prompt_refiner_hidden),
                identity_init=True,
                max_residual=float(coarse_prompt_refiner_max_residual),
                gate_init=float(coarse_prompt_refiner_gate_init),
                gate_max=float(coarse_prompt_refiner_gate_max),
            )
        elif self.coarse_prompt_refiner == "raw_blend_context_cnn":
            self.prompt_refiner = RawBlendPromptComposer(
                hidden_channels=int(coarse_prompt_refiner_hidden),
                identity_init=True,
                gate_init=float(coarse_prompt_refiner_gate_init),
                gate_max=float(coarse_prompt_refiner_gate_max),
            )
        elif self.coarse_prompt_refiner == "precision_recall_endpoint_context_cnn":
            self.prompt_refiner = PrecisionRecallEndpointPromptComposer(
                hidden_channels=int(coarse_prompt_refiner_hidden),
                identity_init=True,
                gate_init=float(coarse_prompt_refiner_gate_init),
                gate_max=float(coarse_prompt_refiner_gate_max),
                precision_bias=float(coarse_prompt_refiner_precision_bias),
                recall_bias=float(coarse_prompt_refiner_recall_bias),
            )
        elif self.coarse_prompt_refiner == "transform_router_context_cnn":
            self.prompt_refiner = TransformRouterPromptComposer(
                hidden_channels=int(coarse_prompt_refiner_hidden),
                identity_init=True,
                gate_init=float(coarse_prompt_refiner_gate_init),
                gate_max=float(coarse_prompt_refiner_gate_max),
                conservative_scale=0.75,
                conservative_bias=float(coarse_prompt_refiner_precision_bias),
            )
        elif self.coarse_prompt_refiner == "learned_precision_recall_endpoint_context_cnn":
            self.prompt_refiner = LearnedPrecisionRecallEndpointPromptComposer(
                hidden_channels=int(coarse_prompt_refiner_hidden),
                identity_init=True,
                gate_init=float(coarse_prompt_refiner_gate_init),
                gate_max=float(coarse_prompt_refiner_gate_max),
                endpoint_residual_max=float(coarse_prompt_refiner_max_residual),
                precision_scale=0.75,
                precision_bias=float(coarse_prompt_refiner_precision_bias),
                recall_scale=1.0,
                recall_bias=float(coarse_prompt_refiner_recall_bias),
            )
        elif self.coarse_prompt_refiner == "teacher_oracle_endpoint_context_cnn":
            self.prompt_refiner = TeacherOracleEndpointPromptComposer(
                hidden_channels=int(coarse_prompt_refiner_hidden),
                identity_init=True,
                gate_init=float(coarse_prompt_refiner_gate_init),
                gate_max=float(coarse_prompt_refiner_gate_max),
                endpoint_residual_max=float(coarse_prompt_refiner_max_residual),
                precision_scale=0.75,
                precision_bias=float(coarse_prompt_refiner_precision_bias),
                recall_scale=1.0,
                recall_bias=float(coarse_prompt_refiner_recall_bias),
            )
        elif self.coarse_prompt_refiner == "post_dual_branch_context_cnn":
            self.prompt_refiner = PostTransformDualBranchPromptRefiner(
                hidden_channels=int(coarse_prompt_refiner_hidden),
                identity_init=True,
                max_residual=float(coarse_prompt_refiner_max_residual),
                gate_init=float(coarse_prompt_refiner_gate_init),
                gate_max=float(coarse_prompt_refiner_gate_max),
            )
        elif self.coarse_prompt_refiner == "feature_guided_post_dual_branch_context_cnn":
            self.prompt_refiner = FeatureGuidedPostTransformDualBranchPromptRefiner(
                hidden_channels=int(coarse_prompt_refiner_hidden),
                identity_init=True,
                max_residual=float(coarse_prompt_refiner_max_residual),
                gate_init=float(coarse_prompt_refiner_gate_init),
                gate_max=float(coarse_prompt_refiner_gate_max),
                feature_channels=int(ferret_output_channels),
            )
        elif self.coarse_prompt_refiner == "semantic_guided_post_dual_branch_context_cnn":
            self.prompt_refiner = SemanticGuidedPostTransformDualBranchPromptRefiner(
                hidden_channels=int(coarse_prompt_refiner_hidden),
                identity_init=True,
                max_residual=float(coarse_prompt_refiner_max_residual),
                gate_init=float(coarse_prompt_refiner_gate_init),
                gate_max=float(coarse_prompt_refiner_gate_max),
            )
        else:
            raise ValueError(f"Unsupported coarse_prompt_refiner={self.coarse_prompt_refiner!r}")

        # Count parameters per submodule
        param_counts = {}
        
        # Count component parameters
        for component_name, component in [
            ('FerretBackbone', self.ferret_backbone),
            # ('GLPModule', self.glp_module),
            ('SharedAdapter', self.adapters)
        ]:
            component_counts = {}
            for name, module in component.named_children():
                count = sum(p.numel() for p in module.parameters())
                component_counts[name] = count
            total_component = sum(component_counts.values())
            component_counts['total'] = total_component
            param_counts[component_name] = component_counts

        logger.debug("Component parameter counts: %s", param_counts)
        logger.info("Using dropout rate: %.3f", dropout_rate)

        # Transforms for preprocessing
        # - local: patch stream (default 256x256)
        # - global: full-image stream (default 512x512)
        self.transforms = SAM2Transforms(resolution=self.output_resolution[0], mask_threshold=0.0)
        self.global_transforms = SAM2Transforms(resolution=self.global_image_size, mask_threshold=0.0)


    def _init_sam2_backend(
        self,
        sam_config: str,
        sam_checkpoint: str,
        train_sam_iou: bool,
    ) -> None:
        """Initialize the original SAM2.1 backend."""
        sam: SAM2Base = build_sam2(sam_config, sam_checkpoint)
        sam.image_size = self.global_image_size
        self.no_mem_embed = sam.no_mem_embed if hasattr(sam, "no_mem_embed") else None
        self.directly_add_no_mem_embed = getattr(sam, "directly_add_no_mem_embed", False)
        self.encoder: ImageEncoder = sam.image_encoder
        self.decoder: MaskDecoder = sam.sam_mask_decoder
        self.sam_prompt_encoder: PromptEncoder = sam.sam_prompt_encoder

        embed_h = max(1, int(self.global_image_size) // 16)
        embed_w = max(1, int(self.global_image_size) // 16)
        self.sam_prompt_encoder.image_embedding_size = (embed_h, embed_w)
        self.decoder.debug_mode = False
        self._freeze_sam_backend(train_sam_iou=train_sam_iou)

    def _init_sam3_interactive_backend(
        self,
        sam_checkpoint: str,
        train_sam_iou: bool,
    ) -> None:
        """Initialize SAM3 with its optional SAM-style interactive decoder path."""
        from sam3.model_builder import build_sam3_image_model

        self.sam3_model = build_sam3_image_model(
            checkpoint_path=sam_checkpoint,
            load_from_HF=False,
            device="cpu",
            eval_mode=True,
            compile=False,
            enable_inst_interactivity=True,
        )
        if self.sam3_model.inst_interactive_predictor is None:
            raise RuntimeError("SAM3 interactive backend did not create inst_interactive_predictor")

        self.sam3_tracker = self.sam3_model.inst_interactive_predictor.model
        self.encoder = self.sam3_model.backbone
        self.decoder = self.sam3_tracker.sam_mask_decoder
        self.sam_prompt_encoder = self.sam3_tracker.sam_prompt_encoder
        self.no_mem_embed = getattr(self.sam3_tracker, "no_mem_embed", None)
        self.directly_add_no_mem_embed = False
        self.sam3_image_transform = v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(int(self.sam3_tracker.image_size), int(self.sam3_tracker.image_size))),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self._freeze_sam_backend(train_sam_iou=train_sam_iou)

    def _freeze_sam_backend(self, train_sam_iou: bool) -> None:
        """Freeze SAM backend components, optionally leaving IoU head trainable."""
        if self.sam3_model is not None:
            for p in self.sam3_model.parameters():
                p.requires_grad = False
        else:
            for p in self.encoder.parameters():
                p.requires_grad = False

        for name, p in self.decoder.named_parameters():
            if "iou_prediction_head" in name and train_sam_iou:
                p.requires_grad = True
                logger.info("Keeping %s trainable for IoU prediction", name)
            else:
                p.requires_grad = False
        for p in self.sam_prompt_encoder.parameters():
            p.requires_grad = False
        self.set_frozen_sam_eval()

    def set_frozen_sam_eval(self) -> None:
        """Keep frozen SAM backend modules in eval mode during adapter training."""
        for module in (
            getattr(self, "sam3_model", None),
            getattr(self, "sam3_tracker", None),
            getattr(self, "encoder", None),
            getattr(self, "decoder", None),
            getattr(self, "sam_prompt_encoder", None),
        ):
            if module is not None:
                module.eval()

    def train(self, mode: bool = True) -> "ForgeryLocalizer":
        """Set training mode while preserving eval mode for frozen SAM modules."""
        super().train(mode)
        if mode:
            self.set_frozen_sam_eval()
        return self

    def _decoder_feature_sizes(self) -> Tuple[Tuple[int, int], List[Tuple[int, int]]]:
        """Return decoder image-embedding and high-res feature spatial sizes."""
        if getattr(self, "sam_backend", "sam2") == "sam3_interactive":
            embed_h, embed_w = self.sam_prompt_encoder.image_embedding_size
            return (int(embed_h), int(embed_w)), [
                (int(embed_h) * 4, int(embed_w) * 4),
                (int(embed_h) * 2, int(embed_w) * 2),
            ]

        out_h, out_w = self.output_resolution
        return (
            max(1, int(out_h) // 16),
            max(1, int(out_w) // 16),
        ), [
            (max(1, int(out_h) // 4), max(1, int(out_w) // 4)),
            (max(1, int(out_h) // 8), max(1, int(out_w) // 8)),
        ]

    @staticmethod
    def _add_ferret_diagnostic_extras(extras: Dict[str, torch.Tensor], ferret_backbone: nn.Module) -> None:
        """Expose lightweight FerretBackbone diagnostic tensors in ``extras``.

        These tensors are consumed only by probes/training diagnostics.  They
        intentionally preserve object identity instead of detaching so callers
        that need gradient-aware diagnostics can opt into them, while no-grad
        probes can safely serialize summaries.
        """
        tau_weights = getattr(ferret_backbone, "_last_lad_tau_weights", None)
        if tau_weights is not None:
            extras["lad_tau_weights"] = tau_weights

    def forward(
        self,
        orig: torch.Tensor,
        streams: Sequence[torch.Tensor] = None,  # Deprecated, kept for compatibility
        output_extras: bool = False,
        *,
        global_image: Optional[torch.Tensor] = None,
        norm_coords: Optional[torch.Tensor] = None,
        global_context: Optional[Tuple[Sequence[torch.Tensor], Tuple[int, int]]] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Run Ferret-SAM localization forward pass.

        Args:
            orig: Local patch tensor of shape ``[B, 3, H, W]`` (typically 256x256).
            streams: Deprecated, kept for compatibility.
            output_extras: Whether to return auxiliary outputs for loss computation.
            global_image: Optional global resized image (typically 512x512) for semantic context.
            norm_coords: Optional normalized patch coords ``[B, 4]`` (x1,y1,x2,y2) in original image space.
            global_context: Optional cached global semantic features returned by ``encode_global``.

        Returns:
            Refined mask logits, optionally with auxiliary outputs.
        """

        # ------------------------------
        # 1. Dual-Stream Feature Extraction
        # ------------------------------
        
        # Stream A: Semantic Stream (Frozen SAM2 Encoder) on global image.
        if global_context is not None:
            semantic_global, global_hw = global_context
            if norm_coords is None:
                raise ValueError("norm_coords must be provided when using global_context")
            semantic_features = self._roi_align_semantic_features(semantic_global, norm_coords, global_hw)
        elif global_image is not None and norm_coords is not None:
            semantic_global, global_hw = self.encode_global(global_image)
            semantic_features = self._roi_align_semantic_features(semantic_global, norm_coords, global_hw)
        else:
            # Backward compatible path: semantic features from the local patch itself.
            semantic_features, _ = self.encode_global(orig)
        
        # Stream B: Forensic Stream (Trainable FerretNet Backbone).  Newer
        # Ferret backbones can return an independent dense prompt map for SAM
        # while keeping the supervised coarse mask on the stable
        # mask_compressor head.  Preserve compatibility with older/fake
        # backbones used by checkpoints and tests.
        adapter_forensic_source = str(getattr(self, "adapter_forensic_source", "final"))
        request_forensic_pyramid = adapter_forensic_source == "final_plus_pyramid"
        try:
            ferret_outputs = self.ferret_backbone(
                orig,
                return_dense_prompt=True,
                return_forensic_pyramid=request_forensic_pyramid,
            )
        except TypeError as exc:
            if "return_forensic_pyramid" in str(exc):
                ferret_outputs = self.ferret_backbone(orig, return_dense_prompt=True)
            elif "return_dense_prompt" not in str(exc):
                raise
            else:
                ferret_outputs = self.ferret_backbone(orig)
        forensic_pyramid = None
        if len(ferret_outputs) == 5:
            (
                forensic_features,
                coarse_mask,
                detection_logit,
                dense_prompt_mask,
                forensic_pyramid,
            ) = ferret_outputs
        elif len(ferret_outputs) == 4:
            forensic_features, coarse_mask, detection_logit, dense_prompt_mask = ferret_outputs
        elif len(ferret_outputs) == 3:
            forensic_features, coarse_mask, detection_logit = ferret_outputs
            dense_prompt_mask = coarse_mask
        else:
            raise RuntimeError(
                "ferret_backbone must return either "
                "(features, coarse_mask, detection_logit), "
                "(features, coarse_mask, detection_logit, dense_prompt_mask), or "
                "(features, coarse_mask, detection_logit, dense_prompt_mask, forensic_pyramid)"
            )
        
        # ------------------------------
        # 3. Cross-Modal Feature Fusion (Adapters)
        # ------------------------------        
        
        # Adapter injection at each semantic feature scale with forensic features
        adapted, adapter_diagnostics = ForgeryLocalizer._adapt_semantic_features(
            self,
            forensic_features,
            semantic_features,
            forensic_pyramid=forensic_pyramid,
        )
        
        # ------------------------------
        # 5. SAM2 Decoder with Heatmap Prompt
        # ------------------------------
        
        # Use adapted features for decoder input
        image_embeddings = adapted[0]
        high_res_features = [adapted[1], adapted[2]]
        
        # For 1024x1024 input, the encoder output is already at the correct scale for decoder input
        # No need for additional upsampling
        # The original code assumed 512x512 input, which required upsampling by factor 2
        
        # Verify the expected feature sizes for the active decoder backend.
        expected_image_embedding_size, expected_high_res_sizes = self._decoder_feature_sizes()

        # Only upsample if feature sizes don't match expected sizes
        if image_embeddings.shape[-2:] != expected_image_embedding_size:
            image_embeddings = F.interpolate(
                image_embeddings, 
                size=expected_image_embedding_size,
                mode='bilinear', 
                align_corners=False
            )
        
        for i in range(len(high_res_features)):
            if high_res_features[i].shape[-2:] != expected_high_res_sizes[i]:
                high_res_features[i] = F.interpolate(
                    high_res_features[i], 
                    size=expected_high_res_sizes[i],
                    mode='bilinear', 
                    align_corners=False
                )
        
        coarse_prompt_source = dense_prompt_mask if dense_prompt_mask is not None else coarse_mask
        prompt_mode = str(getattr(self, "sam3_prompt_mode", "legacy"))
        if prompt_mode in {"dual_avg", "dual_gated", "dual_spatial_gated"}:
            (
                legacy_logits,
                legacy_iou,
                legacy_tokens,
                coarse_prompt,
                prompt_calibrator_diagnostics,
            ) = ForgeryLocalizer._decode_with_dense_prompt(
                self,
                coarse_prompt_source=coarse_prompt_source,
                coarse_mask=coarse_mask,
                dense_prompt_mask=dense_prompt_mask,
                forensic_features=forensic_features,
                detection_logit=detection_logit,
                image_embeddings=image_embeddings,
                high_res_features=high_res_features,
                use_native_mask_resize=False,
                use_dummy_no_point_prompt=False,
            )
            (
                native_logits,
                native_iou,
                _native_tokens,
                _native_prompt,
                _native_diagnostics,
            ) = ForgeryLocalizer._decode_with_dense_prompt(
                self,
                coarse_prompt_source=coarse_prompt_source,
                coarse_mask=coarse_mask,
                dense_prompt_mask=dense_prompt_mask,
                forensic_features=forensic_features,
                detection_logit=detection_logit,
                image_embeddings=image_embeddings,
                high_res_features=high_res_features,
                use_native_mask_resize=True,
                use_dummy_no_point_prompt=True,
            )
            if prompt_mode == "dual_gated":
                fusion_gate = self.dual_prompt_fusion_gate(
                    prompt_source=coarse_prompt_source,
                    forensic_features=forensic_features,
                ).to(device=legacy_logits.device, dtype=legacy_logits.dtype)
            elif prompt_mode == "dual_spatial_gated":
                fusion_gate = self.dual_prompt_fusion_gate(
                    prompt_source=coarse_prompt_source,
                    forensic_features=forensic_features,
                    legacy_logits=legacy_logits,
                    native_logits=native_logits,
                ).to(device=legacy_logits.device, dtype=legacy_logits.dtype)
            else:
                fusion_gate = legacy_logits.new_full(
                    (legacy_logits.shape[0], 1, 1, 1),
                    0.5,
                )
            mask_logits = legacy_logits * (1.0 - fusion_gate) + native_logits * fusion_gate
            iou_gate = fusion_gate.flatten(1).mean(dim=1, keepdim=True).to(dtype=legacy_iou.dtype)
            iou_pred = legacy_iou * (1.0 - iou_gate) + native_iou * iou_gate
            sam_tokens_out = legacy_tokens
            prompt_calibrator_diagnostics = {
                **prompt_calibrator_diagnostics,
                "dual_prompt_fusion_gate": fusion_gate,
                "dual_prompt_legacy_logits": legacy_logits,
                "dual_prompt_native_logits": native_logits,
            }
        else:
            sam3_interactive_backend = getattr(self, "sam_backend", "sam2") == "sam3_interactive"
            use_native_mask_resize = sam3_interactive_backend and prompt_mode in {
                "native",
                "native_resize_only",
            }
            use_dummy_no_point_prompt = sam3_interactive_backend and prompt_mode in {
                "native",
                "legacy_dummy_point",
            }
            (
                mask_logits,
                iou_pred,
                sam_tokens_out,
                coarse_prompt,
                prompt_calibrator_diagnostics,
            ) = ForgeryLocalizer._decode_with_dense_prompt(
                self,
                coarse_prompt_source=coarse_prompt_source,
                coarse_mask=coarse_mask,
                dense_prompt_mask=dense_prompt_mask,
                forensic_features=forensic_features,
                detection_logit=detection_logit,
                image_embeddings=image_embeddings,
                high_res_features=high_res_features,
                use_native_mask_resize=use_native_mask_resize,
                use_dummy_no_point_prompt=use_dummy_no_point_prompt,
            )
        
        # Check and fix iou_pred immediately after decoder output
        # This prevents NaN gradients in iou_prediction_head
        # if torch.isnan(iou_pred).any() or torch.isinf(iou_pred).any():
        #     logger.warning(f"Found NaN/inf in decoder iou_pred, replacing with reasonable values")
        #     # Replace NaN/inf with reasonable values (0.5 for neutral IoU prediction)
        #     iou_pred = torch.nan_to_num(iou_pred, nan=0.5, posinf=1.0, neginf=0.0)
        #     # Also clamp to valid IoU range
        #     iou_pred = torch.clamp(iou_pred, min=0.0, max=1.0)
        # Always clamp iou_pred to ensure it stays within valid range
        # This provides additional safety against extreme values that could cause NaN gradients
        iou_pred = torch.clamp(iou_pred, min=0.0, max=1.0)
        
        processed_mask_logits = self.transforms.postprocess_masks(mask_logits, torch.Size(self.output_resolution))
        final_logit_calibrator_diagnostics: Optional[Dict[str, torch.Tensor]] = None
        final_logit_calibrator = getattr(self, "final_logit_calibrator", None)
        if final_logit_calibrator is not None:
            final_context = {
                "coarse_prompt": coarse_prompt,
                "dense_prompt_mask": dense_prompt_mask,
                "coarse_mask": coarse_mask,
                "detection_logit": detection_logit,
                "forensic_features": forensic_features,
                "image_embeddings": image_embeddings,
                "high_res_features": high_res_features,
            }
            processed_mask_logits, final_logit_calibrator_diagnostics = final_logit_calibrator(
                processed_mask_logits,
                context=final_context,
            )
        
        if output_extras:
            # With multimask_output=False, iou_pred is already shaped [B, 1]
            extras = {
                'iou_pred': iou_pred,
                'sam_tokens_out': sam_tokens_out,
                'coarse_mask': coarse_mask,
                'dense_prompt_mask': dense_prompt_mask,
                'coarse_prompt': coarse_prompt,
                'detection_logit': detection_logit,
                # 'heatmap_prompt': heatmap_prompt,  # Add heatmap prompt to extras
                'forensic_features': forensic_features,  # Add forensic features (LPD input) to extras
            }
            if final_logit_calibrator_diagnostics is not None:
                extras["final_logit_calibrator"] = final_logit_calibrator_diagnostics
                extras["raw_final_logits"] = final_logit_calibrator_diagnostics["pre_calibrator_logits"]
            if adapter_diagnostics is not None:
                extras["adapter_diagnostics"] = adapter_diagnostics
            sample_gates = getattr(self.adapters, "_last_sample_gates", None)
            if sample_gates is not None:
                extras["adapter_sample_gates"] = sample_gates
            if prompt_calibrator_diagnostics is not None:
                extras["prompt_calibrator"] = prompt_calibrator_diagnostics
                if "dual_prompt_fusion_gate" in prompt_calibrator_diagnostics:
                    extras["dual_prompt_fusion_gate"] = prompt_calibrator_diagnostics[
                        "dual_prompt_fusion_gate"
                    ]
                if "dual_prompt_legacy_logits" in prompt_calibrator_diagnostics:
                    extras["dual_prompt_legacy_logits"] = prompt_calibrator_diagnostics[
                        "dual_prompt_legacy_logits"
                    ]
                if "dual_prompt_native_logits" in prompt_calibrator_diagnostics:
                    extras["dual_prompt_native_logits"] = prompt_calibrator_diagnostics[
                        "dual_prompt_native_logits"
                    ]
                post_branch_key_map = {
                    "refiner_fg_gate": "post_prompt_fg_gate",
                    "refiner_bg_gate": "post_prompt_bg_gate",
                    "refiner_fg_residual": "post_prompt_fg_residual",
                    "refiner_bg_residual": "post_prompt_bg_residual",
                    "refiner_fg_delta": "post_prompt_fg_delta",
                    "refiner_bg_delta": "post_prompt_bg_delta",
                    "refiner_blend_gate": "post_prompt_blend_gate",
                    "refiner_blend_delta": "post_prompt_blend_delta",
                }
                for diag_key, extra_key in post_branch_key_map.items():
                    if diag_key in prompt_calibrator_diagnostics:
                        extras[extra_key] = prompt_calibrator_diagnostics[diag_key]
            dense_gate = getattr(self.ferret_backbone, "_last_dense_prompt_gate", None)
            if dense_gate is not None:
                extras["dense_prompt_gate"] = dense_gate
            dense_small_gate = getattr(self.ferret_backbone, "_last_dense_prompt_small_gate", None)
            if dense_small_gate is not None:
                extras["dense_prompt_small_gate"] = dense_small_gate
            dense_signed_delta = getattr(self.ferret_backbone, "_last_dense_prompt_signed_delta", None)
            if dense_signed_delta is not None:
                extras["dense_prompt_signed_delta"] = dense_signed_delta
            dense_signed_gate = getattr(self.ferret_backbone, "_last_dense_prompt_signed_gate", None)
            if dense_signed_gate is not None:
                extras["dense_prompt_signed_gate"] = dense_signed_gate
            dense_pre_unet = getattr(self.ferret_backbone, "_last_dense_prompt_pre_unet", None)
            if dense_pre_unet is not None:
                extras["dense_prompt_pre_unet"] = dense_pre_unet
            dense_unet_delta = getattr(self.ferret_backbone, "_last_dense_prompt_unet_delta", None)
            if dense_unet_delta is not None:
                extras["dense_prompt_unet_delta"] = dense_unet_delta
            dense_unet_gate = getattr(self.ferret_backbone, "_last_dense_prompt_unet_gate", None)
            if dense_unet_gate is not None:
                extras["dense_prompt_unet_gate"] = dense_unet_gate
            dense_area_bias = getattr(self.ferret_backbone, "_last_dense_prompt_area_bias", None)
            if dense_area_bias is not None:
                extras["dense_prompt_area_bias"] = dense_area_bias
            ForgeryLocalizer._add_ferret_diagnostic_extras(extras, self.ferret_backbone)
            dual_branch_diagnostics = {
                "dense_prompt_fg_gate": getattr(self.ferret_backbone, "_last_dense_prompt_fg_gate", None),
                "dense_prompt_bg_gate": getattr(self.ferret_backbone, "_last_dense_prompt_bg_gate", None),
                "dense_prompt_core_gate": getattr(self.ferret_backbone, "_last_dense_prompt_core_gate", None),
                "dense_prompt_fg_residual": getattr(self.ferret_backbone, "_last_dense_prompt_fg_residual", None),
                "dense_prompt_bg_residual": getattr(self.ferret_backbone, "_last_dense_prompt_bg_residual", None),
                "dense_prompt_core_residual": getattr(self.ferret_backbone, "_last_dense_prompt_core_residual", None),
            }
            for key, value in dual_branch_diagnostics.items():
                if value is not None:
                    extras[key] = value

            return processed_mask_logits, extras
        return processed_mask_logits

    def _decode_with_dense_prompt(
        self,
        *,
        coarse_prompt_source: torch.Tensor,
        coarse_mask: torch.Tensor,
        dense_prompt_mask: Optional[torch.Tensor],
        forensic_features: torch.Tensor,
        detection_logit: torch.Tensor,
        image_embeddings: torch.Tensor,
        high_res_features: Sequence[torch.Tensor],
        use_native_mask_resize: bool,
        use_dummy_no_point_prompt: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        coarse_prompt = coarse_prompt_source
        prompt_calibrator_diagnostics: Dict[str, torch.Tensor] = {}
        if coarse_prompt_source is not None:
            if use_native_mask_resize:
                expected_mask_size = tuple(
                    int(v)
                    for v in getattr(
                        self.sam_prompt_encoder,
                        "mask_input_size",
                        (
                            int(image_embeddings.shape[-2]) * 4,
                            int(image_embeddings.shape[-1]) * 4,
                        ),
                    )
                )
            else:
                expected_mask_size = (
                    int(image_embeddings.shape[-2]) * 4,
                    int(image_embeddings.shape[-1]) * 4,
                )
            if coarse_prompt_source.shape[-2:] != expected_mask_size:
                coarse_prompt_source = F.interpolate(
                    coarse_prompt_source,
                    size=expected_mask_size,
                    mode="bilinear",
                    align_corners=False,
                    antialias=use_native_mask_resize,
                )
            calibrator_context = {
                "raw_prompt": coarse_prompt_source,
                "coarse_mask": coarse_mask,
                "dense_prompt_mask": dense_prompt_mask,
                "forensic_features": forensic_features,
                "detection_logit": detection_logit,
                "image_embeddings": image_embeddings,
                "high_res_features": high_res_features,
            }
            coarse_prompt, prompt_calibrator_diagnostics = ForgeryLocalizer._prepare_coarse_prompt(
                self,
                coarse_prompt_source,
                return_diagnostics=True,
                calibrator_context=calibrator_context,
            )

        self.sam_prompt_encoder.image_embedding_size = image_embeddings.shape[-2:]

        prompt_points = None
        if use_dummy_no_point_prompt:
            batch_size = int(image_embeddings.shape[0])
            point_coords = torch.zeros(
                batch_size,
                1,
                2,
                device=image_embeddings.device,
                dtype=image_embeddings.dtype,
            )
            point_labels = -torch.ones(
                batch_size,
                1,
                device=image_embeddings.device,
                dtype=torch.int32,
            )
            prompt_points = (point_coords, point_labels)

        sparse_prompt_embeddings, dense_prompt_embeddings = self.sam_prompt_encoder(
            points=prompt_points,
            boxes=None,
            masks=coarse_prompt,
        )

        mask_logits, iou_pred, sam_tokens_out, _object_score_logits = self.decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        return mask_logits, iou_pred, sam_tokens_out, coarse_prompt, prompt_calibrator_diagnostics

    def _prepare_coarse_prompt(
        self,
        coarse_mask: torch.Tensor,
        return_diagnostics: bool = False,
        calibrator_context: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Convert raw coarse logits into the dense mask prompt sent to SAM.

        SAM prompt encoders expect mask-like logits where negative values denote
        background and positive values denote foreground.  LAD/coarse heads can
        easily produce an all-positive map; optional per-sample centering/z-score
        preserves spatial contrast while restoring signed prompt semantics.
        """
        transform = str(getattr(self, "coarse_prompt_transform", "none"))
        prompt = coarse_mask
        if transform == "none":
            pass
        elif transform in {"center", "zscore"}:
            dims = tuple(range(1, prompt.ndim))
            mean = prompt.mean(dim=dims, keepdim=True)
            prompt = prompt - mean
            if transform == "zscore":
                std = prompt.std(dim=dims, keepdim=True, unbiased=False)
                eps = float(getattr(self, "coarse_prompt_eps", 1e-6))
                prompt = prompt / (std + eps)
        else:
            raise ValueError(f"Unsupported coarse_prompt_transform={transform!r}")

        scale = float(getattr(self, "coarse_prompt_scale", 1.0))
        bias = float(getattr(self, "coarse_prompt_bias", 0.0))
        prompt = prompt * scale + bias
        diagnostics: Dict[str, torch.Tensor] = {}
        calibrator = getattr(self, "prompt_calibrator", None)
        if calibrator is not None:
            pre_calibrator_prompt = prompt
            try:
                prompt, calibrator_diagnostics = calibrator(prompt, context=calibrator_context)
            except TypeError as exc:
                if "unexpected keyword argument 'context'" not in str(exc):
                    raise
                prompt, calibrator_diagnostics = calibrator(prompt)
            diagnostics = {
                **calibrator_diagnostics,
                "pre_calibrator_prompt": pre_calibrator_prompt,
            }
        refiner = getattr(self, "prompt_refiner", None)
        if refiner is not None:
            diagnostics["pre_refiner_prompt"] = prompt.detach()
            try:
                prompt, refiner_diagnostics = refiner(prompt, context=calibrator_context)
            except TypeError as exc:
                if "unexpected keyword argument 'context'" not in str(exc):
                    raise
                prompt, refiner_diagnostics = refiner(prompt)
            diagnostics = {**diagnostics, **{f"refiner_{k}": v for k, v in refiner_diagnostics.items()}}
        prompt = torch.clamp(prompt, min=-10.0, max=10.0)
        if return_diagnostics:
            return prompt, diagnostics
        return prompt

    def _adapt_semantic_features(
        self,
        forensic_features: torch.Tensor,
        semantic_features: Sequence[torch.Tensor],
        forensic_pyramid: Optional[Sequence[torch.Tensor]] = None,
    ) -> Tuple[List[torch.Tensor], Optional[List[Dict[str, torch.Tensor]]]]:
        """Run cross-modal adapters and optionally collect per-scale diagnostics."""
        adapted: List[torch.Tensor] = []
        diagnostics: List[Dict[str, torch.Tensor]] = []
        collect_diagnostics = bool(getattr(self, "adapter_diagnostics_enabled", False))
        pyramid_adapters = getattr(self, "pyramid_adapters", None)
        for i in range(3):
            if collect_diagnostics:
                prompt, stats = self.adapters(
                    forensic_features,
                    semantic_features[i],
                    i,
                    return_diagnostics=True,
                )
                diagnostics.append(stats)
            else:
                prompt = self.adapters(
                    forensic_features,
                    semantic_features[i],
                    i,
                )
            if pyramid_adapters is not None and forensic_pyramid is not None:
                pyramid_input = forensic_pyramid[i]
                if collect_diagnostics:
                    pyramid_prompt, pyramid_stats = pyramid_adapters(
                        pyramid_input,
                        semantic_features[i],
                        i,
                        return_diagnostics=True,
                    )
                    pyramid_stats = {**pyramid_stats, "source": "pyramid"}
                    diagnostics.append(pyramid_stats)
                else:
                    pyramid_prompt = pyramid_adapters(
                        pyramid_input,
                        semantic_features[i],
                        i,
                    )
                prompt = prompt + (pyramid_prompt - semantic_features[i])
            adapted.append(prompt)
        return adapted, diagnostics if collect_diagnostics else None

    def encode_global(
        self, global_image: torch.Tensor
    ) -> Tuple[List[torch.Tensor], Tuple[int, int]]:
        """Encode a global resized image into SAM semantic feature maps.

        Returns:
            semantic_features: [image_embeddings, high_res_s0, high_res_s1]
            global_hw: (H, W) of the original global image tensor.
        """
        if self.sam_backend == "sam3_interactive":
            return self._encode_global_sam3_interactive(global_image)
        return self._encode_global_sam2(global_image)

    def _encode_global_sam2(
        self, global_image: torch.Tensor
    ) -> Tuple[List[torch.Tensor], Tuple[int, int]]:
        global_hw = (int(global_image.shape[-2]), int(global_image.shape[-1]))
        with torch.no_grad():
            out = self.encoder(global_image)
            feats = out["backbone_fpn"]
            image_embeddings, high_res_features = self._apply_post_adapted_processing(feats)
            semantic_features = [image_embeddings, high_res_features[0], high_res_features[1]]
        return semantic_features, global_hw

    def _preprocess_sam3_image_batch(self, image: torch.Tensor) -> torch.Tensor:
        """Apply SAM3 image preprocessing to a BCHW tensor batch."""
        if image.dim() != 4 or image.shape[1] != 3:
            raise ValueError(f"SAM3 image batch must be BCHW with 3 channels, got {tuple(image.shape)}")
        if self.sam3_image_transform is None:
            raise RuntimeError("SAM3 image transform is not initialized")
        if image.is_floating_point() and (
            bool((image.detach().amin() < -0.05).item())
            or bool((image.detach().amax() > 1.05).item())
        ):
            mean = image.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = image.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            image = (image * std + mean).clamp_(0.0, 1.0)
        return self.sam3_image_transform(image)

    def _encode_global_sam3_interactive(
        self, global_image: torch.Tensor
    ) -> Tuple[List[torch.Tensor], Tuple[int, int]]:
        """Encode image with SAM3 and return SAM-style interactive decoder features."""
        if self.sam3_model is None or self.sam3_tracker is None:
            raise RuntimeError("SAM3 interactive backend is not initialized")

        global_hw = (int(global_image.shape[-2]), int(global_image.shape[-1]))
        with torch.no_grad():
            sam3_image = self._preprocess_sam3_image_batch(global_image)
            backbone_out = self.sam3_model.backbone.forward_image(sam3_image)
            sam2_backbone_out = backbone_out.get("sam2_backbone_out")
            if sam2_backbone_out is None:
                raise RuntimeError("SAM3 interactive backend did not produce sam2_backbone_out")

            fpn = list(sam2_backbone_out["backbone_fpn"])
            if len(fpn) != 3:
                raise RuntimeError(f"Expected 3 SAM3 interactive FPN levels, got {len(fpn)}")

            high_res_s0 = self.decoder.conv_s0(fpn[0])
            high_res_s1 = self.decoder.conv_s1(fpn[1])
            image_embeddings = fpn[2]
            if self.no_mem_embed is not None:
                no_mem = self.no_mem_embed.reshape(1, -1, 1, 1).to(
                    device=image_embeddings.device,
                    dtype=image_embeddings.dtype,
                )
                image_embeddings = image_embeddings + no_mem

            semantic_features = [image_embeddings, high_res_s0, high_res_s1]
        return semantic_features, global_hw

    def _roi_align_semantic_features(
        self,
        semantic_features: Sequence[torch.Tensor],
        norm_coords: torch.Tensor,
        global_hw: Tuple[int, int],
    ) -> List[torch.Tensor]:
        """ROI Align global semantic features to the local patch region."""
        if norm_coords.dim() == 1:
            norm_coords = norm_coords.unsqueeze(0)
        if norm_coords.dim() != 2 or norm_coords.shape[-1] != 4:
            raise ValueError("norm_coords must have shape [B, 4]")

        global_h, global_w = int(global_hw[0]), int(global_hw[1])
        if global_h <= 0 or global_w <= 0:
            raise ValueError("global_hw must be positive")

        if getattr(self, "sam_backend", "sam2") == "sam3_interactive":
            image_size, high_res_sizes = self._decoder_feature_sizes()
            roi_sizes = [image_size, high_res_sizes[0], high_res_sizes[1]]
        else:
            out_h, out_w = int(self.output_resolution[0]), int(self.output_resolution[1])
            roi_sizes = [
                (max(1, out_h // 16), max(1, out_w // 16)),
                (max(1, out_h // 4), max(1, out_w // 4)),
                (max(1, out_h // 8), max(1, out_w // 8)),
            ]

        B = int(norm_coords.shape[0])
        device = semantic_features[0].device
        feat_batch = int(semantic_features[0].shape[0])

        coords = norm_coords.to(device=device, dtype=torch.float32)
        full_roi = coords.new_tensor([0.0, 0.0, 1.0, 1.0]).view(1, 4)
        if (
            feat_batch == B
            and torch.allclose(coords, full_roi.expand_as(coords), atol=1e-6, rtol=0.0)
            and all(tuple(feat.shape[-2:]) == tuple(size) for feat, size in zip(semantic_features, roi_sizes))
        ):
            return list(semantic_features)

        x1 = coords[:, 0] * float(global_w)
        y1 = coords[:, 1] * float(global_h)
        x2 = coords[:, 2] * float(global_w)
        y2 = coords[:, 3] * float(global_h)

        x1 = torch.clamp(x1, min=0.0, max=float(max(global_w - 1, 0)))
        y1 = torch.clamp(y1, min=0.0, max=float(max(global_h - 1, 0)))
        x2 = torch.clamp(x2, min=1.0, max=float(global_w))
        y2 = torch.clamp(y2, min=1.0, max=float(global_h))

        eps = 1e-3
        x2 = torch.minimum(x2, x2.new_tensor(float(global_w)))
        y2 = torch.minimum(y2, y2.new_tensor(float(global_h)))
        x2 = torch.maximum(x2, x1 + eps)
        y2 = torch.maximum(y2, y1 + eps)

        if feat_batch == B:
            batch_idx = torch.arange(B, device=device, dtype=torch.float32)
        elif feat_batch == 1:
            batch_idx = torch.zeros(B, device=device, dtype=torch.float32)
        else:
            raise ValueError(
                f"Semantic feature batch={feat_batch} incompatible with norm_coords batch={B}"
            )
        boxes = torch.stack([batch_idx, x1, y1, x2, y2], dim=1)

        aligned_features: List[torch.Tensor] = []
        for feat, output_size in zip(semantic_features, roi_sizes):
            feat_h, feat_w = int(feat.shape[-2]), int(feat.shape[-1])
            if feat_h <= 0 or feat_w <= 0:
                raise ValueError("semantic feature maps must be non-empty")

            spatial_scale_w = float(feat_w) / float(global_w)
            spatial_scale_h = float(feat_h) / float(global_h)
            spatial_scale = float((spatial_scale_w + spatial_scale_h) / 2.0)

            roi_in = feat
            if roi_in.device.type == "cpu" and roi_in.dtype != torch.float32:
                roi_in = roi_in.float()

            roi = roi_align(
                roi_in,
                boxes,
                output_size=output_size,
                spatial_scale=spatial_scale,
                sampling_ratio=-1,
                aligned=True,
            )
            if roi.dtype != feat.dtype:
                roi = roi.to(dtype=feat.dtype)
            aligned_features.append(roi)

        return aligned_features

    def _apply_post_adapted_processing(
        self,
        adapted: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Project SAM features into decoder inputs after adapter fusion."""
        image_embeddings = adapted[-1]
        if self.no_mem_embed is not None:
            no_mem_embed_reshaped = self.no_mem_embed.reshape(1, 256, 1, 1).detach()
            image_embeddings = image_embeddings + no_mem_embed_reshaped
        
        # 3. Use adapted[0] and adapted[1] as high_res_features (highest-res)
        adapted_proj0 = self.decoder.conv_s0(adapted[0])  # [B, 32, 128, 128]
        adapted_proj1 = self.decoder.conv_s1(adapted[1])  # [B, 64, 64, 64]
        high_res_features = [adapted_proj0, adapted_proj1]
        return image_embeddings, high_res_features
