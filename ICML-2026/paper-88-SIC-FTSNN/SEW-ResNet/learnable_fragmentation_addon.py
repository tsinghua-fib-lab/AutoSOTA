"""Learnable fragmentation add-on, aligned with the uploaded ICML paper.

This module implements the input-side learnable fragmentation mechanism described in the
uploaded ICML paper:
- T-1 learnable division lines, each parameterized by (h_k, v_k, r_k)
- bounded reparameterization for line coefficients
- hard masks in the forward path with soft gradients via a straight-through estimator
- optional dynamic fragment-number selection over a candidate set with Gumbel-Softmax
- temporal resampling to T_max and fragment-level mixing for dynamic training
- balance loss and entropy-based decoding utilities

All operations are pure PyTorch and independent from the SNN backend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


__all__ = [
    "FragmentationOutput",
    "LearnableLineFragmentation",
    "DynamicLearnableFragmentation",
    "resample_fragment_sequence",
    "entropy_weighted_decode",
    "EntropyTimeDecoder",
    "MeanTimeDecoder",
]


@dataclass
class FragmentationOutput:
    sequence: Tensor  # [B, T, C, H, W]
    balance_loss: Tensor
    soft_masks: Optional[Tensor] = None  # [T, H, W]
    hard_masks: Optional[Tensor] = None  # [T, H, W]
    selector_probs: Optional[Tensor] = None  # [num_candidates]
    selected_t: Optional[int] = None



def _atanh_safe(x: Tensor) -> Tensor:
    x = x.clamp(-0.999999, 0.999999)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))



def _centered_coordinate_grid(h: int, w: int, device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, Tensor, float]:
    ys = torch.arange(h, device=device, dtype=dtype) - (h - 1) / 2.0
    xs = torch.arange(w, device=device, dtype=dtype) - (w - 1) / 2.0
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    half_diag = float(math.sqrt(((h - 1) / 2.0) ** 2 + ((w - 1) / 2.0) ** 2) + 1e-12)
    return xx, yy, half_diag



def _direction_to_unit_normal(direction: str) -> Tuple[float, float]:
    d = str(direction).strip().lower()
    if d in {"horizontal", "h", "rows"}:
        return 0.0, 1.0
    if d in {"vertical", "v", "cols", "columns"}:
        return 1.0, 0.0
    if d in {"diag_lr", "diag\\", "tl_br"}:
        s2 = math.sqrt(2.0)
        return 1.0 / s2, -1.0 / s2
    if d in {"diag_rl", "diag/", "tr_bl"}:
        s2 = math.sqrt(2.0)
        return 1.0 / s2, 1.0 / s2
    raise ValueError(
        "direction must be one of {'horizontal', 'vertical', 'diag_lr', 'diag_rl'}"
    )



def _equal_width_offsets(num_fragments: int, h: int, w: int, direction: str, device: torch.device, dtype: torch.dtype) -> Tensor:
    if num_fragments < 2:
        raise ValueError("num_fragments must be >= 2")
    a0, b0 = _direction_to_unit_normal(direction)
    xx, yy, half_diag = _centered_coordinate_grid(h, w, device=device, dtype=dtype)
    s = a0 * xx + b0 * yy
    thresholds = torch.linspace(float(s.min()), float(s.max()), steps=num_fragments + 1, device=device, dtype=dtype)[1:-1]
    c = -thresholds
    return _atanh_safe((c / float(half_diag)).clamp(-0.999999, 0.999999))



def _straight_through_binary(prob: Tensor) -> Tensor:
    hard = (prob > 0.5).to(prob.dtype)
    return hard.detach() - prob.detach() + prob



def resample_fragment_sequence(sequence: Tensor, t_max: int) -> Tensor:
    r"""Temporal alignment R_T used in the ICML paper.

    Input shape: [B, T, C, H, W]
    Output shape: [B, T_max, C, H, W]

    The paper's Algorithm 1 uses:
      \hat I_t^(T) = I_{ceil(tT / T_max)}^(T), t in {1, ..., T_max}
    """
    if sequence.dim() != 5:
        raise ValueError(f"Expected [B, T, C, H, W], got {tuple(sequence.shape)}")
    b, t, c, h, w = sequence.shape
    if t == t_max:
        return sequence
    if t > t_max:
        raise ValueError(f"t_max={t_max} must be >= current T={t}")

    device = sequence.device
    idx = ((torch.arange(t_max, device=device) + 1) * t - 1) // t_max  # zero-based ceil mapping
    return sequence.index_select(dim=1, index=idx)



def entropy_weighted_decode(step_logits: Tensor, gamma: float = 1.0, time_dim: int = 1, eps: float = 1e-8) -> Tensor:
    """Entropy-based decoding from the uploaded ICML paper.

    step_logits: [..., T, K] if time_dim=-2/1 or [T, B, K] if time_dim=0.
    Returns logits aggregated over the time dimension.
    """
    if step_logits.dim() < 2:
        raise ValueError("step_logits must have at least 2 dimensions")
    probs = step_logits.softmax(dim=-1)
    entropy = -(probs * (probs.clamp_min(eps).log())).sum(dim=-1)
    weights = torch.softmax(-float(gamma) * entropy, dim=time_dim)
    return (weights.unsqueeze(-1) * step_logits).sum(dim=time_dim)


class MeanTimeDecoder(nn.Module):
    def __init__(self, time_dim: int = 1) -> None:
        super().__init__()
        self.time_dim = int(time_dim)

    def forward(self, step_logits: Tensor) -> Tensor:
        return step_logits.mean(dim=self.time_dim)


class EntropyTimeDecoder(nn.Module):
    def __init__(self, gamma: float = 1.0, time_dim: int = 1) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.time_dim = int(time_dim)

    def forward(self, step_logits: Tensor) -> Tensor:
        return entropy_weighted_decode(step_logits, gamma=self.gamma, time_dim=self.time_dim)


class LearnableLineFragmentation(nn.Module):
    """Fixed-T learnable fragmentation from the ICML paper.

    Parameters are shared across all samples in the batch.
    """

    def __init__(
        self,
        image_size: Tuple[int, int],
        num_fragments: int,
        *,
        init_direction: str = "horizontal",
        hard_forward: bool = True,
        mask_scale: float = 1.0,
        init_noise: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if num_fragments < 2:
            raise ValueError("num_fragments must be >= 2")
        self.h, self.w = int(image_size[0]), int(image_size[1])
        self.num_fragments = int(num_fragments)
        self.num_lines = self.num_fragments - 1
        self.hard_forward = bool(hard_forward)
        self.mask_scale = float(mask_scale)
        self.eps = float(eps)
        self.init_direction = str(init_direction)

        a0, b0 = _direction_to_unit_normal(init_direction)
        self.h_raw = nn.Parameter(torch.full((self.num_lines,), float(a0)))
        self.v_raw = nn.Parameter(torch.full((self.num_lines,), float(b0)))
        self.r_raw = nn.Parameter(torch.zeros(self.num_lines))

        with torch.no_grad():
            r0 = _equal_width_offsets(
                num_fragments=self.num_fragments,
                h=self.h,
                w=self.w,
                direction=init_direction,
                device=self.r_raw.device,
                dtype=self.r_raw.dtype,
            )
            self.r_raw.copy_(r0)
            if init_noise > 0.0:
                self.h_raw.add_(init_noise * torch.randn_like(self.h_raw))
                self.v_raw.add_(init_noise * torch.randn_like(self.v_raw))

    def effective_line_params(self, device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, Tensor, Tensor, float]:
        h = self.h_raw.to(device=device, dtype=dtype)
        v = self.v_raw.to(device=device, dtype=dtype)
        r = self.r_raw.to(device=device, dtype=dtype)
        norm = torch.sqrt(h * h + v * v + self.eps)
        a = h / norm
        b = v / norm
        _, _, half_diag = _centered_coordinate_grid(self.h, self.w, device=device, dtype=dtype)
        c = float(half_diag) * torch.tanh(r)
        return a, b, c, float(half_diag)

    def _build_soft_and_hard_masks(self, device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
        xx, yy, _ = _centered_coordinate_grid(self.h, self.w, device=device, dtype=dtype)
        a, b, c, _ = self.effective_line_params(device=device, dtype=dtype)

        line_scores = a[:, None, None] * xx + b[:, None, None] * yy + c[:, None, None]
        soft_cuts = torch.sigmoid(-self.mask_scale * line_scores)
        hard_cuts = (soft_cuts > 0.5).to(dtype)

        if self.hard_forward:
            cuts_for_forward = _straight_through_binary(soft_cuts)
        else:
            cuts_for_forward = soft_cuts

        soft_masks: List[Tensor] = []
        hard_masks: List[Tensor] = []

        remaining_soft = torch.ones((self.h, self.w), device=device, dtype=dtype)
        remaining_hard = torch.ones((self.h, self.w), device=device, dtype=dtype)

        for idx in range(self.num_lines):
            soft_mask = cuts_for_forward[idx] * remaining_soft
            hard_mask = hard_cuts[idx] * remaining_hard
            soft_masks.append(soft_mask)
            hard_masks.append(hard_mask)
            remaining_soft = remaining_soft * (1.0 - cuts_for_forward[idx])
            remaining_hard = remaining_hard * (1.0 - hard_cuts[idx])

        soft_masks.append(remaining_soft)
        hard_masks.append(remaining_hard)
        return torch.stack(soft_masks, dim=0), torch.stack(hard_masks, dim=0)

    def _balance_loss(self, images: Tensor, soft_masks: Tensor) -> Tensor:
        # Paper: s_t = sum_{x,y} |I(x,y)| * m_t(x,y)
        energy_map = images.abs().sum(dim=1)  # [B, H, W]
        step_energy = (energy_map.unsqueeze(1) * soft_masks.unsqueeze(0)).sum(dim=(2, 3))  # [B, T]
        q = step_energy / step_energy.sum(dim=1, keepdim=True).clamp_min(self.eps)
        target = 1.0 / float(self.num_fragments)
        return ((q - target) ** 2).mean(dim=1).mean()

    def forward(self, images: Tensor) -> FragmentationOutput:
        if images.dim() != 4:
            raise ValueError(f"Expected [B, C, H, W], got {tuple(images.shape)}")
        b, c, h, w = images.shape
        if h != self.h or w != self.w:
            raise ValueError(f"Expected image_size=({self.h}, {self.w}), got ({h}, {w})")

        soft_masks, hard_masks = self._build_soft_and_hard_masks(device=images.device, dtype=images.dtype)
        sequence = images.unsqueeze(1) * soft_masks.unsqueeze(0).unsqueeze(2)
        balance = self._balance_loss(images, soft_masks)
        return FragmentationOutput(
            sequence=sequence,
            balance_loss=balance,
            soft_masks=soft_masks,
            hard_masks=hard_masks,
            selected_t=self.num_fragments,
        )


class DynamicLearnableFragmentation(nn.Module):
    """Dynamic fragment-number selection over a candidate set.

    This matches the ICML paper's formulation more closely than a shared-max-step implementation:
    each candidate T has its own learnable division-line parameters.
    """

    def __init__(
        self,
        image_size: Tuple[int, int],
        candidates: Sequence[int] = (2, 4, 8),
        *,
        init_direction: str = "horizontal",
        gumbel_tau: float = 1.0,
        selector_hard: bool = False,
        hard_forward: bool = True,
        mask_scale: float = 1.0,
        init_noise: float = 0.0,
    ) -> None:
        super().__init__()
        cand = tuple(sorted({int(t) for t in candidates}))
        if not cand:
            raise ValueError("candidates must be non-empty")
        if min(cand) < 2:
            raise ValueError("All candidate fragment counts must be >= 2")

        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.candidates = cand
        self.t_max = max(cand)
        self.gumbel_tau = float(gumbel_tau)
        self.selector_hard = bool(selector_hard)

        self.fragmenters = nn.ModuleDict(
            {
                str(t): LearnableLineFragmentation(
                    image_size=self.image_size,
                    num_fragments=t,
                    init_direction=init_direction,
                    hard_forward=hard_forward,
                    mask_scale=mask_scale,
                    init_noise=init_noise,
                )
                for t in self.candidates
            }
        )
        self.selector_logits = nn.Parameter(torch.zeros(len(self.candidates), dtype=torch.float32))

    def selector_probs(self, sample: bool) -> Tensor:
        if self.training and sample:
            return F.gumbel_softmax(self.selector_logits, tau=self.gumbel_tau, hard=self.selector_hard, dim=0)
        return self.selector_logits.softmax(dim=0)

    def selected_t(self) -> int:
        return int(self.candidates[int(self.selector_logits.argmax().item())])

    def forward(self, images: Tensor, *, mode: Optional[str] = None, sample_selector: bool = True) -> FragmentationOutput:
        if mode is None:
            mode = "mix" if self.training else "selected"
        mode = str(mode).strip().lower()

        outputs: Dict[int, FragmentationOutput] = {t: self.fragmenters[str(t)](images) for t in self.candidates}
        probs = self.selector_probs(sample=sample_selector)

        if mode == "mix":
            mixed_sequence = None
            balance = images.new_tensor(0.0)
            for idx, t in enumerate(self.candidates):
                aligned = resample_fragment_sequence(outputs[t].sequence, self.t_max)
                weighted = aligned * probs[idx].view(1, 1, 1, 1, 1)
                mixed_sequence = weighted if mixed_sequence is None else (mixed_sequence + weighted)
                balance = balance + probs[idx] * outputs[t].balance_loss

            selected_t = int(self.candidates[int(probs.argmax().item())])
            return FragmentationOutput(
                sequence=mixed_sequence,
                balance_loss=balance,
                selector_probs=probs,
                selected_t=selected_t,
            )

        if mode == "selected":
            selected_t = self.selected_t() if not self.training else int(self.candidates[int(probs.argmax().item())])
            out = outputs[selected_t]
            return FragmentationOutput(
                sequence=out.sequence,
                balance_loss=out.balance_loss,
                soft_masks=out.soft_masks,
                hard_masks=out.hard_masks,
                selector_probs=probs,
                selected_t=selected_t,
            )

        raise ValueError("mode must be 'mix' or 'selected'")
