from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FragmentationOutput:
    sequence: torch.Tensor                 # [T, B, C, H, W]
    balance_loss: torch.Tensor
    selector_probs: Optional[torch.Tensor] = None  # [M]
    selected_steps: Optional[int] = None
    masks: Optional[torch.Tensor] = None           # [T, H, W]


def _centered_grid(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    ys = torch.arange(height, device=device, dtype=dtype) - (height - 1) / 2.0
    xs = torch.arange(width, device=device, dtype=dtype) - (width - 1) / 2.0
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    half_diag = float(math.sqrt(((height - 1) / 2.0) ** 2 + ((width - 1) / 2.0) ** 2) + 1e-12)
    return xx, yy, half_diag


def _atanh_safe(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(-0.999999, 0.999999)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def _hard_st(binary_prob: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    hard = (binary_prob >= threshold).to(binary_prob.dtype)
    return hard - binary_prob.detach() + binary_prob


def _normalize_line_params(raw_h: torch.Tensor, raw_v: torch.Tensor, raw_r: torch.Tensor, diag: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    eps = 1e-6
    norm = torch.sqrt(raw_h.square() + raw_v.square() + eps)
    a = raw_h / norm
    b = raw_v / norm
    c = raw_h.new_tensor(float(diag)) * torch.tanh(raw_r)
    return a, b, c


def build_fragment_masks(
    *,
    height: int,
    width: int,
    raw_h: torch.Tensor,
    raw_v: torch.Tensor,
    raw_r: torch.Tensor,
    sharpness: float = 1.0,
    straight_through: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build sequential fragmentation masks.

    The implementation follows Eq. (3)-(5) of the uploaded learnable fragmentation
    paper but uses a temperature-like `sharpness` multiplier on the line score for
    better numerical control.

    Returns
    -------
    masks_use:
        Forward masks, optionally binary with straight-through gradients.
    masks_soft:
        Pure soft masks used for the balance regularizer.
    """
    if raw_h.numel() != raw_v.numel() or raw_h.numel() != raw_r.numel():
        raise ValueError("raw_h, raw_v and raw_r must have the same number of elements")

    device, dtype = raw_h.device, raw_h.dtype
    xx, yy, diag = _centered_grid(height, width, device=device, dtype=dtype)
    a, b, c = _normalize_line_params(raw_h, raw_v, raw_r, diag)

    k = raw_h.numel()
    t = k + 1

    line_scores = a.view(k, 1, 1) * xx + b.view(k, 1, 1) * yy + c.view(k, 1, 1)
    soft_gates = torch.sigmoid(-float(sharpness) * line_scores)
    gates_use = _hard_st(soft_gates) if straight_through else soft_gates

    masks_soft: List[torch.Tensor] = []
    masks_use: List[torch.Tensor] = []

    remaining_soft = torch.ones((height, width), device=device, dtype=dtype)
    remaining_use = torch.ones((height, width), device=device, dtype=dtype)

    for i in range(k):
        mask_soft = remaining_soft * soft_gates[i]
        mask_use = remaining_use * gates_use[i]
        masks_soft.append(mask_soft)
        masks_use.append(mask_use)
        remaining_soft = remaining_soft * (1.0 - soft_gates[i])
        remaining_use = remaining_use * (1.0 - gates_use[i])

    masks_soft.append(remaining_soft)
    masks_use.append(remaining_use)
    return torch.stack(masks_use, dim=0), torch.stack(masks_soft, dim=0)


def fragment_images(
    x: torch.Tensor,
    *,
    raw_h: torch.Tensor,
    raw_v: torch.Tensor,
    raw_r: torch.Tensor,
    sharpness: float = 1.0,
    straight_through: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fragment a static image batch [B,C,H,W] into [T,B,C,H,W]."""
    if x.dim() != 4:
        raise ValueError(f"Expected x with shape [B,C,H,W], got {tuple(x.shape)}")
    bsz, ch, height, width = x.shape
    masks_use, masks_soft = build_fragment_masks(
        height=height,
        width=width,
        raw_h=raw_h,
        raw_v=raw_v,
        raw_r=raw_r,
        sharpness=sharpness,
        straight_through=straight_through,
    )
    seq = x.unsqueeze(0) * masks_use.unsqueeze(1).unsqueeze(2)
    return seq, masks_use, masks_soft


def fragmentation_balance_loss(x: torch.Tensor, masks_soft: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Eq. (8) style balance loss from the learnable fragmentation paper.

    We use |I| summed over channels for color inputs.
    """
    if x.dim() != 4:
        raise ValueError(f"Expected x with shape [B,C,H,W], got {tuple(x.shape)}")
    if masks_soft.dim() != 3:
        raise ValueError(f"Expected masks_soft with shape [T,H,W], got {tuple(masks_soft.shape)}")

    bsz = x.size(0)
    t = masks_soft.size(0)
    pixel_energy = x.abs().sum(dim=1)  # [B,H,W]
    step_energy = (pixel_energy.unsqueeze(0) * masks_soft.unsqueeze(1)).sum(dim=(-1, -2)).transpose(0, 1)  # [B,T]
    q = step_energy / (step_energy.sum(dim=1, keepdim=True) + eps)
    target = 1.0 / float(t)
    return ((q - target) ** 2).mean()


def entropy_weighted_decode(logits_seq: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """Entropy-weighted temporal decoding from the uploaded fragmentation paper.

    Parameters
    ----------
    logits_seq:
        [T, B, K]
    gamma:
        Confidence sharpening coefficient.
    """
    if logits_seq.dim() != 3:
        raise ValueError(f"Expected logits_seq with shape [T,B,K], got {tuple(logits_seq.shape)}")
    probs = logits_seq.softmax(dim=-1)
    entropy = -(probs * (probs.clamp_min(1e-8).log())).sum(dim=-1)  # [T,B]
    weights = torch.softmax(-float(gamma) * entropy, dim=0)
    return (weights.unsqueeze(-1) * logits_seq).sum(dim=0)


class FixedLearnableFragmenter(nn.Module):
    """Learn T-1 line parameters and return a T-step fragment sequence.

    This is the clean, paper-faithful add-on version intended for direct use in
    Spikformer.
    """

    def __init__(
        self,
        *,
        image_size: Tuple[int, int],
        num_steps: int,
        sharpness: float = 1.0,
        straight_through: bool = True,
        init_mode: str = "horizontal_uniform",
    ) -> None:
        super().__init__()
        if num_steps < 2:
            raise ValueError("num_steps must be >= 2")
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.num_steps = int(num_steps)
        self.num_lines = self.num_steps - 1
        self.sharpness = float(sharpness)
        self.straight_through = bool(straight_through)
        self.init_mode = init_mode

        raw_h, raw_v, raw_r = self._make_init(self.num_steps, self.image_size)
        self.raw_h = nn.Parameter(raw_h)
        self.raw_v = nn.Parameter(raw_v)
        self.raw_r = nn.Parameter(raw_r)

    @staticmethod
    def _make_init(num_steps: int, image_size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h, w = image_size
        _, yy, diag = _centered_grid(h, w, device=torch.device("cpu"), dtype=torch.float32)
        # Horizontal lines: y + c = 0. Use equally spaced offsets.
        y_min = float(yy.min().item())
        y_max = float(yy.max().item())
        thresholds = torch.linspace(y_min, y_max, steps=num_steps + 1, dtype=torch.float32)[1:-1]
        raw_h = torch.zeros(num_steps - 1, dtype=torch.float32)
        raw_v = torch.ones(num_steps - 1, dtype=torch.float32)
        raw_r = _atanh_safe((-thresholds / float(diag)).to(torch.float32))
        return raw_h, raw_v, raw_r

    def forward(self, x: torch.Tensor) -> FragmentationOutput:
        seq, masks, masks_soft = fragment_images(
            x,
            raw_h=self.raw_h,
            raw_v=self.raw_v,
            raw_r=self.raw_r,
            sharpness=self.sharpness,
            straight_through=self.straight_through if self.training else False,
        )
        balance = fragmentation_balance_loss(x, masks_soft)
        return FragmentationOutput(
            sequence=seq,
            balance_loss=balance,
            selector_probs=None,
            selected_steps=self.num_steps,
            masks=masks_soft,
        )


class DynamicLearnableFragmenter(nn.Module):
    """Learn both division lines and the number of fragments.

    Training mode returns the paper's *mixed sequence* aligned to Tmax. Eval mode
    returns the sequence for the currently selected T = argmax p(T).
    """

    def __init__(
        self,
        *,
        image_size: Tuple[int, int],
        candidates: Sequence[int] = (2, 4, 8),
        gumbel_tau: float = 1.0,
        sharpness: float = 1.0,
        straight_through: bool = True,
        selector_init: Optional[int] = None,
    ) -> None:
        super().__init__()
        cand = tuple(sorted({int(v) for v in candidates}))
        if not cand or cand[0] < 2:
            raise ValueError("All candidates must be >= 2")
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.candidates = cand
        self.max_steps = max(cand)
        self.gumbel_tau = float(gumbel_tau)
        self.sharpness = float(sharpness)
        self.straight_through = bool(straight_through)

        self.raw_h = nn.ParameterDict()
        self.raw_v = nn.ParameterDict()
        self.raw_r = nn.ParameterDict()
        for t in self.candidates:
            h0, v0, r0 = FixedLearnableFragmenter._make_init(t, self.image_size)
            self.raw_h[str(t)] = nn.Parameter(h0)
            self.raw_v[str(t)] = nn.Parameter(v0)
            self.raw_r[str(t)] = nn.Parameter(r0)

        self.selector_logits = nn.Parameter(torch.zeros(len(self.candidates), dtype=torch.float32))
        if selector_init is None:
            selector_init = self.candidates[len(self.candidates) // 2]
        if selector_init not in self.candidates:
            raise ValueError(f"selector_init={selector_init} must be in {self.candidates}")
        with torch.no_grad():
            self.selector_logits[self.candidates.index(selector_init)] = 2.0

    def selected_steps(self) -> int:
        idx = int(torch.argmax(self.selector_logits).item())
        return int(self.candidates[idx])

    def selector_probs(self) -> torch.Tensor:
        return torch.softmax(self.selector_logits, dim=0)

    def _sample_selector(self) -> torch.Tensor:
        if self.training:
            return F.gumbel_softmax(self.selector_logits, tau=self.gumbel_tau, hard=False, dim=0)
        return torch.softmax(self.selector_logits, dim=0)

    @staticmethod
    def _upsample_time(seq: torch.Tensor, target_steps: int) -> torch.Tensor:
        if seq.dim() != 5:
            raise ValueError(f"Expected seq [T,B,C,H,W], got {tuple(seq.shape)}")
        t = seq.size(0)
        if t == target_steps:
            return seq
        idx = (torch.arange(target_steps, device=seq.device) * t) // target_steps
        return seq.index_select(0, idx)

    def _fragment_candidate(self, x: torch.Tensor, steps: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return fragment_images(
            x,
            raw_h=self.raw_h[str(steps)],
            raw_v=self.raw_v[str(steps)],
            raw_r=self.raw_r[str(steps)],
            sharpness=self.sharpness,
            straight_through=self.straight_through if self.training else False,
        )

    def forward(self, x: torch.Tensor) -> FragmentationOutput:
        probs = self._sample_selector()

        seq_by_t: Dict[int, torch.Tensor] = {}
        masks_by_t: Dict[int, torch.Tensor] = {}
        balance_by_t: Dict[int, torch.Tensor] = {}
        for t in self.candidates:
            seq_t, _masks, masks_soft_t = self._fragment_candidate(x, t)
            seq_by_t[t] = seq_t
            masks_by_t[t] = masks_soft_t
            balance_by_t[t] = fragmentation_balance_loss(x, masks_soft_t)

        # Training: use mixed fragments aligned to Tmax. Eval: use selected T.
        if self.training:
            mixed = None
            weighted_balance = x.new_tensor(0.0)
            for idx, t in enumerate(self.candidates):
                seq_up = self._upsample_time(seq_by_t[t], self.max_steps)
                term = probs[idx].to(dtype=seq_up.dtype) * seq_up
                mixed = term if mixed is None else mixed + term
                weighted_balance = weighted_balance + probs[idx].to(dtype=weighted_balance.dtype) * balance_by_t[t]

            selected = int(self.candidates[int(torch.argmax(probs).item())])
            return FragmentationOutput(
                sequence=mixed,
                balance_loss=weighted_balance,
                selector_probs=probs,
                selected_steps=selected,
                masks=None,
            )

        selected = self.selected_steps()
        return FragmentationOutput(
            sequence=seq_by_t[selected],
            balance_loss=balance_by_t[selected],
            selector_probs=torch.softmax(self.selector_logits, dim=0),
            selected_steps=selected,
            masks=masks_by_t[selected],
        )


__all__ = [
    "FragmentationOutput",
    "build_fragment_masks",
    "fragment_images",
    "fragmentation_balance_loss",
    "entropy_weighted_decode",
    "FixedLearnableFragmenter",
    "DynamicLearnableFragmenter",
]
