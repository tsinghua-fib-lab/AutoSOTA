from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .fragmentation import (
    DynamicLearnableFragmenter,
    FixedLearnableFragmenter,
    FragmentationOutput,
    entropy_weighted_decode,
)
from .spikformer import Spikformer


class FragmentedSpikformer(nn.Module):
    """Thin convenience wrapper that plugs learnable fragmentation in front of Spikformer.

    Usage
    -----
    model = build_spikformer_preset(...)
    fragmenter = DynamicLearnableFragmenter(image_size=(32, 32), candidates=(2,4,8))
    wrapped = FragmentedSpikformer(model, fragmenter, decode='entropy', gamma=1.0)

    logits, aux = wrapped(images, return_aux=True)
    loss = F.cross_entropy(logits, labels) + 0.01 * aux['balance_loss']
    """

    def __init__(
        self,
        backbone: Spikformer,
        fragmenter: Optional[nn.Module] = None,
        *,
        decode: str = "entropy",
        gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.fragmenter = fragmenter
        self.decode = decode.lower().strip()
        self.gamma = float(gamma)

    def _decode(self, logits_seq: torch.Tensor) -> torch.Tensor:
        if self.decode == "mean":
            return logits_seq.mean(dim=0)
        if self.decode == "entropy":
            return entropy_weighted_decode(logits_seq, gamma=self.gamma)
        raise ValueError(f"Unknown decode mode: {self.decode!r}")

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_aux: bool = False,
        return_logits_seq: bool = False,
    ):
        if x.dim() != 4:
            raise ValueError(f"Expected x [B,C,H,W], got {tuple(x.shape)}")

        if self.fragmenter is None:
            logits_seq = self.backbone(x, decode=None, return_logits_seq=True)
            logits = self._decode(logits_seq)
            if return_logits_seq and return_aux:
                return logits, {"balance_loss": logits.new_tensor(0.0), "selected_steps": self.backbone.time_steps, "selector_probs": None}, logits_seq
            if return_aux:
                return logits, {"balance_loss": logits.new_tensor(0.0), "selected_steps": self.backbone.time_steps, "selector_probs": None}
            if return_logits_seq:
                return logits, logits_seq
            return logits

        frag_out = self.fragmenter(x)
        logits_seq = self.backbone.forward_sequence(frag_out.sequence, decode=None, return_logits_seq=True)
        logits = self._decode(logits_seq)

        aux = {
            "balance_loss": frag_out.balance_loss,
            "selected_steps": frag_out.selected_steps,
            "selector_probs": frag_out.selector_probs,
            "masks": frag_out.masks,
        }

        if return_logits_seq and return_aux:
            return logits, aux, logits_seq
        if return_aux:
            return logits, aux
        if return_logits_seq:
            return logits, logits_seq
        return logits


__all__ = ["FragmentedSpikformer"]
