"""Evaluation utilities: perplexity + KL divergence.

We compute:
  * Perplexity (PPL) on WikiText-2 test split
  * KL(P_unquantized || P_quantized) averaged per token
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class EvalResult:
    ppl: float
    nll: float
    kl: Optional[float] = None


@torch.no_grad()
def eval_ppl(
    model: torch.nn.Module,
    tokens_2d: torch.Tensor,
    *,
    max_batches: Optional[int] = None,
) -> Tuple[float, float]:
    """Return (ppl, avg_nll).

    Uses mean-of-means: average NLL per sequence, then average across sequences.
    This matches the reference implementation in matrix-quant.
    """
    model.eval()

    nlls = []
    nb = tokens_2d.shape[0] if max_batches is None else min(int(max_batches), int(tokens_2d.shape[0]))

    device = next(model.parameters()).device

    for i in range(nb):
        batch = tokens_2d[i : i + 1].to(device)
        logits = model(batch, start_pos=0).float()  # Cast to float32 for numerical stability
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        )
        # Average NLL per sequence, then collect
        nll = loss.view(shift_labels.size(0), -1).mean(dim=1)
        nlls.append(nll)

    nlls_tensor = torch.cat(nlls)
    ppl = torch.exp(nlls_tensor.mean())
    return ppl.item(), nlls_tensor.mean().item()


import math

_LOG2_E = math.log2(math.e)  # ≈ 1.4427, for converting nats to bits


@torch.no_grad()
def eval_kl(
    model_ref: torch.nn.Module,
    model_q: torch.nn.Module,
    tokens_2d: torch.Tensor,
    *,
    max_batches: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
) -> float:
    """Average token-level KL(P_ref || P_q) in bits."""
    model_ref.eval()
    model_q.eval()

    total_kl = 0.0
    total_tokens = 0

    nb = tokens_2d.shape[0] if max_batches is None else min(int(max_batches), int(tokens_2d.shape[0]))

    dev_ref = next(model_ref.parameters()).device
    dev_q = next(model_q.parameters()).device
    if dev_ref != dev_q:
        raise ValueError("For KL, both models must be on the same device (for now).")

    for i in range(nb):
        batch = tokens_2d[i : i + 1].to(dev_ref)

        logits_ref = model_ref(batch, start_pos=0)[:, :-1, :].to(dtype)
        logits_q = model_q(batch, start_pos=0)[:, :-1, :].to(dtype)

        logp_ref = F.log_softmax(logits_ref, dim=-1)
        logp_q = F.log_softmax(logits_q, dim=-1)

        p_ref = logp_ref.exp()
        kl_tok = (p_ref * (logp_ref - logp_q)).sum(dim=-1)  # (batch, seqlen-1), in nats

        total_kl += float(kl_tok.sum().item())
        total_tokens += int(kl_tok.numel())

    # Convert from nats to bits
    return (total_kl / max(total_tokens, 1)) * _LOG2_E
