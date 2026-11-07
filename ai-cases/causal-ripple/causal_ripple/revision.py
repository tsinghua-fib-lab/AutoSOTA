"""Paired masked-context probes and bounded joint token selection."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import SamplerConfig


@dataclass
class RippleSelection:
    repair_mask: torch.Tensor
    proposal: torch.Tensor
    proposal_confidence: torch.Tensor
    influence: torch.Tensor
    waves_used: int
    model_calls: int
    accepted_score: torch.Tensor
    root_position: int


def _empty_selection(x, seed_score, seed_proposal, candidate_mask):
    options = dict(device=x.device, dtype=torch.float32)
    return RippleSelection(
        torch.zeros_like(candidate_mask),
        seed_proposal.clone(),
        torch.full(seed_score.shape, float("nan"), **options),
        torch.full(seed_score.shape, float("nan"), **options),
        0,
        0,
        torch.full(seed_score.shape, float("nan"), **options),
        -1,
    )


@torch.no_grad()
def select_revision(
    model,
    x,
    seed_score,
    seed_proposal,
    candidate_mask,
    margin,
    mask_id,
    config: SamplerConfig,
    attention_mask=None,
):
    """Select a seed, then probe the same masked positions in two contexts.

    The original token is the reference event in both likelihoods. A support
    drop or an argmax change makes a probe eligible; the profile controls the
    additional confidence gate and the transfer truncation used by the caller.
    """
    if x.shape[0] != 1:
        raise ValueError("Paired revision requires a single generation trajectory")
    result = _empty_selection(x, seed_score, seed_proposal, candidate_mask)
    positions = torch.nonzero(
        torch.isfinite(seed_score[0]) & candidate_mask[0],
        as_tuple=False,
    ).flatten()
    if not positions.numel():
        return result
    seed = positions[seed_score[0, positions].float().argmax()]
    result.root_position = int(seed)
    result.repair_mask[0, seed] = True
    before = x[0].clone()
    after = before.clone()
    after[seed] = seed_proposal[0, seed]
    frontier = seed.view(1)

    for _ in range(config.waves):
        allowance = config.max_k - int(result.repair_mask.sum())
        if allowance <= 0:
            break
        probes = torch.nonzero(
            candidate_mask[0] & ~result.repair_mask[0],
            as_tuple=False,
        ).flatten()
        if not probes.numel():
            break
        distance = (probes[:, None] - frontier[None, :]).abs().amin(dim=1).float()
        priority = -margin[0, probes].float() - config.distance_weight * distance
        probes = probes[torch.topk(priority, k=min(config.probe_m, len(probes))).indices]
        old_state, new_state = before.clone(), after.clone()
        old_state[probes] = mask_id
        new_state[probes] = mask_id
        states = torch.stack([old_state, new_state])
        attn = attention_mask.expand(2, -1) if attention_mask is not None else None
        probabilities = F.softmax(model(states, attention_mask=attn).logits.float(), dim=-1)
        result.model_calls += 1
        old_p, new_p = probabilities[0, probes], probabilities[1, probes]
        original = x[0, probes]
        old_support = old_p.gather(-1, original[:, None]).squeeze(-1)
        new_support = new_p.gather(-1, original[:, None]).squeeze(-1)
        influence = (torch.log(old_support + 1e-10) - torch.log(new_support + 1e-10)).clamp_min(0)
        new_token = new_p.argmax(-1)
        confidence = new_p.amax(-1)
        flipped = old_p.argmax(-1) != new_token
        eligible = flipped | (influence >= config.influence_threshold)
        eligible &= confidence >= config.min_probe_confidence
        if config.changed_probe_only:
            eligible &= new_token != original
        result.influence[0, probes] = influence.float()
        result.accepted_score[0, probes] = influence.float()
        rows = torch.nonzero(eligible, as_tuple=False).flatten()
        if not rows.numel():
            break
        chosen_rows = rows[torch.topk(influence[rows], k=min(allowance, len(rows))).indices]
        chosen = probes[chosen_rows]
        result.repair_mask[0, chosen] = True
        result.proposal[0, chosen] = new_token[chosen_rows]
        result.proposal_confidence[0, chosen] = confidence[chosen_rows].float()
        before = after.clone()
        after[chosen] = new_token[chosen_rows]
        frontier = chosen
        result.waves_used += 1
    return result
