"""Graph-based proposals used by the Visible-Cascade SafeGraph expert."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .revision import RippleSelection


def _empty_selection(x, seed_score, seed_proposal, candidate_mask):
    float_shape = dict(device=x.device, dtype=torch.float32)
    return RippleSelection(
        repair_mask=torch.zeros_like(candidate_mask),
        proposal=seed_proposal.clone(),
        proposal_confidence=torch.full(seed_score.shape, float("nan"), **float_shape),
        influence=torch.full(seed_score.shape, float("nan"), **float_shape),
        waves_used=0,
        model_calls=0,
        accepted_score=torch.full(seed_score.shape, float("nan"), **float_shape),
        root_position=-1,
    )


def _batched_probabilities(model, states, attention_mask):
    states = torch.stack(states)
    attn = attention_mask.expand(states.shape[0], -1) if attention_mask is not None else None
    return F.softmax(model(states, attention_mask=attn).logits.float(), dim=-1)


def _distribution_features(old_p, new_p, original):
    old_support = old_p.gather(-1, original[:, None]).squeeze(-1)
    new_support = new_p.gather(-1, original[:, None]).squeeze(-1)
    support_drop = (torch.log(old_support + 1e-10) - torch.log(new_support + 1e-10)).clamp_min(0)
    midpoint = 0.5 * (old_p + new_p)
    jsd = 0.5 * (
        (old_p * (torch.log(old_p + 1e-10) - torch.log(midpoint + 1e-10))).sum(-1)
        + (new_p * (torch.log(new_p + 1e-10) - torch.log(midpoint + 1e-10))).sum(-1)
    )
    entropy_old = -(old_p * torch.log(old_p + 1e-10)).sum(-1)
    entropy_new = -(new_p * torch.log(new_p + 1e-10)).sum(-1)
    entropy_drop = entropy_old - entropy_new
    old_argmax = old_p.argmax(-1)
    new_argmax = new_p.argmax(-1)
    flipped = old_argmax != new_argmax
    new_confidence = new_p.amax(-1)
    return (support_drop, jsd, entropy_drop, flipped, new_confidence, new_argmax)


@torch.no_grad()
def grow_graph_repair(
    model,
    x: torch.Tensor,
    seed_score: torch.Tensor,
    seed_proposal: torch.Tensor,
    candidate_mask: torch.Tensor,
    margin: torch.Tensor,
    mask_id: int,
    attention_mask=None,
    roots: int = 2,
    probe_m: int = 12,
    max_k: int = 4,
    threshold: float = 0.1,
    centrality_weight: float = 1.0,
) -> RippleSelection:
    """Select a repair set from a batched multi-root influence graph."""
    if x.shape[0] != 1:
        raise ValueError("graph selection currently expects batch size 1")
    result = _empty_selection(x, seed_score, seed_proposal, candidate_mask)
    finite = torch.isfinite(seed_score[0]) & candidate_mask[0]
    candidates = torch.nonzero(finite, as_tuple=False).flatten()
    if candidates.numel() == 0 or max_k <= 0:
        return result
    root_count = min(max(1, roots), int(candidates.numel()))
    root_score = seed_score[0, candidates].float()
    roots_pos = candidates[torch.topk(root_score, k=root_count).indices]
    available = candidate_mask[0].clone()
    available[roots_pos] = False
    probes = torch.nonzero(available, as_tuple=False).flatten()
    if probes.numel() == 0:
        result.repair_mask[0, roots_pos[0]] = True
        return result
    probe_m_now = min(max(1, probe_m), int(probes.numel()))
    distance = (probes[:, None] - roots_pos[None, :]).abs().amin(dim=1).float()
    priority = -margin[0, probes].float() - 0.01 * distance
    probes = probes[torch.topk(priority, k=probe_m_now).indices]
    states = []
    for root in roots_pos:
        old_state = x[0].clone()
        new_state = old_state.clone()
        new_state[root] = seed_proposal[0, root]
        old_state[probes] = mask_id
        new_state[probes] = mask_id
        states.extend([old_state, new_state])
    probabilities = _batched_probabilities(model, states, attention_mask)
    result.model_calls = 1
    original = x[0, probes]
    edge_scores, new_tokens, new_confidences = ([], [], [])
    for root_index in range(root_count):
        old_p = probabilities[2 * root_index, probes]
        new_p = probabilities[2 * root_index + 1, probes]
        support, jsd, _, _, confidence, token = _distribution_features(old_p, new_p, original)
        edge_scores.append(support + jsd)
        new_tokens.append(token)
        new_confidences.append(confidence)
    edge_scores = torch.stack(edge_scores)
    centrality = edge_scores.topk(min(3, edge_scores.shape[1]), dim=1).values.mean(1)
    root_values = seed_score[0, roots_pos].float() + centrality_weight * centrality
    winner_row = int(root_values.argmax().item())
    winner = roots_pos[winner_row]
    result.root_position = int(winner.item())
    result.repair_mask[0, winner] = True
    node_score = edge_scores[winner_row] + 0.5 * edge_scores.mean(0)
    result.influence[0, probes] = edge_scores[winner_row].float()
    result.accepted_score[0, probes] = node_score.float()
    eligible = torch.nonzero(node_score >= threshold, as_tuple=False).flatten()
    if eligible.numel() and max_k > 1:
        take = min(max_k - 1, int(eligible.numel()))
        rows = eligible[torch.topk(node_score[eligible], k=take).indices]
        chosen = probes[rows]
        result.repair_mask[0, chosen] = True
        result.proposal[0, chosen] = new_tokens[winner_row][rows]
        result.proposal_confidence[0, chosen] = new_confidences[winner_row][rows].float()
        result.waves_used = 1
    return result
