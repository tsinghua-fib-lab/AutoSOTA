"""Cached-confidence remasking under the fixed LLaDA unmasking schedule.

This is the ReMDM-conf plug-in described in CoRe Appendix B.1, rather than
the ReMDM authors' separate 2T-step categorical LLaDA sampler. The confidence
weights follow ReMDM Section 4.1: softmax(-last_unmask_probability).
"""

import torch
import torch.nn.functional as F

from .config import SamplerConfig
from .sampling import _fill_scores, get_num_transfer_tokens


def confidence_weights(cache, positions):
    """Normalize the published confidence weights over eligible positions."""
    return torch.softmax(-cache[positions].float(), dim=0)


@torch.no_grad()
def generate_remdm_conf(
    model,
    prompt,
    steps=128,
    gen_length=512,
    block_length=512,
    temperature=0.0,
    mask_id=126336,
    config=None,
    tokenizer=None,
    trace_callback=None,
):
    """Sample one cached-confidence revision every E steps in the middle half.

    Token predictions stay greedy, with margin-based ordinary transfer as in
    the compared samplers. Remasking is stochastic. Each revised token adds
    one transfer slot; a confidence cache changes only when a token is decoded.
    Prompt tokens and the last protected_suffix positions are never revised.
    The one forward per step includes the remasked state, so cost is T forwards.
    """
    config = config or SamplerConfig(revise_every=8)
    if prompt.shape[0] != 1 or temperature != 0:
        raise ValueError("This comparison uses batch size one and greedy token predictions")
    if steps < 1 or gen_length < 1 or block_length != gen_length:
        raise ValueError("The frozen comparison uses one positive full-length block")
    if gen_length < steps:
        raise ValueError("Each ordinary step must have a positive transfer allocation")
    start, end = prompt.shape[1], prompt.shape[1] + gen_length
    x = torch.full((1, end), mask_id, dtype=torch.long, device=model.device)
    x[:, :start] = prompt
    cache = torch.full_like(x, torch.inf, dtype=torch.float32)
    quota = get_num_transfer_tokens(x[:, start:] == mask_id, steps)
    stats = dict(revision_events=0, remasked_tokens=0, changed_tokens=0)
    for step in range(steps):
        revisions = torch.empty(0, dtype=torch.long, device=x.device)
        old = torch.empty(0, dtype=torch.long, device=x.device)
        if (
            config.revise_every > 0
            and step % config.revise_every == 0
            and int(0.25 * steps) <= step < int(0.75 * steps)
        ):
            eligible = (x[0] != mask_id) & torch.isfinite(cache[0])
            eligible[:start] = False
            eligible[max(start, end - config.protected_suffix) :] = False
            positions = torch.nonzero(eligible, as_tuple=False).flatten()
            if positions.numel():
                index = torch.multinomial(confidence_weights(cache[0], positions), 1)
                revisions = positions[index]
                old = x[0, revisions].clone()
                x[0, revisions] = mask_id
                cache[0, revisions] = torch.inf
                stats["revision_events"] += 1
                stats["remasked_tokens"] += 1
        logits = model(x).logits
        proposal = logits.argmax(dim=-1)
        probabilities = F.softmax(logits, dim=-1)
        scores, confidence = _fill_scores(probabilities, proposal, config.base_masking)
        mask = x == mask_id
        scores = scores[0].masked_fill(~mask[0], -torch.inf)
        count = min(int(quota[0, step]) + len(revisions), int(mask.sum()))
        transfer = torch.topk(scores, count).indices
        x[0, transfer] = proposal[0, transfer]
        cache[0, transfer] = confidence[0, transfer].float()
        if len(revisions):
            immediate = torch.isin(revisions, transfer)
            stats["changed_tokens"] += int((x[0, revisions[immediate]] != old[immediate]).sum())
            if trace_callback is not None:
                trace_callback(dict(step=step, positions=(revisions - start).tolist()))
    if (x[:, start:] == mask_id).any():
        raise RuntimeError("Compensated transfer did not finish decoding")
    return x, stats, {}
