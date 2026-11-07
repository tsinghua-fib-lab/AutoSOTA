"""Margin-scheduled denoising with CORE or paired Causal-Ripple revision.

The filling and verification operations follow the evaluated CORE-based
implementation. Probe selection and transfer truncation are explicit settings.
"""

import torch
import torch.nn.functional as F

from .config import SamplerConfig
from .revision import select_revision

METHODS = {
    "low_confidence",
    "random",
    "topk_margin",
    "core",
    "margin_remask",
    "ripple",
    "safe_graph",
    "structural_core",
}


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    return logits.exp() / ((-torch.log(noise)) ** temperature)


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base, remainder = mask_num // steps, mask_num % steps
    quota = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for batch in range(mask_num.size(0)):
        quota[batch, : remainder[batch]] += 1
    return quota


def _fill_scores(probabilities, proposal, masking):
    confidence = probabilities.gather(-1, proposal.unsqueeze(-1)).squeeze(-1)
    if masking == "margin":
        top2 = probabilities.topk(2, dim=-1).values
        return top2[..., 0] - top2[..., 1], confidence
    return confidence, confidence


@torch.no_grad()
def generate(
    model,
    prompt,
    attention_mask=None,
    steps=128,
    gen_length=512,
    block_length=512,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="ripple",
    mask_id=126336,
    logits_eos_inf=False,
    confidence_eos_eot_inf=False,
    remask_k=1,
    tokenizer=None,
    config=None,
    trace_callback=None,
):
    """Continue a prompt and return ``(token_ids, statistics, diagnostics)``.

    ``config`` selects the published benchmark profile. A ``trace_callback``
    receives token-level revision events without changing the sampler state.
    """
    config = config or SamplerConfig()
    if remasking not in METHODS:
        raise ValueError(f"Unknown sampler: {remasking}")
    if remasking in {"ripple", "safe_graph"} and prompt.shape[0] != 1:
        raise ValueError("Paired revision requires batch size one")
    if cfg_scale != 0:
        raise ValueError("The published configurations use cfg_scale=0")
    if steps < 1 or gen_length < 1 or block_length < 1 or temperature < 0:
        raise ValueError("Lengths and steps must be positive, and temperature nonnegative")
    if gen_length % block_length or steps % (gen_length // block_length):
        raise ValueError("Block length and denoising steps must divide the requested trajectory")
    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length),
        mask_id,
        dtype=torch.long,
        device=model.device,
    )
    x[:, : prompt.shape[1]] = prompt.clone()
    if attention_mask is not None:
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (prompt.shape[0], gen_length),
                    dtype=attention_mask.dtype,
                    device=model.device,
                ),
            ],
            dim=-1,
        )
    prompt_index = x != mask_id
    num_blocks = gen_length // block_length
    block_steps = steps // num_blocks
    stats = dict(
        core_calls=0,
        verified_tokens=0,
        remasked_tokens=0,
        changed_tokens=0,
        core_sum_verified=0.0,
        core_sum_remasked=0.0,
        ripple_calls=0,
        ripple_waves=0,
        guard_rejected=0,
        transaction_rollbacks=0,
    )
    transaction_original = torch.zeros_like(x)
    transaction_touched = torch.zeros_like(x, dtype=torch.bool)

    def forward(state):
        logits = model(state, attention_mask=attention_mask).logits
        if logits_eos_inf or confidence_eos_eot_inf:
            logits[:, :, 126081] = -torch.inf
        if confidence_eos_eot_inf:
            logits[:, :, 126348] = -torch.inf
        return logits

    for block in range(num_blocks):
        block_start = prompt.shape[1] + block * block_length
        block_end = block_start + block_length
        quota = get_num_transfer_tokens(x[:, block_start:block_end] == mask_id, block_steps)
        for step in range(block_steps):
            mask_index = x == mask_id
            repair_mask = torch.zeros_like(x, dtype=torch.bool)
            repair_counts = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            logits = forward(x)
            proposal = add_gumbel_noise(logits, temperature).argmax(dim=-1)
            if remasking == "random":
                fill_score = torch.rand(proposal.shape, device=x.device)
                proposal_confidence = torch.full_like(fill_score, float("nan"))
            else:
                probabilities = F.softmax(logits, dim=-1)
                masking = (
                    "confidence"
                    if remasking == "low_confidence"
                    else "margin"
                    if remasking == "topk_margin"
                    else config.base_masking
                )
                fill_score, proposal_confidence = _fill_scores(probabilities, proposal, masking)

            verify_now = (
                remasking in {"core", "margin_remask", "ripple", "safe_graph", "structural_core"}
                and config.revise_every > 0
                and step % config.revise_every == 0
                and int(0.25 * block_steps) <= step < int(0.75 * block_steps)
            )
            if verify_now:
                stats["core_calls"] += 1
                top2 = probabilities.topk(2, dim=-1).values
                margin = top2[..., 0] - top2[..., 1]
                in_block = torch.zeros_like(x, dtype=torch.bool)
                in_block[:, block_start:block_end] = True
                protected = torch.zeros_like(x, dtype=torch.bool)
                protected[:, max(block_start, block_end - config.protected_suffix) : block_end] = (
                    True
                )
                candidates = ~mask_index & in_block & ~prompt_index & ~protected
                verify_mask = torch.zeros_like(x, dtype=torch.bool)
                for batch in range(x.shape[0]):
                    positions = torch.nonzero(candidates[batch], as_tuple=False).flatten()
                    if positions.numel():
                        chosen = torch.topk(
                            -margin[batch, positions], k=min(config.candidate_m, len(positions))
                        ).indices
                        verify_mask[batch, positions[chosen]] = True
                if verify_mask.any():
                    original = x.clone()
                    masked = x.clone()
                    masked[verify_mask] = mask_id
                    verified_logits = forward(masked)
                    verified_p = F.softmax(verified_logits, dim=-1)
                    verified_proposal = verified_logits.argmax(dim=-1)
                    verified_fill, verified_confidence = _fill_scores(
                        verified_p, verified_proposal, config.base_masking
                    )
                    likelihood = torch.log(
                        verified_p.gather(-1, original.unsqueeze(-1)).squeeze(-1) + 1e-10
                    )
                    seed_score = torch.where(
                        verify_mask
                        & (verified_proposal != original)
                        & (verified_confidence >= 0.30),
                        -likelihood,
                        torch.full_like(likelihood, -float("inf")),
                    )
                    if remasking == "margin_remask":
                        top2_verified = verified_p.topk(2, dim=-1).values
                        verified_margin = top2_verified[..., 0] - top2_verified[..., 1]
                        seed_score = torch.where(
                            verify_mask,
                            -verified_margin,
                            torch.full_like(verified_margin, -float("inf")),
                        )
                    stats["verified_tokens"] += int(verify_mask.sum())
                    stats["core_sum_verified"] += float(likelihood[verify_mask].sum())
                    selection = None
                    if remasking == "ripple":
                        selection = select_revision(
                            model,
                            x,
                            seed_score,
                            verified_proposal,
                            verify_mask,
                            margin,
                            mask_id,
                            config,
                            attention_mask,
                        )
                    elif remasking == "safe_graph":
                        from .graph import grow_graph_repair
                        from .guards import filter_local_safe_selection

                        selection = grow_graph_repair(
                            model,
                            x,
                            seed_score,
                            verified_proposal,
                            verify_mask,
                            margin,
                            mask_id,
                            attention_mask=attention_mask,
                            probe_m=config.probe_m,
                            max_k=config.max_k,
                        )
                        safe = filter_local_safe_selection(
                            x, selection.repair_mask, selection.proposal, tokenizer
                        )
                        stats["guard_rejected"] += int((selection.repair_mask & ~safe).sum())
                        selection.repair_mask = safe
                    if selection is not None:
                        stats["ripple_calls"] += selection.model_calls
                        stats["ripple_waves"] += selection.waves_used
                        repair_mask = selection.repair_mask
                        if remasking == "safe_graph" and repair_mask.any():
                            newly_touched = repair_mask & ~transaction_touched
                            transaction_original[newly_touched] = x[newly_touched]
                            transaction_touched |= repair_mask
                        allowance = int(quota[0, step])
                        selected = torch.nonzero(repair_mask[0], as_tuple=False).flatten()
                        if selected.numel() and selection.root_position not in selected.tolist():
                            selection.root_position = int(selected[0])
                        if len(selected) > allowance:
                            if config.ranked_transfer:
                                root = torch.tensor(
                                    selection.root_position,
                                    device=selected.device,
                                    dtype=selected.dtype,
                                )
                                others = selected[selected != root]
                                scores = selection.accepted_score[0, others].nan_to_num(
                                    nan=-float("inf")
                                )
                                others = others[torch.argsort(scores, descending=True, stable=True)]
                                keep = torch.cat([root.view(1), others[: max(0, allowance - 1)]])[
                                    :allowance
                                ]
                                stats["budget_ranked_trim_events"] = (
                                    stats.get("budget_ranked_trim_events", 0) + 1
                                )
                            else:
                                root = torch.tensor(
                                    selection.root_position,
                                    device=selected.device,
                                    dtype=selected.dtype,
                                )
                                others = selected[selected != root]
                                keep = torch.cat([root.view(1), others[: max(0, allowance - 1)]])[
                                    :allowance
                                ]
                            repair_mask.zero_()
                            repair_mask[0, keep] = True
                        repair_counts = repair_mask.sum(dim=1).long()
                        verified_proposal = torch.where(
                            repair_mask, selection.proposal, verified_proposal
                        )
                        verified_confidence = torch.where(
                            repair_mask & torch.isfinite(selection.proposal_confidence),
                            selection.proposal_confidence,
                            verified_confidence,
                        )
                    else:
                        for batch in range(x.shape[0]):
                            allowance = min(remask_k, int(quota[batch, step]))
                            finite = torch.isfinite(seed_score[batch])
                            count = min(allowance, int(finite.sum()))
                            if count > 0:
                                positions = torch.topk(
                                    seed_score[batch].masked_fill(~finite, -float("inf")), k=count
                                ).indices
                                repair_mask[batch, positions] = True
                                repair_counts[batch] = count
                    stats["remasked_tokens"] += int(repair_mask.sum())
                    if repair_mask.any():
                        stats["core_sum_remasked"] += float(likelihood[repair_mask].sum())
                        stats["changed_tokens"] += int(
                            (verified_proposal[repair_mask] != original[repair_mask]).sum()
                        )
                    if trace_callback is not None:
                        positions = torch.nonzero(repair_mask[0], as_tuple=False).flatten()
                        trace_callback(
                            dict(
                                step=block * block_steps + step,
                                prompt_length=prompt.shape[1],
                                seed=None
                                if selection is None or selection.root_position < 0
                                else selection.root_position - prompt.shape[1],
                                positions=(positions - prompt.shape[1]).tolist(),
                                old_tokens=original[0, positions].tolist(),
                                new_tokens=verified_proposal[0, positions].tolist(),
                                influence=[]
                                if selection is None
                                else selection.influence[0, positions].nan_to_num().tolist(),
                            )
                        )
                    proposal = torch.where(repair_mask, verified_proposal, proposal)
                    fill_score = torch.where(repair_mask, verified_fill, fill_score)
                    proposal_confidence = torch.where(
                        repair_mask, verified_confidence, proposal_confidence
                    )
                    x[repair_mask] = mask_id
                    mask_index = x == mask_id

            fill_score[:, block_end:] = -float("inf")
            proposal = torch.where(mask_index, proposal, x)
            confidence = torch.where(mask_index, fill_score, -float("inf"))
            transfer = torch.zeros_like(proposal, dtype=torch.bool)
            for batch in range(x.shape[0]):
                ordinary = int(quota[batch, step])
                if ordinary <= 0:
                    continue
                allowance = ordinary + int(repair_counts[batch])
                forced = repair_mask[batch] & torch.isfinite(proposal_confidence[batch])
                transfer[batch, forced] = True
                remaining = allowance - int(forced.sum())
                if remaining <= 0:
                    continue
                scores = confidence[batch].clone()
                scores[forced] = -float("inf")
                remaining = min(remaining, int(torch.isfinite(scores).sum()))
                if remaining > 0:
                    transfer[batch, torch.topk(scores, k=remaining).indices] = True
            x[transfer] = proposal[transfer]

    generated_start = prompt.shape[1]
    if remasking == "structural_core" and tokenizer is not None:
        from .structure import expand_structural_mask, syntax_error_seed

        for _ in range(4):
            any_error = False
            for batch in range(x.shape[0]):
                position, _ = syntax_error_seed(
                    x[batch, generated_start:].tolist(), tokenizer, offset=generated_start
                )
                if position is None:
                    continue
                any_error = True
                seed_mask = torch.zeros_like(x, dtype=torch.bool)
                seed_mask[
                    batch, max(generated_start, position - 1) : min(x.shape[1], position + 2)
                ] = True
                valid = torch.zeros_like(seed_mask)
                valid[:, generated_start:] = True
                selected = expand_structural_mask(
                    x, seed_mask, tokenizer, valid_mask=valid, max_total=4
                )
                masked = x.clone()
                masked[selected] = mask_id
                proposal = model(masked, attention_mask=attention_mask).logits.argmax(-1)
                x[selected] = proposal[selected]
                stats["posthoc_calls"] = stats.get("posthoc_calls", 0) + 1
                stats["posthoc_tokens"] = stats.get("posthoc_tokens", 0) + int(selected.sum())
            if not any_error:
                break
    if (x[:, generated_start:] == mask_id).any():
        raise RuntimeError("The completed response contains unfilled masks")
    if remasking == "safe_graph" and transaction_touched.any():
        from .guards import transactional_finalize

        x, rolled_back = transactional_finalize(
            x,
            transaction_original,
            transaction_touched,
            prompt_length=generated_start,
            tokenizer=tokenizer,
        )
        stats["transaction_rollbacks"] += rolled_back
    return x, stats, {}
