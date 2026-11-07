"""Visible-Cascade: local reconstruction followed by static-defect routing."""

import torch

from .config import SamplerConfig
from .guards import python_certificate
from .refinement import (
    hard_deficient,
    rediffuse_positions,
    select_by_certificate,
    unit_suspicion,
    visible_test_score,
)
from .sampling import generate


def _merge_statistics(target, source):
    for key, value in source.items():
        if isinstance(value, (int, float)):
            target[key] = target.get(key, 0) + value


@torch.no_grad()
def visible_cascade(
    model,
    prompt,
    tokenizer,
    config=None,
    steps=128,
    gen_length=512,
    block_length=512,
    temperature=0.0,
    mask_id=126336,
    attention_mask=None,
    base=None,
    certificate=None,
    visible_score=None,
):
    """Refine a completed trajectory using public examples and static evidence.

    ``base`` can supply a previously generated ``(tokens, stats, diagnostics)``
    tuple. The certificate and visible-score callbacks let a benchmark inspect
    its submitted function or snippet rather than prompt demonstrations.
    """
    config = config or SamplerConfig()
    certify = certificate or python_certificate
    score_visible = visible_score or visible_test_score
    kwargs = dict(
        model=model,
        prompt=prompt,
        attention_mask=attention_mask,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=temperature,
        mask_id=mask_id,
        tokenizer=tokenizer,
        config=config,
    )
    tokens, stats, diagnostics = (
        base if base is not None else generate(remasking="ripple", **kwargs)
    )
    stats = dict(stats)
    prompt_length = prompt.shape[1]
    baseline_certificate = certify(tokens[0], prompt_length, tokenizer)
    ranked = unit_suspicion(
        model,
        tokens,
        prompt_length,
        tokenizer,
        mask_id,
        attention_mask=attention_mask,
        max_units=2,
        probe_cap=6,
    )
    visible, total = score_visible(tokens, prompt_length, prompt, tokenizer, timeout=1.0)
    best, best_score = tokens, visible
    repair_calls = 0
    if total and visible < total:
        for _, unit in ranked:
            candidate, calls = rediffuse_positions(
                model, tokens, unit, mask_id, steps=8, attention_mask=attention_mask
            )
            repair_calls += calls
            score, _ = score_visible(candidate, prompt_length, prompt, tokenizer, timeout=1.0)
            if (
                score > best_score
                and certify(candidate[0], prompt_length, tokenizer) >= baseline_certificate
            ):
                best, best_score = candidate, score
    stats.update(post_units=len(ranked), post_repair_calls=repair_calls, full_experts=0)
    if not torch.equal(best, tokens):
        stats.update(post_accepted=1, cascade_reason="visible_test")
        return best, stats, diagnostics
    if not hard_deficient(baseline_certificate):
        stats.update(post_accepted=0, cascade_reason="clean_certificate")
        return tokens, stats, diagnostics
    experts = ["structural_core", "safe_graph"]
    if baseline_certificate.parse_ok:
        experts.reverse()
    candidates = []
    for expert in experts:
        candidate, expert_stats, _ = generate(remasking=expert, **kwargs)
        _merge_statistics(stats, expert_stats)
        stats["full_experts"] += 1
        candidates.append(candidate)
        candidate_certificate = certify(candidate[0], prompt_length, tokenizer)
        if candidate_certificate > baseline_certificate and not hard_deficient(
            candidate_certificate
        ):
            stats.update(post_accepted=1, cascade_reason=expert)
            return candidate, stats, diagnostics
    selected, reason = select_by_certificate(
        tokens,
        candidates,
        prompt_length,
        tokenizer,
        certificate=certify,
    )
    stats.update(post_accepted=int(not torch.equal(selected, tokens)), cascade_reason=reason)
    return selected, stats, diagnostics
