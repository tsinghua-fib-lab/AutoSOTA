"""Statement diagnosis, local re-diffusion, and prompt-visible execution."""

from __future__ import annotations

import math
import os
import re
import subprocess
import sys

import torch
import torch.nn.functional as F

from .guards import StaticCertificate, python_certificate


def select_by_certificate(base, candidates, prompt_length, tokenizer, certificate=None):
    """Prefer a better certificate, breaking ties by the smallest token edit."""
    certify = certificate or python_certificate
    base_certificate = certify(base[0], prompt_length, tokenizer)
    certificates = [certify(candidate[0], prompt_length, tokenizer) for candidate in candidates]
    better = [index for index, value in enumerate(certificates) if value > base_certificate]
    if not better:
        return base, "abstain"
    best = max(certificates[index] for index in better)
    tied = [index for index in better if certificates[index] == best]
    winner = min(tied, key=lambda index: edit_fraction(base, candidates[index], prompt_length))
    return candidates[winner], "certificate"


STOP_TOKEN_IDS = {126081, 126348}


def hard_deficient(certificate: StaticCertificate) -> bool:
    return (
        certificate.parse_ok == 0
        or certificate.negative_undefined < 0
        or certificate.negative_argument_calls < 0
    )


def active_generation_positions(tokens: torch.Tensor, prompt_length: int) -> list[int]:
    """Return generated positions before the first model stop token."""
    positions = []
    for position in range(prompt_length, tokens.shape[-1]):
        if int(tokens[position].item()) in STOP_TOKEN_IDS:
            break
        positions.append(position)
    return positions


def token_units(tokens, prompt_length: int, tokenizer, max_unit_tokens: int = 20):
    """Approximate AST statement units without requiring token/character offsets."""
    positions = active_generation_positions(tokens, prompt_length)
    if not positions:
        return []
    units, current = ([], [])
    for position in positions:
        current.append(position)
        piece = tokenizer.decode(
            [int(tokens[position].item())],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if "\n" in piece or len(current) >= max_unit_tokens:
            units.append(current)
            current = []
    if current:
        units.append(current)
    return [unit for unit in units if unit]


@torch.no_grad()
def unit_suspicion(
    model,
    tokens: torch.Tensor,
    prompt_length: int,
    tokenizer,
    mask_id: int,
    attention_mask=None,
    max_units: int = 3,
    probe_cap: int | None = None,
):
    """Rank completed-code units by leave-unit-out reconstruction difficulty."""
    if tokens.ndim != 2 or tokens.shape[0] != 1:
        raise ValueError("post-draft methods currently require batch size one")
    units = token_units(tokens[0], prompt_length, tokenizer)
    if not units:
        return []
    probe_cap = max(max_units, 6 if probe_cap is None else int(probe_cap))
    if len(units) > probe_cap:
        indices = torch.linspace(0, len(units) - 1, probe_cap).round().long().tolist()
        units = [units[index] for index in sorted(set(indices))]
    scored = []
    for unit in units:
        state = tokens.clone()
        positions = torch.tensor(unit, device=tokens.device)
        original = state[0, positions].clone()
        state[0, positions] = mask_id
        logits = model(state, attention_mask=attention_mask).logits[0, positions].float()
        logp = F.log_softmax(logits, dim=-1).gather(-1, original[:, None]).squeeze(-1)
        predictions = logits.argmax(-1)
        disagreement = (predictions != original).float().mean()
        score = float((-logp.mean() + disagreement).item())
        scored.append((score, unit))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:max_units]


@torch.no_grad()
def rediffuse_positions(
    model,
    base: torch.Tensor,
    positions: list[int],
    mask_id: int,
    steps: int = 8,
    attention_mask=None,
    allowed_proposals: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """Locally inpaint selected positions under the complete draft context."""
    candidate = base.clone()
    if not positions:
        return (candidate, 0)
    index = torch.tensor(sorted(set(positions)), device=candidate.device)
    candidate[0, index] = mask_id
    calls = 0
    for step in range(max(1, steps)):
        remaining = torch.nonzero(candidate[0] == mask_id, as_tuple=False).flatten()
        if remaining.numel() == 0:
            break
        logits = model(candidate, attention_mask=attention_mask).logits[0, remaining].float()
        calls += 1
        probabilities = F.softmax(logits, dim=-1)
        predictions = probabilities.argmax(-1)
        confidence = probabilities.amax(-1)
        if allowed_proposals is not None:
            proposed = allowed_proposals[remaining]
            permitted = proposed >= 0
            predictions = torch.where(permitted, proposed, predictions)
            confidence = torch.where(permitted, torch.ones_like(confidence), confidence)
        take = min(
            int(remaining.numel()), max(1, int(math.ceil(remaining.numel() / max(1, steps - step))))
        )
        rows = confidence.topk(take).indices
        candidate[0, remaining[rows]] = predictions[rows]
    candidate[candidate == mask_id] = base[candidate == mask_id]
    return (candidate, calls)


def edit_fraction(left: torch.Tensor, right: torch.Tensor, prompt_length: int) -> float:
    left = left[:, prompt_length:]
    right = right[:, prompt_length:]
    return float((left != right).float().mean().item())


def _generated_text(tokens, prompt_length: int, tokenizer) -> str:
    text = tokenizer.decode(
        tokens[0, prompt_length:].detach().cpu().tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    text = text.split("[DONE]", 1)[0]
    text = re.sub("^```(?:python)?\\s*", "", text.strip())
    return text.split("```", 1)[0]


def visible_assertions(prompt: torch.Tensor, tokenizer) -> list[str]:
    text = tokenizer.decode(
        prompt[0].detach().cpu().tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    segment = text.rsplit("[DONE]", 1)[-1]
    assertions = [
        line.strip() for line in segment.splitlines() if line.strip().startswith("assert ")
    ]
    if assertions:
        return assertions
    lines = segment.splitlines()
    for index, line in enumerate(lines[:-1]):
        expression = line.strip()
        expected = lines[index + 1].strip()
        if expression.startswith(">>> ") and expected and (not expected.startswith(">>>")):
            assertions.append(f"assert ({expression[4:]}) == ({expected})")
    return assertions


def visible_test_score(
    tokens: torch.Tensor, prompt_length: int, prompt: torch.Tensor, tokenizer, timeout: float = 1.0
) -> tuple[int, int]:
    assertions = visible_assertions(prompt, tokenizer)
    if not assertions:
        return (0, 0)
    program = _generated_text(tokens, prompt_length, tokenizer)
    harness = [program, "\n__dlm_passed = 0"]
    for statement in assertions:
        harness.extend(
            [
                "try:",
                f"    {statement}",
                "    __dlm_passed += 1",
                "except BaseException:",
                "    pass",
            ]
        )
    harness.append(f"print('__DLM_VISIBLE__', __dlm_passed, {len(assertions)})")
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-c", "\n".join(harness)],
            text=True,
            capture_output=True,
            timeout=timeout,
            env={"PATH": os.environ.get("PATH", "")},
        )
    except (subprocess.TimeoutExpired, OSError):
        return (0, len(assertions))
    match = re.search("__DLM_VISIBLE__\\s+(\\d+)\\s+(\\d+)", result.stdout)
    return (int(match.group(1)), int(match.group(2))) if match else (0, len(assertions))
