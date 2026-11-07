"""Token-local quarantine for degenerate non-terminating responses."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RepetitionQuarantineConfig:
    coefficient: float = 0.10
    ngram_size: int = 8
    min_repeats: int = 3
    invalid_tail_tokens: int = 256


def _repetition_onset(tokens: list[int], ngram_size: int, min_repeats: int):
    occurrences: dict[tuple[int, ...], list[int]] = {}
    for position in range(max(0, len(tokens) - ngram_size + 1)):
        ngram = tuple(tokens[position:position + ngram_size])
        positions = occurrences.setdefault(ngram, [])
        positions.append(position)
        if len(positions) >= min_repeats:
            # Once a third occurrence confirms a loop, the second occurrence
            # is the earliest confidently redundant span.
            return positions[1]
    return None


def apply_repetition_quarantine(
    advantages: torch.Tensor,
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
    correctness: torch.Tensor,
    format_valid: torch.Tensor,
    config: RepetitionQuarantineConfig,
):
    if config.coefficient < 0:
        raise ValueError("coefficient must be non-negative")
    if config.ngram_size < 2 or config.min_repeats < 2:
        raise ValueError("ngram_size and min_repeats must be at least two")
    if config.invalid_tail_tokens < 1:
        raise ValueError("invalid_tail_tokens must be positive")

    residual = torch.zeros_like(advantages, dtype=torch.float32)
    active = 0
    repetition_active = 0
    tail_active = 0
    onset_fractions = []
    for row in range(response_ids.shape[0]):
        if bool(correctness[row].item()) or bool(format_valid[row].item()):
            continue
        length = int(response_mask[row].sum().detach().cpu().item())
        tokens = response_ids[row, :length].detach().cpu().tolist()
        repetition = _repetition_onset(
            tokens, config.ngram_size, config.min_repeats
        )
        tail = config.invalid_tail_tokens if length > config.invalid_tail_tokens else None
        candidates = [value for value in (repetition, tail) if value is not None]
        if not candidates:
            continue
        onset = min(candidates)
        residual[row, onset:length] = -1.0
        active += 1
        repetition_active += int(repetition is not None)
        tail_active += int(tail is not None)
        onset_fractions.append(onset / max(length, 1))

    updated = advantages + float(config.coefficient) * residual * response_mask
    metrics = {
        "repetition_quarantine/active_response_fraction": float(
            active / max(response_ids.shape[0], 1)
        ),
        "repetition_quarantine/repetition_response_fraction": float(
            repetition_active / max(response_ids.shape[0], 1)
        ),
        "repetition_quarantine/tail_response_fraction": float(
            tail_active / max(response_ids.shape[0], 1)
        ),
        "repetition_quarantine/quarantined_token_fraction": float(
            residual.abs().sum().detach().cpu().item()
            / max(response_mask.sum().detach().cpu().item(), 1)
        ),
        "repetition_quarantine/onset_fraction_mean": float(
            sum(onset_fractions) / max(len(onset_fractions), 1)
        ) if onset_fractions else 1.0,
    }
    return updated, metrics
