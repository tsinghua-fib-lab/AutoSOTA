"""
Shared helpers for determining winners according to docs/victory_rules.md.

This module centralizes how we interpret verdict strings and combine
human/automated judgments so that every analysis tool follows the same rules.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Iterable, List, Optional, Sequence

from data_models import AutomatedEvaluation, HumanEvaluation
from constants import (
    JUDGE_VERDICT_CLAIMANT_WINS,
    JUDGE_VERDICT_DEFENDER_WINS_INCORRECT,
    JUDGE_VERDICT_DEFENDER_WINS_MINOR,
    JUDGE_VERDICT_MIXED,
    JUDGE_VERDICT_UNKNOWN,
    JUDGE_VERDICT_WRONG_PROBLEM,
    STATUS_FAILED,
)

logger = logging.getLogger(__name__)


class VictorySide(str, Enum):
    """Canonical winner labels."""

    ALICE = "alice"
    BOB = "bob"
    DROP = "drop"


class ConflictingHumanJudgmentError(RuntimeError):
    """Raised when human annotators disagree on the same task."""


_HUMAN_NORMALIZATION = {
    "critique": {
        "correct": JUDGE_VERDICT_CLAIMANT_WINS,
        "incorrect": JUDGE_VERDICT_DEFENDER_WINS_INCORRECT,
        "incorrect but wrong reason": JUDGE_VERDICT_WRONG_PROBLEM,
        "other": JUDGE_VERDICT_UNKNOWN,
        "unknown": JUDGE_VERDICT_UNKNOWN,
        "invalid": JUDGE_VERDICT_UNKNOWN,
    },
    "illposed": {
        "ill-posed": JUDGE_VERDICT_CLAIMANT_WINS,
        "not ill-posed": JUDGE_VERDICT_DEFENDER_WINS_INCORRECT,
        "ill-posed but wrong reason": JUDGE_VERDICT_WRONG_PROBLEM,
        "correct": JUDGE_VERDICT_DEFENDER_WINS_INCORRECT,
        "incorrect": JUDGE_VERDICT_CLAIMANT_WINS,
        "other": JUDGE_VERDICT_UNKNOWN,
        "unknown": JUDGE_VERDICT_UNKNOWN,
        "invalid": JUDGE_VERDICT_UNKNOWN,
    },
}

_ALICE_WINS = {
    "critique": {JUDGE_VERDICT_CLAIMANT_WINS, JUDGE_VERDICT_MIXED},
    "illposed": {JUDGE_VERDICT_CLAIMANT_WINS, JUDGE_VERDICT_MIXED},
}

_BOB_WINS = {
    "critique": {
        JUDGE_VERDICT_DEFENDER_WINS_INCORRECT,
        JUDGE_VERDICT_DEFENDER_WINS_MINOR,
        JUDGE_VERDICT_WRONG_PROBLEM,
    },
    "illposed": {
        JUDGE_VERDICT_DEFENDER_WINS_INCORRECT,
        JUDGE_VERDICT_WRONG_PROBLEM,
    },
}

_DROP_VERDICTS = {
    "critique": {JUDGE_VERDICT_UNKNOWN},
    "illposed": {JUDGE_VERDICT_UNKNOWN},
}


def _claim_type_key(claim_type: str) -> str:
    """Normalize claim types to the two families we care about."""
    if claim_type == "critique_debate":
        return "critique"
    if claim_type not in {"critique", "illposed"}:
        logger.warning("Unknown claim_type '%s'; defaulting to 'critique'", claim_type)
        return "critique"
    return claim_type


def _normalize_verdict(claim_type: str, verdict: Optional[str]) -> Optional[str]:
    if not verdict:
        return None
    verdict_clean = verdict.strip().lower()
    if not verdict_clean:
        return None
    key = _claim_type_key(claim_type)
    mapping = _HUMAN_NORMALIZATION.get(key, {})
    return mapping.get(verdict_clean, verdict_clean)


def verdict_to_victory_side(claim_type: str, verdict: Optional[str]) -> Optional[VictorySide]:
    """
    Map a single verdict string to a winner.

    Returns:
        VictorySide.ALICE if Alice wins, VictorySide.BOB if Bob wins,
        VictorySide.DROP if verdict means neither, or None if verdict is unknown.
    """

    norm = _normalize_verdict(claim_type, verdict)
    if not norm:
        return None

    key = _claim_type_key(claim_type)

    if norm in _ALICE_WINS[key]:
        return VictorySide.ALICE
    if norm in _BOB_WINS[key]:
        return VictorySide.BOB
    if norm in _DROP_VERDICTS[key]:
        return VictorySide.DROP

    logger.warning("Unknown verdict '%s' for claim_type '%s'", norm, key)
    return None


def resolve_victory_from_verdicts(
    claim_type: str,
    *,
    human_verdicts: Optional[Iterable[str]] = None,
    automated_verdicts: Optional[Iterable[str]] = None,
    context: Optional[str] = None,
    log_automated_disagreements: bool = True,
) -> Optional[VictorySide]:
    """
    Resolve a winner following docs/victory_rules.md.

    Human judgments take precedence. If multiple humans disagree, raise an error.
    Automated judgments must be unanimous; otherwise we log an error (if enabled) and return None.
    """

    human_verdicts = human_verdicts or []
    auto_verdicts = automated_verdicts or []

    human_sides: List[VictorySide] = []
    for verdict in human_verdicts:
        side = verdict_to_victory_side(claim_type, verdict)
        if side is not None:
            human_sides.append(side)
    if human_sides:
        if len(set(human_sides)) > 1:
            raise ConflictingHumanJudgmentError(
                f"Conflicting human judgments ({set(human_sides)}) for {context or 'unknown task'}"
            )
        return human_sides[0]

    auto_sides: List[VictorySide] = []
    for verdict in auto_verdicts:
        side = verdict_to_victory_side(claim_type, verdict)
        if side is not None:
            auto_sides.append(side)

    if not auto_sides:
        return None

    unique = set(auto_sides)
    if len(unique) != 1:
        if log_automated_disagreements:
            logger.error(
                "Automated judgments disagree for %s: %s",
                context or "unknown task",
                sorted(side.value for side in unique),
            )
        return None

    return auto_sides[0]


def resolve_automated_victory(
    claim_type: str,
    decisions: Sequence[AutomatedEvaluation],
    *,
    context: Optional[str] = None,
    log_automated_disagreements: bool = True,
) -> Optional[VictorySide]:
    """Convenience wrapper for AutomatedEvaluation objects."""

    verdicts = [
        decision.verdict
        for decision in decisions
        if decision and decision.verdict and decision.status != STATUS_FAILED
    ]
    return resolve_victory_from_verdicts(
        claim_type,
        automated_verdicts=verdicts,
        context=context,
        log_automated_disagreements=log_automated_disagreements,
    )


def resolve_human_victory(
    claim_type: str,
    decisions: Sequence[HumanEvaluation],
    *,
    context: Optional[str] = None,
) -> Optional[VictorySide]:
    """Convenience wrapper for HumanEvaluation objects."""

    verdicts = [decision.verdict for decision in decisions if decision and decision.verdict]
    return resolve_victory_from_verdicts(claim_type, human_verdicts=verdicts, context=context)
