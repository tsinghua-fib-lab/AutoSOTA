"""Abstract interface for multi-rule safety scorers.

RBCBF formulates safety as a set of rule-specific Control Barrier Function margins
{h^(k)(x_{1:t})}_{k in K}. When more than one rule is active simultaneously at the
violation anchor t_u, the terminal violated-rule subset K_term selects which
constraints enter the KL projection.

In practice, K_term is often dominated by a single rule because the intersection of
multiple hard CBF half-spaces over a truncated decoding support tends to be very
restrictive or empty. The framework nonetheless admits multi-rule scorers as a
first-class abstraction; this module defines the protocol any such scorer must
follow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol


@dataclass
class MultiLabelScore:
    """Per-rule scoring result.

    Attributes
    ----------
    probabilities : mapping from rule name to scorer-estimated probability that
        the corresponding rule is violated by the prefix.
    logits        : per-rule raw logits (pre-sigmoid).
    margins       : per-rule CBF margins h^(k); positive = safe, negative = violated.
    p_any_gate    : optional unified "any-rule unsafe" probability for trigger gating.
    p_any_or      : optional max-over-rules probability used as an OR-fusion proxy.
    logit_any     : optional unified gating logit (if the scorer provides a separate gate head).
    """

    probabilities: Dict[str, float]
    logits: Dict[str, float]
    margins: Dict[str, float]
    p_any_gate: Optional[float] = None
    p_any_or: Optional[float] = None
    logit_any: Optional[float] = None


class MultiRuleScorer(Protocol):
    """Protocol for any scorer that returns per-rule safety margins.

    Implementations must expose at least `score_prefix(prompt, prefix) -> MultiLabelScore`.
    The framework calls this protocol via duck typing; concrete classes do not need
    to inherit from this class.
    """

    @property
    def labels(self) -> List[str]:
        """Names of the rules / safety dimensions, in canonical order."""
        ...

    def score_prefix(self, prompt: str, prefix: str) -> MultiLabelScore:
        """Score a (prompt, prefix) pair and return per-rule margins."""
        ...
