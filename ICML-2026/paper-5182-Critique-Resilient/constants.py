"""
Shared constants for verdict values across the research pipeline.

This module centralizes magic string verdicts to improve maintainability
and reduce the risk of typos.
"""

# Critique verdicts (as defined in prompt_library.py:178)
CRITIQUE_VERDICT_CORRECT = "correct"
CRITIQUE_VERDICT_INCORRECT = "incorrect"
CRITIQUE_VERDICT_INSUFFICIENT = "insufficient"
CRITIQUE_VERDICT_OBSCURE = "obscure"
CRITIQUE_VERDICT_UNKNOWN = "unknown"

VALID_CRITIQUE_VERDICTS = {
    CRITIQUE_VERDICT_CORRECT,
    CRITIQUE_VERDICT_INCORRECT,
    CRITIQUE_VERDICT_INSUFFICIENT,
    CRITIQUE_VERDICT_OBSCURE,
    CRITIQUE_VERDICT_UNKNOWN,
}

# Judge verdicts (from automated judging)
JUDGE_VERDICT_CLAIMANT_WINS = "claimant_wins"
JUDGE_VERDICT_DEFENDER_WINS_INCORRECT = "defender_wins_incorrect"
JUDGE_VERDICT_DEFENDER_WINS_MINOR = "defender_wins_minor"
JUDGE_VERDICT_WRONG_PROBLEM = "wrong_problem"
JUDGE_VERDICT_MIXED = "mixed"
JUDGE_VERDICT_UNKNOWN = "unknown"

# Valid verdicts for critique debates
VALID_CRITIQUE_DEBATE_VERDICTS = {
    JUDGE_VERDICT_CLAIMANT_WINS,
    JUDGE_VERDICT_DEFENDER_WINS_INCORRECT,
    JUDGE_VERDICT_DEFENDER_WINS_MINOR,
    JUDGE_VERDICT_WRONG_PROBLEM,
    JUDGE_VERDICT_MIXED,
    JUDGE_VERDICT_UNKNOWN,
}

# Valid verdicts for ill-posed debates (subset of critique debate verdicts)
VALID_ILLPOSED_DEBATE_VERDICTS = {
    JUDGE_VERDICT_CLAIMANT_WINS,
    JUDGE_VERDICT_DEFENDER_WINS_INCORRECT,
    JUDGE_VERDICT_WRONG_PROBLEM,
    JUDGE_VERDICT_MIXED,
    JUDGE_VERDICT_UNKNOWN,
}

# Verdicts where defender wins (answer is considered correct)
DEFENDER_WIN_VERDICTS = {
    JUDGE_VERDICT_DEFENDER_WINS_INCORRECT,
    JUDGE_VERDICT_DEFENDER_WINS_MINOR,
}

# Verdicts where claimant wins (successfully challenged the answer)
CLAIMANT_WIN_VERDICTS = {
    JUDGE_VERDICT_CLAIMANT_WINS,
    JUDGE_VERDICT_MIXED,  # Partial win for claimant
}

# Status values
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"
STATUS_ILL_POSED = "ill-posed"
STATUS_PENDING = "pending"  # Internal-only; do not serialize.

VALID_STATUSES = {
    STATUS_SUCCEEDED,
    STATUS_FAILED,
    STATUS_ILL_POSED,
}
