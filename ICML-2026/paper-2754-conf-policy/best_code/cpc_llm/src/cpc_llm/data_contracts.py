"""Centralized column name constants for DataFrame schemas.

Defines all column names used across the CPC pipeline as constants,
so that a rename is a single-line change and typos are caught at
import time (AttributeError) rather than runtime (KeyError).
"""

# --- Core columns ---
PARTICLE = "particle"
SCORE = "score"
NUM_PARTICLES_GENERATED = "num_particles_generated"

# --- Likelihood columns (dynamic) ---
LIK_PREFIX = "lik_r"
CON_LIK_PREFIX = "con_lik_r"


def lik_col(step: int) -> str:
    """Column name for unconstrained likelihood at a given calibration step.

    Args:
        step: Zero-based calibration step index.

    Returns:
        Column name string, e.g. ``"lik_r0"`` for step 0.
    """
    return f"{LIK_PREFIX}{step}"


def con_lik_col(step: int) -> str:
    """Column name for constrained likelihood at a given calibration step.

    Args:
        step: Zero-based calibration step index.

    Returns:
        Column name string, e.g. ``"con_lik_r0"`` for step 0.
    """
    return f"{CON_LIK_PREFIX}{step}"


# --- Preference pair columns ---
HIGHER_SCORE_PARTICLE = "higher_score_particle"
LOWER_SCORE_PARTICLE = "lower_score_particle"
HIGHER_SCORE = "higher_score"
LOWER_SCORE = "lower_score"

# --- Loglikelihood columns ---
LOGLIKELIHOOD = "loglikelihood"
LIKELIHOOD = "likelihood"
LOWER_PARTICLE_LOGLIKELIHOOD = "lower_particle_loglikelihood"
HIGHER_PARTICLE_LOGLIKELIHOOD = "higher_particle_loglikelihood"

# --- DPO columns ---
PROMPT = "prompt"
CHOSEN = "chosen"
REJECTED = "rejected"
PROMPT_SCORE = "prompt_score"
CHOSEN_SCORE = "chosen_score"
REJECTED_SCORE = "rejected_score"
PROMPT_LOGLIKELIHOOD = "prompt_loglikelihood"
CHOSEN_LOGLIKELIHOOD = "chosen_loglikelihood"
REJECTED_LOGLIKELIHOOD = "rejected_loglikelihood"

# --- Instruction tuning columns ---
INPUT = "input"
TARGET = "target"

# --- Distance/quality columns ---
HAMMING_DISTANCE = "hamming_distance"
