"""Sampler settings for the configurations evaluated in the paper."""

from dataclasses import dataclass, replace

MODEL_ID = "GSAI-ML/LLaDA-8B-Base"
MODEL_REVISION = "0f2787f2d87eac5eed8a087d5ecd24277e6255b2"


@dataclass(frozen=True)
class SamplerConfig:
    candidate_m: int = 32
    revise_every: int = 16
    probe_m: int = 12
    max_k: int = 4
    waves: int = 1
    influence_threshold: float = 0.15
    distance_weight: float = 0.01
    ranked_transfer: bool = False
    changed_probe_only: bool = False
    min_probe_confidence: float = 0.0
    protected_suffix: int = 16
    base_masking: str = "margin"

    def __post_init__(self):
        if self.candidate_m < 1 or self.probe_m < 1 or self.max_k < 1:
            raise ValueError("Candidate, probe, and joint-revision caps must be positive")
        if self.revise_every < 0 or self.waves < 0 or self.protected_suffix < 0:
            raise ValueError("Intervals, waves, and protected lengths cannot be negative")
        if not 0 <= self.min_probe_confidence <= 1:
            raise ValueError("Probe confidence must lie in [0, 1]")
        if self.base_masking not in {"margin", "confidence"}:
            raise ValueError("base_masking must be 'margin' or 'confidence'")


PROFILES = {
    "base": SamplerConfig(),
    "ds1000": SamplerConfig(candidate_m=16, ranked_transfer=True),
    "bigcodebench": SamplerConfig(
        probe_m=48,
        ranked_transfer=True,
        changed_probe_only=True,
        min_probe_confidence=0.30,
    ),
    "ds1000-cascade": SamplerConfig(ranked_transfer=True),
}

BENCHMARK_PROFILES = {
    "humaneval": "base",
    "mbpp": "base",
    "mbppplus": "base",
    "ds1000": "ds1000",
    "bigcodebench": "bigcodebench",
}


def get_config(profile="base", method="ripple"):
    try:
        config = PROFILES[profile]
    except KeyError as error:
        raise ValueError(f"Unknown profile {profile!r}; choose from {list(PROFILES)}") from error
    if method in {"core", "structural_core", "margin-remask", "remdm-conf"}:
        return replace(config, candidate_m=32, revise_every=8)
    return config
