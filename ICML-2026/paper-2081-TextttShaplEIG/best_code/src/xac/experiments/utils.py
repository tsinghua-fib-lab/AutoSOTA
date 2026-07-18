from dataclasses import dataclass

# -----------------------------------------------------------------------------
# Config dataclasses
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class MetaExperimentConfig:
    seed: int = 42
    debug_mode: bool = False
    skip_fitting: bool = False
    time_ops: bool = False
    scalability_mode: bool = False
