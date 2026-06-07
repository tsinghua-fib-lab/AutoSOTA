from dataclasses import dataclass


@dataclass
class dLLMCacheConfig:
    prompt_interval_steps: int = 1
    gen_interval_steps: int = 1
    transfer_ratio: float = 0.0
    cfg_interval_steps: int = 1
