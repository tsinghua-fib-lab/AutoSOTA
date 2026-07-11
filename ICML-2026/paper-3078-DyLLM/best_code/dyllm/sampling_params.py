from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_new_tokens: int = 256
    steps: int = 64
    num_full_steps: int = 16
    block_size: int = 32
    ignore_eos: bool = False
    algorithm: str = "confidence"
    top_p: float = None
    top_k: int = None
    confidence_threshold: float = None
    input_len: int = None
