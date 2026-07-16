from dataclasses import dataclass


@dataclass
class ConfigCode:
    source_folder: str = "static_data/selected"
    peeking: float = 0.0
    peek_frac: float = 0.0
