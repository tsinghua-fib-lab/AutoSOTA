from dataclasses import dataclass
from typing import Optional, List


@dataclass
class QuantizationConfig:
    q: int
    betas: List[float]


@dataclass
class Config:
    act: Optional[QuantizationConfig]
    keys: Optional[QuantizationConfig]
    values: Optional[QuantizationConfig]


def create_config(q, quant_act, quant_kv,
                  act_betas, key_betas, value_betas):
    act = QuantizationConfig(q, act_betas) if quant_act else None
    keys = QuantizationConfig(q, key_betas) if quant_kv else None
    values = QuantizationConfig(q, value_betas) if quant_kv else None
    return Config(act, keys, values)


no_q_config = Config(None, None, None)
