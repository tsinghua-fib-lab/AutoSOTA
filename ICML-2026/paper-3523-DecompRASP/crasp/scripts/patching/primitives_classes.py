import torch
from dataclasses import dataclass
from typing import Optional, Any

@dataclass(frozen=True)
class AttentionInteraction:
    activation_name_to_keep_k: str
    activation_name_to_keep_q: Optional[str] = None

@dataclass(frozen=True)
class LogitsInteraction:
    activation_name_to_keep: str

@dataclass
class Primitive:
    name: str
    is_only_token: bool = False
    contruct_matrix: Optional[Any] = None
    operation: Optional[str] = None
    has_default_scalar: Optional[bool] = None

@dataclass
class AbstractPrimitive:
    name: str
    primitive: Optional['Primitive'] = None
    special_primitive: Optional['Primitive'] = None
    scaling_factor_primitive: Optional[float|str] = None
    pruning_mask: Optional[Any] = None # deprecated
    replacement_matrix: Optional[Any] = None
    coefficients_on_primitives: Optional[Any] = None  # deprecated

@dataclass
class PrimitiveEval:
    zero_parameters: int|list[int]
    total_parameters: int|list[int]
    is_fully_replaced: bool