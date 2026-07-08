# substantive.faircp.structs

from dataclasses import dataclass
from typing import Dict, List

from substantive.faircp.structs.enums import ConformalMethod


@dataclass
class ConformalDetail:
    prompt: str
    label: int
    group: int
    predictions: Dict[ConformalMethod, List[int]]  # conformal method → predicted set


@dataclass
class FairnessInput:
    instances: List[ConformalDetail]
    label_map: Dict[int, str]
    group_map: Dict[int, str]
