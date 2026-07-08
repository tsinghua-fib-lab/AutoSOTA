from dataclasses import dataclass

from substantive.faircp.structs.enums import ConformalMethod


@dataclass
class FairnessExperimentResult:
    index: int
    method: ConformalMethod
    group_text: str
    label_text: str
    result: str
    conformal_set: list[int]
    difficulty: int
