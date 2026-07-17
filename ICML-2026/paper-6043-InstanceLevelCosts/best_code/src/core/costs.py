import math
from dataclasses import dataclass

@dataclass
class CostFields:
    delta_signed: float
    abs_delta: float
    y_star: int

# Reference implementation for converting votes to costs
def votes_to_costs(n_yes: int, n_no: int, eps: float=1.0, tie_positive=True) -> CostFields:
    d = math.log((n_yes + eps) / (n_no + eps))
    y_star = 1 if (d > 0 or (d == 0 and tie_positive)) else 0
    return CostFields(delta_signed=d, abs_delta=abs(d), y_star=y_star)
