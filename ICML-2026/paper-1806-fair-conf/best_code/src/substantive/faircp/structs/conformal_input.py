from abc import ABC
from torch import Tensor
from dataclasses import dataclass

from substantive.faircp.structs.enums import ConformalCategory


@dataclass
class ConformalInput(ABC):
    logits_test: Tensor
    targets_test: Tensor
    logits_val: Tensor
    targets_val: Tensor
    logits_calib: Tensor
    targets_calib: Tensor
    groups_test: Tensor
    used_labels: list[int]
    label_map: dict[int, str]
    group_map: dict[int, str]
    k: int
    cfg: dict


@dataclass
class ConditionalConformalInput(ConformalInput):
    groups_calib: Tensor
    groups_val: Tensor
    dataset_group_conformal_category: ConformalCategory


@dataclass
class AverageKConformalInput(ConformalInput):
    marginal_size: int

@dataclass
class ClusteredLabelConformalInput(ConformalInput):
    pass


@dataclass
class ClusteredGroupConformalInput(ConformalInput):
    groups_calib: Tensor
    groups_val: Tensor
