from __future__ import annotations

from .actmerge import ActMerge
from .cart import CARTMerge
from .dc_merge import DCMerge
from .dare import DAREMerge
from .functional import (
    list_functional_methods,
    merge_actmerge,
    merge_cart,
    merge_dc,
    merge_dare,
    merge_functional,
    merge_isoc,
    merge_isocts,
    merge_pcb,
    merge_raw_matrices,
    merge_task_arithmetic,
    merge_ties,
    merge_tsv,
    merge_wudi,
    merge_weighted_average,
)
from .isoc_merge import IsoCMerge
from .isocts_merge import IsoCTSMerge
from .pcb_merge import PCBMerge
from .task_arithmetic import TaskArithmeticMerge
from .ties_merge import TIESMerge
from .tsv_merge import TSVMerge
from .wudi_merge import WUDIMerge
from .weighted_average import WeightedAverageMerge

__all__ = [
    "ActMerge",
    "CARTMerge",
    "CoreFull",
    "CoreFullTSV",
    "DCMerge",
    "DAREMerge",
    "IsoCMerge",
    "IsoCTSMerge",
    "PCBMerge",
    "TaskArithmeticMerge",
    "TIESMerge",
    "TSVMerge",
    "WUDIMerge",
    "WeightedAverageMerge",
    "list_functional_methods",
    "merge_actmerge",
    "merge_cart",
    "merge_dc",
    "merge_dare",
    "merge_functional",
    "merge_isoc",
    "merge_isocts",
    "merge_pcb",
    "merge_raw_matrices",
    "merge_task_arithmetic",
    "merge_ties",
    "merge_tsv",
    "merge_wudi",
    "merge_weighted_average",
]
