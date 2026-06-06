from enum import Enum
from typing import Union, List


# This enum stores the status for processing every file
class Status(Enum):
    SUCCEEDED = "SUCCEEDED"
    FAILED_SKEWNESS = "FAILED.LOW_MATCHING_SKEWNESS"
    FAILED_PART_RELATIVE_SZ = "FAILED.DIFFERENT_PART_RELATIVE_SZ"
    FAILED_PART_ASPECT_RATIO = "FAILED.DIFFERENT_PART_ASPECT_RATIO"
    FAILED_INPAINTING = "FAILED.INPAINTING"
    FAILED_OVERLAPPING_PART = "FAILED.OVERLAPPING_PART"
    ERROR_UNKNOWN = "ERROR.UNKNOWN"


def check_if_idx_to_be_logged(ds_idxs, idxs_to_log):
    """
    Check if any elements of ds_idx exist in idxs_to_log and return their indices.
    """
    return [i for i, item in enumerate(ds_idxs) if item in idxs_to_log]
