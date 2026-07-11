from dataclasses import dataclass, field
from typing import Optional, List
import torch


@dataclass
class MetaData:
    all_seqs: List[int] = field(default_factory=list)
    running_seqs: List[int] = field(default_factory=list)
    finished_seqs: List[int] = field(default_factory=list)
    running_seqs_tensor: Optional[torch.Tensor] = None


_METADATA = MetaData()


def get_metadata():
    return _METADATA


def set_metadata(all_seqs, running_seqs, finished_seqs):
    global _METADATA
    _METADATA = MetaData(all_seqs=all_seqs, running_seqs=running_seqs, finished_seqs=finished_seqs)


def reset_metadata():
    global _METADATA
    _METADATA = MetaData()
