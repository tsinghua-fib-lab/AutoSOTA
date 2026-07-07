import math
from abc import ABC, abstractmethod
from typing import Literal

import torch
from torch import nn, Tensor


class Distance(nn.Module, ABC):
    @abstractmethod
    def forward(self, pert: Tensor, base: Tensor) -> Tensor:
        pass


class LpNormDistance(Distance):
    def __init__(
        self,
        p: float = 2.0,
        norm_type: Literal["absolute", "relative", "relative_entrywise", "mean"] = "absolute",
        eps: float = 1e-6,
    ):
        """
        Initialize LpNormDistance with configurable norm type.

        Args:
            p: The p-norm parameter (default: 2.0)
            norm_type: Type of norm to compute. Options:
                - "absolute": Basic Lp norm
                - "relative": Relative Lp norm (normalized by base norm)
                - "relative_entrywise": Entry-wise relative Lp norm
                - "mean": Mean-normalized Lp norm
            eps: Small epsilon value to avoid division by zero (default: 1e-6)
        """
        super().__init__()
        valid_types = {"absolute", "relative", "relative_entrywise", "mean"}
        if norm_type not in valid_types:
            raise ValueError(f"Invalid norm_type: {norm_type}. Must be one of: {valid_types}")
        self.p = p
        self.norm_type = norm_type
        self.eps = eps

    def forward(self, pert: Tensor, base: Tensor) -> Tensor:
        pert = pert.flatten().float()
        base = base.flatten().float()

        if self.norm_type == "absolute":
            return torch.norm(pert, p=self.p)
        elif self.norm_type == "relative":
            return torch.norm(pert, p=self.p) / (torch.norm(base, p=self.p) + self.eps)
        elif self.norm_type == "relative_entrywise":
            return torch.norm(pert / (base.abs() + self.eps), p=self.p) / math.pow(pert.numel(), 1 / self.p)
        elif self.norm_type == "mean":
            return torch.norm(pert, p=self.p) / math.pow(pert.numel(), 1 / self.p)
        else:
            raise ValueError(
                f"Unknown norm_type: {self.norm_type}. Must be one of: "
                "'absolute', 'relative', 'relative_entrywise', 'mean'"
            )


def entropy(vec: Tensor, return_val_p: bool = False):
    """
    Compute the entropy of a vector.

    Args:
        vec: The vector to compute the entropy of.
        return_val_p: Whether to return the entropy of. Defaults to False.
    """
    val, counts = torch.unique(vec, return_counts=True)
    p = counts / counts.sum()
    ent = -torch.sum(p * torch.log2(p))
    if return_val_p:
        return ent, val, p
    return ent
