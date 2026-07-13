"""Datasets for residual quantile regression."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


def _squeeze_last_if_4d(t: torch.Tensor | None) -> torch.Tensor | None:
    """If tensor is 4-D with a trailing singleton dim, squeeze it to 3-D."""
    if t is None:
        return None
    if t.dim() == 4:
        return t.squeeze(-1)
    return t


def _ensure_exog_tensor(t: torch.Tensor | None, batch: int, length: int) -> torch.Tensor:
    """Return an exogenous tensor with shape (B, length, exog_dim)."""
    if t is None:
        return torch.zeros(batch, length, 0, dtype=torch.float32)
    return t.float().contiguous()


class ResidualSequenceDataset(Dataset):
    """Pairs context windows with future residuals plus optional exogenous signals."""

    def __init__(
        self,
        context: torch.Tensor,
        residuals: torch.Tensor,
        context_exog: torch.Tensor | None,
        future_exog: torch.Tensor | None,
    ) -> None:
        context = _squeeze_last_if_4d(context)
        residuals = _squeeze_last_if_4d(residuals)

        self.context = context.float().contiguous()
        self.residuals = residuals.float().contiguous()

        batch, steps, _ = self.context.shape[:3]
        horizon = int(self.residuals.shape[1])

        self.context_exog = _ensure_exog_tensor(context_exog, batch, steps)
        self.future_exog = _ensure_exog_tensor(future_exog, batch, horizon)

    def __len__(self) -> int:
        return int(self.context.shape[0])

    def __getitem__(self, idx: int):
        return (
            self.context[idx],
            self.residuals[idx],
            self.context_exog[idx],
            self.future_exog[idx],
        )
