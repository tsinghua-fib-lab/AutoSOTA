#!/usr/bin/env python3
"""
Description: Implementation of a simple RGB vector encoder for
ablations on synthetic data.
"""
import torch
import torch.nn as nn


class RGBEncoder(nn.Module):
    """
    Minimal MLP encoder for a 3‑value RGB vector, with optional token
    reshaping.

    Parameters
    ----------
    dim_hidden : int
        Size of the hidden (intermediate) layer.
    dim_out : int
        Size of the final embedding produced by the encoder.
    as_tokens : bool, optional
        If True, the encoder (or forward call) returns its output as a
        sequence of one token with shape (..., 1, dim_out).  Default: False.
    """

    def __init__(
        self, dim_hidden: int, dim_out: int, *, as_tokens: bool = False
    ) -> None:
        super().__init__()
        self.as_tokens_default = as_tokens

        self.net = nn.Sequential(
            nn.Linear(3, dim_hidden), nn.GELU(), nn.Linear(dim_hidden, dim_out)
        )
        self.seq_len = 1 if as_tokens else None

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        as_tokens: bool | None = None,
    ) -> torch.Tensor:
        """
        Forward pass for the RGB encoder.

        Parameters
        ----------
        x : torch.Tensor of shape (..., 3)
            Batch of RGB vectors.
        as_tokens : bool, optional
            Overrides the default set at construction time.  If True, the
            output gains an extra sequence dimension so every RGB embedding
            becomes a single token.

        Returns
        -------
        torch.Tensor
            Shape (..., dim_out) if as_tokens is False
            Shape (..., 1, dim_out) if as_tokens is True
        """
        y = self.net(x.float())

        # Decide whether to reshape
        flag = self.as_tokens_default if as_tokens is None else as_tokens
        if flag:
            y = y.unsqueeze(-2)  # insert token dimension before embedding dim

        if return_features:
            return y, x

        return y
