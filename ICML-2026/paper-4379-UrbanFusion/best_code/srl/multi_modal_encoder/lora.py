#!/usr/bin/env python3
"""
Description: Implements the LoRA class for low-rank adaptation of linear
projection matrices.
"""

from math import sqrt

import torch.nn as nn
from torch import Tensor


class LoRA(nn.Module):
    def __init__(
        self,
        w: nn.Linear,
        rank: int,
        dim: int,
        initialize: bool = True,
        init_type: str = "xavier_uniform",
        init_settings: dict = None,
        out_dim: int = None,
        scaling: float = None,
    ) -> None:
        """
        Implements a Low Rank Adaptation (LoRA) module.

        Parameters
        ----------
        w : nn.Linear
            The original projection layer.
        rank : int
            The rank of the LoRA module (low-rank approximation).
        dim : int
            The input dimension of the weight matrix.
        initialize : bool, optional
            Whether to initialize weights, by default True.
        init_type : str, optional
            Initialization type: "normal", "kaiming_uniform", "xavier_uniform".
            By default "xavier_uniform".
        init_settings : dict, optional
            Custom initialization settings (e.g., mean/std for "normal").
        out_dim : int, optional
            Output dimension of the LoRA layer, by default same as input dim.
        scaling : float, optional
            Scaling factor for LoRA adjustment, by default equal to rank.
        """

        super().__init__()

        # Ensure `w` is an instance of `nn.Linear`
        if not isinstance(w, nn.Linear):
            raise TypeError(
                f"Expected `w` to be `nn.Linear`, but got {type(w)}"
            )

        # Store parameters
        self.rank = rank
        self.w = w
        self.out_dim = out_dim or dim
        self.scaling = scaling if scaling is not None else rank

        # LoRA matrices (low-rank decomposition)
        self.w_a = nn.Linear(dim, rank, bias=False)
        self.w_b = nn.Linear(rank, self.out_dim, bias=False)

        # Initialize weights if requested
        if initialize:
            self._initialize_weights(init_type, init_settings)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the LoRA Layer.

        Parameters
        ----------
        x : Tensor
            The input tensor.

        Returns
        -------
        out : Tensor
            The output tensor.
        """

        # Compute the LoRA adjustment and scale it
        lora_adjustment = (self.scaling / self.rank) * self.w_b(self.w_a(x))

        # Apply LoRA to the base transformation
        out = self.w(x) + lora_adjustment

        return out

    def _initialize_weights(
        self, init_type: str = "xavier_uniform", init_settings: dict = None
    ) -> None:
        """
        Initializes the weights of the LoRA matrices.

        Parameters
        ----------
        init_type : str, optional
            The type of initialization to use, by default "normal".
            Supported types are "normal", "kaiming_uniform", "xavier_uniform".
        init_settings : dict, optional
            Additional settings for the initialization method such as mean,
            std for "normal", a_squared for "kaiming_uniform", gain for
            "xavier_uniform". By default None.
        """

        # Default initialization settings
        defaults = {
            "normal": {"mean": 0, "std": 0.02},
            "kaiming_uniform": {"a_squared": 5},
            "xavier_uniform": {"gain": 1},
        }

        if init_type not in defaults:
            raise ValueError(
                f"Invalid initialization type '{init_type}'. "
                f"Available types: {list(defaults.keys())}"
            )

        # Merge user settings with defaults
        settings = {**defaults[init_type], **(init_settings or {})}

        # Initialize weights based on type
        if init_type == "normal":
            nn.init.normal_(
                self.w_a.weight, mean=settings["mean"], std=settings["std"]
            )
        elif init_type == "kaiming_uniform":
            nn.init.kaiming_uniform_(
                self.w_a.weight, a=sqrt(settings["a_squared"])
            )
        elif init_type == "xavier_uniform":
            nn.init.xavier_uniform_(self.w_a.weight, gain=settings["gain"])

        # Zero-initialize `w_b.weight` in all cases
        nn.init.zeros_(self.w_b.weight)
