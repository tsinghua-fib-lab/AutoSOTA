"""Dropout schedules used in the experiments."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


DISPLAY_NAMES = {
    "none": "No dropout",
    "constant": "Constant",
    "linear": "Linear (late)",
    "reverse_linear": "Linear (early)",
    "step": "Step (late)",
    "reverse_step": "Step (early)",
    "big_step": "Big step (1/3)",
    "double": "Double (2h)",
    "triple": "Triple (3h)",
    "v_shape": "V-shape",
    "inverse_v": "Inverse V",
}

MARKERS = {
    "none": "s",
    "constant": "o",
    "reverse_step": "D",
    "big_step": "v",
    "step": "^",
    "linear": ">",
    "reverse_linear": "<",
    "double": "P",
    "triple": "X",
    "v_shape": "*",
    "inverse_v": "h",
}


def schedule_layers(schedule: str, depth: int, h_bar: float, h_max: float | None = None) -> list[float]:
    """Build the layer-wise dropout field h_l with mean h_bar."""

    if schedule == "none":
        return [0.0] * depth
    if schedule == "constant":
        return [h_bar] * depth
    if schedule == "double":
        return [2.0 * h_bar] * depth
    if schedule == "triple":
        return [3.0 * h_bar] * depth
    if schedule == "linear":
        if depth == 1:
            return [h_bar]
        return [2.0 * h_bar * i / (depth - 1) for i in range(depth)]
    if schedule == "reverse_linear":
        if depth == 1:
            return [h_bar]
        return [2.0 * h_bar * (depth - 1 - i) / (depth - 1) for i in range(depth)]
    if schedule == "step":
        h_max = h_max or 2.0 * h_bar
        n_drop = max(1, int(math.ceil(h_bar / h_max * depth)))
        h_adj = h_bar * depth / n_drop
        return [0.0] * (depth - n_drop) + [h_adj] * n_drop
    if schedule == "reverse_step":
        h_max = h_max or 2.0 * h_bar
        n_drop = max(1, int(math.ceil(h_bar / h_max * depth)))
        h_adj = h_bar * depth / n_drop
        return [h_adj] * n_drop + [0.0] * (depth - n_drop)
    if schedule == "big_step":
        n_drop = max(1, int(math.ceil(depth / 3)))
        h_adj = h_bar * depth / n_drop
        return [h_adj] * n_drop + [0.0] * (depth - n_drop)
    raise ValueError(f"Unknown schedule: {schedule}")


def effective_xi(h_layers: Sequence[float], activation_class: str = "kinked") -> float:
    """Mean-field estimate of the effective correlation length."""

    power = 1 / 2 if activation_class == "smooth" else 1 / 3
    damage = np.mean([h**power for h in h_layers if h > 0]) if any(h > 0 for h in h_layers) else 0.0
    if damage <= 0:
        return float("inf")
    return 1.0 / damage
