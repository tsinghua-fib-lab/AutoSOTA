from __future__ import annotations


def get_hess_name(layer_id: int, weight: str) -> str:
    sublayer = "feed_forward" if weight in ["w1", "w2", "w3"] else "attention"
    return f"layers.{int(layer_id)}.{sublayer}.{weight}"


def get_weight_name(layer_id: int, weight: str) -> str:
    return f"{get_hess_name(layer_id, weight)}.weight"


def concat_dim(weight: str) -> int:
    return 1 if weight in ["wo", "w2"] else 0
