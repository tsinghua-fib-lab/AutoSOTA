"""Read and write the saved runs used by the paper figures."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def _pack_for_npz(value: Any, arrays: dict[str, np.ndarray]) -> Any:
    if isinstance(value, np.ndarray):
        key = f"arr_{len(arrays):05d}"
        arrays[key] = value
        return {"__ndarray__": key}
    if isinstance(value, np.generic):
        return _pack_for_npz(value.item(), arrays)
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if math.isnan(value):
            return {"__float__": "nan"}
        return {"__float__": "inf" if value > 0 else "-inf"}
    if isinstance(value, dict):
        return {str(key): _pack_for_npz(item, arrays) for key, item in value.items()}
    if isinstance(value, list):
        return [_pack_for_npz(item, arrays) for item in value]
    if isinstance(value, tuple):
        return {"__tuple__": [_pack_for_npz(item, arrays) for item in value]}
    return value


def _unpack_from_npz(value: Any, arrays: dict[str, np.ndarray]) -> Any:
    if isinstance(value, dict):
        if set(value) == {"__ndarray__"}:
            return arrays[value["__ndarray__"]]
        if set(value) == {"__float__"}:
            label = value["__float__"]
            if label == "nan":
                return math.nan
            return math.inf if label == "inf" else -math.inf
        if set(value) == {"__tuple__"}:
            return tuple(_unpack_from_npz(item, arrays) for item in value["__tuple__"])
        return {key: _unpack_from_npz(item, arrays) for key, item in value.items()}
    if isinstance(value, list):
        return [_unpack_from_npz(item, arrays) for item in value]
    return value


def save_npz_result(path: str | Path, data: Any) -> None:
    """Store nested result dictionaries as JSON metadata plus NPZ arrays."""

    arrays: dict[str, np.ndarray] = {}
    metadata = json.dumps(_pack_for_npz(data, arrays), separators=(",", ":"), allow_nan=False)
    payload = {"__metadata__": np.frombuffer(metadata.encode("utf-8"), dtype=np.uint8), **arrays}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def load_npz_result(path: str | Path) -> Any:
    """Load result files written by :func:`save_npz_result` without pickle."""

    with np.load(path, allow_pickle=False) as bundle:
        metadata = bytes(bundle["__metadata__"]).decode("utf-8")
        arrays = {key: bundle[key] for key in bundle.files if key != "__metadata__"}
        return _unpack_from_npz(json.loads(metadata), arrays)


def parse_tuple_key(key: str) -> tuple[Any, ...]:
    """Turn a stringified sweep key back into a tuple."""

    value = ast.literal_eval(key)
    if not isinstance(value, tuple):
        raise ValueError(f"Expected tuple-like key, got {key!r}")
    return value
