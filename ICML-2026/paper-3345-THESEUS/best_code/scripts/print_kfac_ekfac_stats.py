#!/usr/bin/env python3

import argparse
import math
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, DefaultDict, List, Tuple

import torch


Stats = Tuple[float, float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print tensor statistics for KFAC and EKFAC checkpoint payloads."
    )
    parser.add_argument("kfac_path", type=Path, help="Path to a KFAC checkpoint (.pt)")
    parser.add_argument("ekfac_path", type=Path, help="Path to an EKFAC checkpoint (.pt)")
    parser.add_argument(
        "--normalize-results",
        action="store_true",
        help=(
            "Normalize stored tensors the same way the regularizer code consumes them. "
            "For KFAC ffT, this uses the single-checkpoint equivalent ffT / num_examples_ggT."
        ),
    )
    return parser.parse_args()


def load_payload(path: Path) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Expected mapping payload in {path}, got {type(payload).__name__}")
    return payload


def iter_tensors(payload: Mapping[str, Any], prefix: str = "") -> List[Tuple[str, torch.Tensor]]:
    out = []  # type: List[Tuple[str, torch.Tensor]]
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if torch.is_tensor(value):
            out.append((name, value))
            continue
        if isinstance(value, Mapping):
            out.extend(iter_tensors(value, prefix=name))
    return out


def tensor_stats(tensor: torch.Tensor) -> Stats:
    data = tensor.detach().to(device="cpu", dtype=torch.float64)
    flat = data.reshape(-1)
    if flat.numel() == 0:
        nan = float("nan")
        return nan, nan, nan, nan, nan
    return (
        flat.abs().sum().item(),
        torch.linalg.vector_norm(flat).item(),
        flat.max().item(),
        flat.min().item(),
        flat.mean().item(),
    )


def detect_payload_kind(payload: Mapping[str, Any]) -> str:
    if all(key in payload for key in ("aaT", "ggT", "ffT", "num_examples_ggT")):
        return "kfac"
    if all(key in payload for key in ("UA", "UG", "D", "ffT", "num_examples")):
        return "ekfac"
    return "unknown"


def _safe_example_count(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key, 1)
    try:
        return float(max(1, int(value)))
    except (TypeError, ValueError):
        return 1.0


def normalize_tensor_for_reporting(
    tensor_key: str,
    tensor: torch.Tensor,
    payload: Mapping[str, Any],
    payload_kind: str,
) -> torch.Tensor:
    if payload_kind == "kfac":
        num_examples = _safe_example_count(payload, "num_examples_ggT")
        if tensor_key.startswith("aaT."):
            return tensor / num_examples
        if tensor_key.startswith("ggT."):
            return tensor / num_examples
        if tensor_key.startswith("ffT."):
            return tensor / num_examples
        return tensor

    if payload_kind == "ekfac":
        num_examples = _safe_example_count(payload, "num_examples")
        if tensor_key.startswith("ffT."):
            return tensor / num_examples
        return tensor

    return tensor


def same_stats(lhs: Stats, rhs: Stats, *, rel_tol: float = 1e-12, abs_tol: float = 1e-12) -> bool:
    return all(
        (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
        for a, b in zip(lhs, rhs)
    )


def format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{0.0 if value == 0.0 else value:.10g}"


def format_stats(stats: Stats) -> str:
    norm_1, norm_2, max_value, min_value, avg_value = stats
    return (
        f"norm_1={format_float(norm_1)}, "
        f"norm_2={format_float(norm_2)}, "
        f"max={format_float(max_value)}, "
        f"min={format_float(min_value)}, "
        f"avg={format_float(avg_value)}"
    )


def format_shape(tensor: torch.Tensor) -> str:
    return str(tuple(tensor.shape))


def collect_entries(
    source: str,
    path: Path,
    *,
    normalize_results: bool = False,
) -> List[Tuple[str, str, Stats, str]]:
    payload = load_payload(path)
    payload_kind = detect_payload_kind(payload)
    entries = []
    for key, tensor in iter_tensors(payload):
        report_tensor = (
            normalize_tensor_for_reporting(key, tensor, payload, payload_kind)
            if normalize_results
            else tensor
        )
        entries.append((key, format_shape(tensor), tensor_stats(report_tensor), source))
    return entries


def main() -> None:
    args = parse_args()
    entries_by_key = defaultdict(list)  # type: DefaultDict[str, List[Tuple[str, Stats, List[str]]]]

    for source, path in (("kfac", args.kfac_path), ("ekfac", args.ekfac_path)):
        for key, shape, stats, entry_source in collect_entries(
            source,
            path,
            normalize_results=args.normalize_results,
        ):
            groups = entries_by_key[key]
            for existing_shape, existing_stats, sources in groups:
                if existing_shape == shape and same_stats(existing_stats, stats):
                    sources.append(entry_source)
                    break
            else:
                groups.append((shape, stats, [entry_source]))

    for key in sorted(entries_by_key):
        grouped_entries = entries_by_key[key]
        show_sources = len(grouped_entries) > 1 or any(
            len(set(sources)) > 1 for _, _, sources in grouped_entries
        )
        for shape, stats, sources in grouped_entries:
            label = f"{key} [{', '.join(sorted(set(sources)))}]" if show_sources else key
            print(f"{label}: shape={shape}, {format_stats(stats)}")


if __name__ == "__main__":
    main()
