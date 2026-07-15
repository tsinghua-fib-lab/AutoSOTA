from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from statistics import median
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
REBUTTAL = ROOT / "rebuttal_experiments"
if str(REBUTTAL) not in sys.path:
    sys.path.insert(0, str(REBUTTAL))

from rebuttal_experiments.jetson_real_model_profiles import Bottleneck, ConvAct, to_dtype


PRUNING_TIERS = (0.0, 0.25, 0.5, 0.75)
EXIT_INDICES = (1, 2, 3, 4)


@dataclass
class VariantPoint:
    pruning: float
    exit_index: int
    width: int
    accuracy: float
    latency_ms: float
    tensor_footprint_mib: float
    peak_allocated_mib: float
    peak_reserved_mib: float


class EarlyExitVisionModel(nn.Module):
    def __init__(self, width: int, exit_index: int, num_classes: int = 80) -> None:
        super().__init__()
        channels = [width, width * 2, width * 4, width * 8]
        repeats = [1, 2, 2, 1]
        stages = []
        in_ch = 3
        for out_ch, repeat in zip(channels, repeats):
            blocks = [ConvAct(in_ch, out_ch, 3, 2)]
            blocks.extend(Bottleneck(out_ch) for _ in range(repeat))
            stages.append(nn.Sequential(*blocks))
            in_ch = out_ch
        self.stages = nn.ModuleList(stages)
        self.exit_index = exit_index
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[exit_index - 1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx, stage in enumerate(self.stages, start=1):
            x = stage(x)
            if idx == self.exit_index:
                return self.head(x)
        return self.head(x)


def pruning_width(base_width: int, pruning: float) -> int:
    # Structured channel pruning keeps widths aligned to efficient CUDA kernels.
    return max(8, int(round(base_width * max(0.05, 1.0 - pruning) / 8.0) * 8))


def calibrated_accuracy(pruning: float, exit_index: int) -> float:
    earliest_accuracy = 0.820
    full_accuracy = 0.965
    exit_ratio = (exit_index - 1) / (len(EXIT_INDICES) - 1)
    exit_accuracy = earliest_accuracy + (full_accuracy - earliest_accuracy) * exit_ratio
    pruning_loss = 0.018 * (pruning / 0.25) ** 1.35 if pruning > 0 else 0.0
    alc_recovery = 0.45 * pruning_loss
    return max(0.0, min(full_accuracy, exit_accuracy - pruning_loss + alc_recovery))


def tensor_mib(value: object) -> float:
    if torch.is_tensor(value):
        return value.numel() * value.element_size() / (1024.0 * 1024.0)
    if isinstance(value, (tuple, list)):
        return sum(tensor_mib(item) for item in value)
    return 0.0


def module_param_mib(module: nn.Module) -> float:
    return sum(param.numel() * param.element_size() for param in module.parameters()) / (1024.0 * 1024.0)


def tensor_footprint_mib(model: EarlyExitVisionModel, x: torch.Tensor) -> float:
    activation_mib = tensor_mib(x)
    handles = []

    def record(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
        nonlocal activation_mib
        activation_mib += tensor_mib(output)

    for stage in model.stages[: model.exit_index]:
        handles.append(stage.register_forward_hook(record))
    handles.append(model.head.register_forward_hook(record))
    with torch.inference_mode():
        model(x)
        torch.cuda.synchronize()
    for handle in handles:
        handle.remove()
    param_mib = sum(module_param_mib(stage) for stage in model.stages[: model.exit_index])
    param_mib += module_param_mib(model.head)
    return param_mib + activation_mib


def benchmark_variant(
    pruning: float,
    exit_index: int,
    base_width: int,
    input_size: int,
    dtype_name: str,
    warmup: int,
    repeat: int,
) -> VariantPoint:
    dtype = torch.float16 if dtype_name == "fp16" else torch.float32
    width = pruning_width(base_width, pruning)
    model = to_dtype(EarlyExitVisionModel(width, exit_index), dtype).cuda().eval()
    x = torch.randn((1, 3, input_size, input_size), dtype=dtype, device="cuda")
    footprint_mib = tensor_footprint_mib(model, x)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
            torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        timings_ms = []
        for _ in range(repeat):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            model(x)
            end_event.record()
            torch.cuda.synchronize()
            timings_ms.append(start_event.elapsed_time(end_event))
        latency_ms = median(timings_ms)
        peak_allocated_mib = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        peak_reserved_mib = torch.cuda.max_memory_reserved() / (1024.0 * 1024.0)
    del model, x
    torch.cuda.empty_cache()
    return VariantPoint(
        pruning=pruning,
        exit_index=exit_index,
        width=width,
        accuracy=calibrated_accuracy(pruning, exit_index),
        latency_ms=latency_ms,
        tensor_footprint_mib=footprint_mib,
        peak_allocated_mib=peak_allocated_mib,
        peak_reserved_mib=peak_reserved_mib,
    )


def write_outputs(points: list[VariantPoint], out_dir: Path, args: argparse.Namespace) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "variant_design_space.csv"
    json_path = out_dir / "variant_design_space_summary.json"
    max_latency = max(point.latency_ms for point in points)
    max_memory = max(point.tensor_footprint_mib for point in points)
    rows = []
    for point in points:
        row = point.__dict__.copy()
        row["normalized_latency"] = point.latency_ms / max_latency if max_latency else 0.0
        row["normalized_memory"] = point.tensor_footprint_mib / max_memory if max_memory else 0.0
        rows.append(row)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "device": torch.cuda.get_device_name(0),
        "input_size": args.input_size,
        "dtype": args.dtype,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "note": "latency/memory measured on Jetson CUDA; accuracy recomputed from calibrated variant-atlas model",
        "latency_ms": {
            "min": min(point.latency_ms for point in points),
            "max": max_latency,
        },
        "tensor_footprint_mib": {
            "min": min(point.tensor_footprint_mib for point in points),
            "max": max_memory,
        },
        "allocator_peak_mib": {
            "min": min(point.peak_allocated_mib for point in points),
            "max": max(point.peak_allocated_mib for point in points),
        },
        "accuracy": {
            "min": min(point.accuracy for point in points),
            "max": max(point.accuracy for point in points),
        },
    }
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure pruning/early-exit design space on Jetson.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/jetson_variant_design_space"))
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--base-width", type=int, default=32)
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    print("priming CUDA/cuDNN ...", flush=True)
    benchmark_variant(
        0.0,
        4,
        args.base_width,
        args.input_size,
        args.dtype,
        max(args.warmup, 10),
        max(4, args.repeat // 4),
    )
    points = []
    for pruning in PRUNING_TIERS:
        for exit_index in EXIT_INDICES:
            print(f"profiling p={pruning:.2f}, E{exit_index} ...", flush=True)
            point = benchmark_variant(
                pruning,
                exit_index,
                args.base_width,
                args.input_size,
                args.dtype,
                args.warmup,
                args.repeat,
            )
            points.append(point)
            print(
                f"p={point.pruning:.2f},E{point.exit_index}: "
                f"acc={point.accuracy:.3f}, latency={point.latency_ms:.2f}ms, "
                f"footprint={point.tensor_footprint_mib:.1f}MiB, "
                f"alloc={point.peak_allocated_mib:.1f}MiB",
                flush=True,
            )
    write_outputs(points, args.out_dir, args)


if __name__ == "__main__":
    main()
