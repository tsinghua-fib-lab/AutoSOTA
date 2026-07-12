"""
Closed-form merge where each teacher uses only its own dataset frames.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from orb_models.dataset.ase_sqlite_dataset import AseSqliteDataset
from orb_models.forcefield import base as ff_base
from orb_models.forcefield import property_definitions
from orb_models.forcefield import pretrained

try:
    from .train_orb import deterministic_train_val_test_split
except ImportError:  # pragma: no cover - fallback for direct script execution
    import sys
    sys.path.append(str(Path(__file__).resolve().parent))
    from train_orb import deterministic_train_val_test_split  # type: ignore


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def resolve_accum_device(accum_device: str, model_device: torch.device) -> torch.device:
    if accum_device == "auto":
        return model_device
    if accum_device == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("CUDA unavailable; using CPU for accumulators.")
        return torch.device("cpu")
    return torch.device("cpu")


def resolve_solve_device(solve_device: str, accum_device: torch.device) -> torch.device:
    if solve_device == "auto":
        return accum_device
    if solve_device == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("CUDA unavailable; using CPU for solve.")
        return torch.device("cpu")
    return torch.device("cpu")


def extract_state_dict(checkpoint_path: Path) -> Mapping[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def mean_state(states: Sequence[Mapping[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if len(states) < 2:
        raise ValueError("At least two teacher checkpoints are required for closed-form merge.")
    merged: Dict[str, torch.Tensor] = {}
    keys = set(states[0].keys())
    for idx, state in enumerate(states[1:], start=1):
        state_keys = set(state.keys())
        if state_keys != keys:
            missing = keys - state_keys
            extra = state_keys - keys
            raise ValueError(
                f"State dict keys mismatch for teacher {idx}. Missing: {missing}. Extra: {extra}."
            )
    for key in states[0]:
        acc = None
        for state in states:
            acc = state[key].clone() if acc is None else acc + state[key]
        merged[key] = acc / float(len(states))
    return merged


@dataclass
class TeacherSpec:
    name: str
    config: Path
    checkpoint: Path
    dataset: Path


def parse_teacher_specs(args: argparse.Namespace) -> List[TeacherSpec]:
    if args.teacher:
        specs: List[TeacherSpec] = []
        for entry in args.teacher:
            parts = [part.strip() for part in entry.split("|")]
            if len(parts) != 4:
                raise ValueError(
                    "Each --teacher must be formatted as name|config|checkpoint|dataset."
                )
            name, config_path, ckpt_path, dataset_path = parts
            if not name:
                raise ValueError(f"Teacher spec has empty name: '{entry}'")
            specs.append(
                TeacherSpec(
                    name=name,
                    config=Path(config_path),
                    checkpoint=Path(ckpt_path),
                    dataset=Path(dataset_path),
                )
            )
        if len(specs) < 2:
            raise ValueError("At least two --teacher specs are required.")
        return specs

    missing = [
        flag
        for flag, value in (
            ("--config-a", args.config_a),
            ("--checkpoint-a", args.checkpoint_a),
            ("--dataset-a", args.dataset_a),
            ("--config-b", args.config_b),
            ("--checkpoint-b", args.checkpoint_b),
            ("--dataset-b", args.dataset_b),
        )
        if value is None
    ]
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join(missing)}")
    return [
        TeacherSpec(
            name="teacher_a",
            config=args.config_a,
            checkpoint=args.checkpoint_a,
            dataset=args.dataset_a,
        ),
        TeacherSpec(
            name="teacher_b",
            config=args.config_b,
            checkpoint=args.checkpoint_b,
            dataset=args.dataset_b,
        ),
    ]


def build_model(base_model: str, device: torch.device, precision: str, compile_model: bool) -> torch.nn.Module:
    loader = pretrained.ORB_PRETRAINED_MODELS[base_model]
    model = loader(device=device, precision=precision, compile=compile_model, train=True)
    model.eval()
    return model


def make_dataset(dataset_path: Path, dataset_name: str, model: torch.nn.Module, dtype: torch.dtype) -> AseSqliteDataset:
    target_config = property_definitions.PropertyConfig(
        node_names=["forces"],
        graph_names=["energy"],
    )
    return AseSqliteDataset(
        name=dataset_name,
        path=dataset_path,
        system_config=model.system_config,
        target_config=target_config,
        dtype=dtype,
    )


def make_training_loader(
    dataset: AseSqliteDataset,
    config: Dict,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
) -> DataLoader:
    val_fraction = float(config.get("val_fraction", 0.0))
    test_fraction = float(config.get("test_fraction", 0.0))
    split_seed = config.get("split_seed")
    seed = int(split_seed) if split_seed is not None else int(config.get("seed", 42))
    if (val_fraction > 0.0 or test_fraction > 0.0) and len(dataset) > 1:
        train_subset, _, _ = deterministic_train_val_test_split(dataset, val_fraction, test_fraction, seed)
    else:
        train_subset = dataset
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "collate_fn": ff_base.batch_graphs,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(train_subset, **loader_kwargs)


class LinearAccumulator:
    """Accumulate statistics to solve linear regression with bias."""

    def __init__(self, in_features: int, out_features: int, reg: float, device: torch.device):
        self.in_features = in_features
        self.out_features = out_features
        self.reg = reg
        self.device = device
        self.XtX = torch.zeros(in_features + 1, in_features + 1, dtype=torch.float64, device=self.device)
        self.XtY = torch.zeros(in_features + 1, out_features, dtype=torch.float64, device=self.device)
        self.count = 0

    def update(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        if inputs.numel() == 0:
            return
        X = inputs.to(device=self.device, dtype=torch.float64)
        Y = targets.to(device=self.device, dtype=torch.float64)
        n = X.shape[0]
        if n == 0:
            return
        x_sum = X.sum(dim=0)
        y_sum = Y.sum(dim=0)

        self.XtX[:-1, :-1] += X.T @ X
        self.XtX[:-1, -1] += x_sum
        self.XtX[-1, :-1] += x_sum
        self.XtX[-1, -1] += n

        self.XtY[:-1, :] += X.T @ Y
        self.XtY[-1, :] += y_sum
        self.count += n

    def solve(self, solve_device: Optional[torch.device] = None) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if self.count == 0:
            return None
        device = solve_device or self.device
        if device != self.device:
            XtX = self.XtX.to(device=device)
            XtY = self.XtY.to(device=device)
        else:
            XtX = self.XtX
            XtY = self.XtY
        reg_eye = self.reg * torch.eye(XtX.shape[0], dtype=torch.float64, device=device)
        theta = torch.linalg.solve(XtX + reg_eye, XtY)
        weight = theta[:-1].T.to(dtype=torch.float32, device="cpu")
        bias = theta[-1].to(dtype=torch.float32, device="cpu")
        return weight, bias

    def condition_number(self) -> float:
        if self.count == 0:
            return float("nan")
        reg_eye = self.reg * torch.eye(self.XtX.shape[0], dtype=torch.float64, device=self.device)
        matrix = self.XtX + reg_eye
        try:
            cond = torch.linalg.cond(matrix)
            if isinstance(cond, torch.Tensor):
                cond = cond.item()
            return float(cond)
        except RuntimeError:
            return float("inf")


def register_accumulator_hooks(
    model: torch.nn.Module,
    module_names: Iterable[str],
    linear_dims: Dict[str, Tuple[int, int]],
    accumulators: Dict[str, LinearAccumulator],
) -> List[torch.utils.hooks.RemovableHandle]:
    handles: List[torch.utils.hooks.RemovableHandle] = []
    module_name_set = set(module_names)

    for name, module in model.named_modules():
        if name not in module_name_set or not isinstance(module, torch.nn.Linear):
            continue
        in_features, out_features = linear_dims[name]
        accumulator = accumulators[name]

        def hook(
            module: torch.nn.Module,
            inputs: Tuple[torch.Tensor, ...],
            output: torch.Tensor,
            in_features: int = in_features,
            out_features: int = out_features,
            accumulator: LinearAccumulator = accumulator,
        ) -> None:
            inputs_tensor = inputs[0].detach()
            outputs_tensor = output.detach()

            if inputs_tensor.ndim > 2 or inputs_tensor.shape[-1] != in_features:
                inputs_tensor = inputs_tensor.reshape(-1, in_features)
            if outputs_tensor.ndim > 2 or outputs_tensor.shape[-1] != out_features:
                outputs_tensor = outputs_tensor.reshape(-1, out_features)

            accumulator.update(inputs_tensor, outputs_tensor)

        handles.append(module.register_forward_hook(hook))
    return handles


def accumulate_teacher(
    dataset: DataLoader,
    teacher: torch.nn.Module,
    module_names: List[str],
    linear_dims: Dict[str, Tuple[int, int]],
    device: torch.device,
    dtype: torch.dtype,
    accumulators: Dict[str, LinearAccumulator],
    limit: Optional[int],
) -> None:
    handles = register_accumulator_hooks(teacher, module_names, linear_dims, accumulators)
    processed = 0
    try:
        with torch.inference_mode():
            for batch in dataset:
                batch = batch.to(device=device, dtype=dtype)
                teacher(batch)

                processed += batch.n_node.shape[0]
                if limit is not None and processed >= limit:
                    break
    finally:
        for handle in handles:
            handle.remove()


def solve_accumulator(
    name: str,
    accumulator: LinearAccumulator,
    solve_device: torch.device,
) -> Tuple[str, Optional[Tuple[torch.Tensor, torch.Tensor]], Dict[str, object]]:
    cond = accumulator.condition_number()
    entry = {
        "module": name,
        "in_features": accumulator.in_features,
        "out_features": accumulator.out_features,
        "samples": accumulator.count,
        "regularization": accumulator.reg,
        "condition_number": cond,
        "normal_matrix_shape": list(accumulator.XtX.shape),
    }
    result = accumulator.solve(solve_device)
    if result is None:
        entry["skipped"] = True
    return name, result, entry


def reconstruct_linear_layers(
    teachers: Sequence[torch.nn.Module],
    loaders: Sequence[DataLoader],
    module_names: List[str],
    linear_dims: Dict[str, Tuple[int, int]],
    device: torch.device,
    dtype: torch.dtype,
    accumulators: Dict[str, LinearAccumulator],
    solve_device: torch.device,
    limit: Optional[int],
    solve_workers: int,
) -> Tuple[Dict[str, Tuple[torch.Tensor, torch.Tensor]], List[Dict[str, object]], float]:
    for teacher, loader in zip(teachers, loaders):
        accumulate_teacher(loader, teacher, module_names, linear_dims, device, dtype, accumulators, limit)

    if solve_workers <= 0:
        solve_workers = max(1, min(len(module_names), os.cpu_count() or 1))

    weights_biases: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    log_entries_map: Dict[str, Dict[str, object]] = {}

    solve_start = time.perf_counter()
    if solve_workers > 1 and len(module_names) > 1:
        print(f"Solving {len(module_names)} linear layers in parallel with {solve_workers} workers.")
        original_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=solve_workers) as executor:
                futures = [
                    executor.submit(solve_accumulator, name, accumulators[name], solve_device)
                    for name in module_names
                ]
                for future in concurrent.futures.as_completed(futures):
                    name, result, entry = future.result()
                    log_entries_map[name] = entry
                    if result is not None:
                        weights_biases[name] = result
        finally:
            torch.set_num_threads(original_threads)
    else:
        for name in module_names:
            _, result, entry = solve_accumulator(name, accumulators[name], solve_device)
            log_entries_map[name] = entry
            if result is not None:
                weights_biases[name] = result
    solve_seconds = time.perf_counter() - solve_start

    log_entries = [log_entries_map[name] for name in module_names]
    return weights_biases, log_entries, solve_seconds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher",
        action="append",
        default=[],
        help=(
            "Teacher spec as name|config|checkpoint|dataset (repeat to include multiple teachers). "
            "When provided, --config-a/--checkpoint-a/--dataset-a and B variants are ignored."
        ),
    )
    parser.add_argument(
        "--config-a",
        type=Path,
        default=None,
        help="(Deprecated) Config JSON for teacher A.",
    )
    parser.add_argument(
        "--checkpoint-a",
        type=Path,
        default=None,
        help="(Deprecated) Checkpoint for teacher A.",
    )
    parser.add_argument(
        "--dataset-a",
        type=Path,
        default=None,
        help="(Deprecated) Dataset for teacher A.",
    )
    parser.add_argument(
        "--config-b",
        type=Path,
        default=None,
        help="(Deprecated) Config JSON for teacher B.",
    )
    parser.add_argument(
        "--checkpoint-b",
        type=Path,
        default=None,
        help="(Deprecated) Checkpoint for teacher B.",
    )
    parser.add_argument(
        "--dataset-b",
        type=Path,
        default=None,
        help="(Deprecated) Dataset for teacher B.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination path for merged checkpoint (will be overwritten).",
    )
    parser.add_argument("--device", type=str, default="auto", help="Torch device to use.")
    parser.add_argument(
        "--accum-device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for accumulator matrices (auto uses model device).",
    )
    parser.add_argument(
        "--solve-device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for linear solve (auto uses accum-device).",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for feature extraction.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of graphs per dataset.")
    parser.add_argument("--regularization", type=float, default=1e-6, help="Diagonal regularization for linear solve.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers (defaults to max(config_a, config_b) or 0).",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="DataLoader prefetch factor when using workers.",
    )
    parser.add_argument(
        "--solve-workers",
        type=int,
        default=0,
        help="Parallel workers for linear solve (0=auto, 1=disable).",
    )
    parser.add_argument(
        "--keep-metadata",
        action="store_true",
        help="Copy metadata fields from teacher A checkpoint (optimizer, scheduler, etc.).",
    )
    parser.add_argument("--compile-model", action="store_true", help="Enable torch.compile for feature extraction.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    total_start = time.perf_counter()

    teacher_specs = parse_teacher_specs(args)
    configs = [load_json(spec.config) for spec in teacher_specs]

    base_model = configs[0]["base_model"]
    if any(cfg["base_model"] != base_model for cfg in configs[1:]):
        raise ValueError("All teachers must share the same base_model.")

    precision = configs[0].get("precision", "float32-high")
    if any(cfg.get("precision", "float32-high") != precision for cfg in configs[1:]):
        raise ValueError("All teachers must share the same precision setting.")

    device = resolve_device(args.device)
    accum_device = resolve_accum_device(args.accum_device, device)
    solve_device = resolve_solve_device(args.solve_device, accum_device)

    teachers = [
        build_model(base_model, device, precision, compile_model=args.compile_model)
        for _ in teacher_specs
    ]

    states = [extract_state_dict(spec.checkpoint) for spec in teacher_specs]
    for teacher, state in zip(teachers, states):
        teacher.load_state_dict(state)
        teacher.to(device)

    dtype = next(teachers[0].parameters()).dtype

    datasets = [
        make_dataset(spec.dataset, cfg["dataset_name"], teacher, dtype)
        for spec, cfg, teacher in zip(teacher_specs, configs, teachers)
    ]

    num_workers_default = max(int(cfg.get("num_workers", 0)) for cfg in configs)
    num_workers = num_workers_default if args.num_workers is None else args.num_workers
    pin_memory = device.type == "cuda"

    loaders = [
        make_training_loader(
            dataset,
            cfg,
            batch_size=args.batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=args.prefetch_factor,
        )
        for dataset, cfg in zip(datasets, configs)
    ]

    linear_modules = {
        name: module for name, module in teachers[0].named_modules() if isinstance(module, torch.nn.Linear)
    }
    if not linear_modules:
        raise RuntimeError("No torch.nn.Linear modules found in teacher models.")
    for idx, teacher in enumerate(teachers[1:], start=1):
        other_modules = {
            name: module for name, module in teacher.named_modules() if isinstance(module, torch.nn.Linear)
        }
        if set(other_modules.keys()) != set(linear_modules.keys()):
            raise RuntimeError(
                f"Teacher {idx} does not share the same linear modules as teacher 0."
            )
        for name, module in other_modules.items():
            dims = (module.in_features, module.out_features)
            ref_dims = (linear_modules[name].in_features, linear_modules[name].out_features)
            if dims != ref_dims:
                raise RuntimeError(
                    f"Linear module '{name}' dims mismatch for teacher {idx}: {dims} vs {ref_dims}."
                )

    shared_names = sorted(linear_modules.keys())
    linear_dims = {
        name: (linear_modules[name].in_features, linear_modules[name].out_features) for name in shared_names
    }

    accumulators = {
        name: LinearAccumulator(
            in_features=linear_dims[name][0],
            out_features=linear_dims[name][1],
            reg=args.regularization,
            device=accum_device,
        )
        for name in shared_names
    }

    merge_start = time.perf_counter()
    weights_biases, log_entries, solve_seconds = reconstruct_linear_layers(
        teachers=teachers,
        loaders=loaders,
        module_names=shared_names,
        linear_dims=linear_dims,
        device=device,
        dtype=dtype,
        accumulators=accumulators,
        solve_device=solve_device,
        limit=args.limit,
        solve_workers=args.solve_workers,
    )

    merged_state = mean_state(states)

    apply_start = time.perf_counter()
    for name, (weight, bias) in weights_biases.items():
        weight_key = f"{name}.weight"
        bias_key = f"{name}.bias"
        if weight_key in merged_state:
            merged_state[weight_key] = weight.to(merged_state[weight_key].dtype)
        if bias_key in merged_state:
            merged_state[bias_key] = bias.to(merged_state[bias_key].dtype)
    apply_seconds = time.perf_counter() - apply_start

    skipped = set(linear_modules.keys()) - set(weights_biases.keys())
    if skipped:
        print(f"Closed-form merge skipped {len(skipped)} linear layers (fallback to mean): {sorted(skipped)}")

    checkpoint: Dict[str, object] = {"model": merged_state}
    if args.keep_metadata:
        teacher_ckpt = torch.load(teacher_specs[0].checkpoint, map_location="cpu")
        for key in ["optimizer", "scheduler", "history", "best_val_loss", "epoch"]:
            if key in teacher_ckpt:
                checkpoint[key] = teacher_ckpt[key]

    merge_seconds = time.perf_counter() - merge_start

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"Saved individual closed-form merged checkpoint to {args.output}")
    print(f"Solve time (linear regression): {solve_seconds:.6f} s")
    print(f"Apply solved weights time: {apply_seconds:.6f} s")
    if log_entries:
        log_path = args.output.with_suffix(".log")
        with log_path.open("w", encoding="utf-8") as handle:
            for entry in log_entries:
                handle.write(json.dumps(entry) + "\n")
        print(f"Detailed solve log written to {log_path}")
    print(f"Closed-form individual merge compute time: {merge_seconds:.2f}s ({merge_seconds / 60:.2f} min)")
    total_seconds = time.perf_counter() - total_start
    print(f"Total merge time (end-to-end): {total_seconds:.6f} s")


if __name__ == "__main__":
    main()
