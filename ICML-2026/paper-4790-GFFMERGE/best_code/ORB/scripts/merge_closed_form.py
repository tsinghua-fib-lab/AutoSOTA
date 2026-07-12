"""
Closed-form merge of ORB checkpoints combining linear layers via regression.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
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


def build_model(base_model: str, device: torch.device, precision: str, compile_model: bool) -> torch.nn.Module:
    loader = pretrained.ORB_PRETRAINED_MODELS[base_model]
    model = loader(device=device, precision=precision, compile=compile_model, train=True)
    model.eval()
    return model


def make_dataset(
    dataset_path: Path,
    dataset_name: str,
    model: torch.nn.Module,
    dtype: torch.dtype,
) -> AseSqliteDataset:
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
) -> DataLoader:
    """Return DataLoader over the training subset defined by val_fraction/seed."""
    val_fraction = float(config.get("val_fraction", 0.0))
    test_fraction = float(config.get("test_fraction", 0.0))
    split_seed = config.get("split_seed")
    seed = int(split_seed) if split_seed is not None else int(config.get("seed", 42))
    if (val_fraction > 0.0 or test_fraction > 0.0) and len(dataset) > 1:
        train_subset, _, _ = deterministic_train_val_test_split(dataset, val_fraction, test_fraction, seed)
    else:
        train_subset = dataset
    return DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ff_base.batch_graphs,
    )

class LinearAccumulator:
    """Accumulate statistics to solve linear regression with bias."""

    def __init__(self, in_features: int, out_features: int, reg: float):
        self.in_features = in_features
        self.out_features = out_features
        self.reg = reg
        self.XtX = torch.zeros(in_features + 1, in_features + 1, dtype=torch.float64)
        self.XtY = torch.zeros(in_features + 1, out_features, dtype=torch.float64)
        self.count = 0

    def update(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        """inputs [N, in_features], targets [N, out_features]."""
        if inputs.numel() == 0:
            return
        ones = torch.ones(inputs.shape[0], 1, dtype=torch.float64)
        X = torch.cat([inputs.to(torch.float64), ones], dim=-1)
        Y = targets.to(torch.float64)
        self.XtX += X.T @ X
        self.XtY += X.T @ Y
        self.count += inputs.shape[0]

    def solve(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if self.count == 0:
            return None
        reg_eye = self.reg * torch.eye(self.XtX.shape[0], dtype=torch.float64)
        theta = torch.linalg.solve(self.XtX + reg_eye, self.XtY)
        weight = theta[:-1].T.to(torch.float32)
        bias = theta[-1].to(torch.float32)
        return weight, bias


def register_linear_hooks(
    model: torch.nn.Module, module_names: Iterable[str]
) -> Tuple[Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]], List[torch.utils.hooks.RemovableHandle]]:
    storage: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = {name: [] for name in module_names}
    handles: List[torch.utils.hooks.RemovableHandle] = []

    def make_hook(name: str):
        def hook(module: torch.nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            storage[name].append((inputs[0].detach().cpu(), output.detach().cpu()))
        return hook

    for name, module in model.named_modules():
        if name in storage and isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_hook(make_hook(name)))
    return storage, handles


def reconstruct_linear_layers(
    dataset: DataLoader,
    teachers: Sequence[torch.nn.Module],
    module_names: List[str],
    linear_dims: Dict[str, Tuple[int, int]],
    device: torch.device,
    dtype: torch.dtype,
    reg: float,
    limit: Optional[int] = None,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    accumulators = {
        name: LinearAccumulator(in_features=linear_dims[name][0], out_features=linear_dims[name][1], reg=reg)
        for name in module_names
    }

    storages: List[Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]]] = []
    handles: List[torch.utils.hooks.RemovableHandle] = []
    for teacher in teachers:
        storage, teacher_handles = register_linear_hooks(teacher, module_names)
        storages.append(storage)
        handles.extend(teacher_handles)

    processed = 0
    try:
        with torch.no_grad():
            for batch in dataset:
                batch = batch.to(device=device, dtype=dtype)
                for storage in storages:
                    for key in storage:
                        storage[key].clear()

                for teacher in teachers:
                    teacher(batch)

                for name in module_names:
                    lengths = [len(storage[name]) for storage in storages]
                    if len(set(lengths)) != 1:
                        raise RuntimeError(
                            f"Hook counts differ for module {name}: {lengths}"
                        )
                    for storage in storages:
                        for inputs, targets in storage[name]:
                            if inputs.ndim > 2:
                                inputs = inputs.reshape(inputs.shape[0], -1)
                            if targets.ndim > 2:
                                targets = targets.reshape(targets.shape[0], -1)
                            accumulators[name].update(inputs, targets)

                processed += batch.n_node.shape[0]
                if limit is not None and processed >= limit:
                    break
    finally:
        for handle in handles:
            handle.remove()

    weights_biases = {}
    for name, accumulator in accumulators.items():
        result = accumulator.solve()
        if result is not None:
            weights_biases[name] = result
    return weights_biases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to training config JSON.")
    parser.add_argument(
        "--teacher",
        action="append",
        type=Path,
        default=[],
        help="Checkpoint path for a teacher (repeat to include multiple teachers).",
    )
    parser.add_argument(
        "--teacher-a",
        type=Path,
        default=None,
        help="(Deprecated) Checkpoint path for first teacher.",
    )
    parser.add_argument(
        "--teacher-b",
        type=Path,
        default=None,
        help="(Deprecated) Checkpoint path for second teacher.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination path for merged checkpoint (will be overwritten).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Override dataset path (defaults to config['db_path']).",
    )
    parser.add_argument("--device", type=str, default="auto", help="Torch device to use.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for feature extraction.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of graphs.")
    parser.add_argument("--regularization", type=float, default=1e-6, help="Diagonal regularization for linear solve.")
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

    config = load_json(args.config)
    dataset_path = Path(args.dataset) if args.dataset else Path(config["db_path"])
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Provide --dataset to override.")

    base_model = config["base_model"]
    precision = config.get("precision", "float32-high")
    dataset_name = config["dataset_name"]

    device = resolve_device(args.device)

    if args.teacher:
        teacher_paths = list(args.teacher)
    else:
        if args.teacher_a is None or args.teacher_b is None:
            raise ValueError("Provide at least two teachers via --teacher or --teacher-a/--teacher-b.")
        teacher_paths = [args.teacher_a, args.teacher_b]
    if len(teacher_paths) < 2:
        raise ValueError("At least two teacher checkpoints are required for closed-form merge.")

    teachers = [
        build_model(base_model, device, precision, compile_model=args.compile_model)
        for _ in teacher_paths
    ]
    states = [extract_state_dict(path) for path in teacher_paths]
    for teacher, state in zip(teachers, states):
        teacher.load_state_dict(state)
        teacher.to(device)

    dtype = next(teachers[0].parameters()).dtype
    dataset = make_dataset(dataset_path, dataset_name, teachers[0], dtype)
    loader = make_training_loader(dataset, config, batch_size=args.batch_size)

    # Prepare linear module info
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

    weights_biases = reconstruct_linear_layers(
        dataset=loader,
        teachers=teachers,
        module_names=shared_names,
        linear_dims=linear_dims,
        device=device,
        dtype=dtype,
        reg=args.regularization,
        limit=args.limit,
    )

    merged_state = mean_state(states)

    for name, (weight, bias) in weights_biases.items():
        weight_key = f"{name}.weight"
        bias_key = f"{name}.bias"
        if weight_key in merged_state:
            merged_state[weight_key] = weight.to(merged_state[weight_key].dtype)
        if bias_key in merged_state:
            merged_state[bias_key] = bias.to(merged_state[bias_key].dtype)

    skipped = set(linear_modules.keys()) - set(weights_biases.keys())
    if skipped:
        print(f"Closed-form merge skipped {len(skipped)} linear layers (fallback to mean): {sorted(skipped)}")

    checkpoint: Dict[str, object] = {"model": merged_state}
    if args.keep_metadata:
        teacher_ckpt = torch.load(teacher_paths[0], map_location="cpu")
        for key in ["optimizer", "scheduler", "history", "best_val_loss", "epoch"]:
            if key in teacher_ckpt:
                checkpoint[key] = teacher_ckpt[key]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"Saved closed-form merged checkpoint to {args.output}")


if __name__ == "__main__":
    main()
