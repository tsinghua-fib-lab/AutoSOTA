"""
Evaluate fine-tuned ORB checkpoints on ASE SQLite datasets and report MAE/RMSE.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from orb_models.dataset.ase_sqlite_dataset import AseSqliteDataset
from orb_models.forcefield import base as ff_base
from orb_models.forcefield import property_definitions
from orb_models.forcefield import pretrained

try:
    from .train_orb import (
        deterministic_train_val_test_split,
        resolve_data_unit_scale,
        scale_batch_targets,
    )
except ImportError:  # pragma: no cover - fallback for direct script execution
    import sys
    sys.path.append(str(Path(__file__).resolve().parent))
    from train_orb import (  # type: ignore
        deterministic_train_val_test_split,
        resolve_data_unit_scale,
        scale_batch_targets,
    )


@dataclass
class MetricAccumulator:
    """Accumulate absolute and squared errors."""

    abs_sum: float = 0.0
    sq_sum: float = 0.0
    count: int = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        diff = pred - target
        self.abs_sum += torch.sum(torch.abs(diff)).item()
        self.sq_sum += torch.sum(diff * diff).item()
        self.count += diff.numel()

    def finalize(self) -> Dict[str, float]:
        assert self.count > 0, "No samples accumulated."
        mae = self.abs_sum / self.count
        rmse = math.sqrt(self.sq_sum / self.count)
        return {"mae": mae, "rmse": rmse}


def load_config(config_path: Path) -> Dict:
    return json.loads(config_path.read_text())


def load_checkpoint(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    if isinstance(checkpoint, dict):
        return checkpoint  # assume this is already a state dict
    raise ValueError(f"Unsupported checkpoint format at {checkpoint_path}")


def build_model(base_model: str, device: torch.device, precision: str, compile_model: bool) -> torch.nn.Module:
    loader = pretrained.ORB_PRETRAINED_MODELS[base_model]
    model = loader(device=device, precision=precision, compile=compile_model, train=True)
    model.eval()
    return model


def make_dataset(
    dataset_name: str,
    dataset_path: Path,
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


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    target_scale: float,
    limit: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    energy_metrics = MetricAccumulator()
    force_metrics = MetricAccumulator()

    model.eval()
    total_items = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device=device, dtype=dtype)
            scale_batch_targets(batch, target_scale)
            predictions = model.predict(batch)

            pred_energy = predictions["energy"]
            target_energy = batch.system_targets["energy"]
            if pred_energy.dim() > 1:
                pred_energy = pred_energy.squeeze(-1)
            if target_energy.dim() > 1:
                target_energy = target_energy.squeeze(-1)
            energy_metrics.update(pred_energy.cpu(), target_energy.cpu())

            pred_forces = predictions["forces"]
            target_forces = batch.node_targets["forces"]
            force_metrics.update(pred_forces.cpu(), target_forces.cpu())

            total_items += batch.n_node.shape[0]
            if limit is not None and total_items >= limit:
                break

    return {
        "energy": energy_metrics.finalize(),
        "forces": force_metrics.finalize(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to training config JSON.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint (.ckpt).")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Override dataset path (defaults to config['db_path']).",
    )
    parser.add_argument("--device", type=str, default="auto", help="Torch device to use (default auto).")
    parser.add_argument("--batch-size", type=int, default=8, help="Evaluation batch size.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of graphs to evaluate (for quick checks).",
    )
    parser.add_argument("--compile-model", action="store_true", help="Enable torch.compile for evaluation.")
    parser.add_argument(
        "--data-units",
        type=str,
        default=None,
        help="Override dataset units for conversion (kcal/mol or eV).",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["val", "train", "test", "all"],
        default="test",
        help="Dataset split to evaluate (train/val/test per train_orb.py).",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to write metrics (defaults to results/<checkpoint_stem>_eval.txt).",
    )
    return parser


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config)
    dataset_path = Path(args.dataset) if args.dataset else Path(config["db_path"])
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Provide --dataset to override.")

    base_model = config["base_model"]
    precision = config.get("precision", "float32-high")

    device = resolve_device(args.device)
    model = build_model(base_model, device, precision, compile_model=args.compile_model)
    state_dict = load_checkpoint(args.checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)

    dtype = next(model.parameters()).dtype
    dataset = make_dataset(config["dataset_name"], dataset_path, model, dtype)
    data_units = args.data_units or config.get("data_units", "kcal/mol")
    target_scale = resolve_data_unit_scale(data_units)
    metric_scale = 1.0 / target_scale

    split_choice = args.split
    val_fraction = float(config.get("val_fraction", 0.0))
    test_fraction = float(config.get("test_fraction", 0.0))
    split_seed = config.get("split_seed")
    seed = int(split_seed) if split_seed is not None else int(config.get("seed", 42))

    if split_choice == "all" or (val_fraction <= 0.0 and test_fraction <= 0.0) or len(dataset) < 2:
        eval_dataset = dataset
    else:
        train_subset, val_subset, test_subset = deterministic_train_val_test_split(
            dataset, val_fraction, test_fraction, seed
        )
        if split_choice == "val" and val_subset is not None:
            eval_dataset = val_subset
        elif split_choice == "test":
            if test_subset is not None:
                eval_dataset = test_subset
            elif val_subset is not None:
                eval_dataset = val_subset
            else:
                eval_dataset = dataset
        elif split_choice == "train":
            eval_dataset = train_subset
        else:
            eval_dataset = dataset

    loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=ff_base.batch_graphs,
    )

    metrics = evaluate(model, loader, device, dtype, target_scale, limit=args.limit)
    if metric_scale != 1.0:
        for values in metrics.values():
            values["mae"] *= metric_scale
            values["rmse"] *= metric_scale
    lines = [f"Evaluation results for {args.checkpoint} (units: {data_units}):"]
    for key, values in metrics.items():
        lines.append(f"  {key.capitalize()} -> MAE: {values['mae']:.6f}, RMSE: {values['rmse']:.6f}")
    print("\n".join(lines))
    save_path = args.save or Path("results") / f"{args.checkpoint.stem}_eval.txt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
