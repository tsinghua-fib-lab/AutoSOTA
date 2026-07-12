"""
Evaluate a checkpoint by swapping embeddings per sample (ethanol vs malonaldehyde).

This script only evaluates; the fine-tuning path lives in force_head_ft/switch_finetune_embeddings.py.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence

import ase.db
import torch
from torch.utils.data import DataLoader, Subset

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


def load_config(path: Path) -> Dict:
    return json.loads(path.read_text())


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def build_model(base_model: str, device: torch.device, precision: str, compile_model: bool) -> torch.nn.Module:
    loader = pretrained.ORB_PRETRAINED_MODELS[base_model]
    model = loader(device=device, precision=precision, compile=compile_model, train=True)
    return model


def load_checkpoint(path: Path) -> Mapping[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, Mapping) and "model" in checkpoint:
        return checkpoint["model"]
    if isinstance(checkpoint, Mapping):
        return checkpoint
    raise ValueError(f"Unsupported checkpoint format {path}")


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


def tensor_to_float(value: torch.Tensor | float | int) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def accumulate_metrics(
    accumulator: MutableMapping[str, float], metrics: Mapping[str, torch.Tensor | float | int], weight: float
) -> MutableMapping[str, float]:
    for key, value in metrics.items():
        try:
            accumulator[key] = accumulator.get(key, 0.0) + tensor_to_float(value) * weight
        except TypeError:
            continue
    return accumulator


def finalize_metrics(accumulator: Mapping[str, float], total_weight: float) -> Dict[str, float]:
    if total_weight <= 0:
        return {}
    return {key: val / total_weight for key, val in accumulator.items()}


def extract_source_labels(db_path: Path) -> Sequence[str]:
    db = ase.db.connect(str(db_path), serial=True, type="db")
    labels: list[str] = []
    for idx in range(len(db)):
        row = db.get(idx + 1)
        labels.append(row.data.get("source_dataset", row.data.get("dataset", "")))
    return labels


def collect_layernorm_names(model: torch.nn.Module) -> Sequence[str]:
    names = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm):
            names.append(name)
    return names


def extract_ln_params(state_dict: Mapping[str, torch.Tensor], ln_names: Sequence[str]) -> Dict[str, torch.Tensor]:
    params: Dict[str, torch.Tensor] = {}
    for name in ln_names:
        weight_key = f"{name}.weight"
        bias_key = f"{name}.bias"
        if weight_key in state_dict:
            params[weight_key] = state_dict[weight_key].clone()
        if bias_key in state_dict:
            params[bias_key] = state_dict[bias_key].clone()
    return params


def swap_embeddings_inplace(model: torch.nn.Module, label: str, embeddings: Mapping[str, torch.Tensor]) -> None:
    key = label.lower()
    if key not in embeddings:
        raise KeyError(f"No embedding provided for source label '{key}'. Available: {sorted(embeddings.keys())}")
    model.model.atom_emb.embeddings.weight.data.copy_(embeddings[key])


def resolve_original_index(dataset: torch.utils.data.Dataset, idx: int) -> int:
    current_dataset = dataset
    current_idx = idx
    while isinstance(current_dataset, Subset):
        current_idx = current_dataset.indices[current_idx]
        current_dataset = current_dataset.dataset
    return current_idx


def label_for_dataset(
    dataset: torch.utils.data.Dataset,
    idx: int,
    source_labels: Sequence[str],
) -> str:
    orig_idx = resolve_original_index(dataset, idx)
    return source_labels[orig_idx].lower()


def group_indices_by_label(
    dataset: torch.utils.data.Dataset,
    source_labels: Sequence[str],
) -> Dict[str, List[int]]:
    buckets: Dict[str, List[int]] = defaultdict(list)
    for idx in range(len(dataset)):
        label = label_for_dataset(dataset, idx, source_labels)
        buckets[label].append(idx)
    return buckets


def evaluate(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    source_labels: Sequence[str],
    embeddings: Mapping[str, torch.Tensor],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    target_scale: float,
    limit: Optional[int],
) -> Dict[str, float]:
    metrics_sum: Dict[str, float] = {}
    total_weight = 0.0
    raw_force_abs = 0.0
    raw_force_sq = 0.0
    raw_force_count = 0
    raw_energy_abs = 0.0
    raw_energy_sq = 0.0
    raw_energy_count = 0

    label_groups = group_indices_by_label(dataset, source_labels)
    eval_batch_size = max(1, batch_size)

    with torch.no_grad():
        for label, indices in label_groups.items():
            if not indices:
                continue
            subset = Subset(dataset, indices)
            loader = DataLoader(
                subset,
                batch_size=eval_batch_size,
                shuffle=False,
                collate_fn=ff_base.batch_graphs,
            )
            for batch in loader:
                swap_embeddings_inplace(model, label, embeddings)
                batch = batch.to(device=device, dtype=dtype)
                scale_batch_targets(batch, target_scale)
                output = model.loss(batch)
                graphs_in_batch = float(batch.n_node.shape[0])
                accumulate_metrics(metrics_sum, output.log, graphs_in_batch)
                metrics_sum["loss"] = metrics_sum.get("loss", 0.0) + tensor_to_float(output.loss) * graphs_in_batch
                total_weight += graphs_in_batch

                predictions = model.predict(batch)
                pred_forces = predictions["forces"]
                target_forces = batch.node_targets["forces"]
                diff = pred_forces - target_forces
                raw_force_abs += torch.sum(torch.abs(diff)).item()
                raw_force_sq += torch.sum(diff * diff).item()
                raw_force_count += diff.numel()

                if "energy" in predictions:
                    pred_energy = predictions["energy"]
                    target_energy = batch.system_targets["energy"]
                    if pred_energy.dim() > 1:
                        pred_energy = pred_energy.squeeze(-1)
                    if target_energy.dim() > 1:
                        target_energy = target_energy.squeeze(-1)
                    diff_e = pred_energy - target_energy
                    raw_energy_abs += torch.sum(torch.abs(diff_e)).item()
                    raw_energy_sq += torch.sum(diff_e * diff_e).item()
                    raw_energy_count += diff_e.numel()

                if limit is not None and total_weight >= limit:
                    break
            if limit is not None and total_weight >= limit:
                break

    metrics = finalize_metrics(metrics_sum, total_weight)
    if raw_force_count:
        metrics["raw_forces_mae"] = raw_force_abs / raw_force_count
        metrics["raw_forces_rmse"] = math.sqrt(raw_force_sq / raw_force_count)
    if raw_energy_count:
        metrics["raw_energy_mae"] = raw_energy_abs / raw_energy_count
        metrics["raw_energy_rmse"] = math.sqrt(raw_energy_sq / raw_energy_count)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Config JSON (e.g., combined_tiny2 config).")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Merged checkpoint to evaluate.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to combined_sampled.db.")
    parser.add_argument(
        "--source-checkpoint",
        action="append",
        default=None,
        help="Mapping of source label to checkpoint state (e.g., ethanol=path/to.ckpt). Repeat for each label.",
    )
    parser.add_argument(
        "--ethanol-checkpoint",
        type=Path,
        default=None,
        help="(Deprecated) Checkpoint for ethanol embeddings; prefer --source-checkpoint.",
    )
    parser.add_argument(
        "--malonaldehyde-checkpoint",
        type=Path,
        default=None,
        help="(Deprecated) Checkpoint for malonaldehyde embeddings; prefer --source-checkpoint.",
    )
    parser.add_argument("--device", type=str, default="auto", help="Torch device.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (use 1 for per-graph routing).")
    parser.add_argument("--compile-model", action="store_true", help="Enable torch.compile on backbone.")
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test", "all"],
        default="test",
        help="Evaluate only this dataset split (train/val/test per train_orb.py).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of graphs evaluated (<=0 uses full split).",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to write metrics (defaults to results/<checkpoint_stem>_switch_eval.txt).",
    )
    parser.add_argument(
        "--data-units",
        type=str,
        default=None,
        help="Override dataset units for conversion (kcal/mol or eV).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config)
    dataset_path = args.dataset
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    base_model = config["base_model"]
    precision = config.get("precision", "float32-high")
    device = resolve_device(args.device)

    model = build_model(base_model, device, precision, compile_model=args.compile_model)
    model_state = load_checkpoint(args.checkpoint)
    model.load_state_dict(model_state, strict=False)
    if config.get("force_only", False) and hasattr(model, "heads") and "stress" in model.heads:
        del model.heads["stress"]
    model.to(device)

    dtype = next(model.parameters()).dtype
    dataset = make_dataset(config["dataset_name"], dataset_path, model, dtype)
    data_units = args.data_units or config.get("data_units", "kcal/mol")
    target_scale = resolve_data_unit_scale(data_units)
    metric_scale = 1.0 / target_scale

    embed_key = "model.atom_emb.embeddings.weight"

    def parse_source_checkpoints() -> Dict[str, Path]:
        mapping: Dict[str, Path] = {}
        if args.ethanol_checkpoint:
            mapping["ethanol"] = Path(args.ethanol_checkpoint)
        if args.malonaldehyde_checkpoint:
            mapping["malonaldehyde"] = Path(args.malonaldehyde_checkpoint)
        if args.source_checkpoint:
            for entry in args.source_checkpoint:
                if "=" not in entry:
                    raise ValueError(f"--source-checkpoint expects label=path, got '{entry}'")
                label, path_str = entry.split("=", 1)
                mapping[label.strip().lower()] = Path(path_str.strip())
        if not mapping:
            raise ValueError("Provide at least one --source-checkpoint (label=path).")
        return mapping

    source_ckpts = parse_source_checkpoints()
    embeddings: Dict[str, torch.Tensor] = {}
    for label, ckpt_path in source_ckpts.items():
        state = load_checkpoint(ckpt_path)
        if embed_key not in state:
            raise KeyError(f"Checkpoint {ckpt_path} missing embedding key '{embed_key}'")
        embeddings[label] = state[embed_key].to(device)

    source_labels = extract_source_labels(dataset_path)

    val_fraction = float(config.get("val_fraction", 0.0))
    test_fraction = float(config.get("test_fraction", 0.0))
    split_seed = config.get("split_seed")
    seed = int(split_seed) if split_seed is not None else int(config.get("seed", 42))
    split_choice = args.split
    eval_dataset = dataset
    if (val_fraction > 0.0 or test_fraction > 0.0) and len(dataset) > 1:
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
        elif split_choice == "train":
            eval_dataset = train_subset

    limit = args.limit if args.limit and args.limit > 0 else None
    metrics = evaluate(
        model=model,
        dataset=eval_dataset,
        source_labels=source_labels,
        embeddings=embeddings,
        batch_size=args.batch_size,
        device=device,
        dtype=dtype,
        target_scale=target_scale,
        limit=limit,
    )
    if metric_scale != 1.0:
        for key, value in list(metrics.items()):
            if key.endswith(("_mae", "_rmse")) and any(token in key for token in ("force", "forces", "energy")):
                metrics[key] = value * metric_scale

    header = f"Embedding-switched evaluation for {args.checkpoint} (units: {data_units}):"
    lines = [header] + [f"  {key}: {metrics[key]:.6f}" for key in sorted(metrics.keys())]
    print("\n".join(lines))
    save_path = args.save or Path("results") / f"{args.checkpoint.stem}_switch_eval.txt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
