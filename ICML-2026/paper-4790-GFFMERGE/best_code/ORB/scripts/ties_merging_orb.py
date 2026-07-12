"""
TIES-Merging for ORB Models

Implements TIES (TrIm, Elect Sign & Merge) for merging multiple fine-tuned ORB models.
Reference: https://arxiv.org/abs/2306.01708
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

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


def is_param_key(key: str) -> bool:
    """Skip non-parameter buffers like data statistics."""
    if not isinstance(key, str):
        return False
    lowered = key.lower()
    for prefix in ("datamean", "datastd", "data_mean", "data_std"):
        if lowered.startswith(prefix):
            return False
    return True


class TaskVector:
    """Represents the difference between a fine-tuned model and pretrained model."""

    def __init__(self, pretrained_state: Dict, finetuned_state: Dict):
        self.task_vector = {}

        pretrained_keys = set(pretrained_state.keys())
        finetuned_keys = set(finetuned_state.keys())
        common_keys = pretrained_keys & finetuned_keys

        if len(common_keys) == 0:
            raise ValueError(
                f"No common keys found between pretrained and fine-tuned models!"
            )

        for key in pretrained_state.keys():
            if isinstance(pretrained_state[key], torch.Tensor) and key in finetuned_state:
                if isinstance(finetuned_state[key], torch.Tensor):
                    if not is_param_key(key):
                        continue
                    self.task_vector[key] = finetuned_state[key] - pretrained_state[key]


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
    """Extract model state dict from ORB checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    return state_dict


def normalize_state_dict_keys(
    state_dict: Mapping[str, torch.Tensor],
    reference_keys: set,
) -> Dict[str, torch.Tensor]:
    """Normalize state dict keys to match reference model's key format.

    Handles cases where checkpoint keys have/don't have 'model.' prefix
    but the model expects the opposite format.
    """
    state_keys = set(state_dict.keys())

    # Check if keys already match
    if state_keys == reference_keys:
        return dict(state_dict)

    # Check if adding 'model.' prefix would help
    prefixed = {f"model.{k}": v for k, v in state_dict.items()}
    if set(prefixed.keys()) == reference_keys:
        return prefixed

    # Check if removing 'model.' prefix would help
    unprefixed = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            unprefixed[k[6:]] = v
        else:
            unprefixed[k] = v
    if set(unprefixed.keys()) == reference_keys:
        return unprefixed

    # Return original if no transformation helps (will show warnings later)
    return dict(state_dict)


def build_model(base_model: str, device: torch.device, precision: str, compile_model: bool) -> torch.nn.Module:
    """Build an ORB model from pretrained weights."""
    loader = pretrained.ORB_PRETRAINED_MODELS[base_model]
    model = loader(device=device, precision=precision, compile=compile_model, train=True)
    model.eval()
    return model


def make_dataset(dataset_path: Path, dataset_name: str, model: torch.nn.Module, dtype: torch.dtype) -> AseSqliteDataset:
    """Create an ORB dataset for evaluation."""
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


def make_eval_loader(
    dataset: AseSqliteDataset,
    config: Dict,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    use_val_split: bool = True,
) -> DataLoader:
    """Create a DataLoader for evaluation (validation split)."""
    val_fraction = float(config.get("val_fraction", 0.1))
    test_fraction = float(config.get("test_fraction", 0.0))
    split_seed = config.get("split_seed")
    seed = int(split_seed) if split_seed is not None else int(config.get("seed", 42))

    if use_val_split and val_fraction > 0.0 and len(dataset) > 1:
        _, val_subset, _ = deterministic_train_val_test_split(dataset, val_fraction, test_fraction, seed)
        if val_subset is None:
            val_subset = dataset
    else:
        val_subset = dataset

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
    return DataLoader(val_subset, **loader_kwargs)


def trim_task_vectors(
    task_vectors: List[TaskVector],
    density: float
) -> List[TaskVector]:
    """TRIM step: Keep only top-k% of parameters by magnitude."""
    trimmed_vectors = []

    for task_vector in task_vectors:
        trimmed = TaskVector.__new__(TaskVector)
        trimmed.task_vector = {}

        for key, values in task_vector.task_vector.items():
            # Compute absolute values
            abs_values = torch.abs(values)

            # Determine threshold for top-k%
            num_params = abs_values.numel()
            k = max(1, int(num_params * density))

            # Get threshold value (kth largest)
            threshold = torch.topk(abs_values.flatten(), k)[0][-1]

            # Create mask for values above threshold
            mask = abs_values >= threshold

            # Apply mask - keep only top values, zero out rest
            trimmed_values = values * mask.float()
            trimmed.task_vector[key] = trimmed_values

        trimmed_vectors.append(trimmed)

    return trimmed_vectors


def elect_sign(
    task_vectors: List[TaskVector]
) -> Dict[str, torch.Tensor]:
    """ELECT SIGN step: Choose sign based on total magnitude."""
    if not task_vectors:
        return {}

    # Get all keys
    all_keys = set()
    for tv in task_vectors:
        all_keys.update(tv.task_vector.keys())

    elected_signs = {}

    for key in all_keys:
        # Collect all values for this parameter across task vectors
        param_shape = None
        positive_mass = None
        negative_mass = None

        for tv in task_vectors:
            if key not in tv.task_vector:
                continue

            values = tv.task_vector[key]

            if param_shape is None:
                param_shape = values.shape
                positive_mass = torch.zeros_like(values)
                negative_mass = torch.zeros_like(values)

            # Accumulate magnitude of positive and negative values
            positive_mass += torch.where(values > 0, values, torch.zeros_like(values))
            negative_mass += torch.where(values < 0, -values, torch.zeros_like(values))

        # Elect sign based on which has greater total magnitude
        sign = torch.zeros_like(positive_mass)
        sign = torch.where(positive_mass > negative_mass, torch.ones_like(sign), sign)
        sign = torch.where(negative_mass > positive_mass, -torch.ones_like(sign), sign)

        elected_signs[key] = sign

    return elected_signs


def disjoint_merge(
    task_vectors: List[TaskVector],
    elected_signs: Dict[str, torch.Tensor],
    lambda_scale: float = 1.0
) -> Dict[str, torch.Tensor]:
    """DISJOINT MERGE step: Merge values that agree with elected sign."""
    merged_vector = {}

    for key in elected_signs.keys():
        sign = elected_signs[key]

        # Collect values that agree with elected sign
        aligned_values = []

        for tv in task_vectors:
            if key not in tv.task_vector:
                continue

            values = tv.task_vector[key]

            # Keep only values that have the same sign as elected sign
            mask = (sign * values) > 0

            # Zero out disagreeing values
            aligned = values * mask.float()
            aligned_values.append(aligned)

        if not aligned_values:
            merged_vector[key] = torch.zeros_like(sign)
            continue

        # Stack and compute mean, handling zeros appropriately
        stacked = torch.stack(aligned_values)

        # Count non-zero elements for each position
        non_zero_mask = stacked != 0
        num_non_zero = non_zero_mask.sum(dim=0).float()

        # Avoid division by zero
        num_non_zero = torch.where(num_non_zero == 0, torch.ones_like(num_non_zero), num_non_zero)

        # Sum and divide by count of non-zero values
        merged = stacked.sum(dim=0) / num_non_zero

        # Apply lambda scaling
        merged_vector[key] = lambda_scale * merged

    return merged_vector


def apply_ties_merging(
    pretrained_state: Dict,
    task_vectors: List[TaskVector],
    density: float = 0.2,
    lambda_scale: float = 1.0
) -> Dict:
    """Apply full TIES-Merging algorithm."""
    print(f"\nApplying TIES-Merging:")
    print(f"  Density (top-k%): {density*100:.1f}%")
    print(f"  Lambda scale: {lambda_scale}")

    # Step 1: TRIM
    print("  Step 1/3: Trimming redundant parameters...")
    trimmed_vectors = trim_task_vectors(task_vectors, density)

    # Step 2: ELECT SIGN
    print("  Step 2/3: Electing parameter signs...")
    elected_signs = elect_sign(trimmed_vectors)

    # Step 3: DISJOINT MERGE
    print("  Step 3/3: Merging with disjoint mean...")
    merged_task_vector = disjoint_merge(trimmed_vectors, elected_signs, lambda_scale)

    # Add to pretrained model
    merged_state = copy.deepcopy(pretrained_state)
    for key in merged_state.keys():
        if key in merged_task_vector:
            merged_state[key] = pretrained_state[key] + merged_task_vector[key]

    return merged_state


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    target_scale: float,
    force_only: bool = False,
) -> Dict[str, float]:
    """Evaluate model on a dataset and return metrics.

    Args:
        force_only: If True, only compute force metrics (for force-only models).
    """
    model.eval()

    total_energy_abs_error = 0.0
    total_force_abs_error = 0.0
    total_samples = 0
    total_force_components = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device=device, dtype=dtype)
            scale_batch_targets(batch, target_scale)

            # Forward pass using predict() which returns a dict
            predictions = model.predict(batch)

            # Get force predictions and targets
            pred_forces = predictions["forces"]
            target_forces = batch.node_targets["forces"]

            # Compute MAE for forces
            force_abs_error = torch.sum(torch.abs(pred_forces - target_forces)).item()
            total_force_abs_error += force_abs_error
            total_force_components += target_forces.numel()

            # Compute energy metrics only if not force_only
            if not force_only:
                pred_energy = predictions.get("energy")
                target_energy = batch.system_targets.get("energy")

                if pred_energy is not None and target_energy is not None:
                    # Handle dimension mismatches
                    if pred_energy.dim() > 1:
                        pred_energy = pred_energy.squeeze(-1)
                    if target_energy.dim() > 1:
                        target_energy = target_energy.squeeze(-1)

                    energy_abs_error = torch.sum(torch.abs(pred_energy - target_energy)).item()
                    total_energy_abs_error += energy_abs_error
                    total_samples += pred_energy.shape[0]

    metrics = {}
    if total_force_components > 0:
        metrics["val_Force_MAE"] = total_force_abs_error / total_force_components

    if not force_only and total_samples > 0:
        metrics["val_Energy_MAE"] = total_energy_abs_error / total_samples
        metrics["val_Total_Loss"] = metrics["val_Energy_MAE"] + metrics.get("val_Force_MAE", 0.0)
    else:
        # For force-only models, total loss is just force MAE
        metrics["val_Total_Loss"] = metrics.get("val_Force_MAE", float('inf'))

    metric_scale = 1.0 / target_scale if target_scale != 0 else 1.0
    if metric_scale != 1.0:
        if "val_Force_MAE" in metrics:
            metrics["val_Force_MAE"] *= metric_scale
        if "val_Energy_MAE" in metrics:
            metrics["val_Energy_MAE"] *= metric_scale
        if force_only:
            metrics["val_Total_Loss"] = metrics.get("val_Force_MAE", float("inf"))
        else:
            metrics["val_Total_Loss"] = metrics.get("val_Energy_MAE", 0.0) + metrics.get("val_Force_MAE", 0.0)

    return metrics


def grid_search_ties(
    pretrained_state: Dict,
    task_vectors: List[TaskVector],
    model: torch.nn.Module,
    eval_loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    density_values: List[float],
    lambda_values: List[float],
    metric_key: str,
    target_scale: float,
    force_only: bool = False,
) -> Tuple[float, float, Dict[str, float], List[dict]]:
    """Perform grid search to find optimal density and lambda values."""
    all_combinations = list(itertools.product(density_values, lambda_values))

    print(f"\nGrid Search Configuration:")
    print(f"  Density values: {density_values}")
    print(f"  Lambda values: {lambda_values}")
    print(f"  Total combinations: {len(all_combinations)}")
    print(f"  Optimizing for: {metric_key} (lower is better)")
    if force_only:
        print(f"  Mode: Force-only (energy metrics disabled)")

    best_density = None
    best_lambda = None
    best_metric_value = float('inf')
    best_metrics = {}
    all_results = []

    print(f"\n{'='*70}")
    print("Starting TIES-Merging grid search...")
    print(f"{'='*70}")

    for i, (density, lambda_scale) in enumerate(all_combinations):
        # Apply TIES-Merging with current parameters
        merged_state = apply_ties_merging(
            pretrained_state,
            task_vectors,
            density=density,
            lambda_scale=lambda_scale
        )

        # Load merged state into model
        model.load_state_dict(merged_state)
        model.to(device)

        # Evaluate
        try:
            metrics = evaluate_model(
                model,
                eval_loader,
                device,
                dtype,
                target_scale=target_scale,
                force_only=force_only,
            )
            metric_value = metrics.get(metric_key, float('inf'))

            result = {
                "density": density,
                "lambda": lambda_scale,
                "metrics": metrics,
                "metric_value": metric_value,
            }
            all_results.append(result)

            is_best = metric_value < best_metric_value
            if is_best:
                best_density = density
                best_lambda = lambda_scale
                best_metric_value = metric_value
                best_metrics = metrics

            best_marker = " *BEST*" if is_best else ""
            print(f"  [{i+1:3d}/{len(all_combinations)}] density={density:.2f}, λ={lambda_scale:.2f} -> {metric_key}={metric_value:.6f}{best_marker}")

        except Exception as e:
            print(f"  [{i+1:3d}/{len(all_combinations)}] density={density:.2f}, λ={lambda_scale:.2f} -> ERROR: {e}")
            all_results.append({
                "density": density,
                "lambda": lambda_scale,
                "error": str(e),
            })

    print(f"\n{'='*70}")
    print("Grid Search Complete!")
    print(f"{'='*70}")
    print(f"Best density: {best_density}")
    print(f"Best lambda: {best_lambda}")
    print(f"Best {metric_key}: {best_metric_value:.6f}")
    print(f"All best metrics:")
    for key, value in sorted(best_metrics.items()):
        print(f"  {key}: {value:.6f}")

    return best_density, best_lambda, best_metrics, all_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        required=True,
        help="Path to pretrained/base model checkpoint.",
    )

    parser.add_argument(
        "--checkpoint",
        action="append",
        type=Path,
        required=True,
        help="Fine-tuned model checkpoint paths (repeat for each model).",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for merged checkpoint.",
    )

    parser.add_argument(
        "--base-model",
        type=str,
        default="orb-v2",
        help="Base model name from ORB_PRETRAINED_MODELS (default: orb-v2).",
    )

    parser.add_argument(
        "--precision",
        type=str,
        default="float32-high",
        help="Model precision (default: float32-high).",
    )

    # TIES-Merging specific parameters
    ties_group = parser.add_argument_group("TIES-Merging Parameters")

    ties_group.add_argument(
        "--density",
        type=float,
        default=None,
        help="Density parameter for trimming (fraction of params to keep, e.g., 0.2 = top 20%%). "
             "Not required if using --grid-search.",
    )

    ties_group.add_argument(
        "--lambda",
        dest="lambda_scale",
        type=float,
        default=None,
        help="Scaling factor for merged task vector (typical range: 0.5-1.5). "
             "Not required if using --grid-search.",
    )

    # Grid search arguments
    grid_group = parser.add_argument_group("Grid Search Options")

    grid_group.add_argument(
        "--grid-search",
        action="store_true",
        help="Enable grid search to find optimal density and lambda values.",
    )

    grid_group.add_argument(
        "--val-config",
        type=Path,
        help="Config JSON for validation evaluation (required for grid search).",
    )

    grid_group.add_argument(
        "--val-dataset",
        type=Path,
        help="Dataset path for validation (required for grid search).",
    )

    grid_group.add_argument(
        "--density-values",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3],
        help="Density values to try in grid search (default: 0.1 0.2 0.3).",
    )

    grid_group.add_argument(
        "--lambda-values",
        type=float,
        nargs="+",
        default=[0.5, 0.8, 1.0],
        help="Lambda values to try in grid search (default: 0.5 0.8 1.0).",
    )

    grid_group.add_argument(
        "--metric",
        type=str,
        default="val_Total_Loss",
        help="Metric to optimize (lower is better). Default: val_Total_Loss",
    )

    grid_group.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation (default: 8).",
    )

    grid_group.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (default: 0).",
    )

    grid_group.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="DataLoader prefetch factor when using workers.",
    )

    grid_group.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device for evaluation (default: auto).",
    )

    grid_group.add_argument(
        "--save-grid-results",
        type=Path,
        default=None,
        help="Path to save all grid search results as JSON.",
    )

    grid_group.add_argument(
        "--force-only",
        action="store_true",
        help="Evaluate using only force MAE (for force-only models). "
             "When set, val_Total_Loss equals val_Force_MAE.",
    )

    parser.add_argument(
        "--keep-metadata",
        action="store_true",
        help="Copy metadata fields from the first fine-tuned checkpoint (optimizer, scheduler, etc.).",
    )

    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="Enable torch.compile for model.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    total_start = time.perf_counter()

    # Validate arguments based on mode
    if args.grid_search:
        if args.val_config is None or args.val_dataset is None:
            raise ValueError("--val-config and --val-dataset are required when using --grid-search")
        if args.density is not None or args.lambda_scale is not None:
            print("Note: --density and --lambda values will be ignored when using --grid-search")
    else:
        if args.density is None or args.lambda_scale is None:
            raise ValueError(
                "Either provide --density and --lambda values, "
                "or use --grid-search to find optimal values"
            )

    print("TIES-Merging for ORB Models")
    print("=" * 70)
    print(f"Pretrained model: {args.pretrained_checkpoint}")
    print(f"Base model: {args.base_model}")
    print(f"Fine-tuned models ({len(args.checkpoint)}):")
    for i, ckpt in enumerate(args.checkpoint, 1):
        print(f"  [{i}] {ckpt}")

    if args.grid_search:
        print(f"\nMode: Grid Search")
        print(f"  Density values to try: {args.density_values}")
        print(f"  Lambda values to try: {args.lambda_values}")
        print(f"  Validation config: {args.val_config}")
        print(f"  Validation dataset: {args.val_dataset}")
        print(f"  Optimization metric: {args.metric}")
    else:
        print(f"\nMode: Manual Parameters")
        print(f"  Density: {args.density}")
        print(f"  Lambda: {args.lambda_scale}")

    print(f"\nOutput: {args.output}")

    device = resolve_device(args.device)
    print(f"\nUsing device: {device}")

    # Build base model to get architecture
    print("\nBuilding base model...")
    model = build_model(args.base_model, device, args.precision, args.compile_model)
    dtype = next(model.parameters()).dtype

    # Get model's expected key format
    model_keys = set(model.state_dict().keys())

    # Load pretrained model state
    print("Loading pretrained model...")
    raw_pretrained_state = extract_state_dict(args.pretrained_checkpoint)
    pretrained_state = normalize_state_dict_keys(raw_pretrained_state, model_keys)
    print(f"  Loaded {len(pretrained_state)} keys")

    # Verify keys match between pretrained checkpoint and model
    pretrained_keys = set(pretrained_state.keys())
    if model_keys != pretrained_keys:
        missing_in_pretrained = model_keys - pretrained_keys
        extra_in_pretrained = pretrained_keys - model_keys
        if missing_in_pretrained:
            print(f"  ERROR: {len(missing_in_pretrained)} keys in model but not in pretrained checkpoint")
            print(f"    First few: {list(missing_in_pretrained)[:3]}")
        if extra_in_pretrained:
            print(f"  ERROR: {len(extra_in_pretrained)} keys in pretrained but not in model")
            print(f"    First few: {list(extra_in_pretrained)[:3]}")

        # Calculate overlap to determine if this is a complete mismatch
        overlap = len(model_keys & pretrained_keys)
        total = len(model_keys | pretrained_keys)
        overlap_pct = overlap / total * 100 if total > 0 else 0

        print(f"\n  Key overlap: {overlap}/{total} ({overlap_pct:.1f}%)")
        print(f"  Model expects {len(model_keys)} keys, checkpoint has {len(pretrained_keys)} keys")
        print(f"\n  This likely means --base-model ({args.base_model}) does not match the checkpoint.")
        print(f"  Please verify you're using the correct base model for your checkpoints.")
        raise ValueError(
            f"Pretrained checkpoint keys don't match model. "
            f"Expected {len(model_keys)} keys, got {len(pretrained_keys)}. "
            f"Check that --base-model matches your checkpoints."
        )

    # Compute task vectors
    print("\nComputing task vectors...")
    task_vectors = []
    all_checkpoints = []

    for i, ckpt_path in enumerate(args.checkpoint, 1):
        print(f"  Task vector {i}/{len(args.checkpoint)}: {ckpt_path.name}")

        full_checkpoint = torch.load(ckpt_path, map_location="cpu")
        all_checkpoints.append(full_checkpoint)

        raw_finetuned_state = extract_state_dict(ckpt_path)
        finetuned_state = normalize_state_dict_keys(raw_finetuned_state, model_keys)

        task_vector = TaskVector(pretrained_state, finetuned_state)
        task_vectors.append(task_vector)

        # Diagnostics
        total_norm = 0.0
        num_params = 0
        for key, value in task_vector.task_vector.items():
            total_norm += (value ** 2).sum().item()
            num_params += value.numel()
        total_norm = total_norm ** 0.5

        print(f"    Matched {len(task_vector.task_vector)} parameters ({num_params:,} values)")
        print(f"    Task vector L2 norm: {total_norm:.4f}")

    # Determine parameters (either from grid search or manual)
    if args.grid_search:
        val_config = load_json(args.val_config)
        target_scale = resolve_data_unit_scale(val_config.get("data_units", "kcal/mol"))

        # Build evaluation dataset and loader
        print("\nPreparing validation data...")
        val_dataset = make_dataset(
            args.val_dataset,
            val_config.get("dataset_name", "validation"),
            model,
            dtype,
        )

        pin_memory = device.type == "cuda"
        eval_loader = make_eval_loader(
            val_dataset,
            val_config,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            prefetch_factor=args.prefetch_factor,
            use_val_split=True,
        )

        best_density, best_lambda, best_metrics, all_results = grid_search_ties(
            pretrained_state=pretrained_state,
            task_vectors=task_vectors,
            model=model,
            eval_loader=eval_loader,
            device=device,
            dtype=dtype,
            density_values=args.density_values,
            lambda_values=args.lambda_values,
            metric_key=args.metric,
            target_scale=target_scale,
            force_only=args.force_only,
        )

        if args.save_grid_results:
            args.save_grid_results.parent.mkdir(parents=True, exist_ok=True)
            with open(args.save_grid_results, 'w') as f:
                json.dump({
                    "best_density": best_density,
                    "best_lambda": best_lambda,
                    "best_metrics": best_metrics,
                    "all_results": all_results,
                    "density_values_tried": args.density_values,
                    "lambda_values_tried": args.lambda_values,
                    "metric": args.metric,
                }, f, indent=2)
            print(f"\nGrid search results saved to: {args.save_grid_results}")

        density = best_density
        lambda_scale = best_lambda
    else:
        density = args.density
        lambda_scale = args.lambda_scale

    # Apply TIES-Merging with final parameters
    print(f"\nApplying TIES-Merging with density={density}, lambda={lambda_scale}")
    merged_state = apply_ties_merging(
        pretrained_state,
        task_vectors,
        density=density,
        lambda_scale=lambda_scale
    )

    # Prepare checkpoint for saving
    print("\nPreparing merged checkpoint...")
    checkpoint: Dict[str, object] = {"model": merged_state}

    if args.keep_metadata and all_checkpoints:
        print("Copying metadata from first fine-tuned checkpoint...")
        for key in ["optimizer", "scheduler", "history", "best_val_loss", "epoch"]:
            if key in all_checkpoints[0]:
                checkpoint[key] = all_checkpoints[0][key]

    # Store TIES-Merging info in checkpoint metadata
    checkpoint["ties_merging_info"] = {
        "density": density,
        "lambda": lambda_scale,
        "checkpoints": [str(p) for p in args.checkpoint],
        "base_model": args.base_model,
    }

    if args.grid_search:
        checkpoint["ties_merging_info"]["grid_search_metric"] = args.metric

    # Save merged checkpoint
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)

    total_seconds = time.perf_counter() - total_start

    print(f"\n{'='*70}")
    print(f"Merged checkpoint saved: {args.output}")
    print(f"{'='*70}")
    print(f"\nFinal TIES-Merging parameters:")
    print(f"  Density: {density}")
    print(f"  Lambda: {lambda_scale}")
    print(f"\nModels merged:")
    for i, ckpt in enumerate(args.checkpoint, 1):
        print(f"  [{i}] {ckpt.name}")
    print(f"\nTotal merge time: {total_seconds:.2f} s")


if __name__ == "__main__":
    main()
