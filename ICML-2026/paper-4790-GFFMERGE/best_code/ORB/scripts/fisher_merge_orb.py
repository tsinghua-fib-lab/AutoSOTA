"""
Fisher-weighted model merging for ORB force-field models.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from orb_models.forcefield import pretrained
from orb_models.forcefield import base as ff_base
from orb_models.dataset.ase_sqlite_dataset import AseSqliteDataset
from orb_models.forcefield import property_definitions

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

ORB_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--base-model",
        type=str,
        default="orb-v2",
        choices=list(pretrained.ORB_PRETRAINED_MODELS.keys()),
        help="Base ORB model architecture.",
    )

    parser.add_argument(
        "--checkpoint",
        action="append",
        type=Path,
        required=True,
        help="Fine-tuned checkpoint to merge (repeatable).",
    )

    parser.add_argument(
        "--db-path",
        type=Path,
        required=True,
        help="ASE SQLite database path for Fisher approximation.",
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        default="fisher_data",
        help="Name identifier for the dataset.",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional training config JSON to reuse split_seed/val_fraction/test_fraction.",
    )

    parser.add_argument(
        "--output-ckpt",
        type=Path,
        required=True,
        help="Output merged checkpoint path.",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Number of samples to use for Fisher approximation.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for Fisher computation.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device for computation.",
    )

    parser.add_argument(
        "--precision",
        type=str,
        default="float32-high",
        choices=["float32-high", "float32-highest", "float64"],
        help="Floating-point precision.",
    )

    parser.add_argument(
        "--data-units",
        type=str,
        default="kcal/mol",
        help="Units for dataset targets (kcal/mol or eV).",
    )

    parser.add_argument(
        "--fisher-floor",
        type=float,
        default=1e-6,
        help="Minimum Fisher value per-parameter to avoid numerical issues.",
    )

    parser.add_argument(
        "--normalize-fishers",
        action="store_true",
        help="Normalize each model's Fisher so its global L2 norm is 1.",
    )

    parser.add_argument(
        "--favor-target-model",
        action="store_true",
        help=(
            "When all Fishers for a parameter are below fisher_floor, "
            "fallback to the first model's parameter instead of averaging."
        ),
    )

    parser.add_argument(
        "--force-only",
        action="store_true",
        help="Use force-only mode (models fine-tuned without energy head).",
    )

    parser.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help="Optional JSONL log path for diagnostics.",
    )

    return parser


# ---------------------------------------------------------------------------
# Device and dtype utilities
# ---------------------------------------------------------------------------


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def get_dtype(precision: str) -> torch.dtype:
    if precision == "float64":
        return torch.float64
    return torch.float32


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------


def load_base_model(
    base_model: str,
    device: torch.device,
    precision: str,
    force_only: bool = False,
) -> torch.nn.Module:
    """Load ORB pretrained backbone."""
    loader = pretrained.ORB_PRETRAINED_MODELS[base_model]
    model = loader(device=device, precision=precision, compile=False, train=True)

    if force_only:
        # Remove energy and stress heads for force-only models
        if "energy" in model.heads:
            del model.heads["energy"]
        if "stress" in model.heads:
            del model.heads["stress"]
        model.loss_weights.pop("energy", None)
        model.loss_weights.pop("stress", None)

    return model


def extract_state_dict(checkpoint_path: Path) -> Mapping[str, torch.Tensor]:
    """Extract state dict from ORB checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def load_model_with_state(
    base_model: str,
    state_dict: Mapping[str, torch.Tensor],
    device: torch.device,
    precision: str,
    force_only: bool = False,
) -> torch.nn.Module:
    """Load ORB model and apply the given state dict."""
    model = load_base_model(base_model, device, precision, force_only)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------


def build_dataset(
    db_path: Path,
    dataset_name: str,
    model: torch.nn.Module,
    dtype: torch.dtype,
    force_only: bool = False,
) -> AseSqliteDataset:
    """Build ASE SQLite dataset for Fisher computation."""
    if force_only:
        target_config = property_definitions.PropertyConfig(
            node_names=["forces"],
            graph_names=[],
        )
    else:
        target_config = property_definitions.PropertyConfig(
            node_names=["forces"],
            graph_names=["energy"],
        )

    dataset = AseSqliteDataset(
        name=dataset_name,
        path=db_path,
        system_config=model.system_config,
        target_config=target_config,
        dtype=dtype,
    )
    return dataset


def build_dataloader(
    dataset: AseSqliteDataset,
    batch_size: int,
    num_samples: int,
    num_workers: int,
    subset_indices: Optional[Sequence[int]] = None,
) -> DataLoader:
    """Build DataLoader with limited samples."""
    if subset_indices is not None:
        indices = list(subset_indices)
        if num_samples < len(indices):
            indices = indices[:num_samples]
        subset = torch.utils.data.Subset(dataset, indices)
    else:
        total_samples = len(dataset)
        if num_samples < total_samples:
            indices = list(range(num_samples))
            subset = torch.utils.data.Subset(dataset, indices)
        else:
            subset = dataset

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=ff_base.batch_graphs,
        num_workers=num_workers,
    )
    return loader


def resolve_train_subset_indices(
    config_path: Optional[Path],
    dataset: AseSqliteDataset,
) -> Optional[Sequence[int]]:
    if config_path is None:
        return None
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    val_fraction = float(config.get("val_fraction", 0.0) or 0.0)
    test_fraction = float(config.get("test_fraction", 0.0) or 0.0)
    split_seed = config.get("split_seed")
    seed = int(split_seed) if split_seed is not None else int(config.get("seed", 42))
    if val_fraction <= 0.0 and test_fraction <= 0.0:
        return None
    train_subset, _, _ = deterministic_train_val_test_split(
        dataset,
        val_fraction,
        test_fraction,
        seed,
    )
    indices = getattr(train_subset, "indices", None)
    if indices is None:
        return list(range(len(train_subset)))
    return list(indices)


# ---------------------------------------------------------------------------
# Fisher computation
# ---------------------------------------------------------------------------


def compute_fisher_diagonal(
    model: torch.nn.Module,
    dataloader: DataLoader,
    max_batches: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    target_scale: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Approximate diagonal Fisher using squared gradients of the loss.
    F_theta ≈ E[(∂L/∂θ)^2] over the given dataset.
    """
    model.train()
    fisher: Dict[str, torch.Tensor] = {}
    n_steps = 0

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        # Move batch to device
        batch = batch.to(device=device, dtype=dtype)
        scale_batch_targets(batch, target_scale)

        # Forward pass and compute loss
        output = model.loss(batch)
        loss = output.loss

        # Zero gradients and backprop
        model.zero_grad(set_to_none=True)
        loss.backward()

        # Accumulate squared gradients (diagonal Fisher)
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            g2 = param.grad.detach() ** 2
            if name not in fisher:
                fisher[name] = g2.clone().cpu()
            else:
                fisher[name] += g2.cpu()

        n_steps += 1

    if n_steps == 0:
        raise RuntimeError("No batches processed while computing Fisher.")

    # Average over steps
    for name in fisher:
        fisher[name] /= float(n_steps)

    return fisher


def normalize_fisher_l2(fisher: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Normalize Fisher so its global L2 norm is 1."""
    total = 0.0
    for v in fisher.values():
        total += float((v ** 2).sum())
    norm = float(total) ** 0.5
    if norm < 1e-12:
        return fisher
    scale = 1.0 / norm
    return {k: v * scale for k, v in fisher.items()}


# ---------------------------------------------------------------------------
# Fisher merging
# ---------------------------------------------------------------------------


def fisher_merge_states(
    finetuned_states: List[Dict[str, torch.Tensor]],
    fishers: List[Dict[str, torch.Tensor]],
    fisher_floor: float = 1e-6,
    favor_target_model: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Merge multiple fine-tuned state dicts using Fisher-weighted averaging.

    For each parameter θ, the merged value is:
        θ_merged = Σ(F_i * θ_i) / Σ(F_i)

    where F_i is the diagonal Fisher for model i and θ_i is the parameter value.
    """
    num_models = len(finetuned_states)
    if num_models == 0:
        return {}

    # Use key set from first fine-tuned model
    base_keys = list(finetuned_states[0].keys())
    merged: Dict[str, torch.Tensor] = {}

    for key in base_keys:
        thetas = []
        fisher_vals = []

        for i in range(num_models):
            state_i = finetuned_states[i]
            if key not in state_i:
                # Skip this key if any model is missing it
                break

            theta_i = state_i[key]
            F_i = fishers[i].get(key, torch.zeros_like(theta_i, device="cpu"))
            thetas.append(theta_i)
            fisher_vals.append(F_i)

        if len(thetas) != num_models:
            # At least one model was missing this key, skip merging it
            continue

        # Stack Fishers and params on CPU
        theta_stack = torch.stack([t.cpu().float() for t in thetas], dim=0)
        fisher_stack = torch.stack([f.float() for f in fisher_vals], dim=0)

        fisher_sum = fisher_stack.sum(dim=0)
        good_mask = fisher_sum > fisher_floor

        merged_param = torch.empty_like(theta_stack[0])

        if good_mask.any():
            weighted = (fisher_stack * theta_stack).sum(dim=0)
            merged_param[good_mask] = weighted[good_mask] / fisher_sum[good_mask]

        if (~good_mask).any():
            if favor_target_model:
                merged_param[~good_mask] = theta_stack[0][~good_mask]
            else:
                merged_param[~good_mask] = theta_stack.mean(dim=0)[~good_mask]

        # Convert back to original dtype
        merged[key] = merged_param.to(finetuned_states[0][key].dtype)

    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    print("\n" + "=" * 70)
    print("Fisher Merging for ORB")
    print("=" * 70)
    print(f"Base model: {args.base_model}")
    print(f"Fine-tuned models: {len(args.checkpoint)}")
    for i, ckpt in enumerate(args.checkpoint, 1):
        print(f"  [{i}] {ckpt}")
    print(f"Training data (Fisher): {args.db_path}")
    print(f"Num samples: {args.num_samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"Fisher floor: {args.fisher_floor}")
    print(f"Normalize Fishers: {args.normalize_fishers}")
    print(f"Favor target model (model 1) when Fisher small: {args.favor_target_model}")
    print(f"Force-only mode: {args.force_only}")
    print(f"Data units: {args.data_units}")
    print(f"Output: {args.output_ckpt}")
    print("=" * 70)

    device = resolve_device(args.device)
    dtype = get_dtype(args.precision)
    target_scale = resolve_data_unit_scale(args.data_units)
    print(f"Using device: {device}, dtype: {dtype}")

    # Load fine-tuned checkpoints
    print(f"\nLoading {len(args.checkpoint)} fine-tuned checkpoints...")
    finetuned_states: List[Dict[str, torch.Tensor]] = []
    all_finetuned_ckpts: List[dict] = []

    for i, ckpt_path in enumerate(args.checkpoint):
        print(f"  [{i + 1}] Loading {ckpt_path.name}")
        full_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        all_finetuned_ckpts.append(full_ckpt)

        state = extract_state_dict(ckpt_path)
        finetuned_states.append(dict(state))  # Convert to regular dict

    # Build dataset and dataloader (shared for all models)
    print(f"\nBuilding Fisher dataset from {args.db_path}...")

    # Load a reference model to get system_config
    ref_model = load_model_with_state(
        args.base_model,
        finetuned_states[0],
        device,
        args.precision,
        args.force_only,
    )

    dataset = build_dataset(
        args.db_path,
        args.dataset_name,
        ref_model,
        dtype,
        args.force_only,
    )
    print(f"  Dataset size: {len(dataset)} samples")

    train_indices = resolve_train_subset_indices(args.config, dataset)
    if train_indices is not None:
        print(f"  Using train split from {args.config} ({len(train_indices)} samples)")

    dataloader = build_dataloader(
        dataset,
        args.batch_size,
        args.num_samples,
        args.num_workers,
        subset_indices=train_indices,
    )
    sample_count = (
        min(args.num_samples, len(train_indices))
        if train_indices is not None
        else min(args.num_samples, len(dataset))
    )
    print(f"  Using {sample_count} samples for Fisher approximation")

    # Compute Fisher for each fine-tuned model
    fisher_dicts: List[Dict[str, torch.Tensor]] = []

    for i, (ckpt_path, state) in enumerate(zip(args.checkpoint, finetuned_states)):
        print(f"\n[{i + 1}] {ckpt_path.name}")

        # Load model with this state
        model = load_model_with_state(
            args.base_model,
            state,
            device,
            args.precision,
            args.force_only,
        )

        print("  Computing diagonal Fisher approximation...")
        start = time.perf_counter()
        fisher = compute_fisher_diagonal(
            model,
            dataloader,
            max_batches=None,
            device=device,
            dtype=dtype,
            target_scale=target_scale,
        )
        if args.normalize_fishers:
            fisher = normalize_fisher_l2(fisher)
        fisher_dicts.append(fisher)
        elapsed = time.perf_counter() - start
        print(f"  Fisher computed in {elapsed:.2f}s")

        # Free memory
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Perform Fisher merging
    print("\nMerging models with Fisher-weighted averaging...")
    merged_state = fisher_merge_states(
        finetuned_states=finetuned_states,
        fishers=fisher_dicts,
        fisher_floor=args.fisher_floor,
        favor_target_model=args.favor_target_model,
    )

    print(f"  Merged {len(merged_state)} parameters")

    # Build merged checkpoint
    base_ckpt = all_finetuned_ckpts[0] if all_finetuned_ckpts else {}
    merged_ckpt = copy.deepcopy(base_ckpt) if isinstance(base_ckpt, dict) else {}

    # Update state dict
    merged_ckpt["model"] = merged_state

    # Add merge metadata
    merged_ckpt["fisher_merge_info"] = {
        "fisher_floor": args.fisher_floor,
        "normalize_fishers": args.normalize_fishers,
        "favor_target_model": args.favor_target_model,
        "num_models": len(args.checkpoint),
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "base_model": args.base_model,
        "force_only": args.force_only,
    }
    merged_ckpt["merged_from"] = [str(p) for p in args.checkpoint]
    merged_ckpt["merge_strategy"] = "fisher_merging"
    merged_ckpt["merge_num_models"] = len(args.checkpoint)

    # Reset training-related fields
    merged_ckpt.pop("optimizer", None)
    merged_ckpt.pop("scheduler", None)
    merged_ckpt["epoch"] = 0
    merged_ckpt["global_step"] = 0
    merged_ckpt["best_val_loss"] = None

    # Save merged checkpoint
    args.output_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged_ckpt, args.output_ckpt)
    print(f"\nSaved Fisher-merged checkpoint to {args.output_ckpt}")

    # Optional diagnostics log
    if args.log_path:
        args.log_path.parent.mkdir(parents=True, exist_ok=True)
        with args.log_path.open("w", encoding="utf-8") as f:
            meta = {
                "fisher_floor": args.fisher_floor,
                "normalize_fishers": args.normalize_fishers,
                "favor_target_model": args.favor_target_model,
                "checkpoints": [str(p) for p in args.checkpoint],
                "base_model": args.base_model,
                "num_samples": args.num_samples,
                "force_only": args.force_only,
            }
            f.write(json.dumps(meta) + "\n")
        print(f"Diagnostics written to {args.log_path}")

    print("\n" + "=" * 70)
    print("Fisher Merging Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
