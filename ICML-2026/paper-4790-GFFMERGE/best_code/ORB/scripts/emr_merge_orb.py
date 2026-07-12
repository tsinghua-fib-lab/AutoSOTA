"""
EMR-Merging for ORB Models

Implements EMR (Elastic Merge Reduce) for merging multiple fine-tuned ORB models.
EMR elects a unified task vector, then uses per-task masks and rescalers to
reconstruct task-specific models that share the unified representation.
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


# Utilities


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
    """Normalize state dict keys to match reference model's key format."""
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

    # Return original if no transformation helps
    return dict(state_dict)


def build_model(base_model: str, device: torch.device, precision: str, compile_model: bool) -> torch.nn.Module:
    """Build an ORB model from pretrained weights."""
    loader = pretrained.ORB_PRETRAINED_MODELS[base_model]
    model = loader(device=device, precision=precision, compile=compile_model, train=True)
    model.eval()
    return model


def make_dataset(
    dataset_path: Path,
    dataset_name: str,
    model: torch.nn.Module,
    dtype: torch.dtype,
    force_only: bool = False,
) -> AseSqliteDataset:
    """Create an ORB dataset for evaluation."""
    target_config = property_definitions.PropertyConfig(
        node_names=["forces"],
        graph_names=[] if force_only else ["energy"],
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
) -> Tuple[DataLoader, int]:
    """Create a DataLoader for evaluation (validation split) and return sample count."""
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

    num_samples = len(val_subset)

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
    return DataLoader(val_subset, **loader_kwargs), num_samples


# EMR Merging Core


def is_param_key(k: str) -> bool:
    """Check if a key should be included in task vector computation."""
    if not isinstance(k, str):
        return False
    # Exclude non-parameter entries
    exclude = ["datamean", "datastd", "data_mean", "data_std"]
    if any(k.lower().startswith(p) for p in exclude):
        return False
    return True


def build_task_vectors(
    pretrained_state: Dict[str, torch.Tensor],
    finetuned_states: List[Dict[str, torch.Tensor]],
) -> Tuple[List[str], List[Dict[str, torch.Tensor]]]:
    """Build task vectors: tau_i = W_finetuned_i - W_pretrained."""
    # Use intersection of keys present in pretrained and all fine-tuned models
    keys = [
        k
        for k, v in pretrained_state.items()
        if is_param_key(k) and isinstance(v, torch.Tensor)
    ]

    task_vectors: List[Dict[str, torch.Tensor]] = []
    for st in finetuned_states:
        tv: Dict[str, torch.Tensor] = {}
        for k in keys:
            if k in st and isinstance(st[k], torch.Tensor):
                tv[k] = st[k] - pretrained_state[k]
            else:
                tv[k] = torch.zeros_like(pretrained_state[k])
        task_vectors.append(tv)

    return keys, task_vectors


def emr_elect_unified(
    keys: List[str], task_vectors: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """
    Elect unified task vector (tau_uni) using sign voting and max magnitude.

    For each parameter position:
    1. Compute sign vote: sum of signs across all task vectors
    2. Determine dominant sign (gamma_uni)
    3. Among values with matching sign, select the one with max magnitude
    """
    tau_uni: Dict[str, torch.Tensor] = {}

    for k in keys:
        stack = torch.stack([tv[k] for tv in task_vectors], dim=0)  # [n_models, ...]

        # sign_sum in {-n..n}, then sign => {-1,0,1}
        sign_sum = stack.sign().sum(dim=0)
        gamma_uni = sign_sum.sign()

        # mask entries where sign matches gamma_uni
        same_sign = stack.sign() == gamma_uni.unsqueeze(0)
        masked = stack * same_sign  # zero where sign mismatch

        # choose element with max abs value among same-sign entries
        abs_masked = masked.abs()
        idx = abs_masked.argmax(dim=0)  # index along model dimension

        # gather chosen values
        gathered = masked.gather(0, idx.unsqueeze(0)).squeeze(0)

        # where gamma_uni == 0 (no dominant sign), set to zero
        tau_k = torch.where(gamma_uni == 0, torch.zeros_like(gathered), gathered)
        tau_uni[k] = tau_k

    return tau_uni


def emr_build_modulators(
    keys: List[str],
    task_vectors: List[Dict[str, torch.Tensor]],
    tau_uni: Dict[str, torch.Tensor],
    eps: float = 1e-12,
) -> Tuple[List[Dict[str, torch.Tensor]], List[float]]:
    """
    Build task-specific masks (M_i) and rescalers (lambda_i).

    For each task:
    - M_i[k] = 1 where tau_i[k] * tau_uni[k] > 0, else 0
    - lambda_i = sum(|tau_i|) / sum(|M_i * tau_uni|)
    """
    masks: List[Dict[str, torch.Tensor]] = []
    lambdas: List[float] = []

    for tv in task_vectors:
        Mi: Dict[str, torch.Tensor] = {}
        num = 0.0
        den = 0.0

        for k in keys:
            tau_i = tv[k]
            tau_u = tau_uni[k]

            prod = tau_i * tau_u
            mask_k = (prod > 0).to(tau_i.dtype)  # 1 where signs agree, else 0
            Mi[k] = mask_k

            num += float(tau_i.abs().sum())
            den += float((mask_k * tau_u).abs().sum())

        lam = num / (den + eps)
        masks.append(Mi)
        lambdas.append(lam)

    return masks, lambdas


def emr_reconstruct_state_for_task(
    pretrained_state: Dict[str, torch.Tensor],
    keys: List[str],
    tau_uni: Dict[str, torch.Tensor],
    mask_i: Dict[str, torch.Tensor],
    lambda_i: float,
) -> Dict[str, torch.Tensor]:
    """Reconstruct merged state for a specific task: W_i = W_pre + lambda_i * (M_i * tau_uni)."""
    merged = dict(pretrained_state)

    for k in keys:
        base = pretrained_state[k]
        tau_u = tau_uni[k].to(base.device, dtype=base.dtype)
        m_k = mask_i[k].to(base.device, dtype=base.dtype)
        tau_hat = lambda_i * (m_k * tau_u)
        merged[k] = base + tau_hat

    return merged


def save_emr_merged_checkpoints(
    pretrained_state: Dict[str, torch.Tensor],
    finetuned_ckpts: List[dict],
    keys: List[str],
    tau_uni: Dict[str, torch.Tensor],
    masks: List[Dict[str, torch.Tensor]],
    lambdas: List[float],
    output_dir: Path,
    output_names: Optional[List[str]] = None,
    keep_metadata: bool = False,
) -> List[Path]:
    """Save EMR-merged checkpoints for each task."""
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for i, (ft_ckpt, Mi, lam) in enumerate(zip(finetuned_ckpts, masks, lambdas)):
        # Reconstruct merged state
        merged_state = emr_reconstruct_state_for_task(
            pretrained_state=pretrained_state,
            keys=keys,
            tau_uni=tau_uni,
            mask_i=Mi,
            lambda_i=lam,
        )

        # Use fine-tuned ckpt as template for metadata if requested
        if keep_metadata:
            base_ckpt = copy.deepcopy(ft_ckpt)
        else:
            base_ckpt = {}
        base_ckpt["model"] = merged_state

        # Add EMR metadata
        base_ckpt["emr_info"] = {
            "lambda": float(lam),
            "task_index": i,
        }

        if keep_metadata:
            base_ckpt.pop("optimizer", None)
            base_ckpt.pop("scheduler", None)
            base_ckpt["epoch"] = 0
            base_ckpt["global_step"] = 0
            base_ckpt["best_val_loss"] = None

        # Determine output filename
        if output_names and i < len(output_names):
            out_name = output_names[i]
        else:
            out_name = f"emr_task_{i}.ckpt"

        out_path = output_dir / out_name
        torch.save(base_ckpt, out_path)
        saved_paths.append(out_path)
        print(f"  Saved EMR-merged checkpoint {i+1}/{len(finetuned_ckpts)} to {out_path}")

    return saved_paths


def save_emr_unified_checkpoint(
    pretrained_state: Dict[str, torch.Tensor],
    finetuned_ckpts: List[dict],
    keys: List[str],
    tau_uni: Dict[str, torch.Tensor],
    output_dir: Path,
    output_name: str = "emr_unified.ckpt",
    keep_metadata: bool = False,
) -> Path:
    """Save unified EMR model: W_uni = W_pre + tau_uni."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build unified state
    unified_state = dict(pretrained_state)
    for k in keys:
        base = pretrained_state[k]
        tau_u = tau_uni[k].to(base.device, dtype=base.dtype)
        unified_state[k] = base + tau_u

    # Use first fine-tuned checkpoint as template for metadata if requested
    if keep_metadata:
        base_ckpt = copy.deepcopy(finetuned_ckpts[0])
    else:
        base_ckpt = {}
    base_ckpt["model"] = unified_state

    # Add EMR metadata
    base_ckpt["emr_info"] = {
        "model_type": "unified",
        "description": "Universal EMR model using only tau_uni (no per-task masks/rescalers)",
    }

    if keep_metadata:
        base_ckpt.pop("optimizer", None)
        base_ckpt.pop("scheduler", None)
        base_ckpt["epoch"] = 0
        base_ckpt["global_step"] = 0
        base_ckpt["best_val_loss"] = None

    out_path = output_dir / output_name
    torch.save(base_ckpt, out_path)
    print(f"  Saved universal EMR checkpoint to {out_path}")

    return out_path


# Evaluation


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    target_scale: float,
    force_only: bool = False,
) -> Dict[str, float]:
    """Evaluate model on a dataset and return metrics."""
    model.eval()

    total_energy_abs_error = 0.0
    total_force_abs_error = 0.0
    total_samples = 0
    total_force_components = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device=device, dtype=dtype)
            scale_batch_targets(batch, target_scale)
            predictions = model.predict(batch)

            # Force metrics
            pred_forces = predictions["forces"]
            target_forces = batch.node_targets["forces"]
            force_abs_error = torch.sum(torch.abs(pred_forces - target_forces)).item()
            total_force_abs_error += force_abs_error
            total_force_components += target_forces.numel()

            # Energy metrics (unless force_only)
            if not force_only:
                pred_energy = predictions.get("energy")
                target_energy = batch.system_targets.get("energy")

                if pred_energy is not None and target_energy is not None:
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


def evaluate_checkpoint(
    checkpoint_path: Path,
    config: Dict,
    base_model: str,
    precision: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    compile_model: bool = False,
    force_only: bool = False,
    data_units: Optional[str] = None,
) -> Tuple[Dict[str, float], int]:
    """Evaluate a single checkpoint on its task's data."""
    model = build_model(base_model, device, precision, compile_model)
    dtype = next(model.parameters()).dtype

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model_keys = set(model.state_dict().keys())
    normalized_state = normalize_state_dict_keys(state_dict, model_keys)
    model.load_state_dict(normalized_state)
    model.to(device)

    # Prepare dataset
    dataset_path = Path(config.get("db_path", config.get("dataset_path", "")))
    dataset_name = config.get("dataset_name", "eval")
    dataset = make_dataset(dataset_path, dataset_name, model, dtype, force_only=force_only)
    target_scale = resolve_data_unit_scale(data_units or config.get("data_units", "kcal/mol"))

    pin_memory = device.type == "cuda"
    eval_loader, num_samples = make_eval_loader(
        dataset=dataset,
        config=config,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=2,
        use_val_split=True,
    )

    metrics = evaluate_model(
        model,
        eval_loader,
        device,
        dtype,
        target_scale=target_scale,
        force_only=force_only,
    )
    return metrics, num_samples


def aggregate_metrics(
    all_metrics: List[Dict[str, float]],
    all_num_samples: List[int],
) -> Dict[str, float]:
    """Aggregate metrics across tasks using weighted average by number of samples."""
    if not all_metrics:
        return {}

    total_samples = sum(all_num_samples)
    aggregated = {}

    # Get all unique metric keys
    all_keys = set()
    for m in all_metrics:
        all_keys.update(m.keys())

    for key in all_keys:
        weighted_sum = 0.0
        for metrics, n_samples in zip(all_metrics, all_num_samples):
            if key in metrics:
                weighted_sum += metrics[key] * n_samples
        aggregated[key] = weighted_sum / total_samples if total_samples > 0 else 0.0

    return aggregated


# CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        required=True,
        help="Pretrained (base) model checkpoint path.",
    )

    parser.add_argument(
        "--checkpoint",
        action="append",
        type=Path,
        required=True,
        help="Fine-tuned checkpoint to merge (repeatable).",
    )

    parser.add_argument(
        "--config",
        action="append",
        type=Path,
        help=(
            "Config JSON for each fine-tuned checkpoint (repeatable). "
            "If not provided, looks for config.json next to each checkpoint."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for EMR-merged checkpoints (one per task).",
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

    parser.add_argument(
        "--data-units",
        type=str,
        default=None,
        help="Override dataset units for evaluation (kcal/mol or eV).",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation.",
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
        "--log-path",
        type=Path,
        default=None,
        help="Optional JSONL log path for EMR statistics.",
    )

    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate each EMR-merged checkpoint on its task's data and aggregate results.",
    )

    parser.add_argument(
        "--force-only",
        action="store_true",
        help="Evaluate using only force MAE (for force-only models).",
    )

    parser.add_argument(
        "--save-unified",
        action="store_true",
        help="Also save the unified EMR model (W_pre + tau_uni).",
    )

    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="Enable torch.compile for model.",
    )

    parser.add_argument(
        "--keep-metadata",
        action="store_true",
        help="Copy metadata fields from each fine-tuned checkpoint (optimizer, scheduler, etc.).",
    )

    return parser


def load_config(path: Optional[Path], checkpoint: Path) -> dict:
    """Load config from path or look for config.json next to checkpoint."""
    if path is None:
        candidate = checkpoint.parent / "config.json"
        if candidate.exists():
            path = candidate
    if path is None or not path.exists():
        raise FileNotFoundError(
            f"Config not found for {checkpoint}. Provide --config explicitly or add config.json."
        )
    return load_json(path)


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    print("\n" + "=" * 70)
    print("EMR-Merging for ORB Models")
    print("=" * 70)
    print(f"Pretrained checkpoint: {args.pretrained_checkpoint}")
    print(f"Base model: {args.base_model}")
    print(f"Fine-tuned models: {len(args.checkpoint)}")
    for i, ckpt in enumerate(args.checkpoint, 1):
        print(f"  [{i}] {ckpt}")
    print(f"Output directory: {args.output_dir}")
    if args.data_units:
        print(f"Data units override: {args.data_units}")
    print("=" * 70)

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    # Handle config paths
    config_paths = args.config or [None] * len(args.checkpoint)
    if len(config_paths) == 1 and len(args.checkpoint) > 1:
        config_paths = config_paths * len(args.checkpoint)
    if len(config_paths) != len(args.checkpoint):
        raise ValueError("Number of --config must match number of --checkpoint")

    # Build model to get expected keys
    print("\nBuilding base model...")
    model = build_model(args.base_model, device, args.precision, args.compile_model)
    model_keys = set(model.state_dict().keys())
    dtype = next(model.parameters()).dtype

    # Load pretrained checkpoint
    print(f"\nLoading pretrained checkpoint: {args.pretrained_checkpoint}")
    pretrained_ckpt = torch.load(
        args.pretrained_checkpoint, map_location="cpu"
    )
    raw_pretrained_state = pretrained_ckpt["model"] if "model" in pretrained_ckpt else pretrained_ckpt
    pretrained_state = normalize_state_dict_keys(raw_pretrained_state, model_keys)
    print(f"  Loaded {len(pretrained_state)} keys")

    # Load fine-tuned checkpoints
    print(f"\nLoading {len(args.checkpoint)} fine-tuned checkpoints...")
    finetuned_ckpts: List[dict] = []
    finetuned_states: List[Dict[str, torch.Tensor]] = []

    for i, ckpt_path in enumerate(args.checkpoint):
        print(f"  [{i+1}] {ckpt_path.name}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        finetuned_ckpts.append(ckpt)
        raw_state = ckpt["model"] if "model" in ckpt else ckpt
        state = normalize_state_dict_keys(raw_state, model_keys)
        finetuned_states.append(state)

    # Build task vectors
    print("\nBuilding task vectors...")
    start = time.perf_counter()
    keys, task_vectors = build_task_vectors(pretrained_state, finetuned_states)
    print(f"  Parameter keys: {len(keys)}")
    print(f"  Task vectors built in {time.perf_counter() - start:.2f}s")

    # Diagnostics: task vector norms
    for i, tv in enumerate(task_vectors):
        total_norm = sum((v ** 2).sum().item() for v in tv.values()) ** 0.5
        print(f"    Task {i} L2 norm: {total_norm:.4f}")

    # EMR elect: unified task vector
    print("\nElecting unified task vector...")
    start = time.perf_counter()
    tau_uni = emr_elect_unified(keys, task_vectors)
    tau_uni_norm = sum((v ** 2).sum().item() for v in tau_uni.values()) ** 0.5
    print(f"  Unified task vector computed in {time.perf_counter() - start:.2f}s")
    print(f"  Unified task vector L2 norm: {tau_uni_norm:.4f}")

    # Build masks and rescalers
    print("\nBuilding task-specific modulators...")
    start = time.perf_counter()
    masks, lambdas = emr_build_modulators(keys, task_vectors, tau_uni)
    print(f"  Modulators built in {time.perf_counter() - start:.2f}s")
    print(f"  Learned rescalers (lambdas):")
    for i, lam in enumerate(lambdas):
        print(f"    Task {i}: {lam:.6f}")

    # Optionally save unified EMR model
    unified_path = None
    if args.save_unified:
        print("\nSaving universal EMR model...")
        unified_path = save_emr_unified_checkpoint(
            pretrained_state=pretrained_state,
            finetuned_ckpts=finetuned_ckpts,
            keys=keys,
            tau_uni=tau_uni,
            output_dir=args.output_dir,
            keep_metadata=args.keep_metadata,
        )

    # Save EMR-merged checkpoints
    print("\nSaving EMR-merged checkpoints...")
    output_names = [f"emr_task_{i}.ckpt" for i in range(len(args.checkpoint))]
    saved_paths = save_emr_merged_checkpoints(
        pretrained_state=pretrained_state,
        finetuned_ckpts=finetuned_ckpts,
        keys=keys,
        tau_uni=tau_uni,
        masks=masks,
        lambdas=lambdas,
        output_dir=args.output_dir,
        output_names=output_names,
        keep_metadata=args.keep_metadata,
    )

    # Evaluate each task-specific checkpoint on its data
    eval_results = []
    aggregated_metrics = {}
    if args.evaluate:
        print("\n" + "=" * 70)
        print("Evaluating EMR-merged checkpoints on task-specific data")
        print("=" * 70)

        # Load configs for each task
        configs = []
        for i, cfg_path in enumerate(config_paths):
            cfg = load_config(cfg_path, args.checkpoint[i])
            configs.append(cfg)

        all_metrics = []
        all_num_samples = []

        for i, (ckpt_path, cfg) in enumerate(zip(saved_paths, configs)):
            print(f"\n[Task {i}] Evaluating {ckpt_path.name}...")
            try:
                metrics, num_samples = evaluate_checkpoint(
                    checkpoint_path=ckpt_path,
                    config=cfg,
                    base_model=args.base_model,
                    precision=args.precision,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    device=device,
                    compile_model=args.compile_model,
                    force_only=args.force_only,
                    data_units=args.data_units,
                )
                all_metrics.append(metrics)
                all_num_samples.append(num_samples)

                eval_results.append({
                    "task_index": i,
                    "checkpoint": str(ckpt_path),
                    "num_samples": num_samples,
                    "metrics": metrics,
                })

                # Print task metrics
                print(f"  Samples: {num_samples}")
                for key, value in sorted(metrics.items()):
                    print(f"  {key}: {value:.6f}")

            except Exception as e:
                print(f"  Error evaluating task {i}: {e}")
                eval_results.append({
                    "task_index": i,
                    "checkpoint": str(ckpt_path),
                    "error": str(e),
                })

        # Aggregate metrics
        if all_metrics:
            print("\n" + "-" * 70)
            print("AGGREGATED METRICS (weighted by number of samples)")
            print("-" * 70)
            aggregated_metrics = aggregate_metrics(all_metrics, all_num_samples)
            total_samples = sum(all_num_samples)
            print(f"Total samples: {total_samples}")
            for key, value in sorted(aggregated_metrics.items()):
                print(f"  {key}: {value:.6f}")

    # Log statistics if requested
    if args.log_path:
        args.log_path.parent.mkdir(parents=True, exist_ok=True)
        with args.log_path.open("w", encoding="utf-8") as f:
            meta = {
                "method": "EMR_Merging",
                "num_models": len(args.checkpoint),
                "lambdas": lambdas,
                "num_param_keys": len(keys),
                "checkpoints": [str(p) for p in args.checkpoint],
                "pretrained": str(args.pretrained_checkpoint),
                "base_model": args.base_model,
                "saved_task_specific": [str(p) for p in saved_paths],
            }
            if unified_path:
                meta["saved_unified"] = str(unified_path)
            if args.evaluate:
                meta["evaluation"] = {
                    "task_results": eval_results,
                    "aggregated_metrics": aggregated_metrics,
                }
            f.write(json.dumps(meta) + "\n")
        print(f"\nLogged EMR statistics to {args.log_path}")

    print("\n" + "=" * 70)
    print("EMR-Merging Complete")
    print("=" * 70)
    print(f"Saved {len(saved_paths)} task-specific EMR checkpoints to {args.output_dir}")
    if unified_path:
        print(f"Saved 1 universal EMR checkpoint: {unified_path}")

    if args.evaluate and aggregated_metrics:
        print("\n--- SUMMARY ---")
        print(f"Aggregated performance across {len(saved_paths)} tasks:")
        for key, value in sorted(aggregated_metrics.items()):
            print(f"  {key}: {value:.6f}")
    else:
        print("\nEvaluate task-specific checkpoints with evaluate_model.py:")
        for i, path in enumerate(saved_paths):
            print(f"  python evaluate_model.py --checkpoint {path} --config <task{i}_config.json>")
        if unified_path:
            print("\nEvaluate universal multi-task model:")
            print(f"  python evaluate_model.py --checkpoint {unified_path} --config config.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
