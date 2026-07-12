"""
TIES-Merging for M3GNet Models
"""

from __future__ import annotations

import argparse
import copy
import itertools
import os
import sys
import types
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

os.environ.setdefault("MATGL_BACKEND", "DGL")
os.environ.setdefault("DGLBACKEND", "pytorch")
os.environ.setdefault("DGL_SKIP_GRAPHBOLT", "1")


def install_graphbolt_stub() -> None:
    """Provide a minimal GraphBolt stub so DGL can import without the C++ lib."""
    if "dgl.graphbolt" in sys.modules:
        return
    stub = types.ModuleType("dgl.graphbolt")
    def _unavailable(*_args, **_kwargs):
        raise RuntimeError("GraphBolt is not available in this environment.")
    stub.load_from_shared_memory = _unavailable
    stub.__all__ = []
    sys.modules["dgl.graphbolt"] = stub


if os.environ.get("DGL_SKIP_GRAPHBOLT", "0") == "1":
    install_graphbolt_stub()


class TaskVector:
    def __init__(self, pretrained_state: Dict, finetuned_state: Dict):
        self.task_vector = {}

        pretrained_keys = set(pretrained_state.keys())
        finetuned_keys = set(finetuned_state.keys())
        common_keys = pretrained_keys & finetuned_keys

        only_pretrained = pretrained_keys - finetuned_keys
        only_finetuned = finetuned_keys - pretrained_keys

        if len(common_keys) == 0:
            raise ValueError(
                f"No common keys found between pretrained and fine-tuned models!\n"
            )

        for key in pretrained_state.keys():
            if isinstance(pretrained_state[key], torch.Tensor) and key in finetuned_state:
                if isinstance(finetuned_state[key], torch.Tensor):
                    self.task_vector[key] = finetuned_state[key] - pretrained_state[key]


def canonicalize_key(key: str) -> str:
    if key.startswith('model.model.'):
        return key[12:]
    elif key.startswith('model.'):
        return key[6:]
    return key


def load_state_dict_canonical(checkpoint_path: Path) -> tuple[Dict, Dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "state_dict" in checkpoint:
        original_state_dict = checkpoint["state_dict"]
    else:
        original_state_dict = checkpoint

    canonical_state_dict = {}
    for key, value in original_state_dict.items():
        canonical_key = canonicalize_key(key)
        canonical_state_dict[canonical_key] = value

    return canonical_state_dict, original_state_dict


def trim_task_vectors(
    task_vectors: List[TaskVector],
    density: float
) -> List[TaskVector]:
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
            # sign * values > 0 means they agree in sign
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


M3GNET_ROOT = Path(__file__).resolve().parents[1]
if M3GNET_ROOT.name.lower() == "baselines":
    M3GNET_ROOT = M3GNET_ROOT.parent

def resolve_path(path_str: str | Path, root: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def dgl_cuda_ok() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import dgl
        g = dgl.graph((torch.tensor([0]), torch.tensor([0])))
        g = g.to("cuda")
        _ = g.device
        return True
    except Exception:
        return False


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda") if dgl_cuda_ok() else torch.device("cpu")
    if device_arg == "cuda":
        if dgl_cuda_ok():
            return torch.device("cuda")
        print("DGL CUDA backend unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device("cpu")


def read_extxyz(
    path: Path, limit: Optional[int] = None
) -> Tuple[List, List[float], List[np.ndarray]]:
    """Read EXTXYZ and extract structures, energies, forces."""
    import ase.io
    from pymatgen.core import Molecule

    index_str = f":{limit}" if limit is not None else ":"
    atoms_list = ase.io.read(str(path), index=index_str)
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]

    structures = []
    energies: List[float] = []
    forces: List[np.ndarray] = []

    for atoms in atoms_list:
        energy = None
        if atoms.calc is not None:
            try:
                energy = float(atoms.get_potential_energy())
            except Exception:
                energy = None

        if energy is None:
            for key in ("energy", "E", "total_energy", "totalenergy"):
                if key in atoms.info:
                    energy = float(atoms.info[key])
                    break

        if energy is None:
            continue

        force = None
        if atoms.calc is not None:
            try:
                force = np.asarray(atoms.get_forces(), dtype=float)
            except Exception:
                force = None

        if force is None and "forces" in atoms.arrays:
            force = np.asarray(atoms.arrays["forces"], dtype=float)

        if force is None:
            force = np.zeros((len(atoms), 3), dtype=float)

        mol = Molecule(atoms.get_chemical_symbols(), atoms.get_positions())
        structures.append(mol)
        energies.append(energy)
        forces.append(force)

    return structures, energies, forces


def build_eval_dataset(
    name: str,
    structures: List,
    energies: List[float],
    forces: List[np.ndarray],
    converter,
    cache_dir: Path,
):
    """Build matgl MGLDataset for evaluation."""
    from matgl.graph.data import MGLDataset

    labels = {
        "energies": [float(value) for value in energies],
        "forces": [np.asarray(value, dtype=float).tolist() for value in forces],
    }

    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset = MGLDataset(
        structures=structures,
        labels=labels,
        converter=converter,
        include_line_graph=False,
        directory_name=name,
        save_dir=str(cache_dir),
        raw_dir=str(cache_dir),
        save_cache=True,
    )

    if dataset.has_cache():
        dataset.load()
    else:
        dataset.process()
        dataset.save()

    return dataset


def load_eval_loader(
    config: dict,
    batch_size: int,
    num_workers: int,
    split: str,
) -> Tuple:
    """Load evaluation dataloader and module kwargs from config."""
    import yaml
    from torch.utils.data import DataLoader
    from matgl import load_model
    from matgl.ext._pymatgen_dgl import Molecule2Graph
    from matgl.graph.data import collate_fn_pes

    train_cfg = config.get("train", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})

    lr = float(train_cfg.get("lr", 1e-4))
    energy_weight = float(train_cfg.get("energy_weight", 1.0))
    force_weight = float(train_cfg.get("force_weight", 0.1))
    stress_weight = float(train_cfg.get("stress_weight", 0.0))
    decay_steps = int(train_cfg.get("decay_steps", 1000))
    decay_alpha = float(train_cfg.get("decay_alpha", 0.01))

    if split == "test":
        split_path_cfg = data_cfg.get("test_path")
        if split_path_cfg is None:
            raise ValueError("Config missing data.test_path for --split test.")
    else:
        split_path_cfg = data_cfg.get("val_path")
        if split_path_cfg is None:
            raise ValueError("Config missing data.val_path")

    split_paths = split_path_cfg if isinstance(split_path_cfg, (list, tuple)) else [split_path_cfg]
    split_files = [resolve_path(path, M3GNET_ROOT) for path in split_paths]

    cache_dir = resolve_path(data_cfg.get("cache_dir", "data/cache/default"), M3GNET_ROOT)

    pretrained_name = model_cfg.get("pretrained_name", "M3GNet-ANI-1x-Subset-PES")
    cutoff = float(model_cfg.get("cutoff", 5.0))

    potential = load_model(pretrained_name)
    base_model = potential.model
    element_types = getattr(base_model, "element_types", None)

    if element_types is None:
        raise RuntimeError("Pretrained model did not expose element types.")

    structures = []
    energies = []
    forces = []

    for path in split_files:
        s, e, f = read_extxyz(path)
        structures.extend(s)
        energies.extend(e)
        forces.extend(f)

    converter = Molecule2Graph(element_types=element_types, cutoff=cutoff)

    split_dataset = build_eval_dataset(
        name=f"{split}_{hash(tuple(str(p) for p in split_files)) % 10000}",
        structures=structures,
        energies=energies,
        forces=forces,
        converter=converter,
        cache_dir=cache_dir / split,
    )

    collate = lambda batch: collate_fn_pes(batch, include_stress=False, include_line_graph=False)

    eval_loader = DataLoader(
        split_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
    )

    element_refs = None
    if getattr(potential, "element_refs", None) is not None:
        element_refs = potential.element_refs.property_offset.detach().cpu().numpy()

    module_kwargs = {
        "model": base_model,
        "element_refs": element_refs,
        "energy_weight": energy_weight,
        "force_weight": force_weight,
        "stress_weight": stress_weight,
        "data_mean": float(potential.data_mean),
        "data_std": float(potential.data_std),
        "lr": lr,
        "decay_steps": decay_steps,
        "decay_alpha": decay_alpha,
    }

    num_samples = len(structures)
    return eval_loader, module_kwargs, num_samples


def evaluate_merged_state(
    merged_state: Dict,
    base_checkpoint: dict,
    config: dict,
    batch_size: int,
    num_workers: int,
    split: str,
    device: torch.device,
    temp_ckpt_path: Path,
) -> Tuple[Dict[str, float], int]:
    """Evaluate a merged state dict on validation/test data."""
    import lightning as pl
    from matgl.utils.training import PotentialLightningModule

    temp_checkpoint = copy.deepcopy(base_checkpoint)
    temp_checkpoint["state_dict"] = merged_state
    torch.save(temp_checkpoint, temp_ckpt_path)

    eval_loader, module_kwargs, num_samples = load_eval_loader(
        config=config,
        batch_size=batch_size,
        num_workers=num_workers,
        split=split,
    )

    accelerator = "gpu" if device.type == "cuda" else "cpu"
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        inference_mode=False,
    )

    module = PotentialLightningModule.load_from_checkpoint(
        checkpoint_path=str(temp_ckpt_path),
        weights_only=False,
        **module_kwargs,
    )

    metrics_list = trainer.validate(module, dataloaders=eval_loader, verbose=False)
    metrics = metrics_list[0] if metrics_list else {}

    numeric_metrics = {
        key: (float(value) if not isinstance(value, torch.Tensor) else float(value.item()))
        for key, value in metrics.items()
        if isinstance(value, (float, int, torch.Tensor))
    }

    return numeric_metrics, num_samples


def grid_search_ties(
    pretrained_canonical: Dict,
    task_vectors: List[TaskVector],
    finetuned_original_state: Dict,
    base_checkpoint: dict,
    all_checkpoints: List[dict],
    config: dict,
    density_values: List[float],
    lambda_values: List[float],
    batch_size: int,
    num_workers: int,
    split: str,
    device: torch.device,
    metric_key: str,
    temp_dir: Path,
) -> Tuple[float, float, Dict[str, float], List[dict]]:
    import yaml

    all_combinations = list(itertools.product(density_values, lambda_values))

    print(f"\nGrid Search Configuration:")
    print(f"  Density values: {density_values}")
    print(f"  Lambda values: {lambda_values}")
    print(f"  Total combinations: {len(all_combinations)}")
    print(f"  Optimizing for: {metric_key} (lower is better)")
    print(f"  Evaluation split: {split}")

    # Build canonical to original key mapping
    canonical_to_original = {}
    for orig_key in finetuned_original_state.keys():
        canon_key = canonicalize_key(orig_key)
        canonical_to_original[canon_key] = orig_key

    # Average normalization parameters
    norm_params = {}
    for norm_key in ['data_mean', 'data_std', 'model.data_mean', 'model.data_std']:
        values = []
        for ckpt in all_checkpoints:
            if "state_dict" in ckpt and norm_key in ckpt["state_dict"]:
                values.append(ckpt["state_dict"][norm_key])
        if values:
            norm_params[norm_key] = sum(values) / len(values)

    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_ckpt_path = temp_dir / "temp_grid_search_ties.ckpt"

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
        merged_canonical = apply_ties_merging(
            pretrained_canonical,
            task_vectors,
            density=density,
            lambda_scale=lambda_scale
        )

        # Build state dict with original key structure
        new_state_dict = {}
        for k, v in finetuned_original_state.items():
            if not isinstance(v, torch.Tensor):
                new_state_dict[k] = v

        for canon_key, value in merged_canonical.items():
            if isinstance(value, torch.Tensor):
                orig_key = canonical_to_original.get(canon_key, canon_key)
                new_state_dict[orig_key] = value

        for k, v in norm_params.items():
            new_state_dict[k] = v

        # Evaluate
        try:
            metrics, num_samples = evaluate_merged_state(
                merged_state=new_state_dict,
                base_checkpoint=base_checkpoint,
                config=config,
                batch_size=batch_size,
                num_workers=num_workers,
                split=split,
                device=device,
                temp_ckpt_path=temp_ckpt_path,
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
            print(f"  [{i+1:3d}/{len(all_combinations)}] density={density:.2f}, lambda={lambda_scale:.2f} -> {metric_key}={metric_value:.6f}{best_marker}")

        except Exception as e:
            print(f"  [{i+1:3d}/{len(all_combinations)}] density={density:.2f}, lambda={lambda_scale:.2f} -> ERROR: {e}")
            all_results.append({
                "density": density,
                "lambda": lambda_scale,
                "error": str(e),
            })

    # Clean up temp file
    if temp_ckpt_path.exists():
        temp_ckpt_path.unlink()

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
        "--output-ckpt",
        type=Path,
        required=True,
        help="Output path for merged checkpoint.",
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
        help="Config YAML for validation evaluation (required for grid search).",
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
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Data split for grid search evaluation (default: val).",
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
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for evaluation (default: auto).",
    )

    grid_group.add_argument(
        "--save-grid-results",
        type=Path,
        default=None,
        help="Path to save all grid search results as JSON.",
    )

    return parser


def main(argv=None):
    import json
    import yaml

    args = build_parser().parse_args(argv)

    # Validate arguments based on mode
    if args.grid_search:
        if args.val_config is None:
            raise ValueError("--val-config is required when using --grid-search")
        if args.density is not None or args.lambda_scale is not None:
            print("Note: --density and --lambda values will be ignored when using --grid-search")
    else:
        if args.density is None or args.lambda_scale is None:
            raise ValueError(
                "Either provide --density and --lambda values, "
                "or use --grid-search to find optimal values"
            )

    print("TIES-Merging for M3GNet Models")
    print("=" * 70)
    print(f"Pretrained model: {args.pretrained_checkpoint}")
    print(f"Fine-tuned models ({len(args.checkpoint)}):")
    for i, ckpt in enumerate(args.checkpoint, 1):
        print(f"  [{i}] {ckpt}")

    if args.grid_search:
        print(f"\nMode: Grid Search")
        print(f"  Density values to try: {args.density_values}")
        print(f"  Lambda values to try: {args.lambda_values}")
        print(f"  Validation config: {args.val_config}")
        print(f"  Optimization metric: {args.metric}")
    else:
        print(f"\nMode: Manual Parameters")
        print(f"  Density: {args.density}")
        print(f"  Lambda: {args.lambda_scale}")

    print(f"\nOutput: {args.output_ckpt}")

    # Load pretrained model
    print("\nLoading pretrained model...")
    pretrained_canonical, pretrained_original = load_state_dict_canonical(args.pretrained_checkpoint)
    print(f"  Loaded {len(pretrained_canonical)} keys")

    # Compute task vectors
    print("\nComputing task vectors...")
    task_vectors = []
    all_checkpoints = []
    finetuned_original_state = None

    for i, ckpt_path in enumerate(args.checkpoint, 1):
        print(f"  Task vector {i}/{len(args.checkpoint)}: {ckpt_path.name}")

        full_checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        all_checkpoints.append(full_checkpoint)

        finetuned_canonical, finetuned_orig = load_state_dict_canonical(ckpt_path)

        if finetuned_original_state is None:
            finetuned_original_state = finetuned_orig

        task_vector = TaskVector(pretrained_canonical, finetuned_canonical)
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
        with open(args.val_config, 'r') as f:
            val_config = yaml.safe_load(f)

        device = resolve_device(args.device)
        print(f"\nUsing device: {device}")

        best_density, best_lambda, best_metrics, all_results = grid_search_ties(
            pretrained_canonical=pretrained_canonical,
            task_vectors=task_vectors,
            finetuned_original_state=finetuned_original_state,
            base_checkpoint=all_checkpoints[0],
            all_checkpoints=all_checkpoints,
            config=val_config,
            density_values=args.density_values,
            lambda_values=args.lambda_values,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            split=args.split,
            device=device,
            metric_key=args.metric,
            temp_dir=args.output_ckpt.parent,
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
    merged_canonical = apply_ties_merging(
        pretrained_canonical,
        task_vectors,
        density=density,
        lambda_scale=lambda_scale
    )

    # Prepare checkpoint for saving
    print("\nPreparing merged checkpoint...")
    base_checkpoint = all_checkpoints[0]

    # Average normalization parameters
    print("Averaging normalization parameters...")
    norm_params = {}
    for norm_key in ['data_mean', 'data_std', 'model.data_mean', 'model.data_std']:
        values = []
        for ckpt in all_checkpoints:
            if "state_dict" in ckpt and norm_key in ckpt["state_dict"]:
                values.append(ckpt["state_dict"][norm_key])
        if values:
            avg_value = sum(values) / len(values)
            norm_params[norm_key] = avg_value
            print(f"  {norm_key}: averaged from {len(values)} models")

    # Build mapping from canonical keys to original keys
    canonical_to_original = {}
    for orig_key in finetuned_original_state.keys():
        canon_key = canonicalize_key(orig_key)
        canonical_to_original[canon_key] = orig_key

    # Create merged checkpoint
    merged_checkpoint = copy.deepcopy(base_checkpoint)
    new_state_dict = {}

    for k, v in finetuned_original_state.items():
        if not isinstance(v, torch.Tensor):
            new_state_dict[k] = v

    for canon_key, value in merged_canonical.items():
        if isinstance(value, torch.Tensor):
            orig_key = canonical_to_original.get(canon_key, canon_key)
            new_state_dict[orig_key] = value

    for k, v in norm_params.items():
        new_state_dict[k] = v

    merged_checkpoint["state_dict"] = new_state_dict

    # Store TIES-Merging info in checkpoint metadata
    merged_checkpoint["ties_merging_info"] = {
        "density": density,
        "lambda": lambda_scale,
        "checkpoints": [str(p) for p in args.checkpoint],
    }

    if args.grid_search:
        merged_checkpoint["ties_merging_info"]["grid_search_metric"] = args.metric

    # Save merged checkpoint
    args.output_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged_checkpoint, args.output_ckpt)

    print(f"\n{'='*70}")
    print(f"Merged checkpoint saved: {args.output_ckpt}")
    print(f"{'='*70}")
    print(f"\nFinal TIES-Merging parameters:")
    print(f"  Density: {density}")
    print(f"  Lambda: {lambda_scale}")
    print(f"\nModels merged:")
    for i, ckpt in enumerate(args.checkpoint, 1):
        print(f"  [{i}] {ckpt.name}")


if __name__ == "__main__":
    main()
