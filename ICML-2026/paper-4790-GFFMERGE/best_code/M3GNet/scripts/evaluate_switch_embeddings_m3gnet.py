"""
Evaluate a M3GNet checkpoint by swapping embedding weights per sample label.

This is intended for combined datasets where each frame carries a dataset label
such as "ethanol" or "malonaldehyde" in the EXTXYZ metadata.
Supports evaluating either the validation or test split.
"""

from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
import ase.io
from pymatgen.core import Molecule
from torch.utils.data import DataLoader, Subset

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

from matgl import load_model
from matgl.ext._pymatgen_dgl import Molecule2Graph
from matgl.graph.data import MGLDataset, collate_fn_pes
from matgl.utils.training import PotentialLightningModule


M3GNET_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Config YAML defining data/model settings.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Merged checkpoint to evaluate.")
    parser.add_argument(
        "--ethanol-checkpoint",
        type=Path,
        default=None,
        help="(Deprecated) Teacher checkpoint for dataset A; prefer --source-checkpoint.",
    )
    parser.add_argument(
        "--malonaldehyde-checkpoint",
        type=Path,
        default=None,
        help="(Deprecated) Teacher checkpoint for dataset B; prefer --source-checkpoint.",
    )
    parser.add_argument("--label-a", type=str, default="ethanol", help="Label name for dataset A.")
    parser.add_argument("--label-b", type=str, default="malonaldehyde", help="Label name for dataset B.")
    parser.add_argument(
        "--source-checkpoint",
        action="append",
        default=None,
        help="Mapping of label to checkpoint (e.g., aspirin=path/to.ckpt). Repeat for each label.",
    )
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Which split to evaluate.")
    parser.add_argument("--val-path", type=Path, default=None, help="Override val EXTXYZ path.")
    parser.add_argument("--test-path", type=Path, default=None, help="Override test EXTXYZ path.")
    parser.add_argument("--device", type=str, default="auto", help="Device choice: auto, cpu, or cuda.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override DataLoader workers.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of graphs evaluated (<=0 uses full set).",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to write metrics (defaults to results/<checkpoint_stem>_switch_eval.txt).",
    )
    return parser


def resolve_path(path_str: str | Path, root: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def extract_energy(atoms) -> float:
    if atoms.calc is not None:
        return float(atoms.get_potential_energy())
    for key in ("energy", "E", "total_energy"):
        if key in atoms.info:
            return float(atoms.info[key])
    raise ValueError("No energy found on atoms object.")


def extract_forces(atoms) -> np.ndarray:
    if atoms.calc is not None:
        return np.asarray(atoms.get_forces(), dtype=float)
    if "forces" in atoms.arrays:
        return np.asarray(atoms.arrays["forces"], dtype=float)
    raise ValueError("No forces found on atoms object.")


def extract_label(atoms) -> str:
    label = atoms.info.get("source_dataset") or atoms.info.get("dataset") or ""
    return str(label).strip().lower()


def read_extxyz_with_labels(paths: Sequence[Path]) -> Tuple[List[Molecule], List[float], List[np.ndarray], List[str]]:
    structures: List[Molecule] = []
    energies: List[float] = []
    forces: List[np.ndarray] = []
    labels: List[str] = []
    for path in paths:
        atoms_list = ase.io.read(path, index=":")
        if not isinstance(atoms_list, list):
            atoms_list = [atoms_list]
        for atoms in atoms_list:
            energies.append(extract_energy(atoms))
            forces.append(extract_forces(atoms))
            structures.append(Molecule(atoms.get_chemical_symbols(), atoms.get_positions()))
            labels.append(extract_label(atoms))
    return structures, energies, forces, labels


def build_dataset(
    name: str,
    structures: List[Molecule],
    energies: List[float],
    forces: List[np.ndarray],
    converter: Molecule2Graph,
    cache_dir: Path,
) -> MGLDataset:
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


def load_checkpoint_state(path: Path) -> Mapping[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, Mapping) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    if isinstance(ckpt, Mapping):
        return ckpt
    raise ValueError(f"Unsupported checkpoint format: {path}")


def embedding_keys(state_dict: Mapping[str, torch.Tensor]) -> List[str]:
    return [key for key in state_dict.keys() if "embedding" in key]


def extract_embedding_state(
    state_dict: Mapping[str, torch.Tensor],
    keys: Iterable[str],
) -> Dict[str, torch.Tensor]:
    selected: Dict[str, torch.Tensor] = {}
    for key in keys:
        if key in state_dict:
            selected[key] = state_dict[key].detach().clone()
    return selected


def apply_embedding_state(module: PotentialLightningModule, embed_state: Mapping[str, torch.Tensor]) -> None:
    if not embed_state:
        return
    module_state = module.state_dict()
    device = next(module.parameters()).device
    partial: Dict[str, torch.Tensor] = {}
    for key, tensor in embed_state.items():
        if key not in module_state:
            continue
        target = module_state[key]
        partial[key] = tensor.to(device=device, dtype=target.dtype)
    module.load_state_dict(partial, strict=False)


def match_label(label: str, keys: Sequence[str]) -> Optional[str]:
    label = label.lower()
    for key in keys:
        key_norm = key.lower()
        if label == key_norm or key_norm in label:
            return key
    return None


def bucket_labels(
    labels: Sequence[str],
    label_keys: Sequence[str],
    default_label: str,
) -> Tuple[Dict[str, List[int]], List[int]]:
    buckets: Dict[str, List[int]] = {key: [] for key in label_keys}
    unknown: List[int] = []
    for idx, label in enumerate(labels):
        key = match_label(label, label_keys)
        if key is None:
            unknown.append(idx)
            key = default_label
        buckets.setdefault(key, []).append(idx)
    return buckets, unknown


def accumulate_metrics(
    totals: Dict[str, float],
    results: Mapping[str, torch.Tensor | float | int],
    weight: int,
) -> None:
    for key, value in results.items():
        if isinstance(value, torch.Tensor):
            value = float(value.detach().cpu().item())
        else:
            value = float(value)
        totals[key] = totals.get(key, 0.0) + value * weight


def finalize_metrics(totals: Mapping[str, float], total_weight: float) -> Dict[str, float]:
    if total_weight <= 0:
        return {}
    return {key: val / total_weight for key, val in totals.items()}


def evaluate_subset(
    module: PotentialLightningModule,
    dataset: MGLDataset,
    indices: Sequence[int],
    collate_fn,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    limit: Optional[int],
) -> Tuple[Dict[str, float], float]:
    totals: Dict[str, float] = {}
    total_weight = 0.0
    if not indices:
        return totals, total_weight
    subset = Subset(dataset, list(indices))
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    for batch in loader:
        batch = tuple(item.to(device) if hasattr(item, "to") else item for item in batch)
        results, batch_size_actual = module.step(batch)
        accumulate_metrics(totals, results, batch_size_actual)
        total_weight += batch_size_actual
        if limit is not None and total_weight >= limit:
            break
    return totals, total_weight


def parse_source_checkpoints(args: argparse.Namespace) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    if args.ethanol_checkpoint:
        mapping[args.label_a.strip().lower()] = Path(args.ethanol_checkpoint)
    if args.malonaldehyde_checkpoint:
        mapping[args.label_b.strip().lower()] = Path(args.malonaldehyde_checkpoint)
    if args.source_checkpoint:
        for entry in args.source_checkpoint:
            if "=" not in entry:
                raise ValueError(f"--source-checkpoint expects label=path, got '{entry}'")
            label, path_str = entry.split("=", 1)
            label = label.strip().lower()
            path = Path(path_str.strip())
            if not label:
                raise ValueError("Empty label in --source-checkpoint.")
            mapping[label] = path
    if not mapping:
        raise ValueError("Provide at least one embedding checkpoint via --source-checkpoint.")
    return mapping


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    train_cfg = config.get("train", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})

    batch_size = int(args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 8))
    num_workers = int(args.num_workers if args.num_workers is not None else train_cfg.get("num_workers", 0))
    lr = float(train_cfg.get("lr", 1e-4))
    energy_weight = float(train_cfg.get("energy_weight", 1.0))
    force_weight = float(train_cfg.get("force_weight", 0.1))
    stress_weight = float(train_cfg.get("stress_weight", 0.0))
    decay_steps = int(train_cfg.get("decay_steps", 1000))
    decay_alpha = float(train_cfg.get("decay_alpha", 0.01))

    if args.split == "test":
        split_path_cfg = args.test_path or data_cfg.get("test_path")
        if split_path_cfg is None:
            raise ValueError("Config missing data.test_path; provide --test-path.")
    else:
        split_path_cfg = args.val_path or data_cfg.get("val_path")
        if split_path_cfg is None:
            raise ValueError("Config missing data.val_path; provide --val-path.")
    split_paths = split_path_cfg if isinstance(split_path_cfg, (list, tuple)) else [split_path_cfg]
    val_files = [resolve_path(path, M3GNET_ROOT) for path in split_paths]

    cache_dir = resolve_path(data_cfg.get("cache_dir", "data/cache/default"), M3GNET_ROOT)
    pretrained_name = model_cfg.get("pretrained_name", "M3GNet-ANI-1x-Subset-PES")
    cutoff = float(model_cfg.get("cutoff", 5.0))

    structures, energies, forces, labels = read_extxyz_with_labels(val_files)

    potential = load_model(pretrained_name)
    base_model = potential.model
    element_types = getattr(base_model, "element_types", None)
    if element_types is None:
        raise RuntimeError("Pretrained model did not expose element types.")
    converter = Molecule2Graph(element_types=element_types, cutoff=cutoff)

    dataset = build_dataset(
        name=args.split,
        structures=structures,
        energies=energies,
        forces=forces,
        converter=converter,
        cache_dir=cache_dir / f"{args.split}_switch",
    )
    collate = lambda batch: collate_fn_pes(batch, include_stress=False, include_line_graph=False)

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

    module = PotentialLightningModule.load_from_checkpoint(
        checkpoint_path=str(args.checkpoint),
        weights_only=False,
        **module_kwargs,
    )
    device = resolve_device(args.device)
    module.to(device)
    module.eval()

    merged_state = load_checkpoint_state(args.checkpoint)
    emb_keys = embedding_keys(merged_state)
    if not emb_keys:
        raise RuntimeError("No embedding keys found in merged checkpoint.")

    source_ckpts = parse_source_checkpoints(args)
    label_keys = list(source_ckpts.keys())
    default_label = label_keys[0]
    label_buckets, unknown_indices = bucket_labels(labels, label_keys, default_label)
    if unknown_indices:
        print(f"Warning: {len(unknown_indices)} samples had unmapped labels; using '{default_label}' embeddings.")

    label_embeds: Dict[str, Dict[str, torch.Tensor]] = {}
    for label, ckpt_path in source_ckpts.items():
        state = load_checkpoint_state(ckpt_path)
        label_embeds[label] = extract_embedding_state(state, emb_keys)
    limit = args.limit if args.limit and args.limit > 0 else None

    totals: Dict[str, float] = {}
    total_weight = 0.0
    per_label: Dict[str, Dict[str, float]] = {}

    for label, indices in label_buckets.items():
        if not indices:
            continue
        embed_state = label_embeds.get(label, label_embeds[default_label])
        apply_embedding_state(module, embed_state)

        remaining = None if limit is None else max(0, int(limit - total_weight))
        label_totals, label_weight = evaluate_subset(
            module,
            dataset,
            indices,
            collate,
            batch_size,
            num_workers,
            device,
            remaining,
        )
        if label_weight:
            per_label[label] = finalize_metrics(label_totals, label_weight)
            accumulate_metrics(totals, per_label[label], label_weight)
            total_weight += label_weight
        if limit is not None and total_weight >= limit:
            break

    metrics = finalize_metrics(totals, total_weight)
    split_prefix = "test" if args.split == "test" else "val"
    lines = [f"Embedding-switched evaluation for {args.checkpoint}:"]
    for key in sorted(metrics.keys()):
        lines.append(f"  {split_prefix}_{key}={metrics[key]:.6f}")
    for label in sorted(per_label.keys()):
        for key in sorted(per_label[label].keys()):
            lines.append(f"  {label}_{key}={per_label[label][key]:.6f}")

    print("\n".join(lines))
    save_path = args.save or Path("results") / f"{args.checkpoint.stem}_switch_eval.txt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
