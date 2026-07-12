"""
Fine-tune the M3GNet final energy readout + last N graph layers with embedding switching.

Embeddings are swapped based on the dataset label stored in EXTXYZ metadata
("source_dataset" or "dataset"). Loss = energy + force_weight * forces.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import types
import time
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


def collate_fn_no_stress(batch):
    return collate_fn_pes(batch, include_stress=False, include_line_graph=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Config YAML defining data/model settings.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint to fine-tune.")
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
    parser.add_argument("--train-path", type=Path, default=None, help="Override train EXTXYZ path.")
    parser.add_argument("--val-path", type=Path, default=None, help="Override val EXTXYZ path.")
    parser.add_argument("--output", type=Path, default=None, help="Output checkpoint path.")
    parser.add_argument("--device", type=str, default="auto", help="Device choice: auto, cpu, or cuda.")
    parser.add_argument("--epochs", type=int, default=20, help="Fine-tuning epochs.")
    parser.add_argument(
        "--last-n-blocks",
        type=int,
        default=1,
        help="Number of final graph_layers blocks to fine-tune (0 = only final_layer).",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override DataLoader workers.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Fine-tuning learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay for Adam.")
    parser.add_argument("--energy-weight", type=float, default=1.0, help="Energy loss weight.")
    parser.add_argument("--force-weight", type=float, default=0.1, help="Force loss weight.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of graphs per label (<=0 uses all).")
    parser.add_argument("--patience", type=int, default=0, help="Early stopping patience (0=disabled).")
    parser.add_argument("--grad-clip", type=float, default=0.0, help="Gradient clipping max norm (0=disabled).")
    parser.add_argument("--lr-schedule", type=str, default="constant", choices=["constant", "cosine"], help="LR schedule.")
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


def resolve_last_graph_layer_indices(module: PotentialLightningModule, last_n: int) -> List[int]:
    base_model = module.model
    if hasattr(base_model, "model"):
        base_model = base_model.model
    graph_layers = getattr(base_model, "graph_layers", None)
    if graph_layers is None:
        raise RuntimeError("Model does not expose graph_layers.")
    total_layers = len(graph_layers)
    if total_layers == 0:
        raise RuntimeError("Model has no graph_layers.")
    if last_n < 0:
        raise ValueError("--last-n-blocks must be >= 0.")
    if last_n == 0:
        return []
    if last_n > total_layers:
        print(f"Requested last {last_n} blocks; model has {total_layers}. Using all blocks.")
        last_n = total_layers
    return list(range(total_layers - last_n, total_layers))


def freeze_final_layer_and_last_blocks(module: PotentialLightningModule, last_n: int) -> List[str]:
    indices = resolve_last_graph_layer_indices(module, last_n)
    prefixes = [f"model.model.graph_layers.{idx}." for idx in indices]
    trainable: List[str] = []
    for name, param in module.named_parameters():
        if "model.model.final_layer" in name or any(prefix in name for prefix in prefixes):
            param.requires_grad = True
            trainable.append(name)
        else:
            param.requires_grad = False
    if not trainable:
        raise RuntimeError("No parameters selected to fine-tune.")
    return trainable


def evaluate_epoch_metrics(
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


def evaluate_split(
    module: PotentialLightningModule,
    label_loaders: Mapping[str, DataLoader],
    device: torch.device,
    label_embeds: Mapping[str, Mapping[str, torch.Tensor]],
    default_label: str,
) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    total_weight = 0.0
    was_training = module.training
    module.eval()

    for label, loader in label_loaders.items():
        embed_state = label_embeds.get(label, label_embeds[default_label])
        apply_embedding_state(module, embed_state)
        for batch in loader:
            batch = tuple(item.to(device) if hasattr(item, "to") else item for item in batch)
            results, batch_size_actual = module.step(batch)
            evaluate_epoch_metrics(totals, results, batch_size_actual)
            total_weight += batch_size_actual

    if was_training:
        module.train()

    if total_weight <= 0:
        return {}
    return {key: val / total_weight for key, val in totals.items()}


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    train_cfg = config.get("train", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})

    batch_size = int(args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 8))
    num_workers = int(args.num_workers if args.num_workers is not None else train_cfg.get("num_workers", 0))
    lr = float(args.lr)
    energy_weight = float(args.energy_weight)
    force_weight = float(args.force_weight)
    stress_weight = float(train_cfg.get("stress_weight", 0.0))
    decay_steps = int(train_cfg.get("decay_steps", 1000))
    decay_alpha = float(train_cfg.get("decay_alpha", 0.01))

    train_path_cfg = args.train_path or data_cfg.get("train_path")
    if train_path_cfg is None:
        raise ValueError("Config missing data.train_path; provide --train-path.")
    train_paths = train_path_cfg if isinstance(train_path_cfg, (list, tuple)) else [train_path_cfg]
    train_files = [resolve_path(path, M3GNET_ROOT) for path in train_paths]

    val_path_cfg = args.val_path or data_cfg.get("val_path")
    if val_path_cfg is None:
        raise ValueError("Config missing data.val_path; provide --val-path.")
    val_paths = val_path_cfg if isinstance(val_path_cfg, (list, tuple)) else [val_path_cfg]
    val_files = [resolve_path(path, M3GNET_ROOT) for path in val_paths]

    cache_dir = resolve_path(data_cfg.get("cache_dir", "data/cache/default"), M3GNET_ROOT)
    pretrained_name = model_cfg.get("pretrained_name", "M3GNet-ANI-1x-Subset-PES")
    cutoff = float(model_cfg.get("cutoff", 5.0))

    structures, energies, forces, labels = read_extxyz_with_labels(train_files)
    val_structures, val_energies, val_forces, val_labels = read_extxyz_with_labels(val_files)

    potential = load_model(pretrained_name)
    base_model = potential.model
    element_types = getattr(base_model, "element_types", None)
    if element_types is None:
        raise RuntimeError("Pretrained model did not expose element types.")
    converter = Molecule2Graph(element_types=element_types, cutoff=cutoff)

    dataset = build_dataset(
        name="train",
        structures=structures,
        energies=energies,
        forces=forces,
        converter=converter,
        cache_dir=cache_dir / "train_switch_last",
    )
    val_dataset = build_dataset(
        name="val",
        structures=val_structures,
        energies=val_energies,
        forces=val_forces,
        converter=converter,
        cache_dir=cache_dir / "val_switch_last",
    )
    collate = collate_fn_no_stress

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
        "lr": float(train_cfg.get("lr", 1e-4)),
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
    module.train()

    last_n = int(args.last_n_blocks)
    trainable = freeze_final_layer_and_last_blocks(module, last_n)
    if last_n > 0:
        desc = f"final_layer + last {last_n} graph layer(s)"
    else:
        desc = "final_layer only"
    print(f"Fine-tuning {len(trainable)} parameters ({desc}).")

    merged_state = load_checkpoint_state(args.checkpoint)
    emb_keys = embedding_keys(merged_state)
    if not emb_keys:
        raise RuntimeError("No embedding keys found in merged checkpoint.")

    source_ckpts = parse_source_checkpoints(args)
    label_keys = list(source_ckpts.keys())
    default_label = label_keys[0]
    label_buckets, unknown_train = bucket_labels(labels, label_keys, default_label)
    val_label_buckets, unknown_val = bucket_labels(val_labels, label_keys, default_label)
    if isinstance(label_buckets, tuple):
        label_buckets, unknown_train = label_buckets
    if isinstance(val_label_buckets, tuple):
        val_label_buckets, unknown_val = val_label_buckets
    if unknown_train or unknown_val:
        print(
            f"Warning: {len(unknown_train)} train and {len(unknown_val)} val samples had unmapped labels; "
            f"using '{default_label}' embeddings."
        )
    label_embeds: Dict[str, Dict[str, torch.Tensor]] = {}
    for label, ckpt_path in source_ckpts.items():
        state = load_checkpoint_state(ckpt_path)
        label_embeds[label] = extract_embedding_state(state, emb_keys)

    limit = args.limit if args.limit and args.limit > 0 else None

    train_loaders: Dict[str, DataLoader] = {}
    for label, indices in label_buckets.items():
        if not indices:
            continue
        use_indices = list(indices)
        if limit is not None and len(use_indices) > limit:
            use_indices = use_indices[:limit]
        subset = Subset(dataset, use_indices)
        train_loaders[label] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn_no_stress,
        )

    val_loaders: Dict[str, DataLoader] = {}
    for label, indices in val_label_buckets.items():
        if not indices:
            continue
        subset = Subset(val_dataset, list(indices))
        val_loaders[label] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn_no_stress,
        )

    optimizer = torch.optim.Adam(
        [param for param in module.parameters() if param.requires_grad],
        lr=lr,
        weight_decay=float(args.weight_decay),
    )

    scheduler = None
    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=lr / 100.0
        )

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_loss = float("inf")
    best_epoch = -1

    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        totals: Dict[str, float] = {}
        total_weight = 0.0
        for label, loader in train_loaders.items():
            embed_state = label_embeds.get(label, label_embeds[default_label])
            apply_embedding_state(module, embed_state)
            for batch in loader:
                batch = tuple(item.to(device) if hasattr(item, "to") else item for item in batch)
                optimizer.zero_grad(set_to_none=True)
                results, batch_size_actual = module.step(batch)
                loss = results["Total_Loss"]
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in module.parameters() if p.requires_grad],
                        max_norm=args.grad_clip,
                    )
                optimizer.step()

                evaluate_epoch_metrics(totals, results, batch_size_actual)
                total_weight += batch_size_actual

        if total_weight <= 0:
            raise RuntimeError("No training batches processed.")
        avg_metrics = {key: val / total_weight for key, val in totals.items()}
        report = " ".join(f"{key}={avg_metrics[key]:.6f}" for key in sorted(avg_metrics.keys()))

        val_metrics = evaluate_split(
            module=module,
            label_loaders=val_loaders,
            device=device,
            label_embeds=label_embeds,
            default_label=default_label,
        )
        val_report = " ".join(f"val_{key}={val_metrics[key]:.6f}" for key in sorted(val_metrics.keys()))
        print(f"[epoch {epoch:03d}] {report} {val_report}")

        val_loss = float(val_metrics.get("Total_Loss", float("inf")))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}

        if scheduler is not None:
            scheduler.step()

        if args.patience > 0 and (epoch - best_epoch) >= args.patience:
            print(f"Early stopping at epoch {epoch} (best was epoch {best_epoch}, val_loss={best_val_loss:.6f})")
            break

    elapsed = time.perf_counter() - start_time
    print(f"Fine-tuning time: {elapsed:.2f}s ({elapsed / 60:.2f} min)")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
    else:
        module.load_state_dict(best_state)
        print(f"Restored best state from epoch {best_epoch} (val_loss={best_val_loss:.6f})")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        output_ckpt = dict(ckpt)
        output_ckpt["state_dict"] = best_state
    else:
        output_ckpt = {"state_dict": best_state}
    output_ckpt["finetuned_from"] = str(args.checkpoint)
    output_ckpt["finetune_strategy"] = "switch_embeddings_final_layer_last_blocks"
    output_ckpt["finetune_epochs"] = args.epochs
    output_ckpt["finetune_force_weight"] = force_weight
    output_ckpt["finetune_energy_weight"] = energy_weight
    output_ckpt["finetune_best_epoch"] = best_epoch
    output_ckpt["finetune_best_val_loss"] = best_val_loss
    output_ckpt["finetune_last_n_graph_layers"] = last_n

    output_path = args.output
    if output_path is None:
        output_path = args.checkpoint.with_name(
            f"{args.checkpoint.stem}_ft_energy_last_block_ep{args.epochs}.ckpt"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_ckpt, output_path)
    print(f"Saved fine-tuned checkpoint to {output_path}")


if __name__ == "__main__":
    main()
