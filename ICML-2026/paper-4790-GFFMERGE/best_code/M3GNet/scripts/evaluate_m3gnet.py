"""
Evaluate M3GNet checkpoints on the validation or test set defined in the config.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import types
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import yaml
import ase.io
from pymatgen.core import Molecule
from torch.utils.data import DataLoader

os.environ.setdefault("MATGL_BACKEND", "DGL")
os.environ.setdefault("DGLBACKEND", "pytorch")
os.environ.setdefault("DGL_SKIP_GRAPHBOLT", "1")

import lightning as pl

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
    parser.add_argument("--run-dir", type=Path, action="append", default=[], help="Run directory to evaluate.")
    parser.add_argument("--checkpoint", type=Path, action="append", default=[], help="Checkpoint path to evaluate.")
    parser.add_argument("--config", type=Path, default=None, help="Config YAML to use if run dir has none.")
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Which data split to evaluate (val or test).",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device choice: auto, cpu, or cuda.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override DataLoader workers.")
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional CSV output path.")
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


def read_extxyz(path: Path) -> Tuple[List[Molecule], List[float], List[np.ndarray]]:
    atoms_list = ase.io.read(path, index=":")
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    structures: List[Molecule] = []
    energies: List[float] = []
    forces: List[np.ndarray] = []
    for atoms in atoms_list:
        energies.append(extract_energy(atoms))
        forces.append(extract_forces(atoms))
        structures.append(Molecule(atoms.get_chemical_symbols(), atoms.get_positions()))
    return structures, energies, forces


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


def resolve_trainer_settings(device: str) -> Tuple[str, int]:
    if device == "auto":
        return ("gpu", 1) if dgl_cuda_ok() else ("cpu", 1)
    if device == "cuda":
        if dgl_cuda_ok():
            return ("gpu", 1)
        print("DGL CUDA backend unavailable; falling back to CPU.")
        return ("cpu", 1)
    return ("cpu", 1)


def load_eval_loader(
    config: dict,
    batch_size_override: int | None,
    num_workers_override: int | None,
    split: str,
) -> Tuple[DataLoader, dict]:
    train_cfg = config.get("train", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})

    batch_size = batch_size_override if batch_size_override is not None else int(train_cfg.get("batch_size", 8))
    num_workers = num_workers_override if num_workers_override is not None else int(train_cfg.get("num_workers", 0))
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

    structures: List[Molecule] = []
    energies: List[float] = []
    forces: List[np.ndarray] = []
    for path in split_files:
        s, e, f = read_extxyz(path)
        structures.extend(s)
        energies.extend(e)
        forces.extend(f)

    converter = Molecule2Graph(element_types=element_types, cutoff=cutoff)
    split_dataset = build_dataset(
        name=split,
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
    return eval_loader, module_kwargs


def gather_checkpoints(run_dir: Path) -> List[Path]:
    ckpts = list(run_dir.glob("*.ckpt"))
    ckpt_dir = run_dir / "checkpoints"
    if ckpt_dir.is_dir():
        ckpts.extend(ckpt_dir.glob("*.ckpt"))
    return sorted({ckpt.resolve() for ckpt in ckpts})


def format_metrics(metrics: Dict[str, float], split: str) -> str:
    def normalize_key(key: str) -> str:
        if split == "test" and key.startswith("val_"):
            return f"test_{key[4:]}"
        return key

    ordered = sorted(metrics.items(), key=lambda item: (0 if item[0].startswith("val_") else 1, item[0]))
    return " | ".join(f"{normalize_key(key)}={value:.6f}" for key, value in ordered)


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    run_dirs = [path.resolve() for path in args.run_dir]
    checkpoints = [path.resolve() for path in args.checkpoint]

    if not run_dirs and not checkpoints:
        raise SystemExit("Provide --run-dir and/or --checkpoint.")

    work_items: List[Tuple[Path, Path]] = []
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run dir not found: {run_dir}")
        for ckpt in gather_checkpoints(run_dir):
            work_items.append((run_dir, ckpt))

    for ckpt in checkpoints:
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        work_items.append((ckpt.parent, ckpt))

    if not work_items:
        raise SystemExit("No checkpoints found to evaluate.")

    cache: Dict[Path, Tuple[DataLoader, dict]] = {}
    results: List[Dict[str, str | float]] = []

    accelerator, devices = resolve_trainer_settings(args.device)
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        inference_mode=False,
    )

    for run_dir, ckpt in work_items:
        config_path = args.config
        if config_path is None:
            candidate = run_dir / "config.yaml"
            if candidate.exists():
                config_path = candidate
        if config_path is None or not config_path.exists():
            raise FileNotFoundError(
                f"Config not found for {ckpt}. Provide --config or add config.yaml to {run_dir}."
            )

        config_path = config_path.resolve()
        if config_path not in cache:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            val_loader, module_kwargs = load_eval_loader(
                config,
                batch_size_override=args.batch_size,
                num_workers_override=args.num_workers,
                split=args.split,
            )
            cache[config_path] = (val_loader, module_kwargs)
        else:
            val_loader, module_kwargs = cache[config_path]

        module = PotentialLightningModule.load_from_checkpoint(
            checkpoint_path=str(ckpt),
            weights_only=False,
            **module_kwargs,
        )
        metrics_list = trainer.validate(module, dataloaders=val_loader, verbose=False)
        metrics = metrics_list[0] if metrics_list else {}
        numeric_metrics = {
            key: (float(value) if not isinstance(value, torch.Tensor) else float(value.item()))
            for key, value in metrics.items()
            if isinstance(value, (float, int, torch.Tensor))
        }
        print(f"Checkpoint: {ckpt}")
        if numeric_metrics:
            print(format_metrics(numeric_metrics, args.split))
        else:
            print("No metrics returned.")
        record: Dict[str, str | float] = {"checkpoint": str(ckpt), "config": str(config_path)}
        record.update(numeric_metrics)
        results.append(record)

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames: List[str] = ["checkpoint", "config"]
        extra_keys = sorted({key for row in results for key in row.keys() if key not in fieldnames})
        fieldnames.extend(extra_keys)
        with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)


if __name__ == "__main__":
    main()
