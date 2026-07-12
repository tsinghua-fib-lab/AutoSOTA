"""
Fine-tune a pretrained M3GNet potential using MatGL (DGL backend).
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import sys
import time
import types
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import yaml
import ase.io
from pymatgen.core import Molecule

os.environ.setdefault("MATGL_BACKEND", "DGL")
os.environ.setdefault("DGLBACKEND", "pytorch")
os.environ.setdefault("DGL_SKIP_GRAPHBOLT", "1")

import lightning as pl
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

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
from matgl.graph.data import MGLDataset, MGLDataLoader, collate_fn_pes
from matgl.utils.training import PotentialLightningModule


M3GNET_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to M3GNet YAML config.")
    parser.add_argument("--workdir", type=Path, default=None, help="Override output run directory.")
    parser.add_argument("--device", type=str, default="auto", help="Device choice: auto, cpu, or cuda.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--plot", action="store_true", help="Save loss_curve.png under run_dir.")
    parser.add_argument("--plot-name", type=str, default="loss_curve.png", help="Plot filename.")
    parser.add_argument("--progress-bar", action="store_true", help="Enable Lightning progress bar.")
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


def read_loss_curves(csv_path: Path) -> Tuple[List[int], List[float], List[float]]:
    train_by_epoch = {}
    val_by_epoch = {}
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("epoch") in (None, ""):
                continue
            epoch = int(float(row["epoch"]))
            train_val = row.get("train_Total_Loss")
            if train_val not in (None, ""):
                train_by_epoch[epoch] = float(train_val)
            val_val = row.get("val_Total_Loss")
            if val_val not in (None, ""):
                val_by_epoch[epoch] = float(val_val)
    epochs = sorted(set(train_by_epoch) | set(val_by_epoch))
    train_loss = [train_by_epoch.get(epoch, float("nan")) for epoch in epochs]
    val_loss = [val_by_epoch.get(epoch, float("nan")) for epoch in epochs]
    return epochs, train_loss, val_loss


def plot_losses(epochs: List[int], train_loss: List[float], val_loss: List[float], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="train")
    plt.plot(epochs, val_loss, label="valid")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss (log)")
    plt.title("M3GNet Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


class EpochMetricsPrinter(Callback):
    """Print a compact metric summary after each validation epoch."""

    def __init__(self, decimals: int = 6) -> None:
        self.decimals = decimals

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        printable = {}
        for key, value in metrics.items():
            if not (key.startswith("train_") or key.startswith("val_")):
                continue
            try:
                if isinstance(value, torch.Tensor):
                    if value.numel() != 1:
                        continue
                    value = float(value.item())
                elif isinstance(value, (float, int)):
                    value = float(value)
                else:
                    continue
            except Exception:
                continue
            printable[key] = value
        if not printable:
            return
        epoch = trainer.current_epoch + 1
        ordered = sorted(
            printable.items(),
            key=lambda item: (0 if item[0].startswith("val_") else 1, item[0]),
        )
        formatted = " | ".join(f"{key}={value:.{self.decimals}f}" for key, value in ordered)
        print(f"Epoch {epoch} metrics: {formatted}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    train_cfg = config.get("train", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    output_cfg = config.get("output", {})

    epochs = args.epochs if args.epochs is not None else int(train_cfg.get("epochs", 3))
    batch_size = args.batch_size if args.batch_size is not None else int(train_cfg.get("batch_size", 8))
    lr = args.lr if args.lr is not None else float(train_cfg.get("lr", 1e-4))
    energy_weight = float(train_cfg.get("energy_weight", 1.0))
    force_weight = float(train_cfg.get("force_weight", 0.1))
    stress_weight = float(train_cfg.get("stress_weight", 0.0))
    decay_steps = int(train_cfg.get("decay_steps", 1000))
    decay_alpha = float(train_cfg.get("decay_alpha", 0.01))
    num_workers = int(train_cfg.get("num_workers", 0))
    seed = int(train_cfg.get("seed", 1))

    pl.seed_everything(seed, workers=True)

    train_path = resolve_path(data_cfg["train_path"], M3GNET_ROOT)
    val_path = resolve_path(data_cfg["val_path"], M3GNET_ROOT)
    cache_dir = resolve_path(data_cfg.get("cache_dir", "data/cache/default"), M3GNET_ROOT)

    run_dir = resolve_path(output_cfg.get("run_dir", "runs/m3gnet_run"), M3GNET_ROOT)
    if args.workdir is not None:
        run_dir = args.workdir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    pretrained_name = model_cfg.get("pretrained_name", "M3GNet-ANI-1x-Subset-PES")
    cutoff = float(model_cfg.get("cutoff", 5.0))

    potential = load_model(pretrained_name)
    base_model = potential.model
    element_types = getattr(base_model, "element_types", None)
    if element_types is None:
        raise RuntimeError("Pretrained model did not expose element types.")

    train_structures, train_energies, train_forces = read_extxyz(train_path)
    val_structures, val_energies, val_forces = read_extxyz(val_path)

    converter = Molecule2Graph(element_types=element_types, cutoff=cutoff)
    train_dataset = build_dataset(
        name="train",
        structures=train_structures,
        energies=train_energies,
        forces=train_forces,
        converter=converter,
        cache_dir=cache_dir / "train",
    )
    val_dataset = build_dataset(
        name="val",
        structures=val_structures,
        energies=val_energies,
        forces=val_forces,
        converter=converter,
        cache_dir=cache_dir / "val",
    )

    collate = lambda batch: collate_fn_pes(batch, include_stress=False, include_line_graph=False)
    train_loader, val_loader = MGLDataLoader(
        train_dataset,
        val_dataset,
        collate_fn=collate,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    element_refs = None
    if getattr(potential, "element_refs", None) is not None:
        element_refs = potential.element_refs.property_offset.detach().cpu().numpy()

    module = PotentialLightningModule(
        model=base_model,
        element_refs=element_refs,
        energy_weight=energy_weight,
        force_weight=force_weight,
        stress_weight=stress_weight,
        data_mean=float(potential.data_mean),
        data_std=float(potential.data_std),
        lr=lr,
        decay_steps=decay_steps,
        decay_alpha=decay_alpha,
    )

    accelerator, devices = resolve_trainer_settings(args.device)
    logger = CSVLogger(save_dir=str(run_dir), name="logs")
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(run_dir / "checkpoints"),
        save_last=True,
        monitor="val_Total_Loss",
        mode="min",
    )
    callbacks = [checkpoint_cb, EpochMetricsPrinter()]
    trainer = pl.Trainer(
        max_epochs=epochs,
        num_sanity_val_steps=0,
        accelerator=accelerator,
        devices=devices,
        default_root_dir=str(run_dir),
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=1,
        enable_progress_bar=args.progress_bar,
    )
    start_time = time.perf_counter()
    trainer.fit(module, train_loader, val_loader)
    elapsed = time.perf_counter() - start_time
    print(f"Training time: {elapsed:.2f}s ({elapsed / 60:.2f} min)")

    best_path = Path(checkpoint_cb.best_model_path) if checkpoint_cb.best_model_path else None
    if best_path and best_path.exists():
        best_out = run_dir / "best_chk.ckpt"
        shutil.copy2(best_path, best_out)
        print(f"Copied best checkpoint to {best_out}")
    else:
        print("Best checkpoint not found; skipping best_chk.ckpt copy.")

    model_dir = run_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    module.model.save(model_dir)

    shutil.copy2(args.config, run_dir / "config.yaml")

    metrics_path = Path(logger.log_dir) / "metrics.csv"
    if args.plot:
        if not metrics_path.exists():
            raise FileNotFoundError(f"Expected metrics CSV not found: {metrics_path}")
        epochs_list, train_loss, val_loss = read_loss_curves(metrics_path)
        plot_path = run_dir / args.plot_name
        plot_losses(epochs_list, train_loss, val_loss, plot_path)
        print(f"Saved loss curve to {plot_path}")


if __name__ == "__main__":
    main()
