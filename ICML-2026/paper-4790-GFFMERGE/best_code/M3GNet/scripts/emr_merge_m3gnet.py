"""
EMR-Merging for M3GNet / MatGL potentials.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import ase.io
from pymatgen.core import Molecule
from torch.utils.data import DataLoader

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

import lightning as pl

from matgl import load_model
from matgl.ext._pymatgen_dgl import Molecule2Graph
from matgl.graph.data import MGLDataset, collate_fn_pes
from matgl.utils.training import PotentialLightningModule

M3GNET_ROOT = Path(__file__).resolve().parents[1]
if M3GNET_ROOT.name.lower() == "baselines":
    M3GNET_ROOT = M3GNET_ROOT.parent


def resolve_path(path_str: str | Path, root: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (root / path).resolve()


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


def load_eval_loader(
    config: dict,
    batch_size: int,
    num_workers: int,
    split: str,
) -> Tuple[DataLoader, dict]:
    """Load evaluation dataloader and module kwargs from config."""
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


def evaluate_checkpoint(
    checkpoint_path: Path,
    config: dict,
    batch_size: int,
    num_workers: int,
    split: str,
    device: torch.device,
) -> Tuple[Dict[str, float], int]:
    """Evaluate a single checkpoint on its task's data."""
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
        checkpoint_path=str(checkpoint_path),
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


def collate_fn_no_stress(batch):
    return collate_fn_pes(batch, include_stress=False, include_line_graph=False)


# CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        required=True,
        help="Pretrained (base) Lightning checkpoint path.",
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
            "Config YAML for each fine-tuned checkpoint (repeatable). "
            "If not provided, looks for config.yaml next to each checkpoint."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for EMR-merged checkpoints (one per task).",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (not used for EMR, but kept for API consistency).",
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
        choices=["auto", "cpu", "cuda"],
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
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Which data split to evaluate (val or test). Only used with --evaluate.",
    )

    parser.add_argument(
        "--save-unified",
        action="store_true",
        help="Also save the unified EMR model (W_pre + tau_uni).",
    )

    return parser


# Data utilities


def read_extxyz(
    path: Path, limit: Optional[int] = None
) -> Tuple[List[Molecule], List[float], List[np.ndarray]]:
    """Read EXTXYZ and extract structures, energies, forces (not used for EMR, but kept for API consistency)."""
    index_str = f":{limit}" if limit is not None else ":"
    atoms_list = ase.io.read(str(path), index=index_str)
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]

    structures: List[Molecule] = []
    energies: List[float] = []
    forces: List[np.ndarray] = []

    for atoms in atoms_list:
        # Energy
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

        # Forces
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



# Model utilities


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


def load_config(path: Path | None, checkpoint: Path) -> dict:
    if path is None:
        candidate = checkpoint.parent / "config.yaml"
        if candidate.exists():
            path = candidate
    if path is None or not path.exists():
        raise FileNotFoundError(
            f"Config not found for {checkpoint}. Provide --config explicitly or add config.yaml."
        )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def get_state_dict_from_ckpt(ckpt: dict) -> Dict[str, torch.Tensor]:
    sd = ckpt.get("state_dict", None)
    if sd is None and isinstance(ckpt, dict):
        sd = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
    if sd is None:
        raise RuntimeError("Checkpoint did not contain a state_dict.")
    return sd


# EMR Merging Core


def is_param_key(k: str) -> bool:
    if not isinstance(k, str):
        return False
    exclude = [
        "datamean",
        "datastd",
        "data_mean",
        "data_std",
        "model.datamean",
        "model.datastd",
        "model.data_mean",
        "model.data_std",
    ]
    if any(k.startswith(p) for p in exclude):
        return False
    return "model." in k or ("." in k and not k.startswith("_"))


def average_norm_params(checkpoints: List[dict]) -> Dict[str, torch.Tensor]:
    norm_keys = [
        "data_mean",
        "data_std",
        "model.data_mean",
        "model.data_std",
        "datamean",
        "datastd",
        "model.datamean",
        "model.datastd",
    ]
    norm_params: Dict[str, torch.Tensor] = {}
    for norm_key in norm_keys:
        values = []
        for ckpt in checkpoints:
            sd = ckpt.get("state_dict", {})
            if norm_key in sd:
                values.append(sd[norm_key])
        if values:
            norm_params[norm_key] = sum(values) / len(values)
    return norm_params


def build_task_vectors(
    pretrained_state: Dict[str, torch.Tensor],
    finetuned_states: List[Dict[str, torch.Tensor]],
) -> Tuple[List[str], List[Dict[str, torch.Tensor]]]:
    # Use intersection of keys actually present in pretrained and all fine-tuned models
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
    num_models = len(task_vectors)
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
    merged = dict(pretrained_state)

    for k in keys:
        base = pretrained_state[k]
        tau_u = tau_uni[k].to(base.device, dtype=base.dtype)
        m_k = mask_i[k].to(base.device, dtype=base.dtype)
        tau_hat = lambda_i * (m_k * tau_u)
        merged[k] = base + tau_hat

    return merged


def save_emr_merged_checkpoints(
    pretrained_ckpt: dict,
    finetuned_ckpts: List[dict],
    keys: List[str],
    tau_uni: Dict[str, torch.Tensor],
    masks: List[Dict[str, torch.Tensor]],
    lambdas: List[float],
    output_dir: Path,
    output_names: Optional[List[str]] = None,
) -> List[Path]:
    pretrained_state = pretrained_ckpt["state_dict"]
    norm_params = average_norm_params(finetuned_ckpts)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for i, (ft_ckpt, Mi, lam) in enumerate(
        zip(finetuned_ckpts, masks, lambdas)
    ):
        # Reconstruct merged state
        merged_state = emr_reconstruct_state_for_task(
            pretrained_state=pretrained_state,
            keys=keys,
            tau_uni=tau_uni,
            mask_i=Mi,
            lambda_i=lam,
        )

        # Use fine-tuned ckpt as template for metadata & non-param tensors
        base_ckpt = copy.deepcopy(ft_ckpt)
        orig_state = base_ckpt.get("state_dict", {})

        new_state = {}
        # keep non-tensor entries from original state
        for k, v in orig_state.items():
            if not isinstance(v, torch.Tensor):
                new_state[k] = v
        # add averaged normalization parameters
        for k, v in norm_params.items():
            new_state[k] = v
        # add merged parameter tensors
        for k, v in merged_state.items():
            new_state[k] = v

        base_ckpt["state_dict"] = new_state

        # add EMR metadata
        base_ckpt["emr_info"] = {
            "lambda": float(lam),
            "task_index": i,
        }

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
    pretrained_ckpt: dict,
    finetuned_ckpts: List[dict],
    keys: List[str],
    tau_uni: Dict[str, torch.Tensor],
    output_dir: Path,
    output_name: str = "emr_unified.ckpt",
) -> Path:
    pretrained_state = pretrained_ckpt["state_dict"]
    norm_params = average_norm_params(finetuned_ckpts)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build unified state: W_uni = W_pre + tau_uni
    unified_state = dict(pretrained_state)  # shallow copy
    for k in keys:
        base = pretrained_state[k]
        tau_u = tau_uni[k].to(base.device, dtype=base.dtype)
        unified_state[k] = base + tau_u
    
    # Use first fine-tuned checkpoint as template for Lightning metadata
    base_ckpt = copy.deepcopy(finetuned_ckpts[0])
    orig_state = base_ckpt.get("state_dict", {})
    
    new_state = {}
    # Keep non-tensor entries from original state
    for k, v in orig_state.items():
        if not isinstance(v, torch.Tensor):
            new_state[k] = v
    # Add averaged normalization parameters
    for k, v in norm_params.items():
        new_state[k] = v
    # Add unified parameter tensors
    for k, v in unified_state.items():
        new_state[k] = v
    
    base_ckpt["state_dict"] = new_state
    
    # Add EMR metadata
    base_ckpt["emr_info"] = {
        "model_type": "unified",
        "description": "Universal EMR model using only tau_uni (no per-task masks/rescalers)",
    }
    
    out_path = output_dir / output_name
    torch.save(base_ckpt, out_path)
    print(f"  Saved universal EMR checkpoint to {out_path}")
    
    return out_path


# Main



def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    print("\n" + "=" * 70)
    print("EMR-Merging for M3GNet")
    print("=" * 70)
    print(f"Pretrained checkpoint: {args.pretrained_checkpoint}")
    print(f"Fine-tuned models: {len(args.checkpoint)}")
    for i, ckpt in enumerate(args.checkpoint, 1):
        print(f" [{i}] {ckpt}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 70)

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    # Handle config paths
    config_paths = args.config or [None] * len(args.checkpoint)
    if len(config_paths) == 1 and len(args.checkpoint) > 1:
        config_paths = config_paths * len(args.checkpoint)
    if len(config_paths) != len(args.checkpoint):
        raise ValueError("Number of --config must match number of --checkpoint")

    # Load pretrained checkpoint
    print(f"\nLoading pretrained checkpoint: {args.pretrained_checkpoint}")
    pretrained_ckpt = torch.load(
        args.pretrained_checkpoint, map_location="cpu", weights_only=False
    )
    pretrained_state = get_state_dict_from_ckpt(pretrained_ckpt)

    # Load fine-tuned checkpoints
    print(f"\nLoading {len(args.checkpoint)} fine-tuned checkpoints...")
    finetuned_ckpts: List[dict] = []
    finetuned_states: List[Dict[str, torch.Tensor]] = []

    for i, ckpt_path in enumerate(args.checkpoint):
        print(f"  [{i+1}] {ckpt_path.name}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        finetuned_ckpts.append(ckpt)
        state = get_state_dict_from_ckpt(ckpt)
        finetuned_states.append(state)

    # Build task vectors
    print("\nBuilding task vectors...")
    start = time.perf_counter()
    keys, task_vectors = build_task_vectors(pretrained_state, finetuned_states)
    print(f"  Parameter keys: {len(keys)}")
    print(f"  Task vectors built in {time.perf_counter() - start:.2f}s")

    # EMR elect: unified task vector
    print("\nElecting unified task vector...")
    start = time.perf_counter()
    tau_uni = emr_elect_unified(keys, task_vectors)
    print(f"  Unified task vector computed in {time.perf_counter() - start:.2f}s")

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
            pretrained_ckpt=pretrained_ckpt,
            finetuned_ckpts=finetuned_ckpts,
            keys=keys,
            tau_uni=tau_uni,
            output_dir=args.output_dir,
        )

    # Save EMR-merged checkpoints
    print("\nSaving EMR-merged checkpoints...")
    output_names = [f"emr_task_{i}.ckpt" for i in range(len(args.checkpoint))]
    saved_paths = save_emr_merged_checkpoints(
        pretrained_ckpt=pretrained_ckpt,
        finetuned_ckpts=finetuned_ckpts,
        keys=keys,
        tau_uni=tau_uni,
        masks=masks,
        lambdas=lambdas,
        output_dir=args.output_dir,
        output_names=output_names,
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
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    split=args.split,
                    device=device,
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
                    # Rename val_ to test_ if evaluating on test split
                    display_key = key.replace("val_", f"{args.split}_") if key.startswith("val_") else key
                    print(f"  {display_key}: {value:.6f}")

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
                display_key = key.replace("val_", f"{args.split}_") if key.startswith("val_") else key
                print(f"  {display_key}: {value:.6f}")

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
                "saved_task_specific": [str(p) for p in saved_paths],
            }
            if unified_path:
                meta["saved_unified"] = str(unified_path)
            if args.evaluate:
                meta["evaluation"] = {
                    "split": args.split,
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
        print(f"Aggregated {args.split} performance across {len(saved_paths)} tasks:")
        for key, value in sorted(aggregated_metrics.items()):
            display_key = key.replace("val_", f"{args.split}_") if key.startswith("val_") else key
            print(f"  {display_key}: {value:.6f}")
    else:
        print("\nEvaluate task-specific checkpoints with evaluate_m3gnet.py:")
        for i, path in enumerate(saved_paths):
            print(f"  python evaluate_m3gnet.py --checkpoint {path} --config <task{i}_config.yaml> --split test")
        if unified_path:
            print("\nEvaluate universal multi-task model:")
            print(f"  python evaluate_m3gnet.py --checkpoint {unified_path} --config config.yaml --split test")
    print("=" * 70)


if __name__ == "__main__":
    main()
