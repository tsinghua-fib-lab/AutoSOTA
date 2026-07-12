"""
Fisher-weighted model merging for M3GNet / MatGL potentials.
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

from matgl import load_model
from matgl.ext._pymatgen_dgl import Molecule2Graph
from matgl.graph.data import MGLDataset, collate_fn_pes
from matgl.utils.training import PotentialLightningModule

M3GNET_ROOT = Path(__file__).resolve().parents[1]
if M3GNET_ROOT.name.lower() == "baselines":
    M3GNET_ROOT = M3GNET_ROOT.parent


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
        "--train-path",
        type=Path,
        required=True,
        help="EXTXYZ file used to approximate the Fisher (labels must include energy and forces).",
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
        help="Number of EXTXYZ structures to use for Fisher approximation.",
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
        choices=["auto", "cpu", "cuda"],
        help="Device for computation.",
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
        "--log-path",
        type=Path,
        default=None,
        help="Optional JSONL log path for diagnostics.",
    )

    return parser


# Data utilities


def read_extxyz(
    path: Path, limit: Optional[int] = None
) -> Tuple[List[Molecule], List[float], List[np.ndarray]]:
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
            continue

        mol = Molecule(atoms.get_chemical_symbols(), atoms.get_positions())
        structures.append(mol)
        energies.append(energy)
        forces.append(force)

    return structures, energies, forces


def build_dataset(
    structures: List[Molecule],
    energies: List[float],
    forces: List[np.ndarray],
    converter: Molecule2Graph,
    cache_dir: Path,
) -> MGLDataset:
    labels = {
        "energies": [float(e) for e in energies],
        "forces": [np.asarray(f, dtype=float).tolist() for f in forces],
    }

    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = MGLDataset(
        structures=structures,
        labels=labels,
        converter=converter,
        include_line_graph=False,
        directory_name="fisher_merge",
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


def build_module_from_config(
    cfg: dict,
    pretrained_name: str,
    state_dict: dict,
    device: torch.device,
) -> PotentialLightningModule:
    """Construct PotentialLightningModule and load given state_dict."""
    train_cfg = cfg.get("train", {})
    potential = load_model(pretrained_name)
    base_model = potential.model

    element_refs = None
    if getattr(potential, "element_refs", None) is not None:
        element_refs = potential.element_refs.property_offset.detach().cpu().numpy()

    force_weight = float(train_cfg.get("force_weight", 0.1))
    module = PotentialLightningModule(
        model=base_model,
        element_refs=element_refs,
        energy_weight=float(train_cfg.get("energy_weight", 1.0)),
        force_weight=force_weight,
        stress_weight=float(train_cfg.get("stress_weight", 0.0)),
        lr=float(train_cfg.get("lr", 1e-4)),
        decay_steps=int(train_cfg.get("decay_steps", 1000)),
        decay_alpha=float(train_cfg.get("decay_alpha", 0.01)),
        data_mean=float(getattr(potential, "data_mean", 0.0)),
        data_std=float(getattr(potential, "data_std", 1.0)),
    )

    module.load_state_dict(state_dict, strict=False)
    module.to(device)
    if hasattr(module.model, "calc_forces"):
        module.model.calc_forces = force_weight > 0.0
    if hasattr(module.model, "calc_stresses"):
        module.model.calc_stresses = False
    if hasattr(module.model, "calc_hessian"):
        module.model.calc_hessian = False
    return module

# Fisher computation and merging


def compute_fisher_diagonal(
    module: PotentialLightningModule,
    dataloader: DataLoader,
    max_batches: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    """Approximate diagonal Fisher using squared gradients of Total_Loss.
    F_theta ~= E[(dL/dtheta)^2] over the given dataset.
    """
    module.train()
    fisher: Dict[str, torch.Tensor] = {}
    n_steps = 0

    e_weight = float(getattr(module, "energy_weight", 1.0))
    f_weight = float(getattr(module, "force_weight", 0.0))

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        # collate_fn_pes(..., include_stress=False, include_line_graph=False)
        # in your other scripts is unpacked as (g, lat, state_attr, *rest)
        g, lat, state_attr, *rest = batch

        # Move graph and tensors to device
        g = g.to(device)
        lat = lat.to(device) if lat is not None else None
        state_attr = state_attr.to(device) if state_attr is not None else None

        # Extract targets from rest, handling dict or tensor cases
        target_energy = None
        target_forces = None

        for item in rest:
            if isinstance(item, dict):
                if target_energy is None:
                    target_energy = item.get("energies", item.get("e", None))
                if target_forces is None:
                    target_forces = item.get("forces", None)
            elif isinstance(item, (torch.Tensor, list, np.ndarray)):
                # First tensor-like we see, treat as energies if none found yet
                if target_energy is None:
                    target_energy = item

        if target_energy is None:
            # No usable labels in this batch
            continue

        target_energy = torch.as_tensor(
            target_energy, device=device, dtype=torch.float32
        ).view(-1, 1)

        if target_forces is not None:
            target_forces = torch.as_tensor(
                target_forces, device=device, dtype=torch.float32
            )

        # Forward through underlying potential model.
        out = module.model(g=g, lat=lat, state_attr=state_attr)
        if isinstance(out, (tuple, list)):
            e_pred = out[0]
            f_pred = out[1] if len(out) > 1 else None
        else:
            e_pred = out
            f_pred = None
        e_pred = e_pred.view(-1, 1)

        # Energy loss
        loss_e = F.mse_loss(e_pred, target_energy)

        # Optional force loss if both available and weighted
        if f_weight > 0.0 and f_pred is not None and target_forces is not None:
            loss_f = F.mse_loss(f_pred, target_forces)
            loss = e_weight * loss_e + f_weight * loss_f
        else:
            loss = e_weight * loss_e

        module.zero_grad(set_to_none=True)
        loss.backward()

        # Accumulate squared gradients (diagonal Fisher)
        for name, param in module.named_parameters():
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

    for name in fisher:
        fisher[name] /= float(n_steps)

    return fisher

def normalize_fisher_l2(fisher: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    total = 0.0
    for v in fisher.values():
        total += float((v**2).sum())
    norm = float(total) ** 0.5
    if norm < 1e-12:
        return fisher
    scale = 1.0 / norm
    return {k: v * scale for k, v in fisher.items()}


def fisher_merge_states(
    pretrained_state: Dict[str, torch.Tensor],
    finetuned_states: List[Dict[str, torch.Tensor]],
    fishers: List[Dict[str, torch.Tensor]],
    fisher_floor: float = 1e-6,
    favor_target_model: bool = True,
) -> Dict[str, torch.Tensor]:
    num_models = len(finetuned_states)
    if num_models == 0:
        return {}

    # Use key set from fine-tuned model 0 (Lightning-style: "model.<...>")
    base_keys = list(finetuned_states[0].keys())
    merged: Dict[str, torch.Tensor] = {}

    for key in base_keys:
        thetas = []
        fisher_vals = []

        for i in range(num_models):
            state_i = finetuned_states[i]
            # fallback to pretrained_state if key missing in some model
            if key in state_i:
                theta_i = state_i[key]
            else:
                theta_i = pretrained_state.get(key, None)
                if theta_i is None:
                    # skip this key entirely if it doesn't exist anywhere
                    break

            F_i = fishers[i].get(key, torch.zeros_like(theta_i, device="cpu"))
            thetas.append(theta_i)
            fisher_vals.append(F_i)

        if len(thetas) != num_models:
            # at least one model was missing this key, skip merging it
            continue

        # Stack Fishers and params on CPU
        theta_stack = torch.stack([t.cpu() for t in thetas], dim=0)  # [n_models, ...]
        fisher_stack = torch.stack(fisher_vals, dim=0)               # [n_models, ...]

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

        merged[key] = merged_param

    return merged


# Main


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    print("\n" + "=" * 70)
    print("Fisher Merging for M3GNet")
    print("=" * 70)
    print(f"Pretrained checkpoint: {args.pretrained_checkpoint}")
    print(f"Fine-tuned models: {len(args.checkpoint)}")
    for i, ckpt in enumerate(args.checkpoint, 1):
        print(f" [{i}] {ckpt}")
    print(f"Training data (Fisher): {args.train_path}")
    print(f"Num samples: {args.num_samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"Fisher floor: {args.fisher_floor}")
    print(f"Normalize Fishers: {args.normalize_fishers}")
    print(f"Favor target model (model 1) when Fisher small: {args.favor_target_model}")
    print(f"Output: {args.output_ckpt}")
    print("=" * 70)

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    # Handle config paths
    config_paths = args.config or [None] * len(args.checkpoint)
    if len(config_paths) == 1 and len(args.checkpoint) > 1:
        config_paths = config_paths * len(args.checkpoint)
    if len(config_paths) != len(args.checkpoint):
        raise ValueError("Number of --config must match number of --checkpoint")

    # Load pretrained config and state
    pretrained_cfg = load_config(None, args.pretrained_checkpoint)
    model_cfg = pretrained_cfg.get("model", {})
    pretrained_name = model_cfg.get("pretrained_name", "M3GNet-ANI-1x-Subset-PES")
    cutoff = float(model_cfg.get("cutoff", 5.0))

    print(f"\nLoading pretrained model: {pretrained_name}")
    pretrained_ckpt = torch.load(
        args.pretrained_checkpoint, map_location="cpu", weights_only=False
    )
    pretrained_state = get_state_dict_from_ckpt(pretrained_ckpt)

    # Build a reference module to get element types, etc.
    base_module_for_types = build_module_from_config(
        pretrained_cfg, pretrained_name, pretrained_state, device
    )
    element_types = getattr(base_module_for_types.model, "element_types", None)
    if element_types is None:
        potential_ref = load_model(pretrained_name)
        element_types = getattr(potential_ref.model, "element_types", None)
    if element_types is None:
        raise RuntimeError("Could not determine element types from model.")

    # Load fine-tuned checkpoints and build modules
    print(f"\nLoading {len(args.checkpoint)} fine-tuned checkpoints...")
    finetuned_states: List[Dict[str, torch.Tensor]] = []
    fisher_dicts: List[Dict[str, torch.Tensor]] = []
    all_finetuned_ckpts: List[dict] = []

    # Build Fisher dataset once (shared for all models)
    print(f"\nLoading Fisher data from {args.train_path}...")
    structures, energies, forces = read_extxyz(args.train_path, limit=args.num_samples)
    print(f" Loaded {len(structures)} structures with energy + forces")
    converter = Molecule2Graph(element_types=element_types, cutoff=cutoff)
    cache_dir = M3GNET_ROOT / "data" / "cache" / "fisher_merge"
    dataset = build_dataset(structures, energies, forces, converter, cache_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn_no_stress,
    )

    for i, ckpt_path in enumerate(args.checkpoint):
        print(f"\n[{i+1}] {ckpt_path.name}")
        cfg = load_config(config_paths[i], ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        all_finetuned_ckpts.append(ckpt)
        state = get_state_dict_from_ckpt(ckpt)
        finetuned_states.append(state)
        module = build_module_from_config(cfg, pretrained_name, state, device)

        print("  Computing diagonal Fisher approximation...")
        start = time.perf_counter()
        fisher = compute_fisher_diagonal(
            module, dataloader, max_batches=None, device=device
        )
        if args.normalize_fishers:
            fisher = normalize_fisher_l2(fisher)
        fisher_dicts.append(fisher)
        elapsed = time.perf_counter() - start
        print(f"  Fisher computed in {elapsed:.2f}s")

    # Perform Fisher merging
    print("\nMerging models with Fisher-weighted averaging...")
    merged_state = fisher_merge_states(
        pretrained_state=pretrained_state,
        finetuned_states=finetuned_states,
        fishers=fisher_dicts,
        fisher_floor=args.fisher_floor,
        favor_target_model=args.favor_target_model,
    )

    base_ckpt = all_finetuned_ckpts[0] if all_finetuned_ckpts else pretrained_ckpt
    merged_ckpt = copy.deepcopy(base_ckpt) if isinstance(base_ckpt, dict) else {}
    orig_state = merged_ckpt.get("state_dict", {})

    # Average normalization parameters (datamean/datastd) across fine-tuned models
    normparams: Dict[str, torch.Tensor] = {}
    for normkey in [
        "data_mean",
        "data_std",
        "model.data_mean",
        "model.data_std",
        "datamean",
        "datastd",
        "model.datamean",
        "model.datastd",
    ]:
        values = []
        for ckpt in all_finetuned_ckpts:
            sd = ckpt.get("state_dict", {})
            if normkey in sd:
                values.append(sd[normkey])
        if values:
            avg_value = sum(values) / len(values)
            normparams[normkey] = avg_value
            print(f"  {normkey} averaged from {len(values)} fine-tuned models")

    # Build new state_dict: keep non-tensor metadata, add norm params, then merged weights
    new_state: Dict[str, torch.Tensor] = {}

    for k, v in orig_state.items():
        if not isinstance(v, torch.Tensor):
            new_state[k] = v

    for k, v in normparams.items():
        new_state[k] = v

    # merged_state keys are already in Lightning style (e.g. "model.<...>")
    for k, v in merged_state.items():
        new_state[k] = v

    merged_ckpt["state_dict"] = new_state

    # Ensure Lightning-required keys are present
    if "pytorch-lightning_version" not in merged_ckpt:
        try:
            import lightning
            merged_ckpt["pytorch-lightning_version"] = lightning.__version__
        except ImportError:
            try:
                import pytorch_lightning
                merged_ckpt["pytorch-lightning_version"] = pytorch_lightning.__version__
            except ImportError:
                merged_ckpt["pytorch-lightning_version"] = "2.0.0"

    merged_ckpt["fisher_merge_info"] = {
        "fisher_floor": args.fisher_floor,
        "normalize_fishers": args.normalize_fishers,
        "favor_target_model": args.favor_target_model,
        "num_models": len(args.checkpoint),
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
    }
    merged_ckpt["merged_from"] = [str(p) for p in args.checkpoint]
    merged_ckpt["merge_strategy"] = "fisher_merging"
    merged_ckpt["merge_num_models"] = len(args.checkpoint)
    merged_ckpt["pretrained_checkpoint"] = str(args.pretrained_checkpoint)

    merged_ckpt.setdefault("epoch", 0)
    merged_ckpt.setdefault("global_step", 0)

    args.output_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged_ckpt, args.output_ckpt)
    print(f"\nSaved Fisher-merged checkpoint to {args.output_ckpt}")


    if args.log_path:
        args.log_path.parent.mkdir(parents=True, exist_ok=True)
        with args.log_path.open("w", encoding="utf-8") as f:
            meta = {
                "fisher_floor": args.fisher_floor,
                "normalize_fishers": args.normalize_fishers,
                "favor_target_model": args.favor_target_model,
                "checkpoints": [str(p) for p in args.checkpoint],
                "pretrained": str(args.pretrained_checkpoint),
            }
            f.write(json.dumps(meta) + "\n")
        print(f"Diagnostics written to {args.log_path}")

    print("\n" + "=" * 70)
    print("Fisher Merging Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
