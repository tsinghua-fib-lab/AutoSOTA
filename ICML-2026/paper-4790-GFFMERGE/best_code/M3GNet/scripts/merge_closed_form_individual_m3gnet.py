"""
Closed-form merge of M3GNet checkpoints using per-dataset regression
on shared torch.nn.Linear layers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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


def collate_fn_no_stress(batch):
    return collate_fn_pes(batch, include_stress=False, include_line_graph=False)


class LinearAccumulator:
    def __init__(self, in_features: int, out_features: int, reg: float, device: torch.device):
        self.in_features = in_features
        self.out_features = out_features
        self.reg = reg
        self.device = device
        self.XtX = torch.zeros(in_features + 1, in_features + 1, dtype=torch.float64, device=self.device)
        self.XtY = torch.zeros(in_features + 1, out_features, dtype=torch.float64, device=self.device)
        self.count = 0

    def update(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        if inputs.numel() == 0:
            return
        X = inputs.to(device=self.device, dtype=torch.float64)
        Y = targets.to(device=self.device, dtype=torch.float64)
        n = X.shape[0]
        if n == 0:
            return
        x_sum = X.sum(dim=0)
        y_sum = Y.sum(dim=0)

        self.XtX[:-1, :-1] += X.T @ X
        self.XtX[:-1, -1] += x_sum
        self.XtX[-1, :-1] += x_sum
        self.XtX[-1, -1] += n

        self.XtY[:-1, :] += X.T @ Y
        self.XtY[-1, :] += y_sum
        self.count += n

    def solve(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if self.count == 0:
            return None
        reg_eye = self.reg * torch.eye(self.XtX.shape[0], dtype=torch.float64, device=self.device)
        theta = torch.linalg.solve(self.XtX + reg_eye, self.XtY)
        weight = theta[:-1].T.to(dtype=torch.float32, device="cpu")
        bias = theta[-1].to(dtype=torch.float32, device="cpu")
        return weight, bias

    def condition_number(self) -> float:
        if self.count == 0:
            return float("nan")
        reg_eye = self.reg * torch.eye(self.XtX.shape[0], dtype=torch.float64, device=self.device)
        matrix = self.XtX + reg_eye
        try:
            cond = torch.linalg.cond(matrix)
            if isinstance(cond, torch.Tensor):
                cond = cond.item()
            return float(cond)
        except RuntimeError:
            return float("inf")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", type=Path, help="Checkpoint to merge (repeatable).")
    parser.add_argument("--config", action="append", type=Path, help="Config YAML for each checkpoint (repeatable).")
    parser.add_argument("--train-path", action="append", type=Path, help="Override train EXTXYZ per checkpoint.")
    parser.add_argument("--checkpoint-a", type=Path, default=None, help="Checkpoint for dataset A (legacy).")
    parser.add_argument("--checkpoint-b", type=Path, default=None, help="Checkpoint for dataset B (legacy).")
    parser.add_argument("--config-a", type=Path, default=None, help="Config YAML for dataset A (legacy).")
    parser.add_argument("--config-b", type=Path, default=None, help="Config YAML for dataset B (legacy).")
    parser.add_argument("--train-a", type=Path, default=None, help="Override train EXTXYZ for dataset A (legacy).")
    parser.add_argument("--train-b", type=Path, default=None, help="Override train EXTXYZ for dataset B (legacy).")
    parser.add_argument("--output-ckpt", type=Path, required=True, help="Output merged checkpoint.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for regression.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument(
        "--accum-device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for accumulator matrices (auto uses model device).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Max samples per dataset (<=0 means all).",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=1e-6,
        help="Diagonal regularization for linear solve.",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device choice: auto, cpu, or cuda.")
    parser.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help="Optional JSONL log path for solver diagnostics.",
    )
    parser.add_argument(
        "--adaptive-reg",
        action="store_true",
        default=False,
        help="Use per-layer adaptive regularization based on condition numbers.",
    )
    return parser


def resolve_path(path_str: str | Path, root: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (root / path).resolve()


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


def resolve_accum_device(accum_device: str, model_device: torch.device) -> torch.device:
    if accum_device == "auto":
        return model_device
    if accum_device == "cuda":
        if dgl_cuda_ok():
            return torch.device("cuda")
        print("DGL CUDA backend unavailable; using CPU for accumulators.")
        return torch.device("cpu")
    return torch.device("cpu")


def expand_optional_list(
    values: Optional[Sequence[Path]],
    count: int,
    *,
    name: str,
    allow_broadcast: bool = True,
) -> List[Optional[Path]]:
    if not values:
        return [None] * count
    if allow_broadcast and len(values) == 1 and count > 1:
        return [values[0]] * count
    if len(values) != count:
        raise ValueError(f"{name} must have {count} entries; got {len(values)}.")
    return list(values)


def load_config(path: Path | None, checkpoint: Path) -> dict:
    if path is None:
        candidate = checkpoint.parent / "config.yaml"
        if candidate.exists():
            path = candidate
    if path is None or not path.exists():
        raise FileNotFoundError(
            f"Config not found for {checkpoint}. Provide --config or --config-a/--config-b."
        )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_loader(
    data_path: Path,
    converter: Molecule2Graph,
    cache_dir: Path,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    structures, energies, forces = read_extxyz(data_path)
    dataset = build_dataset(
        name=data_path.stem,
        structures=structures,
        energies=energies,
        forces=forces,
        converter=converter,
        cache_dir=cache_dir,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn_no_stress,
    )


def register_linear_hooks(
    model: torch.nn.Module,
    module_names: Iterable[str],
    linear_dims: Dict[str, Tuple[int, int]],
    accumulators: Dict[str, LinearAccumulator],
) -> List[torch.utils.hooks.RemovableHandle]:
    handles: List[torch.utils.hooks.RemovableHandle] = []
    module_name_set = set(module_names)
    for name, module in model.named_modules():
        if name not in module_name_set or not isinstance(module, torch.nn.Linear):
            continue
        in_features, out_features = linear_dims[name]
        accumulator = accumulators[name]

        def hook(
            module: torch.nn.Module,
            inputs: Tuple[torch.Tensor, ...],
            output: torch.Tensor,
            in_features: int = in_features,
            out_features: int = out_features,
            accumulator: LinearAccumulator = accumulator,
        ) -> None:
            inputs_tensor = inputs[0].detach()
            outputs_tensor = output.detach()

            if inputs_tensor.ndim > 2 or inputs_tensor.shape[-1] != in_features:
                inputs_tensor = inputs_tensor.reshape(-1, in_features)
            if outputs_tensor.ndim > 2 or outputs_tensor.shape[-1] != out_features:
                outputs_tensor = outputs_tensor.reshape(-1, out_features)

            accumulator.update(inputs_tensor, outputs_tensor)

        handles.append(module.register_forward_hook(hook))
    return handles


def run_teacher(
    loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    limit: Optional[int],
    register_hooks: callable,
) -> None:
    handles = register_hooks(model)
    processed = 0
    try:
        with torch.inference_mode():
            for batch in loader:
                g, lat, state_attr, *_ = batch
                g = g.to(device)
                lat = lat.to(device)
                state_attr = state_attr.to(device)
                model(g=g, lat=lat, state_attr=state_attr)
                processed += int(getattr(g, "batch_size", 1))
                if limit is not None and processed >= limit:
                    break
    finally:
        for handle in handles:
            handle.remove()


def accumulate_teacher(
    loader: DataLoader,
    model: torch.nn.Module,
    module_names: List[str],
    linear_dims: Dict[str, Tuple[int, int]],
    device: torch.device,
    accumulators: Dict[str, LinearAccumulator],
    limit: Optional[int],
) -> None:
    def register_hooks(target: torch.nn.Module) -> List[torch.utils.hooks.RemovableHandle]:
        return register_linear_hooks(target, module_names, linear_dims, accumulators)

    run_teacher(loader, model, device, limit, register_hooks)


def merge_mean(states: Sequence[Mapping[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not states:
        raise ValueError("No state dicts provided for merging.")
    merged: Dict[str, torch.Tensor] = {}
    key_sets = [set(state.keys()) for state in states]
    if any(key_sets[0] != ks for ks in key_sets[1:]):
        raise ValueError("State dict keys mismatch across checkpoints.")
    for key in states[0]:
        merged[key] = torch.stack([state[key] for state in states], dim=0).mean(dim=0)
    return merged


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.checkpoint:
        if args.checkpoint_a or args.checkpoint_b:
            raise ValueError("Use --checkpoint or --checkpoint-a/--checkpoint-b, not both.")
        checkpoints = list(args.checkpoint)
    else:
        if args.checkpoint_a is None or args.checkpoint_b is None:
            raise ValueError("Provide --checkpoint (repeatable) or both --checkpoint-a and --checkpoint-b.")
        checkpoints = [args.checkpoint_a, args.checkpoint_b]
    if len(checkpoints) < 2:
        raise ValueError("At least two checkpoints are required to merge.")

    if args.config:
        config_paths = expand_optional_list(args.config, len(checkpoints), name="--config")
    else:
        if args.config_a or args.config_b:
            if len(checkpoints) != 2:
                raise ValueError("--config-a/--config-b only supported with two checkpoints.")
            config_paths = [args.config_a, args.config_b]
        else:
            config_paths = [None] * len(checkpoints)

    if args.train_path:
        train_overrides = expand_optional_list(args.train_path, len(checkpoints), name="--train-path")
    else:
        if args.train_a or args.train_b:
            if len(checkpoints) != 2:
                raise ValueError("--train-a/--train-b only supported with two checkpoints.")
            train_overrides = [args.train_a, args.train_b]
        else:
            train_overrides = [None] * len(checkpoints)

    configs = [load_config(config_paths[i], checkpoints[i]) for i in range(len(checkpoints))]

    model_cfg0 = configs[0].get("model", {})
    pretrained_name = model_cfg0.get("pretrained_name", "M3GNet-ANI-1x-Subset-PES")
    cutoff = float(model_cfg0.get("cutoff", 5.0))
    for idx, cfg in enumerate(configs[1:], start=1):
        model_cfg = cfg.get("model", {})
        other_name = model_cfg.get("pretrained_name", "M3GNet-ANI-1x-Subset-PES")
        other_cutoff = float(model_cfg.get("cutoff", 5.0))
        if other_name != pretrained_name:
            raise ValueError(f"Checkpoint {checkpoints[idx]} uses pretrained_name={other_name}; expected {pretrained_name}.")
        if abs(other_cutoff - cutoff) > 1e-6:
            raise ValueError(f"Checkpoint {checkpoints[idx]} uses cutoff={other_cutoff}; expected {cutoff}.")

    train_paths: List[Path] = []
    cache_dirs: List[Path] = []
    for idx, cfg in enumerate(configs):
        data_cfg = cfg.get("data", {})
        train_path = train_overrides[idx] or resolve_path(data_cfg["train_path"], M3GNET_ROOT)
        train_paths.append(train_path)
        cache_root = resolve_path(data_cfg.get("cache_dir", "data/cache/merge"), M3GNET_ROOT)
        cache_dirs.append(cache_root / f"merge_{idx + 1}")

    device = resolve_device(args.device)
    accum_device = resolve_accum_device(args.accum_device, device)

    potential_ref = load_model(pretrained_name)
    base_model_ref = potential_ref.model
    element_types = getattr(base_model_ref, "element_types", None)
    if element_types is None:
        raise RuntimeError("Pretrained model did not expose element types.")
    converter = Molecule2Graph(element_types=element_types, cutoff=cutoff)

    loaders = [
        build_loader(
            data_path=train_paths[idx],
            converter=converter,
            cache_dir=cache_dirs[idx],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )
        for idx in range(len(checkpoints))
    ]

    teachers: List[PotentialLightningModule] = []
    linear_modules_list: List[Dict[str, torch.nn.Linear]] = []
    for idx, ckpt_path in enumerate(checkpoints):
        cfg = configs[idx]
        train_cfg = cfg.get("train", {})
        energy_weight = float(train_cfg.get("energy_weight", 1.0))
        force_weight = float(train_cfg.get("force_weight", 0.1))
        stress_weight = float(train_cfg.get("stress_weight", 0.0))
        lr = float(train_cfg.get("lr", 1e-4))
        decay_steps = int(train_cfg.get("decay_steps", 1000))
        decay_alpha = float(train_cfg.get("decay_alpha", 0.01))

        potential = potential_ref if idx == 0 else load_model(pretrained_name)
        base_model = potential.model
        model_element_types = getattr(base_model, "element_types", None)
        if model_element_types != element_types:
            raise RuntimeError(f"Checkpoint {ckpt_path} has mismatched element types.")

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
        teacher = PotentialLightningModule.load_from_checkpoint(
            checkpoint_path=str(ckpt_path),
            weights_only=False,
            **module_kwargs,
        )
        teacher.eval().to(device)
        if hasattr(teacher.model, "calc_forces"):
            teacher.model.calc_forces = False
        if hasattr(teacher.model, "calc_stresses"):
            teacher.model.calc_stresses = False
        if hasattr(teacher.model, "calc_hessian"):
            teacher.model.calc_hessian = False
        teachers.append(teacher)
        linear_modules_list.append(
            {name: module for name, module in teacher.model.named_modules() if isinstance(module, torch.nn.Linear)}
        )

    shared_names = set(linear_modules_list[0].keys())
    for modules in linear_modules_list[1:]:
        shared_names &= set(modules.keys())
    shared_names = sorted(shared_names)
    if not shared_names:
        raise RuntimeError("No shared torch.nn.Linear modules found between checkpoints.")

    linear_dims: Dict[str, Tuple[int, int]] = {}
    skipped: List[str] = []
    for name in shared_names:
        dims = {(modules[name].in_features, modules[name].out_features) for modules in linear_modules_list}
        if len(dims) != 1:
            skipped.append(name)
            continue
        in_features, out_features = dims.pop()
        linear_dims[name] = (in_features, out_features)

    if skipped:
        preview = ", ".join(skipped[:6])
        suffix = "..." if len(skipped) > 6 else ""
        print(f"Skipping {len(skipped)} mismatched Linear layers: {preview}{suffix}")

    accumulators = {
        name: LinearAccumulator(
            in_features=linear_dims[name][0],
            out_features=linear_dims[name][1],
            reg=args.regularization,
            device=accum_device,
        )
        for name in linear_dims
    }

    start_time = time.perf_counter()
    limit = None if args.limit <= 0 else args.limit
    module_names = list(linear_dims.keys())
    for loader, teacher in zip(loaders, teachers):
        accumulate_teacher(loader, teacher.model, module_names, linear_dims, device, accumulators, limit)

    solved: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    log_entries: List[Dict[str, object]] = []

    # Per-layer adaptive regularization based on condition numbers
    if args.adaptive_reg:
        conds = {}
        for name, accumulator in accumulators.items():
            conds[name] = accumulator.condition_number()
        valid_conds = [v for v in conds.values() if not (v != v or v == float("inf"))]  # exclude NaN and inf
        if valid_conds:
            import statistics
            median_cond = statistics.median(valid_conds)
            print(f"Adaptive regularization: median condition number = {median_cond:.2e}")
            for name, accumulator in accumulators.items():
                kappa = conds[name]
                if kappa == kappa and kappa != float("inf"):  # not NaN, not inf
                    adaptive_reg = max(accumulator.reg, accumulator.reg * kappa / median_cond)
                    accumulator.reg = adaptive_reg
                    print(f"  {name}: kappa={kappa:.2e} -> reg={adaptive_reg:.2e}")
                else:
                    print(f"  {name}: kappa={kappa} -> keeping base reg={accumulator.reg:.2e}")

    for name, accumulator in accumulators.items():
        entry = {
            "layer_type": "torch_linear",
            "module": name,
            "in_features": accumulator.in_features,
            "out_features": accumulator.out_features,
            "samples": accumulator.count,
            "regularization": accumulator.reg,
            "condition_number": accumulator.condition_number(),
        }
        result = accumulator.solve()
        if result is None:
            entry["skipped"] = True
        else:
            solved[name] = result
        log_entries.append(entry)

    ckpts = [torch.load(path, map_location="cpu", weights_only=False) for path in checkpoints]
    state_dicts = [ckpt.get("state_dict", ckpt) for ckpt in ckpts]
    merged_state = merge_mean(state_dicts)

    for name, (weight, bias) in solved.items():
        weight_key = f"model.{name}.weight"
        bias_key = f"model.{name}.bias"
        if weight_key in merged_state:
            merged_state[weight_key] = weight.to(merged_state[weight_key].dtype)
        if bias_key in merged_state:
            merged_state[bias_key] = bias.to(merged_state[bias_key].dtype)

    merged_ckpt = dict(ckpts[0]) if isinstance(ckpts[0], dict) else {"state_dict": merged_state}
    merged_ckpt["state_dict"] = merged_state
    merged_ckpt["merged_from"] = [str(path) for path in checkpoints]
    merged_ckpt["merge_strategy"] = "closed_form_linear"
    merged_ckpt["merge_num_models"] = len(checkpoints)
    merged_ckpt["merge_limit"] = args.limit
    merged_ckpt["merge_regularization"] = args.regularization

    elapsed = time.perf_counter() - start_time

    args.output_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged_ckpt, args.output_ckpt)
    print(f"Saved closed-form merged checkpoint to {args.output_ckpt}")

    if log_entries:
        log_path = args.log_path or args.output_ckpt.with_suffix(".log")
        with log_path.open("w", encoding="utf-8") as handle:
            for entry in log_entries:
                handle.write(json.dumps(entry) + "\n")
        print(f"Detailed solve log written to {log_path}")

    print(f"Closed-form individual merge compute time: {elapsed:.2f}s ({elapsed / 60:.2f} min)")


if __name__ == "__main__":
    main()
