import argparse
import contextlib
import itertools
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import medmnist
import numpy as np
import torch
from medmnist import INFO
from torchvision import transforms

from escnn import nn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from escnn2 import gspaces
from escnn2.gspaces.utils import linear_transform_array_3d
from networks import (
    CNN3D,
    CNN3DResnet,
    PenalizedSteerableApprox3DResnet,
    SteerableCNN3D,
    SteerableCNN3DResnet,
    SteerableRPP3DResnet,
)
import networks.network_new as network_new_module


@dataclass
class FeatureRecord:
    flat: torch.Tensor
    geometric: list[Any] | None = None


@dataclass(frozen=True)
class CheckpointSpec:
    path: Path
    stage: str | None = None
    discovered_from: Path | None = None


@dataclass(frozen=True)
class EvaluationSpec:
    name: str
    description: str
    compare_mode: str


CHECKPOINT_STAGE_PATTERN = re.compile(r"_(init|final)\.pth$")
CHECKPOINT_RUN_ID_PATTERN = re.compile(r"_([a-z0-9]{8})(?:_(init|final))?\.pth$")
DEFAULT_CHECKS = ["hidden-equivariance", "output-invariance"]
EVALUATION_SPECS = {
    "hidden-equivariance": EvaluationSpec(
        name="hidden-equivariance",
        description="Final hidden representation equivariance",
        compare_mode="equivariant",
    ),
    "output-invariance": EvaluationSpec(
        name="output-invariance",
        description="Classifier output invariance",
        compare_mode="invariant",
    ),
}


class _IdentityProjectorR3(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def project(self, kernel: torch.Tensor) -> torch.Tensor:
        return kernel

    def projection_residual_norm(self, kernel: torch.Tensor) -> torch.Tensor:
        return kernel.new_zeros(())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check 3D checkpoint equivariance/invariance for the final hidden "
            "representation and classifier output. Inputs may be checkpoint files "
            "or directories containing init/final checkpoints."
        )
    )
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="+",
        help=(
            "Checkpoint files or directories. Directory inputs are scanned "
            "recursively for *_init.pth and *_final.pth files."
        ),
    )
    parser.add_argument(
        "--group",
        choices=["SO3", "O3"],
        default=None,
        help="Group to test. Defaults to the checkpoint group for steerable models.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Dataset split used for the evaluation batches.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for the equivariance check.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=4,
        help="Number of batches to evaluate.",
    )
    parser.add_argument(
        "--num-transforms",
        type=int,
        default=128,
        help="Number of random group elements to sample.",
    )
    transform_group = parser.add_mutually_exclusive_group()
    transform_group.add_argument(
        "--identity-only",
        action="store_true",
        help="Evaluate only the identity transform.",
    )
    transform_group.add_argument(
        "--right-angle-only",
        action="store_true",
        help=(
            "Evaluate only cube-symmetry rotations, i.e. rotations with "
            "3x3 action matrices containing only 0 and +/-1 entries."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device to use. Defaults to "cuda" when available, else "cpu".',
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help='Dataset root. Defaults to checkpoint config, then "$DATA_ROOT", then "data".',
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for transform sampling.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Allow medmnist to download missing dataset files.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Load the checkpoint with strict=True.",
    )
    parser.add_argument(
        "--stages",
        choices=["init", "final"],
        nargs="+",
        default=["init", "final"],
        help="Checkpoint stages to select when an input path is a directory.",
    )
    parser.add_argument(
        "--checks",
        choices=list(EVALUATION_SPECS),
        nargs="+",
        default=None,
        help=(
            "Checks to run for each selected checkpoint. Defaults to both final-hidden "
            "equivariance and output invariance."
        ),
    )
    parser.add_argument(
        "--full-init",
        action="store_true",
        help=(
            "Build all exact projector buffers during model construction. "
            "By default the checker skips these buffers for ApproxProj checkpoints "
            "because they are not needed for the forward pass."
        ),
    )
    parser.add_argument(
        "--compare-mode",
        choices=["auto", "equivariant", "invariant"],
        default="auto",
        help=(
            "Deprecated alias for selecting a single check when --checks is not set. "
            '"equivariant" maps to hidden-equivariance, "invariant" maps to '
            '"output-invariance".'
        ),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional JSON file for the aggregated results.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"[setup] Random seed set to {seed}.")


def get_default_data_root(config: dict[str, Any]) -> Path:
    if config.get("data_root"):
        return Path(config["data_root"])
    env_root = os.environ.get("DATA_ROOT")
    if env_root:
        return Path(env_root)
    return Path("data")


def load_checkpoint(path: Path) -> dict[str, Any]:
    print(f"[checkpoint] Loading checkpoint from {path} ...")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    print("[checkpoint] Checkpoint loaded.")
    return checkpoint


def infer_run_id(checkpoint_path: Path) -> str | None:
    match = CHECKPOINT_RUN_ID_PATTERN.search(checkpoint_path.name)
    return None if match is None else match.group(1)


def infer_checkpoint_stage(checkpoint_path: Path) -> str | None:
    match = CHECKPOINT_STAGE_PATTERN.search(checkpoint_path.name)
    return None if match is None else match.group(1)


def is_temporary_checkpoint(checkpoint_path: Path) -> bool:
    return "_TMP_" in checkpoint_path.name


def discover_checkpoints_in_directory(
    directory: Path,
    requested_stages: set[str],
) -> list[CheckpointSpec]:
    print(
        f"[discover] Scanning {directory} recursively for checkpoint files "
        f"(stages={sorted(requested_stages)}) ..."
    )
    candidates = sorted(path for path in directory.rglob("*.pth") if path.is_file())
    if not candidates:
        raise RuntimeError(f"No .pth checkpoints found under {directory}.")

    selected = [
        CheckpointSpec(path=path, stage=infer_checkpoint_stage(path), discovered_from=directory)
        for path in candidates
        if infer_checkpoint_stage(path) in requested_stages
    ]
    if selected:
        print(
            f"[discover] Selected {len(selected)} staged checkpoints from {directory}."
        )
        return selected

    if requested_stages == {"init", "final"}:
        legacy = [
            CheckpointSpec(path=path, stage=None, discovered_from=directory)
            for path in candidates
            if not is_temporary_checkpoint(path)
        ]
        if legacy:
            print(
                "[discover] No *_init/_final checkpoints found; falling back to "
                f"{len(legacy)} legacy checkpoint files."
            )
            return legacy

    raise RuntimeError(
        f"No checkpoints matching stages {sorted(requested_stages)} found under {directory}."
    )


def resolve_checkpoint_specs(
    inputs: list[Path],
    requested_stages: list[str],
) -> list[CheckpointSpec]:
    resolved: list[CheckpointSpec] = []
    seen: set[Path] = set()

    for raw_path in inputs:
        path = raw_path.expanduser()
        if path.is_file():
            spec = CheckpointSpec(path=path, stage=infer_checkpoint_stage(path))
            if spec.path not in seen:
                resolved.append(spec)
                seen.add(spec.path)
            continue
        if path.is_dir():
            for spec in discover_checkpoints_in_directory(path, set(requested_stages)):
                if spec.path not in seen:
                    resolved.append(spec)
                    seen.add(spec.path)
            continue
        raise FileNotFoundError(f"Input path does not exist: {path}")

    if not resolved:
        raise RuntimeError("No checkpoints were resolved from the provided inputs.")

    print(f"[discover] Resolved {len(resolved)} checkpoint(s) for evaluation.")
    for index, spec in enumerate(resolved, start=1):
        stage = spec.stage or "legacy"
        print(f"[discover] {index:03d}. stage={stage} path={spec.path}")
    return resolved


def resolve_requested_checks(args: argparse.Namespace) -> list[str]:
    if args.checks is not None:
        requested = args.checks
    elif args.compare_mode == "equivariant":
        requested = ["hidden-equivariance"]
    elif args.compare_mode == "invariant":
        requested = ["output-invariance"]
    else:
        requested = list(DEFAULT_CHECKS)

    ordered_unique = list(dict.fromkeys(requested))
    print(f"[main] Requested checks: {ordered_unique}.")
    return ordered_unique


def load_wandb_config_from_run_id(root: Path, run_id: str) -> dict[str, Any] | None:
    matches = sorted(root.glob(f"wandb/run-*-{run_id}/files/config.yaml"))
    if not matches:
        print(f"[config] No local W&B config found for run id {run_id}.")
        return None
    print(f"[config] Recovering config from {matches[0]} ...")
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "Checkpoint is missing an embedded config and PyYAML is not available "
            "to read the matching local W&B config."
        ) from exc

    with matches[0].open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    config: dict[str, Any] = {}
    for key, value in raw.items():
        if key == "_wandb":
            continue
        if isinstance(value, dict) and "value" in value:
            config[key] = value["value"]
        else:
            config[key] = value
    print("[config] Local W&B config loaded.")
    return config


def recover_config(checkpoint: dict[str, Any], checkpoint_path: Path) -> dict[str, Any]:
    config = checkpoint.get("config")
    if isinstance(config, dict):
        print("[config] Using config embedded in checkpoint.")
        return dict(config)

    run_id = infer_run_id(checkpoint_path)
    if run_id is None:
        raise RuntimeError(
            "Checkpoint does not contain a config and no run id could be inferred from the filename."
        )

    wandb_config = load_wandb_config_from_run_id(ROOT, run_id)
    if wandb_config is None:
        raise RuntimeError(
            f"Checkpoint does not contain a config and no local W&B config was found for run id {run_id}."
        )
    print(f"[config] Recovered config via local W&B run id {run_id}.")
    return wandb_config


def normalise_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(config)
    defaults = {
        "activation": "gated",
        "approx": False,
        "approx_ours": False,
        "basic_wd": 0.0,
        "channels": 6,
        "cnn_match_group": "SO3",
        "conv_wd": 0.0,
        "invariant": True,
        "iteration": 0,
        "L_in": 2,
        "L_out": 4,
        "learn_eq": False,
        "mnist_type": "single",
        "new": False,
        "normalization": True,
        "one_eq": False,
        "penalized_approx": False,
        "projector_basis_frac": None,
        "projector_basis_sample": "first",
        "projector_basis_seed": 0,
        "projector_max_basis": None,
        "ratio": 1.0,
        "resnet": False,
        "restrict": False,
        "rpp": False,
    }
    for key, value in defaults.items():
        cfg.setdefault(key, value)

    dataset = cfg.get("dataset")
    if not dataset:
        raise RuntimeError("Missing dataset in checkpoint config.")
    if "3d" not in dataset:
        raise RuntimeError(f"Only 3D checkpoints are supported, found dataset={dataset}.")

    info = INFO[dataset]
    cfg.setdefault("n_channels", info["n_channels"])
    cfg.setdefault("n_classes", len(info["label"]))
    cfg.setdefault("group", "CNN")
    print(
        "[config] Normalised config:"
        f" dataset={cfg['dataset']}, group={cfg['group']},"
        f" channels={cfg['channels']}, n_classes={cfg['n_classes']}."
    )
    return cfg


def activation_from_name(name: str):
    if name == "gated":
        return nn.GatedNonLinearityUniform
    if name == "QFourier":
        return nn.QuotientFourierELU
    if name == "Fourier":
        return nn.FourierELU
    raise ValueError(f"Unsupported activation setting: {name}")


def resolve_model_class(config: dict[str, Any]):
    network_name = str(config.get("network_name", ""))
    group_name = str(config.get("group", "CNN"))

    if group_name == "CNN":
        return CNN3DResnet if config.get("resnet") else CNN3D
    if "RPP" in network_name or config.get("rpp"):
        return SteerableRPP3DResnet
    if (
        "PenalizedSteerableApprox3DResnet" in network_name
        or config.get("penalized_approx")
        or config.get("approx")
    ):
        return PenalizedSteerableApprox3DResnet
    if config.get("resnet"):
        return SteerableCNN3DResnet
    return SteerableCNN3D


@contextlib.contextmanager
def maybe_fast_init_context(
    cls,
    *,
    full_init: bool,
):
    if full_init:
        yield False
        return
    original_projector = network_new_module.ExactKernelProjectorR3
    network_new_module.ExactKernelProjectorR3 = _IdentityProjectorR3
    try:
        yield True
    finally:
        network_new_module.ExactKernelProjectorR3 = original_projector


def create_model(
    config: dict[str, Any],
    device: torch.device,
    *,
    full_init: bool,
) -> torch.nn.Module:
    cls = resolve_model_class(config)
    activation = activation_from_name(config["activation"])
    print(f"[model] Building {cls.__name__} on device {device} ...")
    start = time.perf_counter()

    with maybe_fast_init_context(cls, full_init=full_init) as fast_init_enabled:
        if fast_init_enabled:
            print(
                "[model] Fast init enabled: skipping ExactKernelProjectorR3 "
                "construction for ApproxProj evaluation."
            )

        if cls in {CNN3D, CNN3DResnet}:
            kwargs = dict(
                mnist_type=config["mnist_type"],
                n_classes=config["n_classes"],
                n_channels=config["n_channels"],
                c=config["channels"],
            )
            model = cls(**kwargs)
            elapsed = time.perf_counter() - start
            print(f"[model] Model instantiated in {elapsed:.2f}s.")
            return model.to(device)

        kwargs = dict(
            mnist_type=config["mnist_type"],
            restrict=config["restrict"],
            n_classes=config["n_classes"],
            n_channels=config["n_channels"],
            learn_eq=config["learn_eq"],
            normalise_basis=config["normalization"],
            one_eq=config["one_eq"],
            channels=config["channels"],
            iteration=config["iteration"],
            L_in=config["L_in"],
            L_out=config["L_out"],
            activation=activation,
            invariant=config["invariant"],
        )

        if cls is PenalizedSteerableApprox3DResnet:
            kwargs.update(
                conv_wd=config.get("conv_wd", 0.0),
                basic_wd=config.get("basic_wd", 0.0),
            )

        model = cls.from_group(config["group"], **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[model] Model instantiated in {elapsed:.2f}s.")
        return model.to(device)


def load_state_dict(
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    strict: bool,
) -> tuple[list[str], list[str]]:
    state_dict = checkpoint["model_state_dict"]
    print(f"[model] Loading state dict with strict={strict} ...")
    start = time.perf_counter()
    result = model.load_state_dict(state_dict, strict=strict)
    if strict:
        elapsed = time.perf_counter() - start
        print(f"[model] State dict loaded successfully in {elapsed:.2f}s.")
        return [], []
    elapsed = time.perf_counter() - start
    print(
        "[model] State dict loaded."
        f" time={elapsed:.2f}s,"
        f" missing_keys={len(result.missing_keys)},"
        f" unexpected_keys={len(result.unexpected_keys)}."
    )
    return list(result.missing_keys), list(result.unexpected_keys)


def build_loader(
    config: dict[str, Any],
    split: str,
    batch_size: int,
    num_workers: int,
    data_root: Path,
    download: bool,
) -> torch.utils.data.DataLoader:
    info = INFO[config["dataset"]]
    data_class = getattr(medmnist, info["python_class"])
    print(
        f"[data] Building {config['dataset']} {split} loader from "
        f"{Path(data_root) / 'medmnist'} ..."
    )

    if "3d" in config["dataset"]:
        transform = transforms.Compose(
            [
                lambda x: torch.FloatTensor(x),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
    else:
        raise RuntimeError("This script only supports 3D datasets.")

    dataset = data_class(
        split=split,
        transform=transform,
        download=download,
        root=os.path.join(str(data_root), "medmnist"),
    )
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    print(
        f"[data] Loader ready: {len(dataset)} samples, batch_size={batch_size}, "
        f"num_workers={num_workers}."
    )
    return loader


class PenultimateExtractor:
    def __init__(self, model: torch.nn.Module):
        self.model = model

    def __call__(self, x: torch.Tensor) -> FeatureRecord:
        if hasattr(self.model, "layers_eq") and hasattr(self.model, "full_net"):
            return self._forward_steerable(x)
        if isinstance(self.model, (CNN3D)):
            return self._forward_cnn3d(x)
        if isinstance(self.model, (CNN3DResnet)):
            return self._forward_cnn3d_resnet(x)
        raise TypeError(f"Unsupported model type: {type(self.model).__name__}")

    def _forward_steerable(self, x: torch.Tensor) -> FeatureRecord:
        gx = self.model.in_type(x)
        outs = [layers(gx) for layers in self.model.layers_eq]
        flat = torch.cat([out.tensor for out in outs], dim=1).reshape(x.shape[0], -1)
        return FeatureRecord(flat=flat, geometric=outs)

    def _forward_cnn3d(self, x: torch.Tensor) -> FeatureRecord:
        x = self.model.upsample(x)
        x = self.model.block_1(x)
        x = self.model.block_2(x)
        x = self.model.pool_1(x)
        x = self.model.block_3(x)
        x = self.model.pool_2(x)
        x = self.model.block_4(x)
        x = self.model.block_5(x)
        x = self.model.pool_3(x)
        x = self.model.block_6(x)
        flat = x.reshape(x.shape[0], -1)
        return FeatureRecord(flat=flat)

    def _forward_cnn3d_resnet(self, x: torch.Tensor) -> FeatureRecord:
        x = self.model.upsample(x)
        x = self.model.resblock_1(x)
        x = self.model.pool_1(x)
        x = self.model.resblock_2(x)
        x = self.model.resblock_3(x)
        flat = x.reshape(x.shape[0], -1)
        return FeatureRecord(flat=flat)


class OutputExtractor:
    def __init__(self, model: torch.nn.Module):
        self.model = model

    def __call__(self, x: torch.Tensor) -> FeatureRecord:
        out = self.model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if hasattr(out, "tensor"):
            out = out.tensor
        flat = out.reshape(out.shape[0], -1)
        return FeatureRecord(flat=flat)


def validate_requested_group(config: dict[str, Any], requested_group: str) -> None:
    model_group = config["group"]
    if model_group == "CNN":
        return
    if requested_group == model_group:
        return
    if model_group == "O3" and requested_group == "SO3":
        return
    raise ValueError(
        f"Cannot test {requested_group} equivariance for a checkpoint built for {model_group}."
    )


def get_sampling_gspace(model: torch.nn.Module, requested_group: str):
    if hasattr(model, "act_r2"):
        return model.act_r2
    if requested_group == "SO3":
        return gspaces.rot3dOnR3(maximum_frequency=6)
    return gspaces.flipRot3dOnR3(maximum_frequency=6)


def _matrix_key(matrix: np.ndarray, decimals: int = 6) -> tuple[float, ...]:
    return tuple(np.round(matrix.reshape(-1), decimals=decimals).tolist())


def _target_right_angle_matrices() -> dict[tuple[float, ...], np.ndarray]:
    targets: dict[tuple[float, ...], np.ndarray] = {}
    eye = np.eye(3, dtype=np.float64)
    for perm in itertools.permutations(range(3)):
        permuted = eye[:, perm]
        for signs in itertools.product([-1.0, 1.0], repeat=3):
            matrix = permuted * np.asarray(signs, dtype=np.float64)
            if np.linalg.det(matrix) > 0:
                targets[_matrix_key(matrix)] = matrix
    return targets


def _collect_candidate_elements(
    gspace,
    requested_group: str,
    num_transforms: int,
) -> list[tuple[Any, np.ndarray]]:
    group = gspace.fibergroup
    candidates: list[Any] = []

    identity = getattr(group, "identity", None)
    if identity is not None:
        candidates.append(identity)

    testing_elements = getattr(group, "testing_elements", None)
    if callable(testing_elements):
        for kwargs in (
            {"n": max(num_transforms, 256)},
            {},
        ):
            try:
                elems = list(testing_elements(**kwargs))
            except TypeError:
                continue
            except Exception:
                continue
            candidates.extend(elems)

    grid = getattr(group, "grid", None)
    if callable(grid):
        for args, kwargs in (
            (("cube",), {}),
            ((), {"type": "cube"}),
        ):
            try:
                elems = list(grid(*args, **kwargs))
            except TypeError:
                continue
            except Exception:
                continue
            candidates.extend(elems)

    unique: list[tuple[Any, np.ndarray]] = []
    seen: set[tuple[float, ...]] = set()
    for element in candidates:
        matrix = np.asarray(gspace.basespace_action(element), dtype=np.float64)
        det = np.linalg.det(matrix)
        if requested_group == "SO3" and det < 0:
            continue
        key = _matrix_key(matrix)
        if key in seen:
            continue
        seen.add(key)
        unique.append((element, matrix))
    return unique


def _identity_element(gspace, requested_group: str, num_transforms: int) -> Any:
    identity_matrix = np.eye(3, dtype=np.float64)
    candidates = _collect_candidate_elements(gspace, requested_group, num_transforms)
    for element, matrix in candidates:
        if np.allclose(matrix, identity_matrix, atol=1e-6):
            print("[group] Using identity transform only.")
            return element

    identity = getattr(gspace.fibergroup, "identity", None)
    if identity is not None:
        print("[group] Using fibergroup identity element.")
        return identity

    raise RuntimeError("Could not recover the identity group element.")


def _right_angle_elements(
    gspace,
    requested_group: str,
    num_transforms: int,
) -> list[Any]:
    targets = _target_right_angle_matrices()
    candidates = _collect_candidate_elements(gspace, requested_group, num_transforms)

    matches: dict[tuple[float, ...], Any] = {}
    for element, matrix in candidates:
        for target_key, target_matrix in targets.items():
            if np.allclose(matrix, target_matrix, atol=1e-6):
                matches[target_key] = element
                break

    if len(matches) != len(targets):
        raise RuntimeError(
            "Could not recover the full set of 24 right-angle SO(3) rotations "
            f"from the group API. Found {len(matches)} / {len(targets)}."
        )

    print("[group] Using the 24 right-angle rotation elements only.")
    return [matches[target_key] for target_key in sorted(matches)]


def sample_group_elements(
    gspace,
    requested_group: str,
    num_transforms: int,
    *,
    identity_only: bool = False,
    right_angle_only: bool = False,
) -> list[Any]:
    if identity_only:
        return [_identity_element(gspace, requested_group, num_transforms)]

    if right_angle_only:
        return _right_angle_elements(gspace, requested_group, num_transforms)

    print(
        f"[group] Sampling {num_transforms} random elements from {requested_group} ..."
    )
    elements = []
    while len(elements) < num_transforms:
        element = gspace.fibergroup.sample()
        matrix = np.asarray(gspace.basespace_action(element), dtype=np.float64)
        det = np.linalg.det(matrix)
        if requested_group == "SO3" and det < 0:
            continue
        elements.append(element)
        print(
            f"[group] Sampled element {len(elements)}/{num_transforms} "
            f"(det={det:.6f})."
        )
    print("[group] Finished sampling group elements.")
    return elements


def transform_input(
    model: torch.nn.Module,
    x: torch.Tensor,
    element: Any,
    gspace,
) -> torch.Tensor:
    if hasattr(model, "in_type") and hasattr(model, "layers_eq"):
        return model.in_type(x).transform(element).tensor

    matrix = np.asarray(gspace.basespace_action(element), dtype=np.float64)
    x_np = x.detach().cpu().numpy()
    transformed = np.stack(
        [linear_transform_array_3d(sample, matrix, exact=True, order=2) for sample in x_np],
        axis=0,
    )
    transformed = np.ascontiguousarray(transformed)
    return torch.from_numpy(transformed).to(device=x.device, dtype=x.dtype)


def transform_reference_features(
    record: FeatureRecord,
    element: Any,
    compare_mode: str,
) -> torch.Tensor:
    if compare_mode == "invariant" or not record.geometric:
        return record.flat

    transformed_parts = []
    for geometric in record.geometric:
        tensor_shape = tuple(geometric.tensor.shape[2:])
        if tensor_shape == (1, 1, 1) and hasattr(geometric, "transform_fibers"):
            transformed = geometric.transform_fibers(element)
        else:
            transformed = geometric.transform(element)
        transformed_parts.append(transformed.tensor)
    return torch.cat(transformed_parts, dim=1).reshape(record.flat.shape)


def compute_absolute_errors(
    predicted: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    diff = predicted - reference
    diff_norm = torch.linalg.norm(diff, dim=1)
    return diff_norm


def compute_relative_errors(
    predicted: torch.Tensor,
    reference: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    abs_err = compute_absolute_errors(predicted, reference)
    ref_norm = torch.linalg.norm(reference, dim=1)
    return abs_err / (ref_norm + eps)


def evaluate_equivariance(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    elements: list[Any],
    gspace,
    extractor: Callable[[torch.Tensor], FeatureRecord],
    compare_mode: str,
    max_batches: int,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    detailed: list[dict[str, Any]] = []
    abs_errors: list[np.ndarray] = []
    rel_errors: list[np.ndarray] = []

    model.eval()
    print("[eval] Starting equivariance evaluation loop ...")
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            print(f"[eval] Processing batch {batch_idx + 1}/{max_batches} ...")
            x = x.to(device)
            print("[eval] Computing baseline penultimate features ...")
            base = extractor(x)
            print("[eval] Baseline features ready. Starting pushforward comparisons ...")

            for transform_idx, element in enumerate(elements):
                print(
                    f"[eval] Batch {batch_idx + 1}: applying transform "
                    f"{transform_idx + 1}/{len(elements)} ..."
                )
                x_transformed = transform_input(model, x, element, gspace)
                print("[eval] Pushforward complete. Running model on transformed input ...")
                pred = extractor(x_transformed)
                print("[eval] Computing transformed reference features ...")
                ref = transform_reference_features(base, element, compare_mode)
                abs_err = compute_absolute_errors(pred.flat, ref)
                rel_err = compute_relative_errors(pred.flat, ref)

                abs_np = abs_err.detach().cpu().numpy()
                rel_np = rel_err.detach().cpu().numpy()
                abs_errors.append(abs_np)
                rel_errors.append(rel_np)

                matrix = np.asarray(gspace.basespace_action(element), dtype=np.float64)
                detailed.append(
                    {
                        "batch_idx": batch_idx,
                        "transform_idx": transform_idx,
                        "determinant": float(np.linalg.det(matrix)),
                        "mean_absolute_l2": float(abs_np.mean()),
                        "max_absolute_l2": float(abs_np.max()),
                        "mean_relative_l2": float(rel_np.mean()),
                        "max_relative_l2": float(rel_np.max()),
                    }
                )
                print(
                    f"[eval] Batch {batch_idx + 1}, transform {transform_idx + 1}: "
                    f"mean_absolute_l2={abs_np.mean():.6e}, "
                    f"max_absolute_l2={abs_np.max():.6e}, "
                    f"mean_relative_l2={rel_np.mean():.6e}, "
                    f"max_relative_l2={rel_np.max():.6e}."
                )

    if not abs_errors:
        raise RuntimeError("No batches were evaluated.")

    abs_all = np.concatenate(abs_errors)
    rel_all = np.concatenate(rel_errors)
    summary = {
        "num_examples": int(abs_all.shape[0]),
        "num_transforms": len(elements),
        "mean_absolute_l2": float(abs_all.mean()),
        "std_absolute_l2": float(abs_all.std()),
        "median_absolute_l2": float(np.median(abs_all)),
        "max_absolute_l2": float(abs_all.max()),
        "mean_relative_l2": float(rel_all.mean()),
        "std_relative_l2": float(rel_all.std()),
        "median_relative_l2": float(np.median(rel_all)),
        "max_relative_l2": float(rel_all.max()),
    }
    print("[eval] Evaluation finished. Aggregated summary computed.")
    return summary, detailed


def to_serialisable_config(config: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, Path):
            out[key] = str(value)
        else:
            out[key] = value
    return out


def run_requested_check(
    spec: EvaluationSpec,
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    elements: list[Any],
    gspace,
    max_batches: int,
    device: torch.device,
) -> dict[str, Any]:
    print(
        f"[eval] Running check '{spec.name}' "
        f"({spec.description}, mode={spec.compare_mode}) ..."
    )
    if spec.name == "hidden-equivariance":
        if not (hasattr(model, "layers_eq") and hasattr(model, "full_net")):
            reason = (
                "Model does not expose a steerable final hidden representation; "
                "skipping hidden-equivariance."
            )
            print(f"[eval] {reason}")
            return {
                "name": spec.name,
                "description": spec.description,
                "compare_mode": spec.compare_mode,
                "status": "skipped",
                "reason": reason,
            }
        extractor = PenultimateExtractor(model)
    elif spec.name == "output-invariance":
        extractor = OutputExtractor(model)
    else:
        raise ValueError(f"Unsupported evaluation check: {spec.name}")

    summary, detailed = evaluate_equivariance(
        model=model,
        loader=loader,
        elements=elements,
        gspace=gspace,
        extractor=extractor,
        compare_mode=spec.compare_mode,
        max_batches=max_batches,
        device=device,
    )
    return {
        "name": spec.name,
        "description": spec.description,
        "compare_mode": spec.compare_mode,
        "status": "completed",
        "summary": summary,
        "details": detailed,
    }


def run_single_checkpoint(
    checkpoint_spec: CheckpointSpec,
    args: argparse.Namespace,
    device: torch.device,
    requested_checks: list[str],
) -> dict[str, Any]:
    checkpoint_path = checkpoint_spec.path
    print("\n" + "=" * 100)
    print(f"[main] Starting checkpoint {checkpoint_path}")
    print("=" * 100)

    checkpoint = load_checkpoint(checkpoint_path)
    config = normalise_config(recover_config(checkpoint, checkpoint_path))
    requested_group = args.group or (
        config["group"] if config["group"] in {"SO3", "O3"} else "SO3"
    )
    validate_requested_group(config, requested_group)
    print(f"[config] Requested test group: {requested_group}.")

    data_root = args.data_root if args.data_root is not None else get_default_data_root(config)
    print(f"[data] Using data root {data_root}.")
    model = create_model(config, device, full_init=args.full_init or args.strict)
    missing_keys, unexpected_keys = load_state_dict(model, checkpoint, strict=args.strict)

    loader = build_loader(
        config=config,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_root=data_root,
        download=args.download,
    )
    gspace = get_sampling_gspace(model, requested_group)
    print(f"[group] Using sampling gspace {type(gspace).__name__}.")
    elements = sample_group_elements(
        gspace,
        requested_group,
        args.num_transforms,
        identity_only=args.identity_only,
        right_angle_only=args.right_angle_only,
    )
    evaluations = [
        run_requested_check(
            spec=EVALUATION_SPECS[check_name],
            model=model,
            loader=loader,
            elements=elements,
            gspace=gspace,
            max_batches=args.max_batches,
            device=device,
        )
        for check_name in requested_checks
    ]
    print("[main] Preparing final result payload.")

    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_stage": checkpoint_spec.stage,
        "discovered_from": (
            None
            if checkpoint_spec.discovered_from is None
            else str(checkpoint_spec.discovered_from)
        ),
        "device": str(device),
        "model_class": type(model).__name__,
        "requested_group": requested_group,
        "transform_mode": (
            "identity"
            if args.identity_only
            else "right-angle"
            if args.right_angle_only
            else "random"
        ),
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "config": to_serialisable_config(config),
        "evaluations": evaluations,
    }

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Stage: {checkpoint_spec.stage or 'legacy'}")
    print(f"Model: {type(model).__name__}")
    print(f"Dataset: {config['dataset']} ({args.split})")
    print(f"Requested group: {requested_group}")
    print(f"Transform mode: {result['transform_mode']}")
    print(f"Transforms: {len(elements)}")
    for evaluation in evaluations:
        print(
            f"Check: {evaluation['name']} "
            f"(status={evaluation['status']}, mode={evaluation['compare_mode']})"
        )
        if evaluation["status"] == "completed":
            summary = evaluation["summary"]
            print(f"Examples evaluated: {summary['num_examples']}")
            print(f"Mean absolute L2 error: {summary['mean_absolute_l2']:.6e}")
            print(f"Median absolute L2 error: {summary['median_absolute_l2']:.6e}")
            print(f"Max absolute L2 error: {summary['max_absolute_l2']:.6e}")
            print(f"Mean relative L2 error: {summary['mean_relative_l2']:.6e}")
            print(f"Median relative L2 error: {summary['median_relative_l2']:.6e}")
            print(f"Max relative L2 error: {summary['max_relative_l2']:.6e}")
        else:
            print(f"Reason: {evaluation['reason']}")
    if missing_keys:
        print(f"Missing keys ({len(missing_keys)}): {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys}")

    print(f"[main] Finished checkpoint {checkpoint_path}.")
    return result


def print_results_table(results: list[dict[str, Any]]) -> None:
    rows = []
    for result in results:
        for evaluation in result["evaluations"]:
            summary = evaluation.get("summary", {})
            rows.append(
                {
                    "checkpoint": Path(result["checkpoint"]).name,
                    "stage": result["checkpoint_stage"] or "legacy",
                    "check": evaluation["name"],
                    "status": evaluation["status"],
                    "model": result["model_class"],
                    "dataset": result["config"]["dataset"],
                    "group": result["requested_group"],
                    "mode": evaluation["compare_mode"],
                    "mean_abs_l2": (
                        f"{summary['mean_absolute_l2']:.6e}"
                        if summary
                        else "-"
                    ),
                    "median_abs_l2": (
                        f"{summary['median_absolute_l2']:.6e}"
                        if summary
                        else "-"
                    ),
                    "max_abs_l2": (
                        f"{summary['max_absolute_l2']:.6e}"
                        if summary
                        else "-"
                    ),
                    "mean_rel_l2": (
                        f"{summary['mean_relative_l2']:.6e}"
                        if summary
                        else "-"
                    ),
                    "median_rel_l2": (
                        f"{summary['median_relative_l2']:.6e}"
                        if summary
                        else "-"
                    ),
                    "max_rel_l2": (
                        f"{summary['max_relative_l2']:.6e}"
                        if summary
                        else "-"
                    ),
                }
            )

    columns = [
        "checkpoint",
        "stage",
        "check",
        "status",
        "model",
        "dataset",
        "group",
        "mode",
        "mean_abs_l2",
        "median_abs_l2",
        "max_abs_l2",
        "mean_rel_l2",
        "median_rel_l2",
        "max_rel_l2",
    ]
    widths = {
        column: max(len(column), max(len(str(row[column])) for row in rows))
        for column in columns
    }

    def format_row(row: dict[str, str]) -> str:
        return " | ".join(str(row[column]).ljust(widths[column]) for column in columns)

    header = {column: column for column in columns}
    separator = " | ".join("-" * widths[column] for column in columns)

    print("\n" + "=" * 100)
    print("Equivariance Error Table")
    print("=" * 100)
    print(format_row(header))
    print(separator)
    for row in rows:
        print(format_row(row))


def main() -> None:
    args = parse_args()
    print("[main] Parsed command line arguments.")
    set_seed(args.seed)
    requested_checks = resolve_requested_checks(args)

    device_name = args.device
    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    print(f"[setup] Using device {device}.")
    checkpoint_specs = resolve_checkpoint_specs(args.inputs, args.stages)

    results = [
        run_single_checkpoint(checkpoint_spec, args, device, requested_checks)
        for checkpoint_spec in checkpoint_specs
    ]
    print_results_table(results)

    if args.json_out is not None:
        print(f"[output] Writing JSON summary to {args.json_out} ...")
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any]
        if len(results) == 1:
            payload = results[0]
        else:
            payload = {"results": results}
        with args.json_out.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print("[output] JSON summary written.")


if __name__ == "__main__":
    main()
