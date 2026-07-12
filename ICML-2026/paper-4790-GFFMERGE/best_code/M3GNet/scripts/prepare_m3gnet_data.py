"""
Prepare MD17/MD22-style XYZ data (or ANI/MD HDF5 molecules) for M3GNet fine-tuning
and emit quick configs.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import ase.io
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
import yaml


KCAL_MOL_TO_EV = 0.0433641153
HARTREE_TO_EV = 27.211386245988
M3GNET_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class SampleResult:
    name: str
    total_frames: int
    sampled_frames: int
    source_path: Path
    source_indices: Sequence[int]


@dataclass
class H5Source:
    path: Path
    group: str


SourceSpec = Path | H5Source


def open_text(path: Path):
    if path.name.lower().endswith((".xyz.gz", ".extxyz.gz")):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def iter_xyz_frames(
    path: Path,
    unit_scale: float,
    input_units: str,
    max_frames: Optional[int],
) -> Iterable[Tuple[int, Atoms]]:
    """Yield (frame_index, atoms) pairs from an MD17/MD22-style XYZ file."""
    with open_text(path) as handle:
        frame = 0
        while True:
            if max_frames is not None and frame >= max_frames:
                break
            header = handle.readline()
            if not header:
                break
            header = header.strip()
            if not header:
                continue
            try:
                natoms = int(header)
            except ValueError as exc:
                raise ValueError(f"Expected atom count at frame {frame} in {path}.") from exc

            comment = handle.readline()
            if not comment:
                raise ValueError(f"Unexpected EOF reading energy line for frame {frame} in {path}.")
            energy_raw = parse_energy_from_comment(comment, path, frame)

            symbols: List[str] = []
            positions: List[Tuple[float, float, float]] = []
            forces_raw: List[Tuple[float, float, float]] = []
            for atom_idx in range(natoms):
                line = handle.readline()
                if not line:
                    raise ValueError(f"Unexpected EOF within frame {frame} in {path}.")
                parts = line.split()
                if len(parts) < 7:
                    raise ValueError(
                        f"Frame {frame} atom {atom_idx} in {path} expected 7 columns, got {len(parts)}."
                    )
                symbols.append(parts[0])
                positions.append((float(parts[1]), float(parts[2]), float(parts[3])))
                forces_raw.append((float(parts[4]), float(parts[5]), float(parts[6])))

            atoms = Atoms(symbols=symbols, positions=positions)
            atoms.info["source_index"] = frame
            atoms.info["energy_raw"] = energy_raw
            atoms.info["energy_raw_unit"] = input_units
            energy = energy_raw * unit_scale
            forces = np.array(forces_raw, dtype=float) * unit_scale
            atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
            yield frame, atoms
            frame += 1


def parse_energy_from_comment(comment: str, path: Path, frame: int) -> float:
    comment = comment.strip()
    tokens = comment.split()
    if tokens:
        try:
            return float(tokens[0])
        except ValueError:
            pass
    for token in tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key.lower() in ("e", "energy"):
            try:
                return float(value)
            except ValueError:
                break
    cleaned = comment.replace(":", " ")
    cleaned_tokens = cleaned.split()
    for idx, token in enumerate(cleaned_tokens[:-1]):
        if token.lower() in ("e", "energy"):
            try:
                return float(cleaned_tokens[idx + 1])
            except ValueError:
                break
    raise ValueError(f"Could not parse energy on frame {frame} in {path}.")


def parse_h5_spec(dataset_arg: str, raw_dir: Path) -> Optional[H5Source]:
    normalized = dataset_arg.replace("\\", "/")
    lower = normalized.lower()
    if ".h5" not in lower:
        return None
    idx = lower.find(".h5")
    h5_part = normalized[: idx + 3]
    rest = normalized[idx + 3 :]
    group = None
    if rest.startswith(":") or rest.startswith("/"):
        group = rest[1:]
    h5_path = Path(h5_part)
    if not h5_path.is_absolute():
        h5_path = raw_dir / h5_path
    return H5Source(path=h5_path, group=group or "")


def iter_h5_frames(
    source: H5Source,
    unit_scale: float,
    input_units: str,
    max_frames: Optional[int],
) -> Iterable[Tuple[int, Atoms]]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to read ANI/MD HDF5 datasets.") from exc

    with h5py.File(source.path, "r") as handle:
        if source.group in handle:
            group = handle[source.group]
        elif "crd.dat" in handle and source.group in handle["crd.dat"]:
            group = handle["crd.dat"][source.group]
        else:
            raise KeyError(f"Group {source.group} not found in {source.path}.")

        species_raw = group["species"][()]
        species = [
            item.decode("utf-8") if isinstance(item, (bytes, bytearray)) else str(item)
            for item in species_raw
        ]
        coords = group["coordinates"]
        forces = group["forces"]
        energies = group["energies"]

        frame = 0
        for idx in range(energies.shape[0]):
            if max_frames is not None and frame >= max_frames:
                break
            atoms = Atoms(symbols=species, positions=coords[idx])
            atoms.info["source_index"] = frame
            atoms.info["energy_raw"] = float(energies[idx])
            atoms.info["energy_raw_unit"] = input_units
            energy = float(energies[idx]) * unit_scale
            forces_frame = np.array(forces[idx], dtype=float) * unit_scale
            atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces_frame)
            yield frame, atoms
            frame += 1


def iter_xyz_sources(
    sources: Sequence[SourceSpec],
    unit_scale: float,
    input_units: str,
    max_frames: Optional[int],
) -> Iterable[Tuple[int, Atoms]]:
    frame = 0
    for source in sources:
        if isinstance(source, H5Source):
            iterator = iter_h5_frames(source, unit_scale, input_units, None)
        else:
            iterator = iter_xyz_frames(source, unit_scale, input_units, None)
        for _, atoms in iterator:
            atoms.info["source_index"] = frame
            yield frame, atoms
            frame += 1
            if max_frames is not None and frame >= max_frames:
                return


def sanitize_dataset_name(name: str) -> str:
    safe = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        else:
            safe.append("_")
    sanitized = "".join(safe).strip("_")
    return sanitized or "dataset"


def strip_xyz_suffix(name: str) -> str:
    lower = name.lower()
    for suffix in (".extxyz.gz", ".xyz.gz", ".extxyz", ".xyz"):
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def dataset_label_from_source(source: Path, raw_dir: Path) -> str:
    try:
        rel = source.resolve().relative_to(raw_dir.resolve())
        parts = list(rel.parts)
    except ValueError:
        parts = [source.name]
    if source.is_file():
        parts[-1] = strip_xyz_suffix(source.name)
    label = "_".join(parts)
    return sanitize_dataset_name(label)


def is_xyz_file(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith((".xyz", ".extxyz", ".xyz.gz", ".extxyz.gz"))


def find_xyz_sources(root: Path) -> List[Path]:
    if root.is_file():
        if not is_xyz_file(root):
            raise FileNotFoundError(f"Unsupported file type: {root}")
        return [root]
    sources = [
        path
        for path in root.rglob("*")
        if path.is_file() and is_xyz_file(path)
    ]
    sources = sorted(sources, key=lambda p: str(p))
    if not sources:
        raise FileNotFoundError(f"No .xyz/.extxyz files found under {root}")
    return sources


def resolve_dataset_sources(
    raw_dir: Path, dataset_arg: str, h5_file: Optional[Path] = None
) -> Tuple[str, List[SourceSpec], Path]:
    candidate = Path(dataset_arg)
    if h5_file is not None:
        if not h5_file.exists():
            raise FileNotFoundError(h5_file)
        group = dataset_arg.strip()
        if not group:
            raise ValueError("When using --h5-file, dataset name must be a non-empty group key.")
        dataset_name = sanitize_dataset_name(group)
        return dataset_name, [H5Source(path=h5_file, group=group)], h5_file

    h5_spec = parse_h5_spec(dataset_arg, raw_dir)
    if h5_spec is not None:
        if not h5_spec.path.exists():
            raise FileNotFoundError(h5_spec.path)
        if not h5_spec.group:
            raise ValueError("H5 dataset spec must include a group, e.g. ani_md_bench.h5:molCaffeine")
        dataset_name = sanitize_dataset_name(h5_spec.group)
        return dataset_name, [h5_spec], h5_spec.path
    if not candidate.is_absolute():
        norm = str(candidate).replace("\\", "/")
        raw_norm = str(raw_dir).replace("\\", "/").rstrip("/")
        if norm.startswith(raw_norm) or norm.startswith("M3GNet/data/raw/") or norm.startswith("data/raw/"):
            candidate = Path(norm)
        else:
            candidate = raw_dir / candidate
    if candidate.exists():
        sources = find_xyz_sources(candidate)
        dataset_name = dataset_label_from_source(candidate, raw_dir)
        return dataset_name, sources, candidate

    file_xyz = raw_dir / f"{dataset_arg}.xyz"
    file_extxyz = raw_dir / f"{dataset_arg}.extxyz"
    file_xyz_gz = raw_dir / f"{dataset_arg}.xyz.gz"
    file_extxyz_gz = raw_dir / f"{dataset_arg}.extxyz.gz"
    if file_xyz.exists():
        return dataset_label_from_source(file_xyz, raw_dir), [file_xyz], file_xyz
    if file_extxyz.exists():
        return dataset_label_from_source(file_extxyz, raw_dir), [file_extxyz], file_extxyz
    if file_xyz_gz.exists():
        return dataset_label_from_source(file_xyz_gz, raw_dir), [file_xyz_gz], file_xyz_gz
    if file_extxyz_gz.exists():
        return dataset_label_from_source(file_extxyz_gz, raw_dir), [file_extxyz_gz], file_extxyz_gz
    if dataset_arg.lower() == "malonaldehyde":
        alt = raw_dir / "malonadehyde.xyz"
        if alt.exists():
            return dataset_label_from_source(alt, raw_dir), [alt], alt
        alt_gz = raw_dir / "malonadehyde.xyz.gz"
        if alt_gz.exists():
            return dataset_label_from_source(alt_gz, raw_dir), [alt_gz], alt_gz
    raise FileNotFoundError(f"Missing raw dataset: {candidate}")


def reservoir_sample(
    sources: Sequence[SourceSpec],
    sample_size: int,
    rng: random.Random,
    unit_scale: float,
    input_units: str,
    max_frames: Optional[int],
) -> Tuple[List[Atoms], int]:
    """Reservoir-sample sample_size frames from path."""
    reservoir: List[Atoms] = []
    total_frames = 0

    for _, atoms in iter_xyz_sources(sources, unit_scale, input_units, max_frames):
        total_frames += 1
        if len(reservoir) < sample_size:
            reservoir.append(atoms)
        else:
            j = rng.randint(0, total_frames - 1)
            if j < sample_size:
                reservoir[j] = atoms

    if not total_frames:
        raise ValueError(f"No frames found in {path}.")

    return reservoir, total_frames


def sample_dataset(
    name: str,
    sources: Sequence[SourceSpec],
    source_root: Path,
    sample_size: int,
    rng: random.Random,
    unit_scale: float,
    input_units: str,
    max_frames: Optional[int],
) -> Tuple[List[Atoms], SampleResult]:
    if sample_size <= 0:
        samples = [atoms for _, atoms in iter_xyz_sources(sources, unit_scale, input_units, max_frames)]
        total_frames = len(samples)
    else:
        samples, total_frames = reservoir_sample(sources, sample_size, rng, unit_scale, input_units, max_frames)
    rng.shuffle(samples)
    for atoms in samples:
        atoms.info["dataset"] = name
    indices = [int(atoms.info["source_index"]) for atoms in samples]
    result = SampleResult(
        name=name,
        total_frames=total_frames,
        sampled_frames=len(samples),
        source_path=source_root,
        source_indices=indices,
    )
    return samples, result


def split_train_val(
    samples: Sequence[Atoms],
    val_fraction: float,
    rng: random.Random,
) -> Tuple[List[Atoms], List[Atoms]]:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1).")
    if val_fraction == 0.0:
        return list(samples), []
    shuffled = list(samples)
    rng.shuffle(shuffled)
    val_size = max(1, int(math.floor(len(shuffled) * val_fraction)))
    val = shuffled[:val_size]
    train = shuffled[val_size:]
    if not train:
        raise ValueError("Validation split too large; no training samples left.")
    return train, val


def split_train_val_test(
    samples: Sequence[Atoms],
    val_fraction: float,
    test_fraction: float,
    rng: random.Random,
) -> Tuple[List[Atoms], List[Atoms], List[Atoms]]:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1).")
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("test_fraction must be in [0, 1).")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1.0.")
    if val_fraction == 0.0 and test_fraction == 0.0:
        return list(samples), [], []
    shuffled = list(samples)
    rng.shuffle(shuffled)
    val_size = max(1, int(math.floor(len(shuffled) * val_fraction))) if val_fraction > 0.0 else 0
    test_size = max(1, int(math.floor(len(shuffled) * test_fraction))) if test_fraction > 0.0 else 0
    val = shuffled[:val_size]
    test = shuffled[val_size : val_size + test_size]
    train = shuffled[val_size + test_size :]
    if not train:
        raise ValueError("Validation/test split too large; no training samples left.")
    return train, val, test


def write_extxyz(samples: Sequence[Atoms], output_path: Path, limit: int) -> int:
    if limit > 0:
        samples = list(samples)[:limit]
    if not samples:
        raise RuntimeError(f"No samples to write for {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ase.io.write(output_path, list(samples), format="extxyz")
    return len(samples)


def make_quick_config(
    dataset_name: str,
    train_path: Path,
    val_path: Path,
    test_path: Path,
    epochs: int,
    model_name: str,
    cutoff: float,
    batch_size: int,
    lr: float,
    energy_weight: float,
    force_weight: float,
    stress_weight: float,
) -> dict:
    def to_root_relative(path: Path) -> Path:
        if path.is_absolute():
            resolved = path
        else:
            parts = path.parts
            if parts and parts[0].lower() == M3GNET_ROOT.name.lower():
                path = Path(*parts[1:])
            resolved = (M3GNET_ROOT / path).resolve()
        try:
            return resolved.relative_to(M3GNET_ROOT)
        except ValueError:
            return resolved

    train_rel = to_root_relative(train_path)
    val_rel = to_root_relative(val_path)
    test_rel = to_root_relative(test_path)
    return {
        "model": {
            "pretrained_name": model_name,
            "cutoff": cutoff,
        },
        "train": {
            "seed": 1,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "energy_weight": energy_weight,
            "force_weight": force_weight,
            "stress_weight": stress_weight,
            "decay_steps": 1000,
            "decay_alpha": 0.01,
            "num_workers": 0,
        },
        "data": {
            "train_path": str(train_rel),
            "val_path": str(val_rel),
            "test_path": str(test_rel),
            "cache_dir": f"data/cache/{dataset_name}",
        },
        "output": {
            "run_dir": f"runs/{dataset_name}",
        },
    }


def write_yaml(config: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=M3GNET_ROOT / "data" / "raw",
        help="Directory containing raw MD17/MD22 XYZ files (e.g., ethanol.xyz).",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Dataset name or path (repeatable). When provided, overrides --dataset-a/--dataset-b.",
    )
    parser.add_argument(
        "--dataset-a",
        type=str,
        default="ethanol",
        help="Name for dataset A (expects <name>.xyz under raw-dir).",
    )
    parser.add_argument(
        "--dataset-b",
        type=str,
        default="malonaldehyde",
        help="Name for dataset B (expects <name>.xyz under raw-dir).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=M3GNET_ROOT / "data" / "prepared",
        help="Destination directory for EXTXYZ outputs.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=M3GNET_ROOT / "configs",
        help="Destination directory for generated M3GNet YAML configs.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1250,
        help="Number of frames to sample for each dataset (<=0 means all).",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of samples reserved for validation (0.0-1.0).",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.1,
        help="Fraction of samples reserved for test (0.0-1.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for sampling and shuffling.",
    )
    parser.add_argument(
        "--extxyz-limit",
        type=int,
        default=-1,
        help="Max frames to export per split (<=0 means all).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Epochs to set in the generated configs (keep tiny for smoke tests).",
    )
    parser.add_argument(
        "--input-units",
        type=str.lower,
        choices=["kcal/mol", "ev", "hartree"],
        default="kcal/mol",
        help="Units of energy labels in raw XYZ (forces assumed per-A).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=-1,
        help="Optional cap on frames read per dataset (<=0 means all frames).",
    )
    parser.add_argument(
        "--h5-file",
        type=Path,
        default=None,
        help="Optional ANI/MD HDF5 file; dataset names should be HDF5 group keys.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="M3GNet-ANI-1x-Subset-PES",
        help="Pretrained M3GNet model name for fine-tuning.",
    )
    parser.add_argument("--cutoff", type=float, default=5.0, help="Graph cutoff radius.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training configs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for training configs.")
    parser.add_argument("--energy-weight", type=float, default=1.0, help="Energy loss weight.")
    parser.add_argument("--force-weight", type=float, default=0.1, help="Force loss weight.")
    parser.add_argument("--stress-weight", type=float, default=0.0, help="Stress loss weight.")
    parser.add_argument(
        "--include-combined",
        action="store_true",
        help="Also emit a combined dataset/config.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.config_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    if args.input_units == "kcal/mol":
        unit_scale = KCAL_MOL_TO_EV
    elif args.input_units == "hartree":
        unit_scale = HARTREE_TO_EV
    else:
        unit_scale = 1.0
    max_frames = None if args.max_frames <= 0 else args.max_frames

    dataset_a = args.dataset_a.strip()
    dataset_b = args.dataset_b.strip()
    if args.datasets:
        dataset_args = [item.strip() for item in args.datasets if item.strip()]
    else:
        if not dataset_a or not dataset_b:
            raise ValueError("dataset names must be non-empty.")
        dataset_args = [dataset_a, dataset_b]
    if len(dataset_args) < 2:
        raise ValueError("Provide at least two datasets (via --dataset or --dataset-a/--dataset-b).")

    datasets = []
    for dataset_arg in dataset_args:
        name, sources, root = resolve_dataset_sources(args.raw_dir, dataset_arg, args.h5_file)
        samples, meta = sample_dataset(
            name,
            sources,
            root,
            args.sample_size,
            rng,
            unit_scale,
            args.input_units,
            max_frames,
        )
        datasets.append((samples, meta))

    summary = {
        "seed": args.seed,
        "sample_size": args.sample_size,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "input_units": args.input_units,
        "unit_scale": unit_scale,
        "datasets": [
            {
                "name": meta.name,
                "total_frames": meta.total_frames,
                "sampled_frames": meta.sampled_frames,
                "source_path": str(meta.source_path),
                "source_indices": meta.source_indices,
            }
            for _, meta in datasets
        ],
    }

    if args.include_combined:
        combined = []
        for samples, _ in datasets:
            combined.extend(samples)
        rng.shuffle(combined)
        summary["datasets"].append(
            {
                "name": "combined",
                "total_frames": len(combined),
                "sampled_frames": len(combined),
                "source_path": str(args.raw_dir),
                "source_indices": [int(atoms.info["source_index"]) for atoms in combined],
            }
        )

    summary_path = args.output_dir / "sampling_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    split_sources = [(meta.name, samples) for samples, meta in datasets]
    if args.include_combined:
        split_sources.append(("combined", combined))

    for name, samples in split_sources:
        train_samples, val_samples, test_samples = split_train_val_test(
            samples, args.val_fraction, args.test_fraction, rng
        )
        train_xyz = args.output_dir / f"{name}_train.extxyz"
        val_xyz = args.output_dir / f"{name}_val.extxyz"
        test_xyz = args.output_dir / f"{name}_test.extxyz"
        write_extxyz(train_samples, train_xyz, args.extxyz_limit)
        write_extxyz(val_samples, val_xyz, args.extxyz_limit)
        write_extxyz(test_samples, test_xyz, args.extxyz_limit)

        cfg = make_quick_config(
            dataset_name=name,
            train_path=train_xyz,
            val_path=val_xyz,
            test_path=test_xyz,
            epochs=args.epochs,
            model_name=args.model_name,
            cutoff=args.cutoff,
            batch_size=args.batch_size,
            lr=args.lr,
            energy_weight=args.energy_weight,
            force_weight=args.force_weight,
            stress_weight=args.stress_weight,
        )
        cfg_path = args.config_dir / f"{name}_quick.yaml"
        write_yaml(cfg, cfg_path)
        print(f"Wrote config -> {cfg_path}")


if __name__ == "__main__":
    main()
