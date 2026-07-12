"""
Sample MD17 xyz data and generate ASE SQLite databases for Kaggle training.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Optional

import numpy as np
import ase.db
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator


@dataclass
class SampleResult:
    """Stores sampling metadata for a single dataset."""

    name: str
    total_frames: int
    sampled_frames: int
    db_path: Path
    source_path: Path
    source_indices: Sequence[int]


def iter_xyz_frames(path: Path) -> Iterable[Tuple[int, Atoms]]:
    """Yield (frame_index, atoms) pairs from an extended XYZ file."""
    with path.open("r", encoding="utf-8") as handle:
        frame = 0
        while True:
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
            energy = parse_energy_from_comment(comment, frame, path)

            symbols: List[str] = []
            positions: List[Tuple[float, float, float]] = []
            forces: List[Tuple[float, float, float]] = []
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
                forces.append((float(parts[4]), float(parts[5]), float(parts[6])))

            atoms = Atoms(symbols=symbols, positions=positions)
            atoms.info["source_index"] = frame
            atoms.info["energy_raw"] = energy
            calc = SinglePointCalculator(atoms, energy=energy, forces=np.array(forces, dtype=float))
            atoms.calc = calc
            yield frame, atoms
            frame += 1


def parse_energy_from_comment(comment: str, frame: int, path: Path) -> float:
    text = comment.strip()
    if not text:
        raise ValueError(f"Empty energy/comment line for frame {frame} in {path}.")
    first = text.split()[0]
    try:
        return float(first)
    except ValueError:
        pass
    for pattern in (
        r"(?:^|\s)(?:Energy|energy|E)=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
        r"(?:^|\s)(?:Energy|energy|E):([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
    ):
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
    match = re.search(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)", text)
    if match:
        return float(match.group(1))
    raise ValueError(f"Could not parse energy for frame {frame} in {path}: '{text}'")


def reservoir_sample(path: Path, sample_size: int, rng: random.Random) -> Tuple[List[Atoms], int]:
    """Reservoir-sample sample_size frames from path."""
    reservoir: List[Atoms] = []
    total_frames = 0

    for frame_idx, atoms in iter_xyz_frames(path):
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


def clone_with_calculator(atoms: Atoms, combined_name: Optional[str] = None) -> Atoms:
    """Create a new Atoms copy retaining calculator-derived energy/forces."""
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    clone = atoms.copy()
    clone.calc = SinglePointCalculator(clone, energy=energy, forces=forces)
    clone.info = dict(atoms.info)
    if combined_name is not None:
        clone.info.setdefault("source_dataset", clone.info.get("dataset"))
        clone.info["dataset"] = combined_name
    return clone


def write_sqlite(atoms_list: Sequence[Atoms], output_path: Path) -> None:
    """Write atoms entries to an ASE SQLite database."""
    if output_path.exists():
        output_path.unlink()

    db = ase.db.connect(str(output_path))
    for atoms in atoms_list:
        metadata = {
            "dataset": atoms.info.get("dataset"),
            "source_index": atoms.info.get("source_index"),
            "source_dataset": atoms.info.get("source_dataset"),
        }
        db.write(atoms, data={k: v for k, v in metadata.items() if v is not None})


def sample_dataset(
    name: str, source: Path, sample_size: int, rng: random.Random, output_dir: Path
) -> Tuple[List[Atoms], SampleResult]:
    """Sample a dataset and persist it to SQLite."""
    samples, total_frames = reservoir_sample(source, sample_size, rng)
    rng.shuffle(samples)
    for atoms in samples:
        atoms.info["dataset"] = name
        atoms.info.setdefault("source_dataset", name)
    db_path = output_dir / f"{name}_sampled.db"
    write_sqlite(samples, db_path)
    indices = [int(atoms.info["source_index"]) for atoms in samples]
    result = SampleResult(
        name=name,
        total_frames=total_frames,
        sampled_frames=len(samples),
        db_path=db_path,
        source_path=source,
        source_indices=indices,
    )
    return samples, result


def parse_dataset_specs(
    args: argparse.Namespace,
) -> List[Tuple[str, Path]]:
    """Resolve dataset name/path pairs from CLI arguments."""
    if args.dataset:
        datasets: List[Tuple[str, Path]] = []
        for entry in args.dataset:
            if "=" not in entry:
                raise ValueError(f"--dataset expects name=path, got '{entry}'")
            name, path_str = entry.split("=", 1)
            name = name.strip()
            path = Path(path_str.strip())
            if not name:
                raise ValueError(f"--dataset has empty name in '{entry}'")
            datasets.append((name, path))
        names = [name for name, _ in datasets]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate dataset names in --dataset: {names}")
        return datasets

    return [
        (args.dataset_a_name, args.ethanol_path),
        (args.dataset_b_name, args.malonaldehyde_path),
    ]


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    repo_root = Path(__file__).resolve().parents[2]
    data_root = repo_root / "data" / "raw"
    default_output = repo_root / "data" / "prepared"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help=(
            "Dataset spec as name=path (repeat for multiple datasets). "
            "When provided, --ethanol-path/--malonaldehyde-path and dataset-a/b names are ignored."
        ),
    )
    parser.add_argument(
        "--ethanol-path",
        type=Path,
        default=data_root / "ethanol.xyz",
        help="Path to the ethanol MD17 XYZ file (used when --dataset is not set).",
    )
    parser.add_argument(
        "--malonaldehyde-path",
        type=Path,
        default=data_root / "malonaldehyde.xyz",
        help="Path to the malonaldehyde MD17 XYZ file (used when --dataset is not set).",
    )
    parser.add_argument(
        "--dataset-a-name",
        type=str,
        default="ethanol",
        help="Logical name for dataset A (used when --dataset is not set).",
    )
    parser.add_argument(
        "--dataset-b-name",
        type=str,
        default="malonaldehyde",
        help="Logical name for dataset B (used when --dataset is not set).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Directory where sampled ASE databases and metadata will be written (defaults to data/prepared).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Number of frames to sample for each standalone dataset.",
    )
    parser.add_argument(
        "--combined-name",
        type=str,
        default="combined",
        help="Base name for the combined dataset database.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for sampling and shuffling.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    datasets = parse_dataset_specs(args)
    if len(datasets) < 1:
        raise ValueError("At least one dataset must be provided.")

    all_samples: List[List[Atoms]] = []
    metas: List[SampleResult] = []
    for name, path in datasets:
        samples, meta = sample_dataset(name, path, args.sample_size, rng, args.output_dir)
        all_samples.append(samples)
        metas.append(meta)

    combined: List[Atoms] = []
    for samples in all_samples:
        for atoms in samples:
            combined.append(clone_with_calculator(atoms, combined_name=args.combined_name))
    rng.shuffle(combined)
    combined_name = args.combined_name
    combined_path = args.output_dir / f"{combined_name}_sampled.db"
    write_sqlite(combined, combined_path)

    combined_meta = SampleResult(
        name=combined_name,
        total_frames=sum(meta.sampled_frames for meta in metas),
        sampled_frames=len(combined),
        db_path=combined_path,
        source_path=args.output_dir,
        source_indices=[int(atoms.info["source_index"]) for atoms in combined],
    )

    summary = {
        "seed": args.seed,
        "sample_size": args.sample_size,
        "datasets": [
            {
                "name": meta.name,
                "total_frames": meta.total_frames,
                "sampled_frames": meta.sampled_frames,
                "db_path": str(meta.db_path),
                "source_path": str(meta.source_path),
                "source_indices": meta.source_indices,
            }
            for meta in metas
        ]
        + [
            {
                "name": combined_meta.name,
                "total_frames": combined_meta.total_frames,
                "sampled_frames": combined_meta.sampled_frames,
                "db_path": str(combined_meta.db_path),
                "source_path": str(combined_meta.source_path),
                "source_indices": combined_meta.source_indices,
            }
        ],
    }
    summary_path = args.output_dir / "sampling_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Prepared datasets written to {args.output_dir}")
    for entry in summary["datasets"]:
        print(
            f"  - {entry['name']}: {entry['sampled_frames']} frames "
            f"(source: {entry['source_path']}) -> {entry['db_path']}"
        )


if __name__ == "__main__":
    main()
