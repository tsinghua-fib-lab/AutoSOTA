"""
Convert DeepMD raw datasets (coord.raw/force.raw/energy.raw) to extended XYZ.
"""

from __future__ import annotations

import argparse
from itertools import zip_longest
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


REQUIRED_FILES = ("coord.raw", "force.raw", "energy.raw", "type.raw", "type_map.raw")


def is_raw_dir(path: Path) -> bool:
    return all((path / name).is_file() for name in REQUIRED_FILES)


def find_raw_dirs(root: Path) -> List[Path]:
    if is_raw_dir(root):
        return [root]
    matches: List[Path] = []
    for candidate in root.rglob("*"):
        if candidate.is_dir() and is_raw_dir(candidate):
            matches.append(candidate)
    return sorted(matches)


def parse_type_map(path: Path) -> List[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line]


def parse_types(path: Path) -> List[int]:
    types: List[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            types.append(int(stripped))
    return types


def iter_frames(
    coord_path: Path,
    force_path: Path,
    energy_path: Path,
) -> Iterable[Tuple[str, str, str, int]]:
    with coord_path.open("r", encoding="utf-8") as coord_f, force_path.open(
        "r", encoding="utf-8"
    ) as force_f, energy_path.open("r", encoding="utf-8") as energy_f:
        for idx, (coord_line, force_line, energy_line) in enumerate(
            zip_longest(coord_f, force_f, energy_f), start=1
        ):
            if coord_line is None or force_line is None or energy_line is None:
                raise ValueError(
                    "coord.raw/force.raw/energy.raw have different number of frames."
                )
            yield coord_line, force_line, energy_line, idx


def convert_raw_dir(
    raw_dir: Path,
    output_path: Path,
    limit: int | None,
) -> int:
    type_map = parse_type_map(raw_dir / "type_map.raw")
    types = parse_types(raw_dir / "type.raw")
    if not type_map:
        raise ValueError(f"Empty type_map.raw in {raw_dir}")
    if not types:
        raise ValueError(f"Empty type.raw in {raw_dir}")

    num_atoms = len(types)
    symbols = [type_map[type_id] for type_id in types]

    coord_path = raw_dir / "coord.raw"
    force_path = raw_dir / "force.raw"
    energy_path = raw_dir / "energy.raw"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames_written = 0
    with output_path.open("w", encoding="utf-8") as out_f:
        for coord_line, force_line, energy_line, idx in iter_frames(
            coord_path, force_path, energy_path
        ):
            if limit is not None and frames_written >= limit:
                break
            coords = [float(val) for val in coord_line.split()]
            forces = [float(val) for val in force_line.split()]
            energy_str = energy_line.split()[0]
            energy = float(energy_str)

            expected = 3 * num_atoms
            if len(coords) != expected:
                raise ValueError(
                    f"coord.raw frame {idx} in {raw_dir} has {len(coords)} values, expected {expected}."
                )
            if len(forces) != expected:
                raise ValueError(
                    f"force.raw frame {idx} in {raw_dir} has {len(forces)} values, expected {expected}."
                )

            out_f.write(f"{num_atoms}\n")
            out_f.write(
                f"Energy={energy:.16e} Properties=species:S:1:pos:R:3:forces:R:3\n"
            )
            for atom_idx, symbol in enumerate(symbols):
                base = 3 * atom_idx
                x, y, z = coords[base : base + 3]
                fx, fy, fz = forces[base : base + 3]
                out_f.write(
                    f"{symbol} {x:.10f} {y:.10f} {z:.10f} {fx:.10f} {fy:.10f} {fz:.10f}\n"
                )
            frames_written += 1
    return frames_written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset spec as name=path to DeepMD root (repeatable).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory to write .xyz files (defaults to data/raw).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of frames to convert.",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="If a dataset root contains multiple raw dirs, concatenate them.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.dataset:
        raise ValueError("Provide at least one --dataset name=path.")

    for entry in args.dataset:
        if "=" not in entry:
            raise ValueError(f"--dataset expects name=path, got '{entry}'")
        name, path_str = entry.split("=", 1)
        name = name.strip()
        root = Path(path_str.strip())
        if not name:
            raise ValueError(f"Dataset name is empty in '{entry}'")
        if not root.exists():
            raise FileNotFoundError(root)

        raw_dirs = find_raw_dirs(root)
        if not raw_dirs:
            raise FileNotFoundError(f"No DeepMD raw directories found under {root}")
        if len(raw_dirs) > 1 and not args.combine:
            listed = "\n  - ".join(str(path) for path in raw_dirs)
            raise ValueError(
                f"Multiple raw dirs found under {root}. Use --combine to merge:\n  - {listed}"
            )

        output_path = args.output_dir / f"{name}.xyz"
        total_frames = 0
        for raw_dir in raw_dirs:
            total_frames += convert_raw_dir(raw_dir, output_path, args.limit)
        print(f"Wrote {output_path} ({total_frames} frames)")


if __name__ == "__main__":
    main()
