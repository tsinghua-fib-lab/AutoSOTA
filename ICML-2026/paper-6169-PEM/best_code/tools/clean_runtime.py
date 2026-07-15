#!/usr/bin/env python3
"""
Remove runtime outputs that should not be committed or shared.

This script deletes only well-known, repo-local paths:
- Results/
- exdata/
- artifacts/
- any __pycache__/ folders
- any *.pyc / *.pyo files

It is safe to run before sharing the repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _iter_runtime_paths(root: Path) -> list[Path]:
    targets: list[Path] = []

    for name in ["Results", "exdata", "artifacts"]:
        p = root / name
        if p.exists():
            targets.append(p)

    targets.extend(root.rglob("__pycache__"))
    targets.extend(root.rglob("*.pyc"))
    targets.extend(root.rglob("*.pyo"))

    # De-duplicate (and normalize) while preserving a stable order.
    uniq: dict[Path, None] = {}
    for p in sorted({t.resolve() for t in targets}):
        try:
            p.relative_to(root.resolve())
        except Exception:
            continue
        uniq[p] = None
    return list(uniq.keys())


def _remove_path(path: Path) -> None:
    if path.is_dir():
        for child in sorted(path.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink(missing_ok=True)
            elif child.is_dir():
                child.rmdir()
        path.rmdir()
    else:
        path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="List what would be removed without deleting.")
    args = parser.parse_args()

    root = _repo_root()
    targets = _iter_runtime_paths(root)

    if not targets:
        print("Nothing to remove.")
        return

    for p in targets:
        rel = p.resolve().relative_to(root.resolve())
        print(("WOULD REMOVE:" if args.dry_run else "REMOVING:"), str(rel))
        if not args.dry_run:
            _remove_path(p)


if __name__ == "__main__":
    main()

