#!/usr/bin/env python3
"""Uniform entry point for every experiment in the paper.

    python run.py <experiment> [experiment-specific args...]

where ``<experiment>`` is one of:

    sbi                 Bayesian-inference benchmark (analytic priors).
    black-hole          Black-hole imaging (InverseBench engine).
    super-resolution    4x ImageNet super-resolution (pixel mean-flow).

Each experiment also exposes the same interface programmatically as
``experiments.<name>.run`` / ``experiments.<name>.cli``. Run

    python run.py <experiment> --help

for the per-experiment flags.
"""

from __future__ import annotations

import importlib
import sys

EXPERIMENTS = {
    "sbi": "experiments.sbi.run",
    "black-hole": "experiments.black_hole.run",
    "super-resolution": "experiments.super_resolution.run",
}


def _usage(stream=sys.stderr) -> None:
    print(__doc__, file=stream)
    print("Available experiments: " + ", ".join(EXPERIMENTS), file=stream)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        _usage(sys.stdout if argv else sys.stderr)
        return 0 if argv else 2
    name = argv[0]
    if name not in EXPERIMENTS:
        print(f"Unknown experiment: {name!r}\n", file=sys.stderr)
        _usage()
        return 2
    module = importlib.import_module(EXPERIMENTS[name])
    # Re-shape argv so the sub-command sees a clean program name + its own args.
    sys.argv = [f"run.py {name}", *argv[1:]]
    return int(module.cli() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
