"""
Mean merge ORB checkpoints by averaging shared parameters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Mapping, Sequence

import torch


def load_config(path: Path) -> Dict:
    return json.loads(path.read_text())


def extract_state_dict(checkpoint_path: Path) -> Mapping[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def merge_mean(states: Sequence[Mapping[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if len(states) < 2:
        raise ValueError("At least two teacher checkpoints are required for mean merge.")
    merged: Dict[str, torch.Tensor] = {}
    keys = set(states[0].keys())
    for idx, state in enumerate(states[1:], start=1):
        state_keys = set(state.keys())
        if state_keys != keys:
            missing = keys - state_keys
            extra = state_keys - keys
            raise ValueError(
                f"State dict keys mismatch for teacher {idx}. Missing: {missing}. Extra: {extra}."
            )

    for key in states[0]:
        acc = None
        for state in states:
            acc = state[key].clone() if acc is None else acc + state[key]
        merged[key] = acc / float(len(states))

    return merged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to training config JSON.")
    parser.add_argument(
        "--teacher",
        action="append",
        type=Path,
        default=[],
        help="Checkpoint path for a teacher (repeat to include multiple teachers).",
    )
    parser.add_argument(
        "--teacher-a",
        type=Path,
        default=None,
        help="(Deprecated) Checkpoint path for first teacher.",
    )
    parser.add_argument(
        "--teacher-b",
        type=Path,
        default=None,
        help="(Deprecated) Checkpoint path for second teacher.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination path for merged checkpoint (will be overwritten).",
    )
    parser.add_argument(
        "--keep-metadata",
        action="store_true",
        help="Copy metadata fields (best_val_loss, history, etc.) from teacher A if present.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    _ = load_config(args.config)  # currently unused, kept for symmetry / validation

    if args.teacher:
        teacher_paths = list(args.teacher)
    else:
        if args.teacher_a is None or args.teacher_b is None:
            raise ValueError("Provide at least two teachers via --teacher or --teacher-a/--teacher-b.")
        teacher_paths = [args.teacher_a, args.teacher_b]
    if len(teacher_paths) < 2:
        raise ValueError("At least two teacher checkpoints are required for mean merge.")

    states = [extract_state_dict(path) for path in teacher_paths]
    merged_state = merge_mean(states)

    checkpoint: Dict[str, object] = {"model": merged_state}
    if args.keep_metadata:
        teacher_ckpt = torch.load(teacher_paths[0], map_location="cpu")
        for key in ["optimizer", "scheduler", "history", "best_val_loss", "epoch"]:
            if key in teacher_ckpt:
                checkpoint[key] = teacher_ckpt[key]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"Saved mean-merged checkpoint to {args.output}")


if __name__ == "__main__":
    main()
