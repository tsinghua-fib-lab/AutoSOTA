"""
Mean-merge M3GNet Lightning checkpoints into a new checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", action="append", type=Path, help="Checkpoint to merge (repeatable).")
    parser.add_argument("--ckpt-a", type=Path, help="First checkpoint to merge (legacy).")
    parser.add_argument("--ckpt-b", type=Path, help="Second checkpoint to merge (legacy).")
    parser.add_argument("--output-ckpt", type=Path, required=True, help="Path to write merged checkpoint.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require identical keys/shapes; fail on any mismatch.",
    )
    return parser


def extract_state_dict(ckpt: object, label: str) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        if all(isinstance(value, torch.Tensor) for value in ckpt.values()):
            return ckpt
    raise ValueError(f"Unsupported checkpoint format for {label}.")


def merge_state_dicts(
    states: Sequence[Dict[str, torch.Tensor]],
    strict: bool,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    merged: Dict[str, torch.Tensor] = {}
    notes: Dict[str, str] = {}
    if not states:
        raise ValueError("No checkpoints provided to merge.")

    key_sets = [set(state.keys()) for state in states]
    if strict and any(key_sets[0] != ks for ks in key_sets[1:]):
        missing = []
        for idx, ks in enumerate(key_sets[1:], start=1):
            if ks != key_sets[0]:
                missing.append(f"ckpt{idx}: missing={sorted(key_sets[0]-ks)} extra={sorted(ks-key_sets[0])}")
        raise ValueError("Key mismatch across checkpoints: " + "; ".join(missing))

    all_keys = set().union(*key_sets)
    for key in sorted(all_keys):
        tensors: List[torch.Tensor] = []
        ref_shape = None
        for state in states:
            if key not in state:
                if strict:
                    raise ValueError(f"Missing key {key} in one checkpoint.")
                continue
            value = state[key]
            if not isinstance(value, torch.Tensor):
                if strict:
                    raise ValueError(f"Non-tensor param for key {key}.")
                continue
            if ref_shape is None:
                ref_shape = value.shape
            if value.shape != ref_shape:
                if strict:
                    raise ValueError(f"Shape mismatch for {key}: {value.shape} vs {ref_shape}")
                continue
            tensors.append(value)

        if not tensors:
            notes[key] = "skipped_no_tensor"
            continue
        if len(tensors) == 1:
            merged[key] = tensors[0]
            notes[key] = "copied_single"
            continue
        merged[key] = torch.stack(tensors, dim=0).mean(dim=0)
        if len(tensors) != len(states):
            notes[key] = f"averaged_{len(tensors)}_of_{len(states)}"
    return merged, notes


def main() -> None:
    args = build_parser().parse_args()

    ckpt_paths = list(args.ckpt or [])
    if not ckpt_paths:
        if args.ckpt_a is None or args.ckpt_b is None:
            raise ValueError("Provide --ckpt (repeatable) or both --ckpt-a/--ckpt-b.")
        ckpt_paths = [args.ckpt_a, args.ckpt_b]
    if len(ckpt_paths) < 2:
        raise ValueError("At least two checkpoints are required to merge.")

    ckpts = [torch.load(path, map_location="cpu", weights_only=False) for path in ckpt_paths]
    states = [extract_state_dict(ckpt, f"ckpt-{idx}") for idx, ckpt in enumerate(ckpts)]

    merged_state, notes = merge_state_dicts(states, strict=args.strict)

    base_ckpt = ckpts[0]
    if isinstance(base_ckpt, dict) and "state_dict" in base_ckpt:
        output_ckpt = dict(base_ckpt)
        output_ckpt["state_dict"] = merged_state
    else:
        output_ckpt = merged_state

    output_ckpt["merged_from"] = [str(path) for path in ckpt_paths]
    output_ckpt["merged_notes"] = notes

    args.output_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_ckpt, args.output_ckpt)

    print(f"Merged checkpoint written to {args.output_ckpt}")


if __name__ == "__main__":
    main()
