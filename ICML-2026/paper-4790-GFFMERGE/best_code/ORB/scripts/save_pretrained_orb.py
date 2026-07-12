"""
Save ORB pretrained model checkpoint to a file.

This utility script downloads and saves the pretrained weights of an ORB model
to a checkpoint file that can be used with merging scripts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from orb_models.forcefield import pretrained


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="orb-v2",
        choices=list(pretrained.ORB_PRETRAINED_MODELS.keys()),
        help="Name of the pretrained model to save (default: orb-v2).",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for the checkpoint file.",
    )

    parser.add_argument(
        "--precision",
        type=str,
        default="float32-high",
        choices=["float32-high", "float32-highest", "float64"],
        help="Model precision (default: float32-high).",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load model on before saving (default: cpu).",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    print(f"Loading pretrained model: {args.model}")
    print(f"  Precision: {args.precision}")
    print(f"  Device: {args.device}")

    # Load the pretrained model
    loader = pretrained.ORB_PRETRAINED_MODELS[args.model]
    model = loader(
        device=args.device,
        precision=args.precision,
        compile=False,
        train=True,
    )

    # Get state dict
    state_dict = model.state_dict()

    # Show key format for debugging
    sample_keys = list(state_dict.keys())[:3]
    print(f"  Sample keys: {sample_keys}")

    # Create checkpoint in the same format as fine-tuned models
    checkpoint = {"model": state_dict}

    # Save checkpoint
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)

    print(f"\nSaved pretrained checkpoint to: {args.output}")
    print(f"  Number of keys: {len(state_dict)}")
    print(f"  Number of parameters: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    main()
