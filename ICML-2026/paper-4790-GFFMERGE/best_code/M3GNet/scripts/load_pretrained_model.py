"""
Save a pretrained M3GNet model as a Lightning-style checkpoint and config.yaml.
"""

from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path

import torch
import yaml

os.environ.setdefault("MATGL_BACKEND", "DGL")
os.environ.setdefault("DGLBACKEND", "pytorch")
os.environ.setdefault("DGL_SKIP_GRAPHBOLT", "1")


def install_graphbolt_stub() -> None:
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
from matgl.utils.training import PotentialLightningModule


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", type=str, required=True, help="Pretrained M3GNet model name.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for checkpoint/config.")
    parser.add_argument("--output-name", type=str, default="original_pretrained_lightning.ckpt")
    parser.add_argument("--cutoff", type=float, default=5.0, help="Graph cutoff radius.")
    parser.add_argument("--energy-weight", type=float, default=1.0, help="Energy loss weight.")
    parser.add_argument("--force-weight", type=float, default=0.1, help="Force loss weight.")
    parser.add_argument("--stress-weight", type=float, default=0.0, help="Stress loss weight.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for config metadata.")
    parser.add_argument("--decay-steps", type=int, default=1000, help="LR decay steps for config metadata.")
    parser.add_argument("--decay-alpha", type=float, default=0.01, help="LR decay alpha for config metadata.")
    parser.add_argument("--seed", type=int, default=1, help="Seed for config metadata.")
    return parser


def build_config(args: argparse.Namespace) -> dict:
    return {
        "model": {
            "pretrained_name": args.model_name,
            "cutoff": args.cutoff,
        },
        "train": {
            "seed": args.seed,
            "batch_size": 8,
            "lr": args.lr,
            "energy_weight": args.energy_weight,
            "force_weight": args.force_weight,
            "stress_weight": args.stress_weight,
            "decay_steps": args.decay_steps,
            "decay_alpha": args.decay_alpha,
            "num_workers": 0,
        },
        "data": {
            "train_path": "",
            "val_path": "",
            "test_path": "",
            "cache_dir": "data/cache/pretrained",
        },
        "output": {"run_dir": "runs/pretrained_original"},
    }


def main() -> None:
    args = build_parser().parse_args()

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading pretrained model: {args.model_name}")
    potential = load_model(args.model_name)
    base_model = potential.model

    element_refs = None
    if getattr(potential, "element_refs", None) is not None:
        element_refs = potential.element_refs.property_offset.detach().cpu().numpy()

    module = PotentialLightningModule(
        model=base_model,
        element_refs=element_refs,
        energy_weight=args.energy_weight,
        force_weight=args.force_weight,
        stress_weight=args.stress_weight,
        lr=args.lr,
        decay_steps=args.decay_steps,
        decay_alpha=args.decay_alpha,
        data_mean=float(getattr(potential, "data_mean", 0.0)),
        data_std=float(getattr(potential, "data_std", 1.0)),
    )

    ckpt_path = out_dir / args.output_name
    checkpoint = {"state_dict": module.state_dict()}
    if "pytorch-lightning_version" not in checkpoint:
        try:
            import lightning

            checkpoint["pytorch-lightning_version"] = lightning.__version__
        except ImportError:
            try:
                import pytorch_lightning

                checkpoint["pytorch-lightning_version"] = pytorch_lightning.__version__
            except ImportError:
                checkpoint["pytorch-lightning_version"] = "2.0.0"

    torch.save(checkpoint, ckpt_path)
    print(f"Saved pretrained checkpoint to: {ckpt_path}")

    cfg_path = out_dir / "config.yaml"
    cfg = build_config(args)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"Wrote config to: {cfg_path}")


if __name__ == "__main__":
    main()
