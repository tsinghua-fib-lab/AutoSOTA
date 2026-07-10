#!/usr/bin/env python3
"""Phase 2: Query-level path conformal calibration on D_cal."""

import argparse
import json
import os

from cpr.cli_args import add_common_args, apply_config, load_yaml_config, resolve_env_paths
from cpr.core import CPRCore
from cpr.pipeline import (
    load_datasets,
    build_encoder,
    core_kwargs,
    calibrate_phase,
    load_checkpoint,
)


def main():
    parser = argparse.ArgumentParser(description="CPR Phase 2: conformal calibration")
    add_common_args(parser)
    parser.add_argument(
        "--rcvnet_checkpoint",
        type=str,
        default="checkpoints/rcvnet.pt",
        help="Phase 1 checkpoint (optional)",
    )
    parser.add_argument(
        "--out_threshold",
        type=str,
        default="checkpoints/tau_{alpha}.json",
    )
    args = parser.parse_args()

    if args.config:
        apply_config(args, load_yaml_config(args.config))
    args = resolve_env_paths(args)

    train_items, calib_items, _, global_triples = load_datasets(args)
    encoder = build_encoder(args)

    if os.path.isfile(args.rcvnet_checkpoint):
        core = CPRCore(**core_kwargs(
            args, train_items, global_triples, encoder, skip_training=True
        ))
        load_checkpoint(args.rcvnet_checkpoint, core)
        print(f"[Loaded] {args.rcvnet_checkpoint}")
    else:
        print("[Warn] No RCVNet checkpoint; running Phase 1 training first.")
        from cpr.pipeline import train_phase
        core = train_phase(args, train_items, global_triples, encoder)

    stats = calibrate_phase(args, core, calib_items)

    out_path = args.out_threshold.format(alpha=args.alpha)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[Done] Threshold saved to {out_path}")


if __name__ == "__main__":
    main()
