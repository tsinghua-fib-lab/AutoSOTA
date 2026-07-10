#!/usr/bin/env python3
"""Phase 1: Train RCVNet with PUCT on D_train."""

import argparse

from cpr.cli_args import add_common_args, apply_config, load_yaml_config, resolve_env_paths
from cpr.pipeline import load_datasets, build_encoder, train_phase, save_checkpoint


def main():
    parser = argparse.ArgumentParser(description="CPR Phase 1: RCVNet training")
    add_common_args(parser)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/rcvnet.pt")
    args = parser.parse_args()

    if args.config:
        apply_config(args, load_yaml_config(args.config))
    args = resolve_env_paths(args)
    args.puct_calib = True

    train_items, _, _, global_triples = load_datasets(args)
    encoder = build_encoder(args)
    core = train_phase(args, train_items, global_triples, encoder)
    save_checkpoint(args.checkpoint, core, {}, args)
    print(f"\n[Done] RCVNet checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
