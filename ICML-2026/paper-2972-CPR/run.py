#!/usr/bin/env python3
"""CPR full pipeline: train RCVNet -> calibrate -> evaluate."""

import argparse
import json

from cpr.cli_args import add_common_args, apply_config, load_yaml_config, resolve_env_paths
from cpr.pipeline import (
    calibrate_phase,
    evaluate_phase,
    load_datasets,
    build_encoder,
    train_phase,
    save_checkpoint,
)


def main():
    parser = argparse.ArgumentParser(description="Conformal Path Reasoning (CPR)")
    add_common_args(parser)
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--out_json", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None, help="Save checkpoint path")
    parser.add_argument(
        "--risk_levels",
        type=str,
        default=None,
        help="Comma-separated alphas e.g. 0.3,0.4,0.5 for multi-run eval",
    )
    args = parser.parse_args()

    if args.config:
        apply_config(args, load_yaml_config(args.config))
    args = resolve_env_paths(args)

    train_items, calib_items, test_items, global_triples = load_datasets(args)
    encoder = build_encoder(args)

    alphas = [args.alpha]
    if args.risk_levels:
        alphas = [float(x.strip()) for x in args.risk_levels.split(",")]

    all_results = {}
    for alpha in alphas:
        args.alpha = alpha
        print(f"\n{'='*60}\nRisk level alpha={alpha}\n{'='*60}")

        core = train_phase(args, train_items, global_triples, encoder)
        calib_stats = calibrate_phase(args, core, calib_items)
        test_eval = evaluate_phase(core, test_items)

        print(
            f"\n[Test] ECR={test_eval['ecr']:.3f}, APSS={test_eval['apss']:.3f}, "
            f"Efficiency={test_eval['coverage_efficiency']:.4f}"
        )

        all_results[str(alpha)] = {
            "calib": calib_stats,
            "test_eval": test_eval,
        }

        if args.checkpoint:
            ckpt_path = args.checkpoint.replace("{alpha}", str(alpha))
            save_checkpoint(ckpt_path, core, calib_stats, args)

    if args.out_json:
        out = {"results": all_results, "conformal_mode": args.conformal_mode}
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2, default=str)
        print(f"\nSaved results to {args.out_json}")


if __name__ == "__main__":
    main()
