from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scgfm.data.node_to_graph import NodeToGraphConverter
from scgfm.utils import load_config, resolve_device


def parse_args():
    parser = argparse.ArgumentParser(description="Build node ego-graph datasets for node-level few-shot evaluation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.device is not None:
        config["device"] = args.device
    if args.data_root is not None:
        config["data_root"] = args.data_root
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    device = resolve_device(config.get("device", "auto"))
    build_cfg = config.get("build", {})
    converter = NodeToGraphConverter(
        data_root=config.get("data_root", "data"),
        output_root=config.get("output_dir", "data/node_graphs"),
        device=device,
        seed=int(config.get("seed", 42)),
    )
    for dataset_name in build_cfg.get("datasets", ["Cora"]):
        out = converter.convert(
            dataset_name=dataset_name,
            size=int(build_cfg.get("size", 100)),
            alpha=float(build_cfg.get("alpha", 0.15)),
            batch_size=int(build_cfg.get("batch_size", 256)),
            samples_per_class=int(build_cfg.get("samples_per_class", 300)),
            max_iter=int(build_cfg.get("max_iter", 100)),
            tol=float(build_cfg.get("tol", 1e-6)),
        )
        print(f"Saved {out}")


if __name__ == "__main__":
    main()

