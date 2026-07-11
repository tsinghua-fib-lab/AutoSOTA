from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scgfm.data.datasets import load_tu_graphs
from scgfm.training import pretrain_bases
from scgfm.utils import load_config, resolve_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain SCGFM geometric bases on graph datasets.")
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

    set_seed(int(config.get("seed", 42)))
    device = resolve_device(config.get("device", "auto"))
    data_cfg = config.get("data", {})
    graphs = load_tu_graphs(
        data_root=config.get("data_root", "data"),
        names=list(data_cfg.get("datasets", ["MUTAG"])),
        max_nodes=data_cfg.get("max_nodes"),
        use_node_attr=bool(data_cfg.get("use_node_attr", True)),
        drop_node_features=bool(data_cfg.get("drop_node_features", True)),
    )
    print(f"Loaded {len(graphs)} training graphs on {device}.")
    pretrain_bases(graphs, config, device, config.get("output_dir", "outputs/graph_pretrain"))


if __name__ == "__main__":
    main()
