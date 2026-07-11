from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scgfm.data.datasets import load_node_graph_file
from scgfm.training import pretrain_bases
from scgfm.utils import load_config, resolve_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain SCGFM geometric bases on generated node ego-graphs.")
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
    datasets = config.get("data", {}).get("datasets", ["photo", "computers"])
    graphs = []
    for name in datasets:
        dataset_graphs, _ = load_node_graph_file(config.get("data_root", "data"), name)
        if bool(config.get("data", {}).get("drop_node_features", True)):
            for graph in dataset_graphs:
                graph.x = None
        graphs.extend(dataset_graphs)
    print(f"Loaded {len(graphs)} node ego-graphs on {device}.")
    pretrain_bases(graphs, config, device, config.get("output_dir", "outputs/node_pretrain"))


if __name__ == "__main__":
    main()
