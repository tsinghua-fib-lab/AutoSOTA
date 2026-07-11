from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scgfm.checkpoint import load_model_from_checkpoint
from scgfm.data.datasets import load_node_graph_file
from scgfm.encoders import SCGFMEncoder
from scgfm.fewshot import evaluate_fewshot
from scgfm.utils import ensure_dir, load_config, resolve_device, set_seed, write_csv, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate node-level few-shot transfer on generated ego-graphs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
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
    model, _ = load_model_from_checkpoint(args.checkpoint, device)

    dataset_name = config.get("data", {}).get("dataset", "cora")
    graphs, num_classes = load_node_graph_file(config.get("data_root", "data"), dataset_name)
    del num_classes

    encoder_cfg = config.get("encoder", {})
    encoder = SCGFMEncoder(
        model,
        tau=encoder_cfg.get("tau"),
        device=device,
        max_dim=int(encoder_cfg.get("max_dim", 100)),
        num_projections=int(encoder_cfg.get("num_projections", 200)),
        top_k=int(encoder_cfg.get("top_k", 8)),
    )
    embeddings, labels = encoder.encode_dataset(graphs)

    fewshot = config.get("fewshot", {})
    metrics, rows = evaluate_fewshot(
        embeddings,
        labels,
        k_shot=int(fewshot.get("k_shot", 5)),
        n_query=int(fewshot.get("n_query", 50)),
        n_runs=int(fewshot.get("n_runs", 50)),
        seed=int(config.get("seed", 42)),
        device=device,
        pca_components=int(fewshot.get("pca_components", 0)),
        use_adapter=bool(fewshot.get("use_adapter", False)),
    )
    metrics["dataset"] = dataset_name
    out_dir = ensure_dir(config.get("output_dir", "outputs/node_fewshot"))
    write_json(out_dir / "metrics.json", metrics)
    write_csv(out_dir / "results.csv", rows)
    print(metrics)


if __name__ == "__main__":
    main()

