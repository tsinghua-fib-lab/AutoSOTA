import os
import sys
import pickle
import random
import argparse
from functools import partial

from cdatasets import DatasetBuilder, PromptFormatter
from eap import Graph, attribute, evaluate_baseline, evaluate_graph
from .utils import (
    load_model,
    seed_everything,
    parse_key_value_pairs,
    make_dataset,
    get_metric,
    get_extraction,
    extraction_schema,
)

import torch.nn.functional as F
from transformer_lens import HookedTransformer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="model")
    parser.add_argument("ofname", type=str, help="output filename")
    parser.add_argument("checkpoint", type=int, help="checkpoint")
    parser.add_argument("--batch_size", type=int, help="batch size", default=32)
    parser.add_argument("--ndevices", type=int, help="number of devices", default=1)
    parser.add_argument("--large_model", action="store_true", help="use large model")
    parser.add_argument("--seed", type=int, help="random seed", default=42)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DatasetBuilder.ids.keys()),
        help="dataset name",
        required=True,
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=list(PromptFormatter.ids.keys()),
        help="format name",
        required=True,
    )
    parser.add_argument("--data_params", nargs="*", default=[], help="dataset params")
    parser.add_argument("--format_params", nargs="*", default=[], help="format params")
    parser.add_argument(
        "--patching_metric", type=str, default="kl", help="patching metric"
    )
    parser.add_argument(
        "--extraction",
        type=str,
        default="last_token",
        help="method for extracting comparison tokens",
    )
    parser.add_argument("--ig_steps", type=int, default=5, help="number of IG steps")
    args = parser.parse_args()
    args.data_params = parse_key_value_pairs(args.data_params)
    args.format_params = parse_key_value_pairs(args.format_params)
    return args


if __name__ == "__main__":
    opts = parse_args()
    seed_everything(opts.seed)

    dataset = make_dataset(
        opts.dataset, opts.data_params, opts.format, opts.format_params
    )
    model = load_model(opts.model_name, opts.checkpoint, large_model=opts.large_model)

    model = HookedTransformer.from_pretrained(opts.model_name, n_devices=opts.ndevices)
    dataloader = dataset.to_dataloader(model, opts.batch_size)

    pure_metric = get_metric(opts.patching_metric)
    extraction = get_extraction(opts.extraction)

    metric = extraction_schema(extraction, model)(pure_metric)

    g = Graph.from_model(model)
    attribute(model, g, dataloader, metric, method="EAP-IG", ig_steps=opts.ig_steps)
    g.apply_topn(200, absolute=False)
    g.to_json(f"{opts.ofname}.json")
    g.prune_dead_nodes()

    baseline = evaluate_baseline(model, dataloader, metric)
    results = evaluate_graph(model, g, dataloader, metric)

    diff = (results - baseline).mean().item()

    print(f"The circuit incurred extra {diff} loss.")

    gz = g.to_graphviz()
    gz.draw(f"{opts.ofname}.png", prog="dot")
