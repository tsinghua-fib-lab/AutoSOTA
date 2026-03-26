import json
import argparse

from cdatasets import DatasetBuilder, PromptFormatter
from eap import Graph, evaluate_graph
from .utils import (
    seed_everything,
    parse_key_value_pairs,
    make_dataset,
    get_metric,
    get_extraction,
    extraction_schema,
    eval_pass,
)

import numpy as np
import torch.nn.functional as F
from transformer_lens import HookedTransformer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="model")
    parser.add_argument("ofname", type=str, help="output filename")
    parser.add_argument("graph_file", type=str, help="graph with scores after pruning")
    parser.add_argument("--batch_size", type=int, help="batch size", default=32)
    parser.add_argument("--ndevices", type=int, help="number of devices", default=1)
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
    model = HookedTransformer.from_pretrained(opts.model_name, n_devices=opts.ndevices)
    dataloader = dataset.to_dataloader(model, opts.batch_size)

    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True

    pure_metric = get_metric(opts.patching_metric)
    extraction = get_extraction(opts.extraction)
    metric = extraction_schema(extraction, model)(pure_metric)

    g = Graph.from_json(opts.graph_file)

    ## get the total number of edges and the take a fraction of it, go 5% at a time
    n_components = len(g.edges)
    perf = []
    components = []
    for prct in np.logspace(0, np.log10(n_components), 20):
        remained_components = int(prct)
        components.append(remained_components)
        g.apply_topn(remained_components, absolute=True)
        g.prune_dead_nodes()

        empty = not g.nodes["logits"].in_graph
        if empty and prct != 1:
            continue

        result = evaluate_graph(model, g, dataloader, metric)
        perf.append(result.mean().item())
        print("n_comps:", remained_components, "result", result.mean().item())

    pareto = {"perf": perf, "components": components}

    json.dump(pareto, open(f"{opts.ofname}-pareto.json", "w+"))
