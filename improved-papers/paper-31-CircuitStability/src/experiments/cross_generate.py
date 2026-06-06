import json
import argparse

from cdatasets import DatasetBuilder, PromptFormatter
from eap import Graph, evaluate_graph, evaluate_graph_generate
from .utils import (
    seed_everything,
    parse_key_value_pairs,
    make_dataset,
)

import tqdm
import torch
import numpy as np
import torch.nn.functional as F
from transformer_lens import HookedTransformer


from typing import Callable, List, Union
from functools import partial

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from einops import einsum

from eap.graph import Graph, InputNode, LogitNode, AttentionNode, MLPNode, Node, Edge
from eap.attribute import make_hooks_and_matrices, tokenize_plus


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="model")
    parser.add_argument("ofname", type=str, help="output filename")
    parser.add_argument("graph_file", type=str, help="graph with scores after pruning")
    parser.add_argument("--batch_size", type=int, help="batch size", default=64)
    parser.add_argument("--ndevices", type=int, help="number of devices", default=2)
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


@torch.inference_mode()
def eval_pass(model, dataloader, graph, max_new_tokens=15):
    model.eval()
    inputs, out_texts, labels = [], [], []

    graph.prune_dead_nodes(prune_childless=True, prune_parentless=True)

    empty_circuit = not graph.nodes["logits"].in_graph
    if empty_circuit:
        print("Warning: empty circuit")

    # we construct the in_graph matrix, which is a binary matrix indicating which edges are in the circuit
    # we invert it so that we add in the corrupting activation differences for edges not in the circuit
    in_graph_matrix = torch.zeros(
        (graph.n_forward, graph.n_backward), device="cuda", dtype=model.cfg.dtype
    )
    for edge in graph.edges.values():
        if edge.in_graph:
            in_graph_matrix[
                graph.forward_index(edge.parent, attn_slice=False),
                graph.backward_index(edge.child, qkv=edge.qkv, attn_slice=False),
            ] = 1

    in_graph_matrix = 1 - in_graph_matrix

    for clean, corrupted, label in tqdm.tqdm(dataloader):
        outputs = evaluate_graph_generate(
            model, graph, clean, corrupted, max_new_tokens=max_new_tokens, in_graph_matrix=in_graph_matrix
        )
        inputs.extend(model.to_string(clean[0]))
        out_texts.extend(model.to_string(outputs))
        labels.extend(label)
    return inputs, out_texts, labels


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

    g = Graph.from_json(opts.graph_file)

    ## get the total number of edges and the take a fraction of it, go 5% at a time
    n_components = len(g.edges)
    components = []
    out_texts = []
    for prct in np.logspace(0, np.log10(n_components), 20):
        remained_components = int(prct)
        components.append(remained_components)
        g.apply_topn(remained_components, absolute=True)
        g.prune_dead_nodes()

        empty = not g.nodes["logits"].in_graph
        if empty and prct != 1:
            continue

        ins, outs, labels = eval_pass(model, dataloader, g)
        out_texts.append({
            "inputs": ins,
            "outputs": outs,
            "labels": labels
        })
        print("Sample outputs:", outs[:5])
        print("n_comps:", remained_components)

    pareto = {"results": out_texts, "components": components}

    json.dump(pareto, open(f"{opts.ofname}-pareto.json", "w+"))
