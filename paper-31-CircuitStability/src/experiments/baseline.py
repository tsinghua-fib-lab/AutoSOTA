import os
import sys
import pickle
import random
import argparse
from functools import partial

from cdatasets import DatasetBuilder, PromptFormatter
from eap import Graph, attribute, evaluate_baseline, evaluate_graph
from .utils import (
    seed_everything,
    parse_key_value_pairs,
    make_dataset,
    get_metric,
    get_extraction,
    extraction_schema,
    eval_pass,
    eval_choice,
)

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="model")
    parser.add_argument("ofname", type=str, help="output filename")
    parser.add_argument("--batch_size", type=int, help="batch size", default=8)
    parser.add_argument("--ndevices", type=int, help="number of devices", default=1)
    parser.add_argument("--seed", type=int, help="random seed", default=42)
    parser.add_argument("--generate", action="store_true", default=False)
    parser.add_argument("--max_new_tokens", type=int, help="max new tokens", default=15)
    parser.add_argument("--model_path", type=str, default=None, help="local model path override")
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

    if opts.model_path and os.path.exists(opts.model_path):
        print(f"Loading model from local path: {opts.model_path}")
        hf_model = AutoModelForCausalLM.from_pretrained(opts.model_path, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(opts.model_path, local_files_only=True)
        model = HookedTransformer.from_pretrained(
            opts.model_name, 
            hf_model=hf_model, 
            tokenizer=tokenizer,
            n_devices=opts.ndevices
        )
    else:
        model = HookedTransformer.from_pretrained(opts.model_name, n_devices=opts.ndevices)
    loader = dataset.to_dataloader(model, opts.batch_size)

    if (
        not opts.generate
        and hasattr(dataset, "choices")
        and opts.format != "chain-of-thought"
    ):
        print("evaluating multiple choice score")
        inputs, out_texts, labels = eval_choice(model, loader, dataset.choices)
    else:
        print(f"generating with max {opts.max_new_tokens} tokens")
        inputs, out_texts, labels = eval_pass(model, loader, opts.max_new_tokens)
    d = {
        "input": inputs,
        "output": out_texts,
        "target": labels,
    }
    pickle.dump(d, open(f"{opts.ofname}-benchmark.pkl", "wb+"))
