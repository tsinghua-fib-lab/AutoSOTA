import os
from dotenv import load_dotenv

load_dotenv()

import argparse
import pickle
import numpy as np
from src.utils import load_data
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from typing import Any, Dict, Iterable, List, Optional, Tuple
from data import get_dataset_generator, BaseDatasetGenerator
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

MODEL_LAYER_DEFAULTS: Dict[str, int] = {
    "Qwen/Qwen3-8B": 24,
    "Qwen/Qwen3-14B": 26,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": 17,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": 30
}

def get_args_parser():
    parser = argparse.ArgumentParser('Args', add_help=False)
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-8B')
    parser.add_argument('--layer_idx', type=int, default=None)
    parser.add_argument('--max_samples_per_prompt', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    return parser

def get_training_samples(data_generator, tokenizer, question: str, trace: str) -> Optional[str]:
    prompt = data_generator.apply_chat_template(tokenizer, question, trace)
    return prompt

def load_probe_model(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(model_name,device_map="auto", torch_dtype="auto", attn_implementation="flash_attention_2").eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return tokenizer, model

def build_prompt_samples(data: Iterable[Dict[str, Any]], data_generator: Optional[BaseDatasetGenerator], tokenizer) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for idx, row in enumerate(data):
        question = row.get("question", "")
        original = row.get("reasoning_well_posed", "")
        alternative = row.get("reasoning_ill_posed", "")
        trace_original = get_training_samples(data_generator, tokenizer, question, original)
        if trace_original:
            samples.append({"prompt": trace_original, "label": 0, "row_index": idx, "variant": "original"})
        trace_ill_posed = get_training_samples(data_generator, tokenizer, question, alternative)
        if trace_ill_posed:
            samples.append({"prompt": trace_ill_posed, "label": 1, "row_index": idx, "variant": "alternative"})
    return samples

def find_subseq(hay: torch.Tensor, needle: torch.Tensor) -> int:
    if needle.numel() == 0 or hay.numel() < needle.numel():
        return -1
    for i in range(0, hay.size(0) - needle.size(0) + 1):
        if torch.equal(hay[i:i + needle.size(0)], needle):
            return i
    return -1

def reasoning_token_indices(input_ids_row: torch.Tensor, think_ids: torch.Tensor, end_ids: torch.Tensor) -> List[int]:
    start = find_subseq(input_ids_row, think_ids)
    if start == -1:
        return []
    span_start = start + think_ids.size(0)
    end = find_subseq(input_ids_row, end_ids)
    span_end = end if end != -1 else input_ids_row.size(0)
    if span_end <= span_start:
        return []
    return list(range(span_start, span_end))

def random_sample_indices(abs_span, seed: int, max_samples_per_prompt: int) -> List[int]:
    rng = random.Random(seed)
    limit = len(abs_span) if max_samples_per_prompt <= 0 else min(max_samples_per_prompt, len(abs_span))
    sample_positions = abs_span if limit == len(abs_span) else rng.sample(abs_span, k=limit)
    return sample_positions

def collect_activations(
    samples: List[Dict[str, Any]],
    tokenizer,
    model,
    layer_idx: int,
    batch_size: int,
    sample_method,
) -> List[Dict[str, Any]]:
    if not samples:
        return []
    if layer_idx < 0 or layer_idx >= len(model.model.layers):
        raise ValueError(f"layer_idx={layer_idx} out of range for model with {len(model.model.layers)} layers")
    
    model_device = next(model.parameters()).device
    think_ids_device = tokenizer("<think>", add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(model_device)
    end_ids_device = tokenizer("</think>", add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(model_device)
    think_ids_cpu = think_ids_device.cpu()
    end_ids_cpu = end_ids_device.cpu()
    hook_buffer: Dict[str, Optional[torch.Tensor]] = {"tensor": None}

    def attn_hook(module, inputs, outputs):
        hook_buffer["tensor"] = outputs[0].detach().cpu()

    handle = model.model.layers[layer_idx].self_attn.register_forward_hook(attn_hook)
    dataset: List[Dict[str, Any]] = []
    try:
        for batch_start in tqdm(range(0, len(samples), batch_size)):
            batch = samples[batch_start:batch_start + batch_size]
            prompts = [item["prompt"] for item in batch]
            enc_cpu = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
            enc_device = {k: v.to(model_device) for k, v in enc_cpu.items()}
            with torch.no_grad():
                _ = model(**enc_device, use_cache=False)
            feats = hook_buffer.get("tensor")
            hook_buffer["tensor"] = None
            if feats is None:
                continue
            for offset, sample in enumerate(batch):
                attn_mask = enc_cpu["attention_mask"][offset]
                valid_pos = torch.nonzero(attn_mask == 1, as_tuple=False).squeeze(-1).tolist()
                if not valid_pos:
                    continue
                rel_span = reasoning_token_indices(enc_cpu["input_ids"][offset, valid_pos], think_ids_cpu, end_ids_cpu)
                if not rel_span:
                    continue
                
                # now, abs_span is the collection of absolute indices of reasoning tokens
                abs_span = [valid_pos[idx] for idx in rel_span]
                if not abs_span:
                    continue
                sample_positions = sample_method(abs_span)

                token_features = feats[offset, sample_positions, :].to(dtype=torch.float32).numpy()
                for j, pos in enumerate(sample_positions):
                    dataset.append({
                        "x": token_features[j],
                        "label": sample["label"],
                        "meta": {
                            "prompt_index": batch_start + offset,
                            "token_index": int(pos),
                            "row_index": sample.get("row_index"),
                            "variant": sample.get("variant"),
                            "layer": layer_idx,
                            "total_tokens": int(len(valid_pos))
                        }
                    })
                # free the memory for this sample
                del token_features
            # free the memory for this batch
            del feats
    finally:
        handle.remove()
    return dataset


def classify_thinking_content(prompt_samples, tokenizer, model, args):
    dataset = collect_activations(
        prompt_samples,
        tokenizer,
        model,
        args.layer_idx,
        args.batch_size,
        lambda abs_span: random_sample_indices(abs_span, args.seed, args.max_samples_per_prompt)
    )

    if not dataset:
        return {
            "dataset_size": 0,
            "layer_idx": int(args.layer_idx),
            "layer_source": args.layer_source,
            "message": "No activations extracted from reasoning spans."
        }

    X = np.stack([item["x"] for item in dataset])
    y = np.array([item["label"] for item in dataset])
    groups = np.array([item['meta']["row_index"] for item in dataset])  # <-- group by row

    unique_labels = np.unique(y)
    if unique_labels.size < 2:
        return {
            "dataset_size": len(dataset),
            "layer_idx": int(args.layer_idx),
            "layer_source": args.layer_source,
            "message": "Only one class present; linear probe not trained."
        }

    # ---------- GROUP-DISJOINT SPLIT ----------
    # No group appears in both train and test.
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf = LogisticRegression(max_iter=1000)  # raise max_iter to avoid non-convergence
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, probs)

    # save the dataset and the model to "probe/"
    os.makedirs(f"probe/{args.model_name}", exist_ok=True)
    with open(f"probe/{args.model_name}/layer{args.layer_idx}.pkl", "wb") as f:
        pickle.dump(dataset, f)
    with open(f"probe/{args.model_name}/probe_model_layer{args.layer_idx}.pkl", "wb") as f:
        pickle.dump(clf, f)

    # Useful diagnostics: how many groups in each split?
    n_groups_total = len(np.unique(groups))
    n_groups_train = len(np.unique(groups[train_idx]))
    n_groups_test = len(np.unique(groups[test_idx]))

    return {
        "dataset_size": len(dataset),
        "feature_dim": int(X.shape[1]),
        "layer_idx": int(args.layer_idx),
        "layer_source": args.layer_source,
        "n_groups_total": int(n_groups_total),
        "n_groups_train": int(n_groups_train),
        "n_groups_test": int(n_groups_test),
        "roc_auc": float(roc_auc),
    }

def resolve_layer_index(user_layer_idx: Optional[int], model_name: str, model) -> Tuple[int, str]:
    backbone = getattr(model, "model", None)
    if backbone is None or not hasattr(backbone, "layers"):
        raise AttributeError("Model backbone missing `.model.layers` for probing.")
    layer_count = len(backbone.layers)
    if user_layer_idx is not None:
        if 0 <= user_layer_idx < layer_count:
            return user_layer_idx, "user-specified"
        raise ValueError(f"layer_idx={user_layer_idx} out of range for model with {layer_count} layers")
    id_str = model_name
    default = MODEL_LAYER_DEFAULTS.get(id_str)
    source = "model default map"
    if default is None:
        match = next(((pattern, value) for pattern, value in MODEL_LAYER_DEFAULTS.items() if pattern in id_str), None)
        if match:
            pattern, default = match
            source = f"pattern '{pattern}'"
    if default is None:
        default = layer_count - 1
        source = "last layer fallback"
    resolved = min(default, layer_count - 1)
    return resolved, source


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    train_data = load_data(args.model_name, "gsm8k")
    data_generator = get_dataset_generator("gsm8k")
    tokenizer, model = load_probe_model(args.model_name)
    requested_layer = args.layer_idx
    args.layer_idx, args.layer_source = resolve_layer_index(requested_layer, args.model_name, model)
    if requested_layer is None:
        print(f"Auto-selected layer_idx={args.layer_idx} for {args.model_name} (source: {args.layer_source}).")
    prompt_samples = build_prompt_samples(train_data, data_generator, tokenizer)
    if not prompt_samples:
        print("No prompts containing both question and reasoning content were found.")
        return
    print("\n--- Running activation-based linear probe ---")
    results = classify_thinking_content(prompt_samples, tokenizer, model, args)
    # save the probing results to "probe/"
    with open(f"probe/{args.model_name}/training_results_layer{args.layer_idx}.json", "wb") as f:
        f.write(json.dumps(results, indent=4).encode("utf-8"))
    return results

if __name__ == "__main__":
    results = main()
