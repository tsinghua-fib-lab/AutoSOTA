#!/usr/bin/env python3
"""Training-free VA evaluation on SciFact using Llama-2-7B.

Reproduces NDCG@10 metrics from Table 1 of the paper:
  VA (Llama-2, layers 20-27): NDCG@10 = 54.58 on SciFact

Usage:
  python eval_va_scifact.py --model_path /path/to/model --layers 20-27
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add repo to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mteb import MTEB
import mteb

# HF env setup
os.environ.pop("HF_ENDPOINT", None)

from llama_model import VaLlamaForCausalLM
from transformers import AutoTokenizer


# SciFact task prompt (from mteb_evaluation.py)
SCIFACT_PROMPT = "Given a scientific claim, retrieve documents that support or refute the claim:"


class VAEvaluator:
    """Training-free VA: loads a model and aggregates value vectors across selected layers."""

    def __init__(self, model_path: str, layers: list, max_length: int = 512):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.layers = layers  # 0-indexed layer indices

        print(f"Loading model from {model_path} ...")
        self.model = VaLlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
        )
        self.model.eval()
        print(f"Model loaded on {self.device}. Aggregating layers: {[l+1 for l in layers]} (1-indexed)")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.dim = self.model.config.hidden_size
        print(f"Model dim: {self.dim}")

    def encode(self, sentences, prompt_name=None, batch_size=32, **kwargs):
        """MTEB encoder interface. Adds task prompt for queries."""
        # For retrieval tasks, prompt_name indicates query vs corpus
        # We add the SciFact prompt for both queries and corpus (matching paper)
        if prompt_name is not None:
            sentences = [SCIFACT_PROMPT + '\n' + text for text in sentences]

        all_embeddings = []
        with torch.no_grad():
            for start in tqdm(range(0, len(sentences), batch_size), desc="Encoding", leave=False):
                batch = sentences[start:start + batch_size]
                inputs = self.tokenizer(
                    batch,
                    add_special_tokens=True,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Forward pass through VaLlamaForCausalLM to get all_values
                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_hidden_states=False,
                )

                # all_values is a tuple of (batch, seq_len, hidden_size) per layer
                all_values = outputs.all_values

                # Select layers and aggregate
                selected_values = [all_values[i] for i in self.layers]

                # Compute cross-layer variance for each token position
                # Stack selected values: (num_layers, batch, seq_len, hidden_dim)
                stacked_vals = torch.stack(selected_values, dim=0)
                # Variance across layers (dim=0): (batch, seq_len, hidden_dim)
                val_variance = stacked_vals.var(dim=0, unbiased=False)
                # Aggregate variance across hidden dim -> per-token importance
                token_var = val_variance.mean(dim=-1)  # (batch, seq_len)

                # Normalize variance to get token weights (blend with uniform)
                mask_2d = inputs["attention_mask"].float()
                token_var = token_var * mask_2d
                var_sum = token_var.sum(dim=1, keepdim=True).clamp(min=1e-6)
                var_weights = token_var / var_sum  # (batch, seq_len)

                # Uniform weights for blending
                uniform_weights = mask_2d / mask_2d.sum(dim=1, keepdim=True).clamp(min=1e-6)
                # Blend: 0.8 variance + 0.2 uniform (conservative)
                token_weights = 0.8 * var_weights + 0.2 * uniform_weights

                # Weighted mean pool over tokens
                layer_embeddings = []
                for val in selected_values:
                    weighted_val = val * token_weights.unsqueeze(-1)
                    layer_emb = weighted_val.sum(dim=1)  # (batch, hidden_dim)
                    layer_embeddings.append(layer_emb)

                # Compute per-layer weights via cosine dissimilarity to group mean
                stacked_layer_embs = torch.stack(layer_embeddings, dim=0)  # (num_layers, batch, hidden_dim)
                # Group mean across layers
                layer_group_mean = stacked_layer_embs.mean(dim=0, keepdim=True)  # (1, batch, hidden_dim)
                # Cosine similarity of each layer to group mean
                stacked_norm = F.normalize(stacked_layer_embs, p=2, dim=-1)
                group_norm = F.normalize(layer_group_mean, p=2, dim=-1)
                cos_sim = (stacked_norm * group_norm).sum(dim=-1)  # (num_layers, batch)
                # Convert to dissimilarity and compute weights
                dissimilarity = 1.0 - cos_sim  # (num_layers, batch)
                layer_weights = F.softmax(dissimilarity.mean(dim=1), dim=0)  # (num_layers,) - temperature=1
                # Weighted mean over layers
                embedding = (stacked_layer_embs * layer_weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=0)
                all_embeddings.append(F.normalize(embedding, p=2, dim=-1).cpu().to(torch.float32))

                # Free GPU memory explicitly
                del outputs, all_values, stacked_vals, val_variance, selected_values, stacked_layer_embs, layer_embeddings, embedding
                torch.cuda.empty_cache()

        return torch.cat(all_embeddings, dim=0).numpy()


def parse_layers(layer_str: str, num_layers: int = 32) -> list:
    """Parse layer range like '20-27' into 0-indexed list."""
    if '-' in layer_str:
        parts = layer_str.split('-')
        start, end = int(parts[0]), int(parts[1])
        # If values are large, assume 0-indexed; otherwise assume 1-indexed
        if start >= num_layers or end >= num_layers:
            return list(range(start, end + 1))
        else:
            return list(range(start - 1, end))
    else:
        indices = [int(x.strip()) for x in layer_str.split(',')]
        if max(indices) >= num_layers:
            return indices
        else:
            return [i - 1 for i in indices]


def main():
    parser = argparse.ArgumentParser(description="VA evaluation on SciFact")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--layers", type=str, default="20-27")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    layers = parse_layers(args.layers)
    print(f"Selected layers (0-indexed): {layers}")
    print(f"Selected layers (1-indexed): {[l+1 for l in layers]}")

    evaluator = VAEvaluator(
        model_path=args.model_path,
        layers=layers,
        max_length=args.max_length,
    )

    print("\n=== Running MTEB evaluation on SciFact ===")
    evaluation = MTEB(tasks=mteb.get_tasks(tasks=["SciFact"]))
    results = evaluation.run(
        evaluator,
        output_folder="./va_scifact_results",
        eval_splits=["test"],
        verbosity=2,
        overwrite_results=True,
        encode_kwargs={"batch_size": args.batch_size},
    )

    # Print results (MTEB v1.12 returns list of MTEBResults)
    print(f"\n=== Results (got {len(results)} items) ===")
    print(f"Results type: {type(results)}")
    if results:
        print(f"First item type: {type(results[0])}")
        print(f"First item: {results[0]}")
    ndcg10 = None
    for result in results:
        print(f"Task: {result.task_name}")
        result_dict = result.to_dict()
        for split, scores_list in result_dict.get("scores", {}).items():
            print(f"  Split: {split}")
            for score_entry in scores_list:
                for metric_name, metric_value in score_entry.items():
                    if isinstance(metric_value, (int, float)):
                        print(f"    {metric_name}: {metric_value:.4f}")
                        if metric_name == "ndcg_at_10":
                            ndcg10 = metric_value
                    elif metric_name not in ("hf_subset", "languages", "main_score"):
                        print(f"    {metric_name}: {metric_value}")

    # Extract NDCG@10
    if ndcg10 is not None:
        print(f"\n{'='*60}")
        print(f"SciFact NDCG@10: {ndcg10:.2f}")
        print(f"Paper target:    54.58")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
