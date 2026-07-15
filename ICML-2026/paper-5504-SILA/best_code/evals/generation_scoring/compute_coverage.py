#!/usr/bin/env python3
"""Compute coverage metric from eval results JSON."""
import json
import sys

def compute_coverage(filepath):
    with open(filepath) as f:
        data = json.load(f)
    
    evaluations = data.get("evaluations", [])
    total_latents = len(evaluations)
    latents_with_hits = 0
    
    for eval_entry in evaluations:
        has_hit = False
        for scale_eval in eval_entry.get("scale_evaluations", []):
            for label_entry in scale_eval.get("generated_labels", []):
                for sample in label_entry.get("reward_samples", []):
                    if sample.get("error") is not None:
                        continue
                    activations = sample.get("per_token_activations", [])
                    # Check for any nonzero activation (skip index 0 = BOS)
                    if any(a > 0.0 for a in activations[1:] if len(activations) > 1):
                        has_hit = True
                        break
                if has_hit:
                    break
            if has_hit:
                break
        if has_hit:
            latents_with_hits += 1
    
    coverage = (latents_with_hits / total_latents * 100.0) if total_latents > 0 else 0.0
    print(f"Total latents: {total_latents}")
    print(f"Latents with hits: {latents_with_hits}")
    print(f"Coverage: {coverage:.2f}%")
    return coverage

if __name__ == "__main__":
    if len(sys.argv) < 2:
        filepath = "/repo/evals/generation_scoring/results/eval_results_layer_19_fast_eval_50.json"
    else:
        filepath = sys.argv[1]
    compute_coverage(filepath)
