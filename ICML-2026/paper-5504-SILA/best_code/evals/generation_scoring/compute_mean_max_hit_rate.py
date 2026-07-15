#!/usr/bin/env python3
"""
Compute the mean-max hit rate metric for SAE evaluation results.

Metric definition:
- For each latent: compute MEAN hit rate for each of the 6 labels (across 10 trials each)
- For each latent: take the MAX of these 6 mean hit rates
- Report the MEAN of these max values across all latents

Hit definition: A trial is a "hit" if ANY activation after index 0 (skipping BOS token) is > 0.0
"""

import json
import sys
from collections import defaultdict


def is_hit(per_token_activations):
    """
    Check if this trial is a hit.
    A trial is a hit if any activation AFTER THE FIRST token (index 0) is > 0.0
    """
    if len(per_token_activations) <= 1:
        return False
    
    # Check indices 1, 2, 3, ... (skipping index 0 which is BOS)
    for activation in per_token_activations[1:]:
        if activation > 0.0:
            return True
    return False


def compute_hit_rate_for_label(reward_samples):
    """
    Compute the hit rate for a single label across its reward samples.
    Returns (num_hits, num_valid_samples, hit_rate, sample_details)
    """
    sample_details = []
    num_hits = 0
    num_valid = 0
    
    for sample in reward_samples:
        # Skip samples with errors
        if sample.get('error') is not None:
            continue
        
        num_valid += 1
        hit = is_hit(sample['per_token_activations'])
        if hit:
            num_hits += 1
        
        sample_details.append({
            'sample_index': sample.get('sample_index'),
            'is_hit': hit,
            'num_activations': len(sample['per_token_activations']),
            'num_positive_after_bos': sum(1 for a in sample['per_token_activations'][1:] if a > 0.0),
        })
    
    hit_rate = num_hits / num_valid if num_valid > 0 else 0.0
    return num_hits, num_valid, hit_rate, sample_details


def process_label_dataset_file(filepath, verbose=False, split_60_trials=False):
    """
    Process a label_dataset_merged format file.
    Structure: evaluations list where each entry has latent_index, label, reward_samples
    Multiple entries can have the same latent_index (different paraphrased labels)
    
    If split_60_trials=True, assumes each label has 60 trials and splits them into
    6 groups of 10 trials each to match the best-of-6 protocol.
    """
    print(f"\n{'='*80}")
    print(f"Processing label_dataset file: {filepath}")
    if split_60_trials:
        print("Mode: Splitting 60 trials into 6 groups of 10")
    print(f"{'='*80}\n")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Group evaluations by latent_index
    latents = defaultdict(list)
    for eval_entry in data['evaluations']:
        latent_idx = eval_entry['latent_index']
        latents[latent_idx].append(eval_entry)
    
    print(f"Total unique latents: {len(latents)}")
    
    # For each latent, compute max hit rate
    latent_max_hit_rates = []
    
    for latent_idx in sorted(latents.keys()):
        labels = latents[latent_idx]
        
        if verbose and len(latent_max_hit_rates) < 3:  # Show details for first 3 latents
            print(f"\n--- Latent {latent_idx} ---")
            if split_60_trials:
                print(f"Number of labels: {len(labels)} (will split into 6 groups each)")
            else:
                print(f"Number of labels: {len(labels)}")
        
        label_hit_rates = []
        
        for i, label_entry in enumerate(labels):
            label_text = label_entry.get('label', 'N/A')
            
            if split_60_trials:
                # Split the 60 trials into 6 groups of 10
                reward_samples = label_entry['reward_samples']
                if verbose and len(latent_max_hit_rates) < 3:
                    print(f"\n  Original label: {label_text[:70]}...")
                    print(f"    Total trials: {len(reward_samples)}")
                    print(f"    Splitting into 6 groups of 10:")
                
                for group_idx in range(6):
                    start_idx = group_idx * 10
                    end_idx = start_idx + 10
                    group_samples = reward_samples[start_idx:end_idx]
                    
                    num_hits, num_valid, hit_rate, sample_details = compute_hit_rate_for_label(
                        group_samples
                    )
                    label_hit_rates.append(hit_rate)
                    
                    if verbose and len(latent_max_hit_rates) < 3:
                        print(f"      Group {group_idx} (samples {start_idx}-{end_idx-1}): "
                              f"Hits: {num_hits}/{num_valid} = {hit_rate:.4f}")
            else:
                # Normal processing - each label_entry is one label with ~10 trials
                num_hits, num_valid, hit_rate, sample_details = compute_hit_rate_for_label(
                    label_entry['reward_samples']
                )
                label_hit_rates.append(hit_rate)
                
                if verbose and len(latent_max_hit_rates) < 3:
                    print(f"\n  Label {i}: {label_text[:70]}...")
                    print(f"    Hits: {num_hits}/{num_valid} = {hit_rate:.4f}")
                    if len(sample_details) <= 10:  # Show all samples if <=10
                        for detail in sample_details:
                            print(f"      Sample {detail['sample_index']}: " 
                                  f"hit={detail['is_hit']}, "
                                  f"tokens={detail['num_activations']}, "
                                  f"positive_after_bos={detail['num_positive_after_bos']}")
        
        # Take the max hit rate across all labels for this latent
        max_hit_rate = max(label_hit_rates) if label_hit_rates else 0.0
        latent_max_hit_rates.append(max_hit_rate)
        
        if verbose and len(latent_max_hit_rates) <= 3:
            print(f"\n  All label/group hit rates: {[f'{hr:.4f}' for hr in label_hit_rates]}")
            print(f"  MAX hit rate for latent {latent_idx}: {max_hit_rate:.4f}")
    
    # Compute mean of max hit rates
    mean_max_hit_rate = sum(latent_max_hit_rates) / len(latent_max_hit_rates) if latent_max_hit_rates else 0.0
    
    print(f"\n{'='*80}")
    print(f"RESULTS for {filepath}")
    print(f"{'='*80}")
    print(f"Number of latents evaluated: {len(latent_max_hit_rates)}")
    print(f"Max hit rates across latents (first 10): {[f'{hr:.4f}' for hr in latent_max_hit_rates[:10]]}")
    print(f"\nFINAL METRIC - Mean of max hit rates: {mean_max_hit_rate:.6f}")
    print(f"{'='*80}\n")
    
    return mean_max_hit_rate, latent_max_hit_rates


def process_label_generator_file(filepath, verbose=False):
    """
    Process a label_generator_merged format file.
    Structure: evaluations list where each entry has latent_index and scale_evaluations
    Each scale_evaluation has a scale_value and generated_labels
    """
    print(f"\n{'='*80}")
    print(f"Processing label_generator file: {filepath}")
    print(f"{'='*80}\n")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    print(f"Total evaluations (latents): {len(data['evaluations'])}")
    
    latent_max_hit_rates = []
    
    for eval_entry in data['evaluations']:
        latent_idx = eval_entry['latent_index']
        scale_evaluations = eval_entry['scale_evaluations']
        
        if verbose and len(latent_max_hit_rates) < 3:  # Show details for first 3 latents
            print(f"\n--- Latent {latent_idx} ---")
            print(f"Number of scale_evaluations: {len(scale_evaluations)}")
        
        label_hit_rates = []
        
        for scale_eval in scale_evaluations:
            scale_value = scale_eval['scale_value']
            generated_labels = scale_eval['generated_labels']
            
            # There should typically be 1 generated label per scale
            for j, label_entry in enumerate(generated_labels):
                label_text = label_entry.get('label', 'N/A')
                num_hits, num_valid, hit_rate, sample_details = compute_hit_rate_for_label(
                    label_entry['reward_samples']
                )
                label_hit_rates.append(hit_rate)
                
                if verbose and len(latent_max_hit_rates) < 3:
                    print(f"\n  Scale {scale_value} (label {j}): {label_text[:70]}...")
                    print(f"    Hits: {num_hits}/{num_valid} = {hit_rate:.4f}")
                    if len(sample_details) <= 10:
                        for detail in sample_details:
                            print(f"      Sample {detail['sample_index']}: "
                                  f"hit={detail['is_hit']}, "
                                  f"tokens={detail['num_activations']}, "
                                  f"positive_after_bos={detail['num_positive_after_bos']}")
        
        # Take the max hit rate across all labels (scales) for this latent
        max_hit_rate = max(label_hit_rates) if label_hit_rates else 0.0
        latent_max_hit_rates.append(max_hit_rate)
        
        if verbose and len(latent_max_hit_rates) <= 3:
            print(f"\n  All label hit rates: {[f'{hr:.4f}' for hr in label_hit_rates]}")
            print(f"  MAX hit rate for latent {latent_idx}: {max_hit_rate:.4f}")
    
    # Compute mean of max hit rates
    mean_max_hit_rate = sum(latent_max_hit_rates) / len(latent_max_hit_rates) if latent_max_hit_rates else 0.0
    
    print(f"\n{'='*80}")
    print(f"RESULTS for {filepath}")
    print(f"{'='*80}")
    print(f"Number of latents evaluated: {len(latent_max_hit_rates)}")
    print(f"Max hit rates across latents (first 10): {[f'{hr:.4f}' for hr in latent_max_hit_rates[:10]]}")
    print(f"\nFINAL METRIC - Mean of max hit rates: {mean_max_hit_rate:.6f}")
    print(f"{'='*80}\n")
    
    return mean_max_hit_rate, latent_max_hit_rates


def main():
    if len(sys.argv) < 2:
        print("Usage: python compute_mean_max_hit_rate.py <filepath> [--verbose] [--split-60]")
        print("\nOptions:")
        print("  --verbose, -v    Show detailed output for debugging")
        print("  --split-60       Split 60 trials into 6 groups of 10 (for 60x files)")
        print("\nExample:")
        print("  python compute_mean_max_hit_rate.py results/merged_results_6paraphrases_llamascope.json --verbose")
        print("  python compute_mean_max_hit_rate.py results/merged_results_llamascope_60x.json --split-60")
        sys.exit(1)
    
    filepath = sys.argv[1]
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    split_60_trials = '--split-60' in sys.argv
    
    # Detect file type by reading the evaluation_mode
    with open(filepath, 'r') as f:
        data = json.load(f)
        evaluation_mode = data.get('evaluation_mode', '')
    
    if 'label_dataset' in evaluation_mode:
        mean_max_hit_rate, latent_max_hit_rates = process_label_dataset_file(filepath, verbose, split_60_trials)
    elif 'label_generator' in evaluation_mode:
        mean_max_hit_rate, latent_max_hit_rates = process_label_generator_file(filepath, verbose)
    else:
        print(f"Unknown evaluation_mode: {evaluation_mode}")
        sys.exit(1)
    
    return mean_max_hit_rate


if __name__ == "__main__":
    main()

