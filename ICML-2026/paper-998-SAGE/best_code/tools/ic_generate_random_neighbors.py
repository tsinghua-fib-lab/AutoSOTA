"""
ic_generate_random_neighbors.py

Generate random neighbor files for Image Classification tasks, used for comparison experiments with the KNN neighbor method.
Each sample randomly selects 9 other samples as "neighbors" (excluding itself).

The generated file format is consistent with the neighbor file output by knn_image.py,
and can be directly used for evaluation with neighbor_based_ic_evaluator_vllm.py.

Usage:
    # Generate a single file
    python tools/ic_generate_random_neighbors.py \
        --predictions outputs/image_classification/CIFAR-10_internvl3.5-8b_xxx.json \
        --output outputs/image_classification/CIFAR-10_internvl_random_image_neighbors.jsonl \
        --k 10 --seed 42

    # Batch generate all files (CIFAR-10 + ImageNet-1k, 3 models)
    python tools/ic_generate_random_neighbors.py --batch
"""

import argparse
import json
import os
import random
import glob
from pathlib import Path
from typing import List, Dict, Optional


def load_predictions(file_path: str) -> List[Dict]:
    """Load prediction data file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_random_neighbors(
    predictions: List[Dict],
    source_file: str,
    k: int = 10,
    seed: int = 42
) -> List[Dict]:
    """
    Generate random neighbors for each sample

    Args:
        predictions: List of prediction data
        source_file: Source file name
        k: Number of neighbors (including itself, so actually selects k-1 randomly)
        seed: Random seed

    Returns:
        List of neighbor data, format consistent with knn_image.py output
    """
    random.seed(seed)
    n_samples = len(predictions)
    all_indices = list(range(n_samples))
    
    results = []
    
    for i, item in enumerate(predictions):
        idx = item.get('index', i)
        
        # Randomly select from all samples (excluding itself)
        candidates = [j for j in all_indices if j != i]
        
        # Randomly select k-1 neighbors (consistent with original format: 9 neighbors)
        random_neighbors = random.sample(candidates, min(k - 1, len(candidates)))
        
        # Build neighbor list, cosine set to 0 to indicate random selection
        neighbors = []
        for neighbor_idx in random_neighbors:
            neighbors.append({
                'global_id': neighbor_idx,
                'index': neighbor_idx,
                'cosine': 0.0  # Randomly selected, no actual cosine similarity
            })
        
        # Build result
        result = {
            'global_id': i,
            'source_file': source_file,
            'row_in_file': i,
            'index': idx,
            'neighbors': neighbors
        }
        results.append(result)
    
    return results


def save_jsonl(data: List[Dict], output_path: str):
    """Save in JSONL format"""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f'Saved: {output_path} ({len(data)} entries)')


def generate_single(predictions_path: str, output_path: str, k: int = 10, seed: int = 42):
    """Generate a single random neighbor file"""
    print(f'Loading prediction data: {predictions_path}')
    predictions = load_predictions(predictions_path)
    print(f'   Total {len(predictions)} samples')

    source_file = os.path.basename(predictions_path)

    print(f'Generating random neighbors (k={k}, seed={seed})...')
    results = generate_random_neighbors(predictions, source_file, k=k, seed=seed)
    
    save_jsonl(results, output_path)


def find_prediction_file(output_dir: str, dataset: str, model_pattern: str) -> Optional[str]:
    """Find matching prediction file"""
    pattern = os.path.join(output_dir, f'{dataset}_{model_pattern}*.json')
    files = glob.glob(pattern)
    if files:
        # Return the most recent file
        return sorted(files)[-1]
    return None


def generate_batch(base_dir: str, seed: int = 42):
    """Batch generate all random neighbor files"""
    output_dir = os.path.join(base_dir, 'outputs/image_classification')
    
    # Configuration: (dataset, model_pattern, model_short_name)
    configs = [
        # CIFAR-10
        ('CIFAR-10', 'internvl3.5-8b', 'internvl'),
        ('CIFAR-10', 'qwen3-vl-8b', 'qwen'),
        ('CIFAR-10', 'sailvl-8b', 'sailvl'),
        # ImageNet-1k
        ('ImageNet-1k', 'internvl3.5-8b', 'internvl'),
        ('ImageNet-1k', 'qwen3-vl-8b', 'qwen'),
        ('ImageNet-1k', 'sailvl-8b', 'sailvl'),
    ]

    print('=' * 60)
    print('Batch generating IC random neighbor files')
    print('=' * 60)
    
    generated = 0
    skipped = 0
    
    for dataset, model_pattern, model_short in configs:
        # Find prediction file
        pred_file = find_prediction_file(output_dir, dataset, model_pattern)
        
        if not pred_file:
            print(f'Skipped (prediction file does not exist): {dataset}_{model_pattern}')
            skipped += 1
            continue
        
        # Output filename: {Dataset}_{model}_random_image_neighbors.jsonl
        output_file = f'{dataset}_{model_short}_random_image_neighbors.jsonl'
        output_path = os.path.join(output_dir, output_file)
        
        print(f'\n--- {output_file} ---')
        generate_single(pred_file, output_path, k=10, seed=seed)
        generated += 1
    
    print('\n' + '=' * 60)
    print(f'Done! Generated: {generated}, Skipped: {skipped}')
    print('=' * 60)


def main():
    parser = argparse.ArgumentParser(description='Generate random neighbor files for IC tasks')
    parser.add_argument('--predictions', type=str, help='Prediction data file path')
    parser.add_argument('--output', '-o', type=str, help='Output file path')
    parser.add_argument('--k', type=int, default=10, help='Number of neighbors (default 10, actually selects 9 randomly)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch', action='store_true', help='Batch generate all files')
    parser.add_argument('--base-dir', type=str,
                        default='.',
                        help='Project root directory')
    
    args = parser.parse_args()
    
    if args.batch:
        generate_batch(args.base_dir, seed=args.seed)
    elif args.predictions and args.output:
        generate_single(args.predictions, args.output, k=args.k, seed=args.seed)
    else:
        parser.print_help()
        print('\nPlease specify --batch or both --predictions and --output')


if __name__ == '__main__':
    main()

