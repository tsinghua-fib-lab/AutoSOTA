"""
vlm_generate_random_neighbors.py

Generate random neighbor files, used for comparison experiments with the KNN neighbor method.
Each sample randomly selects 10 other samples as "neighbors" (excluding itself).

The generated file format is consistent with the neighbor file output by knn_image_vlm.py,
and can be directly used for evaluation with neighbor_based_vlm_evaluator_vllm.py.

Usage:
    # Generate a single file
    python tools/vlm_generate_random_neighbors.py \
        --vlm-data outputs/vlm_tagging/COCO_internvl3.5-8b_v1.json \
        --output outputs/vlm_tagging/COCO_internvl_random_image_neighbors.jsonl \
        --k 10 --seed 42

    # Batch generate all 4 files
    python tools/vlm_generate_random_neighbors.py --batch
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Dict


def load_vlm_data(file_path: str) -> List[Dict]:
    """Load VLM data file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_random_neighbors(
    vlm_data: List[Dict],
    source_file: str,
    k: int = 10,
    seed: int = 42
) -> List[Dict]:
    """
    Generate random neighbors for each sample

    Args:
        vlm_data: List of VLM data
        source_file: Source file name
        k: Number of neighbors (including itself, so actually selects k randomly)
        seed: Random seed

    Returns:
        List of neighbor data, format consistent with knn_image_vlm.py output
    """
    random.seed(seed)
    n_samples = len(vlm_data)
    all_indices = list(range(n_samples))
    
    results = []
    
    for i, item in enumerate(vlm_data):
        idx = item.get('index', i)
        
        # Randomly select k from all samples (excluding itself)
        candidates = [j for j in all_indices if j != i]
        
        # Randomly select k neighbors (the last position will be itself, added by evaluator)
        # So here we select k-1 random samples, consistent with original format (9 neighbors)
        random_neighbors = random.sample(candidates, min(k - 1, len(candidates)))
        
        # Build neighbor list, cosine set to 0 to indicate random selection
        neighbors = []
        for neighbor_idx in random_neighbors:
            neighbors.append({
                'global_id': neighbor_idx,
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


def generate_single(vlm_data_path: str, output_path: str, k: int = 10, seed: int = 42):
    """Generate a single random neighbor file"""
    print(f'Loading VLM data: {vlm_data_path}')
    vlm_data = load_vlm_data(vlm_data_path)
    print(f'   Total {len(vlm_data)} samples')

    source_file = os.path.basename(vlm_data_path)

    print(f'Generating random neighbors (k={k}, seed={seed})...')
    results = generate_random_neighbors(vlm_data, source_file, k=k, seed=seed)
    
    save_jsonl(results, output_path)


def generate_batch(base_dir: str, seed: int = 42):
    """Batch generate all 4 random neighbor files"""
    output_dir = os.path.join(base_dir, 'outputs/vlm_tagging')
    
    # Configuration: (vlm_data file, output filename)
    configs = [
        ('COCO_qwen3-vl-8b_v1.json', 'COCO_qwen_random_image_neighbors.jsonl'),
        ('COCO_internvl3.5-8b_v1.json', 'COCO_internvl_random_image_neighbors.jsonl'),
        ('Flickr30k_qwen3-vl-8b_v1.json', 'Flickr30k_qwen_random_image_neighbors.jsonl'),
        ('Flickr30k_internvl3.5-8b_v1.json', 'Flickr30k_internvl_random_image_neighbors.jsonl'),
    ]
    
    print('=' * 60)
    print('Batch generating random neighbor files')
    print('=' * 60)
    
    for vlm_file, output_file in configs:
        vlm_path = os.path.join(output_dir, vlm_file)
        output_path = os.path.join(output_dir, output_file)
        
        if not os.path.exists(vlm_path):
            print(f'Skipped (file does not exist): {vlm_path}')
            continue
        
        print(f'\n--- {output_file} ---')
        generate_single(vlm_path, output_path, k=10, seed=seed)
    
    print('\n' + '=' * 60)
    print('All files generated successfully!')
    print('=' * 60)


def main():
    parser = argparse.ArgumentParser(description='Generate random neighbor files')
    parser.add_argument('--vlm-data', type=str, help='VLM data file path')
    parser.add_argument('--output', '-o', type=str, help='Output file path')
    parser.add_argument('--k', type=int, default=10, help='Number of neighbors (default 10, actually selects 9 randomly)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch', action='store_true', help='Batch generate all 4 files')
    parser.add_argument('--base-dir', type=str,
                        default='.',
                        help='Project root directory')
    
    args = parser.parse_args()
    
    if args.batch:
        generate_batch(args.base_dir, seed=args.seed)
    elif args.vlm_data and args.output:
        generate_single(args.vlm_data, args.output, k=args.k, seed=args.seed)
    else:
        parser.print_help()
        print('\nPlease specify --batch or both --vlm-data and --output')


if __name__ == '__main__':
    main()

