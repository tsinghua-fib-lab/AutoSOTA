#!/usr/bin/env python3
"""
Generate TinyBenchmarks item IDs for all datasets
TinyBenchmarks provides curated small subsets of benchmark datasets
"""

from datasets import load_dataset
import pandas as pd
import os

print("=" * 70)
print("Generating TinyBenchmarks Item IDs for All Datasets")
print("=" * 70)
print()

# TinyBenchmarks configuration
# Each dataset may have different splits and configurations
TINY_CONFIGS = [
    {
        'name': 'ARC',
        'dataset': 'tinyBenchmarks/tinyAI2_arc',
        'config': 'ARC-Challenge',
        'split': 'test',
        'id_field': 'id'
    },
    {
        'name': 'GSM8K',
        'dataset': 'tinyBenchmarks/tinyGSM8K',
        'config': 'main',
        'split': 'test',
        'id_field': None  # Will use index
    },
    {
        'name': 'HellaSwag',
        'dataset': 'tinyBenchmarks/tinyHellaswag',
        'config': None,
        'split': 'validation',
        'id_field': 'ind'
    },
    {
        'name': 'TruthfulQA',
        'dataset': 'tinyBenchmarks/tinyTruthfulQA',
        'config': 'multiple_choice',
        'split': 'validation',
        'id_field': None  # Will need manual mapping
    },
    {
        'name': 'Winogrande',
        'dataset': 'tinyBenchmarks/tinyWinogrande',
        'config': 'winogrande_xl',
        'split': 'validation',
        'id_field': None  # Will use index
    }
]

# Path to benchmark prompts for matching (if needed)
import argparse as _argparse
_parser = _argparse.ArgumentParser(add_help=False)
_parser.add_argument('--prompts_dir', default=None)
_known, _ = _parser.parse_known_args()
PROMPTS_DIR = _known.prompts_dir or 'data/metabench_data/benchmark-data'

def get_item_ids_from_tiny(config):
    """Extract item IDs from tiny benchmark dataset"""
    name = config['name']
    print(f"Processing {name}...")
    
    try:
        # Load tiny dataset
        if config['config']:
            ds = load_dataset(config['dataset'], config['config'])[config['split']]
        else:
            ds = load_dataset(config['dataset'])[config['split']]
        
        n_items = len(ds)
        print(f"  Found {n_items} items in tiny{name}")
        
        # Try to extract item IDs
        item_ids = []
        
        if config['id_field'] and config['id_field'] in ds.features:
            # Use the ID field directly
            item_ids = ds[config['id_field']]
            print(f"  ✓ Using '{config['id_field']}' field for item IDs")
        else:
            # Need to match with original dataset or use prompts
            print(f"  ⚠ No direct ID field, attempting to match with prompts...")
            
            prompts_path = f"{PROMPTS_DIR}/{name.lower()}_prompts.csv"
            if os.path.exists(prompts_path):
                prompts_df = pd.read_csv(prompts_path)
                
                # Get questions from tiny dataset
                if 'question' in ds.features:
                    questions = ds['question']
                elif 'input_formatted' in ds.features:
                    questions = ds['input_formatted']
                else:
                    print(f"  ✗ Cannot find question field in tiny{name}")
                    return None
                
                # Match questions to prompts
                for question in questions:
                    found = False
                    for _, row in prompts_df.iterrows():
                        prompt = str(row['prompt'])
                        if question in prompt:
                            item_ids.append(row['item'])
                            found = True
                            break
                    if not found:
                        # Use index as fallback
                        item_ids.append(len(item_ids))
                
                print(f"  ✓ Matched {sum([1 for x in item_ids if x != 'not_found'])} items via prompts")
            else:
                # Fallback: use indices
                print(f"  ⚠ Prompts file not found, using indices as item IDs")
                item_ids = list(range(n_items))
        
        # Create DataFrame
        df = pd.DataFrame({'item_id': item_ids})
        
        # Save to CSV
        output_path = f'tiny{name.lower()}_item_ids.csv'
        df.to_csv(output_path, index=False)
        
        print(f"  ✓ Saved to {output_path}")
        print(f"  First few items: {', '.join(map(str, item_ids[:10]))}")
        
        return df
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None

# Process all datasets
print()
results = {}
for config in TINY_CONFIGS:
    df = get_item_ids_from_tiny(config)
    if df is not None:
        results[config['name']] = df
    print()

print("=" * 70)
print(f"Successfully processed {len(results)}/{len(TINY_CONFIGS)} datasets")
print("=" * 70)
print()
print("Generated files:")
for name in results.keys():
    print(f"  - tiny{name.lower()}_item_ids.csv")
print()
