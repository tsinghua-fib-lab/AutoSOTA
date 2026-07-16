#!/usr/bin/env python3
"""
Generate Metabench item IDs for all datasets
Creates both primary and secondary item ID files for each dataset
"""

from datasets import load_dataset
import pandas as pd
import os

# List of all metabench datasets
DATASETS = ['ARC', 'GSM8K', 'HellaSwag', 'TruthfulQA', 'Winogrande']

print("=" * 70)
print("Generating Metabench Item IDs for All Datasets")
print("=" * 70)
print()

for dataset_name in DATASETS:
    print(f"Processing {dataset_name}...")
    
    try:
        # Load metabench dataset
        ds = load_dataset("HCAI/metabench", dataset_name)
        
        # Extract item IDs for primary and secondary
        item_ids_primary = ds['primary']['metabench_idx']
        item_ids_secondary = ds['secondary']['metabench_idx']
        
        # Create DataFrames
        df_primary = pd.DataFrame({'item_id': item_ids_primary})
        df_secondary = pd.DataFrame({'item_id': item_ids_secondary})
        
        # Save to CSV files
        output_path_primary = f'metabench_{dataset_name.lower()}_item_ids_primary.csv'
        output_path_secondary = f'metabench_{dataset_name.lower()}_item_ids_secondary.csv'
        
        df_primary.to_csv(output_path_primary, index=False)
        df_secondary.to_csv(output_path_secondary, index=False)
        
        print(f"  ✓ Primary: {len(item_ids_primary)} items → {output_path_primary}")
        print(f"  ✓ Secondary: {len(item_ids_secondary)} items → {output_path_secondary}")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print()

print("=" * 70)
print("All metabench item IDs generated!")
print("=" * 70)
