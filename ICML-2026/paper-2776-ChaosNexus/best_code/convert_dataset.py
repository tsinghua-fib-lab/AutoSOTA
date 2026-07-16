import os, sys
import pandas as pd
import numpy as np
from pathlib import Path

def convert_parquet_to_subdirs(input_parquet, output_base_dir, max_systems=None):
    """Convert a single parquet file to subdirectory structure expected by code."""
    print(f"Reading {input_parquet}...")
    df = pd.read_parquet(input_parquet)
    print(f"  Rows: {len(df)}, unique systems: {df['_source_directory'].nunique()}")
    
    # Group by system name
    for sys_name, group in df.groupby('_source_directory'):
        sys_dir = Path(output_base_dir) / sys_name
        os.makedirs(sys_dir, exist_ok=True)
        
        for idx, (_, row) in enumerate(group.iterrows()):
            # Parse source filename: "{sample_idx}_T-{length}.arrow"
            fname = row['_source_filename']
            out_path = sys_dir / fname
            
            # Write single-row parquet
            single_df = pd.DataFrame([row])
            single_df.to_parquet(out_path, index=False)
        
        if (len(df.groupby('_source_directory').size()) <= 5 or 
            list(df.groupby('_source_directory').size().index).index(sys_name) % 200 == 0):
            pass  # suppress per-system logging for speed
    
    print(f"  Wrote {len(df)} files to {output_base_dir} ({df['_source_directory'].nunique()} systems)")

if __name__ == '__main__':
    indir = '/autosota_cache/tmp/skew40_raw'
    outdir = '/repo/data'
    
    # Convert train
    convert_parquet_to_subdirs(
        os.path.join(indir, 'train.parquet'),
        os.path.join(outdir, 'train')
    )
    
    # Convert test
    convert_parquet_to_subdirs(
        os.path.join(indir, 'test_zeroshot.parquet'),
        os.path.join(outdir, 'test')
    )
    
    print("Conversion complete!")
    # Print summary
    for split in ['train', 'test']:
        d = Path(outdir) / split
        n_dirs = len(list(d.iterdir()))
        n_files = len(list(d.rglob('*')))
        print(f"  {split}/: {n_dirs} subdirs, {n_files} files")
