#!/usr/bin/env python3
"""
Alibaba Cluster Trace v2018 Data Downloader

Downloads the real Alibaba cluster trace data from Alibaba's OSS mirror.

Data Source:
    Primary: Alibaba OSS (China): http://clusterdata2018pubcn.oss-cn-beijing.aliyuncs.com/
    Backup: GitHub: https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018

Data Statistics (Full Dataset):
    - File: batch_task.tar.gz (~285MB compressed)
    - Rows: ~4.2 million tasks
    - Time span: 8 days of cluster operations
    - Columns: task_name, instance_num, job_name, task_type, status,
               start_time, end_time, plan_cpu, plan_mem

Usage:
    # Download real data (full ~4.2M rows)
    python download_data.py

    # Download and limit to first N rows
    python download_data.py --max-rows 100000

    # Create synthetic data for quick testing
    python download_data.py --synthetic --samples 10000

    # Force re-download even if file exists
    python download_data.py --force
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import requests
from tqdm import tqdm
import tarfile
import gzip
import shutil
import json
from datetime import datetime


# Alibaba OSS mirror (most reliable)
ALIBABA_OSS_URL = "http://clusterdata2018pubcn.oss-cn-beijing.aliyuncs.com/batch_task.tar.gz"

# Backup URLs
BACKUP_URLS = [
    "https://github.com/alibaba/clusterdata/raw/master/cluster-trace-v2018/batch_task.tar.gz",
]


def download_file(url: str, output_path: Path, chunk_size: int = 8192) -> bool:
    """Download a file with progress bar."""
    try:
        print(f"Downloading from: {url}")
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        print(f"Downloaded: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
        return True

    except requests.exceptions.Timeout:
        print(f"Download timed out. The file is large (~285MB), try again or use manual download.")
        return False
    except Exception as e:
        print(f"Download failed: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def extract_tar_gz(tar_path: Path, output_dir: Path) -> Path:
    """Extract .tar.gz file and return path to CSV."""
    try:
        print(f"Extracting: {tar_path}")
        with tarfile.open(tar_path, 'r:gz') as tar:
            # List contents
            members = tar.getnames()
            print(f"  Archive contains: {members}")
            tar.extractall(path=output_dir)

        # Find the extracted CSV
        for f in output_dir.iterdir():
            if f.suffix == '.csv' and 'batch_task' in f.name.lower():
                print(f"Extracted: {f}")
                return f

        # Check subdirectories
        for subdir in output_dir.iterdir():
            if subdir.is_dir():
                for f in subdir.iterdir():
                    if f.suffix == '.csv' and 'batch_task' in f.name.lower():
                        print(f"Extracted: {f}")
                        return f

        raise FileNotFoundError("Could not find batch_task.csv in archive")

    except Exception as e:
        print(f"Extraction failed: {e}")
        return None


def process_data(input_path: Path, output_path: Path, max_rows: int = None) -> dict:
    """
    Process the raw Alibaba data and save cleaned version.

    The Alibaba batch_task.csv has NO header row. Columns are:
    0: task_name, 1: instance_num, 2: job_name, 3: task_type,
    4: status, 5: start_time, 6: end_time, 7: plan_cpu, 8: plan_mem

    Note: plan_cpu is 0-100 scale, plan_mem is 0-1 scale (needs *100)

    Returns statistics about the data.
    """
    print(f"\nProcessing data...")
    print(f"  Input: {input_path}")

    # Define column names (Alibaba data has no header)
    column_names = [
        'task_name', 'instance_num', 'job_name', 'task_type',
        'status', 'start_time', 'end_time', 'plan_cpu', 'plan_mem'
    ]

    # Read data (may be large)
    print("  Reading CSV (this may take a moment for large files)...")

    if max_rows:
        df = pd.read_csv(input_path, nrows=max_rows, header=None, names=column_names)
        print(f"  Loaded first {max_rows:,} rows")
    else:
        df = pd.read_csv(input_path, header=None, names=column_names)
        print(f"  Loaded all {len(df):,} rows")

    # Check columns
    print(f"  Columns: {list(df.columns)}")

    # Convert plan_mem from 0-1 to 0-100 scale (to match plan_cpu)
    print("  Converting plan_mem from 0-1 to 0-100 scale...")
    df['plan_mem'] = df['plan_mem'] * 100

    required_cols = ['plan_cpu', 'plan_mem', 'start_time', 'end_time']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Filter valid rows
    original_count = len(df)

    # Remove rows with missing values in required columns
    df = df.dropna(subset=required_cols)

    # Filter for valid timestamps (end_time > start_time)
    df = df[df['end_time'] > df['start_time']]

    # Filter for valid resource values (cpu, mem > 0)
    df = df[(df['plan_cpu'] > 0) & (df['plan_mem'] > 0)]

    # Sort by start_time to preserve temporal order
    df = df.sort_values('start_time').reset_index(drop=True)

    filtered_count = len(df)
    print(f"  Filtered: {original_count:,} -> {filtered_count:,} rows ({filtered_count/original_count*100:.1f}%)")

    # Compute statistics
    duration = df['end_time'] - df['start_time']
    stats = {
        'total_rows': filtered_count,
        'original_rows': original_count,
        'time_span_seconds': int(df['end_time'].max() - df['start_time'].min()),
        'time_span_days': float((df['end_time'].max() - df['start_time'].min()) / 86400),
        'cpu_mean': float(df['plan_cpu'].mean()),
        'cpu_std': float(df['plan_cpu'].std()),
        'mem_mean': float(df['plan_mem'].mean()),
        'mem_std': float(df['plan_mem'].std()),
        'duration_mean': float(duration.mean()),
        'duration_std': float(duration.std()),
        'duration_min': float(duration.min()),
        'duration_max': float(duration.max()),
        'download_date': datetime.now().isoformat(),
        'source': 'alibaba_cluster_trace_v2018',
    }

    # Save processed data
    df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")

    return stats


def create_synthetic_data(output_path: Path, n_samples: int = 10000) -> dict:
    """Create synthetic data matching Alibaba trace structure."""
    print(f"\nCreating synthetic data ({n_samples:,} samples)...")
    print("NOTE: This is synthetic data for testing. Use real data for experiments.")

    np.random.seed(42)

    # Generate data matching Alibaba trace characteristics
    # Based on real data statistics
    data = {
        'task_name': [f'task_{i}' for i in range(n_samples)],
        'instance_num': np.random.randint(1, 100, n_samples),
        'job_name': [f'job_{i // 10}' for i in range(n_samples)],
        'task_type': np.random.choice(['', 'map', 'reduce'], n_samples),
        'status': np.random.choice(['Terminated', 'Failed', 'Running', 'Waiting'],
                                   n_samples, p=[0.7, 0.1, 0.15, 0.05]),
        'plan_cpu': np.random.exponential(20, n_samples).clip(1, 100),
        'plan_mem': np.random.exponential(25, n_samples).clip(1, 100),
        'start_time': np.sort(np.random.randint(0, 8 * 24 * 3600, n_samples)),  # 8 days
    }

    # Duration: exponential distribution with mean ~300 seconds
    duration = np.random.exponential(300, n_samples).astype(int).clip(1, 7200)
    data['end_time'] = data['start_time'] + duration

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)

    stats = {
        'total_rows': n_samples,
        'original_rows': n_samples,
        'synthetic': True,
        'time_span_seconds': int(df['end_time'].max() - df['start_time'].min()),
        'time_span_days': float((df['end_time'].max() - df['start_time'].min()) / 86400),
        'cpu_mean': float(df['plan_cpu'].mean()),
        'mem_mean': float(df['plan_mem'].mean()),
        'duration_mean': float(duration.mean()),
        'download_date': datetime.now().isoformat(),
        'source': 'synthetic',
    }

    print(f"Saved: {output_path}")
    return stats


def save_metadata(stats: dict, output_dir: Path):
    """Save data metadata/statistics."""
    meta_path = output_dir / 'data_metadata.json'
    with open(meta_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved metadata: {meta_path}")


def print_stats(stats: dict):
    """Print data statistics."""
    print("\n" + "=" * 70)
    print("DATA STATISTICS")
    print("=" * 70)
    print(f"  Source: {stats.get('source', 'unknown')}")
    print(f"  Total rows: {stats['total_rows']:,}")
    if stats.get('original_rows') and stats['original_rows'] != stats['total_rows']:
        print(f"  Original rows: {stats['original_rows']:,}")
    print(f"  Time span: {stats['time_span_days']:.1f} days")
    print(f"  CPU - mean: {stats['cpu_mean']:.2f}, std: {stats.get('cpu_std', 0):.2f}")
    print(f"  Memory - mean: {stats['mem_mean']:.2f}, std: {stats.get('mem_std', 0):.2f}")
    print(f"  Duration - mean: {stats['duration_mean']:.1f}s, range: [{stats.get('duration_min', 0):.0f}, {stats.get('duration_max', 0):.0f}]s")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Download Alibaba Cluster Trace v2018',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python download_data.py                    # Download full dataset (~4.2M rows)
    python download_data.py --max-rows 100000  # Download and keep first 100K rows
    python download_data.py --synthetic        # Create synthetic test data
    python download_data.py --force            # Force re-download
        """
    )
    parser.add_argument('--output', type=str, default='./data/raw',
                       help='Output directory for downloaded data')
    parser.add_argument('--synthetic', action='store_true',
                       help='Create synthetic data instead of downloading')
    parser.add_argument('--samples', type=int, default=10000,
                       help='Number of samples for synthetic data')
    parser.add_argument('--max-rows', type=int, default=None,
                       help='Maximum rows to keep from real data (None = all)')
    parser.add_argument('--force', action='store_true',
                       help='Force re-download even if file exists')

    args = parser.parse_args()

    # Setup paths
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / 'batch_task.csv'
    tar_path = output_dir / 'batch_task.tar.gz'
    meta_path = output_dir / 'data_metadata.json'

    print("=" * 70)
    print("ALIBABA CLUSTER TRACE v2018 DATA DOWNLOADER")
    print("=" * 70)

    # Check if data already exists
    if csv_path.exists() and not args.force and not args.synthetic:
        print(f"\nData already exists at: {csv_path}")
        if meta_path.exists():
            with open(meta_path) as f:
                stats = json.load(f)
            print_stats(stats)
        print("\nUse --force to re-download.")
        return 0

    # Synthetic data mode
    if args.synthetic:
        stats = create_synthetic_data(csv_path, args.samples)
        save_metadata(stats, output_dir)
        print_stats(stats)
        return 0

    # Download real data
    print("\nDownloading real Alibaba trace data...")
    print("Note: This file is ~285MB compressed, download may take several minutes.")
    print("      The full dataset contains ~4.2 million rows.\n")

    # Try primary URL
    success = download_file(ALIBABA_OSS_URL, tar_path)

    # Try backup URLs if primary fails
    if not success:
        for backup_url in BACKUP_URLS:
            print(f"\nTrying backup URL...")
            success = download_file(backup_url, tar_path)
            if success:
                break

    if not success:
        print("\n" + "=" * 70)
        print("AUTOMATIC DOWNLOAD FAILED")
        print("=" * 70)
        print("\nPlease manually download the data:")
        print(f"\n1. Visit: http://clusterdata2018pubcn.oss-cn-beijing.aliyuncs.com/batch_task.tar.gz")
        print(f"2. Save to: {tar_path}")
        print(f"3. Run this script again\n")
        print("Or use synthetic data for testing:")
        print(f"    python {sys.argv[0]} --synthetic")
        print("\n" + "=" * 70)
        return 1

    # Extract
    extracted_path = extract_tar_gz(tar_path, output_dir)
    if not extracted_path:
        print("Extraction failed!")
        return 1

    # Process data
    try:
        stats = process_data(extracted_path, csv_path, max_rows=args.max_rows)
        save_metadata(stats, output_dir)
        print_stats(stats)
    except Exception as e:
        print(f"Processing failed: {e}")
        return 1

    # Clean up
    if tar_path.exists():
        tar_path.unlink()
        print(f"Cleaned up: {tar_path}")

    # Remove original extracted file if different from output
    if extracted_path != csv_path and extracted_path.exists():
        extracted_path.unlink()

    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"\nData saved to: {csv_path}")
    print(f"Metadata saved to: {meta_path}")
    print("\nYou can now run experiments with:")
    print("    python run_experiment.py --smoke  # Quick test")
    print("    python run_experiment.py          # Full experiment")

    return 0


if __name__ == '__main__':
    sys.exit(main())
