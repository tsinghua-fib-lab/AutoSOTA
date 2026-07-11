import os
import pandas as pd
import h5py
import numpy as np

root = "/media/data/BEND_backups/data"
tasks = [
    "enhancer_annotation",
    "gene_finding",
    "chromatin_accessibility",
    "histone_modification",
    "cpg_methylation",
    "disease_vep",
    "expression_vep"
]

def dataset_summary(task, label_depth=None):
    print("=" * 80)
    print(f"📊 DATASET: {task}")
    print("=" * 80)

    bed_file = os.path.join(root, task, f"{task}.bed")
    hdf5_file = os.path.join(root, task, f"{task}.hdf5")

    if not os.path.exists(bed_file):
        print(f"❌ BED file not found: {bed_file}")
        return
    bed = pd.read_csv(bed_file, sep='\t', low_memory=False)

    # Basic info
    print("\n[Basic Info]")
    print(f"Total entries: {len(bed)}")
    print(f"Columns: {list(bed.columns)}")
    print(f"Splits: {bed['split'].unique().tolist() if 'split' in bed.columns else 'N/A'}")

    # Chromosome and strand distribution
    print("\n[Chromosome Distribution]")
    print(bed.iloc[:, 0].value_counts())

    if 'strand' in bed.columns:
        print("\n[Strand Distribution]")
        print(bed['strand'].value_counts())

    # Sequence length stats
    lengths = bed.iloc[:, 2] - bed.iloc[:, 1]
    print("\n[Sequence Length Stats]")
    print(f"Min: {lengths.min()}, Max: {lengths.max()}, Mean: {lengths.mean():.2f}, Median: {lengths.median()}")

    # Label summary from HDF5
    if os.path.exists(hdf5_file):
        print(f"\n--- Loading HDF5 file: {hdf5_file} ---")
        with h5py.File(hdf5_file, 'r') as h5:
            labels = h5['labels'][:]
            print("\n[Label Info]")
            print(f"Shape: {labels.shape}")
            if labels.ndim == 2:
                nonzero_per_sample = (labels > 0).sum(axis=1)
                print(f"Average non-zero labels per sample: {nonzero_per_sample.mean():.2f}")
                print(f"Label frequency: {np.sum(labels, axis=0)}")
                if label_depth and labels.shape[1] != label_depth:
                    print(f"⚠️ Warning: Label depth mismatch (expected {label_depth}, found {labels.shape[1]})")
            elif labels.ndim == 1:
                if isinstance(labels[0], (np.ndarray, list)):
                    print("Multi-label classification (ragged 1D) detected.")
                    label_counts = {}
                    for item in labels:
                        for label in item:
                            label_counts[label] = label_counts.get(label, 0) + 1
                    print("Label counts (top 10):")
                    for label, count in sorted(label_counts.items())[:10]:
                        print(f"Label {label}: {count}")
                else:
                    print("Single-label classification detected.")
                    label_counts = pd.Series(labels).value_counts().sort_index()
                    print("Label counts:")
                    print(label_counts)
                    if label_depth and label_counts.index.max() >= label_depth:
                        print(f"⚠️ Warning: Label depth too small. Max label index = {label_counts.index.max()}")
    else:
        print("⚠️ No HDF5 file found.")


# Loop through all datasets
for task in tasks:
    dataset_summary(task, label_depth=64)
