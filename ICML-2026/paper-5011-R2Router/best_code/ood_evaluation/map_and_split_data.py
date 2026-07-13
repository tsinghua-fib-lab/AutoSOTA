"""
Map SPROUT dataset to existing CSV files and create category-based splits.

This script:
1. Loads the SPROUT dataset with category information
2. Matches it with existing LLM CSV files
3. Splits data by category for OOD evaluation
4. Splits embeddings correspondingly
"""
import os
import pickle
import pandas as pd
import numpy as np
from datasets import load_dataset
from collections import defaultdict

print("=" * 80)
print("LOADING DATA")
print("=" * 80)

# Load SPROUT dataset
print("\nLoading SPROUT dataset...")
cache_dir = "./cache"
sprout = load_dataset("CARROT-LLM-Routing/SPROUT", cache_dir=cache_dir)
train_data = sprout['train']

# Load embeddings
print("Loading embeddings...")
with open("data/prompt_embeddings.pkl", "rb") as f:
    emb_data = pickle.load(f)
    if isinstance(emb_data, dict):
        embeddings = emb_data["embeddings"]
    else:
        embeddings = emb_data

print(f"  Embeddings shape: {embeddings.shape}")
print(f"  SPROUT train size: {len(train_data)}")

# Verify they match
assert len(embeddings) == len(train_data), \
    f"Size mismatch! Embeddings: {len(embeddings)}, SPROUT: {len(train_data)}"

print("\n✓ Embeddings and SPROUT dataset sizes match!")

# Extract category information
categories = train_data['dataset']
category_list = sorted(set(categories))

print(f"\n✓ Found {len(category_list)} unique categories")

# Create category indices mapping
print("\n" + "=" * 80)
print("CREATING CATEGORY INDICES")
print("=" * 80)

category_indices = defaultdict(list)
for idx, cat in enumerate(categories):
    category_indices[cat].append(idx)

# Print category sizes
print("\nCategory sizes:")
for cat in category_list:
    print(f"  {cat}: {len(category_indices[cat])} queries")

# Create output directory
os.makedirs("./ood_evaluation/category_splits", exist_ok=True)

# Save category indices
print("\n" + "=" * 80)
print("SAVING CATEGORY INDICES")
print("=" * 80)

category_indices_file = "./ood_evaluation/category_splits/category_indices.pkl"
with open(category_indices_file, 'wb') as f:
    pickle.dump(dict(category_indices), f)
print(f"✓ Saved category indices to {category_indices_file}")

# Split embeddings by category
print("\n" + "=" * 80)
print("SPLITTING EMBEDDINGS BY CATEGORY")
print("=" * 80)

os.makedirs("./ood_evaluation/category_splits/embeddings", exist_ok=True)

category_embeddings = {}
for cat in category_list:
    indices = category_indices[cat]
    cat_emb = embeddings[indices]
    category_embeddings[cat] = cat_emb

    # Save individual category embeddings
    cat_name_safe = cat.replace('/', '_')
    emb_file = f"./ood_evaluation/category_splits/embeddings/{cat_name_safe}.pkl"
    with open(emb_file, 'wb') as f:
        pickle.dump(cat_emb, f)

    print(f"  {cat}: saved {cat_emb.shape[0]} embeddings")

print("\n✓ Split embeddings by category")

# Now split CSV files for each LLM
print("\n" + "=" * 80)
print("SPLITTING CSV FILES BY CATEGORY")
print("=" * 80)

# Find all CSV files in data directory
csv_files = [f for f in os.listdir("data") if f.endswith('.csv')]
print(f"\nFound {len(csv_files)} CSV files:")
for f in csv_files:
    print(f"  - {f}")

os.makedirs("./ood_evaluation/category_splits/csv", exist_ok=True)

# Split each CSV file
for csv_file in csv_files:
    csv_path = os.path.join("data", csv_file)
    print(f"\nProcessing {csv_file}...")

    df = pd.read_csv(csv_path)

    # Verify size matches
    if len(df) != len(embeddings):
        print(f"  ⚠️  Warning: {csv_file} has {len(df)} rows, expected {len(embeddings)}")
        continue

    # Split by category
    for cat in category_list:
        indices = category_indices[cat]
        cat_df = df.iloc[indices].reset_index(drop=True)

        # Save category-specific CSV
        cat_name_safe = cat.replace('/', '_')
        model_name = csv_file.replace('.csv', '')
        cat_csv_dir = f"./ood_evaluation/category_splits/csv/{cat_name_safe}"
        os.makedirs(cat_csv_dir, exist_ok=True)

        cat_csv_file = f"{cat_csv_dir}/{model_name}.csv"
        cat_df.to_csv(cat_csv_file, index=False)

    print(f"  ✓ Split {csv_file} into {len(category_list)} categories")

print("\n" + "=" * 80)
print("CREATING OOD TRAIN/TEST SPLITS")
print("=" * 80)

# For each category, create train/test split where:
# - Test = this category
# - Train = all other categories

ood_splits = {}

for test_cat in category_list:
    # Get test indices (this category)
    test_indices = category_indices[test_cat]

    # Get train indices (all other categories)
    train_indices = []
    for cat in category_list:
        if cat != test_cat:
            train_indices.extend(category_indices[cat])

    train_indices = sorted(train_indices)
    test_indices = sorted(test_indices)

    ood_splits[test_cat] = {
        'train_idx': np.array(train_indices),
        'test_idx': np.array(test_indices),
        'train_size': len(train_indices),
        'test_size': len(test_indices)
    }

    print(f"\n{test_cat}:")
    print(f"  Train: {len(train_indices)} queries (from {len(category_list)-1} categories)")
    print(f"  Test: {len(test_indices)} queries (OOD)")

# Save OOD splits
ood_splits_file = "./ood_evaluation/category_splits/ood_splits.pkl"
with open(ood_splits_file, 'wb') as f:
    pickle.dump(ood_splits, f)
print(f"\n✓ Saved OOD splits to {ood_splits_file}")

# Create a summary CSV
print("\n" + "=" * 80)
print("CREATING SUMMARY")
print("=" * 80)

summary_data = []
for cat in category_list:
    split = ood_splits[cat]
    summary_data.append({
        'test_category': cat,
        'train_size': split['train_size'],
        'test_size': split['test_size'],
        'test_percentage': 100 * split['test_size'] / len(embeddings)
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv("./ood_evaluation/category_splits/ood_splits_summary.csv", index=False)
print("\n✓ Saved OOD splits summary to './ood_evaluation/category_splits/ood_splits_summary.csv'")

print("\n" + "=" * 80)
print("COMPLETE!")
print("=" * 80)
print("\nData organization:")
print("  - Category indices: ./ood_evaluation/category_splits/category_indices.pkl")
print("  - OOD splits: ./ood_evaluation/category_splits/ood_splits.pkl")
print("  - Category embeddings: ./ood_evaluation/category_splits/embeddings/")
print("  - Category CSVs: ./ood_evaluation/category_splits/csv/")
print("\nYou can now run OOD evaluation by training on all categories except one")
print("and testing on the held-out category!")
