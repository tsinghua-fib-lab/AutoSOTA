"""Download and cache the SST-2 dataset locally."""
import os
import argparse
from datasets import load_dataset
from release_utils import repo_path


def download_sst2(output_dir: str):
    """Download the SST-2 dataset and save its splits to disk."""
    
    print("Loading SST-2 dataset from HuggingFace...")
    dataset = load_dataset("glue", "sst2")
    
    print(f"Dataset splits:")
    print(f"  Train: {len(dataset['train'])} examples")
    print(f"  Validation: {len(dataset['validation'])} examples")
    print(f"  Test: {len(dataset['test'])} examples")
    
    os.makedirs(output_dir, exist_ok=True)
    
    dataset["train"].save_to_disk(os.path.join(output_dir, "train"))
    dataset["validation"].save_to_disk(os.path.join(output_dir, "validation"))
    dataset["test"].save_to_disk(os.path.join(output_dir, "test"))
    
    print(f"\n✓ SST-2 dataset saved to: {output_dir}")
    
    train_labels = dataset["train"]["label"]
    n_neg = sum(1 for l in train_labels if l == 0)
    n_pos = sum(1 for l in train_labels if l == 1)
    print(f"\nLabel distribution in training set:")
    print(f"  Negative (0): {n_neg} ({100*n_neg/len(train_labels):.1f}%)")
    print(f"  Positive (1): {n_pos} ({100*n_pos/len(train_labels):.1f}%)")
    
    print(f"\nSample examples:")
    for i in range(3):
        example = dataset["train"][i]
        label_str = "Positive" if example["label"] == 1 else "Negative"
        print(f"  [{label_str}] {example['sentence'][:100]}...")
    
    return output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=repo_path("data", "sst2"))
    args = parser.parse_args()
    
    download_sst2(args.output_dir)
