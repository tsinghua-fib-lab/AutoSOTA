import os
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
from release_utils import ensure_dir, repo_path


def download_and_save(base_dir: str, tokenizer_name: str):
    local_data_dir = os.path.join(base_dir, "raw")
    local_tokenizer_dir = os.path.join(base_dir, "tokenizer")
    ensure_dir(base_dir)

    print(f">>> 1. Downloading WikiText-103 Dataset to {local_data_dir}...")
    
    try:
        # Use the raw WikiText-103 dataset variant for a straightforward download flow.
        dataset = load_dataset("wikitext", "wikitext-103-v1")
        dataset.save_to_disk(local_data_dir)
        print(f">>> Success! Dataset saved to: {local_data_dir}")
    except Exception as e:
        print(f">>> Error downloading WikiText: {e}")

    print(f"\n>>> 2. Downloading Tokenizer to {local_tokenizer_dir}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.save_pretrained(local_tokenizer_dir)
        print(f">>> Success! Tokenizer saved to: {local_tokenizer_dir}")
    except Exception as e:
        print(f">>> Error downloading Tokenizer: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=repo_path("data", "wikitext"))
    parser.add_argument("--tokenizer_name", type=str, default="EleutherAI/gpt-neox-20b")
    args = parser.parse_args()
    download_and_save(args.output_dir, args.tokenizer_name)
