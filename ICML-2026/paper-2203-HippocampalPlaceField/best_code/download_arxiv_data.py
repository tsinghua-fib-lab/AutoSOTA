import os
import argparse
from datasets import load_dataset, disable_progress_bar, DownloadConfig
from transformers import AutoTokenizer
from release_utils import ensure_dir, repo_path


def download_and_save(base_dir: str, tokenizer_name: str):
    train_dir = os.path.join(base_dir, "arxiv_train")
    val_dir = os.path.join(base_dir, "arxiv_validation")
    local_tokenizer_dir = os.path.join(base_dir, "tokenizer")
    ensure_dir(base_dir)
    
    # disable_progress_bar()

    print(f">>> 1. Downloading RedPajama-ArXiv Dataset...")
    try:
        # Keep the optional download config here for resumable downloads if needed.
        # dl_config = DownloadConfig(resume_download=True)

        dataset = load_dataset(
            "togethercomputer/RedPajama-Data-1T", 
            "arxiv", 
            num_proc=8, 
            trust_remote_code=True,
            # download_config=dl_config
        )
        
        full_train = dataset["train"]
        
        print(">>> Splitting dataset and reserving 2,000 examples for validation...")
        splits = full_train.train_test_split(test_size=2000, seed=6198)
        
        print(f">>> Saving training split to {train_dir}...")
        splits["train"].save_to_disk(train_dir)
        
        print(f">>> Saving validation split to {val_dir}...")
        splits["test"].save_to_disk(val_dir)
        
        print(">>> Dataset processing completed successfully.")
        
    except Exception as e:
        print(f">>> Error downloading or processing ArXiv: {e}")

    print(f"\n>>> 2. Downloading Tokenizer to {local_tokenizer_dir}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        tokenizer.save_pretrained(local_tokenizer_dir)
        print(f">>> Success! Tokenizer saved to: {local_tokenizer_dir}")
    except Exception as e:
        print(f">>> Error downloading Tokenizer: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=repo_path("data", "arxiv"))
    parser.add_argument("--tokenizer_name", type=str, default="allenai/olmo-1b")
    args = parser.parse_args()
    download_and_save(args.output_dir, args.tokenizer_name)
