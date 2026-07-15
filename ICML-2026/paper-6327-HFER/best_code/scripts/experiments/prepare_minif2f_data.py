import os
import shutil
import random
import argparse

def main():
    parser = argparse.ArgumentParser(description="Prepare MiniF2F data for experiment (Sampling).")
    parser.add_argument("--source-dir", type=str, default="data/proofs_minif2f", help="Source directory")
    parser.add_argument("--dest-dir", type=str, default="data/minif2f_moe_prepared", help="Destination directory")
    parser.add_argument("--target", type=int, default=50, help="Number of samples per class")
    args = parser.parse_args()

    # Define sources
    # Try 'valid_ground_truth' first for high quality, else 'valid'
    valid_src = os.path.join(args.source_dir, "valid_ground_truth")
    if not os.path.exists(valid_src) or len(os.listdir(valid_src)) < args.target:
        print(f"Ground truth invalid or insufficient, trying 'valid'...")
        valid_src = os.path.join(args.source_dir, "valid")
    
    invalid_src = os.path.join(args.source_dir, "invalid")

    # Define destinations
    dest_valid = os.path.join(args.dest_dir, "valid")
    dest_invalid = os.path.join(args.dest_dir, "invalid")

    os.makedirs(dest_valid, exist_ok=True)
    os.makedirs(dest_invalid, exist_ok=True)

    # Collect files
    valid_files = [f for f in os.listdir(valid_src) if f.endswith(".lean") or f.endswith(".txt")]
    invalid_files = [f for f in os.listdir(invalid_src) if f.endswith(".lean") or f.endswith(".txt")]

    print(f"Found {len(valid_files)} valid and {len(invalid_files)} invalid files.")

    # Sample
    if len(valid_files) > args.target:
        valid_sample = random.sample(valid_files, args.target)
    else:
        valid_sample = valid_files
    
    if len(invalid_files) > args.target:
        invalid_sample = random.sample(invalid_files, args.target)
    else:
        invalid_sample = invalid_files

    print(f"Selected {len(valid_sample)} Valid, {len(invalid_sample)} Invalid.")

    # Copy
    for f in valid_sample:
        shutil.copy(os.path.join(valid_src, f), os.path.join(dest_valid, f))
        
    for f in invalid_sample:
        shutil.copy(os.path.join(invalid_src, f), os.path.join(dest_invalid, f))

    print("Data preparation complete.")

if __name__ == "__main__":
    main()
