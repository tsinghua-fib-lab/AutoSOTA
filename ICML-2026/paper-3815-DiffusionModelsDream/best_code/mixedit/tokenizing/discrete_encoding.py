import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(os.path.abspath('../'))
try:
    from maskedit.datasets.dataset import dataset as FmtDataset
except ImportError:
    print("WARNING: Could not import 'maskedit.datasets.dataset'. Ensure your python path is correct.")
    # creating a dummy class just so the script doesn't crash if checking syntax
    class FmtDataset:
        def __init__(self, data, labels): self.norm = data; self.data = data; self.iblank = 0
        def blank_mod(self, ind): return ind

# ==========================================
# Core Logic Functions
# ==========================================

def compress_to_decimal(binary_dataset):
    """
    Compresses a (B, D) binary dataset by:
    1. Finding and removing columns where all values are 1.
    2. Converting the remaining (B, D_filtered) binary rows to decimal.
    """
    if not isinstance(binary_dataset, np.ndarray):
        binary_dataset = np.array(binary_dataset)
        
    B, original_D = binary_dataset.shape
    
    # 1. Find columns to remove (where all values are 1)
    remove_mask = binary_dataset.all(axis=0)
    all_ones_indices = np.where(remove_mask)[0]
    
    # 2. Create the filtered dataset (keeping columns that are NOT all 1s)
    keep_mask = ~remove_mask
    filtered_dataset = binary_dataset[:, keep_mask]
    
    D_filtered = filtered_dataset.shape[1]
    
    # 3. Convert to decimal
    if D_filtered == 0:
        decimals_bx1 = np.zeros((B, 1), dtype=int)
    else:
        powers = np.arange(D_filtered)[::-1]
        multipliers = 2 ** powers
        decimals = filtered_dataset @ multipliers
        decimals_bx1 = decimals[:, np.newaxis]
        
    return decimals_bx1, all_ones_indices, original_D

def decompress_from_decimal(decimals_bx1, all_ones_indices, original_D):
    """
    Decompresses a (B, 1) decimal array back into a (B, D) binary array.
    """
    B = decimals_bx1.shape[0]
    D_filtered = original_D - len(all_ones_indices)
    
    reconstructed = np.zeros((B, original_D), dtype=int)
    
    # Step 1: Add back the "all 1s" columns
    reconstructed[:, all_ones_indices] = 1
    
    # Step 2: Add back the filtered data
    if D_filtered > 0:
        decimals = decimals_bx1.ravel() 
        m = 2 ** np.arange(D_filtered)[::-1]
        
        # Broadcasting to get binary
        filtered_binary = (decimals[:, np.newaxis] // m) % 2
        
        keep_mask = np.ones(original_D, dtype=bool)
        keep_mask[all_ones_indices] = False
        
        reconstructed[:, keep_mask] = filtered_binary
    
    return reconstructed

def compress_decimal_representation(dec_data):
    """
    Tokenizes the unique decimal values to a smaller range (0..N).
    Returns the tokenized data and the dictionary map.
    """
    h, c = np.unique(dec_data, return_counts=True)
    # Map decimal_val -> token_id
    tokenizer = {k: v for k, v in zip(h, range(len(h)))}
    # Apply map
    tokenized_data = np.array([tokenizer[v[0]] for v in dec_data]).reshape((-1, 1))
    return tokenized_data, tokenizer

def convert_samples_to_binary(samples, tokenizer, all_ones_indices, original_D):
    """
    Full pipeline reverse: Tokenized -> Decimal -> Binary
    """
    # Reverse the tokenizer: token_id -> decimal_val
    reverse_tokenizer = {int(v): int(k) for k, v in tokenizer.items()}
    
    # Convert samples back to decimal
    dec_gen = np.array([reverse_tokenizer[int(v)] for v in samples]).reshape((-1, 1))
    
    # Decompress decimal to binary
    return decompress_from_decimal(dec_gen, all_ones_indices, original_D)

# ==========================================
# Main Execution
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Binary Encoding/Decoding Pipeline")
    
    # Input args
    parser.add_argument('--data_dir', type=str, default="./training_data/maskedit/data", help="Base directory for data")
    parser.add_argument('--csv_file', type=str, default="train_set.csv", help="Input CSV filename")
    parser.add_argument('--pkl_file', type=str, default="indices_01.pkl", help="Input Pickle filename")
    
    # Output args (arguments for np.save)
    parser.add_argument('--out_dir', type=str, default=".", help="Directory to save outputs")
    parser.add_argument('--save_indices', type=str, default="all_ones_indices.npy")
    parser.add_argument('--save_dims', type=str, default="original_D.npy")
    parser.add_argument('--save_tokens', type=str, default="tokenized_uav.npy")
    parser.add_argument('--save_obs', type=str, default="uav_observations.npy")
    parser.add_argument('--save_tokenizer', type=str, default="tokenizer.npy")
    
    # Flags
    parser.add_argument('--plot', action='store_true', help="Show probability plot")
    parser.add_argument('--gpu', type=str, default='0', help="CUDA_VISIBLE_DEVICES ID")

    args = parser.parse_args()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # --- 1. Load Data ---
    pth = Path(args.data_dir)
    print(f"Loading data from {pth}...")
    
    df = pd.read_csv(pth / args.csv_file)
    lablist = list(df.keys())
    data_raw = np.expand_dims(df.to_numpy(), axis=-1)

    with open(pth / args.pkl_file, "rb") as f:
        indices = pickle.load(f)["indices"]

    # --- 2. Dataset Wrappers & Masking ---
    # Using the imported class 'dataset' as 'FmtDataset' to avoid variable name collision later
    full_data = FmtDataset(data_raw, labels=lablist)
    indices = full_data.blank_mod(indices)
    indices = indices[:-1] # Remove topmask

    # Extract sequences
    data_sequences = full_data.norm[:, full_data.iblank:].squeeze(-1)

    # This is the Ground Truth Binary Matrix
    binary_matrix = np.array(data_sequences)
    print(f"Data shape: {binary_matrix.shape}")

    # --- 3. Optional Plotting ---
    if args.plot:
        sl = binary_matrix.shape[1]
        probs_of_one = binary_matrix.mean(axis=0)
        plt.figure(figsize=(10, 6))
        plt.bar(np.arange(sl), probs_of_one, label="Prob(1)")
        plt.title("Probability of '1' at Each Position")
        plt.xlabel("Sequence Position")
        plt.ylabel("Probability")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

    # --- 4. Compression Pipeline ---
    print("Compressing to decimal...")
    # Step A: Binary -> Decimal (handling all-ones columns)
    decimals_bx1, all_ones_indices, original_D = compress_to_decimal(binary_matrix)
    
    # Step B: Decimal -> Tokenized
    print("Tokenizing decimals...")
    decimals_tokenized, tokenizer = compress_decimal_representation(decimals_bx1)

    # --- 5. Saving Artifacts ---
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving artifacts to {out_path}...")
    np.save(out_path / args.save_indices, all_ones_indices)
    np.save(out_path / args.save_dims, original_D)
    np.save(out_path / args.save_tokens, decimals_tokenized)
    np.save(out_path / args.save_tokenizer, tokenizer)

    # Save Observations (Metadata/Auxiliary data)
    inds = [
        '/res_metrics/avg_drag', '/res_metrics/avg_l', '/res_metrics/comp_weights/total',
        '/res_metrics/cruise_aoa', '/res_metrics/cruise_fom', '/res_metrics/cruise_l', '/res_metrics/soc'
    ]
    # Indices extracted from snippet [72,73,76,78,79,80,81]
    obs_cols = [72, 73, 76, 78, 79, 80, 81]
    
    dic = {
        "norm_data": np.asarray(full_data.norm[:, obs_cols]),
        "inds": inds,
        "unnorm_data": np.asarray(full_data.data[:, obs_cols])
    }
    np.save(out_path / args.save_obs, dic)

    # --- 6. Decompression / Verification ---
    print("\n--- Verifying Reconstruction ---")
    
    recon_data = convert_samples_to_binary(
        decimals_tokenized, 
        tokenizer, 
        all_ones_indices, 
        original_D
    )

    print(f"Original Shape: {binary_matrix.shape}")
    print(f"Recon Shape:    {recon_data.shape}")
    
    # Verify exact match
    if np.array_equal(binary_matrix, recon_data):
        print("SUCCESS: recon_data exactly matches dataset (data_sequences).")
    else:
        print("FAILURE: Mismatch detected between original and reconstructed data.")
        # Debugging info
        diff = np.sum(binary_matrix != recon_data)
        print(f"Number of mismatched elements: {diff}")

if __name__ == "__main__":
    main()