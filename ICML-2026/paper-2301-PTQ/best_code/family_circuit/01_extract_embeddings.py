import sys
import os
import argparse
import numpy as np
import polars as pl
import torch
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append('../training')
sys.path.append('..')
from circuit_utils.esm_activation import ESMInference
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    
ESM_WEIGHTS = os.environ.get("ESM_WEIGHTS")
PARQUET_PATH = os.environ.get("PARQUET_PATH")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "families")
MASTER_NAME = os.environ.get("MASTER_NPZ_NAME", "all_acts.npz")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 16))
MIN_POSITIVES = int(os.environ.get("MIN_POSITIVES", 2))

def load_data(limit=None, target=None):
    print(f"Loading parquet from {PARQUET_PATH}...")
    df = pl.read_parquet(PARQUET_PATH)
    
    # Expand InterPro IDs
    exploded = df.select(
        pl.col("Sequence"),
        pl.col("InterPro").str.strip_chars(";").str.split(";")
    ).explode("InterPro").filter(pl.col("InterPro").str.len_chars() > 0)

    # Filter by size
    counts = exploded.group_by("InterPro").len()
    valid_families = counts.filter(pl.col("len") >= MIN_POSITIVES)["InterPro"].to_list()
    if target:
        if target not in valid_families:
            print(f"Warning: Target {target} has insufficient data.")
        valid_families = [target]
    elif limit:
        top_families = counts.filter(pl.col("InterPro").is_in(valid_families)) \
                             .sort("len", descending=True).head(limit)["InterPro"].to_list()
        valid_families = top_families

    # Keep only relevant rows and find negative class examples
    filtered_df = exploded.filter(pl.col("InterPro").is_in(valid_families))
    if target:
        pos_seqs = filtered_df["Sequence"].unique().to_list()
        neg_df = df.filter(~pl.col("Sequence").is_in(pos_seqs)).sample(n=len(pos_seqs)*4, seed=42)
        all_seqs = pos_seqs + neg_df["Sequence"].to_list()
        all_ids = [target] * len(pos_seqs) + ["NEGATIVE"] * len(neg_df)
        return all_seqs, all_ids
    
    return filtered_df["Sequence"].to_list(), filtered_df["InterPro"].to_list()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--hidden_size", type=int, default=320)
    parser.add_argument("--source", type=str, default="mlp_output", choices=["layer_output", "mlp_output"], help="Source of embeddings: 'layer_output' (Normed) or 'mlp_output' (Raw FC2).")
    # Dummy args
    parser.add_argument("--root_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    args, _ = parser.parse_known_args()

    Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)
    save_path = Path(OUTPUT_DIR) / (f"{args.target}.npz" if args.target else MASTER_NAME)

    if save_path.exists() and not args.overwrite:
        print(f"File {save_path} exists. Skipping.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seqs, ids = load_data(limit=args.limit, target=args.target)
    inference = ESMInference(device, esm_weights_path=ESM_WEIGHTS, num_layers=args.layers, d_model=args.hidden_size)
        
    print(f"Extracting embeddings ({args.source}) for {len(seqs)} sequences...")
    embeddings = []
    for i in tqdm(range(0, len(seqs), BATCH_SIZE)):
        batch_seqs = seqs[i : i + BATCH_SIZE]
        batch_emb = inference.get_embeddings(batch_seqs, source=args.source, mean_pool=True)
        embeddings.append(batch_emb)
    full_embeddings = np.concatenate(embeddings, axis=0)
    np.savez_compressed(save_path, embeddings=full_embeddings, sequences=np.array(seqs), interpro_ids=np.array(ids), source=args.source)
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    main()