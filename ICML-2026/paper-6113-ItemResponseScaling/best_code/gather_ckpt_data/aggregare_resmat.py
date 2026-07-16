import os
from huggingface_hub import snapshot_download
from collections import defaultdict
import pickle
import pandas as pd
import sys
sys.path.append("..")
from utils import visualize_response_matrix
from tqdm import tqdm
import random

def aggregate(file_list):
    assert len(file_list) == 2, "Expected exactly two files to aggregate."

    with open(file_list[0], "rb") as f1, open(file_list[1], "rb") as f2:
        df1 = pickle.load(f1)
        df2 = pickle.load(f2)
    merged = pd.concat([df1, df2])

    if not merged.index.duplicated().any():
        merged = merged.sort_index(key=lambda idx: idx.str.split("-").str[-1].astype(int))
        return merged

    keep_pos = []
    for label, group in merged.groupby(level=0, sort=False):
        # find all the integer positions of this label in merged.index
        positions = [i for i, lab in enumerate(merged.index) if lab == label]

        if len(positions) == 1:
            keep_pos.append(positions[0])
        else:
            # For each pos, count non-null entries
            counts = [merged.iloc[i].notna().sum() for i in positions]
            # find the max count
            max_count = max(counts)
            # get all positions with that max count
            best = [pos for pos, cnt in zip(positions, counts) if cnt == max_count]
            keep_pos.append(best[0])

    # now select by integer location, then sort
    result = merged.iloc[keep_pos]
    return result.sort_index(key=lambda idx: idx.str.split("-").str[-1].astype(int))

if __name__ == "__main__":
    cache_folder = snapshot_download(
        repo_id="stair-lab/irsl_response_matrix",
        repo_type="dataset",
    )

    save_dir = "../data/gather_ckpt_data/aggregate_matrix"
    os.makedirs(save_dir, exist_ok=True)
    output_dir = "../result/gather_ckpt_data/aggregate_matrix"
    os.makedirs(output_dir, exist_ok=True)

    all_files = os.listdir(cache_folder)
    grouped_files = defaultdict(list)

    for f in all_files:
        if f.startswith("results_") and f.endswith(".pkl"):
            parts = f.split("_")
            group_name = "_".join(parts[:2])  # e.g., results_pythia-6.9b
            grouped_files[group_name].append(f"{cache_folder}/{f}")

    for group, files in tqdm(grouped_files.items()):
        output_path = f"{save_dir}/{group}.pkl"
        if len(files) == 1:
            with open(files[0], "rb") as f:
                data = pickle.load(f)
            with open(output_path, "wb") as f:
                pickle.dump(data, f)
            visualize_response_matrix(data, data, f"{output_dir}/response_matrix_{group}.png")
        else:
            merged_data = aggregate(files)
            with open(output_path, "wb") as f:
                pickle.dump(merged_data, f)
            visualize_response_matrix(merged_data, merged_data, f"{output_dir}/response_matrix_{group}.png")
