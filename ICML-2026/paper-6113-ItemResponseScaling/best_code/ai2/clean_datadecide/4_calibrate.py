import argparse
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from huggingface_hub import snapshot_download

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR.parent.parent))
from utils import calibrate

REPO_ID = "yuhengtu/irsl_datadecide"
SNAPSHOT_DIR = Path(snapshot_download(repo_id=REPO_ID, repo_type="dataset"))
PROB_PATH = SNAPSHOT_DIR / "3_prob_matrix.parquet"
BINARY_PATH = SNAPSHOT_DIR / "3_binary_matrix.parquet"
OUTPUT_PROB_PATH = BASE_DIR / "4_prob_matrix_with_difficulty.parquet"
OUTPUT_BINARY_PATH = BASE_DIR / "4_binary_matrix_with_difficulty.parquet"
BATCH_SIZE = 1024
CUDAS = [4, 5, 6, 7]

parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true", help="Limit calibration to a subset of columns and skip writing output.")
parser.add_argument("--dry-run-cols", type=int, default=5000, help="Number of columns to keep in dry-run mode.")
parser.add_argument("--loss-kind", type=str, default="beta", choices=["beta", "binary"], help="Loss used in calibrate().")
args = parser.parse_args()

input_path = PROB_PATH if args.loss_kind == "beta" else BINARY_PATH
output_path = OUTPUT_PROB_PATH if args.loss_kind == "beta" else OUTPUT_BINARY_PATH

resmat_df = pd.read_parquet(input_path)

if args.dry_run:
    original_cols = len(resmat_df.columns)
    selected_columns = resmat_df.columns[: args.dry_run_cols]
    resmat_df = resmat_df[selected_columns]
    print(f"Dry run enabled: limiting to {len(selected_columns)} columns (of {original_cols}) and skipping write.")

train_df = resmat_df[resmat_df.index.get_level_values("model_split") == "train"].copy()
test_df = resmat_df[resmat_df.index.get_level_values("model_split") == "test"].copy()

for name, df in [("train", train_df), ("test", test_df)]:
    total_cells = df.size
    nan_count = int(np.isnan(df.to_numpy()).sum())
    nan_pct = (nan_count / total_cells * 100) if total_cells else 0.0
    print(f"{name} rows: {len(df)}, questions: {df.shape[1]}")
    print(f"{name} NaN: {nan_count} cells ({nan_pct:.2f}%)")

def _calibrate_chunk(chunk_array: np.ndarray, device_str: str) -> np.ndarray:
    torch.cuda.set_device(int(device_str.split(":")[-1]))
    probmat_chunk = torch.tensor(chunk_array, dtype=torch.float32, device=device_str)
    return calibrate(probmat_chunk, device=device_str, batch_size=BATCH_SIZE, loss_kind=args.loss_kind)

train_np = train_df.to_numpy(dtype=np.float32)
n_items = train_np.shape[1]
print(f"Using {len(CUDAS)} CUDA devices {CUDAS}; splitting {n_items} columns across devices.")
col_indices = np.array_split(np.arange(n_items), len(CUDAS))
chunks = [(cols[0], cols[-1] + 1) for cols in col_indices if len(cols) > 0]

z_optimized = np.empty(n_items, dtype=np.float32)
with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
    futures = []
    for device_idx, (start, end) in enumerate(chunks):
        device_str = f"cuda:{CUDAS[device_idx]}"
        fut = executor.submit(_calibrate_chunk, train_np[:, start:end], device_str)
        futures.append((start, end, fut))
    for start, end, fut in futures:
        z_chunk = fut.result()
        z_optimized[start:end] = z_chunk

mean_train = np.nanmean(train_df.to_numpy(), axis=0)
mean_test = np.nanmean(test_df.to_numpy(), axis=0)
rho_train, _ = spearmanr(z_optimized, mean_train)
rho_test, _ = spearmanr(z_optimized, mean_test)
print(
    f"\ncorr with np.nanmean(train_df[questions], axis=0) = {rho_train:.6f}"
    f"\ncorr with np.nanmean(test_df[questions], axis=0) = {rho_test:.6f}\n"
)

resmat_df_with_difficulty = resmat_df.copy()
resmat_df_with_difficulty.columns = pd.MultiIndex.from_arrays(
    [resmat_df.columns, z_optimized],
    names=["question_id", "difficulty"],
)
if not args.dry_run:
    resmat_df_with_difficulty.to_parquet(output_path)
