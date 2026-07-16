import argparse
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
BASE_DIR = PROJECT_ROOT / "data"
sys.path.append(str(PROJECT_ROOT.parent))
from utils import calibrate_2pl

DRY_RUN_N_COLS = 128
DRY_RUN_N_ROWS = 64
CUDA = 5

parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--loss-kind", type=str, default="beta", choices=["beta", "binary"])
parser.add_argument("--data-root", type=Path, default=BASE_DIR)
args = parser.parse_args()

data_root = args.data_root
data_root.mkdir(parents=True, exist_ok=True)
input_path = data_root / ("3_prob_matrix.parquet" if args.loss_kind == "beta" else "3_binary_matrix.parquet")
output_path = data_root / (
    "4_prob_matrix_calibrated_2pl.parquet" if args.loss_kind == "beta" else "4_binary_matrix_calibrated_2pl.parquet"
)
resmat_df = pd.read_parquet(input_path)

if args.dry_run:
    resmat_df = resmat_df.iloc[:DRY_RUN_N_ROWS, :DRY_RUN_N_COLS]
    print(f"Dry run: limiting to {resmat_df.shape[0]} rows and {resmat_df.shape[1]} columns")

train_df = resmat_df[resmat_df.index.get_level_values("model_split") == "train"].copy()
test_df = resmat_df[resmat_df.index.get_level_values("model_split") == "test"].copy()
train_np = train_df.to_numpy(dtype=np.float32)
n_testtakers, n_items = train_np.shape
bench_names = train_df.columns.get_level_values("bench_name").map(
    lambda b: "mmlu" if b.startswith("mmlu") else b
)
unique_bench_names = sorted(bench_names.unique())

bench_thetas = {}
final_zs = np.full(n_items, np.nan, dtype=np.float32)
final_alphas = np.full(n_items, np.nan, dtype=np.float32)
for bench in tqdm(unique_bench_names, desc="benches"):
    bench_mask = bench_names == bench    
    bench_resmat = train_np[:, bench_mask]

    calibrate_result = calibrate_2pl(
        resmat=torch.tensor(bench_resmat),
        device=f'cuda:{CUDA}',
        loss_kind=args.loss_kind,
    )
    z_optimized = calibrate_result['z']
    alpha_optimized = calibrate_result['alpha']
    theta_optimized = calibrate_result['theta']
    bench_thetas[bench] = theta_optimized
    final_zs[bench_mask] = z_optimized
    final_alphas[bench_mask] = alpha_optimized
    
    col_mean = np.nanmean(bench_resmat, axis=0)
    row_mean = np.nanmean(bench_resmat, axis=1)
    col_rho, _ = spearmanr(z_optimized, col_mean)
    row_rho, _ = spearmanr(theta_optimized, row_mean)
    print(
        f"Bench: {bench}"
        f"\nz corr with col mean = {col_rho:.6f}"
        f"\ntheta corr with row mean = {row_rho:.6f}"
    )

assert not np.isnan(final_zs).any() and not np.isnan(final_alphas).any()
resmat_df_new = resmat_df.copy()
col_levels = [resmat_df.columns.get_level_values(i) for i in range(resmat_df.columns.nlevels)]
resmat_df_new.columns = pd.MultiIndex.from_arrays(
    [*col_levels, final_zs, final_alphas],
    names=[*(resmat_df.columns.names), "difficulty", "discrimination"],
)

theta_columns = [f"ability_{bench}" for bench in unique_bench_names]
theta_df = pd.DataFrame(np.nan, index=resmat_df_new.index, columns=theta_columns, dtype=float)
theta_values = np.column_stack([bench_thetas[b] for b in unique_bench_names])
theta_df.loc[train_df.index, theta_columns] = theta_values

org_index_cols = list(resmat_df.index.names)
combined = pd.concat([resmat_df_new, theta_df], axis=1)
combined = combined.reset_index()
new_index_cols = org_index_cols + theta_columns
combined = combined.set_index(new_index_cols)
combined.columns = pd.MultiIndex.from_tuples(combined.columns, names=resmat_df_new.columns.names)

print(f"Shape: {combined.shape}")
idx = combined.index
print("\nRow index names:", idx.names)
print("Row index sample (first 5):", list(islice(idx, 5)))
cols = combined.columns
print("\nColumn index names:", cols.names)
print("Column index sample (first 5):", list(islice(cols, 5)))

if not args.dry_run:
    combined.to_parquet(output_path)
