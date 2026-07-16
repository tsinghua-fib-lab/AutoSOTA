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
from utils import calibrate_1pl_z, calibrate_1pl_theta

DRY_RUN_N_COLS = 128
DRY_RUN_BATCH_SIZE = 32
NON_DRY_RUN_BATCH_SIZE = 4096 # 4096
CUDAS = [5]

parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--loss-kind", type=str, default="beta", choices=["beta", "binary"])
parser.add_argument("--data-root", type=Path, default=BASE_DIR)
args = parser.parse_args()

data_root = args.data_root
data_root.mkdir(parents=True, exist_ok=True)
input_path = data_root / ("3_prob_matrix.parquet" if args.loss_kind == "beta" else "3_binary_matrix.parquet")
output_path = data_root / (
    "4_prob_matrix_calibrated.parquet" if args.loss_kind == "beta" else "4_binary_matrix_calibrated.parquet"
)
batch_size = DRY_RUN_BATCH_SIZE if args.dry_run else NON_DRY_RUN_BATCH_SIZE

resmat_df = pd.read_parquet(input_path)

if args.dry_run:
    selected_columns = resmat_df.columns[: DRY_RUN_N_COLS]
    resmat_df = resmat_df[selected_columns]
    print(f"Dry run: limiting to {len(selected_columns)} columns")

train_df = resmat_df[resmat_df.index.get_level_values("model_split") == "train"].copy()
test_df = resmat_df[resmat_df.index.get_level_values("model_split") == "test"].copy()
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# fit z
print("# fit z")
def _calibrate_z_chunk(chunk_array: np.ndarray, device_str: str) -> np.ndarray:
    torch.cuda.set_device(int(device_str.split(":")[-1]))
    mat_chunk = torch.tensor(chunk_array, dtype=torch.float32, device=device_str)
    return calibrate_1pl_z(mat_chunk, device=device_str, batch_size=batch_size, loss_kind=args.loss_kind)

train_np = train_df.to_numpy(dtype=np.float32)
n_items = train_np.shape[1]
print(f"Using {len(CUDAS)} CUDA devices {CUDAS}; splitting {n_items} columns across devices.")
col_indices = np.array_split(np.arange(n_items), len(CUDAS))
chunks = [(cols[0], cols[-1] + 1) for cols in col_indices]

z_optimized = np.empty(n_items, dtype=np.float32)
with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
    futures = []
    for device_idx, (start, end) in enumerate(chunks):
        device_str = f"cuda:{CUDAS[device_idx]}"
        fut = executor.submit(_calibrate_z_chunk, train_np[:, start:end], device_str)
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
    f"\ncorr with np.nanmean(test_df[questions], axis=0) = {rho_test:.6f}"
)

# fit theta
print("# fit theta")
theta_device = f"cuda:{CUDAS[0]}"
bench_names = train_df.columns.get_level_values("bench_name").map(
    lambda b: "mmlu" if b.startswith("mmlu") else b
)
unique_bench_names = sorted(bench_names.unique())

print(f"Processing {len(unique_bench_names)} benches: {unique_bench_names}")
bench_thetas = {}
for bench in tqdm(unique_bench_names, desc="benches"):
    bench_mask = bench_names == bench    
    bench_theta = calibrate_1pl_theta(
        resmat=torch.tensor(train_np[:, bench_mask], dtype=torch.float32),
        device=theta_device,
        zs=torch.tensor(z_optimized[bench_mask], dtype=torch.float32),
        loss_kind=args.loss_kind,
    )
    bench_thetas[bench] = bench_theta
    
# save data
print("# save data")
resmat_df_new = resmat_df.copy()
col_levels = [resmat_df.columns.get_level_values(i) for i in range(resmat_df.columns.nlevels)]
resmat_df_new.columns = pd.MultiIndex.from_arrays(
    [*col_levels, z_optimized],
    names=[*(resmat_df.columns.names), "difficulty"],
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
