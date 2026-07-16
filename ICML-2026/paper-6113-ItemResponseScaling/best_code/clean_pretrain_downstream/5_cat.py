import argparse
from pathlib import Path
import sys
import os
import torch
import numpy as np
import pandas as pd
from itertools import islice
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from tqdm import tqdm
from tueplots import bundles
bundles.icml2024()
from huggingface_hub import snapshot_download
import pickle

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT.parent))
from utils import cat_beta_1pl, cat_binary_1pl, cat_beta_2pl, cat_binary_2pl

N_MAX_PLOT = 50
DRY_RUN_N_ROWS = 64

parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--loss-kind", type=str, default="beta", choices=["beta", "binary"])
parser.add_argument("--irt-model", type=str, default="1pl", choices=["1pl", "2pl"])
parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results")
args = parser.parse_args()

n_cpus = int((os.cpu_count()) * 0.8)
print(f"Using {n_cpus} CPUs for parallel processing.")

irt_suffix = "_2pl" if args.irt_model == "2pl" else ""
plots_dir = args.results_root / f"5_cat{irt_suffix}"
data_root = args.data_root
data_root.mkdir(parents=True, exist_ok=True)

if args.loss_kind == "beta":
    input_path = data_root / f"4_prob_matrix_calibrated{irt_suffix}.parquet"
    matrix_output_path = data_root / f"5_prob_matrix_cated{irt_suffix}.parquet"
    cat_fn = cat_beta_2pl if args.irt_model == "2pl" else cat_beta_1pl
    fig_prefix = "beta"
else:
    input_path = data_root / f"4_binary_matrix_calibrated{irt_suffix}.parquet"
    matrix_output_path = data_root / f"5_binary_matrix_cated{irt_suffix}.parquet"
    cat_fn = cat_binary_2pl if args.irt_model == "2pl" else cat_binary_1pl
    fig_prefix = "binary"
os.makedirs(plots_dir, exist_ok=True)

resmat_df = pd.read_parquet(input_path)
if args.dry_run:
    resmat_df = resmat_df.sample(n=DRY_RUN_N_ROWS, random_state=0)
    print(f"Dry run: limiting to {len(resmat_df)} rows")
test_df = resmat_df[resmat_df.index.get_level_values("model_split") == "test"].copy()
print(f"resmat_df.shape: {resmat_df.shape}, test_df.shape: {test_df.shape}")

test_ys = test_df.to_numpy(dtype=np.float32)
n_models = test_ys.shape[0]
bench_names = test_df.columns.get_level_values("bench_name").map(
    lambda b: "mmlu" if b.startswith("mmlu") else b
)
unique_bench_names = sorted(bench_names.unique())
zs = test_df.columns.get_level_values("difficulty").to_numpy(dtype=np.float32)
if args.irt_model == "2pl":
    alphas = test_df.columns.get_level_values("discrimination").to_numpy(dtype=np.float32)

bench_thetas = {}
for bench in tqdm(unique_bench_names):
    bench_mask = bench_names == bench
    bench_test_ys = torch.tensor(test_ys[:, bench_mask], dtype=torch.float32)
    bench_zs = torch.tensor(zs[bench_mask], dtype=torch.float32)
    
    if args.irt_model == "2pl":
        bench_alphas = torch.tensor(alphas[bench_mask], dtype=torch.float32)
        results = Parallel(n_jobs=n_cpus)(
            delayed(cat_fn)(bench_test_ys[i], bench_alphas, bench_zs, "cpu")
            for i in range(n_models)
        )
    else:
        results = Parallel(n_jobs=n_cpus)(
            delayed(cat_fn)(bench_test_ys[i], bench_zs, "cpu")
            for i in range(n_models)
        )
    fig_name = plots_dir / f"{fig_prefix}_{bench}_theta_convergence.png"
        
    thetass = np.asarray(results, dtype=np.float32)
    bench_thetas[bench] = thetass[:, -1]

    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        fig, axes = plt.subplots(
            nrows=N_MAX_PLOT,
            ncols=1,
            figsize=(6, 2 * N_MAX_PLOT),
            sharex=True,
        )
        plot_idxs = np.random.permutation(n_models)[:N_MAX_PLOT].tolist()
        for i_plot, ax in enumerate(axes[:N_MAX_PLOT]):
            model_idx = plot_idxs[i_plot]
            ax.plot(np.arange(thetass.shape[-1]), thetass[model_idx])
            ax.set_ylabel(r"$\theta$", fontsize=12)
            ax.tick_params(axis="both", labelsize=10)
        axes[-1].set_xlabel("Budget", fontsize=12)
        plt.tight_layout()
        plt.savefig(fig_name, dpi=100, bbox_inches="tight")
        plt.close()

theta_columns = [f"ability_{bench}" for bench in unique_bench_names]
resmat_df_reset = resmat_df.reset_index()
test_df_reset = test_df.reset_index()
index_cols = [col for col in resmat_df.index.names if col not in theta_columns]
resmat_df_reset = resmat_df_reset.set_index(index_cols)
test_df_reset = test_df_reset.set_index(index_cols)

theta_values = np.column_stack([bench_thetas[b] for b in unique_bench_names])
resmat_df_reset.loc[test_df_reset.index, theta_columns] = theta_values
assert not resmat_df_reset[theta_columns].isna().any().any()

combined = resmat_df_reset.reset_index().set_index(resmat_df.index.names)
combined.columns = pd.MultiIndex.from_tuples(combined.columns, names=resmat_df.columns.names)

print(f"Shape: {combined.shape}")
idx = combined.index
print("\nRow index names:", idx.names)
print("Row index sample (first 5):", list(islice(idx, 5)))
cols = combined.columns
print("\nColumn index names:", cols.names)
print("Column index sample (first 5):", list(islice(cols, 5)))

if not args.dry_run:
    combined.to_parquet(matrix_output_path)
