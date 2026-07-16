import argparse
from pathlib import Path
import sys
import os

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from tqdm import tqdm
from tueplots import bundles
bundles.icml2024()
from huggingface_hub import snapshot_download

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR.parent.parent))
from utils import cat_beta_1pl, cat_binary_1pl

REPO_ID = "yuhengtu/irsl_datadecide"
SNAPSHOT_DIR = Path(snapshot_download(repo_id=REPO_ID, repo_type="dataset"))

parser = argparse.ArgumentParser()
parser.add_argument("--loss-kind", type=str, default="beta", choices=["beta", "binary"])
args = parser.parse_args()

n_cpus = int((os.cpu_count()) * 0.8)
print(f"Using {n_cpus} CPUs for parallel processing.")

input_path = (
    SNAPSHOT_DIR / "4_prob_matrix_with_difficulty.parquet"
    if args.loss_kind == "beta"
    else SNAPSHOT_DIR / "4_binary_matrix_with_difficulty.parquet"
)
output_path = (
    BASE_DIR / "5_prob_matrix_with_theta.parquet"
    if args.loss_kind == "beta"
    else BASE_DIR / "5_binary_matrix_with_theta.parquet"
)

df = pd.read_parquet(input_path)
df = df[df.index.get_level_values("model_split") == "test"].copy()

question_ids = df.columns.get_level_values("question_id")
bench_names = question_ids.map(lambda q: q.rsplit("_", 1)[0])
# treat all mmlu sub-benches as "mmlu"
bench_names = bench_names.map(lambda b: "mmlu" if b.startswith("mmlu") else b).unique()
print(f"Processing {len(bench_names)} benches: {bench_names.tolist()}")

plots_dir = BASE_DIR / "results/5_cat_convergence"
os.makedirs(plots_dir, exist_ok=True)

bench_thetas = {}
for bench in tqdm(bench_names):
    bench_mask = question_ids.str.startswith(bench)
    bench_cols = df.columns[bench_mask]

    scen_matrix = torch.tensor(df[bench_cols].to_numpy(dtype=np.float32))
    scen_zs = torch.tensor(bench_cols.get_level_values("difficulty").to_numpy(dtype=np.float32))
    n_models = scen_matrix.shape[0]

    if args.loss_kind == "binary":
        thetass = Parallel(n_jobs=n_cpus)(
            delayed(cat_binary_1pl)(scen_matrix[i], scen_zs, "cpu")
            for i in range(n_models)
        )
        fig_name = plots_dir / f"{bench}_binary_theta_convergence.png"
    else:
        thetass = Parallel(n_jobs=n_cpus)(
            delayed(cat_beta_1pl)(scen_matrix[i], scen_zs, "cpu")
            for i in range(n_models)
        )
        fig_name = plots_dir / f"{bench}_beta_theta_convergence.png"
        
    thetass = torch.tensor(thetass, dtype=torch.float)
    final_thetas = thetass[:, -1]

    max_plot = 50
    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        fig, axes = plt.subplots(
            nrows=max_plot,
            ncols=1,
            figsize=(6, 2 * max_plot),
            sharex=True,
        )
        plot_idxs = torch.randperm(n_models)[:max_plot].tolist()
        axes_list = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        for i_plot, ax in enumerate(axes_list[:max_plot]):
            model_idx = plot_idxs[i_plot]
            ax.plot(np.arange(thetass.shape[-1]), thetass[model_idx].cpu().numpy())
            ax.set_ylabel(r"$\theta$", fontsize=12)
            ax.tick_params(axis="both", labelsize=10)
        axes_list[-1].set_xlabel("Budget", fontsize=12)
        plt.tight_layout()
        plt.savefig(fig_name, dpi=100, bbox_inches="tight")
        plt.close()

    bench_thetas[bench] = final_thetas.cpu().numpy()

theta_columns = [f"model_theta_{bench}" for bench in bench_names]
theta_df = pd.DataFrame(
    np.column_stack([bench_thetas[b] for b in bench_names]),
    index=df.index,
    columns=theta_columns,
)
org_index_cols = list(df.index.names)
combined = pd.concat([df, theta_df], axis=1)
combined = combined.reset_index()
new_index_cols = org_index_cols + theta_columns
combined = combined.set_index(new_index_cols)
combined.to_parquet(output_path)
