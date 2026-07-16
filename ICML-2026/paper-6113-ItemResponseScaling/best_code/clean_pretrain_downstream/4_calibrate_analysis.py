import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.special import expit
from tueplots import bundles
bundles.icml2024()
rng = np.random.default_rng(0)
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent / "data"

parser = argparse.ArgumentParser()
parser.add_argument("--loss-kind", type=str, default="beta", choices=["beta", "binary"])
parser.add_argument("--irt-model", type=str, default="1pl", choices=["1pl", "2pl"])
args = parser.parse_args()

SAMPLE_QUESTIONS = 5

if args.irt_model == "2pl":
    input_path = (
        BASE_DIR / "4_prob_matrix_calibrated_2pl.parquet"
        if args.loss_kind == "beta"
        else BASE_DIR / "4_binary_matrix_calibrated_2pl.parquet"
    )
    output_root = Path(__file__).resolve().parent / "results" / "4_calibrate_analysis_2pl"
else:
    input_path = (
        BASE_DIR / "4_prob_matrix_calibrated.parquet"
        if args.loss_kind == "beta"
        else BASE_DIR / "4_binary_matrix_calibrated.parquet"
    )
    output_root = Path(__file__).resolve().parent / "results" / "4_calibrate_analysis"
output_root.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(input_path)
train_df = df[df.index.get_level_values("model_split") == "train"].copy()
ys = train_df.to_numpy(dtype=np.float32)
bench_names = train_df.columns.get_level_values("bench_name").map(
    lambda b: "mmlu" if b.startswith("mmlu") else b
)
unique_bench_names = sorted(bench_names.unique())
zs = train_df.columns.get_level_values("difficulty").to_numpy(dtype=np.float32)
if args.irt_model == "2pl":
    alphas = train_df.columns.get_level_values("discrimination").to_numpy(dtype=np.float32)

for bench in unique_bench_names:
    bench_thetas = train_df.index.get_level_values(f"ability_{bench}").to_numpy(dtype=np.float32)
    
    bench_mask = bench_names == bench
    bench_ys = ys[:, bench_mask]
    bench_zs = zs[bench_mask]
    if args.irt_model == "2pl":
        bench_alphas = alphas[bench_mask]

    sample_idxs = rng.choice(bench_ys.shape[1], size=SAMPLE_QUESTIONS, replace=False)
    theta_arange = np.linspace(bench_thetas.min() - 1, bench_thetas.max() + 1, 200)

    for idx in sample_idxs:
        y_j = bench_ys[:, idx]
        z_j = bench_zs[idx]
        if args.irt_model == "2pl":
            alpha_j = bench_alphas[idx]
            curve = expit(alpha_j * (theta_arange + z_j))
        else:
            curve = expit(theta_arange + z_j)

        with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
            plt.figure(figsize=(6, 4))
            plt.scatter(bench_thetas, y_j, s=10, label="Responses")
            plt.plot(theta_arange, curve, color="red", label="IRT Prob")
            plt.xlabel(r"$\theta$", fontsize=14)
            plt.ylabel("IRT Prob / Responses", fontsize=14)
            plt.ylim(0, 1)
            plt.legend(fontsize=12)
            plt.title(f"{bench}, {idx}", fontsize=14)
            plt.tight_layout()
            out_path = output_root / f"{args.loss_kind}_{bench}_{idx}.png"
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close()
