import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy.special import expit
from torch.optim import LBFGS
from tqdm import tqdm
from tueplots import bundles

bundles.icml2024()
rng = np.random.default_rng(0)

BASE_DIR = Path(__file__).resolve().parent
import sys
sys.path.append(str(BASE_DIR.parent.parent))
from utils import trainer, beta_nll

device = torch.device("cuda:0")

parser = argparse.ArgumentParser()
parser.add_argument("--loss-kind", type=str, default="beta", choices=["beta", "binary"])
parser.add_argument("--sample-questions", type=int, default=5, help="Number of questions to plot per bench.")
args = parser.parse_args()

input_path = (
    BASE_DIR / "4_prob_matrix_with_difficulty.parquet"
    if args.loss_kind == "beta"
    else BASE_DIR / "4_binary_matrix_with_difficulty.parquet"
)
output_root = BASE_DIR / "results" / "4_IRT_curve_train"
output_root.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(input_path)
train_df = df[df.index.get_level_values("model_split") == "train"].copy()
question_ids = train_df.columns.get_level_values("question_id")
bench_names = question_ids.map(lambda q: q.rsplit("_", 1)[0])
bench_names = bench_names.map(lambda b: "mmlu" if b.startswith("mmlu") else b).unique()
print(f"Processing {len(bench_names)} benches: {bench_names.tolist()}")

for bench in tqdm(bench_names, desc="benches"):
    bench_mask = question_ids.str.startswith(bench)
    bench_cols = train_df.columns[bench_mask]
    zs = torch.tensor(
        bench_cols.get_level_values("difficulty").to_numpy(dtype=np.float64),
        device=device,
    )
    data_np = train_df[bench_cols].to_numpy(dtype=np.float64)
    data_t = torch.tensor(data_np, device=device)

    n_test_takers, n_items = data_t.shape
    phi = 10.0

    thetas = torch.randn(n_test_takers, requires_grad=True, device=device)
    optim_theta = LBFGS([thetas], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")

    def closure_theta():
        optim_theta.zero_grad()
        probs = torch.sigmoid(thetas[:, None] + zs[None, :])
        loss = torch.nanmean(beta_nll(data_t, probs, phi))
        loss.backward()
        return loss

    thetas = trainer([thetas], optim_theta, closure_theta)[0].detach()

    sample_idxs = rng.choice(n_items, size=args.sample_questions, replace=False)
    theta_range = torch.linspace(thetas.min() - 1, thetas.max() + 1, steps=200, device=device)

    theta_np = thetas.cpu().numpy()
    for idx in sample_idxs:
        qid = bench_cols.get_level_values("question_id")[idx]
        z_j = zs[idx].item()
        resp = data_np[:, idx]
        curve = expit(theta_range.cpu().numpy() + z_j)

        plt.figure(figsize=(6, 4))
        plt.scatter(theta_np, resp, s=10, label="responses")
        plt.plot(theta_range.cpu().numpy(), curve, color="red", label="IRT curve")
        plt.xlabel(r"$\theta$", fontsize=14)
        plt.ylabel("Probability", fontsize=14)
        plt.ylim(0, 1)
        plt.legend(fontsize=12)
        plt.title(qid, fontsize=14)
        plt.tight_layout()
        out_path = output_root / f"{args.loss_kind}_{qid}.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
