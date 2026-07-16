import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from tqdm import tqdm
from tueplots import bundles

torch.manual_seed(0)
torch.set_num_threads(1)
bundles.icml2024()

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR.parent))
from utils import calibrate_1pl_theta, cat_beta_1pl

INPUT_HF_REPO = "irsl_testtime_resmat2"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results" / "cat_budget_ablation"
DEVICE = "cpu"
LOAD_PLOT_DATA = False
NROWS = 2
NCOLS = 2
ITEM_BUDGET = 30
MAX_BUDGET = 100
INIT_FRAC = 0
PLOT_METRIC = "mse"

LABEL = "1PL"
INPUT_FILE = f"1_calibreated_{INPUT_HF_REPO}.pt"
CAT_FN = cat_beta_1pl
LINE_COLOR = "#1f77b4"
LINESTYLE = "-"


def cat_theta_trace(
    ys: np.ndarray,
    zs: np.ndarray,
    cat_fn,
    device: str,
    item_budget: int,
    init_frac: float,
) -> np.ndarray:
    ys_t = torch.tensor(ys, dtype=torch.float32)
    zs_t = torch.tensor(zs, dtype=torch.float32)
    return np.asarray(
        cat_fn(ys_t, zs_t, device, budget=item_budget, init_frac=init_frac),
        dtype=np.float32,
    )

if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"cat_budget_ablation_{INPUT_HF_REPO}.csv"
    n_jobs = int(os.cpu_count() * 0.8)

    if LOAD_PLOT_DATA:
        results_df = pd.read_csv(csv_path)
        shared_datasets = sorted(results_df["dataset"].unique().tolist())
    else:
        payload = torch.load(DATA_DIR / INPUT_FILE, map_location="cpu", weights_only=False)
        data_tensor = np.array(payload["data_tensor"], dtype=np.float32)
        model_names = list(payload["models"])
        test_models = list(payload["test_models"])
        test_model_indices = [i for i, m in enumerate(model_names) if m in set(test_models)]
        datasets = list(payload["datasets"])
        zs = np.array(payload["zs"], dtype=np.float32)
        shared_datasets = sorted(set(datasets))
        sample_budgets = np.arange(1, MAX_BUDGET + 1)
        data_tensor = data_tensor[test_model_indices]
        full_probmat = np.nanmean(data_tensor, axis=-1)

        records = []
        for dataset in tqdm(shared_datasets, desc=LABEL):
            item_indices = [i for i, d in enumerate(datasets) if d == dataset]
            bench_full_probmat = full_probmat[:, item_indices]
            bench_zs = zs[item_indices]
            gt_thetas = calibrate_1pl_theta(
                resmat=torch.tensor(bench_full_probmat, dtype=torch.float32),
                device=DEVICE,
                zs=torch.tensor(bench_zs, dtype=torch.float32),
                loss_kind="beta",
            )

            for sample_budget in tqdm(sample_budgets, desc=f"{dataset} sample_budget", leave=False):
                bench_probmat = np.nanmean(data_tensor[:, item_indices, :sample_budget], axis=-1)
                traces = np.asarray(
                    Parallel(n_jobs=n_jobs)(
                        delayed(cat_theta_trace)(
                            ys=bench_probmat[i],
                            zs=bench_zs,
                            cat_fn=CAT_FN,
                            device=DEVICE,
                            item_budget=ITEM_BUDGET,
                            init_frac=INIT_FRAC,
                        )
                        for i in range(bench_probmat.shape[0])
                    ),
                    dtype=np.float32,
                )
                theta_est = traces[:, -1]
                errors = theta_est - gt_thetas
                mae = np.abs(errors).mean()
                mse = np.square(errors).mean()
                records.append(
                    {
                        "dataset": dataset,
                        "sample_budget": sample_budget,
                        "mae": float(mae),
                        "mse": float(mse),
                    }
                )

        results_df = pd.DataFrame.from_records(records)
        results_df.to_csv(csv_path, index=False)

    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        fig, axes = plt.subplots(
            nrows=NROWS,
            ncols=NCOLS,
            figsize=(14, 9),
            sharex=True,
            sharey=False,
        )
        axes = np.atleast_1d(axes).ravel()
        for i_ax, (ax, dataset) in enumerate(zip(axes, shared_datasets)):
            dataset_df = results_df[results_df["dataset"] == dataset]
            ax.plot(
                dataset_df["sample_budget"],
                dataset_df[PLOT_METRIC],
                color=LINE_COLOR,
                linestyle=LINESTYLE,
                linewidth=2.6,
                alpha=0.8,
            )
            dataset_max = dataset_df[PLOT_METRIC].max()
            ax.set_ylim(0, dataset_max * 1.12 if dataset_max > 0 else 1.0)
            ax.set_xlim(1, MAX_BUDGET)
            ax.axvline(50, color="black", linestyle="--", linewidth=1.2, alpha=0.9)
            ax.set_title(dataset, fontsize=22, pad=12)
            if i_ax < NCOLS:
                ax.set_xlabel("")
            else:
                ax.set_xlabel("Sample Budget", fontsize=20)
            if i_ax % NCOLS == 0:
                ax.set_ylabel(PLOT_METRIC.upper(), fontsize=20)
            else:
                ax.set_ylabel("")
            ax.tick_params(axis="both", labelsize=18)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.margins(x=0.01)
        fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=1.6, h_pad=1.4)
        fig.savefig(RESULTS_DIR / f"cat_budget_ablation_{INPUT_HF_REPO}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
