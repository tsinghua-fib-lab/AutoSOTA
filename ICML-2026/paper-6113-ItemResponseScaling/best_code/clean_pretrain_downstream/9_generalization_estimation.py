from pathlib import Path
import numpy as np
import pandas as pd
import torch
import pickle
import sys
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent / "data"
sys.path.append(str(BASE_DIR.parent))
from utils import calibrate_1pl_theta, calculate_flops

DEVICE = "cuda:2"
INPUT_PATH = BASE_DIR / "4_prob_matrix_calibrated.parquet"
OUTPUT_PATH = BASE_DIR / "9_generalization_estimation.pkl"

resmat_df = pd.read_parquet(INPUT_PATH)
test_df = resmat_df[resmat_df.index.get_level_values("model_split") == "test"].copy()

test_ys = test_df.to_numpy(dtype=np.float32)
bench_names = test_df.columns.get_level_values("bench_name").map(
    lambda b: "mmlu" if b.startswith("mmlu") else b
)
unique_bench_names = sorted(bench_names.unique())
zs = test_df.columns.get_level_values("difficulty").to_numpy(dtype=np.float32)

index_df = test_df.index.to_frame(index=False)
model_data_mix = index_df["model_data_mix"].tolist()
model_size = index_df["model_size"].tolist()
model_step = index_df["model_step"].astype(int).tolist()
max_step = (
    index_df.groupby(["model_data_mix", "model_size"], as_index=False)["model_step"]
    .max()
    .rename(columns={"model_step": "max_model_step"})
)
index_with_max = index_df.merge(max_step, on=["model_data_mix", "model_size"], how="left")
is_final = (index_with_max["model_step"] == index_with_max["max_model_step"]).to_numpy()
flops = [
    calculate_flops(size, step) if is_final[i] else np.nan
    for i, (size, step) in enumerate(zip(model_size, model_step))
]

output_dict = {}
for bench in tqdm(unique_bench_names, desc="benches"):
    bench_mask = bench_names == bench
    bench_ys = test_ys[:, bench_mask]
    bench_zs = zs[bench_mask]

    col_means = np.nanmean(bench_ys, axis=0)
    order = np.argsort(col_means)  # small -> hard, large -> easy
    half = len(order) // 2
    hard_local = order[:half]
    easy_local = order[half:]

    easy_ys = torch.tensor(bench_ys[:, easy_local])
    easy_zs = torch.tensor(bench_zs[easy_local])
    thetas = calibrate_1pl_theta(easy_ys, DEVICE, easy_zs)

    output_dict[bench] = {
        "thetas_from_easy": thetas,
        "model_data_mix": model_data_mix,
        "model_size": model_size,
        "model_step": model_step,
        "flops": flops,
        "easy_ys": bench_ys[:, easy_local],
        "easy_zs": bench_zs[easy_local],
        "hard_ys": bench_ys[:, hard_local],
        "hard_zs": bench_zs[hard_local],
    }

with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(output_dict, f)
