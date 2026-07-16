import argparse
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import sys
torch.manual_seed(0)
torch.set_num_threads(1)

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR.parent))
from utils import calibrate_1pl_theta

DEVICE = "cuda:7"
DATA_DIR = BASE_DIR / "data"

parser = argparse.ArgumentParser()
parser.add_argument("--input-hf-repo", type=str, default="irsl_testtime_resmat2", choices=["irsl_testtime_resmat1", "irsl_testtime_resmat2"])
args = parser.parse_args()

input_path = DATA_DIR / f"1_calibreated_{args.input_hf_repo}.pt"
payload = torch.load(input_path, map_location="cpu", weights_only=False)

data_tensor = np.array(payload["data_tensor"], dtype=np.float32)
model_names = list(payload["models"])
test_models = list(payload["test_models"])
test_model_indices = [i for i, m in enumerate(model_names) if m in set(test_models)]
datasets = list(payload["datasets"])
zs = np.array(payload["zs"], dtype=np.float32)

output_dict = {}
unique_datasets = sorted(set(datasets))
for dataset in tqdm(unique_datasets, desc="datasets"):
    item_indices = [i for i, d in enumerate(datasets) if d == dataset]
    bench_tensor = data_tensor[test_model_indices][:, item_indices, :]
    bench_zs = zs[item_indices]

    bench_probmat_full = np.nanmean(bench_tensor, axis=-1)
    col_means = np.nanmean(bench_probmat_full, axis=0)
    order = np.argsort(col_means)  # small -> hard, large -> easy
    half = len(order) // 2
    hard_local = order[:half].tolist()
    easy_local = order[half:].tolist()

    easy_probmat = torch.tensor(bench_probmat_full[:, easy_local])
    easy_zs = torch.tensor(bench_zs[easy_local])
    thetas = calibrate_1pl_theta(easy_probmat, DEVICE, easy_zs)

    easy_tensor = bench_tensor[:, easy_local, :]
    hard_tensor = bench_tensor[:, hard_local, :]
    hard_zs = torch.tensor(bench_zs[hard_local])

    output_dict[dataset] = {
        "theta_from_easy": thetas,
        "model_names": test_models,
        "easy_zs": easy_zs.cpu().numpy(),
        "hard_zs": hard_zs.cpu().numpy(),
        "easy_tensor": easy_tensor,
        "hard_tensor": hard_tensor,
    }

output_path = DATA_DIR / f"4_generalization_estimation_{args.input_hf_repo}.pt"
torch.save(output_dict, output_path)
