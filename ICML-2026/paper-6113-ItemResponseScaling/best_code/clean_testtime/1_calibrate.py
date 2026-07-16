import argparse
import os
from pathlib import Path
import numpy as np
import torch
from huggingface_hub import snapshot_download
from tqdm import tqdm
import sys
torch.manual_seed(0)
torch.set_num_threads(1)

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR.parent))
from utils import calibrate_1pl_z, calibrate_1pl_theta, calibrate_2pl

HF_CACHE_ROOT = BASE_DIR / ".hf_cache"
os.environ["XDG_CACHE_HOME"] = str(HF_CACHE_ROOT)
os.environ["HF_HOME"] = str(HF_CACHE_ROOT / "huggingface")
os.environ["HF_HUB_CACHE"] = str(HF_CACHE_ROOT / "huggingface" / "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.environ["HF_HUB_CACHE"]
os.environ["HF_XET_CACHE"] = str(HF_CACHE_ROOT / "huggingface" / "xet")
os.environ["HF_ASSETS_CACHE"] = str(HF_CACHE_ROOT / "huggingface" / "assets")
os.environ["HF_HUB_DISABLE_XET"] = "1"

for path in (
    HF_CACHE_ROOT,
    Path(os.environ["HF_HOME"]),
    Path(os.environ["HF_HUB_CACHE"]),
    Path(os.environ["HF_XET_CACHE"]),
    Path(os.environ["HF_ASSETS_CACHE"]),
):
    path.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:7"
N_MODELS_FOR_TEST = 4

parser = argparse.ArgumentParser()
parser.add_argument("--input-hf-repo", type=str, default="irsl_testtime_resmat2", choices=["irsl_testtime_resmat1", "irsl_testtime_resmat2"])
parser.add_argument("--irt-model", type=str, default="1pl", choices=["1pl", "2pl"])
parser.add_argument("--split-seed", type=int, default=0)
parser.add_argument("--data-root", type=Path, default=BASE_DIR / "data")
args = parser.parse_args()

output_dir = args.data_root
output_dir.mkdir(parents=True, exist_ok=True)

cache_dir = snapshot_download(
    repo_id=f"stair-lab/{args.input_hf_repo}",
    repo_type="dataset",
    cache_dir=os.environ["HF_HUB_CACHE"],
)
testtime_resmat = torch.load(f"{cache_dir}/resmat.pt", map_location="cpu", weights_only=False)
data_tensor = testtime_resmat["data_tensor"].numpy()
model_names = list(testtime_resmat["models"])
questions = testtime_resmat["questions"]
if args.input_hf_repo == "irsl_testtime_resmat1":
    helm_zs = np.array(testtime_resmat["zs"])
    datasets = testtime_resmat["scenarios"]
else:
    datasets = testtime_resmat["datasets"]

rng = np.random.default_rng(args.split_seed)
perm = rng.permutation(len(model_names))
data_tensor = data_tensor[perm]
model_names = [model_names[idx] for idx in perm]

n_models_for_train = data_tensor.shape[0] - N_MODELS_FOR_TEST
train_model_names = model_names[:n_models_for_train]
test_model_names = model_names[n_models_for_train:]
print(f"split_seed={args.split_seed}")
print(f"train_model_names={train_model_names}")
print(f"test_model_names={test_model_names}")

train_probmat = torch.nanmean(
    torch.tensor(data_tensor[:n_models_for_train, :, :], dtype=torch.float, device=DEVICE),
    dim=-1,
)
assert not torch.isnan(train_probmat).any()

bench_names = list(datasets)
unique_benches = sorted(set(bench_names))

if args.irt_model == "1pl":
    z_optimized = calibrate_1pl_z(train_probmat, DEVICE)
    bench_thetas = {}
    for bench in tqdm(unique_benches, desc="benches"):
        mask_np = np.array([b == bench for b in bench_names])
        bench_prob = train_probmat[:, torch.tensor(mask_np, device=DEVICE)]
        bench_zs = torch.tensor(z_optimized[mask_np], dtype=torch.float, device=DEVICE)
        theta = calibrate_1pl_theta(bench_prob, DEVICE, bench_zs)
        bench_thetas[bench] = list(zip(train_model_names, theta))
    alphas_optimized = None
else:
    n_items = train_probmat.shape[1]
    final_zs = np.full(n_items, np.nan, dtype=np.float32)
    final_alphas = np.full(n_items, np.nan, dtype=np.float32)
    bench_thetas = {}
    for bench in tqdm(unique_benches, desc="benches"):
        mask_np = np.array([b == bench for b in bench_names])
        bench_prob = train_probmat[:, torch.tensor(mask_np, device=DEVICE)]
        calibrate_res = calibrate_2pl(resmat=bench_prob, device=DEVICE, max_epochs=200)
        z_optimized = calibrate_res["z"]
        alpha_optimized = calibrate_res["alpha"]
        theta_optimized = calibrate_res["theta"]
        final_zs[mask_np] = z_optimized
        final_alphas[mask_np] = alpha_optimized
        bench_thetas[bench] = list(zip(train_model_names, theta_optimized))
    z_optimized = final_zs
    alphas_optimized = final_alphas

assert len(questions) == len(datasets) == data_tensor.shape[1] == z_optimized.shape[0]

irt_suffix = "_2pl" if args.irt_model == "2pl" else ""
out_path = output_dir / f"1_calibreated_{args.input_hf_repo}{irt_suffix}.pt"
payload = {
    "data_tensor": data_tensor,
    "models": model_names,
    "test_models": test_model_names,
    "questions": questions,
    "datasets": datasets,
    "zs": z_optimized,
    "alphas": alphas_optimized,
    "train_thetas": bench_thetas,
    "split_seed": args.split_seed,
}
if args.input_hf_repo == "irsl_testtime_resmat1":
    payload["helm_zs"] = helm_zs
torch.save(payload, out_path)
