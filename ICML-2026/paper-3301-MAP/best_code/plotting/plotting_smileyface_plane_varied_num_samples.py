import os
import sys
import re
import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

# Move to the project root (one level up from current file)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

from datasets import SmileyFaceDataset
from trainers import DDPMTrainer
from utils.metrics import coverage, MMD, ensure_tensor_2d, filter_valid_samples
from plotting.plotting_smileyface_plane import to_intrinsic_2d_plane


def _save_pdf_png(fig, output_path, **kwargs):
    fig.savefig(output_path, **kwargs)
    if output_path.lower().endswith(".pdf"):
        fig.savefig(output_path[:-4] + ".png", **kwargs)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fixed plane definition (same as other smileyface scripts)
A = torch.tensor([1.0, 2.0, 3.0]).unsqueeze(0)
b = torch.tensor([1.0])

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--seed", type=int, default=42, help="Random seed")
args, _ = parser.parse_known_args()

# Experiment defaults
fixed_sigma = 0.01
epochs = 200
hidden_dim = 64
time_embed_dim = 32
timesteps = 250
time_concat = True
time_embed_choice = "default"
random_seed = args.seed
trials = 10

# Prepare true data (no noise) and project to 2D intrinsic coordinates
torch.manual_seed(random_seed)
np.random.seed(random_seed)
dataset_true = SmileyFaceDataset(
    num_samples=10000,
    A=A,
    b=b,
    lifted=False,
    noise_level=0.0,
    device=device,
    seed=random_seed,
)
data_points = torch.stack([dataset_true[i] for i in range(len(dataset_true))])
D2 = 2
with torch.no_grad():
    data_points_plain_2d = to_intrinsic_2d_plane(data_points.cpu(), A.cpu(), b.cpu())
true_tensor_2d = ensure_tensor_2d(data_points_plain_2d, D2).cpu()
true_tensor_2d = filter_valid_samples(true_tensor_2d).cpu()


def _list_num_sample_checkpoints(model_dir, pattern):
    """Return sorted list of (N, path) pairs matching a filename pattern with num_samples.
    pattern should include a capturing group for the number of samples.
    """
    out = []
    try:
        for fname in os.listdir(model_dir):
            m = re.match(pattern, fname)
            if m:
                try:
                    N = int(m.group(1))
                    out.append((N, os.path.join(model_dir, fname)))
                except Exception:
                    continue
    except Exception:
        pass
    # sort by N ascending
    out.sort(key=lambda x: x[0])
    return out


model_dir = os.path.join("models", "smileyface_plane")

# Patterns to match Lifted DDPM and baseline DDPM (NONPROJECT) checkpoints with num_samples
# Example filenames we expect:
#  - model_DDPM_epoch_200_num_samples_1000_noise_level_0.01_time_default_seed_42.pth
#  - model_DDPM_NONPROJECT_epoch_200_num_samples_1000_noise_level_0.0_time_default_seed_42.pth
lifted_pat = rf"model_DDPM_epoch_{epochs}_num_samples_(\d+)_noise_level_{fixed_sigma}_time_{time_embed_choice}_seed_{random_seed}\.pth"
baseline_pat = rf"model_DDPM_NONPROJECT_epoch_{epochs}_num_samples_(\d+)_noise_level_0\.0_time_{time_embed_choice}_seed_{random_seed}\.pth"

lifted_ckpts = _list_num_sample_checkpoints(model_dir, lifted_pat)
baseline_ckpts = _list_num_sample_checkpoints(model_dir, baseline_pat)

if len(lifted_ckpts) == 0:
    print(f"No lifted DDPM checkpoints found in {model_dir} for sigma={fixed_sigma}.")
if len(baseline_ckpts) == 0:
    print(f"No baseline DDPM (NONPROJECT) checkpoints found in {model_dir}.")

# Build maps from N to path for quick lookup
lifted_map = {N: path for N, path in lifted_ckpts}
baseline_map = {N: path for N, path in baseline_ckpts}
candidate_N = sorted(set(lifted_map.keys()) & set(baseline_map.keys()))
if len(candidate_N) == 0:
    # If there is no overlap, just use lifted Ns
    candidate_N = sorted(lifted_map.keys())

# Initialize the trainer objects once (weights loaded per-iteration)
trainer_lifted = DDPMTrainer(
    data_points.squeeze(),
    project_x0_sample=True,
    timesteps=timesteps,
    constraints_dict={"linear_equality": (A.to(device), b.to(device))},
    hidden_dim=hidden_dim,
    time_concat=time_concat,
    time_embed_dim=time_embed_dim,
    time_conditioning=time_embed_choice,
)

trainer_plain = DDPMTrainer(
    data_points.squeeze(),
    project_x0_sample=False,
    timesteps=timesteps,
    constraints_dict={"linear_equality": (A.to(device), b.to(device))},
    hidden_dim=hidden_dim,
    time_concat=time_concat,
    time_embed_dim=time_embed_dim,
    time_conditioning=time_embed_choice,
)


# Accumulators
Ns = []
coverage_mean_lifted = []
coverage_std_lifted = []
MMD_mean_lifted = []
MMD_std_lifted = []

coverage_mean_plain = []
coverage_std_plain = []
MMD_mean_plain = []
MMD_std_plain = []

scores_list = []  # lifted score curves per N (trial 0)
scores_plain_first = None

for N in candidate_N:
    lifted_path = lifted_map.get(N, None)
    baseline_path = baseline_map.get(N, None)
    if lifted_path is None:
        continue

    # Load lifted checkpoint
    checkpoint = torch.load(lifted_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    has_timeembed_keys = any(k.startswith("time_embed_module.") for k in state.keys()) if isinstance(state, dict) else False
    if (not has_timeembed_keys) and getattr(trainer_lifted.denoiser, "time_embed_module", None) is not None:
        trainer_lifted.denoiser.time_embed_module = None
    trainer_lifted.denoiser.load_state_dict(state if isinstance(state, dict) else {})
    trainer_lifted.denoiser.eval()

    # Trials for lifted
    trial_cov = []
    trial_mmd = []
    for t in range(trials):
        with torch.no_grad():
            samples_lifted, _ = trainer_lifted.sample(num_samples=N)
        # collect scores from first trial
        if t == 0:
            try:
                scores_list.append(list(trainer_lifted.scores))
            except Exception:
                scores_list.append([])
        # project to plane 2D intrinsic
        with torch.no_grad():
            samples_lifted_2d = to_intrinsic_2d_plane(torch.tensor(samples_lifted), A, b)
        samples_2d = ensure_tensor_2d(samples_lifted_2d, D2).cpu()
        samples_2d = filter_valid_samples(samples_2d).cpu()
        # metrics
        try:
            cov_val = float(coverage(true_tensor_2d, samples_2d))
        except Exception:
            cov_val = float('nan')
        try:
            mmd_val = float(MMD(samples_2d, true_tensor_2d))
        except Exception:
            mmd_val = float('nan')
        trial_cov.append(cov_val)
        trial_mmd.append(mmd_val)

    Ns.append(N)
    coverage_mean_lifted.append(float(np.nanmean(np.array(trial_cov))))
    coverage_std_lifted.append(float(np.nanstd(np.array(trial_cov))))
    MMD_mean_lifted.append(float(np.nanmean(np.array(trial_mmd))))
    MMD_std_lifted.append(float(np.nanstd(np.array(trial_mmd))))

    # Baseline for this N, if available
    if baseline_path is not None:
        checkpoint_b = torch.load(baseline_path, map_location=device)
        state_b = checkpoint_b.get("model_state_dict", checkpoint_b)
        has_timeembed_keys_b = any(k.startswith("time_embed_module.") for k in state_b.keys()) if isinstance(state_b, dict) else False
        if (not has_timeembed_keys_b) and getattr(trainer_plain.denoiser, "time_embed_module", None) is not None:
            trainer_plain.denoiser.time_embed_module = None
        trainer_plain.denoiser.load_state_dict(state_b if isinstance(state_b, dict) else {})
        trainer_plain.denoiser.eval()

        trial_cov_p = []
        trial_mmd_p = []
        for t in range(trials):
            with torch.no_grad():
                samples_plain, _ = trainer_plain.sample(num_samples=N)
            if t == 0 and scores_plain_first is None:
                try:
                    scores_plain_first = list(trainer_plain.scores)
                except Exception:
                    scores_plain_first = []
            with torch.no_grad():
                samples_plain_2d = to_intrinsic_2d_plane(torch.tensor(samples_plain), A, b)
            samples_plain_2d = ensure_tensor_2d(samples_plain_2d, D2).cpu()
            samples_plain_2d = filter_valid_samples(samples_plain_2d).cpu()
            try:
                trial_cov_p.append(float(coverage(true_tensor_2d, samples_plain_2d)))
            except Exception:
                trial_cov_p.append(float('nan'))
            try:
                trial_mmd_p.append(float(MMD(samples_plain_2d, true_tensor_2d)))
            except Exception:
                trial_mmd_p.append(float('nan'))

        coverage_mean_plain.append(float(np.nanmean(np.array(trial_cov_p))))
        coverage_std_plain.append(float(np.nanstd(np.array(trial_cov_p))))
        MMD_mean_plain.append(float(np.nanmean(np.array(trial_mmd_p))))
        MMD_std_plain.append(float(np.nanstd(np.array(trial_mmd_p))))
    else:
        coverage_mean_plain.append(float('nan'))
        coverage_std_plain.append(float('nan'))
        MMD_mean_plain.append(float('nan'))
        MMD_std_plain.append(float('nan'))


# Save metrics JSON
output_dir = "results/smileyface_plane"
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, "metrics_varied_num_samples.json"), "w") as f:
    json.dump(
        {
            "num_samples_list": Ns,
            "lifted": {
                "coverage_mean": coverage_mean_lifted,
                "coverage_std": coverage_std_lifted,
                "MMD_mean": MMD_mean_lifted,
                "MMD_std": MMD_std_lifted,
            },
            "plain": {
                "coverage_mean": coverage_mean_plain,
                "coverage_std": coverage_std_plain,
                "MMD_mean": MMD_mean_plain,
                "MMD_std": MMD_std_plain,
            },
        },
        f,
    )


# Plot Coverage vs number of training samples
plt.figure(figsize=(10, 8))
plt.plot(Ns, coverage_mean_lifted, marker="o", label="Lifted DDPM")
plt.fill_between(Ns,
                 np.array(coverage_mean_lifted) - np.array(coverage_std_lifted),
                 np.array(coverage_mean_lifted) + np.array(coverage_std_lifted),
                 alpha=0.25)
plt.plot(Ns, coverage_mean_plain, marker="s", linestyle="--", color="r", label="DDPM")
plt.fill_between(Ns,
                 np.array(coverage_mean_plain) - np.array(coverage_std_plain),
                 np.array(coverage_mean_plain) + np.array(coverage_std_plain),
                 color="r", alpha=0.1)
plt.xlabel("Number of training samples (N)")
plt.ylabel("Coverage")
plt.grid(True)
plt.legend()
_save_pdf_png(plt.gcf(), os.path.join(output_dir, "coverage_vs_num_samples.pdf"))


# Plot MMD vs number of training samples
plt.figure(figsize=(10, 8))
plt.plot(Ns, MMD_mean_lifted, marker="o", label="Lifted DDPM")
plt.fill_between(Ns,
                 np.array(MMD_mean_lifted) - np.array(MMD_std_lifted),
                 np.array(MMD_mean_lifted) + np.array(MMD_std_lifted),
                 alpha=0.25)
plt.plot(Ns, MMD_mean_plain, marker="s", linestyle="--", color="r", label="DDPM")
plt.fill_between(Ns,
                 np.array(MMD_mean_plain) - np.array(MMD_std_plain),
                 np.array(MMD_mean_plain) + np.array(MMD_std_plain),
                 color="r", alpha=0.1)
plt.xlabel("Number of training samples (N)")
plt.ylabel("MMD")
plt.grid(True)
plt.legend()
_save_pdf_png(plt.gcf(), os.path.join(output_dir, "MMD_vs_num_samples.pdf"))
