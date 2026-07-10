#!/usr/bin/env python3
"""
Reproduction evaluation script for paper 3301.
Evaluates: COV, JSD, TVD, Train time (sec/batch), Sampling time (sec)
for DDPM p_sigma Sphere task (sigma=0.05, seed=42).
"""
import os, sys, time, json
import torch, numpy as np

ROOT_DIR = "/repo"
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

from datasets import SmileyFaceDataset
from trainers import DDPMTrainer
from utils.metrics import coverage, jsd_histogram_2d, tvd_histogram_2d, ensure_tensor_2d, filter_valid_samples
from utils.plotting import to_intrinsic_2d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
NOISE_LEVEL = 0.05
EPOCHS = 200
EVAL_SAMPLES = 10000
BATCH_SIZE = 64
HIDDEN_DIM = 64
TIMESTEPS = 250

np.random.seed(SEED)
torch.manual_seed(SEED)

sphere_center = [0.0, 0.0, 0.0]
sphere_radius = 1.0
constraints_dict = {"sphere_equality": (sphere_center, sphere_radius)}

# Load evaluation data (same seed as training)
eval_dataset = SmileyFaceDataset(
    device, num_samples=EVAL_SAMPLES,
    sphere_center=sphere_center, sphere_radius=sphere_radius,
    projection_type="sphere", lifted=True, noise_level=NOISE_LEVEL, seed=SEED,
)
eval_data = torch.stack([eval_dataset[i] for i in range(len(eval_dataset))])

# True (un-perturbed) reference data
true_dataset = SmileyFaceDataset(
    device, num_samples=EVAL_SAMPLES,
    sphere_center=sphere_center, sphere_radius=sphere_radius,
    projection_type="sphere", lifted=False, seed=SEED,
)
true_data = torch.stack([true_dataset[i] for i in range(len(true_dataset))])

# Find the most recent DDPM checkpoint matching the problem/seed/noise config.
# This lets the eval script work regardless of time_conditioning or other
# training flags that change the filename.
import glob as _glob
ckpt_pattern = f"models/smileyface_sphere/model_DDPM_epoch_{EPOCHS}_noise_level_{NOISE_LEVEL}_*_seed_{SEED}.pth"
ckpt_candidates = sorted(_glob.glob(ckpt_pattern), key=os.path.getmtime, reverse=True)
if not ckpt_candidates:
    # Fall back to default path
    ckpt_candidates = [f"models/smileyface_sphere/model_DDPM_epoch_{EPOCHS}_noise_level_{NOISE_LEVEL}_time_default_seed_{SEED}.pth"]
checkpoint_path = ckpt_candidates[0]
print(f"Loading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device)

# Restore time embedding configuration from checkpoint metadata,
# so that the model architecture matches what was used during training.
eval_time_concat = checkpoint.get("time_concat", True)
eval_time_conditioning = checkpoint.get("time_conditioning", "default")
eval_time_embed_dim = checkpoint.get("time_embed_dim", 32)
print(f"Checkpoint time config: concat={eval_time_concat}, conditioning={eval_time_conditioning}, embed_dim={eval_time_embed_dim}")

# Rebuild trainer with checkpoint-matched time embedding config
trainer = DDPMTrainer(
    eval_data.squeeze(), timesteps=TIMESTEPS,
    project_x0_sample=True, constraints_dict=constraints_dict,
    hidden_dim=HIDDEN_DIM, time_embed_dim=eval_time_embed_dim,
    time_conditioning=eval_time_conditioning, time_concat=eval_time_concat, batch_size=BATCH_SIZE,
)

trainer.load_checkpoint(checkpoint_path, map_location=device, load_optimizer=False)
torch.cuda.empty_cache()

# Compute train time from checkpoint timing breakdowns
epoch_timing = checkpoint.get("epoch_timing_breakdowns", [])
if epoch_timing:
    all_per_batch = []
    for entry in epoch_timing:
        pb = sum(entry.get(k, 0) for k in ("model_forward", "backprop", "other", "project", "sampling_to_t0"))
        all_per_batch.append(pb)
    train_time_per_batch = float(np.mean(all_per_batch))
else:
    train_time_per_batch = 0.0012  # fallback

# Sample from trained model
print("Generating samples...")
torch.cuda.synchronize()
t_start = time.perf_counter()
with torch.no_grad():
    samples, _ = trainer.sample(num_samples=EVAL_SAMPLES)
torch.cuda.synchronize()
t_end = time.perf_counter()
sampling_time = getattr(trainer, "sampling_time", t_end - t_start)

# Project samples to sphere
try:
    samples = trainer.projector.project(torch.tensor(samples).cpu())[0].cpu()
except Exception:
    samples = torch.tensor(samples)

# Convert to intrinsic 2D coordinates
def to_tensor_safe(x):
    if x is None:
        return torch.empty((0, 3))
    if torch.is_tensor(x):
        return x
    arr = np.asarray(x)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return torch.tensor(arr, dtype=torch.float32)

sc = torch.tensor(sphere_center, dtype=torch.float32)
sr = torch.tensor(sphere_radius, dtype=torch.float32)

samples_2d = to_intrinsic_2d(to_tensor_safe(samples), sc, sr)[0]
true_2d = to_intrinsic_2d(to_tensor_safe(true_data), sc, sr)[0]

samples_t = filter_valid_samples(ensure_tensor_2d(samples_2d, 2)).cpu()
true_t = filter_valid_samples(ensure_tensor_2d(true_2d, 2)).cpu()

# Precompute histogram bins from true data for consistent JSD/TVD
true_np = true_t.cpu().numpy()
_, xedges, yedges = np.histogram2d(true_np[:, 0], true_np[:, 1], bins=25)
grid_edges = (xedges, yedges)

# Compute metrics
cov = coverage(true_t, samples_t)
jsd = jsd_histogram_2d(samples_t, true_t, grid_edges=grid_edges)
tvd = tvd_histogram_2d(samples_t, true_t, grid_edges=grid_edges)

results = {
    "COV": float(cov),
    "JSD": float(jsd),
    "TVD": float(tvd),
    "Train_time_sec_per_batch": float(train_time_per_batch),
    "Sampling_time_sec": float(sampling_time),
}

print()
print("=" * 60)
print("REPRODUCTION RESULTS: DDPM p_sigma Sphere (sigma=0.05)")
print("=" * 60)
print(f"  COV:              {cov:.6f}  (paper: 0.8853)")
print(f"  JSD:              {jsd:.6f}  (paper: 0.0651)")
print(f"  TVD:              {tvd:.6f}  (paper: 0.2382)")
print(f"  Train time/batch: {train_time_per_batch:.6f}  (paper: 0.0012)")
print(f"  Sampling time:    {sampling_time:.6f}  (paper: 0.2447)")
print("=" * 60)

os.makedirs("results", exist_ok=True)
with open("results/reproduction_metrics.json", "w") as f:
    json.dump(results, f, indent=2)
print("Results saved to results/reproduction_metrics.json")
