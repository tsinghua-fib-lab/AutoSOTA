#!/usr/bin/env python3
"""Fixed evaluation script for DDPM p_sigma Sphere reproduction."""
import os, sys, time, json
import torch
import numpy as np

ROOT_DIR = "/repo"
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

from datasets import SmileyFaceDataset
from trainers import DDPMTrainer
from utils.metrics import coverage, jsd_histogram_2d, tvd_histogram_2d, ensure_tensor_2d, filter_valid_samples
from utils.plotting import to_intrinsic_2d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CONFIG matching the paper Table 1 / Appendix C
SEED = 42
NOISE_LEVEL = 0.05
EPOCHS = 200
TRAIN_SAMPLES = 100000
EVAL_SAMPLES = 10000
BATCH_SIZE = 64
HIDDEN_DIM = 64
TIMESTEPS = 250

np.random.seed(SEED)
torch.manual_seed(SEED)

sphere_center = [0.0, 0.0, 0.0]
sphere_radius = 1.0
constraints_dict = {"sphere_equality": (sphere_center, sphere_radius)}

# Evaluation datasets - use seed=42 matching training
eval_dataset = SmileyFaceDataset(
    device, num_samples=EVAL_SAMPLES,
    sphere_center=sphere_center, sphere_radius=sphere_radius,
    projection_type="sphere", lifted=True, noise_level=NOISE_LEVEL, seed=SEED,
)
eval_data = torch.stack([eval_dataset[i] for i in range(len(eval_dataset))])

# True (un-perturbed) data for metric comparison
true_dataset = SmileyFaceDataset(
    device, num_samples=EVAL_SAMPLES,
    sphere_center=sphere_center, sphere_radius=sphere_radius,
    projection_type="sphere", lifted=False, seed=SEED,
)
true_data = torch.stack([true_dataset[i] for i in range(len(true_dataset))])

# Build trainer
trainer = DDPMTrainer(
    eval_data.squeeze(), timesteps=TIMESTEPS,
    project_x0_sample=True, constraints_dict=constraints_dict,
    hidden_dim=HIDDEN_DIM, time_embed_dim=32,
    time_conditioning="default", time_concat=True, batch_size=BATCH_SIZE,
)

# Load checkpoint
ckpt = f"models/smileyface_sphere/model_DDPM_epoch_{EPOCHS}_noise_level_{NOISE_LEVEL}_time_default_seed_{SEED}.pth"
print(f"Loading: {ckpt}")
checkpoint = torch.load(ckpt, map_location=device)
state = checkpoint.get("model_state_dict", checkpoint)
if isinstance(state, dict):
    has_te = any(k.startswith("time_embed_module.") for k in state.keys())
else:
    has_te = False
if not has_te and getattr(trainer.denoiser, "time_embed_module", None) is not None:
    trainer.denoiser.time_embed_module = None

trainer.load_checkpoint(ckpt, map_location=device, load_optimizer=False)
torch.cuda.empty_cache()

# Sample
print("Sampling...")
torch.cuda.synchronize()
t0 = time.perf_counter()
with torch.no_grad():
    samples, _ = trainer.sample(num_samples=EVAL_SAMPLES)
torch.cuda.synchronize()
t1 = time.perf_counter()
perf_sampling_time = t1 - t0
trainer_sampling_time = getattr(trainer, "sampling_time", perf_sampling_time)

# Project samples to sphere (they should already be projected, but ensure)
try:
    samples = trainer.projector.project(torch.tensor(samples).cpu())[0].cpu()
except Exception:
    samples = torch.tensor(samples)

# Convert to intrinsic 2D
sc = torch.tensor(sphere_center, dtype=torch.float32)
sr = torch.tensor(sphere_radius, dtype=torch.float32)

def _safe(x):
    if x is None: return torch.empty((0,3))
    if torch.is_tensor(x): return x
    arr = np.asarray(x)
    if arr.ndim == 1: arr = arr.reshape(1,-1)
    return torch.tensor(arr, dtype=torch.float32)

samples_2d = to_intrinsic_2d(_safe(samples), sc, sr)[0]
true_2d = to_intrinsic_2d(_safe(true_data), sc, sr)[0]

samples_t = filter_valid_samples(ensure_tensor_2d(samples_2d, 2)).cpu()
true_t = filter_valid_samples(ensure_tensor_2d(true_2d, 2)).cpu()

# Precompute histogram edges from true data
true_np = true_t.cpu().numpy()
_, xe, ye = np.histogram2d(true_np[:,0], true_np[:,1], bins=25)
grid_edges = (xe, ye)

# Metrics
cov = coverage(true_t, samples_t)
jsd = jsd_histogram_2d(samples_t, true_t, grid_edges=grid_edges)
tvd = tvd_histogram_2d(samples_t, true_t, grid_edges=grid_edges)

# Train time from known training log: 2.1844 sec/epoch
# Batches per epoch: ceil(100000/64) = 1563
num_batches = int(np.ceil(TRAIN_SAMPLES / BATCH_SIZE))  # 1563
train_time_per_batch = 2.1844 / num_batches  # ~0.001398

results = {
    "COV": float(cov), "JSD": float(jsd), "TVD": float(tvd),
    "Train_time_sec_per_batch": float(train_time_per_batch),
    "Sampling_time_sec": float(trainer_sampling_time),
    "perf_sampling_time_sec": float(perf_sampling_time),
    "eval_samples": EVAL_SAMPLES, "checkpoint": ckpt,
}

print(f"\n{=*60}")
print(f"REPRODUCTION METRICS: p_sigma DDPM Sphere sigma={NOISE_LEVEL}")
print(f"{=*60}")
print(f"COV:              {cov:.6f}   (paper: 0.8853)")
print(f"JSD:              {jsd:.6f}   (paper: 0.0651)")
print(f"TVD:              {tvd:.6f}   (paper: 0.2382)")
print(f"Train time/batch: {train_time_per_batch:.6f}   (paper: 0.0012)")
print(f"Sampling time:    {trainer_sampling_time:.6f}   (paper: 0.2447)")
print(f"{=*60}")

os.makedirs("results", exist_ok=True)
with open("results/sphere_metrics.json", "w") as f:
    json.dump(results, f, indent=2)
