import scipy.linalg
import torch.nn as nn
import os
import sys
import argparse

# Move to the project root (one level up from current file)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

# --- Import classifier from training script ---
from train_mnist_classifier import SimpleMNISTClassifier

def load_mnist_classifier(path="models/mnist_classifier.pth", device=None):
    model = SimpleMNISTClassifier()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model
def get_classifier_features(samples, classifier, device=None, batch_size=256):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert to tensor
    x = samples if isinstance(samples, torch.Tensor) else torch.tensor(samples)

    # Force float32
    if x.dtype != torch.float32:
        x = x.float()

    # Scale to [0,1] if needed
    # (assumes either [0,255] uint-like or already [0,1])
    if x.numel() > 0:
        xmax = float(x.max().detach().cpu())
        if xmax > 1.0:
            x = x / 255.0

    # Ensure shape is (N,1,28,28)
    if x.dim() == 2:
        if x.shape[1] == 28 * 28:
            x = x.view(-1, 1, 28, 28)
        else:
            raise ValueError(f"Expected 2D input with 784 features, got {x.shape}")
    elif x.dim() == 3:
        # (N,28,28)
        x = x.unsqueeze(1)
    elif x.dim() == 4:
        # could be (N,28,28,1) or (N,1,28,28) or (N,3,28,28)
        if x.shape[-1] == 1 and x.shape[1] != 1:
            x = x.permute(0, 3, 1, 2)  # (N,H,W,1)->(N,1,H,W)
    else:
        raise ValueError(f"Unexpected samples shape: {tuple(x.shape)}")

    x = x.to(device)

    features = []
    classifier.eval()
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            batch = x[i:i + batch_size]
            _, feats = classifier(batch, return_features=True)  # your CNN should support this
            features.append(feats.detach().cpu())

    feats = torch.cat(features, dim=0).numpy()
    return feats

def compute_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    # Standard FID formula
    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    return float(fid)

def classifier_accuracy(samples, classifier, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(samples, torch.Tensor):
        x = samples
    else:
        x = torch.tensor(samples)
    if x.dtype != torch.float32:
        x = x.float()
        # Diagnostic: print min/max before normalization
        # Only print min/max if x is not empty
        if x.numel() > 0:
            x_min, x_max = float(x.min()), float(x.max())
            print(f"[Classifier] Input min: {x_min:.4f}, max: {x_max:.4f}")
        else:
            x_min, x_max = 0.0, 0.0
            print("[Classifier] Input is empty.")
        # Always min-max normalize to [0,1] as in transforms.ToTensor()
        if x.numel() > 0 and (x_max > 1.0 or x_min < 0.0 or x_max > 1.01 or x_min < -0.01):
            x = (x - x_min) / (x_max - x_min + 1e-8)
            print(f"[Classifier] After normalization: min {float(x.min()):.4f}, max {float(x.max()):.4f}")
    
    # Ensure shape is (N,1,28,28)
    if x.dim() == 2:
        if x.shape[1] == 28 * 28:
            x = x.view(-1, 1, 28, 28)
        else:
            raise ValueError(f"Expected 2D input with 784 features, got {x.shape[1]}")
    elif x.dim() == 3:
        # (N,28,28)
        x = x.unsqueeze(1)
    elif x.dim() == 4:
        # could be (N,28,28,1) or (N,1,28,28) or (N,3,28,28)
        if x.shape[-1] == 1 and x.shape[1] != 1:
            x = x.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"Unexpected samples shape: {tuple(x.shape)}")
    
    x = x.to(device)
    with torch.no_grad():
        logits = classifier(x)
        preds = logits.argmax(dim=1)
    # No ground truth for generated samples, so just return predicted label histogram
    return preds.cpu().numpy()
import os
import sys

# Move to the project root (one level up from current file)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from geomloss import SamplesLoss

from datasets import MNISTFixedSumDataset
from trainers import DDPMTrainer
from utils.constraints import FixedSumProjector
from utils.plotting import count_trainable_params
from utils.metrics import coverage, filter_valid_samples

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parse command-line arguments
parser = argparse.ArgumentParser(description='MNIST Fixed-Sum Diffusion Evaluation')
parser.add_argument('--num_eval_samples', type=int, default=10000, help='Number of samples to evaluate; MNIST evaluation is forced to the full 10k test set')
parser.add_argument('--num_samples', type=int, default=10000, help='Number of training samples used')
parser.add_argument('--target_sum', type=float, default=100.0, help='Target pixel sum constraint')
parser.add_argument('--noise_level', type=float, default=0.001, help='Noise level (sigma) for lifted diffusion')
parser.add_argument('--hidden_dim', type=int, default=1024, help='Hidden dimension for models')
parser.add_argument('--time_embed_dim', type=int, default=64, help='Time embedding dimension')
parser.add_argument('--timesteps', type=int, default=250, help='Number of diffusion timesteps')
parser.add_argument('--time_embed_choice', type=str, default='default', help='Time embedding type')
parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs (for checkpoint loading)')
parser.add_argument('--random_seed', type=int, default=42, help='Random seed (for checkpoint matching)')
parser.add_argument('--seed', type=int, default=None, help='Random seed (alias for --random_seed)')
parser.add_argument('--n_trials', type=int, default=3, help='Number of trials to average timing/metrics over')
parser.add_argument('--compute_table', action='store_true', default=True, help='Compute and write the metrics table (enabled by default)')
parser.add_argument('--metric_sigma', type=float, default=0.001, help='Deprecated; metrics_table.tex is written for the current --noise_level')
parser.add_argument('--metric_num_samples', type=int, default=None, help='Deprecated; metrics_table.tex is written for the current --num_samples')
args = parser.parse_args()

MNIST_TEST_SET_SIZE = 10000
# Respect the requested number of evaluation samples, but cap at the full test-set size
num_eval_samples = args.num_eval_samples if args.num_eval_samples is not None else MNIST_TEST_SET_SIZE
if num_eval_samples > MNIST_TEST_SET_SIZE:
    print(f"--num_eval_samples={num_eval_samples} exceeds test set size {MNIST_TEST_SET_SIZE}; capping to {MNIST_TEST_SET_SIZE}")
    num_eval_samples = MNIST_TEST_SET_SIZE
num_samples = args.num_samples
target_sum = args.target_sum
noise_level = args.noise_level
hidden_dim = args.hidden_dim
time_embed_dim = args.time_embed_dim
timesteps = args.timesteps
time_embed_choice = args.time_embed_choice
epochs = args.epochs
random_seed = args.seed if args.seed is not None else args.random_seed
n_trials = args.n_trials
compute_table = args.compute_table
metric_sigma = args.metric_sigma
metric_num_samples = args.metric_num_samples

torch.manual_seed(random_seed)
np.random.seed(random_seed)

def _attach_time_embed_if_needed(denoiser, state_dict, device):
    """Placeholder function - time embedding utilities have been removed."""
    pass
dataset = MNISTFixedSumDataset(device=device, pixel_sum=target_sum, noise_level=noise_level, lifted=True, train=False)
if len(dataset) != MNIST_TEST_SET_SIZE:
    raise ValueError(f"Expected MNIST test set to contain {MNIST_TEST_SET_SIZE} samples, got {len(dataset)}")
data_points = torch.stack([dataset[i] for i in range(num_eval_samples)])

# Trainer: Lifted Diffusion Score (project_x0_sample=True)
print("Lifted Diffusion Model")
trainer_lifted = DDPMTrainer(
    data_points.squeeze(),
    timesteps=timesteps,
    project_x0_sample=True,
    projector=FixedSumProjector(target_sum=target_sum),
    hidden_dim=hidden_dim,
    time_embed_dim=time_embed_dim,
    size=784,
    image=True
    # constraints_dict={"linear_equality": {torch.ones(1, data_points.shape[1]).to(device), torch.tensor([target_sum]).to(device)}},
)
checkpoint_path = f'models/mnist/model_DDPM_epoch_{epochs}_num_samples_{num_samples}_noise_level_{noise_level}_time_{time_embed_choice}_seed_{random_seed}.pth'
# Load checkpoint dict first so we can detect any time-embedding keys and attach/remove modules
checkpoint = torch.load(checkpoint_path, map_location=device)
state_dict = checkpoint.get('model_state_dict', checkpoint)
_attach_time_embed_if_needed(trainer_lifted.denoiser, state_dict if isinstance(state_dict, dict) else {}, device)
has_timeembed = any(k.startswith('time_embed_module.') for k in state_dict.keys()) if isinstance(state_dict, dict) else False
if not has_timeembed and getattr(trainer_lifted.denoiser, 'time_embed_module', None) is not None:
    trainer_lifted.denoiser.time_embed_module = None
# Use trainer.load_checkpoint to restore timing metadata (epoch_timing_breakdowns etc.) in addition to model weights
trainer_lifted.load_checkpoint(checkpoint_path, map_location=device, load_optimizer=False)
torch.cuda.empty_cache()
trainer_lifted.denoiser.eval()
with torch.no_grad():
    samples_lifted, _ = trainer_lifted.sample(num_samples=num_eval_samples)
try:
    samples_lifted = trainer_lifted.projector.project(torch.tensor(samples_lifted).cpu())[0].cpu()
except Exception:
    samples_lifted = torch.tensor(samples_lifted)

# Trainer: PDM (project_x0_sample=True, no lifting)
dataset_plain = MNISTFixedSumDataset(pixel_sum=target_sum, noise_level=0.0, device=device, train=False)
if len(dataset_plain) != MNIST_TEST_SET_SIZE:
    raise ValueError(f"Expected MNIST test set to contain {MNIST_TEST_SET_SIZE} samples, got {len(dataset_plain)}")
data_points_plain = torch.stack([dataset_plain[i] for i in range(num_eval_samples)])
trainer_PDM = DDPMTrainer(
    data_points_plain.squeeze(),
    timesteps=timesteps,
        project_x0_sample=True,
    projector=FixedSumProjector(target_sum=target_sum),
    hidden_dim=hidden_dim,
    time_embed_dim=time_embed_dim,
    size=784,
    image=True
    # constraints_dict={"linear_equality": {torch.ones(1, data_points_plain.shape[1]).to(device), torch.tensor([target_sum]).to(device)}},
)
print("Projected Diffusion Model")
checkpoint_path = f'models/mnist/model_DDPM_NONPROJECT_epoch_{epochs}_num_samples_{num_samples}_noise_level_0.0_time_{time_embed_choice}_seed_{random_seed}.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)
state_dict = checkpoint.get('model_state_dict', checkpoint)
_attach_time_embed_if_needed(trainer_PDM.denoiser, state_dict if isinstance(state_dict, dict) else {}, device)
has_timeembed = any(k.startswith('time_embed_module.') for k in state_dict.keys()) if isinstance(state_dict, dict) else False
if not has_timeembed and getattr(trainer_PDM.denoiser, 'time_embed_module', None) is not None:
    trainer_PDM.denoiser.time_embed_module = None
trainer_PDM.load_checkpoint(checkpoint_path, map_location=device, load_optimizer=False)
trainer_PDM.denoiser.eval()
with torch.no_grad():
    samples_PDM, _ = trainer_PDM.sample(num_samples=num_eval_samples, PDM=True)

# Trainer: Traditional DDPM Score (project_x0_sample=False)
print("Traditional DDPM Model")
trainer_plain = DDPMTrainer(
    data_points_plain.squeeze(),
    timesteps=timesteps,
    project_x0_sample=False,
    projector=FixedSumProjector(target_sum=target_sum),
    hidden_dim=hidden_dim,
    time_embed_dim=time_embed_dim,
    size=784,
    image=True
    # constraints_dict={"linear_equality": {torch.ones(1, data_points_plain.shape[1]).to(device), torch.tensor([target_sum]).to(device)}},
)
checkpoint_path = f'models/mnist/model_DDPM_NONPROJECT_epoch_{epochs}_num_samples_{num_samples}_noise_level_0.0_time_{time_embed_choice}_seed_{random_seed}.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)
state_dict = checkpoint.get('model_state_dict', checkpoint)
_attach_time_embed_if_needed(trainer_plain.denoiser, state_dict if isinstance(state_dict, dict) else {}, device)
has_timeembed = any(k.startswith('time_embed_module.') for k in state_dict.keys()) if isinstance(state_dict, dict) else False
if not has_timeembed and getattr(trainer_plain.denoiser, 'time_embed_module', None) is not None:
    trainer_plain.denoiser.time_embed_module = None
trainer_plain.load_checkpoint(checkpoint_path, map_location=device, load_optimizer=False)
with torch.no_grad():
    samples_plain, norms = trainer_plain.sample(num_samples=num_eval_samples)
print("Average deviation of Traditional DDPM samples from the fixed sum hyperplane:", norms)
# We'll compute an externally-projected DDPM sample and measure how long that projection takes
# below (outside of the trainers map) so the plotted DDPM bar stays without projection time
# while DDPM_projected will show the external projection cost.
samples_plain_projected = None
# Trainer: PIDM (same as above)
print("Physics-Informed Diffusion Model")
trainer_PIDM = DDPMTrainer(
    data_points_plain.squeeze(),
    timesteps=timesteps,
    project_x0_sample=False,
    projector=FixedSumProjector(target_sum=target_sum),
    hidden_dim=hidden_dim,
    time_embed_dim=time_embed_dim,
    size=784,
    image=True
    # constraints_dict={"linear_equality": {torch.ones(1, data_points_plain.shape[1]).to(device), torch.tensor([target_sum]).to(device)}},
)
checkpoint_path = f'models/mnist/model_PIDM_epoch_{epochs}_num_samples_{num_samples}_noise_level_0.0_time_{time_embed_choice}_seed_{random_seed}.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)
state_dict = checkpoint.get('model_state_dict', checkpoint)
_attach_time_embed_if_needed(trainer_PIDM.denoiser, state_dict if isinstance(state_dict, dict) else {}, device)
has_timeembed = any(k.startswith('time_embed_module.') for k in state_dict.keys()) if isinstance(state_dict, dict) else False
if not has_timeembed and getattr(trainer_PIDM.denoiser, 'time_embed_module', None) is not None:
    trainer_PIDM.denoiser.time_embed_module = None
trainer_PIDM.load_checkpoint(checkpoint_path, map_location=device, load_optimizer=False)
trainer_PIDM.denoiser.eval()
with torch.no_grad():
    samples_PIDM, norms_PIDM = trainer_PIDM.sample(num_samples=num_eval_samples)
print("Average deviation of PIDM samples from the plane:", norms_PIDM)

from utils.plotting import plot_scores_vs_time

# Score curves
scores_lifted = np.array(trainer_lifted.scores, dtype=np.float64)
scores_plain = np.array(trainer_plain.scores, dtype=np.float64)
os.makedirs("results/mnist/", exist_ok=True)
plot_scores_vs_time(
    scores_list=[scores_lifted],
    scores_plain=scores_plain,
    sigma_list=[1.0],
    output_path="results/mnist/scores_mnist.pdf",
)

D = data_points_plain.shape[1]
# We'll run multiple trials of sampling and average timing/metric results if requested.
# `n_trials` is taken from command-line `--n_trials` (default 3).

# Map method names to trainer objects for sampling
primary_trainers = {
    'Lifted': locals().get('trainer_lifted', None),
    'PDM':   locals().get('trainer_PDM', None),
    'DDPM':  locals().get('trainer_plain', None),
    'PIDM':  locals().get('trainer_PIDM', None),
}

# Accumulators for timings and metrics
acc = {}
for name in list(primary_trainers.keys()) + ['DDPM (proj.)']:
    acc[name] = {
        'model': 0.0,
        'proj': 0.0,
        'sampling': 0.0,
        'coverage': 0.0,
        'count': 0,
    }

# Store first-trial samples for visualization
first_samples = {}

true_tensor = filter_valid_samples(data_points_plain.view(-1, D)).cpu()

# Try to load classifier and precompute real-data embeddings for embedding-based coverage
try:
    classifier = load_mnist_classifier("models/mnist_classifier.pth", device=device)
    feats_real_for_coverage = get_classifier_features(true_tensor, classifier, device=device).astype(np.float64)
    classifier_loaded_for_coverage = True
except Exception:
    classifier = None
    feats_real_for_coverage = None
    classifier_loaded_for_coverage = False

def _embedding_coverage(samples_tensor):
    if not classifier_loaded_for_coverage:
        return float('nan')
    feats_gen = get_classifier_features(samples_tensor, classifier, device=device).astype(np.float64)
    return float(coverage(feats_real_for_coverage, feats_gen))

# Run trials
for trial in range(n_trials):
    print(f"Sampling trial {trial+1}/{n_trials}...")
    # For each primary method, call its trainer.sample() and measure metrics
    for name, tr in primary_trainers.items():
        if tr is None:
            continue
        sample_kwargs = {'PDM': True} if name == 'PDM' else {}
        with torch.no_grad():
            try:
                samples, norms = tr.sample(num_samples=num_eval_samples, **sample_kwargs)
                if name == 'Lifted':
                    try:
                        samples = tr.projector.project(torch.tensor(samples).cpu())[0].cpu()
                    except Exception:
                        samples = torch.tensor(samples)
            except Exception as e:
                print(f"Sampling error for {name} on trial {trial}: {e}")
                samples = np.array([])
            # Define x as the tensor to be normalized and diagnosed
            if isinstance(samples, np.ndarray):
                x = torch.from_numpy(samples)
            else:
                x = samples
            if hasattr(x, 'to'):
                x = x.to(torch.float32)
            # Diagnostic: print min/max before normalization
            if hasattr(x, 'numel') and x.numel() > 0:
                x_min, x_max = float(x.min()), float(x.max())
                print(f"[Features] Input min: {x_min:.4f}, max: {x_max:.4f}")
            else:
                x_min, x_max = 0.0, 0.0
                print("[Features] Input is empty.")
            # Always min-max normalize to [0,1] as in transforms.ToTensor()
            if hasattr(x, 'numel') and x.numel() > 0 and (x_max > 1.0 or x_min < 0.0 or x_max > 1.01 or x_min < -0.01):
                x = (x - x_min) / (x_max - x_min + 1e-8)
                print(f"[Features] After normalization: min {float(x.min()):.4f}, max {float(x.max()):.4f}")
        # Capture timing summaries from the trainer (best-effort)
        model_t = getattr(tr, 'total_model_forward_sample_time', None)
        if model_t is None:
            # fallback to summing per-call list
            lst = getattr(tr, 'model_forward_times', []) or []
            try:
                model_t = float(np.sum([float(x) for x in lst])) if len(lst) > 0 else float('nan')
            except Exception:
                model_t = float('nan')

        proj_t = getattr(tr, 'total_projection_sample_time', None)
        if proj_t is None:
            lst = getattr(tr, 'projection_sample_times', []) or []
            try:
                proj_t = float(np.sum([float(x) for x in lst])) if len(lst) > 0 else float('nan')
            except Exception:
                proj_t = float('nan')

        samp_t = getattr(tr, 'sampling_time', float('nan'))

        # Compute metrics from samples (filter invalids)
        try:
            samples_tensor = filter_valid_samples(torch.tensor(samples).view(-1, D)).cpu()
            cov = _embedding_coverage(samples_tensor)
        except Exception:
            samples_tensor = filter_valid_samples(torch.tensor([]).view(-1, D)).cpu() if hasattr(torch, 'tensor') else torch.tensor([])
            cov = float('nan')

        # Accumulate
        acc[name]['model'] += (float(model_t) if model_t is not None else float('nan'))
        acc[name]['proj'] += (float(proj_t) if proj_t is not None else float('nan'))
        acc[name]['sampling'] += (float(samp_t) if samp_t is not None else float('nan'))
        acc[name]['coverage'] += (float(cov) if cov is not None else float('nan'))
        acc[name]['count'] += 1

        if trial == 0:
            first_samples[name] = samples

            # For DDPM, also measure external projection to create DDPM (proj.) entry
        if name == 'DDPM':
            ext_proj_time = float('nan')
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                samples_plain_projected, _, _ = tr.projector.project(torch.tensor(samples).cpu())
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                ext_proj_time = float(t1 - t0)
                samples_plain_projected = samples_plain_projected.cpu() if hasattr(samples_plain_projected, 'cpu') else samples_plain_projected
            except Exception:
                try:
                    samples_plain_projected, _, _ = tr.projector.project(torch.tensor(samples).cpu())
                    samples_plain_projected = samples_plain_projected.cpu()
                except Exception:
                    samples_plain_projected = torch.tensor([])

            # compute metrics for projected DDPM
            try:
                sp_tensor = filter_valid_samples(samples_plain_projected.detach().clone() if isinstance(samples_plain_projected, torch.Tensor) else torch.tensor(samples_plain_projected)).view(-1, D).cpu()
                cov_p = _embedding_coverage(sp_tensor)
            except Exception:
                cov_p = float('nan')

            # model time for DDPM (proj.) is same as DDPM model time for this trial
            acc['DDPM (proj.)']['model'] += (float(model_t) if model_t is not None else float('nan'))
            acc['DDPM (proj.)']['proj'] += ext_proj_time
            # sampling time for projected case considered as base sampling + external projection
            base_samp = float(samp_t) if samp_t is not None else float('nan')
            acc['DDPM (proj.)']['sampling'] += (base_samp + (ext_proj_time if not np.isnan(ext_proj_time) else 0.0))
            acc['DDPM (proj.)']['coverage'] += (float(cov_p) if cov_p is not None else float('nan'))
            acc['DDPM (proj.)']['count'] += 1
            if trial == 0:
                first_samples['DDPM (proj.)'] = samples_plain_projected

# After trials, compute averages and prepare final tensors for plotting/metrics
avg = {}
for name, stats in acc.items():
    cnt = stats['count'] if stats['count'] > 0 else 1
    avg[name] = {
        'model': stats['model'] / cnt,
        'proj': stats['proj'] / cnt,
        'sampling': stats['sampling'] / cnt,
        'coverage': stats['coverage'] / cnt,
    }

# Prepare final sample tensors (from first trial) for visualization and comparison
# Prepare final sample tensors (from first trial) for visualization and comparison
def _to_tensor_or_empty(x):
    try:
        if isinstance(x, torch.Tensor):
            x = x.detach().clone()
        else:
            x = torch.tensor(x)
        return filter_valid_samples(x.view(-1, D)).cpu()
    except Exception:
        return filter_valid_samples(torch.tensor([]).view(-1, D)).cpu()

samples_PDM_tensor    = _to_tensor_or_empty(first_samples.get('PDM', []))
samples_PIDM_tensor   = _to_tensor_or_empty(first_samples.get('PIDM', []))
samples_lifted_tensor = _to_tensor_or_empty(first_samples.get('Lifted', []))
samples_plain_tensor  = _to_tensor_or_empty(first_samples.get('DDPM', []))
samples_plain_projected_tensor = _to_tensor_or_empty(first_samples.get('DDPM (proj.)', []))

# --- Classifier-based accuracy on generated samples ---
# --- Classifier-based accuracy on generated samples ---
try:
    classifier = load_mnist_classifier("models/mnist_classifier.pth", device=device)
    import numpy as np
    import matplotlib.pyplot as plt
    method_names = [
        ("PDM", samples_PDM_tensor),
        ("PIDM", samples_PIDM_tensor),
        ("Lifted", samples_lifted_tensor),
        ("DDPM", samples_plain_tensor),
        ("Projected DDPM", samples_plain_projected_tensor),
    ]
    # Add the data (true_tensor) as a method for plotting
    method_names_with_data = method_names + [("Data", true_tensor)]
    bincounts = {}
    for name, tensor in method_names_with_data:
        preds = classifier_accuracy(tensor, classifier, device=device)
        bincount = np.bincount(preds, minlength=10)
        bincounts[name] = bincount
        total = bincount.sum()
        print(f"Classifier predictions for {name} (N={total}):")
        for i, count in enumerate(bincount):
            print(f"  {i}: {count} ({count/total*100:.1f}%)")

    print("Skipping bar plot output: classifier_grouped_bar_chart")

    # =========================
    # Label-distribution distance to Data (recommended: Jensen–Shannon distance)
    # =========================

    def _to_prob(counts, eps=1e-12):
        counts = np.asarray(counts, dtype=np.float64)
        s = counts.sum()
        if s <= 0:
            p = np.ones_like(counts) / len(counts)
        else:
            p = counts / s
        # smooth to avoid log(0), then renormalize
        p = p + eps
        return p / p.sum()

    def js_distance_from_counts(counts_p, counts_q, eps=1e-12):
        """
        Jensen–Shannon *distance* between two discrete histograms.
        Returns sqrt(JS divergence). Values in [0, sqrt(log 2)] if using natural logs.
        (For convenience, we also print a normalized version in [0,1].)
        """
        p = _to_prob(counts_p, eps=eps)
        q = _to_prob(counts_q, eps=eps)
        m = 0.5 * (p + q)

        def kl(a, b):
            return np.sum(a * np.log(a / b))

        js = 0.5 * kl(p, m) + 0.5 * kl(q, m)
        return float(np.sqrt(js))

    # Compute distances to the reference ("Data")
    ref_name = "Data"
    if ref_name not in bincounts:
        raise KeyError(f"Expected '{ref_name}' in bincounts, got: {list(bincounts.keys())}")

    ref_counts = bincounts[ref_name]

    # sqrt(log 2) is the max JS distance when using natural logs
    js_max = float(np.sqrt(np.log(2.0)))

    print("\n=== Label-distribution distance to Data (Jensen–Shannon distance) ===")
    results_js = []
    for name, _ in method_names:  # exclude "Data" itself
        d = js_distance_from_counts(bincounts[name], ref_counts, eps=1e-12)
        d_norm = d / js_max  # normalized to [0,1]
        results_js.append((name, d, d_norm))

    # sort best (closest to Data) -> worst
    results_js.sort(key=lambda t: t[1])

    for name, d, d_norm in results_js:
        print(f"{name:>14s}:  JS-dist = {d:.6f}   (normalized {d_norm:.3f} in [0,1])")

    # --- Precompute REAL stats once ---
    feats_real = get_classifier_features(true_tensor, classifier, device=device)
    feats_real = feats_real.astype(np.float64)
    mu_real = feats_real.mean(axis=0)
    sigma_real = np.cov(feats_real, rowvar=False)

    def fid_for_samples(gen_tensor, name):
        feats_gen = get_classifier_features(gen_tensor, classifier, device=device).astype(np.float64)
        mu_gen = feats_gen.mean(axis=0)
        sigma_gen = np.cov(feats_gen, rowvar=False)
        fid = compute_fid(mu_gen, sigma_gen, mu_real, sigma_real)
        print(f"FID (classifier features) for {name}: {fid:.4f}")
        return fid

    print("\n--- FID (classifier features) ---")
    fid_for_samples(samples_PDM_tensor, "PDM")
    fid_for_samples(samples_PIDM_tensor, "PIDM")
    fid_for_samples(samples_lifted_tensor, "Lifted")
    fid_for_samples(samples_plain_tensor, "DDPM")
    fid_for_samples(samples_plain_projected_tensor, "Projected DDPM")
except Exception as e:
    print("Could not load classifier or classify generated samples:", e)

# Coverage
print(f"Coverage (PDM):    {_embedding_coverage(samples_PDM_tensor)}")
print(f"Coverage (PIDM):   {_embedding_coverage(samples_PIDM_tensor)}")
print(f"Coverage (Lifted): {_embedding_coverage(samples_lifted_tensor)}")
print(f"Coverage (DDPM):   {_embedding_coverage(samples_plain_tensor)}")
print(f"Coverage (Projected DDPM): {_embedding_coverage(samples_plain_projected_tensor)}")
# Visualize a grid of generated images for each method
os.makedirs("results/mnist/", exist_ok=True)
def save_grid(samples, fname):
    # Ensure numpy array for indexing
    try:
        samples = np.asarray(samples)
    except Exception:
        pass
    # Plot up to 25 samples in a 5x5 grid
    n = min(25, len(samples)) if hasattr(samples, '__len__') else 25
    fig, axs = plt.subplots(5, 5, figsize=(5, 5))
    for i, ax in enumerate(axs.flat):
        if i >= n:
            ax.axis('off')
            continue
        img = samples[i].reshape(28, 28)
        ax.imshow(img, cmap='gray')
        # Hide ticks and frames for tight visual packing
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
    # Reduce spacing between images
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.05, hspace=0.05)
    fig.savefig(fname, dpi=200)
    plt.close(fig)

save_grid(samples_PDM, f'results/mnist/PDM_samples_{num_samples}.png')
save_grid(samples_plain, f'results/mnist/DDPM_samples_{num_samples}.png')
save_grid(samples_PIDM, f'results/mnist/PIDM_samples_{num_samples}.png')
save_grid(samples_lifted, f'results/mnist/lifted_sigma_{noise_level}_samples_{num_samples}.png')
save_grid(data_points_plain.cpu().numpy(), f'results/mnist/data_samples_{num_samples}.png')

import numpy as _np


def _total_model_time(tr):
    if tr is None:
        return _np.nan
    if hasattr(tr, 'total_model_forward_sample_time'):
        try:
            return float(getattr(tr, 'total_model_forward_sample_time'))
        except Exception:
            pass
    if hasattr(tr, 'model_forward_times'):
        lst = getattr(tr, 'model_forward_times') or []
        try:
            return float(_np.sum([float(x) for x in lst])) if len(lst) > 0 else _np.nan
        except Exception:
            return _np.nan
    return _np.nan

def _total_proj_time(tr):
    if tr is None:
        import time
        return _np.nan
    if hasattr(tr, 'total_projection_sample_time'):
        try:
            return float(getattr(tr, 'total_projection_sample_time'))
        except Exception:
            pass
    if hasattr(tr, 'projection_sample_times'):
        lst = getattr(tr, 'projection_sample_times') or []
        try:
            return float(_np.sum([float(x) for x in lst])) if len(lst) > 0 else _np.nan
        except Exception:
            return _np.nan
    return _np.nan

method_names = ['Lifted', 'PDM', 'DDPM', 'DDPM (proj.)', 'PIDM']
trainers_map = {
    'Lifted': locals().get('trainer_lifted', None),
    'PDM':   locals().get('trainer_PDM', None),
    'DDPM':  locals().get('trainer_plain', None),
    # DDPM (proj.) will be represented visually but we don't mutate trainers for it
    'DDPM (proj.)': None,
    'PIDM':  locals().get('trainer_PIDM', None),
}

# Measure an external projection of the DDPM samples (timed) for the DDPM_projected bar.
proj_time_plain_projection = float('nan')
try:
    if trainer_plain is not None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        samples_plain_projected, _, _ = trainer_plain.projector.project(torch.tensor(samples_plain).cpu())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        proj_time_plain_projection = float(t1 - t0)
        samples_plain_projected = samples_plain_projected.cpu() if hasattr(samples_plain_projected, 'cpu') else samples_plain_projected
except Exception:
    try:
        # Best-effort fallback (no timing)
        samples_plain_projected, _, _ = trainer_plain.projector.project(torch.tensor(samples_plain).cpu())
        samples_plain_projected = samples_plain_projected.cpu()
    except Exception:
        samples_plain_projected = torch.tensor([])
model_vals = []
proj_vals = []
other_vals = []

# Average timings across n_trials. Keep first-trial samples for visual outputs and
# run additional trials to collect timing/metric data. We try to avoid mutating
# trainer internals permanently; per-trial resets are performed when possible.
def _compute_avg_stats(method_names, trainers_map, n_trials):
    # Initialize accumulators using the existing (first) run already performed above
    stats = {}
    for name in method_names:
        tr = trainers_map.get(name)
        m0 = _total_model_time(tr)
        p0 = _total_proj_time(tr)
        s0 = getattr(tr, 'sampling_time', _np.nan) if tr is not None else _np.nan
        stats[name] = {
            'm_sum': 0.0 if _np.isnan(m0) else float(m0),
            'p_sum': 0.0 if _np.isnan(p0) else float(p0),
            's_sum': 0.0 if _np.isnan(s0) else float(s0),
            'count': 1 if not (_np.isnan(m0) and _np.isnan(p0) and _np.isnan(s0)) else 0,
            'external_proj_list': []
        }

    # Include any external projection time measured earlier for DDPM_projected (first trial)
    if 'proj_time_plain_projection' in locals():
        try:
            if not _np.isnan(proj_time_plain_projection):
                stats.setdefault('DDPM_projected', {'external_proj_list': []}).setdefault('external_proj_list', []).append(float(proj_time_plain_projection))
        except Exception:
            pass

    # per-method sample kwargs if special modes required
    sample_kwargs_map = {'PDM': {'PDM': True}}

    # Run additional trials (best-effort). We will reuse trainer objects and reset
    # timing lists before each trial when possible.
    for t in range(max(0, n_trials - 1)):
        for name in method_names:
            tr = trainers_map.get(name)
            if tr is None:
                continue
            # clear per-trial timing lists if present
            for attr in ('model_forward_times', 'projection_sample_times', 'projection_times'):
                try:
                    setattr(tr, attr, [])
                except Exception:
                    pass
            try:
                tr.total_model_forward_sample_time = 0.0
            except Exception:
                pass
            try:
                tr.total_projection_sample_time = 0.0
            except Exception:
                pass

            # Choose kwargs for sampling (PDM case)
            kwargs = sample_kwargs_map.get(name, {})

            # Invoke sampling (best-effort). Keep sampled output if needed for external projection timing.
            try:
                with torch.no_grad():
                    # Use the trainer instance directly
                    res = tr.sample(num_samples=num_eval_samples, **kwargs)
            except Exception:
                # If a trial fails, skip it
                continue

            # After trial, collect measured times from trainer
            m = _total_model_time(tr)
            p = _total_proj_time(tr)
            s = getattr(tr, 'sampling_time', _np.nan)
            if not _np.isnan(m):
                stats[name]['m_sum'] += float(m)
            if not _np.isnan(p):
                stats[name]['p_sum'] += float(p)
            if not _np.isnan(s):
                stats[name]['s_sum'] += float(s)
            stats[name]['count'] += 1

            # For DDPM_projected, measure external projection time for the plain DDPM output
            if name == 'DDPM':
                try:
                    samples_out = res[0] if isinstance(res, (list, tuple)) and len(res) > 0 else res
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    samples_plain_projected_tmp, _, _ = trainer_plain.projector.project(torch.tensor(samples_out).cpu())
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    stats.setdefault('DDPM_projected', {}).setdefault('external_proj_list', []).append(float(t1 - t0))
                except Exception:
                    pass

    # Compute averages
    avg_stats = {}
    # First compute for non-DDPM_projected
    for name in method_names:
        if name == 'DDPM_projected':
            continue
        st = stats.get(name, None)
        if st is None or st.get('count', 0) == 0:
            avg_stats[name] = {'m': _np.nan, 'p': _np.nan, 's': _np.nan}
            continue
        cnt = float(st['count'])
        m_avg = float(st['m_sum']) / cnt if st['m_sum'] != 0 else (st['m_sum'] / cnt if st['m_sum'] == 0 and cnt > 0 else _np.nan)
        p_avg = float(st['p_sum']) / cnt if st['p_sum'] != 0 else (st['p_sum'] / cnt if st['p_sum'] == 0 and cnt > 0 else _np.nan)
        s_avg = float(st['s_sum']) / cnt if st['s_sum'] != 0 else (st['s_sum'] / cnt if st['s_sum'] == 0 and cnt > 0 else _np.nan)
        avg_stats[name] = {'m': m_avg, 'p': p_avg, 's': s_avg}

    # Now compute DDPM_projected using averaged DDPM model time + averaged external projection
    ext_list = stats.get('DDPM_projected', {}).get('external_proj_list', [])
    ext_mean = float(_np.mean(ext_list)) if len(ext_list) > 0 else _np.nan
    ddpm_base = avg_stats.get('DDPM', {'m': _np.nan, 'p': _np.nan, 's': _np.nan})
    avg_stats['DDPM_projected'] = {
        'm': ddpm_base.get('m', _np.nan),
        'p': ext_mean,
        's': (ddpm_base.get('s', _np.nan) + ext_mean) if _np.isfinite(ddpm_base.get('s', _np.nan)) and _np.isfinite(ext_mean) else _np.nan,
    }
    # Also expose the projected DDPM stats under the display name used elsewhere
    try:
        avg_stats['DDPM (proj.)'] = avg_stats.get('DDPM_projected', {'m': _np.nan, 'p': _np.nan, 's': _np.nan})
    except Exception:
        pass

    return avg_stats


avg_stats = _compute_avg_stats(method_names, trainers_map, n_trials)

for name in method_names:
    stats = avg_stats.get(name, {'m': _np.nan, 'p': _np.nan, 's': _np.nan})
    m = stats['m']
    p = stats['p']
    s = stats['s']
    if name == 'DDPM':
        p = 0.0
    if _np.isfinite(s) and _np.isfinite(m) and _np.isfinite(p):
        other = max(0.0, float(s) - float(m) - float(p))
    else:
        other = _np.nan
    model_vals.append(m)
    proj_vals.append(p)
    other_vals.append(other)

outdir = 'results/mnist'
os.makedirs(outdir, exist_ok=True)

# ---- Training time breakdown plot (uses checkpoint-loaded epoch timing breakdowns) ----
def _avg_training_components(tr):
    import numpy as _np

    # Return averaged (model_forward, project, backprop, other_rest)
    if tr is None:
        return _np.nan, _np.nan, _np.nan, _np.nan
    etb = getattr(tr, 'epoch_timing_breakdowns', None)
    if not etb:
        return _np.nan, _np.nan, _np.nan, _np.nan
    model_vals = [_np.nan if d is None else d.get('model_forward', _np.nan) for d in etb]
    proj_vals = [_np.nan if d is None else d.get('project', _np.nan) for d in etb]
    backprop_vals = [_np.nan if d is None else d.get('backprop', 0.0) for d in etb]
    # Keep 'other' separate from backprop so we can plot backprop explicitly
    other_rest_vals = [_np.nan if d is None else (d.get('other', 0.0) + d.get('sampling_to_t0', 0.0)) for d in etb]
    try:
        m = float(_np.nanmean(model_vals))
    except Exception:
        m = _np.nan
    try:
        p = float(_np.nanmean(proj_vals))
    except Exception:
        p = _np.nan
    try:
        bp = float(_np.nanmean(backprop_vals))
    except Exception:
        bp = _np.nan
    try:
        o = float(_np.nanmean(other_rest_vals))
    except Exception:
        o = _np.nan
    return m, p, bp, o

train_method_names = ['Lifted', 'PDM', 'DDPM', 'PIDM']
trainers_map_small = {
    'Lifted': locals().get('trainer_lifted', None),
    'PDM': locals().get('trainer_PDM', None),
    'DDPM': locals().get('trainer_plain', None),
    'PIDM': locals().get('trainer_PIDM', None),
}
model_t = []
proj_t = []
backprop_t = []
other_t = []
for name in train_method_names:
    tr = trainers_map_small.get(name)
    m, p, bp, o = _avg_training_components(tr)
    model_t.append(m)
    proj_t.append(p)
    backprop_t.append(bp)
    other_t.append(o)

print("Skipping bar plot output: mnist training_time_breakdown")

# --- Build and save metrics table (LaTeX + CSV) ---
try:
    # Build general_metrics using averaged sampling times (avg) and training breakdowns
    # Compute training-time totals from averaged per-epoch components
    sample_time_map = {}
    try:
        for name in avg.keys():
            sample_time_map[name] = avg.get(name, {}).get('sampling', float('nan'))
    except Exception:
        sample_time_map = {name: float('nan') for name in avg.keys()}

    # Compute averaged training components like other plotting scripts
    def _avg_training_components(tr):
        import numpy as _np
        if tr is None:
            return _np.nan, _np.nan, _np.nan, _np.nan
        etb = getattr(tr, 'epoch_timing_breakdowns', None)
        if not etb:
            return _np.nan, _np.nan, _np.nan, _np.nan
        model_vals = [_np.nan if d is None else d.get('model_forward', _np.nan) for d in etb]
        proj_vals = [_np.nan if d is None else d.get('project', _np.nan) for d in etb]
        backprop_vals = [_np.nan if d is None else d.get('backprop', 0.0) for d in etb]
        other_rest_vals = [_np.nan if d is None else (d.get('other', 0.0) + d.get('sampling_to_t0', 0.0)) for d in etb]
        try:
            m = float(_np.nanmean(model_vals))
        except Exception:
            m = _np.nan
        try:
            p = float(_np.nanmean(proj_vals))
        except Exception:
            p = _np.nan
        try:
            bp = float(_np.nanmean(backprop_vals))
        except Exception:
            bp = _np.nan
        try:
            o = float(_np.nanmean(other_rest_vals))
        except Exception:
            o = _np.nan
        return m, p, bp, o

    train_method_names = ['Lifted', 'PDM', 'DDPM', 'PIDM']
    trainers_map_small = {
        'Lifted': locals().get('trainer_lifted', None),
        'PDM': locals().get('trainer_PDM', None),
        'DDPM': locals().get('trainer_plain', None),
        'PIDM': locals().get('trainer_PIDM', None),
    }
    model_t = []
    proj_t = []
    backprop_t = []
    other_t = []
    for name in train_method_names:
        tr = trainers_map_small.get(name)
        m, p, bp, o = _avg_training_components(tr)
        model_t.append(m)
        proj_t.append(p)
        backprop_t.append(bp)
        other_t.append(o)

    train_time_map = {}
    try:
        for i, name in enumerate(train_method_names):
            comps = [model_t[i] if i < len(model_t) else float('nan'),
                        proj_t[i] if i < len(proj_t) else float('nan'),
                        backprop_t[i] if i < len(backprop_t) else float('nan'),
                        other_t[i] if i < len(other_t) else float('nan')]
            if all([not np.isfinite(c) for c in comps]):
                total = float('nan')
            else:
                total = float(sum([float(c) for c in comps if np.isfinite(c)]))
            train_time_map[name] = total
        train_time_map['DDPM (proj.)'] = train_time_map.get('DDPM', float('nan'))
    except Exception:
        train_time_map = {name: float('nan') for name in train_method_names}

    # Compute FID for each method
    fid_map = {}
    try:
        fid_map['PDM'] = fid_for_samples(samples_PDM_tensor, "PDM")
        fid_map['PIDM'] = fid_for_samples(samples_PIDM_tensor, "PIDM")
        fid_map['Lifted'] = fid_for_samples(samples_lifted_tensor, "Lifted")
        fid_map['DDPM'] = fid_for_samples(samples_plain_tensor, "DDPM")
        fid_map['Projected DDPM'] = fid_for_samples(samples_plain_projected_tensor, "Projected DDPM")
    except Exception:
        fid_map = {}

    # Create JSD map from results_js
    jsd_map = {}
    try:
        for name, d, d_norm in results_js:
            jsd_map[name] = d
    except Exception:
        jsd_map = {}

    general_metrics = {}
    for method in avg.keys():
        vals = avg.get(method, {})
        samp_val = sample_time_map.get(method, float('nan'))
        samp_entry = float(samp_val) if np.isfinite(samp_val) else 'n/a'
        # Map training time name for projected variant
        train_name = method if method != 'DDPM_projected' else 'DDPM (proj.)'
        train_val = train_time_map.get(train_name if train_name in train_time_map else method, float('nan'))
        train_entry = float(train_val) if np.isfinite(train_val) else 'n/a'
        
        # Map method names for FID/JSD lookups
        # avg.keys() uses "DDPM (proj.)" but FID/JSD maps use "Projected DDPM"
        if method == 'DDPM (proj.)':
            fid_key = 'Projected DDPM'
            jsd_key = 'Projected DDPM'
        else:
            fid_key = method
            jsd_key = method
        
        # Format FID in scientific notation
        fid_val = fid_map.get(fid_key, float('nan'))
        if isinstance(fid_val, float) and np.isfinite(fid_val):
            try:
                fid_str = f"{fid_val:.3e}"
                mant, exp = fid_str.split('e')
                fid_entry = f"${mant}\\times10^{{{int(exp)}}}$"
            except Exception:
                fid_entry = f"{fid_val:.3e}"
        else:
            fid_entry = 'n/a'
        
        general_metrics[method] = {
            'Train time (s/epoch)': train_entry,
            'Sampling time (s)': samp_entry,
            'COV': float(vals.get('coverage', 0.0)) if vals is not None else 0.0,
            'FID': fid_entry,
            'Class JSD': jsd_map.get(jsd_key, float('nan')),
        }

    if compute_table:
        from plotting.paper_tables import write_mnist_metrics_table
        outdir = 'results/mnist'
        os.makedirs(outdir, exist_ok=True)

        rows = []
        for key, label in [
            ('Lifted', r'$p_{\sigma}$ (ours)'),
            ('PDM', 'PDM'),
            ('DDPM', 'DDPM'),
            ('PIDM', 'PIDM'),
            ('DDPM (proj.)', 'DDPM (proj.)'),
        ]:
            vals = general_metrics[key]
            fid_key = 'Projected DDPM' if key == 'DDPM (proj.)' else key
            rows.append(
                {
                    'method': label,
                    'Train time (s/epoch)': vals['Train time (s/epoch)'],
                    'Sampling time (s)': vals['Sampling time (s)'],
                    'COV': float(vals['COV']),
                    'FID': float(fid_map.get(fid_key, float('nan'))),
                    'Class JSD': float(jsd_map.get(fid_key, float('nan'))),
                }
            )

        write_mnist_metrics_table(
            rows,
            out_tex_path=os.path.join(outdir, 'metrics_table.tex'),
            caption=f"MNIST metrics at $\\sigma = {noise_level}$ with {num_samples:,} training samples.",
        )
    else:
        raise RuntimeError("MNIST metrics table generation is disabled.")
except Exception as e:
    raise RuntimeError("Failed to save metrics table for MNIST") from e

print("\n--- Model Trainable Parameter Counts ---")
try:
    print(f"Lifted Model (trainer_lifted.denoiser): {count_trainable_params(trainer_lifted.denoiser):,}")
except Exception as e:
    print(f"Could not count Lifted model params: {e}")
try:
    print(f"Traditional DDPM Model (trainer_plain.denoiser): {count_trainable_params(trainer_plain.denoiser):,}")
except Exception as e:
    print(f"Could not count Traditional DDPM params: {e}")
try:
    print(f"Projected Diffusion Model (trainer_PDM.denoiser): {count_trainable_params(trainer_PDM.denoiser):,}")
except Exception as e:
    print(f"Could not count PDM params: {e}")
try:
    print(f"Physics-Informed Diffusion Model (trainer_PIDM.denoiser): {count_trainable_params(trainer_PIDM.denoiser):,}")
except Exception as e:
    print(f"Could not count PIDM params: {e}")

# # Optionally: PCA scatter for samples
# from sklearn.decomposition import PCA
# n_components = 2
# pca = PCA(n_components=n_components)
# proj_2d = pca.fit_transform(samples_lifted.cpu().numpy())
# plt.figure(figsize=(6,6))
# plt.scatter(proj_2d[:,0], proj_2d[:,1], s=1, alpha=0.5)
# plt.title(f"Lifted DDPM MNIST digits (sum={target_sum}) - PCA 2D")
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.savefig("results/mnist/pca_lifted.pdf")
# plt.close()
