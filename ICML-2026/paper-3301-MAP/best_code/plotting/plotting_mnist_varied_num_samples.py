import os

# Keep these harmless guards for environments that preload torchvision elsewhere.
os.environ.setdefault("TORCHVISION_DISABLE_NMS_EXPORT", "1")
os.environ.setdefault("TORCHVISION_DISABLE_META_REGISTRATIONS", "1")

import sys
import re
import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.linalg

# Move to the project root (one level up from current file)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

from train_mnist_classifier import SimpleMNISTClassifier
from datasets import MNISTFixedSumDataset
from trainers import DDPMTrainer
from utils.constraints import FixedSumProjector
from utils.metrics import coverage, filter_valid_samples


def _save_pdf_png(fig, output_path, **kwargs):
    fig.savefig(output_path, **kwargs)
    if output_path.lower().endswith(".pdf"):
        fig.savefig(output_path[:-4] + ".png", **kwargs)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--num_eval_samples", type=int, default=10000, help="Accepted for wrapper compatibility; MNIST evaluation is forced to 10000")
args, _ = parser.parse_known_args()

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
    
    x = samples if isinstance(samples, torch.Tensor) else torch.tensor(samples)
    if x.dtype != torch.float32:
        x = x.float()
    if x.numel() > 0:
        xmax = float(x.max().detach().cpu())
        if xmax > 1.0:
            x = x / 255.0
    
    if x.dim() == 2:
        if x.shape[1] == 28 * 28:
            x = x.view(-1, 1, 28, 28)
        else:
            raise ValueError(f"Expected 2D input with 784 features, got {x.shape[1]}")
    elif x.dim() == 3:
        x = x.unsqueeze(1)
    elif x.dim() == 4:
        if x.shape[-1] == 1 and x.shape[1] != 1:
            x = x.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"Unexpected samples shape: {tuple(x.shape)}")
    
    x = x.to(device)
    features = []
    classifier.eval()
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            batch = x[i:i + batch_size]
            _, feats = classifier(batch, return_features=True)
            features.append(feats.detach().cpu())
    
    feats = torch.cat(features, dim=0).numpy()
    return feats

def compute_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
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
        if x.numel() > 0:
            x_min, x_max = float(x.min()), float(x.max())
            if x_max > 1.0 or x_min < 0.0 or x_max > 1.01 or x_min < -0.01:
                x = (x - x_min) / (x_max - x_min + 1e-8)
    
    if x.dim() == 2:
        if x.shape[1] == 28 * 28:
            x = x.view(-1, 1, 28, 28)
        else:
            raise ValueError(f"Expected 2D input with 784 features, got {x.shape[1]}")
    elif x.dim() == 3:
        x = x.unsqueeze(1)
    elif x.dim() == 4:
        if x.shape[-1] == 1 and x.shape[1] != 1:
            x = x.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"Unexpected samples shape: {tuple(x.shape)}")
    
    x = x.to(device)
    with torch.no_grad():
        logits = classifier(x)
        preds = logits.argmax(dim=1)
    return preds.cpu().numpy()

def _to_prob(counts, eps=1e-12):
    counts = np.asarray(counts, dtype=np.float64)
    s = counts.sum()
    if s <= 0:
        p = np.ones_like(counts) / len(counts)
    else:
        p = counts / s
    p = p + eps
    return p / p.sum()

def js_distance_from_counts(counts_p, counts_q, eps=1e-12):
    """Jensen–Shannon distance between two discrete histograms."""
    p = _to_prob(counts_p, eps=eps)
    q = _to_prob(counts_q, eps=eps)
    m = 0.5 * (p + q)

    def kl(a, b):
        return np.sum(a * np.log(a / b))

    js = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    return float(np.sqrt(js))

MNIST_TEST_SET_SIZE = 10000
if args.num_eval_samples != MNIST_TEST_SET_SIZE:
    print(f"Ignoring --num_eval_samples={args.num_eval_samples}; MNIST evaluation uses the full {MNIST_TEST_SET_SIZE} sample test set.")
num_eval_samples = MNIST_TEST_SET_SIZE  # samples generated per trial
target_sum = 100.0
hidden_dim = 1024
time_embed_dim = 64
timesteps = 250
time_embed_choice = 'default'
epochs = 1000
random_seed = args.seed
trials = 3

torch.manual_seed(random_seed)
np.random.seed(random_seed)

# Coverage convention: prefer embedding (classifier) coverage when available.
# Coverage is embedding-only; if the classifier is unavailable, coverage is NaN.

# Allow evaluating multiple fixed sigmas; default includes 0.01
fixed_sigma_list = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
models_dir = os.path.join('models', 'mnist')


def _attach_time_embed_if_needed(denoiser, state_dict, device):
    """Placeholder function - time embedding utilities have been removed."""
    pass
def _discover_available_for_sigma(sig):
    pattern = re.compile(rf"model_DDPM_epoch_{epochs}_num_samples_(\d+)_noise_level_{sig}_time_{time_embed_choice}_seed_{random_seed}\.pth")
    found = []
    for fname in os.listdir(models_dir):
        m = pattern.match(fname)
        if m:
            n = int(m.group(1))
            found.append((n, os.path.join(models_dir, fname)))
    found.sort(key=lambda x: x[0])
    return found

def _cached_subset(num_samples, noise_level, lifted, seed, device, data_root="./data"):
    fname = f"fixedsum_mnist_train_n{num_samples}_nl{noise_level}_lifted{int(lifted)}_seed{seed}.pt"
    path = os.path.join(data_root, "cache", fname)
    blob = torch.load(path, map_location=device)
    X = blob["images"].to(torch.float32)
    return X

def _mnist_test_set(noise_level, lifted, device):
    dataset = MNISTFixedSumDataset(
        device=device,
        pixel_sum=target_sum,
        noise_level=noise_level,
        lifted=lifted,
        train=False,
    )
    if len(dataset) != MNIST_TEST_SET_SIZE:
        raise ValueError(f"Expected MNIST test set to contain {MNIST_TEST_SET_SIZE} samples, got {len(dataset)}")
    return torch.stack([dataset[i] for i in range(MNIST_TEST_SET_SIZE)])

# Reference baseline dataset: full zero-noise MNIST test set.
baseline_cached = _mnist_test_set(noise_level=0.0, lifted=False, device=device)
D = int(baseline_cached.shape[1])
true_tensor = filter_valid_samples(baseline_cached.view(-1, D)).cpu()

# Load classifier once for all evaluations
try:
    classifier = load_mnist_classifier("models/mnist_classifier.pth", device=device)
    print("Classifier loaded successfully")
    # Precompute REAL data stats once
    feats_real = get_classifier_features(true_tensor, classifier, device=device)
    feats_real = feats_real.astype(np.float64)
    mu_real = feats_real.mean(axis=0)
    sigma_real = np.cov(feats_real, rowvar=False)
    # Get reference class distribution
    preds_real = classifier_accuracy(true_tensor, classifier, device=device)
    ref_counts = np.bincount(preds_real, minlength=10)
    js_max = float(np.sqrt(np.log(2.0)))
    classifier_loaded = True
except Exception as e:
    print(f"Could not load classifier: {e}")
    classifier_loaded = False

# Coverage is always measured against the zero-noise MNIST reference set.
coverage_ref_tensor = true_tensor
if classifier_loaded:
    feats_real_for_metrics = get_classifier_features(coverage_ref_tensor, classifier, device=device).astype(np.float64)
    mu_real_for_metrics = feats_real_for_metrics.mean(axis=0)
    sigma_real_for_metrics = np.cov(feats_real_for_metrics, rowvar=False)
    preds_real_for_metrics = classifier_accuracy(coverage_ref_tensor, classifier, device=device)
    ref_counts_for_metrics = np.bincount(preds_real_for_metrics, minlength=10)


def _embedding_coverage(samples_tensor):
    if not classifier_loaded:
        return float('nan')
    feats_gen = get_classifier_features(samples_tensor, classifier, device=device).astype(np.float64)
    return float(coverage(feats_real_for_metrics, feats_gen))

results_by_sigma = {}

for fixed_sigma in fixed_sigma_list:
    available = _discover_available_for_sigma(fixed_sigma)
    if len(available) == 0:
        print(f"Warning: No checkpoints found in {models_dir} for sigma={fixed_sigma} with num_samples pattern. Skipping.")
        continue

    # Filter to only test specific num_samples values
    allowed_num_samples = {100, 1000, 10000}
    available = [(n, p) for n, p in available if n in allowed_num_samples]
    if len(available) == 0:
        print(f"Warning: No checkpoints found for sigma={fixed_sigma} with allowed num_samples {allowed_num_samples}. Skipping.")
        continue

    num_train_list = [n for n, _ in available]

    # Accumulators for lifted model across num_samples
    coverage_list = []
    coverage_std_list = []
    FID_list = []
    FID_std_list = []
    JS_list = []
    JS_std_list = []

    # Baseline DDPM accumulators (varying per num_samples)
    coverage_plain_list = []
    coverage_plain_std_list = []
    FID_plain_list = []
    FID_plain_std_list = []
    JS_plain_list = []
    JS_plain_std_list = []

    for n_train, ckpt_path in available:
        # Cached training subset for this sigma and num_samples
        data_points_eval = _mnist_test_set(noise_level=fixed_sigma, lifted=True, device=device)
        D = int(data_points_eval.shape[1])
        ref_tensor = filter_valid_samples(data_points_eval.view(-1, D)).cpu()
        # Lifted trainer
        trainer_lifted = DDPMTrainer(
            data_points_eval.squeeze(),
            timesteps=timesteps,
            project_x0_sample=True,
            projector=FixedSumProjector(target_sum=target_sum),
            hidden_dim=hidden_dim,
            time_embed_dim=time_embed_dim,
            size=D,
            image=True,
        )

        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        _attach_time_embed_if_needed(trainer_lifted.denoiser, state_dict, device)
        has_timeembed = isinstance(state_dict, dict) and any(k.startswith('time_embed_module.') for k in state_dict.keys())
        if not has_timeembed and getattr(trainer_lifted.denoiser, 'time_embed_module', None) is not None:
            trainer_lifted.denoiser.time_embed_module = None
        try:
            trainer_lifted.load_checkpoint(ckpt_path, map_location=device, load_optimizer=False)
        except Exception:
            if isinstance(state_dict, dict):
                trainer_lifted.denoiser.load_state_dict(state_dict)
        trainer_lifted.denoiser.eval()

        # Trials
        trial_coverages = []
        trial_fids = []
        trial_js_dists = []
        for t in range(trials):
            with torch.no_grad():
                samples_lifted, _ = trainer_lifted.sample(num_samples=num_eval_samples)
            try:
                samples_lifted = trainer_lifted.projector.project(torch.tensor(samples_lifted).cpu())[0].cpu()
            except Exception:
                samples_lifted = torch.tensor(samples_lifted)
            samples_tensor = filter_valid_samples(torch.tensor(samples_lifted).view(-1, D)).cpu()
            try:
                trial_coverages.append(_embedding_coverage(samples_tensor))
            except Exception:
                trial_coverages.append(float('nan'))
            
            # Classifier-based metrics
            if classifier_loaded:
                try:
                    feats_gen = get_classifier_features(samples_tensor, classifier, device=device).astype(np.float64)
                    mu_gen = feats_gen.mean(axis=0)
                    sigma_gen = np.cov(feats_gen, rowvar=False)
                    fid_val = float(compute_fid(mu_gen, sigma_gen, mu_real_for_metrics, sigma_real_for_metrics))
                    trial_fids.append(fid_val)
                except Exception:
                    trial_fids.append(float('nan'))
                
                try:
                    preds_gen = classifier_accuracy(samples_tensor, classifier, device=device)
                    gen_counts = np.bincount(preds_gen, minlength=10)
                    js_val = float(js_distance_from_counts(gen_counts, ref_counts_for_metrics))
                    trial_js_dists.append(js_val)
                except Exception:
                    trial_js_dists.append(float('nan'))

        coverage_list.append(float(np.nanmean(np.array(trial_coverages))))
        coverage_std_list.append(float(np.nanstd(np.array(trial_coverages))))
        
        if classifier_loaded:
            fid_arr = np.array(trial_fids, dtype=np.float64)
            js_arr = np.array(trial_js_dists, dtype=np.float64)
            FID_list.append(float(np.nanmean(fid_arr)))
            FID_std_list.append(float(np.nanstd(fid_arr)))
            JS_list.append(float(np.nanmean(js_arr)))
            JS_std_list.append(float(np.nanstd(js_arr)))
        else:
            FID_list.append(float('nan'))
            FID_std_list.append(float('nan'))
            JS_list.append(float('nan'))
            JS_std_list.append(float('nan'))

        # Baseline projected DDPM for the same training size: try to load num_samples variant, fallback to generic
        ddpm_fname = f"model_DDPM_NONPROJECT_epoch_{epochs}_num_samples_{n_train}_noise_level_0.0_time_{time_embed_choice}_seed_{random_seed}.pth"
        ddpm_path = os.path.join(models_dir, ddpm_fname)
        if not os.path.exists(ddpm_path):
            ddpm_fname_fallback = f"model_DDPM_NONPROJECT_epoch_{epochs}_noise_level_0.0_time_{time_embed_choice}_seed_{random_seed}.pth"
            ddpm_path = os.path.join(models_dir, ddpm_fname_fallback)
            if not os.path.exists(ddpm_path):
                # last fallback without time/seed suffix
                ddpm_fname_fallback2 = f"model_DDPM_NONPROJECT_epoch_{epochs}_noise_level_0.0.pth"
                ddpm_path = os.path.join(models_dir, ddpm_fname_fallback2)

        data_points_true_cached = baseline_cached
        ref_plain_tensor = filter_valid_samples(data_points_true_cached.view(-1, D)).cpu()
        trainer_plain = DDPMTrainer(
            data_points_true_cached.squeeze(),
            timesteps=timesteps,
            project_x0_sample=False,
            projector=FixedSumProjector(target_sum=target_sum),
            hidden_dim=hidden_dim,
            time_embed_dim=time_embed_dim,
            size=D,
            image=True,
        )
        checkpoint_plain = torch.load(ddpm_path, map_location=device)
        state_plain = checkpoint_plain.get('model_state_dict', checkpoint_plain)
        _attach_time_embed_if_needed(trainer_plain.denoiser, state_plain if isinstance(state_plain, dict) else {}, device)
        has_timeembed_plain = isinstance(state_plain, dict) and any(k.startswith('time_embed_module.') for k in state_plain.keys())
        if not has_timeembed_plain and getattr(trainer_plain.denoiser, 'time_embed_module', None) is not None:
            trainer_plain.denoiser.time_embed_module = None
        try:
            trainer_plain.load_checkpoint(ddpm_path, map_location=device, load_optimizer=False)
        except Exception:
            if isinstance(state_plain, dict):
                trainer_plain.denoiser.load_state_dict(state_plain)
        trainer_plain.denoiser.eval()

        # Trials for baseline
        trial_coverages_plain = []
        trial_fids_plain = []
        trial_js_dists_plain = []
        for t in range(trials):
            with torch.no_grad():
                samples_plain, _ = trainer_plain.sample(num_samples=num_eval_samples)
                try:
                    samples_plain = trainer_plain.projector.project(torch.tensor(samples_plain).cpu())[0].cpu()
                except Exception:
                    samples_plain = torch.tensor(samples_plain)
            samples_plain_tensor = filter_valid_samples(torch.tensor(samples_plain).view(-1, D)).cpu()
            try:
                trial_coverages_plain.append(_embedding_coverage(samples_plain_tensor))
            except Exception:
                trial_coverages_plain.append(float('nan'))
            
            # Classifier-based metrics for baseline
            if classifier_loaded:
                try:
                    feats_gen = get_classifier_features(samples_plain_tensor, classifier, device=device).astype(np.float64)
                    mu_gen = feats_gen.mean(axis=0)
                    sigma_gen = np.cov(feats_gen, rowvar=False)
                    fid_val = float(compute_fid(mu_gen, sigma_gen, mu_real_for_metrics, sigma_real_for_metrics))
                    trial_fids_plain.append(fid_val)
                except Exception:
                    trial_fids_plain.append(float('nan'))
                
                try:
                    preds_gen = classifier_accuracy(samples_plain_tensor, classifier, device=device)
                    gen_counts = np.bincount(preds_gen, minlength=10)
                    js_val = float(js_distance_from_counts(gen_counts, ref_counts_for_metrics))
                    trial_js_dists_plain.append(js_val)
                except Exception:
                    trial_js_dists_plain.append(float('nan'))

        coverage_plain_list.append(float(np.nanmean(np.array(trial_coverages_plain))))
        coverage_plain_std_list.append(float(np.nanstd(np.array(trial_coverages_plain))))
        
        if classifier_loaded:
            fid_arr_plain = np.array(trial_fids_plain, dtype=np.float64)
            js_arr_plain = np.array(trial_js_dists_plain, dtype=np.float64)
            FID_plain_list.append(float(np.nanmean(fid_arr_plain)))
            FID_plain_std_list.append(float(np.nanstd(fid_arr_plain)))
            JS_plain_list.append(float(np.nanmean(js_arr_plain)))
            JS_plain_std_list.append(float(np.nanstd(js_arr_plain)))
        else:
            FID_plain_list.append(float('nan'))
            FID_plain_std_list.append(float('nan'))
            JS_plain_list.append(float('nan'))
            JS_plain_std_list.append(float('nan'))

    # Store results for this sigma
    results_by_sigma[fixed_sigma] = {
        'num_train_list': num_train_list,
        'coverage_mean': coverage_list,
        'coverage_std': coverage_std_list,
        'FID_mean': FID_list,
        'FID_std': FID_std_list,
        'JS_distance_mean': JS_list,
        'JS_distance_std': JS_std_list,
        'plain': {
            'coverage_mean': coverage_plain_list,
            'coverage_std': coverage_plain_std_list,
            'FID_mean': FID_plain_list,
            'FID_std': FID_plain_std_list,
            'JS_distance_mean': JS_plain_list,
            'JS_distance_std': JS_plain_std_list,
        },
    }


# Save metrics JSON
output_dir = os.path.join('results', 'mnist')
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, 'metrics_varied_num_samples.json'), 'w') as f:
    json.dump({'results_by_sigma': results_by_sigma}, f)

# Sort results by sigma for plotting
sorted_items = sorted(results_by_sigma.items(), key=lambda x: float(x[0]))
num_sigmas = len(sorted_items)

# Prepare colormaps if needed (for multiple sigmas)
if num_sigmas > 1:
    from matplotlib.colors import Normalize
    sigmas_numeric = np.array([float(sig) for sig, _ in sorted_items])
    norm_cov = Normalize(vmin=sigmas_numeric.min(), vmax=sigmas_numeric.max())
    cmap_cov = plt.cm.viridis
else:
    norm_cov = None
    cmap_cov = None


# 3-panel combined figure function
def build_combined_3panel_figure(output_dir, sorted_items, norm_cov, cmap_cov, num_sigmas):
    """Create a 1x3 figure with Coverage, FID, and JSD vs num_samples for all sigmas."""
    mpl.rcParams.update({
        "figure.figsize": (11.5, 3.8),
        "font.size": 14,
        "axes.labelsize": 15,
        "axes.titlesize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "lines.linewidth": 2.6,
        "lines.markersize": 7,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })
    
    line_width = 2.0
    marker_size = 5
    # Use consistent proportions for 3-panel figures
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
    
    # Panel labels
    panel_labels = ["Coverage", "FID", "Class JSD"]
    
    # Plot each sigma with color-coded hues
    for i, (sig, res) in enumerate(sorted_items):
        nlist = np.array(res['num_train_list'], dtype=np.int64)
        cov_arr = np.array(res['coverage_mean'], dtype=np.float64)
        cov_std_arr = np.array(res['coverage_std'], dtype=np.float64)
        fid_arr = np.array(res['FID_mean'], dtype=np.float64)
        fid_std_arr = np.array(res['FID_std'], dtype=np.float64)
        js_arr = np.array(res['JS_distance_mean'], dtype=np.float64)
        js_std_arr = np.array(res['JS_distance_std'], dtype=np.float64)
        
        color_i = cmap_cov(norm_cov(float(sig))) if norm_cov is not None else cmap_cov(0.5)
        
        # Coverage panel
        axes[0].plot(nlist, cov_arr, marker='o', markersize=marker_size, linestyle='-', 
                     color=color_i, label=f'σ={sig}' if num_sigmas > 1 else '$p_{\\sigma}$ (ours)')
        axes[0].fill_between(nlist, cov_arr - cov_std_arr, cov_arr + cov_std_arr, 
                             color=color_i, alpha=0.20)
        
        # FID panel
        axes[1].plot(nlist, fid_arr, marker='o', markersize=marker_size, linestyle='-', 
                     color=color_i)
        axes[1].fill_between(nlist, fid_arr - fid_std_arr, fid_arr + fid_std_arr, 
                             color=color_i, alpha=0.20)
        
        # JSD panel
        axes[2].plot(nlist, js_arr, marker='o', markersize=marker_size, linestyle='-', 
                     color=color_i)
        axes[2].fill_between(nlist, js_arr - js_std_arr, js_arr + js_std_arr, 
                             color=color_i, alpha=0.20)
    
    # Add baseline projected DDPM (from first sigma's baseline data)
    if len(sorted_items) >= 1:
        base_nlist = np.array(sorted_items[0][1]['num_train_list'], dtype=np.int64)
        cov_plain_arr = np.array(sorted_items[0][1]['plain']['coverage_mean'], dtype=np.float64)
        cov_plain_std_arr = np.array(sorted_items[0][1]['plain']['coverage_std'], dtype=np.float64)
        fid_plain_arr = np.array(sorted_items[0][1]['plain']['FID_mean'], dtype=np.float64)
        fid_plain_std_arr = np.array(sorted_items[0][1]['plain']['FID_std'], dtype=np.float64)
        js_plain_arr = np.array(sorted_items[0][1]['plain']['JS_distance_mean'], dtype=np.float64)
        js_plain_std_arr = np.array(sorted_items[0][1]['plain']['JS_distance_std'], dtype=np.float64)
        
        # Coverage baseline
        axes[0].plot(base_nlist, cov_plain_arr, marker='s', markersize=marker_size, 
                 linestyle='--', color='black', label='DDPM (proj.)')
        axes[0].fill_between(base_nlist, cov_plain_arr - cov_plain_std_arr, 
                             cov_plain_arr + cov_plain_std_arr, color='black', alpha=0.12)
        
        # FID baseline
        axes[1].plot(base_nlist, fid_plain_arr, marker='s', markersize=marker_size, 
                 linestyle='--', color='black', label='DDPM (proj.)')
        axes[1].fill_between(base_nlist, fid_plain_arr - fid_plain_std_arr, 
                             fid_plain_arr + fid_plain_std_arr, color='black', alpha=0.12)
        
        # JSD baseline
        axes[2].plot(base_nlist, js_plain_arr, marker='s', markersize=marker_size, 
                 linestyle='--', color='black', label='DDPM (proj.)')
        axes[2].fill_between(base_nlist, js_plain_arr - js_plain_std_arr, 
                             js_plain_arr + js_plain_std_arr, color='black', alpha=0.12)
    
    # Configure axes
    for idx, ax in enumerate(axes):
        ax.set_xscale('log')
        ax.set_xlabel('Training size')
        ax.set_ylabel(panel_labels[idx])
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.tick_params(axis='both', which='both', direction='out')
    
    # Add legend
    if num_sigmas == 1:
        axes[0].legend(frameon=False, loc='best')
    else:
        # Add colorbar for multiple sigmas below the panels
        from matplotlib.colors import LogNorm
        norm = LogNorm(vmin=min([float(s) for s, _ in sorted_items]), 
                                   vmax=max([float(s) for s, _ in sorted_items]))
        sm = plt.cm.ScalarMappable(cmap=cmap_cov, norm=norm)
        sm.set_array([])
        
        # Add marker legend
        from matplotlib.lines import Line2D
        lifted_handle = Line2D([0], [0], marker='o', color='gray', linestyle='-', label='$p_{\\sigma}$ (ours)')
        ddpm_handle = Line2D([0], [0], marker='s', color='black', linestyle='--', label='DDPM')
        
        # Reserve space at bottom for legend and colorbar
        # Standardize spacing to match other 3-panel figures
        # Use tight layout then slightly tighten horizontal margins and reduce panel spacing
        fig.tight_layout()
        try:
            fig.subplots_adjust(left=0.07, right=0.97, bottom=0.22, top=0.93, wspace=0.35)
        except Exception:
            pass

        # Position legend below plots (lowered to avoid overlapping x-labels)
        fig.legend(handles=[lifted_handle, ddpm_handle], loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.08))

        # Position colorbar below the legend (aligned with left/right margins)
        # Position a centered, slightly narrower colorbar aligned to tightened margins
        # left=0.08 and width=0.84 centers at 0.5
        cbar_ax = fig.add_axes([0.10, -0.18, 0.80, 0.025])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('σ (noise level)', fontsize=13)
    out_path = os.path.join(output_dir, "combined_metrics_3panel_num_samples.pdf")
    try:
        _save_pdf_png(fig, out_path, bbox_inches="tight")
    except Exception:
        out_path = os.path.join(output_dir, "combined_metrics_3panel_num_samples_fallback.pdf")
        _save_pdf_png(fig, out_path, bbox_inches="tight")
    finally:
        plt.close(fig)
    print(f"Saved 3-panel combined figure to {out_path}")


# Global plotting style for single-column figures
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "lines.linewidth": 2.0,
})

# Build 3-panel figure with all sigmas
if len(sorted_items) >= 1 and classifier_loaded:
    build_combined_3panel_figure(output_dir, sorted_items, norm_cov, cmap_cov, num_sigmas)

# Plots: FID vs number of training samples
if classifier_loaded:
    fig_fid, ax_fid = plt.subplots(figsize=(4.5, 3.2))
    for i, (sig, res) in enumerate(sorted_items):
        nlist = np.array(res['num_train_list'], dtype=np.int64)
        fid_arr = np.array(res['FID_mean'], dtype=np.float64)
        fid_std_arr = np.array(res['FID_std'], dtype=np.float64)
        color_i = cmap_cov(norm_cov(sig)) if norm_cov is not None else cmap_cov(0.5)
        ax_fid.plot(nlist, fid_arr, marker='o', markersize=5, linestyle='-', color=color_i, label=f'Lifted DDPM (σ={sig})' if num_sigmas == 1 else None)
        ax_fid.fill_between(nlist, fid_arr - fid_std_arr, fid_arr + fid_std_arr, color=color_i, alpha=0.20)

    if len(sorted_items) >= 1:
        base_nlist = np.array(sorted_items[0][1]['num_train_list'], dtype=np.int64)
        fid_plain_arr = np.array(sorted_items[0][1]['plain']['FID_mean'], dtype=np.float64)
        fid_plain_std_arr = np.array(sorted_items[0][1]['plain']['FID_std'], dtype=np.float64)
        ax_fid.plot(base_nlist, fid_plain_arr, marker='s', markersize=5, linestyle='--', color='black', label='DDPM')
        ax_fid.fill_between(base_nlist, fid_plain_arr - fid_plain_std_arr, fid_plain_arr + fid_plain_std_arr, color='black', alpha=0.12)

    if num_sigmas == 1:
        ax_fid.legend(frameon=False, loc='best')
    
    ax_fid.set_xscale('log')
    ax_fid.set_xlabel('Training dataset size', labelpad=6)
    ax_fid.set_ylabel('FID (Classifier Features)')
    ax_fid.grid(True, alpha=0.3, linestyle=':')
    ax_fid.tick_params(axis='both', which='both', direction='out')
    fig_fid.tight_layout()
    _save_pdf_png(fig_fid, os.path.join(output_dir, 'FID_vs_num_samples.pdf'), bbox_inches='tight')

# Plots: JS distance vs number of training samples
if classifier_loaded:
    fig_js, ax_js = plt.subplots(figsize=(4.5, 3.2))
    for i, (sig, res) in enumerate(sorted_items):
        nlist = np.array(res['num_train_list'], dtype=np.int64)
        js_arr = np.array(res['JS_distance_mean'], dtype=np.float64)
        js_std_arr = np.array(res['JS_distance_std'], dtype=np.float64)
        color_i = cmap_cov(norm_cov(sig)) if norm_cov is not None else cmap_cov(0.5)
        ax_js.plot(nlist, js_arr, marker='o', markersize=5, linestyle='-', color=color_i, label=f'Lifted DDPM (σ={sig})' if num_sigmas == 1 else None)
        ax_js.fill_between(nlist, js_arr - js_std_arr, js_arr + js_std_arr, color=color_i, alpha=0.20)

    if len(sorted_items) >= 1:
        base_nlist = np.array(sorted_items[0][1]['num_train_list'], dtype=np.int64)
        js_plain_arr = np.array(sorted_items[0][1]['plain']['JS_distance_mean'], dtype=np.float64)
        js_plain_std_arr = np.array(sorted_items[0][1]['plain']['JS_distance_std'], dtype=np.float64)
        ax_js.plot(base_nlist, js_plain_arr, marker='s', markersize=5, linestyle='--', color='black', label='DDPM')
        ax_js.fill_between(base_nlist, js_plain_arr - js_plain_std_arr, js_plain_arr + js_plain_std_arr, color='black', alpha=0.12)

    if num_sigmas == 1:
        ax_js.legend(frameon=False, loc='best')
    
    ax_js.set_xscale('log')
    ax_js.set_xlabel('Training dataset size', labelpad=6)
    ax_js.set_ylabel('JS Distance (Class Distribution)')
    ax_js.grid(True, alpha=0.3, linestyle=':')
    ax_js.tick_params(axis='both', which='both', direction='out')
    fig_js.tight_layout()
    _save_pdf_png(fig_js, os.path.join(output_dir, 'JS_distance_vs_num_samples.pdf'), bbox_inches='tight')

# Plots: Coverage vs number of training samples (single-column friendly)
fig_cov, ax_cov = plt.subplots(figsize=(4.5, 3.2))
for i, (sig, res) in enumerate(sorted_items):
    nlist = np.array(res['num_train_list'], dtype=np.int64)
    cov_arr = np.array(res['coverage_mean'], dtype=np.float64)
    cov_std_arr = np.array(res['coverage_std'], dtype=np.float64)
    # Lifted DDPM: circles
    color_i = cmap_cov(norm_cov(sig)) if norm_cov is not None else cmap_cov(0.5)
    ax_cov.plot(nlist, cov_arr, marker='o', markersize=5, linestyle='-', color=color_i, label=f'Lifted DDPM (σ={sig})' if num_sigmas == 1 else None)
    ax_cov.fill_between(nlist, cov_arr - cov_std_arr, cov_arr + cov_std_arr, color=color_i, alpha=0.20)

# Single DDPM baseline (shared across σ): use the first sigma's baseline arrays
if len(sorted_items) >= 1:
    base_nlist = np.array(sorted_items[0][1]['num_train_list'], dtype=np.int64)
    cov_plain_arr = np.array(sorted_items[0][1]['plain']['coverage_mean'], dtype=np.float64)
    cov_plain_std_arr = np.array(sorted_items[0][1]['plain']['coverage_std'], dtype=np.float64)
    ax_cov.plot(base_nlist, cov_plain_arr, marker='s', markersize=5, linestyle='--', color='black', label='DDPM (proj.)')
    ax_cov.fill_between(base_nlist, cov_plain_arr - cov_plain_std_arr, cov_plain_arr + cov_plain_std_arr, color='black', alpha=0.12)

# Legend strategy:
if num_sigmas == 1:
    # Single sigma: show method legend entries (Lifted and projected DDPM)
    ax_cov.legend(frameon=False, loc='best')
else:
    # Multiple sigmas: add colorbar for σ and keep projected DDPM as a single black dashed line
    # Marker legend for methods
    from matplotlib.lines import Line2D
    lifted_handle = Line2D([0], [0], marker='o', color='gray', linestyle='-', label='Ours (per σ)')
    ddpm_handle = Line2D([0], [0], marker='s', color='black', linestyle='--', label='DDPM (proj.)')
    ax_cov.legend(handles=[lifted_handle, ddpm_handle])
    # Add colorbar for sigma
    from matplotlib.colors import LogNorm
    norm = LogNorm(vmin=float(min(results_by_sigma.keys())), vmax=float(max(results_by_sigma.keys())))
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    sm.set_array([])
    cbar = fig_cov.colorbar(sm, ax=ax_cov, orientation='horizontal', pad=0.25)
    cbar.set_label('sigma', fontsize=11)

ax_cov.set_xscale('log')
ax_cov.set_xlabel('Training dataset size', labelpad=6)
ax_cov.set_ylabel('Coverage')
# No title for paper-ready figure
ax_cov.grid(True, alpha=0.3, linestyle=':')
ax_cov.tick_params(axis='both', which='both', direction='out')
fig_cov.tight_layout()
_save_pdf_png(fig_cov, os.path.join(output_dir, 'coverage_vs_num_samples.pdf'), bbox_inches='tight')
