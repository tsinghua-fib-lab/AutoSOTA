import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib as mpl

# Move to the project root (one level up from current file)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

from train_mnist_classifier import SimpleMNISTClassifier
from datasets import MNISTFixedSumDataset


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

from trainers import DDPMTrainer
from utils.constraints import FixedSumProjector
from utils.metrics import coverage, filter_valid_samples


def _save_pdf_png(fig, output_path, **kwargs):
    fig.savefig(output_path, **kwargs)
    if output_path.lower().endswith(".pdf"):
        fig.savefig(output_path[:-4] + ".png", **kwargs)
import scipy.linalg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--num_samples", type=int, default=10000, help="Accepted for wrapper compatibility; MNIST evaluation is forced to 10000")
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

    # Convert to tensor
    x = samples if isinstance(samples, torch.Tensor) else torch.tensor(samples)

    # Force float32
    if x.dtype != torch.float32:
        x = x.float()

    # Scale to [0,1] if needed
    if x.numel() > 0:
        xmax = float(x.max().detach().cpu())
        if xmax > 1.0:
            x = x / 255.0

    # Ensure shape is (N,1,28,28)
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
    
    # Ensure shape is (N,1,28,28)
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

def set_paper_style():
    mpl.rcParams.update({
        "figure.figsize": (3.2, 2.4),
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "lines.linewidth": 2.6,
        "lines.markersize": 7,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })

# MNIST defaults
MNIST_TEST_SET_SIZE = 10000
if args.num_samples != MNIST_TEST_SET_SIZE:
    print(f"Ignoring --num_samples={args.num_samples}; MNIST evaluation/checkpoint selection uses {MNIST_TEST_SET_SIZE}.")
if args.num_eval_samples != MNIST_TEST_SET_SIZE:
    print(f"Ignoring --num_eval_samples={args.num_eval_samples}; MNIST evaluation uses the full {MNIST_TEST_SET_SIZE} sample test set.")
num_samples = MNIST_TEST_SET_SIZE
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

# Evaluate models trained at various noise levels (include the MNIST default 1.0)
sigma_list = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]


def _attach_time_embed_if_needed(denoiser, state_dict, device):
    """Placeholder function - time embedding utilities have been removed."""
    pass
data_points_true = _mnist_test_set(noise_level=0.0, lifted=False, device=device)
D = int(data_points_true.shape[1])
true_tensor = filter_valid_samples(data_points_true.view(-1, D)).cpu()


# Load classifier once for all sigma evaluations
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
    js_max = float(np.sqrt(np.log(2.0)))  # max JS distance
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

# Accumulators across sigmas
coverage_list = []
coverage_std_list = []
FID_list = []
FID_std_list = []
JS_list = []
JS_std_list = []
scores_list = []  # per-sigma score curves (from trial 0)

set_paper_style()
# Evaluate Lifted DDPM models for each sigma
for sigma in sigma_list:
    # Load cached training subset for this sigma (lifted=True, matches training noise level)
    data_points_eval = _mnist_test_set(noise_level=sigma, lifted=True, device=device)
    ref_tensor = filter_valid_samples(data_points_eval.view(-1, D)).cpu()
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

    # Load checkpoint for this sigma
    checkpoint_path = f"models/mnist/model_DDPM_epoch_{epochs}_num_samples_{num_samples}_noise_level_{sigma}_time_{time_embed_choice}_seed_{random_seed}.pth"
    if not os.path.exists(checkpoint_path):
        fallback = f"models/mnist/model_DDPM_epoch_{epochs}_noise_level_{sigma}_time_{time_embed_choice}_seed_{random_seed}.pth"
        if os.path.exists(fallback):
            checkpoint_path = fallback
    if not os.path.exists(checkpoint_path):
        fallback = f"models/mnist/model_DDPM_epoch_{epochs}_noise_level_{sigma}.pth"
        if os.path.exists(fallback):
            checkpoint_path = fallback
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    # Attach or remove time-embedding to match checkpoint
    _attach_time_embed_if_needed(trainer_lifted.denoiser, state_dict, device)
    has_timeembed = isinstance(state_dict, dict) and any(k.startswith('time_embed_module.') for k in state_dict.keys())
    if not has_timeembed and getattr(trainer_lifted.denoiser, 'time_embed_module', None) is not None:
        trainer_lifted.denoiser.time_embed_module = None

    # Restore weights (and, if available, timing metadata)
    try:
        trainer_lifted.load_checkpoint(checkpoint_path, map_location=device, load_optimizer=False)
    except Exception:
        trainer_lifted.denoiser.load_state_dict(state_dict if isinstance(state_dict, dict) else {})

    trainer_lifted.denoiser.eval()

    # Trials to estimate variability
    trial_coverages = []
    trial_fids = []
    trial_js_dists = []
    for t in range(trials):
        with torch.no_grad():
            samples_lifted, _ = trainer_lifted.sample(num_samples=num_samples)
            try:
                samples_lifted = trainer_lifted.projector.project(torch.tensor(samples_lifted).cpu())[0].cpu()
            except Exception:
                samples_lifted = torch.tensor(samples_lifted)

        if t == 0:
            # collect per-time-step score norms
            try:
                scores_list.append(list(trainer_lifted.scores))
            except Exception:
                scores_list.append([])

        # metrics
        samples_tensor = filter_valid_samples(torch.tensor(samples_lifted).view(-1, D)).cpu()
        try:
            cov_val = _embedding_coverage(samples_tensor)
        except Exception:
            cov_val = float('nan')
        trial_coverages.append(cov_val)
        
        # Classifier-based metrics
        if classifier_loaded:
            try:
                # FID
                feats_gen = get_classifier_features(samples_tensor, classifier, device=device).astype(np.float64)
                mu_gen = feats_gen.mean(axis=0)
                sigma_gen = np.cov(feats_gen, rowvar=False)
                fid_val = float(compute_fid(mu_gen, sigma_gen, mu_real_for_metrics, sigma_real_for_metrics))
                trial_fids.append(fid_val)
            except Exception:
                trial_fids.append(float('nan'))
            
            try:
                # JS distance
                preds_gen = classifier_accuracy(samples_tensor, classifier, device=device)
                gen_counts = np.bincount(preds_gen, minlength=10)
                js_val = float(js_distance_from_counts(gen_counts, ref_counts_for_metrics))
                trial_js_dists.append(js_val)
            except Exception:
                trial_js_dists.append(float('nan'))

    # aggregate
    cov_arr = np.array(trial_coverages, dtype=np.float64)
    coverage_list.append(float(np.nanmean(cov_arr)))
    coverage_std_list.append(float(np.nanstd(cov_arr)))
    
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


# Baseline: Traditional DDPM (NONPROJECT), trained at noise_level=0.0
print("Traditional DDPM Model (baseline)")
trainer_plain = DDPMTrainer(
    data_points_true.squeeze(),
    timesteps=timesteps,
    project_x0_sample=False,
    projector=FixedSumProjector(target_sum=target_sum),
    hidden_dim=hidden_dim,
    time_embed_dim=time_embed_dim,
    size=D,
    image=True,
)
checkpoint_path = f"models/mnist/model_DDPM_NONPROJECT_epoch_{epochs}_num_samples_{num_samples}_noise_level_0.0_time_{time_embed_choice}_seed_{random_seed}.pth"
if not os.path.exists(checkpoint_path):
    fallback = f"models/mnist/model_DDPM_NONPROJECT_epoch_{epochs}_noise_level_0.0_time_{time_embed_choice}_seed_{random_seed}.pth"
    if os.path.exists(fallback):
        checkpoint_path = fallback
if not os.path.exists(checkpoint_path):
    fallback = f"models/mnist/model_DDPM_NONPROJECT_epoch_{epochs}_noise_level_0.0.pth"
    if os.path.exists(fallback):
        checkpoint_path = fallback
checkpoint = torch.load(checkpoint_path, map_location=device)
state_plain = checkpoint.get('model_state_dict', checkpoint)
_attach_time_embed_if_needed(trainer_plain.denoiser, state_plain if isinstance(state_plain, dict) else {}, device)
has_timeembed_plain = isinstance(state_plain, dict) and any(k.startswith('time_embed_module.') for k in state_plain.keys())
if not has_timeembed_plain and getattr(trainer_plain.denoiser, 'time_embed_module', None) is not None:
    trainer_plain.denoiser.time_embed_module = None
try:
    trainer_plain.load_checkpoint(checkpoint_path, map_location=device, load_optimizer=False)
except Exception:
    trainer_plain.denoiser.load_state_dict(state_plain if isinstance(state_plain, dict) else {})
trainer_plain.denoiser.eval()

# Trials for baseline
plain_coverages = []
plain_fids = []
plain_js_dists = []
scores_plain = []
for t in range(trials):
    with torch.no_grad():
        samples_plain, _ = trainer_plain.sample(num_samples=num_samples)
        try:
            samples_plain = trainer_plain.projector.project(torch.tensor(samples_plain).cpu())[0].cpu()
        except Exception:
            samples_plain = torch.tensor(samples_plain)
    if t == 0:
        try:
            scores_plain = list(trainer_plain.scores)
        except Exception:
            scores_plain = []
    samples_plain_tensor = filter_valid_samples(torch.tensor(samples_plain).view(-1, D)).cpu()
    try:
        try:
            plain_coverages.append(_embedding_coverage(samples_plain_tensor))
        except Exception:
            plain_coverages.append(float('nan'))
    except Exception:
        plain_coverages.append(float('nan'))
    
    # Classifier-based metrics for baseline
    if classifier_loaded:
        try:
            feats_gen = get_classifier_features(samples_plain_tensor, classifier, device=device).astype(np.float64)
            mu_gen = feats_gen.mean(axis=0)
            sigma_gen = np.cov(feats_gen, rowvar=False)
            fid_val = float(compute_fid(mu_gen, sigma_gen, mu_real_for_metrics, sigma_real_for_metrics))
            plain_fids.append(fid_val)
        except Exception:
            plain_fids.append(float('nan'))
        
        try:
            preds_gen = classifier_accuracy(samples_plain_tensor, classifier, device=device)
            gen_counts = np.bincount(preds_gen, minlength=10)
            js_val = float(js_distance_from_counts(gen_counts, ref_counts_for_metrics))
            plain_js_dists.append(js_val)
        except Exception:
            plain_js_dists.append(float('nan'))

coverage_plain = float(np.nanmean(np.array(plain_coverages)))
coverage_plain_std = float(np.nanstd(np.array(plain_coverages)))

if classifier_loaded:
    FID_plain = float(np.nanmean(np.array(plain_fids)))
    FID_plain_std = float(np.nanstd(np.array(plain_fids)))
    JS_plain = float(np.nanmean(np.array(plain_js_dists)))
    JS_plain_std = float(np.nanstd(np.array(plain_js_dists)))
else:
    FID_plain = float('nan')
    FID_plain_std = float('nan')
    JS_plain = float('nan')
    JS_plain_std = float('nan')


# Save metrics JSON
output_dir = "results/mnist"
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, "metrics_varied_sigmas.json"), "w") as f:
    json.dump(
        {
            "sigma_list": sigma_list,
            "coverage_mean": coverage_list,
            "coverage_std": coverage_std_list,
            "FID_mean": FID_list,
            "FID_std": FID_std_list,
            "JS_distance_mean": JS_list,
            "JS_distance_std": JS_std_list,
            "plain": {
                "coverage_mean": coverage_plain,
                "coverage_std": coverage_plain_std,
                "FID_mean": FID_plain,
                "FID_std": FID_plain_std,
                "JS_distance_mean": JS_plain,
                "JS_distance_std": JS_plain_std,
            },
        },
        f,
    )

# ------------------------------------------------------------------
# Coverage figure in unified paper style
# ------------------------------------------------------------------
def build_combined_3panel_figure(
    output_dir,
    sigma_list,
    coverage_list,
    coverage_std_list,
    FID_list,
    FID_std_list,
    JS_list,
    JS_std_list,
    coverage_plain,
    coverage_plain_std,
    FID_plain,
    FID_plain_std,
    JS_plain,
    JS_plain_std,
):
    """Create a 1x3 figure with Coverage, FID, and JSD vs sigma.

    Style mirrors plotting_smileyface_plane_varied_sigmas.py exactly:
      - log-scale x axis (sigma)
      - ours curve with markers and shaded error band
      - baseline DDPM horizontal dashed line + band
      - legend stacked (ours vs baseline) placed below axes
    """
    set_paper_style()
    color_map = {"OURS": "#1f77b4", "DDPM": "#d62728"}
    line_width = 3.0
    marker_size = 7

    # Use consistent proportions for 3-panel figures
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2), sharex=True)
    metric_data = [
        ("Coverage", np.array(coverage_list, dtype=float), np.array(coverage_std_list, dtype=float), coverage_plain, coverage_plain_std),
        ("FID", np.array(FID_list, dtype=float), np.array(FID_std_list, dtype=float), FID_plain, FID_plain_std),
        ("JSD", np.array(JS_list, dtype=float), np.array(JS_std_list, dtype=float), JS_plain, JS_plain_std),
    ]
    handles = []
    labels = []

    for ax, (label, vals, stds, base_mean, base_std) in zip(axes, metric_data):
        # ours curve
        h, = ax.plot(
            sigma_list,
            vals,
            marker="o",
            markersize=marker_size,
            linewidth=line_width,
            color=color_map["OURS"],
            label=r"$p_{\sigma}$ (ours)",
            zorder=2,
        )
        # error band (min thickness enforcement similar to sphere script)
        try:
            arr = vals.astype(float)
            std_arr = stds.astype(float)
            mask = np.isfinite(arr) & np.isfinite(std_arr)
            if mask.any():
                sig = np.array(sigma_list)[mask]
                min_band = 0.005 if label == "Coverage" else 0.002
                delta = np.maximum(std_arr[mask], min_band)
                lower = arr[mask] - delta
                upper = arr[mask] + delta
                if label in ("FID", "JSD"):
                    eps = 1e-12
                    lower = np.maximum(lower, eps)
                    upper = np.maximum(upper, eps)
                ax.fill_between(sig, lower, upper, color=color_map["OURS"], alpha=0.25, linewidth=0, zorder=1)
        except Exception:
            pass

        # Collect legend handle once
        if ax is axes[0]:
            handles.append(h)
            labels.append(r"$p_{\sigma}$ (ours)")

        # baseline line and band
        hb = ax.axhline(
            y=base_mean,
            color=color_map["DDPM"],
            linestyle="--",
            linewidth=line_width,
            alpha=0.9,
            label="DDPM (proj.)",
        )
        if ax is axes[0]:
            handles.append(hb)
            labels.append("DDPM (proj.)")
        if base_std is not None and np.isfinite(base_std):
            eps = 1e-12 if label in ("FID", "JSD") else 0.0
            low = max(base_mean - base_std, eps)
            high = max(base_mean + base_std, eps)
            ax.fill_between(
                sigma_list,
                [low] * len(sigma_list),
                [high] * len(sigma_list),
                color=color_map["DDPM"],
                alpha=0.10,
            )

        # axes cosmetics
        ax.set_xscale("log")
        ax.set_xlabel("σ")
        ax.set_ylabel(label)
        # Set x-limits and ticks to standard increments for clarity
        try:
            min_sigma = float(min(sigma_list))
            max_sigma = float(max(sigma_list))
            ax.set_xlim(min_sigma, max_sigma)
            # Major ticks at powers of 10 only
            from matplotlib.ticker import LogLocator, FuncFormatter
            ax.xaxis.set_major_locator(LogLocator(base=10.0))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}"))
            # Remove minor tick labels entirely to avoid 5s
            ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=[]))
        except Exception:
            pass
        # Adaptive y-limits with small padding to make lines clearer
        try:
            values_for_ylim = [vals]
            if np.isfinite(base_mean):
                values_for_ylim.append(np.array([base_mean]))
            y_all = np.concatenate([v.flatten() for v in values_for_ylim])
            y_all = y_all[np.isfinite(y_all)]
            if label == "Coverage":
                if y_all.size > 0:
                    ymin = max(0.0, float(y_all.min()) - 0.03)
                    ymax = min(1.0, float(y_all.max()) + 0.03)
                    ax.set_ylim(ymin, ymax)
                else:
                    ax.set_ylim(0, 1)
            else:
                if y_all.size > 0:
                    ymin = 0.0
                    ymax = max(1e-12, float(y_all.max()) * 1.10)
                    ax.set_ylim(ymin, ymax)
                else:
                    ax.set_ylim(bottom=0)
        except Exception:
            pass
        ax.grid(True)

    try:
        ncols = max(1, len(labels))
        fig.legend(handles, labels, loc="lower center", ncol=ncols, frameon=False, bbox_to_anchor=(0.5, -0.14))
    except Exception:
        pass

    fig.tight_layout()
    try:
        fig.subplots_adjust(left=0.12, right=0.98, bottom=0.18, top=0.92, wspace=0.35)
    except Exception:
        pass
    out_path = os.path.join(output_dir, "combined_metrics_3panel.pdf")
    try:
        _save_pdf_png(fig, out_path, bbox_inches="tight")
    except Exception:
        out_path = os.path.join(output_dir, "combined_metrics_3panel_fallback.pdf")
        _save_pdf_png(fig, out_path, bbox_inches="tight")
    finally:
        plt.close(fig)


# Build the combined 3-panel figure (if classifier loaded)
if classifier_loaded:
    build_combined_3panel_figure(
        output_dir,
        sigma_list,
        coverage_list,
        coverage_std_list,
        FID_list,
        FID_std_list,
        JS_list,
        JS_std_list,
        coverage_plain,
        coverage_plain_std,
        FID_plain,
        FID_plain_std,
        JS_plain,
        JS_plain_std,
    )


# Plot score curves vs time, color-coded by sigma
if "scores_list" in globals() and len(scores_list) > 0:
    set_paper_style()
    cmap = plt.cm.viridis
    try:
        norm = mpl.colors.LogNorm(vmin=float(min(sigma_list)), vmax=float(max(sigma_list)))
    except Exception:
        norm = mpl.colors.Normalize(vmin=float(min(sigma_list)), vmax=float(max(sigma_list)))

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    fig, ax = plt.subplots(figsize=(3.5, 4.0))
    nan_label_plotted = False
    for i, sigma in enumerate(sigma_list):
        raw = scores_list[i] if i < len(scores_list) else []
        try:
            scores_sigma = np.array([float(x) for x in raw])
        except Exception:
            scores_sigma = np.array(raw)
        invalid_mask = ~np.isfinite(scores_sigma)
        valid_mask = np.isfinite(scores_sigma)
        color = cmap(norm(sigma))
        ax.plot(scores_sigma, color=color, linewidth=2.0)
        if np.any(valid_mask):
            top_y = np.nanmax(scores_sigma[valid_mask]) * 1.1
        else:
            top_y = 1.0
        if np.any(invalid_mask):
            label = "NaN or Inf" if not nan_label_plotted else ""
            ax.plot(
                np.where(invalid_mask)[0],
                [top_y] * np.sum(invalid_mask),
                "x",
                color=color,
                markersize=5,
                label=label,
            )
            nan_label_plotted = True

    # Plot DDPM baseline scores if present
    try:
        scores_plain_plot = np.array([float(x) for x in scores_plain])
    except Exception:
        scores_plain_plot = np.array(scores_plain)
    invalid_mask_plain = ~np.isfinite(scores_plain_plot)
    valid_mask_plain = np.isfinite(scores_plain_plot)
    if np.any(valid_mask_plain):
        top_y_plain = np.nanmax(scores_plain_plot[valid_mask_plain]) * 1.1
    else:
        top_y_plain = 1.0
    ax.plot(scores_plain_plot, label="DDPM", linestyle="--", color="red", linewidth=2.6)
    if np.any(invalid_mask_plain):
        ax.plot(
            np.where(invalid_mask_plain)[0],
            [top_y_plain] * np.sum(invalid_mask_plain),
            "x",
            color="red",
            markersize=5,
            label="NaN or Inf" if not nan_label_plotted else "",
        )

    ax.set_yscale("log")

    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", pad=0.20)
    cbar.set_label("σ", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    # Prefer ticks at powers of ten within the sigma range and label as 10^{exp}
    try:
        min_s = float(min(sigma_list))
        max_s = float(max(sigma_list))
        if min_s > 0 and max_s > 0:
            exp_min = int(np.floor(np.log10(min_s)))
            exp_max = int(np.ceil(np.log10(max_s)))
            exps = np.arange(exp_min, exp_max + 1)
            ticks = (10.0 ** exps).tolist()
            ticks = [t for t in ticks if t >= min_s and t <= max_s]
            if len(ticks) >= 1:
                cbar.set_ticks(ticks)
                cbar.set_ticklabels([r"$10^{%d}$" % int(np.round(np.log10(t))) for t in ticks])
            else:
                cbar.set_ticks([min_s, max_s])
                cbar.set_ticklabels([f"{min_s:.3g}", f"{max_s:.3g}"])
    except Exception:
        try:
            cbar.set_ticks(sigma_list)
            cbar.set_ticklabels([str(s) for s in sigma_list])
        except Exception:
            pass

    ax.legend(fontsize=9, frameon=False, loc="upper left")
    num_points = len(scores_plain_plot)
    xticks = np.linspace(0, num_points - 1, num=6, dtype=int)
    xtick_labels = [f"{num_points - 1 - x}" for x in xticks]
    xtick_labels[0] = str(timesteps)
    xtick_labels[-1] = "0"
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, fontsize=9)
    ax.set_xlabel(r"$t$ (reversed)", fontsize=11, labelpad=8)
    ax.set_ylabel(
        r"Median $\nabla_x(t) \, \log p_t(x(t))$", fontsize=11
    )
    ax.grid(True)
    fig.subplots_adjust(bottom=0.20)
    _save_pdf_png(fig, os.path.join(output_dir, "scores_vs_time.pdf"), bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------
