#!/usr/bin/env python3
"""
Compute Pseudo-True Radius for MNIST Rotation (Simplified PGD-based)

This script computes the "pseudo-true" worst-case radius by finding the maximum
perturbation that causes change ≤ ε_y using Projected Gradient Descent (PGD).

Simplified approach:
1. Use PGD to find worst-case perturbation (much faster than differential evolution)
2. Use bisection to find R such that max_change(R) = eps_y
3. Multiple random restarts to avoid local maxima
4. Optional: MC sanity check to verify

Note on PGD vs MC:
- PGD finds local maxima (with multiple restarts, should be close to global)
- If PGD finds LARGER value than MC: Good! Means we found better worst-case
- If PGD finds SMALLER value: Possible underestimation (local max < global max)
- With multiple random restarts (default: 5), PGD should find good solutions
- MC sampling is simpler but may miss worst-case (only samples, doesn't optimize)
- Recommendation: Use PGD with multiple restarts (this script), verify with MC if needed

Usage:
    # Quick test on 5 samples
    python experiments/mnist_rotation/compute_mnist_pseudo_true_radius_simple.py \
        --variance_gradient mnist_rotation_full_cert_n100_20251106_033225.json \
        --n_points 5 \
        --n_mc 50000 \
        --device cuda
    
    # Full run on all 100 samples
    python experiments/mnist_rotation/compute_mnist_pseudo_true_radius_simple.py \
        --variance_gradient mnist_rotation_full_cert_n100_20251106_033225.json \
        --n_points 100 \
        --n_mc 50000 \
        --device cuda
"""

import json
import numpy as np
import torch
import argparse
from pathlib import Path
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from scipy.optimize import brentq
from PIL import Image
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))
sys.path.append(str(Path(__file__).resolve().parent))

from e2cnn_rotation_model import RotationEquivariantCNN_Simple, cos_sin_to_angle


# MNIST normalization constants (used throughout this repo)
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def denormalize_mnist(x_norm: np.ndarray) -> np.ndarray:
    """Convert normalized MNIST image to raw [0,1] pixel space."""
    return x_norm * MNIST_STD + MNIST_MEAN


def normalize_mnist(x_raw: np.ndarray) -> np.ndarray:
    """Convert raw [0,1] MNIST image to normalized space."""
    return (x_raw - MNIST_MEAN) / MNIST_STD


def load_json(json_path: str) -> Dict:
    """Load JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def load_model(model_path: str, device: str = 'cpu') -> torch.nn.Module:
    """Load trained MNIST rotation model."""
    model = RotationEquivariantCNN_Simple(N=8)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def load_mnist_image(image_idx: int, use_rotation_dataset: bool = False) -> np.ndarray:
    """Load MNIST test image by index in RAW [0,1] pixel space.
    
    Args:
        image_idx: Index of the image
        use_rotation_dataset: If True, load from rotated MNIST dataset
    """
    if use_rotation_dataset:
        # Load a SINGLE rotated MNIST sample on-demand (do NOT generate the full dataset).
        #
        # This matches `experiments/mnist_rotation/dataset_generator.py` logic:
        # - rotation_range=(0,360), augmentation_factor=1, seed=42
        # - rotate with background_color=33, expand=True, then resize back to 28x28 (BILINEAR)
        #
        # NOTE: This recreates the same rotation dataset indexing:
        #   base_idx = image_idx // augmentation_factor
        #   aug_idx  = image_idx % augmentation_factor
        import torchvision
        from torchvision import transforms
        import random
        
        rotation_range = (0.0, 360.0)
        augmentation_factor = 1
        seed = 42
        background_color = 33
        
        # Map rotation-dataset index -> original MNIST index
        base_idx = int(image_idx // augmentation_factor)
        aug_idx = int(image_idx % augmentation_factor)
        
        # Load original MNIST test sample in raw [0,1]
        raw_transform = transforms.Compose([transforms.ToTensor()])
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=raw_transform
        )
        if base_idx >= len(test_dataset):
            raise ValueError(f"Rotation base index {base_idx} out of range (max: {len(test_dataset)-1})")
        
        image_tensor, _ = test_dataset[base_idx]  # (1,28,28) raw
        image_raw = image_tensor.squeeze(0).numpy()
        image_raw = np.clip(image_raw, 0.0, 1.0)
        
        # Reproduce the python `random` stream used by MNISTRotationDataset generation
        rng = random.Random(seed)
        # Skip draws up to the required (base_idx, aug_idx) position:
        # generation loops: for i in range(len(dataset)): for _ in range(augmentation_factor): draw angle
        draws_to_skip = base_idx * augmentation_factor + aug_idx
        for _ in range(draws_to_skip):
            rng.uniform(*rotation_range)
        angle = rng.uniform(*rotation_range)
        
        # Rotate using the same PIL pipeline as dataset_generator
        pil_img = Image.fromarray((image_raw * 255.0).astype(np.uint8), mode="L")
        rotated = pil_img.rotate(angle, fillcolor=background_color, expand=True)
        rotated = rotated.resize((28, 28), Image.BILINEAR)
        
        rotated_np = np.asarray(rotated).astype(np.float32) / 255.0  # raw [0,1]
        rotated_np = np.clip(rotated_np, 0.0, 1.0)
        return rotated_np
    else:
        # Load from original MNIST
        import torchvision
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),  # raw [0,1]
        ])
        
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )
        
        if image_idx >= len(test_dataset):
            raise ValueError(f"Image index {image_idx} out of range (max: {len(test_dataset)-1})")
        
        image_tensor, _ = test_dataset[image_idx]
        image_np = image_tensor.squeeze(0).numpy()  # (28, 28) in [0,1]
        image_np = np.clip(image_np, 0.0, 1.0)
        return image_np


def estimate_g_sigma_batch(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    sigma: float,
    N: int,
    batch_size: int = 1000,
    device: str = 'cpu',
    noise_fixed: Optional[torch.Tensor] = None
) -> float:
    """
    Estimate E[g_σ(x)] using batched Monte Carlo with proper circular averaging.
    
    Returns:
        Mean angle in radians
    """
    cos_sum = 0.0
    sin_sum = 0.0
    
    # IMPORTANT:
    # - Perturbations/noise are applied in RAW [0,1] pixel space (to match other certifiers)
    # - Images are normalized only right before the model forward pass
    #
    # Ensure correct shape: (1, 1, 28, 28)
    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
    elif image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    
    image_tensor = image_tensor.to(device)
    
    # Use fixed noise (CRN) or generate new
    if noise_fixed is not None:
        all_noise = noise_fixed
    else:
        all_noise = torch.randn(N, 1, 28, 28).to(device) * sigma
    
    num_batches = (N + batch_size - 1) // batch_size
    
    model.eval()
    with torch.no_grad():
        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, N)
            batch_noise = all_noise[start:end]
            
            batch_input_raw = (image_tensor + batch_noise).clamp(0.0, 1.0)
            batch_input_norm = (batch_input_raw - MNIST_MEAN) / MNIST_STD
            pred_cos_sin = model(batch_input_norm)
            
            cos_sum += pred_cos_sin[:, 0].sum().item()
            sin_sum += pred_cos_sin[:, 1].sum().item()
    
    avg_cos = cos_sum / N
    avg_sin = sin_sum / N
    mean_angle = np.arctan2(avg_sin, avg_cos)
    
    return float(mean_angle)


def angular_distance(a1: float, a2: float) -> float:
    """Compute shortest angular distance between two angles in radians."""
    diff = np.abs(a1 - a2)
    return float(np.minimum(diff, 2 * np.pi - diff))


def find_worst_case_pgd_simple(
    model: torch.nn.Module,
    image: np.ndarray,
    clean_g: float,
    sigma: float,
    R_max: float,
    N_mc: int = 50000,
    n_restarts: int = 5,
    n_steps: int = 50,
    step_size: float = 0.01,
    attack_n_mc: int = 2000,
    device: str = 'cpu',
    seed: int = 42,
    verbose: bool = False
) -> Tuple[float, Dict]:
    """
    Find worst-case perturbation using PGD with multiple random restarts.
    
    Returns:
        (max_change, info_dict)
    """
    # Pre-generate noise bank for CRN (RAW pixel space noise)
    torch.manual_seed(seed)
    noise_bank = torch.randn(N_mc, 1, 28, 28).to(device) * sigma
    
    # Convert image to tensor
    image_tensor = torch.from_numpy(image).float()
    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
    elif image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)
    
    # A) Freeze model parameters (critical for avoiding graph contamination)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    
    # --- IMPORTANT: clean reference must be constant (no graph) ---
    # Target direction (clean expectation) - create as plain tensor, no grad
    target_cos = np.cos(clean_g)
    target_sin = np.sin(clean_g)
    target = torch.tensor([target_cos, target_sin], device=device, dtype=torch.float32)  # no grad
    
    best_max_change = 0.0
    best_delta = None
    
    # Use subset of noise for gradient computation (faster)
    attack_N = min(N_mc, int(attack_n_mc))
    noise_subset = noise_bank[:attack_N]  # Fixed noise bank (CRN)
    
    # Multiple random restarts
    for restart in range(n_restarts):
        if verbose and restart > 0:
            print(f"    PGD restart {restart+1}/{n_restarts}...")
        # B) Initialize delta as a clean leaf tensor (avoid .data assignment)
        torch.manual_seed(seed + restart)
        # Small random initialization
        delta_init = torch.randn_like(image_tensor) * 0.01
        delta_norm = delta_init.view(delta_init.shape[0], -1).norm(p=2, dim=1, keepdim=True)
        if delta_norm.item() > R_max:
            delta_init = delta_init * (R_max / (delta_norm + 1e-10)).view(-1, 1, 1, 1)
        # Create clean leaf tensor (no .data assignment)
        delta = delta_init.detach().clone().to(device)
        delta.requires_grad_(True)
        
        # PGD loop
        for step in range(n_steps):
            if verbose and step % 20 == 0 and step > 0:
                print(f"      PGD step {step}/{n_steps}...")
            # Forward pass: build a fresh graph each iteration
            # image_tensor and noise_subset are fixed (no grad), only delta has grad
            perturbed_raw = (image_tensor + delta + noise_subset).clamp(0.0, 1.0)  # (N_attack,1,28,28)
            perturbed_norm = (perturbed_raw - MNIST_MEAN) / MNIST_STD
            pred_cos_sin = model(perturbed_norm)  # (N_attack, 2) cos/sin
            mean_vec = pred_cos_sin.mean(dim=0)    # (2,)
            
            # Similarity to clean direction (bigger sim => smaller angular change)
            # We want to maximize angular change, so minimize similarity
            sim = (mean_vec * target).sum()
            loss = sim
            
            # C) Use autograd.grad instead of loss.backward() (avoids touching model param grads)
            g, = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)
            
            # Update step
            with torch.no_grad():
                # Normalize gradient (optional but stabilizes)
                g_norm = g.view(g.shape[0], -1).norm(p=2, dim=1, keepdim=True)
                g = g / (g_norm + 1e-12)
                g = g.view(*delta.shape)
                
                # Gradient descent on sim (minimize similarity = maximize angular change)
                delta -= step_size * g
                
                # Project onto L2 ball
                dnorm = delta.view(delta.shape[0], -1).norm(p=2, dim=1, keepdim=True)
                if dnorm.item() > R_max:
                    factor = (R_max / (dnorm + 1e-12)).view(-1, 1, 1, 1)
                    delta *= factor
            
            # --- CRITICAL: make delta a leaf again, drop old graph ---
            # This is the key fix from collaborator's suggestion!
            delta = delta.detach()
            delta.requires_grad_(True)
        
        # Evaluate final perturbation with full noise bank
        with torch.no_grad():
            final_delta = delta.detach()
            perturbed_g = estimate_g_sigma_batch(
                model, (image_tensor + final_delta).clamp(0.0, 1.0), sigma, N_mc,
                batch_size=1000, device=device, noise_fixed=noise_bank
            )
            
            change = angular_distance(perturbed_g, clean_g)
            
            if change > best_max_change:
                best_max_change = change
                best_delta = final_delta.clone()
    
    return best_max_change, {
        'best_delta_l2_raw': float(best_delta.norm().item()) if best_delta is not None else 0.0,
        'best_delta_l2_norm': float(best_delta.norm().item() / MNIST_STD) if best_delta is not None else 0.0,
        'attack_n_mc': int(attack_N),
        'n_steps': int(n_steps),
        'step_size': float(step_size),
    }


def compute_pseudo_true_radius_simple(
    model: torch.nn.Module,
    image: np.ndarray,
    sigma: float,
    eps_y: float,
    N_mc: int = 50000,
    R_max: float = 10.0,
    n_restarts: int = 5,
    n_steps: int = 50,
    step_size: float = 0.01,
    attack_n_mc: int = 2000,
    device: str = 'cpu',
    seed: int = 42,
    verbose: bool = False
) -> Tuple[float, Dict]:
    """
    Compute pseudo-true radius using bisection + PGD.
    
    Returns:
        (R_true, info_dict)
    """
    # Compute clean baseline
    if verbose:
        print(f"  Computing clean expectation (N_mc={N_mc})...")
    image_tensor = torch.from_numpy(image).float()
    clean_g = estimate_g_sigma_batch(
        model, image_tensor, sigma, N_mc,
        batch_size=1000, device=device
    )
    
    if verbose:
        print(f"  Clean expectation: {np.degrees(clean_g):.2f}°")
    else:
        print(f"  Clean expectation: {np.degrees(clean_g):.2f}° (computing worst-case perturbation...)")
    
    # Bisection search
    def max_change_at_R(R: float) -> float:
        """Find maximum change at radius R."""
        if verbose:
            print(f"  Testing R = {R:.4f}...")
        max_change, _ = find_worst_case_pgd_simple(
            model, image, clean_g, sigma, R, N_mc, n_restarts,
            n_steps=n_steps, step_size=step_size, attack_n_mc=attack_n_mc, device=device, seed=seed, verbose=verbose
        )
        if verbose:
            print(f"    max_change = {max_change:.6f} rad = {np.degrees(max_change):.2f}°")
        return max_change
    
    # Find R such that max_change(R) = eps_y
    R_low = 0.0
    R_high = R_max
    
    # Check if R_max is safe
    max_change_at_R_max = max_change_at_R(R_high)
    if max_change_at_R_max <= eps_y:
        return R_high, {
            'method': 'pgd', 
            'converged': True,
            'hit_R_max': True,
            'max_change_at_R_max': float(max_change_at_R_max),
            'note': 'True radius is at least R_max (may be larger)'
        }
    
    # Check if R=0 is unsafe
    if max_change_at_R(R_low) > eps_y:
        return 0.0, {'method': 'pgd', 'converged': True, 'note': 'Even R=0 violates eps_y'}
    
    # Bisection
    try:
        def f(R: float) -> float:
            return max_change_at_R(R) - eps_y
        
        R_true = brentq(f, R_low, R_high, xtol=1e-3, rtol=1e-3)
        # Get the actual max_change at the found radius for verification
        max_change_at_R_true, _ = find_worst_case_pgd_simple(
            model, image, clean_g, sigma, R_true, N_mc, n_restarts,
            n_steps=n_steps, step_size=step_size, attack_n_mc=attack_n_mc, device=device, seed=seed, verbose=verbose
        )
        return float(R_true), {
            'method': 'pgd', 
            'converged': True,
            'hit_R_max': False,
            'max_change_at_R_true': float(max_change_at_R_true)
        }
    except (ValueError, RuntimeError) as e:
        # Fallback: return conservative estimate
        if verbose:
            print(f"  Warning: Bisection failed: {e}")
        return 0.0, {'method': 'pgd', 'converged': False, 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description="Compute pseudo-true radius using simplified PGD")
    parser.add_argument(
        "--variance_gradient",
        type=str,
        required=True,
        help="JSON file with variance/gradient estimates"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="experiments/mnist_rotation/e2cnn_rotation_model.pth",
        help="Path to trained model weights"
    )
    parser.add_argument(
        "--n_points",
        type=int,
        default=10,
        help="Number of points to compute (default: 10 for quick test)"
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Starting index in the samples list (for parallel processing, default: 0)"
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Noise standard deviation (auto-detected from data if not specified)"
    )
    parser.add_argument(
        "--eps_y_deg",
        type=float,
        default=10.0,
        help="Output tolerance in degrees"
    )
    parser.add_argument(
        "--n_mc",
        type=int,
        default=50000,
        help="Number of MC samples for expectation estimation"
    )
    parser.add_argument(
        "--R_max",
        type=float,
        default=10.0,
        help="Maximum radius to search (default: 10.0 pixels, reasonable for 784-dim MNIST images)"
    )
    parser.add_argument(
        "--n_restarts",
        type=int,
        default=5,
        help="Number of random restarts for PGD"
    )
    parser.add_argument(
        "--pgd_steps",
        type=int,
        default=50,
        help="Number of PGD steps per restart (default: 50)"
    )
    parser.add_argument(
        "--pgd_step_size",
        type=float,
        default=0.01,
        help="PGD step size in raw pixel units (default: 0.01)"
    )
    parser.add_argument(
        "--attack_n_mc",
        type=int,
        default=2000,
        help="Number of noise samples used inside PGD gradient loop (default: 2000; final evaluation still uses --n_mc)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="Device to use ('cpu' or 'cuda')"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file (auto-generated if not specified)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--verbose",
        action='store_true',
        help="Verbose output"
    )
    parser.add_argument(
        "--use_rotation_dataset",
        action='store_true',
        help="Use rotated MNIST dataset (matches estimation if --use_rotation_dataset was used)"
    )
    parser.add_argument(
        "--test_manifest",
        type=str,
        default=None,
        help="Optional rotated test-set manifest NPZ (from experiments/mnist_rotation/generate_rotated_mnist_test_manifest.py). "
             "If provided, images/angles are loaded from the manifest instead of regenerating rotations."
    )
    
    args = parser.parse_args()
    
    # Auto-detect sigma
    if args.sigma is None:
        vg_data = load_json(args.variance_gradient)
        args.sigma = vg_data['parameters']['sigma']
        print(f"Auto-detected σ = {args.sigma} from data file")
    
    eps_y_rad = np.radians(args.eps_y_deg)
    
    print(f"\n{'='*80}")
    print(f"Computing Pseudo-True Radius (Simplified PGD)")
    print(f"{'='*80}")
    print(f"Variance/gradient data: {args.variance_gradient}")
    print(f"Model: {args.model_path}")
    print(f"σ = {args.sigma}")
    print(f"ε_y = {args.eps_y_deg}° = {eps_y_rad:.6f} rad")
    print(f"N_points = {args.n_points}")
    print(f"N_mc = {args.n_mc}")
    print(f"R_max = {args.R_max}")
    print(f"PGD: n_steps={args.pgd_steps}, step_size={args.pgd_step_size}, attack_n_mc={args.attack_n_mc}")
    print(f"Device = {args.device}")
    print(f"{'='*80}\n")
    if args.use_rotation_dataset:
        print("Note: --use_rotation_dataset now loads rotated MNIST samples lazily (one image at a time).")
        print("      This avoids generating the full rotated dataset (which is very slow).\n")
    if args.test_manifest:
        print(f"Using test manifest: {args.test_manifest}\n")
    
    # Load data
    print("Loading data...")
    vg_data = load_json(args.variance_gradient)
    samples = vg_data.get('samples', [])

    # Optional: load rotated images from a manifest (npz) for reproducibility and speed.
    manifest_by_idx = None
    if args.test_manifest:
        npz = np.load(args.test_manifest)
        m_idxs = npz["test_dataset_idx"].astype(int).tolist()
        m_images = npz["images_raw"].astype(np.float32)
        m_angles = npz["true_angle_deg"].astype(np.float32)
        m_labels = npz["digit_label"].astype(int)
        manifest_by_idx = {}
        for j, idx in enumerate(m_idxs):
            manifest_by_idx[int(idx)] = {
                "image_raw": m_images[j],
                "true_angle_deg": float(m_angles[j]),
                "digit_label": int(m_labels[j]),
            }
    
    # Auto-detect if rotated dataset was used
    if not args.use_rotation_dataset:
        use_rotation = vg_data.get('parameters', {}).get('use_rotation_dataset', False)
        if use_rotation:
            print("⚠️  Auto-detected: Estimation used rotated dataset, enabling --use_rotation_dataset")
            args.use_rotation_dataset = True
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path, device=args.device)
    print("✓ Model loaded\n")
    
    # Process samples
    start_idx = args.start_idx
    end_idx = min(start_idx + args.n_points, len(samples))
    n_points = end_idx - start_idx
    
    if start_idx >= len(samples):
        print(f"ERROR: start_idx={start_idx} is out of range (max: {len(samples)-1})")
        return
    
    print(f"Computing pseudo-true radius for {n_points} samples (indices {start_idx} to {end_idx-1})...")
    print(f"Using {'rotated' if args.use_rotation_dataset else 'original'} MNIST dataset\n")
    
    results = []
    
    for i in range(start_idx, end_idx):
        sample = samples[i]
        local_idx = i - start_idx  # Local index within this job (0, 1, 2, ...)
        sample_idx = sample.get('test_dataset_idx', sample.get('sample_idx', i))
        
        if args.verbose:
            print(f"\nSample {local_idx+1}/{n_points} (global_idx={i}, image_idx={sample_idx})")
        
        # Try to load image from manifest / JSON / dataset (in that priority order).
        # We always convert to RAW [0,1] pixel space before optimization.
        image = None
        true_angle_deg = None
        digit_label = sample.get('digit_label', None)

        if manifest_by_idx is not None and int(sample_idx) in manifest_by_idx:
            m = manifest_by_idx[int(sample_idx)]
            image = np.array(m["image_raw"], dtype=np.float32)
            true_angle_deg = float(m["true_angle_deg"])
            digit_label = m["digit_label"]

        if 'image_normalized' in sample:
            image = denormalize_mnist(np.array(sample['image_normalized']))
        elif 'image' in sample:
            image = np.array(sample['image'])
            # If this looks out of [0,1], assume it's normalized MNIST and denormalize.
            if image.min() < -1e-3 or image.max() > 1.1:
                image = denormalize_mnist(image)
        
        # Fall back to loading from dataset
        if image is None:
            image = load_mnist_image(sample_idx, use_rotation_dataset=args.use_rotation_dataset)
        else:
            image = np.clip(image, 0.0, 1.0)
        
        # Compute pseudo-true radius
        R_true, info = compute_pseudo_true_radius_simple(
            model, image, args.sigma, eps_y_rad, args.n_mc,
            args.R_max, args.n_restarts, args.pgd_steps, args.pgd_step_size, args.attack_n_mc,
            args.device, args.seed + i, args.verbose
        )
        
        R_true_norm = float(R_true / MNIST_STD)
        result = {
            'sample_idx': sample_idx,
            'test_dataset_idx': sample_idx,
            'digit_label': digit_label,
            'true_angle_deg': true_angle_deg,
            'true_angle_rad': float(np.radians(true_angle_deg)) if true_angle_deg is not None else None,
            'R_true_raw': float(R_true),
            'R_true_norm': R_true_norm,
            'info': info
        }
        results.append(result)
        
        if args.verbose:
            print(f"  R_true = {R_true:.6f}")
        else:
            if (local_idx + 1) % 5 == 0:
                print(f"  Processed {local_idx + 1}/{n_points} samples...")
    
    print(f"\n✓ Computed pseudo-true radius for {len(results)} samples\n")
    
    # Summary statistics
    R_trues = [r['R_true_raw'] for r in results]
    summary = {
        'n_samples': len(results),
        'mean_R_true': float(np.mean(R_trues)),
        'median_R_true': float(np.median(R_trues)),
        'std_R_true': float(np.std(R_trues)),
        'min_R_true': float(np.min(R_trues)),
        'max_R_true': float(np.max(R_trues))
    }
    
    print("Summary Statistics:")
    print(f"  Mean R_true:   {summary['mean_R_true']:.6f}")
    print(f"  Median R_true: {summary['median_R_true']:.6f}")
    print(f"  Std dev:       {summary['std_R_true']:.6f}")
    print(f"  Range:         [{summary['min_R_true']:.6f}, {summary['max_R_true']:.6f}]\n")
    
    # Save results
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"mnist_pseudo_true_radius_simple_sigma{args.sigma}_eps{args.eps_y_deg}deg_n{args.n_points}_{timestamp}.json"
    
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'variance_gradient_file': args.variance_gradient,
            'model_path': args.model_path,
            'sigma': args.sigma,
            'eps_y_deg': args.eps_y_deg,
            'eps_y_rad': eps_y_rad,
            'n_points': args.n_points,
            'start_idx': args.start_idx,
            'n_mc': args.n_mc,
            'R_max': args.R_max,
            'n_restarts': args.n_restarts,
            'pgd_steps': args.pgd_steps,
            'pgd_step_size': args.pgd_step_size,
            'attack_n_mc': args.attack_n_mc,
            'device': args.device,
            'input_space': 'raw_[0,1]_pixels',
            'model_input_space': 'normalized_(x-0.1307)/0.3081',
            'test_manifest': args.test_manifest,
            'note': 'Pseudo-radius runs PGD/noise in raw pixel space and normalizes only for the model forward pass (to match other certifiers).'
        },
        'summary': summary,
        'results': results
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ Saved results to: {args.output}\n")


if __name__ == "__main__":
    main()

