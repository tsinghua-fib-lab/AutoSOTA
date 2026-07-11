#!/usr/bin/env python3
"""
MNIST Rotation Certification with Alpha-Trimming

Certifies MNIST rotation predictions using the α-trimming method from:
"Certified Adversarial Robustness via Randomized α-Smoothing for Regression Models"
(Rekavandi et al., NeurIPS 2024)

This script runs on the same test images as a previous certification run to enable
fair comparison between methods.

Usage:
    # Using indices from previous JSON:
    python experiments/mnist_rotation/mnist_alpha_trimming_certification.py \\
        --previous_json mnist_rotation_full_cert_n100_20251102_115355.json \\
        --sigma 0.75 --eps_y 10.0 --alpha 0.35
"""

import torch
import numpy as np
import json
import argparse
from pathlib import Path
import sys
from datetime import datetime
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    def tqdm(iterable, desc=None, **kwargs):
        return iterable

# Add paths
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))
sys.path.append(str(Path(__file__).resolve().parent))

from regression_certifiers.certify.alpha_trimming_certifier import AlphaTrimmingCertifier
from dataset_generator import load_mnist_rotation_datasets
from e2cnn_rotation_model import RotationEquivariantCNN_Simple, cos_sin_to_angle


def load_model_and_data(model_path, use_rotation_dataset=False, device='cpu'):
    """
    Load trained model and test dataset.
    
    Args:
        model_path: Path to trained model
        use_rotation_dataset: If True, use rotated MNIST with ground truth angles.
                              If False, use original MNIST (true angle = 0°).
        device: Device to use
    
    Returns:
        model, test_dataset
    """
    import torchvision
    from torchvision import transforms
    
    # Load model
    model = RotationEquivariantCNN_Simple(N=8).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✓ Loaded model from {model_path}")
    
    if use_rotation_dataset:
        # Load MNIST rotation dataset with ground truth angles
        print("Loading MNIST rotation dataset with ground truth angles...")
        _, test_loader = load_mnist_rotation_datasets(
            rotation_range=(0.0, 360.0),
            augmentation_factor=1,
            batch_size=1,
            seed=42
        )
        test_dataset = test_loader.dataset
        print(f"✓ Loaded rotation test dataset ({len(test_dataset)} images with ground truth angles)")
    else:
        # Load original MNIST test set (no rotation, true angle = 0°)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )
        print(f"✓ Loaded original MNIST test dataset ({len(test_dataset)} images, no rotation - true angle = 0°)")
    
    return model, test_dataset


def model_predict_angle(model, image_tensor, device='cpu'):
    """
    Predict rotation angle from image.
    
    Args:
        model: Trained rotation prediction model
        image_tensor: Input image tensor (already normalized)
        device: Device to use
        
    Returns:
        Predicted angle in radians [-π, π]
    """
    model.eval()
    with torch.no_grad():
        # Ensure correct shape: [1, 1, 28, 28]
        if image_tensor.dim() == 2:
            image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
        elif image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        image_tensor = image_tensor.to(device)
        
        # Model outputs (cos θ, sin θ)
        cos_sin = model(image_tensor)
        
        # Convert to angle in radians
        angles_deg = cos_sin_to_angle(cos_sin)
        angles_rad = angles_deg * np.pi / 180.0
        
        return float(angles_rad[0])


def certify_with_alpha_trimming(
    model,
    image_tensor,
    sigma,
    eps_y_deg,
    alpha,
    device,
    n_tr=1000,
    n_sample=16,
    P=0.9,
    seed=None
):
    """
    Certify using alpha-trimming method (corrected implementation).
    
    Following Rekavandi et al., NeurIPS 2024:
    - Uses clean prediction as center (stability certification)
    - Estimates p_A with n_tr trials (NO trimming)
    - Computes q from (alpha, n_sample, P) via Binomial inverse
    - Uses circular distance for angles
    
    Args:
        model: Trained model
        image_tensor: Input image (normalized)
        sigma: Noise standard deviation
        eps_y_deg: Output tolerance in DEGREES (for circular distance)
        alpha: Trimming rate
        device: Device to use
        n_tr: Number of samples for p_A estimation (Clopper-Pearson)
        n_sample: Number of samples for g_alpha / Binomial mapping to q
        P: Target success probability in radius formula
        seed: Random seed
        
    Returns:
        Certified radius (float)
    """
    # Denormalize image for noise addition
    mean_val, std_val = 0.1307, 0.3081
    image_np = image_tensor.squeeze().cpu().numpy()
    image_denorm = image_np * std_val + mean_val
    
    # For GPU efficiency, create a batched model function
    # The certifier will call this many times, so we batch internally
    batch_cache = {'noises': [], 'batch_size': 1000 if device == 'cuda' else 100}
    
    def f_bounded(x):
        """Function from pixel space to angle in DEGREES."""
        # Reshape from (784,) to (28, 28)
        x_2d = x.reshape(28, 28)
        # Renormalize
        x_norm = (x_2d - mean_val) / std_val
        # Predict
        x_tensor = torch.from_numpy(x_norm).float().unsqueeze(0).unsqueeze(0).to(device)
        angle_rad = model_predict_angle(model, x_tensor, device)
        # Convert to degrees for circular distance
        return np.degrees(angle_rad)
    
    def f_bounded_batch(X):
        """Batched version: process multiple inputs at once for GPU efficiency.
        
        Args:
            X: (N, 784) array of flattened images
            
        Returns:
            (N,) array of angles in DEGREES
        """
        N = X.shape[0]
        angles_deg = np.zeros(N)
        
        # Reshape to (N, 28, 28)
        X_2d = X.reshape(N, 28, 28)
        # Normalize
        X_norm = (X_2d - mean_val) / std_val
        
        # Convert to tensor and batch process
        batch_size = 1000 if device == 'cuda' else 100
        model.eval()
        with torch.no_grad():
            for i in range(0, N, batch_size):
                end_idx = min(i + batch_size, N)
                batch_images = X_norm[i:end_idx]
                
                # Convert to tensor [B, 1, 28, 28]
                batch_tensor = torch.from_numpy(batch_images).float().unsqueeze(1).to(device)
                
                # Predict in batch
                pred_cos_sin = model(batch_tensor)
                
                # Convert to angles in degrees
                angles_deg_batch = cos_sin_to_angle(pred_cos_sin).cpu().numpy()
                angles_deg[i:end_idx] = angles_deg_batch
        
        return angles_deg
    
    # Initialize certifier with circular=True for angles
    certifier = AlphaTrimmingCertifier(
        sigma=sigma,
        eps_y=eps_y_deg,
        alpha=alpha,
        n_tr=n_tr,
        n_sample=n_sample,
        confidence=0.95,
        P=P,
        center='pred',  # Clean prediction (stability)
        circular=True   # Use circular distance for angles
    )
    
    # For GPU efficiency, manually batch the p_A estimation when using GPU
    # The certifier calls model_fn one at a time, which is slow on GPU
    z = image_denorm.flatten()
    
    if device == 'cuda' and n_tr > 100:
        # Manually compute p_A with batching for GPU efficiency
        from regression_certifiers.certify.alpha_trimming_certifier import (
            within_eps, clopper_pearson_lower, probability_success_from_alpha, radius_from_probabilities
        )
        
        rng_tr = np.random.default_rng(42 if seed is None else seed)
        
        # Generate all noise samples at once
        noise_samples = rng_tr.normal(0.0, sigma, size=(n_tr, z.shape[0]))
        perturbed_inputs = z[None, :] + noise_samples  # (n_tr, 784)
        
        # Batch process all predictions (much faster on GPU)
        preds_deg = f_bounded_batch(perturbed_inputs)
        
        # Get center (clean prediction)
        center_val = float(f_bounded(z))
        
        # Count how many are within eps_y
        k = sum(within_eps(pred, center_val, eps_y_deg, circular=True) for pred in preds_deg)
        
        # Compute p_A LCB
        pA_lcb = clopper_pearson_lower(k, n_tr, certifier.delta)
        
        # Compute q from (alpha, n_sample, P)
        q = probability_success_from_alpha(alpha, n_sample, P)
        
        # Compute radius
        radius = radius_from_probabilities(pA_lcb, q, sigma)
        
        return float(radius)
    else:
        # Use standard certifier (CPU or small n_tr)
        radius = certifier.certify_point(
            z=z,
            model_fn=f_bounded,
            seed=seed
        )
        return float(radius)


def main():
    parser = argparse.ArgumentParser(
        description='MNIST Rotation Certification with Alpha-Trimming',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model_path", type=str,
                       default="experiments/mnist_rotation/e2cnn_rotation_model.pth",
                       help="Path to trained model")
    parser.add_argument("--previous_json", type=str, default=None,
                       help="Path to previous certification JSON (to reuse indices). If not provided, will select new samples.")
    parser.add_argument("--n_test", type=int, default=100,
                       help="Number of test samples to certify (used if --previous_json not provided)")
    parser.add_argument("--stratified", action="store_true",
                       help="Use stratified sampling by digit class (used if --previous_json not provided)")
    parser.add_argument("--start_idx", type=int, default=0,
                       help="Starting index in test set (used if --previous_json not provided and --stratified not used)")
    parser.add_argument("--use_rotation_dataset", action="store_true",
                       help="Use rotated MNIST dataset with ground truth angles (recommended for proper rotation regression evaluation). If False, uses original MNIST (true angle = 0°).")
    parser.add_argument("--sigma", type=float, default=0.75,
                       help="Noise standard deviation")
    parser.add_argument("--eps_y", type=float, default=10.0,
                       help="Output tolerance in degrees")
    parser.add_argument("--alpha", type=float, default=0.35,
                       help="Trimming rate (fraction to trim from each tail)")
    parser.add_argument("--n_tr", type=int, default=10000,
                       help="Number of samples for p_A estimation (Clopper-Pearson). Should match N used in (E, C)+M and (E, C, G)+M methods (typically 10000)")
    parser.add_argument("--n_sample", type=int, default=500,
                       help="Number of samples for g_alpha / Binomial mapping to q (should be large: n_sample * (1-2*alpha) >= 50-200)")
    parser.add_argument("--P", type=float, default=0.9,
                       help="Target success probability in radius formula")
    parser.add_argument("--device", type=str, default="cpu",
                       choices=["cpu", "cuda", "mps"],
                       help="Device to use")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file path (auto-generated if not provided)")
    
    args = parser.parse_args()
    
    print("="*80)
    print("MNIST ROTATION CERTIFICATION WITH ALPHA-TRIMMING")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Parameters: σ={args.sigma}, ε_y={args.eps_y}°, α={args.alpha}")
    print(f"  n_tr={args.n_tr} (for p_A), n_sample={args.n_sample} (for g_alpha/q), P={args.P}")
    print(f"Device: {args.device}")
    print("="*80)
    
    # Load model and data
    model, test_dataset = load_model_and_data(args.model_path, args.use_rotation_dataset, args.device)
    
    print(f"\nε_y = {args.eps_y}° (using circular distance)")
    if args.use_rotation_dataset:
        print("⚠ Using rotated MNIST dataset - this is recommended for proper rotation regression evaluation!")
    else:
        print("⚠ Using original MNIST (non-rotated) - true angles are 0°. This tests model calibration, not rotation prediction.")
    
    # Initialize variables for comparison (only used if previous_json is provided)
    prev_sigma = None
    prev_eps_y = None
    prev_data = None
    
    # Determine test indices
    if args.previous_json:
        # Load test indices from previous run
        print(f"\nLoading test indices from: {args.previous_json}")
        with open(args.previous_json, 'r') as f:
            prev_data = json.load(f)
        
        # Check if previous run used rotation dataset
        prev_use_rotation = prev_data.get('parameters', {}).get('use_rotation_dataset', False)
        if prev_use_rotation != args.use_rotation_dataset:
            print(f"\n⚠ WARNING: Dataset mismatch!")
            print(f"  Previous run used rotation dataset: {prev_use_rotation}")
            print(f"  Current run uses rotation dataset: {args.use_rotation_dataset}")
            print(f"  This may cause issues if indices don't match between datasets!")
        
        # Extract test indices
        test_indices = prev_data.get('selected_test_indices', None)
        if test_indices is None:
            # Try to extract from samples
            test_indices = [
                s.get('test_dataset_idx', s.get('image_idx', i)) 
                for i, s in enumerate(prev_data.get('samples', []))
            ]
        
        print(f"✓ Loaded {len(test_indices)} test indices from previous run")
        
        # Also extract previous method's parameters for comparison
        prev_params = prev_data.get('parameters', {})
        prev_sigma = prev_params.get('sigma', None)
        prev_eps_y = prev_params.get('eps_y_deg', prev_params.get('eps_y', None))
        
        print(f"\nPrevious run parameters (for comparison):")
        print(f"  σ={prev_sigma}, ε_y={prev_eps_y}°")
        print(f"  Used rotation dataset: {prev_use_rotation}")
    else:
        # Select new test samples
        print(f"\nSelecting {args.n_test} test samples...")
        if args.stratified:
            # Use stratified sampling (similar to mnist_rotation_full_certification.py)
            from torch.utils.data import DataLoader
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            
            n_per_digit = args.n_test // 10
            extra_samples = args.n_test % 10
            
            samples_by_digit = {i: [] for i in range(10)}
            for idx, batch in enumerate(test_loader):
                if args.use_rotation_dataset:
                    # get_original_label returns just the label (int), not a tuple
                    digit_label = test_dataset.get_original_label(idx)
                else:
                    _, digit_label = batch
                    digit_label = digit_label.item() if isinstance(digit_label, torch.Tensor) else digit_label
                
                if len(samples_by_digit[digit_label]) < n_per_digit + (1 if digit_label < extra_samples else 0):
                    samples_by_digit[digit_label].append(idx)
            
            test_indices = []
            for digit in range(10):
                test_indices.extend(samples_by_digit[digit])
            
            print(f"✓ Selected {len(test_indices)} samples using stratified sampling (10 per digit)")
        else:
            # Use consecutive samples starting from start_idx
            test_indices = list(range(args.start_idx, args.start_idx + args.n_test))
            if args.start_idx + args.n_test > len(test_dataset):
                print(f"⚠ WARNING: Requested {args.n_test} samples starting from {args.start_idx}, but dataset only has {len(test_dataset)} samples")
                test_indices = list(range(args.start_idx, len(test_dataset)))
            print(f"✓ Selected {len(test_indices)} consecutive samples starting from index {args.start_idx}")
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Prepare results storage
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'experiment_type': 'mnist_rotation_alpha_trimming',
        'timestamp': timestamp,
        'parameters': {
            'sigma': args.sigma,
            'eps_y_deg': args.eps_y,
            'alpha': args.alpha,
            'n_tr': args.n_tr,
            'n_sample': args.n_sample,
            'P': args.P,
            'center': 'pred',
            'circular': True,
            'seed': args.seed,
            'test_indices': test_indices,
            'use_rotation_dataset': args.use_rotation_dataset
        },
        'comparison': {
            'previous_json': args.previous_json,
            'previous_sigma': prev_sigma if args.previous_json else None,
            'previous_eps_y': prev_eps_y if args.previous_json else None
        },
        'method': 'Alpha-Trimming (adapted from Rekavandi et al., NeurIPS 2024)',
        'samples': []
    }
    
    # Check effective sample size
    n_eff = args.n_sample * (1 - 2 * args.alpha)
    print(f"\nEffective sample size: n_eff = n_sample * (1 - 2*alpha) = {args.n_sample} * (1 - 2*{args.alpha}) = {n_eff:.1f}")
    if n_eff < 50:
        print(f"⚠ WARNING: Effective sample size ({n_eff:.1f}) is below recommended minimum (50-200)")
        print(f"  Consider increasing n_sample or decreasing alpha")
    elif n_eff > 200:
        print(f"✓ Effective sample size ({n_eff:.1f}) is in recommended range (50-200)")
    else:
        print(f"✓ Effective sample size ({n_eff:.1f}) is in recommended range (50-200)")
    
    # Certify each test image
    print(f"\nCertifying {len(test_indices)} test images with α-trimming...")
    print("="*80)
    
    radii = []
    for i, test_idx in enumerate(tqdm(test_indices, desc="Certifying")):
        # Get test image and true angle
        if args.use_rotation_dataset:
            # Rotated dataset returns (image, true_angle_deg)
            image, true_angle_deg = test_dataset[test_idx]
            true_angle_rad = np.radians(true_angle_deg)
            # Get original digit label
            digit_label = test_dataset.get_original_label(test_idx)
        else:
            # Original MNIST returns (image, digit_label)
            image, digit_label = test_dataset[test_idx]
            true_angle_deg = 0.0  # Original MNIST images are not rotated
            true_angle_rad = 0.0
        
        # Get clean prediction
        clean_pred_rad = model_predict_angle(model, image.unsqueeze(0).to(args.device), args.device)
        clean_pred_deg = np.degrees(clean_pred_rad)
        
        # Compute prediction error (circular distance)
        def angdiff_deg(a, b):
            """Compute smallest signed difference a-b in [-180, 180] degrees."""
            return (a - b + 180.0) % 360.0 - 180.0
        
        pred_error_deg = abs(angdiff_deg(clean_pred_deg, true_angle_deg))
        pred_error_rad = abs(angdiff_deg(clean_pred_deg, true_angle_deg) * np.pi / 180.0)
        
        # Certify with alpha-trimming
        print(f"\n[{i+1}/{len(test_indices)}] Image {test_idx} (digit {digit_label}):")
        print(f"  True angle: {true_angle_deg:.2f}°")
        print(f"  Clean prediction: {clean_pred_deg:.2f}°")
        print(f"  Prediction error: {pred_error_deg:.2f}°")
        
        radius = certify_with_alpha_trimming(
            model=model,
            image_tensor=image,
            sigma=args.sigma,
            eps_y_deg=args.eps_y,
            alpha=args.alpha,
            device=args.device,
            n_tr=args.n_tr,
            n_sample=args.n_sample,
            P=args.P,
            seed=args.seed + i
        )
        
        print(f"  Certified radius: {radius:.6f} pixels")
        
        # Store results
        sample_result = {
            'sample_idx': i,
            'test_dataset_idx': test_idx,
            'digit_label': int(digit_label) if isinstance(digit_label, (int, np.integer)) else int(digit_label.item() if hasattr(digit_label, 'item') else digit_label),
            'true_angle_deg': float(true_angle_deg),
            'true_angle_rad': float(true_angle_rad),
            'clean_pred_deg': float(clean_pred_deg),
            'clean_pred_rad': float(clean_pred_rad),
            'pred_error_deg': float(pred_error_deg),
            'pred_error_rad': float(pred_error_rad),
            'certified_radius': float(radius)
        }
        
        results['samples'].append(sample_result)
        radii.append(radius)
    
    # Save results
    if args.output is None:
        output_file = f"mnist_alpha_trimming_n{len(test_indices)}_{timestamp}.json"
    else:
        output_file = args.output
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✓ Results saved to: {output_file}")
    print("="*80)
    
    # Print summary statistics
    radii = np.array(radii)
    
    print("\nSUMMARY STATISTICS:")
    print(f"  Mean radius:      {np.mean(radii):.6f} pixels")
    print(f"  Median radius:    {np.median(radii):.6f} pixels")
    print(f"  Std radius:       {np.std(radii):.6f} pixels")
    print(f"  Min radius:       {np.min(radii):.6f} pixels")
    print(f"  Max radius:       {np.max(radii):.6f} pixels")
    print(f"  Certified (r>0):  {np.sum(radii > 0)}/{len(radii)} "
          f"({100*np.mean(radii > 0):.1f}%)")
    
    # Compare with previous run if available
    if prev_data is not None and 'samples' in prev_data:
        print("\nCOMPARISON WITH PREVIOUS RUN:")
        prev_radii = []
        for s in prev_data['samples']:
            # Try different possible keys for radius
            r = s.get('certified_radius', s.get('r_empirical', s.get('radius', None)))
            if r is not None:
                prev_radii.append(r)
        
        if prev_radii:
            prev_radii = np.array(prev_radii)
            print(f"  Previous method mean:     {np.mean(prev_radii):.6f} pixels")
            print(f"  Alpha-trimming mean:      {np.mean(radii):.6f} pixels")
            print(f"  Difference:               {np.mean(radii) - np.mean(prev_radii):.6f} pixels")
            print(f"  Relative difference:      {100*(np.mean(radii) - np.mean(prev_radii))/np.mean(prev_radii):.2f}%")
    
    print("="*80)


if __name__ == '__main__':
    main()

