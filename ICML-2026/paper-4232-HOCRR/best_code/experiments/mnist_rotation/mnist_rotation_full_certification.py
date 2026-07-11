#!/usr/bin/env python3
"""
Full MNIST Rotation Certification Experiment

Certifies multiple MNIST rotation test images to demonstrate practical performance
of the bounded certifier on a real regression task.

This complements the single-sample convergence analysis with statistics over multiple samples.

Usage:
    # Certify 100 test samples with N=[1000, 5000, 10000]
    python experiments/mnist_rotation/mnist_rotation_full_certification.py --n_test 100 --n_trials 5
    
    # Quick test with 10 samples
    python experiments/mnist_rotation/mnist_rotation_full_certification.py --n_test 10 --n_trials 3
    
    # Full experiment with 1000 samples
    python experiments/mnist_rotation/mnist_rotation_full_certification.py --n_test 1000 --n_trials 5
"""

import torch
import numpy as np
import json
import argparse
from pathlib import Path
import sys
from datetime import datetime
from tqdm import tqdm

# Add paths
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))
sys.path.append(str(Path(__file__).resolve().parent))

from regression_certifiers.certify.variance_gradient_certifier import VarianceGradientCertifier
from bounded_certifier_convergence_analysis import BoundedCertifierConvergenceValidator
from dataset_generator import load_mnist_rotation_datasets
from e2cnn_rotation_model import RotationEquivariantCNN_Simple, cos_sin_to_angle


def load_model_and_data(model_path, use_rotation_dataset=False, device='cpu'):
    """Load trained model and test dataset."""
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
        print(f"✓ Loaded rotation test dataset ({len(test_loader.dataset)} images with ground truth angles)")
    else:
        # Load original MNIST test set (no ground truth angles)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )
        
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
        print(f"✓ Loaded MNIST test dataset ({len(test_dataset)} images, no ground truth angles)")
    
    return model, test_loader


def model_predict_angle(model, image_normalized_np, device='cpu'):
    """
    Predict angle from NORMALIZED image (numpy array).
    
    Args:
        model: Trained e2cnn model
        image_normalized_np: (28, 28) numpy array, already normalized with mean=0.1307, std=0.3081
        device: Device
        
    Returns:
        Predicted angle in radians [-π, π]
    """
    # Convert to tensor [1, 1, 28, 28]
    image_tensor = torch.from_numpy(image_normalized_np).float().unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_cos_sin = model(image_tensor)
    
    # Get angle in degrees [-180, 180]
    angle_degrees = cos_sin_to_angle(pred_cos_sin).item()
    
    # Convert to radians [-π, π]
    angle_radians = np.radians(angle_degrees)
    return angle_radians


def model_predict_angles_batch(model, images_normalized_np, device='cpu', batch_size=1000):
    """
    Predict angles from NORMALIZED images in batches for GPU efficiency.
    
    Args:
        model: Trained e2cnn model
        images_normalized_np: (N, 28, 28) numpy array, already normalized
        device: Device
        batch_size: Batch size for processing (larger for GPU)
        
    Returns:
        Array of predicted angles in radians [-π, π], shape (N,)
    """
    N = images_normalized_np.shape[0]
    angles_rad = np.zeros(N)
    
    model.eval()
    with torch.no_grad():
        for i in range(0, N, batch_size):
            end_idx = min(i + batch_size, N)
            batch_images = images_normalized_np[i:end_idx]
            
            # Convert to tensor [B, 1, 28, 28]
            batch_tensor = torch.from_numpy(batch_images).float().unsqueeze(1).to(device)
            
            # Predict in batch
            pred_cos_sin = model(batch_tensor)
            
            # Convert to angles
            angles_deg = cos_sin_to_angle(pred_cos_sin).cpu().numpy()
            angles_rad[i:end_idx] = np.radians(angles_deg)
    
    return angles_rad


def select_stratified_samples(test_loader, n_samples, use_rotation_dataset=False):
    """
    Select samples using stratified sampling by digit class.
    
    Args:
        test_loader: DataLoader for test dataset
        n_samples: Total number of samples to select
        use_rotation_dataset: Whether using rotation dataset with ground truth
        
    Returns:
        List of sample indices evenly distributed across digit classes
    """
    n_per_digit = n_samples // 10  # 10 digit classes
    extra_samples = n_samples % 10  # Extra samples to reach n_samples
    
    print(f"Stratified sampling: {n_per_digit} samples per digit class")
    if extra_samples > 0:
        print(f"  + {extra_samples} additional samples from first {extra_samples} classes")
    
    # Collect samples by digit class
    samples_by_digit = {i: [] for i in range(10)}
    
    for idx, batch in enumerate(test_loader):
        if use_rotation_dataset:
            # Rotation dataset: need to get original label
            original_label = test_loader.dataset.get_original_label(idx)
        else:
            # Original MNIST: label is in batch[1]
            _, labels = batch
            original_label = labels[0].item()
        
        # Add to appropriate digit class
        if len(samples_by_digit[original_label]) < n_per_digit:
            samples_by_digit[original_label].append(idx)
        
        # Check if we have enough samples for all digits
        if all(len(v) >= n_per_digit for v in samples_by_digit.values()):
            # Add extra samples if needed
            if extra_samples > 0:
                for digit in range(extra_samples):
                    if len(samples_by_digit[digit]) == n_per_digit:
                        # Need one more for this digit
                        for idx2 in range(idx + 1, len(test_loader.dataset)):
                            if use_rotation_dataset:
                                label2 = test_loader.dataset.get_original_label(idx2)
                            else:
                                _, labels2 = test_loader.dataset[idx2]
                                label2 = labels2.item() if isinstance(labels2, torch.Tensor) else labels2
                            
                            if label2 == digit and idx2 not in samples_by_digit[digit]:
                                samples_by_digit[digit].append(idx2)
                                break
            break
    
    # Flatten the dict to get list of indices
    selected_indices = []
    for digit in range(10):
        selected_indices.extend(samples_by_digit[digit])
    
    # Sort indices for sequential processing
    selected_indices.sort()
    
    print(f"✓ Selected {len(selected_indices)} samples:")
    for digit in range(10):
        print(f"  Digit {digit}: {len(samples_by_digit[digit])} samples")
    
    return selected_indices


def certify_one_sample(model, test_image, validator, N_values, n_trials, 
                       sigma, device, seed, true_angle_deg=None, skip_bootstrap=False):
    """
    Certify one test sample with multiple N values.
    
    Uses UNBOUNDED VarianceGradientCertifier (theoretical guarantees).
    Saves variance and gradient norm estimates (with CIs) for later radius computation.
    
    Args:
        model: Trained model
        test_image: Test image (28, 28) in [0, 1] pixel space
        validator: BoundedCertifierConvergenceValidator (for U-statistic estimators)
        N_values: List of sample sizes to test
        n_trials: Number of trials per N
        sigma: Noise standard deviation
        device: Device to use
        seed: Random seed
        true_angle_deg: Optional ground truth angle (if available from rotation dataset)
    
    Returns dictionary with results for all N values (variance and gradient estimates only).
    """
    # Flatten the test image
    test_image_flat = test_image.flatten()  # (784,)
    
    # Define unbounded scalar function (returns angle in radians)
    def f_unbounded(img_flat):
        """Unbounded scalar function: takes flattened [0,1] image, returns angle in radians."""
        img_2d = img_flat.reshape(28, 28)
        img_2d = np.clip(img_2d, 0.0, 1.0)
        img_normalized = (img_2d - 0.1307) / 0.3081
        angle_rad = model_predict_angle(model, img_normalized, device=device)
        return angle_rad  # No clipping - unbounded
    
    # Get clean prediction for this sample
    test_image_normalized = (test_image - 0.1307) / 0.3081
    clean_pred_rad = model_predict_angle(model, test_image_normalized, device)
    clean_pred_deg = np.degrees(clean_pred_rad)
    
    # Results for this sample
    sample_results = {
        'clean_pred_rad': float(clean_pred_rad),
        'clean_pred_deg': float(clean_pred_deg),
        'results_by_N': {N: [] for N in N_values}
    }
    
    # Add ground truth angle if available
    if true_angle_deg is not None:
        sample_results['true_angle_deg'] = float(true_angle_deg)
        sample_results['clean_error_deg'] = float(abs(clean_pred_deg - true_angle_deg))
    
    trial_count = 0
    for N in N_values:
        for i in range(n_trials):
            trial_seed = seed + trial_count if seed is not None else None
            rng = np.random.default_rng(trial_seed)
            
            # Generate pixel-wise noise with INDEPENDENT SAMPLING
            eta_samples = rng.normal(0.0, sigma, size=(N, 784))
            
            # Evaluate function using batched processing for GPU efficiency
            # Prepare batched inputs: (N, 784) -> (N, 28, 28)
            perturbed_images_flat = test_image_flat[None, :] + eta_samples  # (N, 784)
            perturbed_images = perturbed_images_flat.reshape(N, 28, 28)  # (N, 28, 28)
            perturbed_images = np.clip(perturbed_images, 0.0, 1.0)
            
            # Normalize
            perturbed_images_normalized = (perturbed_images - 0.1307) / 0.3081
            
            # Predict in batches (much faster on GPU)
            batch_size = 1000 if device == 'cuda' else 100  # Larger batches for GPU
            f_values = model_predict_angles_batch(model, perturbed_images_normalized, device=device, batch_size=batch_size)
            
            # Estimate variance using U-statistic (with two types of CIs)
            C_hat_analytical, C_lower_analytical, C_upper_analytical = \
                validator.u_statistic_variance_estimator_alpha_half(f_values)
            
            # Also get bootstrap CI for variance (optional, can be slow)
            if skip_bootstrap:
                C_hat_bootstrap = C_hat_analytical
                C_lower_bootstrap = C_lower_analytical
                C_upper_bootstrap = C_upper_analytical
            else:
                # Reduce B for speed if N is large
                B_bootstrap = 500 if N >= 10000 else 1000
                C_hat_bootstrap, C_lower_bootstrap, C_upper_bootstrap = \
                    validator.u_statistic_variance_estimator_bootstrap(f_values, B=B_bootstrap, rng=rng)
            
            # Estimate θ = ||G||² using U-statistic with z-critical CI
            theta_hat, theta_lower, theta_upper = validator.compute_theta_ci_with_z_critical(
                f_values, eta_samples, confidence=validator.confidence
            )
            
            # Estimate gradient norm (||G||) from variance + gradient estimator
            # This gives point estimate and CIs for ||G|| directly
            G_norm_hat, G_norm_lower, G_norm_upper = \
                validator.u_statistic_gradient_norm_estimator_alpha_half(f_values, eta_samples)
            
            # Store results (estimates only - radius computed later from saved estimates)
            trial_result = {
                # Variance estimators (two types of CIs)
                'C_hat': float(C_hat_analytical),
                'C_lower_analytical': float(C_lower_analytical),
                'C_upper_analytical': float(C_upper_analytical),
                'C_lower_bootstrap': float(C_lower_bootstrap),
                'C_upper_bootstrap': float(C_upper_bootstrap),
                
                # Squared gradient norm (θ = ||G||²)
                'theta_hat': float(theta_hat),
                'theta_lower': float(theta_lower),
                'theta_upper': float(theta_upper),
                
                # Gradient norm (||G||) - direct estimate
                'G_norm_hat': float(G_norm_hat),
                'G_norm_lower': float(G_norm_lower),
                'G_norm_upper': float(G_norm_upper),
                
                # Mean function value (for reference)
                'g_z_hat': float(np.mean(f_values)),
                
                # Metadata
                'N_samples': N,
                'trial': i,
                
                # Note: Radius NOT computed here - will be computed later from saved estimates
                # for any desired eps_y using VarianceGradientCertifier.variance_gradient_certificate()
            }
            
            sample_results['results_by_N'][N].append(trial_result)
            trial_count += 1
    
    return sample_results


def main():
    parser = argparse.ArgumentParser(description="MNIST rotation full certification")
    parser.add_argument("--model_path", type=str, 
                       default="experiments/mnist_rotation/e2cnn_rotation_model.pth",
                       help="Path to trained model")
    parser.add_argument("--n_test", type=int, default=100,
                       help="Number of test samples to certify")
    parser.add_argument("--start_idx", type=int, default=0,
                       help="Starting index in test set (ignored if using stratified sampling)")
    parser.add_argument("--sigma", type=float, default=0.75,
                       help="Noise standard deviation")
    parser.add_argument("--eps_y", type=float, default=10.0,
                       help="Output tolerance in degrees")
    parser.add_argument("--N_values", nargs="+", type=int,
                       default=[1000, 5000, 10000],
                       help="Sample sizes to use for certification")
    parser.add_argument("--n_trials", type=int, default=5,
                       help="Number of trials per sample per N")
    parser.add_argument("--use_rotation_dataset", action="store_true",
                       help="Use rotation dataset with ground truth angles")
    parser.add_argument("--stratified", action="store_true",
                       help="Use stratified sampling by digit class")
    parser.add_argument("--device", type=str, default="cpu",
                       choices=["cpu", "cuda", "mps"],
                       help="Device to use")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file path (auto-generated if not provided)")
    parser.add_argument("--skip_bootstrap", action="store_true",
                       help="Skip bootstrap CI computation (faster, but less robust)")
    parser.add_argument("--confidence", type=float, default=0.95,
                       help="Confidence level for confidence intervals (default: 0.95 = 5%% failure probability). Use 0.90 for 10%% failure probability.")
    
    args = parser.parse_args()
    
    print("="*80)
    print("MNIST ROTATION FULL CERTIFICATION EXPERIMENT")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Test samples: {args.n_test}")
    if args.stratified:
        print(f"Sampling: Stratified (balanced across digit classes)")
    else:
        print(f"Sampling: Sequential (starting from index {args.start_idx})")
    print(f"Dataset: {'Rotation dataset (with ground truth angles)' if args.use_rotation_dataset else 'Original MNIST (no ground truth)'}")
    print(f"Parameters: σ={args.sigma}, ε_y={args.eps_y}°")
    print(f"Confidence level: {args.confidence:.2f} (failure probability: {(1-args.confidence)*100:.1f}%%)")
    print(f"N values: {args.N_values}")
    print(f"Trials per sample: {args.n_trials}")
    print(f"Device: {args.device}")
    total_evaluations = args.n_test * len(args.N_values) * args.n_trials * sum(args.N_values)
    print(f"Total certifications: {args.n_test} samples × {len(args.N_values)} N values × {args.n_trials} trials = {args.n_test * len(args.N_values) * args.n_trials}")
    print(f"Total model evaluations: ~{total_evaluations:,} (this determines runtime)")
    if not args.skip_bootstrap:
        print(f"⚠️  Bootstrap CI enabled - this adds significant overhead. Use --skip_bootstrap for faster runs.")
    print("="*80)
    
    # Load model and data
    model, test_loader = load_model_and_data(args.model_path, args.use_rotation_dataset, args.device)
    
    # Initialize validator (eps_y not needed for estimates, but kept for compatibility)
    eps_y_rad = np.radians(args.eps_y)  # Saved for reference, but not used in unbounded certifier
    validator = BoundedCertifierConvergenceValidator(sigma=args.sigma, eps_y=eps_y_rad, confidence=args.confidence)
    
    # Results container
    results = {
        'experiment_type': 'mnist_rotation_full_certification',
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'sigma': args.sigma,
            'eps_y_deg': args.eps_y,  # Saved for reference only - radius computed later from estimates
            'eps_y_rad': eps_y_rad,
            'method': 'unbounded_variance_gradient',  # Uses VarianceGradientCertifier (theoretical guarantees)
            'N_values': args.N_values,
            'n_trials': args.n_trials,
            'n_test': args.n_test,
            'start_idx': args.start_idx,
            'seed': args.seed,
            'confidence': args.confidence,
            'failure_probability': 1.0 - args.confidence,
            'use_rotation_dataset': args.use_rotation_dataset,
            'stratified': args.stratified,
            'sampling_note': 'Stratified: 10 samples per digit class' if args.stratified else f'Sequential: indices {args.start_idx} to {args.start_idx + args.n_test - 1}',
            'note': 'This run saves variance and gradient norm estimates only. Radius can be computed later for any eps_y using VarianceGradientCertifier.variance_gradient_certificate()'
        },
        'samples': [],
        'selected_test_indices': []  # Will be populated with actual test dataset indices
    }
    
    # Select samples (stratified or sequential)
    if args.stratified:
        print("\nSelecting samples using stratified sampling...")
        selected_indices = select_stratified_samples(test_loader, args.n_test, args.use_rotation_dataset)
    else:
        print(f"\nUsing sequential sampling starting from index {args.start_idx}...")
        selected_indices = list(range(args.start_idx, min(args.start_idx + args.n_test, len(test_loader.dataset))))
    
    # Save the selected indices for reproducibility
    results['selected_test_indices'] = selected_indices
    print(f"Selected test dataset indices: {selected_indices[:10]}...{selected_indices[-3:]}" if len(selected_indices) > 13 else f"Selected test dataset indices: {selected_indices}")
    
    # Certify test samples
    print(f"\nCertifying {len(selected_indices)} test samples...")
    print("(This may take a while - each sample requires many model evaluations)")
    print("="*80)
    for sample_idx, dataset_idx in enumerate(tqdm(selected_indices, desc="Certifying samples")):
        # Get sample from dataset
        if args.use_rotation_dataset:
            # Rotation dataset returns (image, angle)
            images, true_angle_deg = test_loader.dataset[dataset_idx]
            # Get original digit label
            digit_label = test_loader.dataset.get_original_label(dataset_idx)
        else:
            # Original MNIST returns (image, label)
            images, digit_label = test_loader.dataset[dataset_idx]
            true_angle_deg = None
            digit_label = digit_label.item() if isinstance(digit_label, torch.Tensor) else digit_label
        
        # Get denormalized image [0, 1]
        if images.dim() == 3:  # [C, H, W]
            test_image_normalized = images[0].cpu().numpy()  # (28, 28)
        else:  # [H, W]
            test_image_normalized = images.cpu().numpy()  # (28, 28)
        test_image = test_image_normalized * 0.3081 + 0.1307  # Denormalize
        test_image = np.clip(test_image, 0.0, 1.0)
        
        # Certify this sample
        sample_seed = args.seed + dataset_idx * 10000  # Unique seed per sample
        sample_results = certify_one_sample(
            model, test_image, validator, args.N_values, args.n_trials,
            args.sigma, args.device, sample_seed, true_angle_deg, args.skip_bootstrap
        )
        
        # Add metadata
        sample_results['test_dataset_idx'] = dataset_idx  # Actual index in test dataset
        sample_results['sample_idx'] = sample_idx  # Index in selected samples (0, 1, 2, ...)
        sample_results['digit_label'] = digit_label
        
        # Legacy field for backwards compatibility
        sample_results['image_idx'] = dataset_idx
        
        results['samples'].append(sample_results)
    
    # Save results
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"mnist_rotation_full_cert_n{args.n_test}_{timestamp}.json"
    else:
        output_file = args.output

    # Ensure output directory exists
    output_path = Path(output_file)
    if output_path.parent != Path(""):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("CERTIFICATION COMPLETE!")
    print("="*80)
    print(f"\n✓ Saved results: {output_file}")
    print(f"✓ Certified {len(results['samples'])} samples")
    print(f"✓ Each sample certified with N={args.N_values} × {args.n_trials} trials")
    
    # Compute summary statistics (variance and gradient estimates)
    print("\n📊 SUMMARY STATISTICS (Estimates Only - Radius computed later):")
    for N in args.N_values:
        C_hats = []
        G_norm_hats = []
        for sample in results['samples']:
            for trial in sample['results_by_N'][N]:
                C_hats.append(trial['C_hat'])
                G_norm_hats.append(trial['G_norm_hat'])
        
        C_hats = np.array(C_hats)
        G_norm_hats = np.array(G_norm_hats)
        
        print(f"\nN = {N}:")
        print(f"  Variance (C):")
        print(f"    Mean: {np.mean(C_hats):.6f} ± {np.std(C_hats):.6f}")
        print(f"    Median: {np.median(C_hats):.6f}")
        print(f"    Range: [{np.min(C_hats):.6f}, {np.max(C_hats):.6f}]")
        print(f"  Gradient Norm (||G||):")
        print(f"    Mean: {np.mean(G_norm_hats):.6f} ± {np.std(G_norm_hats):.6f}")
        print(f"    Median: {np.median(G_norm_hats):.6f}")
        print(f"    Range: [{np.min(G_norm_hats):.6f}, {np.max(G_norm_hats):.6f}]")
    
    print("\n" + "="*80)
    print("Next steps:")
    print(f"  1. Compute radii for any desired eps_y from saved estimates:")
    print(f"     Use VarianceGradientCertifier.variance_gradient_certificate(C_ucb, G_ucb, eps_y)")
    print(f"  2. Postprocess results with scripts in experiments/mnist_rotation/")
    print(f"  3. Compare N={args.N_values[0]} vs N={args.N_values[-1]}")
    print("="*80)


if __name__ == "__main__":
    main()

