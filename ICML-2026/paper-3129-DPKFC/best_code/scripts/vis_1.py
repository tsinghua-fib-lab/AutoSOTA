import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from kfac_functs import MLP, CNN, KFACRecorder, compute_covariances


def get_pink_noise_batch(batch_size, channels, img_size, device, alpha=1.0):
    """
    Generates 'Pink' noise (1/f decay) to mimic natural image statistics.

    White noise has uncorrelated pixels (A ≈ Identity matrix).
    Pink noise has spatially correlated pixels that mimic the power spectrum of natural images.

    Args:
        batch_size: Number of images to generate
        channels: Number of channels (1 for grayscale, 3 for RGB)
        img_size: Image dimensions (assumes square)
        device: torch device
        alpha: Spectral decay factor
               0.0 = White Noise (uncorrelated)
               1.0 = Pink Noise (natural image-like)
               2.0 = Brownian Noise (very smooth/cloudy)

    Returns:
        Tensor of shape [batch_size, channels, img_size, img_size]
    """
    # 1. Generate White Noise in Frequency Domain (complex for phase)
    white_noise_freq = torch.rand(batch_size, channels, img_size, img_size, dtype=torch.cfloat, device=device)

    # 2. Create Frequency Grid (fx, fy)
    freqs = torch.fft.fftfreq(img_size, device=device)
    fx, fy = torch.meshgrid(freqs, freqs, indexing='ij')

    # Calculate magnitude of frequency f = sqrt(fx^2 + fy^2)
    f = torch.sqrt(fx**2 + fy**2)

    # Avoid division by zero at DC component (f=0)
    f[0, 0] = 1.0

    # 3. Apply 1/f^alpha scaling (scale amplitude)
    scale = 1.0 / (f ** alpha)
    scale[0, 0] = 0.0  # Zero out DC component (mean center the image)

    # Reshape scale to broadcast: [1, 1, H, W]
    scale = scale.view(1, 1, img_size, img_size)

    # 4. Filter the noise in frequency domain
    pink_noise_freq = white_noise_freq * scale

    # 5. Inverse FFT to get spatial domain
    pink_noise = torch.fft.ifft2(pink_noise_freq).real

    # 6. Normalize per-image (instance normalization) to ensure each sample has ~unit variance
    std = pink_noise.view(batch_size, -1).std(dim=1, keepdim=True)
    pink_noise = pink_noise / (std.view(batch_size, 1, 1, 1) + 1e-8) * 0.5

    return pink_noise


def get_kfac_eigenvalues(model, data_loader, recorder, noise_type=None, device='cuda'):
    """
    Computes the eigenvalues of the KFAC approx (A \\otimes G) for all layers.

    Args:
        model: The neural network model
        data_loader: DataLoader for real data (used for shape reference even with noise)
        recorder: KFACRecorder instance
        noise_type: None for real data, 'white' for white noise, 'pink' for pink noise
        device: torch device
    """
    model.eval()
    model.zero_grad()

    # 1. Accumulate Statistics
    # We only need a few batches to estimate the curvature shape
    for i, (data, target) in enumerate(data_loader):
        if i > 100: break # 100 batches is enough for spectrum estimation
        data = data.to(device)

        if noise_type == 'white':
            # Overwrite data with White Noise (uncorrelated)
            data = torch.rand_like(data)
            output, _ = model(data)
            target = torch.distributions.Categorical(logits=output).sample()
        elif noise_type == 'pink':
            # Overwrite data with Pink Noise (spatially correlated, 1/f spectrum)
            batch_size, channels, h, w = data.shape
            data = get_pink_noise_batch(batch_size, channels, h, device=device, alpha=1.0)
            output, _ = model(data)
            target = torch.distributions.Categorical(logits=output).sample()
        else:
            # Real Data: Use model predictive distribution (True Fisher)
            # or ground truth (Empirical Fisher).
            # Standard KFAC uses model predictions.
            output, _ = model(data)
            target = torch.distributions.Categorical(logits=output).sample()

        # Backward to capture 'G' (backprops)
        loss = torch.nn.CrossEntropyLoss()(output, target)
        loss.backward()

    # 2. Compute Covariances
    cov_A, cov_G = compute_covariances(model, recorder.activations, recorder.backprops)

    # 3. Compute Eigenvalues for all layers
    layer_evals = {}
    for layer_name in cov_A.keys():
        print(f"Layer: {layer_name}, Num Eigenvalues: {cov_A[layer_name].shape[0] * cov_G[layer_name].shape[0]}")

        # Eigendecomposition of A
        evals_A = torch.linalg.eigvalsh(cov_A[layer_name]).cpu().numpy()
        # Eigendecomposition of G
        evals_G = torch.linalg.eigvalsh(cov_G[layer_name]).cpu().numpy()

        # Outer product to get full KFAC eigenvalues
        # (lambda_i * mu_j) for all i, j
        kfac_evals = np.outer(evals_A, evals_G).flatten()

        # Sort descending
        kfac_evals.sort()
        layer_evals[layer_name] = kfac_evals[::-1]

    return layer_evals


def get_kfac_eigenvalues_hybrid(model, data_loader, recorder, a_source='public', g_source='public_noise',
                                 num_classes=10, device='cuda'):
    """
    Computes KFAC eigenvalues with separate control over A and G sources.

    This implements the three main preconditioner types:
    - 'public': A from public data, G from public data + public labels (full public)
    - 'hybrid': A from public data, G from public data + random labels (no task supervision)
    - 'pink_noise': A from pink noise, G from pink noise + random labels (full noise)

    Args:
        model: The neural network model
        data_loader: DataLoader for data (public or private)
        recorder: KFACRecorder instance
        a_source: 'public' or 'pink' - source for activation covariances
        g_source: 'public' (real labels), 'public_noise' (random labels), or 'pink_noise'
        num_classes: Number of classes for random label generation
        device: torch device
    """
    model.eval()
    model.zero_grad()

    # We need separate passes for A and G if sources differ
    # For simplicity, we'll do a combined pass when possible

    for i, (data, target) in enumerate(data_loader):
        if i > 100: break
        data = data.to(device)
        target = target.to(device)
        batch_size, channels, h, w = data.shape

        # Determine input data for this batch
        if a_source == 'pink':
            input_data = get_pink_noise_batch(batch_size, channels, h, device=device, alpha=1.0)
        else:  # 'public'
            input_data = data

        # Forward pass
        output, _ = model(input_data)

        # Determine labels for backward pass (affects G)
        if g_source == 'public':
            # Use real public labels
            labels = target
        elif g_source == 'public_noise' or g_source == 'pink_noise':
            # Use random labels (no task supervision)
            labels = torch.randint(0, num_classes, (batch_size,), device=device)
        else:
            labels = target

        # Backward pass
        loss = torch.nn.CrossEntropyLoss()(output, labels)
        loss.backward()

    # Compute covariances
    cov_A, cov_G = compute_covariances(model, recorder.activations, recorder.backprops)

    # Compute eigenvalues
    layer_evals = {}
    for layer_name in cov_A.keys():
        evals_A = torch.linalg.eigvalsh(cov_A[layer_name]).cpu().numpy()
        evals_G = torch.linalg.eigvalsh(cov_G[layer_name]).cpu().numpy()
        kfac_evals = np.outer(evals_A, evals_G).flatten()
        kfac_evals.sort()
        layer_evals[layer_name] = kfac_evals[::-1]

    return layer_evals

def compute_spectrum_data(model, train_loader, recorder, device='cuda'):
    """
    Compute KFAC eigenvalue spectra for different preconditioner types.

    Returns spectra for:
    - Oracle (private data)
    - Public Full Fashion MNIST (A_pub + G_pub with real labels)
    - Public Full CIFAR-10 (A_pub + G_pub with real labels) - shows domain mismatch
    - Hybrid (Fashion MNIST images + random labels)
    - Pink Noise Full (pink noise images + random labels)
    """
    # Load Fashion MNIST as public data (good domain match for MNIST)
    transform = transforms.Compose([transforms.ToTensor()])
    fashion_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    fashion_loader = DataLoader(fashion_dataset, batch_size=512, shuffle=True)

    # Load CIFAR-10 as alternative public data (poor domain match - natural images vs digits)
    cifar_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),  # Convert RGB to grayscale
        transforms.Resize((28, 28)),  # Resize to match MNIST
    ])
    cifar_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=cifar_transform)
    cifar_loader = DataLoader(cifar_dataset, batch_size=512, shuffle=True)

    # 1. Oracle: Private data (MNIST) - upper bound
    print("Computing Oracle Spectrum (Private MNIST)...")
    recorder.activations.clear(); recorder.backprops.clear()
    oracle_evals = get_kfac_eigenvalues(model, train_loader, recorder, noise_type=None, device=device)

    # 2. Public Full Fashion MNIST: A_pub + G_pub (Fashion MNIST with real labels)
    # Good domain match: grayscale, centered objects, similar spatial structure
    print("Computing Public Full Spectrum (Fashion MNIST + real labels)...")
    recorder.activations.clear(); recorder.backprops.clear()
    public_fashion_evals = get_kfac_eigenvalues_hybrid(
        model, fashion_loader, recorder,
        a_source='public', g_source='public',
        num_classes=10, device=device
    )

    # 3. Public Full CIFAR-10: A_pub + G_pub (CIFAR-10 with real labels)
    # Poor domain match: natural images, textures, different spatial structure
    print("Computing Public Full Spectrum (CIFAR-10 + real labels)...")
    recorder.activations.clear(); recorder.backprops.clear()
    public_cifar_evals = get_kfac_eigenvalues_hybrid(
        model, cifar_loader, recorder,
        a_source='public', g_source='public',
        num_classes=10, device=device
    )

    # 4. Hybrid: A_pub + G_pub_noise (Fashion MNIST images + random labels)
    print("Computing Hybrid Spectrum (Fashion MNIST + random labels)...")
    recorder.activations.clear(); recorder.backprops.clear()
    hybrid_evals = get_kfac_eigenvalues_hybrid(
        model, fashion_loader, recorder,
        a_source='public', g_source='public_noise',
        num_classes=10, device=device
    )

    # 5. Pink Noise Full: A_pink + G_pink_noise (pink noise + random labels)
    print("Computing Pink Noise Full Spectrum...")
    recorder.activations.clear(); recorder.backprops.clear()
    pink_noise_evals = get_kfac_eigenvalues_hybrid(
        model, train_loader, recorder,  # Use train_loader just for shape reference
        a_source='pink', g_source='pink_noise',
        num_classes=10, device=device
    )

    return oracle_evals, public_fashion_evals, public_cifar_evals, hybrid_evals, pink_noise_evals

def compute_condition_number(evals, eps=1e-10):
    """Compute condition number from eigenvalues (max/min of positive evals)."""
    positive_evals = evals[evals > eps]
    if len(positive_evals) < 2:
        return float('inf')
    return positive_evals[0] / positive_evals[-1]  # Already sorted descending


def plot_unified_spectrum(mlp_data, cnn_data):
    """
    Plot KFAC eigenvalue spectra for different preconditioner types.

    Shows 5 curves per layer:
    - Oracle (private data) - blue solid
    - Public Fashion MNIST (A_pub + G_pub) - green solid (good domain match)
    - Public CIFAR-10 (A_pub + G_pub) - red solid (poor domain match)
    - Hybrid (A_pub + G_pub_noise) - orange dashed
    - Pink Noise (A_pink + G_pink_noise) - purple dash-dot
    """
    print("Creating unified plot...")

    # Create 3x2 subplot (3 rows, 2 columns) - MLP in first column, CNN in second column
    fig, axes = plt.subplots(3, 2, figsize=(6, 7))

    # Extract MLP data (first column)
    mlp_oracle = mlp_data['oracle_evals']
    mlp_public_fashion = mlp_data['public_fashion_evals']
    mlp_public_cifar = mlp_data['public_cifar_evals']
    mlp_hybrid = mlp_data['hybrid_evals']
    mlp_pink_noise = mlp_data['pink_noise_evals']
    mlp_layers = mlp_data['fc_layers']

    # Extract CNN data (second column)
    cnn_oracle = cnn_data['oracle_evals']
    cnn_public_fashion = cnn_data['public_fashion_evals']
    cnn_public_cifar = cnn_data['public_cifar_evals']
    cnn_hybrid = cnn_data['hybrid_evals']
    cnn_pink_noise = cnn_data['pink_noise_evals']
    cnn_layers = cnn_data['conv_layers'] + cnn_data['fc_layers']  # conv1, conv2, fc1 (excluding fc2)

    print(f"MLP layers: {mlp_layers}")
    print(f"CNN layers: {cnn_layers}")

    # Plot MLP layers (first column)
    for i, layer_name in enumerate(mlp_layers):
        print(f"Plotting MLP layer {i}: {layer_name}")
        ax = axes[i, 0]  # First column

        # Get data for this layer (full arrays for condition number)
        oracle_full = mlp_oracle[layer_name]
        public_fashion_full = mlp_public_fashion[layer_name]
        public_cifar_full = mlp_public_cifar[layer_name]
        hybrid_full = mlp_hybrid[layer_name]
        pink_noise_full = mlp_pink_noise[layer_name]

        # Compute condition numbers
        kappa_oracle = compute_condition_number(oracle_full)
        kappa_fashion = compute_condition_number(public_fashion_full)
        kappa_cifar = compute_condition_number(public_cifar_full)
        kappa_hybrid = compute_condition_number(hybrid_full)
        kappa_pink = compute_condition_number(pink_noise_full)

        # Find best and worst match (smallest/largest ratio to oracle)
        kappas = {
            'Fashion': kappa_fashion,
            'CIFAR': kappa_cifar,
            'Hybrid': kappa_hybrid,
            'Pink': kappa_pink,
        }
        ratios = {k: abs(np.log10(v) - np.log10(kappa_oracle)) for k, v in kappas.items()}
        best_match = min(ratios.keys(), key=lambda k: ratios[k])
        worst_match = max(ratios.keys(), key=lambda k: ratios[k])

        # Values for plotting (possibly subsampled)
        oracle_vals = oracle_full
        public_fashion_vals = public_fashion_full
        public_cifar_vals = public_cifar_full
        hybrid_vals = hybrid_full
        pink_noise_vals = pink_noise_full

        # Limit to first 1000 points for large layers to avoid overcrowding
        max_points = 1000
        if len(oracle_vals) > max_points:
            indices = np.linspace(0, len(oracle_vals)-1, max_points, dtype=int)
            oracle_vals = oracle_vals[indices]
            public_fashion_vals = public_fashion_vals[indices]
            public_cifar_vals = public_cifar_vals[indices]
            hybrid_vals = hybrid_vals[indices]
            pink_noise_vals = pink_noise_vals[indices]
            x_vals = indices
        else:
            x_vals = np.arange(len(oracle_vals))

        ax.plot(x_vals, oracle_vals, label='Oracle (Private)', alpha=0.8, linewidth=1.5, color='blue')
        ax.plot(x_vals, public_fashion_vals, label='Public (FashionMNIST)', alpha=0.8, linewidth=1.5, color='green')
        ax.plot(x_vals, public_cifar_vals, label='Public (CIFAR-10)', alpha=0.8, linewidth=1.5, color='red')
        ax.plot(x_vals, hybrid_vals, label='Hybrid (A_pub⊗G_noise)', linestyle='--', alpha=0.8, linewidth=1.5, color='orange')
        ax.plot(x_vals, pink_noise_vals, label='Pink Noise', linestyle='-.', alpha=0.8, linewidth=1.5, color='purple')
        ax.set_yscale('log')
        # Add y-axis label only in first column
        ax.set_ylabel("Eigenvalue", fontsize=8, labelpad=0)

        # Show only min and max on y-axis in scientific notation
        # Filter positive values for log scale
        all_vals = np.concatenate([oracle_vals, public_fashion_vals, public_cifar_vals,
                                   hybrid_vals, pink_noise_vals])
        positive_vals = all_vals[all_vals > 0]
        y_min = positive_vals.min()
        y_max = positive_vals.max()
        ax.set_ylim(y_min * 0.5, y_max * 2)  # Add some padding
        ax.set_yticks([y_min, y_max])
        ax.set_yticklabels([f'{y_min:.0e}', f'{y_max:.0e}'], fontsize=7)
        ax.yaxis.set_minor_locator(NullLocator())  # Remove minor ticks

        # Add title with model type and layer name
        ax.set_title(f'MLP: {layer_name}', fontsize=10)

        # Add annotation with Oracle κ, best match and worst match (top right)
        annotation_text = (f"κ(Oracle)={kappa_oracle:.1e}\n"
                          f"Best: {best_match} (κ={kappas[best_match]:.1e})\n"
                          f"Worst: {worst_match} (κ={kappas[worst_match]:.1e})")
        ax.text(0.96, 0.96, annotation_text, transform=ax.transAxes,
                fontsize=7, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='wheat', alpha=0.7))

        # Only add x-label for bottom row (i == 2)
        if i == 2:
            ax.set_xlabel("Index", fontsize=9, labelpad=-8)
        ax.grid(True, which="major", ls="-", alpha=0.3)

        # Show only max value on x-axis to indicate scale
        max_val = x_vals[-1]
        ax.set_xticks([max_val])
        ax.set_xticklabels([f'{max_val:.0e}'], fontsize=7)

    # Plot CNN layers (second column)
    for i, layer_name in enumerate(cnn_layers):
        print(f"Plotting CNN layer {i}: {layer_name}")
        ax = axes[i, 1]  # Second column

        # Get data for this layer (full arrays for condition number)
        oracle_full = cnn_oracle[layer_name]
        public_fashion_full = cnn_public_fashion[layer_name]
        public_cifar_full = cnn_public_cifar[layer_name]
        hybrid_full = cnn_hybrid[layer_name]
        pink_noise_full = cnn_pink_noise[layer_name]

        # Compute condition numbers
        kappa_oracle = compute_condition_number(oracle_full)
        kappa_fashion = compute_condition_number(public_fashion_full)
        kappa_cifar = compute_condition_number(public_cifar_full)
        kappa_hybrid = compute_condition_number(hybrid_full)
        kappa_pink = compute_condition_number(pink_noise_full)

        # Find best and worst match (smallest/largest ratio to oracle)
        kappas = {
            'Fashion': kappa_fashion,
            'CIFAR': kappa_cifar,
            'Hybrid': kappa_hybrid,
            'Pink': kappa_pink,
        }
        ratios = {k: abs(np.log10(v) - np.log10(kappa_oracle)) for k, v in kappas.items()}
        best_match = min(ratios.keys(), key=lambda k: ratios[k])
        worst_match = max(ratios.keys(), key=lambda k: ratios[k])

        # Values for plotting (possibly subsampled)
        oracle_vals = oracle_full
        public_fashion_vals = public_fashion_full
        public_cifar_vals = public_cifar_full
        hybrid_vals = hybrid_full
        pink_noise_vals = pink_noise_full

        # Limit to first 1000 points for large layers to avoid overcrowding
        max_points = 1000
        if len(oracle_vals) > max_points:
            indices = np.linspace(0, len(oracle_vals)-1, max_points, dtype=int)
            oracle_vals = oracle_vals[indices]
            public_fashion_vals = public_fashion_vals[indices]
            public_cifar_vals = public_cifar_vals[indices]
            hybrid_vals = hybrid_vals[indices]
            pink_noise_vals = pink_noise_vals[indices]
            x_vals = indices
        else:
            x_vals = np.arange(len(oracle_vals))

        ax.plot(x_vals, oracle_vals, label='Oracle (Private)', alpha=0.8, linewidth=1.5, color='blue')
        ax.plot(x_vals, public_fashion_vals, label='Public (FashionMNIST)', alpha=0.8, linewidth=1.5, color='green')
        ax.plot(x_vals, public_cifar_vals, label='Public (CIFAR-10)', alpha=0.8, linewidth=1.5, color='red')
        ax.plot(x_vals, hybrid_vals, label='Hybrid (A_pub⊗G_noise)', linestyle='--', alpha=0.8, linewidth=1.5, color='orange')
        ax.plot(x_vals, pink_noise_vals, label='Pink Noise', linestyle='-.', alpha=0.8, linewidth=1.5, color='purple')
        ax.set_yscale('log')

        # Show only min and max on y-axis in scientific notation
        # Filter positive values for log scale
        all_vals = np.concatenate([oracle_vals, public_fashion_vals, public_cifar_vals,
                                   hybrid_vals, pink_noise_vals])
        positive_vals = all_vals[all_vals > 0]
        y_min = positive_vals.min()
        y_max = positive_vals.max()
        ax.set_ylim(y_min * 0.5, y_max * 2)  # Add some padding
        ax.set_yticks([y_min, y_max])
        ax.set_yticklabels([f'{y_min:.0e}', f'{y_max:.0e}'], fontsize=7)
        ax.yaxis.set_minor_locator(NullLocator())  # Remove minor ticks

        # Add title with model type and layer name
        ax.set_title(f'CNN: {layer_name}', fontsize=10)

        # Add annotation with Oracle κ, best match and worst match (top right)
        annotation_text = (f"κ(Oracle)={kappa_oracle:.1e}\n"
                          f"Best: {best_match} (κ={kappas[best_match]:.1e})\n"
                          f"Worst: {worst_match} (κ={kappas[worst_match]:.1e})")
        ax.text(0.96, 0.96, annotation_text, transform=ax.transAxes,
                fontsize=7, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='wheat', alpha=0.7))

        # Only add x-label for bottom row (i == 2)
        if i == 2:
            ax.set_xlabel("Index", fontsize=9, labelpad=-8)
        ax.grid(True, which="major", ls="-", alpha=0.3)

        # Show only max value on x-axis to indicate scale
        max_val = x_vals[-1]
        ax.set_xticks([max_val])
        ax.set_xticklabels([f'{max_val:.0e}'], fontsize=7)

    # Add horizontal legend at the bottom of all subplots
    lines = []
    labels = ['Oracle (Private)', 'Public (FashionMNIST)', 'Public (CIFAR-10)', 'Hybrid (A_pub⊗G_noise)', 'Pink Noise']
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    styles = ['-', '-', '-', '--', '-.']

    for label, color, style in zip(labels, colors, styles):
        lines.append(Line2D([0], [0], color=color, linestyle=style, linewidth=1.5, alpha=0.8, label=label))

    fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02),
              ncol=5, fontsize=6.5, framealpha=0.9, columnspacing=0.8)

    print("Saving plot...")
    plt.tight_layout(pad=0.3, rect=(0, 0.035, 1, 1))  # Make space for bottom legend
    plt.savefig("spectrum_ablation_unified.png", dpi=300, bbox_inches='tight')
    plt.savefig("spectrum_ablation_unified.pdf", dpi=300, bbox_inches='tight', format='pdf')
    print("Plot saved as spectrum_ablation_unified.png and .pdf")
    plt.show()

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load MNIST data (private data)
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

    print("=== MLP Analysis ===")
    # Create MLP model and recorder
    mlp_model = MLP().to(device)
    mlp_recorder = KFACRecorder(mlp_model)

    # Compute spectrum data for MLP (Oracle, Public Fashion, Public CIFAR, Hybrid, Pink Noise)
    mlp_oracle, mlp_public_fashion, mlp_public_cifar, mlp_hybrid, mlp_pink_noise = compute_spectrum_data(
        mlp_model, train_loader, mlp_recorder, str(device)
    )
    mlp_data = {
        'fc_layers': list(mlp_oracle.keys()),
        'oracle_evals': mlp_oracle,
        'public_fashion_evals': mlp_public_fashion,
        'public_cifar_evals': mlp_public_cifar,
        'hybrid_evals': mlp_hybrid,
        'pink_noise_evals': mlp_pink_noise
    }

    # Clean up MLP
    mlp_recorder.remove()

    print("\n=== CNN Analysis ===")
    # Create CNN model and recorder
    cnn_model = CNN().to(device)
    cnn_recorder = KFACRecorder(cnn_model)

    # Compute spectrum data for CNN (Oracle, Public Fashion, Public CIFAR, Hybrid, Pink Noise)
    cnn_oracle, cnn_public_fashion, cnn_public_cifar, cnn_hybrid, cnn_pink_noise = compute_spectrum_data(
        cnn_model, train_loader, cnn_recorder, str(device)
    )
    layer_names = list(cnn_oracle.keys())
    conv_layers = [name for name in layer_names if 'conv' in name]
    fc_layers = [name for name in layer_names if 'fc' in name and name != 'fc2']  # Exclude fc2

    cnn_data = {
        'conv_layers': conv_layers,
        'fc_layers': fc_layers,
        'oracle_evals': cnn_oracle,
        'public_fashion_evals': cnn_public_fashion,
        'public_cifar_evals': cnn_public_cifar,
        'hybrid_evals': cnn_hybrid,
        'pink_noise_evals': cnn_pink_noise
    }

    # Clean up CNN
    cnn_recorder.remove()

    # Plot unified spectrum
    print("\n=== Creating Unified Plot ===")
    plot_unified_spectrum(mlp_data, cnn_data)