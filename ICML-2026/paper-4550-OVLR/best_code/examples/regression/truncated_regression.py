"""
OVLR Example: Truncated L1 Regression for Robust Outlier Resistance

This script demonstrates OVLR's ability to optimize truncated loss functions
that provide robustness to outliers. Standard backpropagation fails here because
the gradient is zero outside the truncation threshold.

Reference:
    OVLR: Efficient, Scalable, and Robust Training via
    Output-Level Variance-Reduced Likelihood Ratio
    ICML 2026
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os


class TruncatedL1Loss(nn.Module):
    """
    Truncated L1 loss: gradients vanish outside the truncation threshold
    when using standard backpropagation.

    L = clamp(|y_pred - y_true|, max=threshold)
    """
    def __init__(self, threshold=1.0, reduction='none'):
        super().__init__()
        self.threshold = threshold
        self.reduction = reduction

    def forward(self, input, target):
        diff = torch.abs(input - target)
        loss = torch.clamp(diff, max=self.threshold)
        if self.reduction == 'mean':
            return loss.mean()
        return loss


def generate_data(n_samples, outlier_ratio=0.3, outlier_shift=15.0, is_train=True):
    """Generate synthetic regression data with outliers."""
    seed = 42 if is_train else 2024
    torch.manual_seed(seed)

    x = torch.linspace(-6, 6, n_samples).unsqueeze(1)
    y_clean = torch.sin(x)
    y_noisy = y_clean + 0.05 * torch.randn_like(y_clean)

    if is_train and outlier_ratio > 0:
        n_outliers = int(n_samples * outlier_ratio)
        indices = torch.randperm(n_samples)[:n_outliers]
        y_noisy[indices] += outlier_shift + torch.randn(n_outliers, 1) * 0.5

    return x, y_clean, y_noisy


class MLP(nn.Module):
    """Simple MLP for regression."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


def train_model(method, model, x_train, y_train, optimizer, estimator, epochs, trunc_threshold=1.0):
    """Train model with specified method."""
    losses = []

    mse_crit = nn.MSELoss(reduction='none')
    l1_crit = nn.L1Loss(reduction='none')
    trunc_crit = TruncatedL1Loss(threshold=trunc_threshold, reduction='none')

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        if method.startswith('OVLR'):
            outputs = model(x_train)
            if method == 'OVLR-Truncated':
                criterion = trunc_crit
            elif method == 'OVLR-L1':
                criterion = l1_crit
            else:  # OVLR-MSE
                criterion = mse_crit

            loss = estimator(outputs, y_train, criterion, loss_fn_reduction='mean')
        else:
            pred = model(x_train)
            if method == 'BP-MSE':
                loss = mse_crit(pred, y_train).mean()
            elif method == 'BP-L1':
                loss = l1_crit(pred, y_train).mean()
            elif method == 'BP-Truncated':
                loss = trunc_crit(pred, y_train).mean()
            loss.backward()

        optimizer.step()

        if epoch % 50 == 0:
            losses.append(loss.item())

    return model


def main():
    parser = argparse.ArgumentParser(description='OVLR: Truncated L1 Regression')
    parser.add_argument('--n-train', type=int, default=400, help='number of training samples')
    parser.add_argument('--n-test', type=int, default=100, help='number of test samples')
    parser.add_argument('--epochs', type=int, default=2000, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--n-repeat', type=int, default=400, help='number of noisy samples for OVLR')
    parser.add_argument('--noise-scale', type=float, default=2.0, help='OVLR noise scale (sigma)')
    parser.add_argument('--trunc-threshold', type=float, default=0.5, help='truncation threshold')
    parser.add_argument('--outlier-ratio', type=float, default=0.3, help='fraction of outliers')
    parser.add_argument('--outlier-shift', type=float, default=15.0, help='outlier shift magnitude')
    parser.add_argument('--save-dir', type=str, default='./results_regression', help='save directory')
    parser.add_argument('--no-cuda', action='store_true', help='disable CUDA')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # Generate data
    print("Generating data...")
    x_train, y_clean_train, y_train = generate_data(
        args.n_train, args.outlier_ratio, args.outlier_shift, is_train=True
    )
    x_test, y_clean_test, _ = generate_data(
        args.n_test, 0, 0, is_train=False
    )

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_clean_test = y_clean_test.to(device)

    # Methods to compare
    methods = {
        'BP-MSE': {'color': 'gray', 'linestyle': '--', 'label': 'BP (MSE)'},
        'BP-L1': {'color': 'orange', 'linestyle': '-.', 'label': 'BP (L1)'},
        'OVLR-L1': {'color': 'cyan', 'linestyle': '-.', 'label': 'OVLR (L1)'},
        'BP-Truncated': {'color': 'red', 'linestyle': ':', 'label': 'BP (Truncated)'},
        'OVLR-Truncated': {'color': 'blue', 'linestyle': '-', 'label': 'OVLR (Truncated)'},
    }

    results = {}

    # Plot setup
    plt.figure(figsize=(10, 6))
    plt.scatter(x_train.cpu(), y_train.cpu(), color='black', s=10, alpha=0.15, label='Noisy Data')
    plt.plot(x_test.cpu(), y_clean_test.cpu(), 'k', linewidth=2, alpha=0.3, label='Ground Truth')

    # Import OVLR
    from ovlr import OVLRGradientEstimator, get_noise_fn

    for name, config in methods.items():
        print(f"\nTraining {name}...")
        model = MLP().to(device)

        # Bad initialization to challenge BP-Truncated
        with torch.no_grad():
            model.net[-1].bias.fill_(2.0)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        if name.startswith('OVLR'):
            noise_fn = get_noise_fn(mode="symmetric", noise_scale=args.noise_scale)
            estimator = OVLRGradientEstimator(noise_fn, n_repeat=args.n_repeat)
        else:
            estimator = None

        model = train_model(name, model, x_train, y_train, optimizer, estimator, args.epochs, args.trunc_threshold)

        # Evaluate
        model.eval()
        with torch.no_grad():
            preds = model(x_test)
            test_mse = nn.MSELoss()(preds, y_clean_test).item()
            results[name] = test_mse

        plt.plot(x_test.cpu(), preds.cpu(), color=config['color'],
                 linestyle=config['linestyle'], linewidth=2.5,
                 label=f"{config['label']} (MSE: {test_mse:.3f})")

    plt.title(f"Robust Regression Comparison (+{args.outlier_shift} Shift, {args.outlier_ratio:.0%} Outliers)", fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(-3, 8)
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'regression_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to {args.save_dir}/regression_comparison.png")

    print("\n" + "=" * 50)
    print("FINAL TEST MSE (on Clean Ground Truth)")
    print("=" * 50)
    for method_name, mse in sorted(results.items(), key=lambda x: x[1]):
        print(f"{method_name:15s}: {mse:.5f}")


if __name__ == '__main__':
    main()
