#!/usr/bin/env python3
"""Evaluation for paper 4529 — KL-DRO on California Housing.
GPU-optimized with configurable optimization levers.

Flags:
  --baseline          Run unregularized baseline (no rho/alpha)
  --n-seeds N         Number of seeds (default 10)
  --lr LR             Learning rate (default 5e-7)
  --n-epochs N        Training epochs (default 50)
  --quick             3 seeds, proposed only
  --lr-schedule S     constant | linear | step (default constant)
  --step-size S       Step size for step schedule (default 15)
  --step-gamma G      Gamma for step schedule (default 0.5)
  --use-ema           Use EMA weights for final evaluation
  --ema-decay D       EMA decay rate (default 0.999)
  --float64-softplus  Use float64 precision for softplus computation
  --rho-anneal        Anneal rho from high to low over training
  --rho-start R       Starting rho for annealing (default 1e-1)
  --rho-end R         Final rho for annealing (default 1e-4)
  --grad-clip C       Max gradient norm for clipping (0 = disabled)
  --alpha-lr-mult M   Multiplier for alpha learning rate (default 1.0)
  --alpha-init V      Initial value for alpha (default 0.0)
"""
import sys, os, math, argparse, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------------------------
def get_data():
    data_path = '/datasets/california_housing_data.npy'
    target_path = '/datasets/california_housing_target.npy'
    if os.path.exists(data_path) and os.path.exists(target_path):
        return np.load(data_path), np.load(target_path)
    from sklearn.datasets import fetch_openml
    housing = fetch_openml(name='california_housing', version=1, as_frame=False, parser='auto')
    raw = housing.data[:, [0, 1, 2, 3, 4, 5, 6, 7]].astype(float)
    target = housing.target.astype(float) / 100_000.0
    bedrooms = raw[:, 4].copy()
    bedrooms[np.isnan(bedrooms)] = np.nanmedian(bedrooms)
    X = np.column_stack([
        raw[:, 7], raw[:, 1], raw[:, 2] / raw[:, 6],
        bedrooms / raw[:, 6], raw[:, 5],
        raw[:, 5] / raw[:, 6], raw[:, 1], raw[:, 0]
    ])
    os.makedirs('/datasets', exist_ok=True)
    np.save(data_path, X); np.save(target_path, target)
    return X, target

# ---------------------------------------------------------------------------
class LinRegModel(nn.Module):
    def __init__(self, input_dim, with_alpha=False, alpha_init=0.0, device='cpu'):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        sd = torch.load('/repo/linreg_weights.pt', map_location=device)
        with torch.no_grad():
            self.linear.weight.copy_(sd['weights'].unsqueeze(0))
            self.linear.bias.copy_(sd['bias'])
        if with_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha_init, device=device))

    def forward(self, x):
        return self.linear(x)

# ---------------------------------------------------------------------------
def compute_objective(model, X, y, lam):
    with torch.no_grad():
        preds = model(X)
        errors = (preds - y) ** 2
        L = errors.squeeze(1)
    lse = torch.logsumexp(L / lam, dim=0).item() - math.log(len(y))
    return lam * lse

# ---------------------------------------------------------------------------
def softplus_loss(L, alpha, lam, rho, use_float64=False):
    """Compute the proposed KL-DRO loss."""
    if use_float64:
        exponent = ((L.double() - alpha.double()) / lam + math.log(rho)).float()
        return (lam / rho) * F.softplus(exponent).mean() + alpha
    else:
        exponent = (L - alpha) / lam + math.log(rho)
        return (lam / rho) * F.softplus(exponent).mean() + alpha

# ---------------------------------------------------------------------------
def train_one_run(X, y, lam, rho, lr, seed, args, device='cpu'):
    """Train one seed. X, y already on device."""
    torch.manual_seed(seed)
    n = len(X)
    idx = torch.randperm(n, device=device)
    batch_size = 10  # |D| = 10 per paper

    model = LinRegModel(X.shape[1], with_alpha=(rho is not None),
                        alpha_init=args.alpha_init, device=device).to(device)

    # Per-parameter learning rates
    if rho is not None and args.alpha_lr_mult != 1.0:
        param_groups = [
            {'params': [model.linear.weight, model.linear.bias], 'lr': lr},
            {'params': [model.alpha], 'lr': lr * args.alpha_lr_mult}
        ]
        optimizer = torch.optim.SGD(param_groups, momentum=0.9, nesterov=True)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)

    # LR scheduler
    scheduler = None
    if args.lr_schedule == 'linear':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda e: 1.0 - e / args.n_epochs)
    elif args.lr_schedule == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.step_gamma)

    # EMA model (deepcopy after first backward)
    ema_model = None
    if args.use_ema:
        ema_model = copy.deepcopy(model)
        for p in ema_model.parameters():
            p.requires_grad = False

    # Rho annealing schedule
    def get_rho(epoch):
        if not args.rho_anneal or rho is None:
            return rho
        log_start = math.log10(args.rho_start)
        log_end = math.log10(args.rho_end)
        frac = epoch / args.n_epochs
        return 10.0 ** (log_start + frac * (log_end - log_start))

    for epoch in range(1, args.n_epochs + 1):
        current_rho = get_rho(epoch)
        for start in range(0, n, batch_size):
            batch_idx = idx[start:start + batch_size]
            Xb, yb = X[batch_idx], y[batch_idx]

            optimizer.zero_grad()
            preds = model(Xb)
            errors = (preds - yb) ** 2
            L = errors.squeeze(1)

            if rho is None:  # baseline
                with torch.no_grad():
                    p = torch.softmax(L / lam, dim=0)
                loss = torch.sum(p * L)
            else:
                loss = softplus_loss(L, model.alpha, lam, current_rho, args.float64_softplus)

            # NaN detection
            if not torch.isfinite(loss):
                return float('nan')

            loss.backward()

            # Gradient clipping
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            # EMA update
            if ema_model is not None:
                with torch.no_grad():
                    for p_ema, p_model in zip(ema_model.parameters(), model.parameters()):
                        p_ema.mul_(args.ema_decay).add_(p_model, alpha=1.0 - args.ema_decay)

        if scheduler is not None:
            scheduler.step()

        # Reshuffle
        idx = torch.randperm(n, device=device)

    eval_model = ema_model if ema_model is not None else model
    return compute_objective(eval_model, X, y, lam)

# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--n-seeds', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-7)
    parser.add_argument('--n-epochs', type=int, default=50)
    parser.add_argument('--quick', action='store_true')
    # Optimization flags
    parser.add_argument('--lr-schedule', default='constant', choices=['constant', 'linear', 'step'])
    parser.add_argument('--step-size', type=int, default=15)
    parser.add_argument('--step-gamma', type=float, default=0.5)
    parser.add_argument('--use-ema', action='store_true')
    parser.add_argument('--ema-decay', type=float, default=0.999)
    parser.add_argument('--float64-softplus', action='store_true')
    parser.add_argument('--rho-anneal', action='store_true')
    parser.add_argument('--rho-start', type=float, default=1e-1)
    parser.add_argument('--rho-end', type=float, default=1e-4)
    parser.add_argument('--grad-clip', type=float, default=0.0)
    parser.add_argument('--alpha-lr-mult', type=float, default=1.0)
    parser.add_argument('--alpha-init', type=float, default=0.0)
    args = parser.parse_args()

    if args.quick:
        args.n_seeds = 3

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mode = "baseline" if args.baseline else "proposed"

    # Build description
    flags = []
    if args.lr_schedule != 'constant':
        flags.append(f"lr={args.lr_schedule}")
    if args.use_ema:
        flags.append(f"ema={args.ema_decay}")
    if args.float64_softplus:
        flags.append("f64")
    if args.rho_anneal:
        flags.append(f"rho_anneal({args.rho_start:.0e}->{args.rho_end:.0e})")
    if args.grad_clip > 0:
        flags.append(f"clip={args.grad_clip}")
    if args.alpha_lr_mult != 1.0:
        flags.append(f"alpha_lr={args.alpha_lr_mult}x")
    if args.alpha_init != 0.0:
        flags.append(f"alpha_init={args.alpha_init}")
    flag_str = " + ".join(flags) if flags else "baseline config"
    print(f"=== Paper 4529: {mode} [{flag_str}] ===", flush=True)
    print(f"Device: {device}, lr={args.lr:.0e}, seeds={args.n_seeds}, epochs={args.n_epochs}", flush=True)

    # Load data
    X_np, y_np = get_data()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_np)
    tX = torch.from_numpy(X_scaled.astype(np.float32)).to(device)
    ty = torch.from_numpy(y_np.astype(np.float32)).unsqueeze(1).to(device)
    print(f"Data: {len(tX)} samples", flush=True)

    # LS init
    lr_model = LinearRegression(fit_intercept=True).fit(X_scaled, y_np)
    torch.save({
        'weights': torch.tensor(lr_model.coef_, dtype=torch.float32),
        'bias': torch.tensor(lr_model.intercept_, dtype=torch.float32)
    }, '/repo/linreg_weights.pt')

    lam = 5.0
    rho = 1e-3 if not args.baseline else None

    results = []
    failures = 0
    for seed in range(args.n_seeds):
        obj = train_one_run(tX, ty, lam, rho, args.lr, seed, args, device=device)
        if np.isfinite(obj):
            results.append(obj)
            print(f"  seed {seed}: {obj:.4f}", flush=True)
        else:
            failures += 1
            print(f"  seed {seed}: FAILED (NaN/inf)", flush=True)

    if not results:
        print("ERROR: All seeds failed!", flush=True)
        sys.exit(1)

    mean_v = np.mean(results)
    std_v = np.std(results)
    print(f"\n{'='*60}", flush=True)
    print(f"RESULT ({mode}): {mean_v:.4f} +/- {std_v:.4f}  ({len(results)}/{args.n_seeds} ok, {failures} failed)", flush=True)
    print(f"Config: {flag_str}", flush=True)
    print(f"{'='*60}", flush=True)

    metric_name = "baseline_objective" if args.baseline else "proposed_objective"
    print(f"\nMETRIC:{metric_name}={mean_v:.4f}", flush=True)
    print(f"METRIC:{metric_name}_std={std_v:.4f}", flush=True)

if __name__ == "__main__":
    main()
