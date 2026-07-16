"""
Helmholtz 2D PINN with multi-layer SIREN MLP and Hermite-NGP encoding.
Adaptive-LR variant for the high-frequency a=100 case (larger collocation
pool, more BC points).

Usage:
    python examples/helmholtz2d_a100.py
    python examples/helmholtz2d_a100.py --epochs 100000 --layers 2
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Tuple, Optional

# =============================================================================
# Problem Setup: Helmholtz 2D
# =============================================================================
# PDE: Laplacian(u) + k^2 * u = f
# Domain: [0, 1] x [0, 1]
# Exact: u = sin(a1*pi*x) * sin(a2*pi*y)
# Source: f = (k^2 - (a1^2 + a2^2)*pi^2) * sin(a1*pi*x) * sin(a2*pi*y)

a1, a2 = 100.0, 100.0  # Frequency parameters (hard-coded for paper a=100 setting)
K = 1.0             # Helmholtz parameter
PI = np.pi

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Import CUDA extension for multi-layer SIREN
try:
    import hermite_mlp_cuda_v2
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("WARNING: hermite_mlp_cuda_v2 not available. Run: python setup_mlp_cuda_v2.py install")


def exact_solution(x, y):
    """Exact solution: u = sin(a1*pi*x) * sin(a2*pi*y)"""
    return torch.sin(a1 * PI * x) * torch.sin(a2 * PI * y)


def source_term(x, y):
    """Source term f for the Helmholtz equation."""
    return (K**2 - (a1**2 + a2**2) * PI**2) * torch.sin(a1 * PI * x) * torch.sin(a2 * PI * y)


# =============================================================================
# CUDA V3 Multi-layer SIREN (from helmholtz2d_cuda_mlp_v3_train.py)
# =============================================================================

class HermiteLayerFunctionV3(torch.autograd.Function):
    """CUDA forward + PyTorch backward for Hermite propagation."""

    @staticmethod
    def forward(ctx, h, dh_dx, dh_dy, d2h_dxx, d2h_dyy, weight, bias, omega, apply_activation):
        outputs = hermite_mlp_cuda_v2.forward(
            h.contiguous(), dh_dx.contiguous(), dh_dy.contiguous(),
            d2h_dxx.contiguous(), d2h_dyy.contiguous(),
            weight.contiguous(), bias.contiguous(),
            omega, apply_activation
        )
        out_h, out_dx, out_dy, out_dxx, out_dyy, save_z, save_dz_dx, save_dz_dy, save_d2z_dxx, save_d2z_dyy = outputs

        ctx.save_for_backward(
            h, dh_dx, dh_dy, d2h_dxx, d2h_dyy,
            weight,
            save_z, save_dz_dx, save_dz_dy
        )
        ctx.omega = omega
        ctx.apply_activation = apply_activation

        return out_h, out_dx, out_dy, out_dxx, out_dyy

    @staticmethod
    def backward(ctx, grad_h, grad_dx, grad_dy, grad_dxx, grad_dyy):
        h, dh_dx, dh_dy, d2h_dxx, d2h_dyy, weight, z, dz_dx, dz_dy = ctx.saved_tensors
        omega = ctx.omega
        apply_activation = ctx.apply_activation

        omega2 = omega * omega
        omega3 = omega2 * omega

        if apply_activation:
            sin_z = torch.sin(omega * z)
            cos_z = torch.cos(omega * z)

            h_p = omega * cos_z
            h_pp = -omega2 * sin_z
            h_ppp = -omega3 * cos_z

            grad_z = grad_h * h_p
            grad_z = grad_z + grad_dx * h_pp * dz_dx
            grad_z = grad_z + grad_dy * h_pp * dz_dy

            d2z_dxx = d2h_dxx @ weight.T
            grad_z = grad_z + grad_dxx * (h_ppp * dz_dx * dz_dx + h_pp * d2z_dxx)

            d2z_dyy = d2h_dyy @ weight.T
            grad_z = grad_z + grad_dyy * (h_ppp * dz_dy * dz_dy + h_pp * d2z_dyy)

            grad_dz_dx = grad_dx * h_p + grad_dxx * 2 * h_pp * dz_dx
            grad_dz_dy = grad_dy * h_p + grad_dyy * 2 * h_pp * dz_dy
            grad_d2z_dxx = grad_dxx * h_p
            grad_d2z_dyy = grad_dyy * h_p
        else:
            grad_z = grad_h
            grad_dz_dx = grad_dx
            grad_dz_dy = grad_dy
            grad_d2z_dxx = grad_dxx
            grad_d2z_dyy = grad_dyy

        grad_h_in = grad_z @ weight
        grad_dh_dx_in = grad_dz_dx @ weight
        grad_dh_dy_in = grad_dz_dy @ weight
        grad_d2h_dxx_in = grad_d2z_dxx @ weight
        grad_d2h_dyy_in = grad_d2z_dyy @ weight

        grad_weight = grad_z.T @ h
        grad_weight = grad_weight + grad_dz_dx.T @ dh_dx
        grad_weight = grad_weight + grad_dz_dy.T @ dh_dy
        grad_weight = grad_weight + grad_d2z_dxx.T @ d2h_dxx
        grad_weight = grad_weight + grad_d2z_dyy.T @ d2h_dyy

        grad_bias = grad_z.sum(dim=0)

        return grad_h_in, grad_dh_dx_in, grad_dh_dy_in, grad_d2h_dxx_in, grad_d2h_dyy_in, grad_weight, grad_bias, None, None


class MultiSIREN(nn.Module):
    """
    Multi-layer SIREN MLP with CUDA V3 (supports multiple hidden layers).

    Uses CUDA forward + PyTorch backward for Hermite derivative propagation.
    """

    def __init__(self, input_dim, hidden_dim=256, n_layers=2, omega_0=0.5):
        super().__init__()
        self.omega_0 = omega_0
        self.n_layers = n_layers

        # Build layers
        self.layers = nn.ModuleList()
        dims = [input_dim] + [hidden_dim] * n_layers
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
        self.output_layer = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self):
        """SIREN-specific initialization."""
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                if i == 0:
                    bound = 1.0 / layer.in_features
                else:
                    bound = np.sqrt(6.0 / layer.in_features) / self.omega_0
                layer.weight.uniform_(-bound, bound)
                layer.bias.uniform_(-bound, bound)
            bound = np.sqrt(6.0 / self.output_layer.in_features) / self.omega_0
            self.output_layer.weight.uniform_(-bound, bound)
            self.output_layer.bias.zero_()

    def forward(self, x):
        """Standard forward pass."""
        h = x
        for layer in self.layers:
            h = torch.sin(self.omega_0 * layer(h))
        return self.output_layer(h)

    def forward_with_laplacian_cuda(self, enc, dx, dy, dxx, dyy):
        """Forward with full Hermite propagation using CUDA V3."""
        omega = self.omega_0
        h, dh_dx, dh_dy, d2h_dxx, d2h_dyy = enc, dx, dy, dxx, dyy

        # Hidden layers with activation
        for layer in self.layers:
            h, dh_dx, dh_dy, d2h_dxx, d2h_dyy = HermiteLayerFunctionV3.apply(
                h, dh_dx, dh_dy, d2h_dxx, d2h_dyy,
                layer.weight, layer.bias, omega, True
            )

        # Output layer without activation
        u, du_dx, du_dy, d2u_dxx, d2u_dyy = HermiteLayerFunctionV3.apply(
            h, dh_dx, dh_dy, d2h_dxx, d2h_dyy,
            self.output_layer.weight, self.output_layer.bias, omega, False
        )

        laplacian = d2u_dxx + d2u_dyy
        return u, laplacian


# =============================================================================
# Hermite-NGP PINN Model with Multi-layer SIREN
# =============================================================================

class HermiteNGP_PINN(nn.Module):
    """
    Physics-Informed Neural Network with Hermite Hash Encoding and Multi-layer SIREN.

    Features:
    - Analytic derivatives from Hermite encoding
    - Multi-layer SIREN MLP with CUDA forward
    - Curriculum learning (progressive level activation)
    - GradNorm (adaptive loss balancing)
    """
    def __init__(self, config=None):
        super().__init__()

        # Default best configuration
        config = config or {}
        self.n_levels = config.get('n_levels', 8)
        self.log2_hashmap_size_1 = config.get('log2_hashmap_size_1', 16)
        self.log2_hashmap_size_2 = config.get('log2_hashmap_size_2', 16)
        self.log2_hashmap_size_3 = config.get('log2_hashmap_size_3', 16)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.n_layers = config.get('n_layers', 2)
        self.omega = config.get('omega', 0.5)
        self.bc_weight_cap = config.get('bc_weight_cap', 10000.0)

        # Hermite Hash Encoding (CUDA)
        from hermite_ngp.encoding.hermite_encoding_cuda import HermiteHashEncodingCUDA
        self.encoding = HermiteHashEncodingCUDA(
            n_input_dims=2,
            n_levels=self.n_levels,
            n_features_per_level=2,
            log2_hashmap_size_1=self.log2_hashmap_size_1,
            log2_hashmap_size_2=self.log2_hashmap_size_2,
            log2_hashmap_size_3=self.log2_hashmap_size_3,
            base_resolution=4,
            per_level_scale=2.0,
        )

        # Multi-layer SIREN MLP
        encoding_dim = self.n_levels * 2
        self.mlp = MultiSIREN(encoding_dim, self.hidden_dim, self.n_layers, omega_0=self.omega)

        # Training state
        self.bc_weight = 5000.0
        self.register_buffer('level_grad_mask', torch.ones(self.n_levels))

        # Curriculum phases (coarse-to-fine)
        self.phases = [
            (0, float('inf'), list(range(8))),  # All levels from start
        ]

        self.to(device)

        # Pre-computed fixed BC points (static shapes)
        n_bc_per_edge = 5000
        t = torch.linspace(0.01, 0.99, n_bc_per_edge, device=device)
        self.fixed_bc_pts = torch.cat([
            torch.stack([t, torch.zeros_like(t)], dim=1),  # bottom
            torch.stack([t, torch.ones_like(t)], dim=1),   # top
            torch.stack([torch.zeros_like(t), t], dim=1),  # left
            torch.stack([torch.ones_like(t), t], dim=1),   # right
        ])
        self.fixed_bc_vals = exact_solution(self.fixed_bc_pts[:, 0], self.fixed_bc_pts[:, 1])

    def get_active_levels(self, epoch):
        """Get active levels for curriculum learning."""
        for start, end, levels in self.phases:
            if start <= epoch < end:
                return levels
        return list(range(self.n_levels))

    def freeze_levels(self, levels_to_freeze):
        """Freeze gradients for specified levels."""
        self.level_grad_mask[:] = 1.0
        for l in levels_to_freeze:
            self.level_grad_mask[l] = 0.0

    def apply_level_mask(self):
        """Apply gradient mask to hash table."""
        if self.encoding.hash_table_1.grad is not None:
            mask = self.level_grad_mask.view(-1, 1, 1)
            self.encoding.hash_table_1.grad *= mask

        if self.encoding.hash_table_2.grad is not None:
            mask = self.level_grad_mask.view(-1, 1, 1)
            self.encoding.hash_table_2.grad *= mask

        if self.encoding.hash_table_3.grad is not None:
            mask = self.level_grad_mask.view(-1, 1, 1)
            self.encoding.hash_table_3.grad *= mask

    def forward(self, x):
        """Standard forward pass."""
        enc = self.encoding(x)
        return self.mlp(enc)

    def forward_with_laplacian(self, x):
        """
        Forward pass with analytic Laplacian using CUDA MLP.

        Returns:
            u: [N, 1] function values
            laplacian: [N, 1] Laplacian values
        """
        # Hermite encoding with analytic derivatives
        enc, dx, dy, dxx, dyy = self.encoding.forward_with_second_derivatives_cuda(x)

        # MLP with Hermite propagation (CUDA)
        u, laplacian = self.mlp.forward_with_laplacian_cuda(enc, dx, dy, dxx, dyy)

        return u, laplacian

    def loss_pde(self, pts):
        """PDE loss: (Laplacian(u) + k^2*u - f)^2"""
        u, lap = self.forward_with_laplacian(pts)
        f = source_term(pts[:, 0], pts[:, 1]).unsqueeze(-1)
        residual = lap + K**2 * u - f
        return (residual**2).mean()

    def loss_bc(self, bc_pts, bc_vals):
        """Boundary condition loss: (u - u_exact)^2"""
        u = self.forward(bc_pts)
        return ((u.squeeze() - bc_vals)**2).mean()

    def generate_bc_points(self, n_per_edge):
        """Generate boundary points on all 4 edges."""
        t = torch.rand(n_per_edge, device=device)
        pts = torch.cat([
            torch.stack([t, torch.zeros(n_per_edge, device=device)], dim=1),  # bottom
            torch.stack([t, torch.ones(n_per_edge, device=device)], dim=1),   # top
            torch.stack([torch.zeros(n_per_edge, device=device), t], dim=1),  # left
            torch.stack([torch.ones(n_per_edge, device=device), t], dim=1),   # right
        ])
        vals = exact_solution(pts[:, 0], pts[:, 1])
        return pts, vals

    def evaluate(self, resolution=100):
        """Evaluate L2 error on uniform grid."""
        with torch.no_grad():
            g = torch.linspace(0, 1, resolution, device=device)
            X, Y = torch.meshgrid(g, g, indexing='ij')
            pts = torch.stack([X.flatten(), Y.flatten()], dim=1)
            u_pred = self.forward(pts).reshape(resolution, resolution)
            u_exact = exact_solution(X, Y)
            l2_error = (torch.sqrt(((u_pred - u_exact)**2).sum()) / torch.sqrt((u_exact**2).sum())).item()

            return l2_error, u_pred.cpu().numpy(), u_exact.cpu().numpy()


# =============================================================================
# Exponential Moving Average
# =============================================================================

class EMA:
    """Exponential Moving Average for model parameters."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        """Update shadow weights."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self, model):
        """Apply shadow weights to model (save backup)."""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model):
        """Restore original weights from backup."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


# =============================================================================
# Training
# =============================================================================

def get_grad_norm(model):
    """Compute total gradient norm."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.norm()**2
    return total.sqrt()


def train(config):
    """
    Main training function.

    Args:
        config: Training configuration dict
    """
    # Extract config
    n_epochs = config.get('n_epochs', 20000)
    seed = config.get('seed', 456)
    lr = config.get('lr', 1e-3)
    num_collocation = config.get('num_collocation', 10000)
    num_bc_per_edge = config.get('num_bc_per_edge', 5000)
    eval_interval = config.get('eval_interval', 5000)
    save_plots = config.get('save_plots', True)

    # Adaptive LR config
    use_adaptive_lr = config.get('use_adaptive_lr', True)
    lr_patience = config.get('lr_patience', 2000)
    min_lr = config.get('min_lr', 1e-7)

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print("Helmholtz 2D - Multi-layer SIREN (v2 with Adaptive LR)")
    print("=" * 70)
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Epochs: {n_epochs}, Seed: {seed}")
    print(f"Hidden: {config.get('hidden_dim', 256)}, Layers: {config.get('n_layers', 2)}")
    if use_adaptive_lr:
        print(f"Adaptive LR: ON (patience={lr_patience}, min_lr={min_lr:.0e})")
    else:
        print(f"Adaptive LR: OFF")
    print("=" * 70)

    if not CUDA_AVAILABLE:
        print("\nERROR: CUDA extension not available!")
        print("Please run: python setup_mlp_cuda_v2.py install")
        return None, None, None

    # Create model
    model = HermiteNGP_PINN(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15000, gamma=0.5)

    # EMA
    ema = EMA(model, decay=0.999)

    # Warmup
    print("\nWarmup...")
    for _ in range(20):
        pts = torch.rand(num_collocation, 2, device=device)
        bc_pts, bc_vals = model.generate_bc_points(num_bc_per_edge)
        loss_pde = model.loss_pde(pts)
        loss_bc = model.loss_bc(bc_pts, bc_vals)
        loss = loss_pde + model.bc_weight * loss_bc
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.update(model)

    # Training loop
    best_l2 = float('inf')
    best_epoch = 0
    best_state = None
    history = []
    epochs_since_improvement = 0

    # Adjust eval interval for short runs
    actual_eval_interval = min(eval_interval, max(100, n_epochs // 10))

    print("\nTraining...")
    t0 = time.perf_counter()

    for epoch in range(n_epochs):
        # Curriculum: get active levels
        active_levels = model.get_active_levels(epoch)
        frozen_levels = [l for l in range(model.n_levels) if l not in active_levels]
        model.freeze_levels(frozen_levels)

        # Sample collocation points (random every epoch)
        pts = torch.rand(num_collocation, 2, device=device)

        # Sample BC points (random every epoch)
        bc_pts, bc_vals = model.generate_bc_points(num_bc_per_edge)

        # GradNorm: adaptive loss balancing (every 100 epochs)
        if (epoch + 1) % 100 == 0:
            l_pde = model.loss_pde(pts)
            l_bc = model.loss_bc(bc_pts, bc_vals)

            optimizer.zero_grad()
            l_pde.backward(retain_graph=True)
            model.apply_level_mask()
            grad_pde = get_grad_norm(model)

            optimizer.zero_grad()
            l_bc.backward(retain_graph=True)
            model.apply_level_mask()
            grad_bc = get_grad_norm(model)

            if grad_bc > 1e-8:
                ratio = (grad_pde / grad_bc).item()
                model.bc_weight = 0.9 * model.bc_weight + 0.1 * ratio
                model.bc_weight = max(1.0, min(model.bc_weight_cap, model.bc_weight))

            loss = l_pde + model.bc_weight * l_bc
        else:
            loss_pde = model.loss_pde(pts)
            loss_bc = model.loss_bc(bc_pts, bc_vals)
            loss = loss_pde + model.bc_weight * loss_bc

        # Backward and update
        optimizer.zero_grad()
        loss.backward()
        model.apply_level_mask()
        optimizer.step()
        scheduler.step()
        ema.update(model)

        # Evaluation
        if (epoch + 1) % actual_eval_interval == 0:
            ema.apply_shadow(model)
            l2, u_pred, u_exact = model.evaluate()
            ema.restore(model)

            if l2 < best_l2:
                best_l2 = l2
                best_epoch = epoch + 1
                best_state = {k: v.clone() for k, v in ema.shadow.items()}
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += actual_eval_interval

            # Adaptive LR reduction
            lr_current = scheduler.get_last_lr()[0]
            if use_adaptive_lr and epochs_since_improvement >= lr_patience and lr_current > min_lr:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = max(param_group['lr'] * 0.5, min_lr)
                lr_current = optimizer.param_groups[0]['lr']
                print(f"  >> LR reduced to {lr_current:.2e} (no improvement for {lr_patience} epochs)")
                epochs_since_improvement = 0

            elapsed = time.perf_counter() - t0
            history.append((epoch + 1, l2, best_l2))

            print(f"  Epoch {epoch+1:6d}: L2={l2:.4e}, best={best_l2:.4e} @{best_epoch}, "
                  f"bc_w={model.bc_weight:.1f}, lr={lr_current:.2e}, time={elapsed:.0f}s")

    # Restore best model
    if best_state is not None:
        for name, param in model.named_parameters():
            if name in best_state:
                param.data = best_state[name]

    # Final evaluation
    l2_final, u_pred, u_exact = model.evaluate(resolution=200)
    elapsed_total = time.perf_counter() - t0
    ms_per_epoch = elapsed_total / n_epochs * 1000

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Best L2 Error: {best_l2:.4e} @ epoch {best_epoch}")
    print(f"Final L2 Error: {l2_final:.4e}")
    print(f"Total Time: {elapsed_total:.1f}s ({ms_per_epoch:.2f} ms/epoch)")

    # Prepare results dict
    results = {
        'best_l2': best_l2,
        'best_epoch': best_epoch,
        'final_l2': l2_final,
        'total_time': elapsed_total,
        'ms_per_epoch': ms_per_epoch,
        'n_params': n_params,
        'history': history,
        'u_pred': u_pred,
        'u_exact': u_exact,
    }

    # Save visualization
    if save_plots:
        save_visualization(u_pred, u_exact, best_l2, best_epoch, config)

    # Save results to NPZ
    if config.get('save_npz', True):
        save_results_npz(model, config, results, config.get('output_path'))

    return best_l2, best_epoch, history


def save_results_npz(model, config, results, output_path=None):
    """
    Save training results to NPZ file.

    Args:
        model: Trained model
        config: Training configuration
        results: Dict containing training results
        output_path: Optional custom output path
    """
    if output_path is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(output_dir, 'helmholtz2d_multi_siren_v2_results.npz')

    # Extract model state dict (convert to numpy)
    model_state = {}
    for name, param in model.named_parameters():
        model_state[f'param_{name.replace(".", "_")}'] = param.detach().cpu().numpy()

    # Prepare data dict
    save_dict = {
        # Config
        'config_n_epochs': config.get('n_epochs', 0),
        'config_seed': config.get('seed', 0),
        'config_lr': config.get('lr', 0),
        'config_hidden_dim': config.get('hidden_dim', 0),
        'config_n_layers': config.get('n_layers', 0),
        'config_omega': config.get('omega', 0),
        'config_n_levels': config.get('n_levels', 0),
        'config_num_collocation': config.get('num_collocation', 0),
        'config_num_bc_per_edge': config.get('num_bc_per_edge', 0),

        # Results
        'best_l2': results.get('best_l2', 0),
        'best_epoch': results.get('best_epoch', 0),
        'final_l2': results.get('final_l2', 0),
        'total_time': results.get('total_time', 0),
        'ms_per_epoch': results.get('ms_per_epoch', 0),
        'n_params': results.get('n_params', 0),

        # History (epochs, l2, best_l2)
        'history_epochs': np.array([h[0] for h in results.get('history', [])]),
        'history_l2': np.array([h[1] for h in results.get('history', [])]),
        'history_best_l2': np.array([h[2] for h in results.get('history', [])]),

        # Predictions
        'u_pred': results.get('u_pred', np.array([])),
        'u_exact': results.get('u_exact', np.array([])),
    }

    # Add model state
    save_dict.update(model_state)

    # Save
    np.savez_compressed(output_path, **save_dict)
    print(f"Results saved: {output_path}")

    return output_path


def save_visualization(u_pred, u_exact, best_l2, best_epoch, config):
    """Save visualization of results."""
    import matplotlib.pyplot as plt

    # Create grid
    res = u_pred.shape[0]
    x = np.linspace(0, 1, res)
    y = np.linspace(0, 1, res)
    X, Y = np.meshgrid(x, y, indexing='ij')

    error = np.abs(u_pred - u_exact)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Exact solution
    im0 = axes[0].contourf(X, Y, u_exact, levels=50, cmap='RdBu_r')
    axes[0].set_title('Exact Solution')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0])

    # Prediction
    im1 = axes[1].contourf(X, Y, u_pred, levels=50, cmap='RdBu_r')
    axes[1].set_title(f'Prediction (L2={best_l2:.2e})')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[1])

    # Error
    im2 = axes[2].contourf(X, Y, error, levels=50, cmap='hot')
    axes[2].set_title(f'|Error| (max={error.max():.2e})')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    plt.colorbar(im2, ax=axes[2])

    # Cross-section
    mid = res // 2
    axes[3].plot(x, u_exact[:, mid], 'b-', label='Exact', linewidth=2)
    axes[3].plot(x, u_pred[:, mid], 'r--', label='Prediction', linewidth=2)
    axes[3].set_title('Cross-section at y=0.5')
    axes[3].set_xlabel('x')
    axes[3].set_ylabel('u')
    axes[3].legend()
    axes[3].grid(True)

    n_layers = config.get('n_layers', 2)
    plt.suptitle(f'Helmholtz 2D (Multi-SIREN {n_layers}L): Best L2={best_l2:.4e} @ epoch {best_epoch}', fontsize=14)
    plt.tight_layout()

    # Save
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, 'helmholtz2d_multi_siren_v2_result.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nVisualization saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Helmholtz 2D - Multi-layer SIREN (v2 with Adaptive LR)')
    parser.add_argument('--epochs', type=int, default=100000, help='Number of epochs')
    parser.add_argument('--seed', type=int, default=456, help='Random seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--omega', type=float, default=0.5, help='SIREN omega')
    parser.add_argument('--hidden', type=int, default=128, help='MLP hidden dim')
    parser.add_argument('--layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--no-plots', action='store_true', help='Disable plot saving')
    parser.add_argument('--no-save', action='store_true', help='Disable NPZ results saving')
    parser.add_argument('--output', type=str, default=None, help='Custom output path for NPZ file')
    parser.add_argument('--lr-patience', type=int, default=1000000, help='Epochs without improvement before LR reduction')
    parser.add_argument('--min-lr', type=float, default=5e-6, help='Minimum learning rate')
    parser.add_argument('--no-adaptive-lr', action='store_true', help='Disable adaptive LR reduction')
    args = parser.parse_args()

    config = {
        'n_epochs': args.epochs,
        'seed': args.seed,
        'lr': args.lr,
        'omega': args.omega,
        'hidden_dim': args.hidden,
        'n_layers': args.layers,
        'save_plots': not args.no_plots,
        'save_npz': not args.no_save,
        'output_path': args.output,
        'lr_patience': args.lr_patience,
        'min_lr': args.min_lr,
        'use_adaptive_lr': not args.no_adaptive_lr,

        # Fixed best config (paper a=100 setting, recovered from saved npz)
        'n_levels': 8,
        'log2_hashmap_size_1': 16,
        'log2_hashmap_size_2': 16,
        'log2_hashmap_size_3': 16,
        'bc_weight_cap': 50000.0,
        'num_collocation': 100000,    # paper: 100k (not 10k ablation default)
        'num_bc_per_edge': 25000,     # paper: 25k (not 5k)
        'eval_interval': 500,
    }

    train(config)
