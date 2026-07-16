"""
1+1D advection PINN with multi-layer SIREN MLP and Hermite-NGP encoding.

PDE:               du/dt + c * du/dx = 0
Domain:            x in [0, 2pi], t in [0, 1]
IC:                u(x, 0) = sin(x)
Periodic BC:       u(0, t) = u(2pi, t)
Exact solution:    u(x, t) = sin(x - c*t)

Usage:
    python examples/convection1d.py
    python examples/convection1d.py --epochs 60000 --c 80 --layers 2
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
# Problem Setup: 1D Advection Equation
# =============================================================================
# PDE: du/dt + c * du/dx = 0
# Domain: x in [0, 2pi], t in [0, 1]
# IC: u(x, 0) = sin(x)
# Periodic BC: u(0, t) = u(2pi, t)
# Exact solution: u(x, t) = sin(x - c*t)

C_DEFAULT = 80.0  # Advection speed
PI = np.pi
TWO_PI = 2 * np.pi

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Import CUDA extension for multi-layer SIREN
try:
    import hermite_mlp_cuda_v2
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("WARNING: hermite_mlp_cuda_v2 not available. Run: python setup_mlp_cuda_v2.py install")


def exact_solution(x, t, c):
    """Exact solution: u(x, t) = sin(x - c*t)"""
    return torch.sin(x - c * t)


def initial_condition(x):
    """Initial condition: u(x, 0) = sin(x)"""
    return torch.sin(x)


# =============================================================================
# CUDA V3 Multi-layer SIREN (from helmholtz2d_multi_siren_v2.py)
# =============================================================================

class HermiteLayerFunctionV3(torch.autograd.Function):
    """CUDA forward + PyTorch backward for Hermite propagation (2D style, adapted for 1D advection)."""

    @staticmethod
    def forward(ctx, h, dh_dx, dh_dt, d2h_dxx, d2h_dtt, weight, bias, omega, apply_activation):
        outputs = hermite_mlp_cuda_v2.forward(
            h.contiguous(), dh_dx.contiguous(), dh_dt.contiguous(),
            d2h_dxx.contiguous(), d2h_dtt.contiguous(),
            weight.contiguous(), bias.contiguous(),
            omega, apply_activation
        )
        out_h, out_dx, out_dt, out_dxx, out_dtt, save_z, save_dz_dx, save_dz_dt, save_d2z_dxx, save_d2z_dtt = outputs

        ctx.save_for_backward(
            h, dh_dx, dh_dt, d2h_dxx, d2h_dtt,
            weight,
            save_z, save_dz_dx, save_dz_dt
        )
        ctx.omega = omega
        ctx.apply_activation = apply_activation

        return out_h, out_dx, out_dt, out_dxx, out_dtt

    @staticmethod
    def backward(ctx, grad_h, grad_dx, grad_dt, grad_dxx, grad_dtt):
        h, dh_dx, dh_dt, d2h_dxx, d2h_dtt, weight, z, dz_dx, dz_dt = ctx.saved_tensors
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
            grad_z = grad_z + grad_dt * h_pp * dz_dt

            d2z_dxx = d2h_dxx @ weight.T
            grad_z = grad_z + grad_dxx * (h_ppp * dz_dx * dz_dx + h_pp * d2z_dxx)

            d2z_dtt = d2h_dtt @ weight.T
            grad_z = grad_z + grad_dtt * (h_ppp * dz_dt * dz_dt + h_pp * d2z_dtt)

            grad_dz_dx = grad_dx * h_p + grad_dxx * 2 * h_pp * dz_dx
            grad_dz_dt = grad_dt * h_p + grad_dtt * 2 * h_pp * dz_dt
            grad_d2z_dxx = grad_dxx * h_p
            grad_d2z_dtt = grad_dtt * h_p
        else:
            grad_z = grad_h
            grad_dz_dx = grad_dx
            grad_dz_dt = grad_dt
            grad_d2z_dxx = grad_dxx
            grad_d2z_dtt = grad_dtt

        grad_h_in = grad_z @ weight
        grad_dh_dx_in = grad_dz_dx @ weight
        grad_dh_dt_in = grad_dz_dt @ weight
        grad_d2h_dxx_in = grad_d2z_dxx @ weight
        grad_d2h_dtt_in = grad_d2z_dtt @ weight

        grad_weight = grad_z.T @ h
        grad_weight = grad_weight + grad_dz_dx.T @ dh_dx
        grad_weight = grad_weight + grad_dz_dt.T @ dh_dt
        grad_weight = grad_weight + grad_d2z_dxx.T @ d2h_dxx
        grad_weight = grad_weight + grad_d2z_dtt.T @ d2h_dtt

        grad_bias = grad_z.sum(dim=0)

        return grad_h_in, grad_dh_dx_in, grad_dh_dt_in, grad_d2h_dxx_in, grad_d2h_dtt_in, grad_weight, grad_bias, None, None


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

    def forward_with_derivatives_cuda(self, enc, dx, dt, dxx, dtt):
        """
        Forward with full Hermite propagation using CUDA V3.

        Returns:
            u: [N, 1] function values
            u_x: [N, 1] derivative w.r.t. x (normalized)
            u_t: [N, 1] derivative w.r.t. t
        """
        omega = self.omega_0
        h, dh_dx, dh_dt, d2h_dxx, d2h_dtt = enc, dx, dt, dxx, dtt

        # Hidden layers with activation
        for layer in self.layers:
            h, dh_dx, dh_dt, d2h_dxx, d2h_dtt = HermiteLayerFunctionV3.apply(
                h, dh_dx, dh_dt, d2h_dxx, d2h_dtt,
                layer.weight, layer.bias, omega, True
            )

        # Output layer without activation
        u, du_dx, du_dt, d2u_dxx, d2u_dtt = HermiteLayerFunctionV3.apply(
            h, dh_dx, dh_dt, d2h_dxx, d2h_dtt,
            self.output_layer.weight, self.output_layer.bias, omega, False
        )

        return u, du_dx, du_dt


# =============================================================================
# Hermite-NGP PINN Model with Multi-layer SIREN
# =============================================================================

class HermiteNGP_PINN(nn.Module):
    """
    Physics-Informed Neural Network with Hermite Hash Encoding and Multi-layer SIREN.

    Features:
    - Analytic derivatives from Hermite encoding
    - Multi-layer SIREN MLP with CUDA forward
    - GradNorm (adaptive loss balancing)
    - Causal training
    """
    def __init__(self, config=None):
        super().__init__()

        # Default best configuration
        config = config or {}
        self.c = config.get('c', C_DEFAULT)  # Advection speed
        self.n_levels = config.get('n_levels', 8)
        self.log2_hashmap_size_1 = config.get('log2_hashmap_size_1', 16)
        self.log2_hashmap_size_2 = config.get('log2_hashmap_size_2', 16)
        self.log2_hashmap_size_3 = config.get('log2_hashmap_size_3', 16)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.n_layers = config.get('n_layers', 2)
        self.omega = config.get('omega', 0.5)
        self.ic_weight_cap = config.get('ic_weight_cap', 100000.0)
        self.bc_weight_cap = config.get('bc_weight_cap', 100000.0)

        # Curriculum phases: (start_epoch, end_epoch, [active_level_indices])
        self.phases = []

        # Causal training config
        self.use_causal = config.get('use_causal', True)
        self.num_chunks = config.get('num_chunks', 16)
        self.causal_tol = config.get('causal_tol', 1.0)
        self.causal_tol_min = config.get('causal_tol_min', 0.01)
        self.causal_anneal_epochs = config.get('causal_anneal_epochs', 30000)

        # Hermite Hash Encoding (CUDA)
        from hermite_ngp.encoding.hermite_encoding_cuda import HermiteHashEncodingCUDA
        self.encoding = HermiteHashEncodingCUDA(
            n_input_dims=2,  # (x_norm, t_norm)
            n_levels=self.n_levels,
            n_features_per_level=2,
            log2_hashmap_size_1=self.log2_hashmap_size_1,
            log2_hashmap_size_2=self.log2_hashmap_size_2,
            log2_hashmap_size_3=self.log2_hashmap_size_3,
            base_resolution=4,
            per_level_scale=1.5,
        )

        # Multi-layer SIREN MLP
        encoding_dim = self.n_levels * 2
        self.mlp = MultiSIREN(encoding_dim, self.hidden_dim, self.n_layers, omega_0=self.omega)

        # Training state
        self.ic_weight = 100000.0
        self.bc_weight = 100000.0
        self.register_buffer('level_grad_mask', torch.ones(self.n_levels))

        self.to(device)

        # Config for IC/BC points
        self.num_ic = config.get('num_ic_per_edge', 5000)
        self.num_bc = config.get('num_bc_per_edge', 5000)

    def get_active_levels(self, epoch):
        """Get list of active levels for given epoch based on phase schedule."""
        for start, end, levels in self.phases:
            if start <= epoch < end:
                return levels
        return list(range(self.n_levels))

    def freeze_levels(self, levels_to_freeze):
        """Freeze gradients for specified levels."""
        self.level_grad_mask[:] = 1.0
        for l in levels_to_freeze:
            self.level_grad_mask[l] = 0.0

    def set_active_levels(self, levels):
        """Set active levels for phase learning."""
        self.level_grad_mask[:] = 0.0
        for l in levels:
            self.level_grad_mask[l] = 1.0

    def apply_level_mask(self):
        """Apply gradient mask to hash tables."""
        mask = self.level_grad_mask.view(-1, 1, 1)
        for ht in [self.encoding.hash_table_1, self.encoding.hash_table_2, self.encoding.hash_table_3]:
            if ht.grad is not None:
                ht.grad *= mask

    def forward(self, x):
        """Standard forward pass."""
        enc = self.encoding(x)
        return self.mlp(enc)

    def forward_with_derivatives(self, x):
        """
        Forward pass with analytic first derivatives using CUDA MLP.

        Returns:
            u: [N, 1] function values
            u_x: [N, 1] spatial derivative (physical coords)
            u_t: [N, 1] time derivative
        """
        # Hermite encoding with analytic derivatives
        enc, dx, dt, dxx, dtt = self.encoding.forward_with_second_derivatives_cuda(x)

        # MLP with Hermite propagation (CUDA)
        u, du_dx, du_dt = self.mlp.forward_with_derivatives_cuda(enc, dx, dt, dxx, dtt)

        # Chain rule: du/dx_phys = du/dx_norm / (2*pi)
        u_x = du_dx / TWO_PI
        u_t = du_dt

        return u, u_x, u_t

    def loss_pde(self, pts):
        """PDE loss: (u_t + c*u_x)^2"""
        u, u_x, u_t = self.forward_with_derivatives(pts)
        residual = u_t + self.c * u_x
        return (residual**2).mean()

    def get_causal_tol(self, epoch):
        """Get annealed causal tolerance for given epoch."""
        if epoch >= self.causal_anneal_epochs:
            return self.causal_tol_min
        progress = epoch / self.causal_anneal_epochs
        return self.causal_tol * (1 - progress) + self.causal_tol_min * progress

    def loss_pde_causal(self, pts, num_chunks=16, causal_tol=1.0):
        """
        Causal PDE loss with time-based weighting.

        Points are sorted by time, split into chunks, and weighted by
        accumulated error from earlier time chunks.
        """
        # Sort points by time (column 1)
        t_vals = pts[:, 1]
        sorted_idx = torch.argsort(t_vals)
        pts_sorted = pts[sorted_idx]

        # Compute residuals for all points
        u, u_x, u_t = self.forward_with_derivatives(pts_sorted)
        residual_sq = (u_t + self.c * u_x) ** 2

        # Split into chunks
        n = len(pts_sorted)
        chunk_size = n // num_chunks
        chunk_losses = []

        for i in range(num_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < num_chunks - 1 else n
            chunk_loss = residual_sq[start:end].mean()
            chunk_losses.append(chunk_loss)

        chunk_losses = torch.stack(chunk_losses)

        # Compute causal weights: w_i = exp(-tol * sum_{j<i} L_j)
        cumsum = torch.cumsum(chunk_losses.detach(), dim=0)
        cumsum_shifted = torch.cat([torch.zeros(1, device=pts.device), cumsum[:-1]])
        weights = torch.exp(-causal_tol * cumsum_shifted)

        # Weighted mean
        total_loss = (chunk_losses * weights).sum() / (weights.sum() + 1e-8)
        return total_loss

    def loss_ic(self, ic_pts, ic_vals):
        """Initial condition loss: (u - u_exact)^2"""
        u = self.forward(ic_pts)
        return ((u.squeeze() - ic_vals)**2).mean()

    def loss_bc(self, bc_pts_left, bc_pts_right):
        """Periodic boundary condition loss: (u(0,t) - u(2pi,t))^2"""
        u_left = self.forward(bc_pts_left)
        u_right = self.forward(bc_pts_right)
        return ((u_left - u_right)**2).mean()

    def generate_pde_points(self, n):
        """Generate collocation points in [0,1] x [0,1] (normalized domain)."""
        return torch.rand(n, 2, device=device)

    def generate_ic_points(self, n_per_edge):
        """Generate IC points on t=0."""
        x_norm = torch.rand(n_per_edge, device=device)
        t_norm = torch.zeros_like(x_norm)
        pts = torch.stack([x_norm, t_norm], dim=1)
        x_phys = x_norm * TWO_PI
        vals = initial_condition(x_phys)
        return pts, vals

    def generate_bc_points(self, n_per_edge):
        """Generate BC points for periodic boundary."""
        t = torch.rand(n_per_edge, device=device)
        pts_left = torch.stack([torch.zeros_like(t), t], dim=1)
        pts_right = torch.stack([torch.ones_like(t), t], dim=1)
        return pts_left, pts_right

    def evaluate(self, resolution=100):
        """Evaluate L2 error on uniform grid."""
        with torch.no_grad():
            # Physical domain
            x_phys = torch.linspace(0, TWO_PI, resolution, device=device)
            t = torch.linspace(0, 1, resolution, device=device)
            X_phys, T = torch.meshgrid(x_phys, t, indexing='ij')

            # Normalize for model input
            X_norm = X_phys / TWO_PI
            pts = torch.stack([X_norm.flatten(), T.flatten()], dim=1)

            u_pred = self.forward(pts).reshape(resolution, resolution)
            u_exact = exact_solution(X_phys, T, self.c)

            # Relative L2 error
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
    n_epochs = config.get('n_epochs', 60000)
    seed = config.get('seed', 456)
    lr = config.get('lr', 1e-4)
    c = config.get('c', C_DEFAULT)
    num_collocation = config.get('num_collocation', 10000)
    num_ic_per_edge = config.get('num_ic_per_edge', 5000)
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
    print("1D Advection Equation (v3 with Multi-layer SIREN)")
    print("=" * 70)
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Advection speed c = {c}")
    print(f"Domain: x in [0, 2pi], t in [0, 1]")
    print(f"Wave wraps {c / TWO_PI:.1f} times during t in [0, 1]")
    print(f"Epochs: {n_epochs}, Seed: {seed}")
    print(f"Hidden: {config.get('hidden_dim', 256)}, Layers: {config.get('n_layers', 2)}")
    use_causal = config.get('use_causal', True)
    if use_causal:
        print(f"Causal training: ON (tol: {config.get('causal_tol', 1.0)} -> {config.get('causal_tol_min', 0.01)} over {config.get('causal_anneal_epochs', 30000)} epochs)")
    else:
        print(f"Causal training: OFF")
    if use_adaptive_lr:
        print(f"Adaptive LR: ON (patience={lr_patience}, min_lr={min_lr:.0e})")
    else:
        print(f"Adaptive LR: OFF")
    print("=" * 70)

    if not CUDA_AVAILABLE:
        print("\nERROR: CUDA extension not available!")
        print("Please run: python setup_mlp_cuda_v2.py install")
        return None, None, None, None, None

    # Create model
    config['c'] = c
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
    for warmup_epoch in range(20):
        pts = torch.rand(num_collocation, 2, device=device)
        ic_pts, ic_vals = model.generate_ic_points(model.num_ic)
        bc_pts_left, bc_pts_right = model.generate_bc_points(model.num_bc)

        if model.use_causal:
            loss_pde = model.loss_pde_causal(pts, model.num_chunks, model.causal_tol)
        else:
            loss_pde = model.loss_pde(pts)
        loss_ic = model.loss_ic(ic_pts, ic_vals)
        loss_bc = model.loss_bc(bc_pts_left, bc_pts_right)
        loss = loss_pde + model.ic_weight * loss_ic + model.bc_weight * loss_bc

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

    # Track phase transitions
    current_active = None

    for epoch in range(n_epochs):
        # Phase learning: update active levels
        new_active = model.get_active_levels(epoch)
        if new_active != current_active:
            model.set_active_levels(new_active)
            if current_active is not None:
                print(f"  [Phase] Epoch {epoch}: activating levels {new_active}")
            current_active = new_active

        # Sample random points every epoch
        pts = torch.rand(num_collocation, 2, device=device)
        ic_pts, ic_vals = model.generate_ic_points(model.num_ic)
        bc_pts_left, bc_pts_right = model.generate_bc_points(model.num_bc)

        # Get current causal tolerance
        current_tol = model.get_causal_tol(epoch) if model.use_causal else 0

        # GradNorm: adaptive loss balancing (every 100 epochs)
        if (epoch + 1) % 100 == 0:
            if model.use_causal:
                l_pde = model.loss_pde_causal(pts, model.num_chunks, current_tol)
            else:
                l_pde = model.loss_pde(pts)
            l_ic = model.loss_ic(ic_pts, ic_vals)
            l_bc = model.loss_bc(bc_pts_left, bc_pts_right)

            optimizer.zero_grad()
            l_pde.backward(retain_graph=True)
            model.apply_level_mask()
            grad_pde = get_grad_norm(model)

            optimizer.zero_grad()
            l_ic.backward(retain_graph=True)
            model.apply_level_mask()
            grad_ic = get_grad_norm(model)

            optimizer.zero_grad()
            l_bc.backward(retain_graph=True)
            model.apply_level_mask()
            grad_bc = get_grad_norm(model)

            if grad_ic > 1e-8:
                ratio = (grad_pde / grad_ic).item()
                model.ic_weight = 0.9 * model.ic_weight + 0.1 * ratio
                model.ic_weight = max(1.0, min(model.ic_weight_cap, model.ic_weight))

            if grad_bc > 1e-8:
                ratio = (grad_pde / grad_bc).item()
                model.bc_weight = 0.9 * model.bc_weight + 0.1 * ratio
                model.bc_weight = max(1.0, min(model.bc_weight_cap, model.bc_weight))

            loss = l_pde + model.ic_weight * l_ic + model.bc_weight * l_bc
        else:
            # Compute loss
            if model.use_causal:
                loss_pde = model.loss_pde_causal(pts, model.num_chunks, current_tol)
            else:
                loss_pde = model.loss_pde(pts)
            loss_ic = model.loss_ic(ic_pts, ic_vals)
            loss_bc = model.loss_bc(bc_pts_left, bc_pts_right)
            loss = loss_pde + model.ic_weight * loss_ic + model.bc_weight * loss_bc

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

            causal_tol_current = model.get_causal_tol(epoch) if model.use_causal else 0
            print(f"  Epoch {epoch+1:6d}: L2={l2:.4e}, best={best_l2:.4e} @{best_epoch}, "
                  f"ic_w={model.ic_weight:.1f}, bc_w={model.bc_weight:.1f}, "
                  f"causal_tol={causal_tol_current:.3f}, lr={lr_current:.2e}, time={elapsed:.0f}s")

    # Restore best model
    if best_state is not None:
        for name, param in model.named_parameters():
            if name in best_state:
                param.data = best_state[name]

    # Final evaluation
    l2_final, u_pred, u_exact = model.evaluate(resolution=200)
    elapsed_total = time.perf_counter() - t0

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Best L2 Error: {best_l2:.4e} @ epoch {best_epoch}")
    print(f"Total Time: {elapsed_total:.1f}s ({elapsed_total/n_epochs*1000:.2f} ms/epoch)")

    # Collect results
    results = {
        'best_l2': best_l2,
        'best_epoch': best_epoch,
        'final_l2': l2_final,
        'total_time': elapsed_total,
        'ms_per_epoch': elapsed_total / n_epochs * 1000,
        'n_params': n_params,
        'history': history,
        'u_pred': u_pred,
        'u_exact': u_exact,
    }

    # Save visualization
    if save_plots:
        save_visualization(u_pred, u_exact, best_l2, best_epoch, config)

    # Save NPZ results
    save_npz = config.get('save_npz', True)
    output_path = config.get('output_path', None)
    if save_npz:
        save_results_npz(model, config, results, output_path)

    return best_l2, best_epoch, history, model, results


def save_results_npz(model, config, results, output_path=None):
    """Save model state, config, and results to NPZ file."""
    if output_path is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(output_dir, 'advection1d_v3_results.npz')

    # Collect model state
    model_state = {}
    for name, param in model.named_parameters():
        model_state[f'param_{name.replace(".", "_")}'] = param.detach().cpu().numpy()

    # Build save dict
    save_dict = {
        # Config
        'config_n_epochs': config.get('n_epochs', 0),
        'config_seed': config.get('seed', 0),
        'config_lr': config.get('lr', 0),
        'config_c': config.get('c', C_DEFAULT),
        'config_omega': config.get('omega', 0.5),
        'config_hidden_dim': config.get('hidden_dim', 256),
        'config_n_layers': config.get('n_layers', 2),
        'config_n_levels': config.get('n_levels', 8),
        'config_use_causal': config.get('use_causal', True),
        'config_num_chunks': config.get('num_chunks', 16),
        'config_causal_tol': config.get('causal_tol', 1.0),
        'config_causal_tol_min': config.get('causal_tol_min', 0.01),
        'config_causal_anneal_epochs': config.get('causal_anneal_epochs', 30000),
        'config_num_collocation': config.get('num_collocation', 10000),
        'config_num_ic_per_edge': config.get('num_ic_per_edge', 5000),
        'config_num_bc_per_edge': config.get('num_bc_per_edge', 5000),
        'config_lr_patience': config.get('lr_patience', 2000),
        'config_min_lr': config.get('min_lr', 1e-7),
        'config_use_adaptive_lr': config.get('use_adaptive_lr', True),

        # Results
        'best_l2': results.get('best_l2', 0),
        'best_epoch': results.get('best_epoch', 0),
        'final_l2': results.get('final_l2', 0),
        'total_time': results.get('total_time', 0),
        'ms_per_epoch': results.get('ms_per_epoch', 0),
        'n_params': results.get('n_params', 0),

        # History
        'history_epochs': np.array([h[0] for h in results.get('history', [])]),
        'history_l2': np.array([h[1] for h in results.get('history', [])]),
        'history_best_l2': np.array([h[2] for h in results.get('history', [])]),

        # Predictions
        'u_pred': results.get('u_pred', np.array([])),
        'u_exact': results.get('u_exact', np.array([])),
    }

    # Add model parameters
    save_dict.update(model_state)

    # Save
    np.savez_compressed(output_path, **save_dict)
    print(f"Results saved: {output_path}")


def save_visualization(u_pred, u_exact, best_l2, best_epoch, config):
    """Save visualization of results."""
    import matplotlib.pyplot as plt

    c = config.get('c', C_DEFAULT)
    res = u_pred.shape[0]
    x = np.linspace(0, TWO_PI, res)
    t = np.linspace(0, 1, res)
    X, T = np.meshgrid(x, t, indexing='ij')

    error = np.abs(u_pred - u_exact)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: 2D contour plots
    im0 = axes[0, 0].contourf(T, X, u_exact, levels=50, cmap='RdBu_r')
    axes[0, 0].set_title('Exact Solution')
    axes[0, 0].set_xlabel('t')
    axes[0, 0].set_ylabel('x')
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].contourf(T, X, u_pred, levels=50, cmap='RdBu_r')
    axes[0, 1].set_title(f'Prediction (L2={best_l2:.2e})')
    axes[0, 1].set_xlabel('t')
    axes[0, 1].set_ylabel('x')
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[0, 2].contourf(T, X, error, levels=50, cmap='hot')
    axes[0, 2].set_title(f'|Error| (max={error.max():.2e})')
    axes[0, 2].set_xlabel('t')
    axes[0, 2].set_ylabel('x')
    plt.colorbar(im2, ax=axes[0, 2])

    # Row 2: 1D slices
    time_slices = [0, res//4, res//2, 3*res//4, res-1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(time_slices)))

    for i, t_idx in enumerate(time_slices):
        t_val = t[t_idx]
        axes[1, 0].plot(x, u_exact[:, t_idx], '-', color=colors[i],
                        label=f't={t_val:.2f}', linewidth=2)
    axes[1, 0].set_title('Exact Solution at Different Times')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('u')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    for i, t_idx in enumerate(time_slices):
        t_val = t[t_idx]
        axes[1, 1].plot(x, u_pred[:, t_idx], '-', color=colors[i],
                        label=f't={t_val:.2f}', linewidth=2)
    axes[1, 1].set_title('Prediction at Different Times')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('u')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # Comparison at t=0.5
    mid_t = res // 2
    axes[1, 2].plot(x, u_exact[:, mid_t], 'b-', label='Exact', linewidth=2)
    axes[1, 2].plot(x, u_pred[:, mid_t], 'r--', label='Prediction', linewidth=2)
    axes[1, 2].set_title(f'Comparison at t={t[mid_t]:.2f}')
    axes[1, 2].set_xlabel('x')
    axes[1, 2].set_ylabel('u')
    axes[1, 2].legend()
    axes[1, 2].grid(True)

    n_layers = config.get('n_layers', 2)
    plt.suptitle(f'1D Advection (c={c}, Multi-SIREN {n_layers}L): Best L2={best_l2:.4e} @ epoch {best_epoch}', fontsize=14)
    plt.tight_layout()

    # Save
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, 'advection1d_v3_result.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nVisualization saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='1D Advection Equation (v3 with Multi-layer SIREN)')
    parser.add_argument('--epochs', type=int, default=100000, help='Number of epochs')
    parser.add_argument('--seed', type=int, default=456, help='Random seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--c', type=float, default=30.0, help='Advection speed')
    parser.add_argument('--omega', type=float, default=0.5, help='SIREN omega')
    parser.add_argument('--hidden', type=int, default=128, help='MLP hidden dim')
    parser.add_argument('--layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--no-plots', action='store_true', help='Disable plot saving')
    parser.add_argument('--n-levels', type=int, default=8, help='Number of hash encoding levels')
    parser.add_argument('--no-causal', action='store_true', help='Disable causal training')
    parser.add_argument('--num-chunks', type=int, default=16, help='Number of time chunks for causal')
    parser.add_argument('--causal-tol', type=float, default=1.0, help='Initial causal tolerance')
    parser.add_argument('--causal-tol-min', type=float, default=0.01, help='Final causal tolerance')
    parser.add_argument('--causal-anneal', type=int, default=30000, help='Epochs to anneal causal tolerance')
    parser.add_argument('--lr-patience', type=int, default=2000, help='Epochs without improvement before LR reduction')
    parser.add_argument('--min-lr', type=float, default=1e-7, help='Minimum learning rate')
    parser.add_argument('--no-adaptive-lr', action='store_true', help='Disable adaptive LR reduction')
    parser.add_argument('--no-save', action='store_true', help='Disable NPZ results saving')
    parser.add_argument('--output', type=str, default=None, help='Output NPZ file path')
    args = parser.parse_args()

    config = {
        'n_epochs': args.epochs,
        'seed': args.seed,
        'lr': args.lr,
        'c': args.c,
        'omega': args.omega,
        'hidden_dim': args.hidden,
        'n_layers': args.layers,
        'save_plots': not args.no_plots,
        'use_causal': not args.no_causal,
        'num_chunks': args.num_chunks,
        'causal_tol': args.causal_tol,
        'causal_tol_min': args.causal_tol_min,
        'causal_anneal_epochs': args.causal_anneal,
        'lr_patience': args.lr_patience,
        'min_lr': args.min_lr,
        'use_adaptive_lr': not args.no_adaptive_lr,

        # Hash encoding config
        'n_levels': args.n_levels,
        'log2_hashmap_size_1': 16,
        'log2_hashmap_size_2': 16,
        'log2_hashmap_size_3': 16,
        'ic_weight_cap': 5000.0,
        'bc_weight_cap': 5000.0,
        'num_collocation': 10000,
        'num_ic_per_edge': 5000,
        'num_bc_per_edge': 5000,
        'eval_interval': 100,

        # Save config
        'save_npz': not args.no_save,
        'output_path': args.output,
    }

    train(config)
