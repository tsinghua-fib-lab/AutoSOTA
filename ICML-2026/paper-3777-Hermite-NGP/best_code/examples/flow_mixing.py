"""
Flow mixing 2D + time PINN with multi-layer SIREN MLP and Hermite-NGP
encoding. Fluid swirls around the origin with angular velocity that
depends on radial distance.

PDE:               du/dt + a(x,y) * du/dx + b(x,y) * du/dy = 0
Domain:            t in [0, 4], (x, y) in [-4, 4]^2

Velocity field:
    r   = sqrt(x^2 + y^2)
    v_t = (1/cosh(r))^2 * tanh(r)
    w   = (v_t / v_max) / r
    a   = -(v_t/v_max) * (y/r)
    b   =  (v_t/v_max) * (x/r)

Exact solution:    u(t, x, y) = -tanh((y/2)*cos(w*t) - (x/2)*sin(w*t))

Usage:
    python examples/flow_mixing.py
    python examples/flow_mixing.py --epochs 200000 --cosine-scheduler
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

# =============================================================================
# Problem Setup: Flow Mixing 3D
# =============================================================================
# PDE: du/dt + a*du/dx + b*du/dy = 0
# Domain: t in [0, 4], (x, y) in [-4, 4]^2
# Exact: u = -tanh((y/2)*cos(w*t) - (x/2)*sin(w*t))

V_MAX = 0.385  # Maximum tangential velocity
PI = np.pi

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Import CUDA extension
try:
    import hermite_mlp_cuda_3d_v2
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("WARNING: hermite_mlp_cuda_3d_v2 not available. Run: python setup_mlp_cuda_v2.py install")


def flow_params(x, y, v_max=V_MAX):
    """
    Compute flow parameters at spatial points.

    Returns:
        r: radial distance
        v_t: tangential velocity
        omega: angular velocity
        a: x-velocity component (for PDE)
        b: y-velocity component (for PDE)
    """
    r = torch.sqrt(x**2 + y**2 + 1e-8)  # Add eps to avoid division by zero
    v_t = (1.0 / torch.cosh(r))**2 * torch.tanh(r)
    omega = (v_t / v_max) / r
    a = -(v_t / v_max) * (y / r)
    b = (v_t / v_max) * (x / r)
    return r, v_t, omega, a, b


def exact_solution(t, x, y, v_max=V_MAX):
    """Exact solution: u = -tanh((y/2)*cos(w*t) - (x/2)*sin(w*t))"""
    _, _, omega, _, _ = flow_params(x, y, v_max)
    return -torch.tanh((y / 2) * torch.cos(omega * t) - (x / 2) * torch.sin(omega * t))


def initial_condition(x, y, v_max=V_MAX):
    """Initial condition at t=0: u(0, x, y) = -tanh(y/2)"""
    # At t=0: cos(0)=1, sin(0)=0, so u = -tanh(y/2)
    return -torch.tanh(y / 2)


# =============================================================================
# Hermite Hash Encoding (from hermite_ngp)
# =============================================================================
from hermite_ngp.encoding.hermite_encoding_cuda import HermiteHashEncodingCUDA_3D as HermiteHashEncoding


# =============================================================================
# CUDA V3 MLP: CUDA Forward + PyTorch Backward (3D version)
# =============================================================================

class HermiteLayerFunction3D_V3(torch.autograd.Function):
    """CUDA forward + PyTorch backward for Hermite propagation (3D)."""

    @staticmethod
    def forward(ctx, h, dh_dx, dh_dy, dh_dz, d2h_dxx, d2h_dyy, d2h_dzz, weight, bias, omega, apply_activation):
        outputs = hermite_mlp_cuda_3d_v2.forward(
            h.contiguous(), dh_dx.contiguous(), dh_dy.contiguous(), dh_dz.contiguous(),
            d2h_dxx.contiguous(), d2h_dyy.contiguous(), d2h_dzz.contiguous(),
            weight.contiguous(), bias.contiguous(),
            omega, apply_activation
        )
        out_h, out_dx, out_dy, out_dz, out_dxx, out_dyy, out_dzz = outputs[:7]
        save_z, save_dz_dx, save_dz_dy, save_dz_dz, save_d2z_dxx, save_d2z_dyy, save_d2z_dzz = outputs[7:]

        ctx.save_for_backward(
            h, dh_dx, dh_dy, dh_dz, d2h_dxx, d2h_dyy, d2h_dzz,
            weight,
            save_z, save_dz_dx, save_dz_dy, save_dz_dz
        )
        ctx.omega = omega
        ctx.apply_activation = apply_activation

        return out_h, out_dx, out_dy, out_dz, out_dxx, out_dyy, out_dzz

    @staticmethod
    def backward(ctx, grad_h, grad_dx, grad_dy, grad_dz, grad_dxx, grad_dyy, grad_dzz):
        h, dh_dx, dh_dy, dh_dz, d2h_dxx, d2h_dyy, d2h_dzz, weight, z, dz_dx, dz_dy, dz_dz = ctx.saved_tensors
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
            grad_z = grad_z + grad_dz * h_pp * dz_dz

            d2z_dxx = d2h_dxx @ weight.T
            grad_z = grad_z + grad_dxx * (h_ppp * dz_dx * dz_dx + h_pp * d2z_dxx)

            d2z_dyy = d2h_dyy @ weight.T
            grad_z = grad_z + grad_dyy * (h_ppp * dz_dy * dz_dy + h_pp * d2z_dyy)

            d2z_dzz = d2h_dzz @ weight.T
            grad_z = grad_z + grad_dzz * (h_ppp * dz_dz * dz_dz + h_pp * d2z_dzz)

            grad_dz_dx = grad_dx * h_p + grad_dxx * 2 * h_pp * dz_dx
            grad_dz_dy = grad_dy * h_p + grad_dyy * 2 * h_pp * dz_dy
            grad_dz_dz = grad_dz * h_p + grad_dzz * 2 * h_pp * dz_dz

            grad_d2z_dxx = grad_dxx * h_p
            grad_d2z_dyy = grad_dyy * h_p
            grad_d2z_dzz = grad_dzz * h_p
        else:
            grad_z = grad_h
            grad_dz_dx = grad_dx
            grad_dz_dy = grad_dy
            grad_dz_dz = grad_dz
            grad_d2z_dxx = grad_dxx
            grad_d2z_dyy = grad_dyy
            grad_d2z_dzz = grad_dzz

        grad_h_in = grad_z @ weight
        grad_dh_dx_in = grad_dz_dx @ weight
        grad_dh_dy_in = grad_dz_dy @ weight
        grad_dh_dz_in = grad_dz_dz @ weight
        grad_d2h_dxx_in = grad_d2z_dxx @ weight
        grad_d2h_dyy_in = grad_d2z_dyy @ weight
        grad_d2h_dzz_in = grad_d2z_dzz @ weight

        grad_weight = grad_z.T @ h
        grad_weight = grad_weight + grad_dz_dx.T @ dh_dx
        grad_weight = grad_weight + grad_dz_dy.T @ dh_dy
        grad_weight = grad_weight + grad_dz_dz.T @ dh_dz
        grad_weight = grad_weight + grad_d2z_dxx.T @ d2h_dxx
        grad_weight = grad_weight + grad_d2z_dyy.T @ d2h_dyy
        grad_weight = grad_weight + grad_d2z_dzz.T @ d2h_dzz

        grad_bias = grad_z.sum(dim=0)

        return (grad_h_in, grad_dh_dx_in, grad_dh_dy_in, grad_dh_dz_in,
                grad_d2h_dxx_in, grad_d2h_dyy_in, grad_d2h_dzz_in,
                grad_weight, grad_bias, None, None)


class SIREN_CUDA_3D_V3(nn.Module):
    """SIREN MLP with CUDA V3 for 3D (supports multiple hidden layers)."""

    def __init__(self, input_dim, hidden_dim=256, n_layers=2, omega_0=0.5):
        super().__init__()
        self.omega_0 = omega_0
        self.n_layers = n_layers

        self.layers = nn.ModuleList()
        dims = [input_dim] + [hidden_dim] * n_layers
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
        self.output_layer = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self):
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
        h = x
        for layer in self.layers:
            h = torch.sin(self.omega_0 * layer(h))
        return self.output_layer(h)

    def forward_with_derivatives_cuda(self, enc, dx, dy, dz, dxx, dyy, dzz):
        """
        Forward with Hermite propagation for first derivatives only.

        Coordinate mapping: dim0 (t), dim1 (x), dim2 (y)
        Returns: u, du_dt, du_dx, du_dy
        """
        omega = self.omega_0
        h, dh_dt, dh_dx, dh_dy, d2h_dtt, d2h_dxx, d2h_dyy = enc, dx, dy, dz, dxx, dyy, dzz

        for layer in self.layers:
            h, dh_dt, dh_dx, dh_dy, d2h_dtt, d2h_dxx, d2h_dyy = HermiteLayerFunction3D_V3.apply(
                h, dh_dt, dh_dx, dh_dy, d2h_dtt, d2h_dxx, d2h_dyy,
                layer.weight, layer.bias, omega, True
            )

        u, du_dt, du_dx, du_dy, _, _, _ = HermiteLayerFunction3D_V3.apply(
            h, dh_dt, dh_dx, dh_dy, d2h_dtt, d2h_dxx, d2h_dyy,
            self.output_layer.weight, self.output_layer.bias, omega, False
        )

        return u, du_dt, du_dx, du_dy


# =============================================================================
# Model
# =============================================================================

class HermiteNGP_PINN_FlowMixing(nn.Module):
    """PINN for Flow Mixing with Hermite encoding + CUDA MLP."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.v_max = config.get('v_max', V_MAX)

        # Domain bounds
        self.t_min, self.t_max = 0.0, 4.0
        self.x_min, self.x_max = -4.0, 4.0
        self.y_min, self.y_max = -4.0, 4.0

        # Hermite Hash Encoding
        n_levels = config.get('n_levels', 8)
        log2_hashmap_size = config.get('log2_hashmap_size', 14)
        self.encoding = HermiteHashEncoding(
            n_input_dims=3,
            n_levels=n_levels,
            n_features_per_level=2,
            log2_hashmap_size_1=log2_hashmap_size,
            log2_hashmap_size_2=log2_hashmap_size,
            log2_hashmap_size_3=log2_hashmap_size,
            log2_hashmap_size_4=log2_hashmap_size,
            base_resolution=4,
            per_level_scale=1.5
        ).to(device)

        # SIREN MLP with CUDA V3
        input_dim = self.encoding.output_dim
        hidden_dim = config.get('hidden_dim', 256)
        n_layers = config.get('n_layers', 2)
        omega = config.get('omega', 0.5)
        self.mlp = SIREN_CUDA_3D_V3(input_dim, hidden_dim, n_layers, omega).to(device)

        self.n_levels = n_levels
        self.pde_weight = config.get('pde_weight', 10.0)
        self.ic_weight = config.get('ic_weight', 1.0)
        self.bc_weight = config.get('bc_weight', 1.0)
        self.ic_weight_cap = config.get('ic_weight_cap', 1000.0)
        self.bc_weight_cap = config.get('bc_weight_cap', 1000.0)

        # Sampling config
        self.num_ic = config.get('num_ic', 5000)
        self.num_bc_per_edge = config.get('num_bc_per_edge', 2000)

        # Level mask for curriculum
        self.register_buffer('level_grad_mask', torch.ones(n_levels, device=device))

        # Curriculum phases (coarse-to-fine)
        self.phases = [
            (0, 5000, [0, 1, 2, 3]),           # First 5k: coarse levels only
            (5000, 15000, [0, 1, 2, 3, 4, 5]), # 5k-15k: add medium levels
            (15000, float('inf'), list(range(n_levels))),  # 15k+: all levels
        ]

    def get_active_levels(self, epoch):
        """Get active levels for curriculum learning."""
        for start, end, levels in self.phases:
            if start <= epoch < end:
                return levels
        return list(range(self.n_levels))

    def normalize_coords(self, t, x, y):
        """Normalize coordinates to [0, 1] for hash encoding."""
        t_norm = (t - self.t_min) / (self.t_max - self.t_min)
        x_norm = (x - self.x_min) / (self.x_max - self.x_min)
        y_norm = (y - self.y_min) / (self.y_max - self.y_min)
        return t_norm, x_norm, y_norm

    def generate_ic_points(self, n_points=None):
        """Generate initial condition points at t=0."""
        if n_points is None:
            n_points = self.num_ic

        x = torch.rand(n_points, device=device) * (self.x_max - self.x_min) + self.x_min
        y = torch.rand(n_points, device=device) * (self.y_max - self.y_min) + self.y_min
        t = torch.zeros(n_points, device=device)

        # Normalize for encoding
        t_norm, x_norm, y_norm = self.normalize_coords(t, x, y)
        pts = torch.stack([t_norm, x_norm, y_norm], dim=1)

        # Exact IC values
        vals = exact_solution(t, x, y, self.v_max)
        return pts, vals, t, x, y

    def generate_bc_points(self, n_per_edge=None):
        """Generate boundary points on all 4 spatial edges for all t."""
        if n_per_edge is None:
            n_per_edge = self.num_bc_per_edge

        pts_list = []
        t_list, x_list, y_list = [], [], []

        # x = -4 boundary
        t = torch.rand(n_per_edge, device=device) * self.t_max
        x = torch.full((n_per_edge,), self.x_min, device=device)
        y = torch.rand(n_per_edge, device=device) * (self.y_max - self.y_min) + self.y_min
        t_norm, x_norm, y_norm = self.normalize_coords(t, x, y)
        pts_list.append(torch.stack([t_norm, x_norm, y_norm], dim=1))
        t_list.append(t); x_list.append(x); y_list.append(y)

        # x = 4 boundary
        t = torch.rand(n_per_edge, device=device) * self.t_max
        x = torch.full((n_per_edge,), self.x_max, device=device)
        y = torch.rand(n_per_edge, device=device) * (self.y_max - self.y_min) + self.y_min
        t_norm, x_norm, y_norm = self.normalize_coords(t, x, y)
        pts_list.append(torch.stack([t_norm, x_norm, y_norm], dim=1))
        t_list.append(t); x_list.append(x); y_list.append(y)

        # y = -4 boundary
        t = torch.rand(n_per_edge, device=device) * self.t_max
        x = torch.rand(n_per_edge, device=device) * (self.x_max - self.x_min) + self.x_min
        y = torch.full((n_per_edge,), self.y_min, device=device)
        t_norm, x_norm, y_norm = self.normalize_coords(t, x, y)
        pts_list.append(torch.stack([t_norm, x_norm, y_norm], dim=1))
        t_list.append(t); x_list.append(x); y_list.append(y)

        # y = 4 boundary
        t = torch.rand(n_per_edge, device=device) * self.t_max
        x = torch.rand(n_per_edge, device=device) * (self.x_max - self.x_min) + self.x_min
        y = torch.full((n_per_edge,), self.y_max, device=device)
        t_norm, x_norm, y_norm = self.normalize_coords(t, x, y)
        pts_list.append(torch.stack([t_norm, x_norm, y_norm], dim=1))
        t_list.append(t); x_list.append(x); y_list.append(y)

        pts = torch.cat(pts_list, dim=0)
        t_all = torch.cat(t_list)
        x_all = torch.cat(x_list)
        y_all = torch.cat(y_list)

        # Exact BC values
        vals = exact_solution(t_all, x_all, y_all, self.v_max)
        return pts, vals

    def freeze_levels(self, levels_to_freeze):
        self.level_grad_mask[:] = 1.0
        for l in levels_to_freeze:
            self.level_grad_mask[l] = 0.0

    def apply_level_mask(self):
        mask = self.level_grad_mask.view(-1, 1, 1)
        for ht in [self.encoding.hash_table_1, self.encoding.hash_table_2,
                   self.encoding.hash_table_3, self.encoding.hash_table_4]:
            if ht.grad is not None:
                ht.grad *= mask

    def forward(self, pts):
        """Forward pass with normalized coordinates."""
        enc = self.encoding(pts)
        return self.mlp(enc)

    def forward_with_derivatives(self, pts, t_raw, x_raw, y_raw):
        """
        Forward pass returning u and first derivatives.

        Input pts: normalized coordinates (N, 3) with [t_norm, x_norm, y_norm]
        t_raw, x_raw, y_raw: original (unnormalized) coordinates for velocity field

        Returns: u, du_dt, du_dx, du_dy (in original coordinate scale)
        """
        enc, dx, dy, dz, dxx, dyy, dzz = self.encoding.forward_with_second_derivatives_cuda(pts)
        u, du_dt_norm, du_dx_norm, du_dy_norm = self.mlp.forward_with_derivatives_cuda(
            enc, dx, dy, dz, dxx, dyy, dzz
        )

        # Scale derivatives back to original coordinates
        # d/dt_raw = d/dt_norm * dt_norm/dt_raw = d/dt_norm / (t_max - t_min)
        du_dt = du_dt_norm / (self.t_max - self.t_min)
        du_dx = du_dx_norm / (self.x_max - self.x_min)
        du_dy = du_dy_norm / (self.y_max - self.y_min)

        return u, du_dt, du_dx, du_dy

    def loss_pde(self, pts, t_raw, x_raw, y_raw):
        """
        PDE residual loss: du/dt + a*du/dx + b*du/dy = 0
        """
        u, du_dt, du_dx, du_dy = self.forward_with_derivatives(pts, t_raw, x_raw, y_raw)

        # Get velocity field at these points
        _, _, _, a, b = flow_params(x_raw, y_raw, self.v_max)

        # Residual
        residual = du_dt + a.unsqueeze(-1) * du_dx + b.unsqueeze(-1) * du_dy
        return (residual**2).mean()

    def loss_ic(self, ic_pts, ic_vals):
        """Initial condition loss."""
        u = self.forward(ic_pts)
        return ((u.squeeze() - ic_vals)**2).mean()

    def loss_bc(self, bc_pts, bc_vals):
        """Boundary condition loss."""
        u = self.forward(bc_pts)
        return ((u.squeeze() - bc_vals)**2).mean()

    def evaluate(self, resolution=50):
        """
        Evaluate L2 error on full 3D domain against exact solution.

        Returns:
            mean_l2: average L2 error across time slices
            l2_errors: dict mapping t -> L2 error
        """
        time_slices = [0.0, 1.0, 2.0, 3.0, 4.0]
        l2_errors = {}

        with torch.no_grad():
            x = torch.linspace(self.x_min, self.x_max, resolution, device=device)
            y = torch.linspace(self.y_min, self.y_max, resolution, device=device)
            X, Y = torch.meshgrid(x, y, indexing='ij')

            for t_val in time_slices:
                T = torch.full_like(X, t_val)

                # Normalize
                t_norm, x_norm, y_norm = self.normalize_coords(T, X, Y)
                pts = torch.stack([t_norm.flatten(), x_norm.flatten(), y_norm.flatten()], dim=1)

                u_pred = self.forward(pts).reshape(resolution, resolution)
                u_exact = exact_solution(T, X, Y, self.v_max)

                l2 = torch.sqrt(((u_pred - u_exact)**2).sum()) / torch.sqrt((u_exact**2).sum() + 1e-8)
                l2_errors[t_val] = l2.item()

        mean_l2 = sum(l2_errors.values()) / len(l2_errors)
        return mean_l2, l2_errors


# =============================================================================
# EMA
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
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


# =============================================================================
# Training
# =============================================================================

def get_grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.norm()**2
    return total.sqrt()


def train(config):
    n_epochs = config.get('n_epochs', 30000)
    seed = config.get('seed', 456)
    lr = config.get('lr', 1e-3)
    num_collocation = config.get('num_collocation', 100000)
    eval_interval = config.get('eval_interval', 1000)
    save_plots = config.get('save_plots', True)

    # Adaptive LR config
    use_adaptive_lr = config.get('use_adaptive_lr', True)
    lr_patience = config.get('lr_patience', 2000)
    min_lr = config.get('min_lr', 1e-6)
    use_cosine_scheduler = config.get('use_cosine_scheduler', True)
    use_warm_restart = config.get('use_warm_restart', False)
    restart_period = config.get('restart_period', 5000)
    restart_mult = config.get('restart_mult', 2)

    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print("Flow Mixing 3D - CUDA MLP")
    print("=" * 70)
    print(f"PDE: du/dt + a*du/dx + b*du/dy = 0 (advection)")
    print(f"v_max = {config.get('v_max', V_MAX)}")
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Epochs: {n_epochs}, Seed: {seed}")
    print(f"Hidden: {config.get('hidden_dim', 256)}, Layers: {config.get('n_layers', 2)}")
    if use_adaptive_lr:
        print(f"Adaptive LR: ON (patience={lr_patience}, min_lr={min_lr:.0e})")
    if use_warm_restart:
        print(f"Scheduler: CosineAnnealingWarmRestarts (T_0={restart_period}, T_mult={restart_mult})")
    elif use_cosine_scheduler:
        print(f"Scheduler: CosineAnnealing (T_max={n_epochs})")
    else:
        print(f"Scheduler: StepLR (step=15000, gamma=0.5)")
    print("=" * 70)

    if not CUDA_AVAILABLE:
        print("\nERROR: CUDA extension not available!")
        print("Please run: python setup_mlp_cuda_v2.py install")
        return None

    model = HermiteNGP_PINN_FlowMixing(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if use_warm_restart:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=restart_period, T_mult=restart_mult, eta_min=min_lr
        )
    elif use_cosine_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=min_lr)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15000, gamma=0.5)

    # EMA
    ema = EMA(model, decay=0.999)

    # Warmup
    print("\nWarmup...")
    for _ in range(20):
        # Collocation points (random in domain)
        t_raw = torch.rand(num_collocation, device=device) * model.t_max
        x_raw = torch.rand(num_collocation, device=device) * (model.x_max - model.x_min) + model.x_min
        y_raw = torch.rand(num_collocation, device=device) * (model.y_max - model.y_min) + model.y_min
        t_norm, x_norm, y_norm = model.normalize_coords(t_raw, x_raw, y_raw)
        pts = torch.stack([t_norm, x_norm, y_norm], dim=1)

        ic_pts, ic_vals, _, _, _ = model.generate_ic_points()
        bc_pts, bc_vals = model.generate_bc_points()

        loss_pde = model.loss_pde(pts, t_raw, x_raw, y_raw)
        loss_ic = model.loss_ic(ic_pts, ic_vals)
        loss_bc = model.loss_bc(bc_pts, bc_vals)
        loss = model.pde_weight * loss_pde + model.ic_weight * loss_ic + model.bc_weight * loss_bc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.update(model)

    # Training
    best_l2 = float('inf')
    best_epoch = 0
    best_state = None
    history = []
    epochs_since_improvement = 0

    actual_eval_interval = min(eval_interval, max(100, n_epochs // 10))

    print("\nTraining...")
    t0 = time.perf_counter()

    for epoch in range(n_epochs):
        # Curriculum: get active levels
        active_levels = model.get_active_levels(epoch)
        frozen_levels = [l for l in range(model.n_levels) if l not in active_levels]
        model.freeze_levels(frozen_levels)

        # Sample collocation points
        t_raw = torch.rand(num_collocation, device=device) * model.t_max
        x_raw = torch.rand(num_collocation, device=device) * (model.x_max - model.x_min) + model.x_min
        y_raw = torch.rand(num_collocation, device=device) * (model.y_max - model.y_min) + model.y_min
        t_norm, x_norm, y_norm = model.normalize_coords(t_raw, x_raw, y_raw)
        pts = torch.stack([t_norm, x_norm, y_norm], dim=1)

        ic_pts, ic_vals, _, _, _ = model.generate_ic_points()
        bc_pts, bc_vals = model.generate_bc_points()

        # GradNorm: adaptive loss balancing (every 100 epochs)
        if (epoch + 1) % 100 == 0:
            l_pde = model.loss_pde(pts, t_raw, x_raw, y_raw)
            l_ic = model.loss_ic(ic_pts, ic_vals)
            l_bc = model.loss_bc(bc_pts, bc_vals)

            # Compute gradient norm for PDE loss
            optimizer.zero_grad()
            l_pde.backward(retain_graph=True)
            model.apply_level_mask()
            grad_pde = get_grad_norm(model)

            # Compute gradient norm for IC+BC loss
            optimizer.zero_grad()
            (l_ic + l_bc).backward(retain_graph=True)
            model.apply_level_mask()
            grad_icbc = get_grad_norm(model)

            # Update weights based on gradient ratio
            if grad_icbc > 1e-8:
                ratio = (grad_pde / grad_icbc).item()
                model.ic_weight = 0.9 * model.ic_weight + 0.1 * ratio
                model.bc_weight = 0.9 * model.bc_weight + 0.1 * ratio
                model.ic_weight = max(1.0, min(model.ic_weight_cap, model.ic_weight))
                model.bc_weight = max(1.0, min(model.bc_weight_cap, model.bc_weight))

            loss = model.pde_weight * l_pde + model.ic_weight * l_ic + model.bc_weight * l_bc
        else:
            loss_pde = model.loss_pde(pts, t_raw, x_raw, y_raw)
            loss_ic = model.loss_ic(ic_pts, ic_vals)
            loss_bc = model.loss_bc(bc_pts, bc_vals)
            loss = model.pde_weight * loss_pde + model.ic_weight * loss_ic + model.bc_weight * loss_bc

        optimizer.zero_grad()
        loss.backward()
        model.apply_level_mask()
        optimizer.step()
        scheduler.step()
        ema.update(model)

        # Evaluate
        if (epoch + 1) % actual_eval_interval == 0 or epoch == 0:
            ema.apply_shadow(model)
            mean_l2, l2_errors = model.evaluate()
            ema.restore(model)

            if mean_l2 < best_l2:
                best_l2 = mean_l2
                best_epoch = epoch + 1
                best_state = {k: v.clone() for k, v in ema.shadow.items()}
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += actual_eval_interval

            # Adaptive LR reduction
            lr_current = optimizer.param_groups[0]['lr']
            if use_adaptive_lr and epochs_since_improvement >= lr_patience and lr_current > min_lr:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = max(param_group['lr'] * 0.5, min_lr)
                lr_current = optimizer.param_groups[0]['lr']
                print(f"  >> LR reduced to {lr_current:.2e}")
                epochs_since_improvement = 0

            elapsed = time.perf_counter() - t0
            history.append((epoch + 1, mean_l2, best_l2))

            print(f"  Epoch {epoch+1:6d}: L2={mean_l2:.4e}, best={best_l2:.4e} @{best_epoch}, "
                  f"ic_w={model.ic_weight:.1f}, bc_w={model.bc_weight:.1f}, lr={lr_current:.2e}, time={elapsed:.0f}s")

    # Restore best model
    if best_state is not None:
        for name, param in model.named_parameters():
            if name in best_state:
                param.data = best_state[name]

    # Final evaluation
    mean_l2_final, l2_errors_final = model.evaluate(resolution=100)
    elapsed_total = time.perf_counter() - t0
    ms_per_epoch = elapsed_total / n_epochs * 1000

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Best L2 Error: {best_l2:.4e} @ epoch {best_epoch}")
    print(f"Final L2 Error: {mean_l2_final:.4e}")
    print("L2 errors by time slice:")
    for t_val in sorted(l2_errors_final.keys()):
        print(f"  t={t_val:.1f}: {l2_errors_final[t_val]:.4e}")
    print(f"Total Time: {elapsed_total:.1f}s ({ms_per_epoch:.2f} ms/epoch)")

    # Prepare results dict
    results = {
        'best_l2': best_l2,
        'best_epoch': best_epoch,
        'final_l2': mean_l2_final,
        'l2_errors_by_time': l2_errors_final,
        'total_time': elapsed_total,
        'ms_per_epoch': ms_per_epoch,
        'n_params': n_params,
        'history': history,
    }

    # Save visualization
    if save_plots:
        save_visualization(model, config, results)

    # Save results to NPZ
    if config.get('save_npz', True):
        save_results_npz(model, config, results, config.get('output_path'))

    return results


def save_results_npz(model, config, results, output_path=None):
    """Save training results to NPZ file."""
    if output_path is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(output_dir, 'flow_mixing3d_results.npz')

    # Generate prediction grids at different time slices
    resolution = 100
    time_slices = [0.0, 1.0, 2.0, 3.0, 4.0]

    u_preds = {}
    u_exacts = {}
    with torch.no_grad():
        x = torch.linspace(model.x_min, model.x_max, resolution, device=device)
        y = torch.linspace(model.y_min, model.y_max, resolution, device=device)
        X, Y = torch.meshgrid(x, y, indexing='ij')

        for t_val in time_slices:
            T = torch.full_like(X, t_val)
            t_norm, x_norm, y_norm = model.normalize_coords(T, X, Y)
            pts = torch.stack([t_norm.flatten(), x_norm.flatten(), y_norm.flatten()], dim=1)

            u_pred = model.forward(pts).reshape(resolution, resolution)
            u_exact = exact_solution(T, X, Y, model.v_max)

            u_preds[f'u_pred_t{t_val:.1f}'] = u_pred.cpu().numpy()
            u_exacts[f'u_exact_t{t_val:.1f}'] = u_exact.cpu().numpy()

    # Prepare data dict
    save_dict = {
        # Config
        'config_n_epochs': config.get('n_epochs', 0),
        'config_seed': config.get('seed', 0),
        'config_lr': config.get('lr', 0),
        'config_v_max': config.get('v_max', V_MAX),
        'config_hidden_dim': config.get('hidden_dim', 0),
        'config_n_layers': config.get('n_layers', 0),

        # Results
        'best_l2': results.get('best_l2', 0),
        'best_epoch': results.get('best_epoch', 0),
        'final_l2': results.get('final_l2', 0),
        'total_time': results.get('total_time', 0),
        'ms_per_epoch': results.get('ms_per_epoch', 0),
        'n_params': results.get('n_params', 0),

        # L2 errors by time
        'time_slices': np.array(time_slices),

        # History
        'history_epochs': np.array([h[0] for h in results.get('history', [])]),
        'history_l2': np.array([h[1] for h in results.get('history', [])]),
        'history_best_l2': np.array([h[2] for h in results.get('history', [])]),

        # Coordinates
        'x': x.cpu().numpy(),
        'y': y.cpu().numpy(),
    }
    save_dict.update(u_preds)
    save_dict.update(u_exacts)

    # L2 by time
    for t_val, l2 in results.get('l2_errors_by_time', {}).items():
        save_dict[f'l2_t{t_val:.1f}'] = l2

    # Save
    np.savez_compressed(output_path, **save_dict)
    print(f"Results saved: {output_path}")

    # Save model state separately (for easier loading)
    params_path = output_path.replace('.npz', '_params.pth')
    torch.save(model.state_dict(), params_path)
    print(f"Model params saved: {params_path}")

    return output_path


def save_visualization(model, config, results):
    """Save visualization of results at different time slices."""
    resolution = 100
    time_slices = [0.0, 1.0, 2.0, 3.0, 4.0]

    with torch.no_grad():
        x = torch.linspace(model.x_min, model.x_max, resolution, device=device)
        y = torch.linspace(model.y_min, model.y_max, resolution, device=device)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        X_np, Y_np = X.cpu().numpy(), Y.cpu().numpy()

        fig, axes = plt.subplots(3, len(time_slices), figsize=(4*len(time_slices), 10))

        for i, t_val in enumerate(time_slices):
            T = torch.full_like(X, t_val)
            t_norm, x_norm, y_norm = model.normalize_coords(T, X, Y)
            pts = torch.stack([t_norm.flatten(), x_norm.flatten(), y_norm.flatten()], dim=1)

            u_pred = model.forward(pts).reshape(resolution, resolution).cpu().numpy()
            u_exact = exact_solution(T, X, Y, model.v_max).cpu().numpy()
            error = np.abs(u_pred - u_exact)

            vmax = 1.0

            # Row 0: Exact
            im0 = axes[0, i].contourf(X_np, Y_np, u_exact, levels=50, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            axes[0, i].set_title(f'Exact t={t_val:.1f}')
            axes[0, i].set_aspect('equal')
            plt.colorbar(im0, ax=axes[0, i], shrink=0.6)

            # Row 1: Prediction
            im1 = axes[1, i].contourf(X_np, Y_np, u_pred, levels=50, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            axes[1, i].set_title(f'Pred t={t_val:.1f}')
            axes[1, i].set_aspect('equal')
            plt.colorbar(im1, ax=axes[1, i], shrink=0.6)

            # Row 2: Error
            l2 = results['l2_errors_by_time'].get(t_val, 0)
            im2 = axes[2, i].contourf(X_np, Y_np, error, levels=50, cmap='hot')
            axes[2, i].set_title(f'|Error| L2={l2:.2e}')
            axes[2, i].set_aspect('equal')
            plt.colorbar(im2, ax=axes[2, i], shrink=0.6)

        axes[0, 0].set_ylabel('Exact')
        axes[1, 0].set_ylabel('Prediction')
        axes[2, 0].set_ylabel('Error')

    plt.suptitle(f'Flow Mixing 3D: du/dt + a*du/dx + b*du/dy = 0\n'
                 f'Best L2: {results["best_l2"]:.4e} @ epoch {results["best_epoch"]}',
                 fontsize=12)
    plt.tight_layout()

    # Save
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, 'flow_mixing3d_result.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nVisualization saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flow Mixing 3D - CUDA MLP')
    parser.add_argument('--epochs', type=int, default=200000, help='Number of epochs')
    parser.add_argument('--seed', type=int, default=456, help='Random seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--v-max', type=float, default=0.385, help='Maximum tangential velocity')
    parser.add_argument('--omega', type=float, default=0.5, help='SIREN omega')
    parser.add_argument('--hidden', type=int, default=128, help='MLP hidden dim')
    parser.add_argument('--layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--no-plots', action='store_true', help='Disable plot saving')
    parser.add_argument('--no-save', action='store_true', help='Disable NPZ results saving')
    parser.add_argument('--output', type=str, default=None, help='Custom output path for NPZ file')
    parser.add_argument('--lr-patience', type=int, default=2000000, help='Epochs without improvement before LR reduction')
    parser.add_argument('--min-lr', type=float, default=1e-5, help='Minimum learning rate')
    parser.add_argument('--no-adaptive-lr', action='store_true', help='Disable adaptive LR reduction')
    parser.add_argument('--cosine-scheduler', action='store_true', default=True, help='Use cosine annealing scheduler (default: True)')
    parser.add_argument('--no-cosine-scheduler', action='store_true', help='Disable cosine annealing scheduler')
    parser.add_argument('--warm-restart', action='store_true', help='Use cosine annealing with warm restarts')
    parser.add_argument('--restart-period', type=int, default=5000, help='Initial restart period T_0 for warm restart')
    parser.add_argument('--restart-mult', type=int, default=2, help='Period multiplier T_mult for warm restart')
    args = parser.parse_args()

    config = {
        'n_epochs': args.epochs,
        'seed': args.seed,
        'lr': args.lr,
        'v_max': args.v_max,
        'omega': args.omega,
        'hidden_dim': args.hidden,
        'n_layers': args.layers,
        'save_plots': not args.no_plots,
        'save_npz': not args.no_save,
        'output_path': args.output,
        'lr_patience': args.lr_patience,
        'min_lr': args.min_lr,
        'use_adaptive_lr': not args.no_adaptive_lr,
        'use_cosine_scheduler': args.cosine_scheduler and not args.no_cosine_scheduler,
        'use_warm_restart': args.warm_restart,
        'restart_period': args.restart_period,
        'restart_mult': args.restart_mult,

        # Fixed config
        'n_levels': 8,
        'log2_hashmap_size': 16,
        'pde_weight': 10.0,
        'ic_weight': 1.0,
        'bc_weight': 1.0,
        'ic_weight_cap': 1000.0,
        'bc_weight_cap': 1000.0,
        'num_collocation': 50000,
        'num_bc_per_edge': 3000,
        'num_ic': 10000,
        'eval_interval': 1000,
    }

    train(config)
