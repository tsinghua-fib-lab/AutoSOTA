"""
Navier-Stokes 2D + time (Taylor-Green vortex) PINN with multi-layer SIREN
MLP and Hermite-NGP encoding.

PDE system:
    u_t + u*u_x + v*u_y = -p_x + nu*(u_xx + u_yy)
    v_t + u*v_x + v*v_y = -p_y + nu*(v_xx + v_yy)
    u_x + v_y = 0   (incompressibility)

Domain: t in [0, 1], x, y in [0, 2*pi].

Exact solution:
    u = -cos(x)*sin(y)*exp(-2*nu*t)
    v =  sin(x)*cos(y)*exp(-2*nu*t)
    p = -0.25*(cos(2x) + cos(2y))*exp(-4*nu*t)

Usage:
    python examples/taylor_green.py
    python examples/taylor_green.py --epochs 100000 --nu 0.01
    python examples/taylor_green.py --warm-restart --restart-period 10000
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Import CUDA extension
try:
    import hermite_mlp_cuda_3d_v2
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("WARNING: hermite_mlp_cuda_3d_v2 not available. Run: python setup_mlp_cuda_v2.py install")

# Problem parameters
PI = np.pi
TWO_PI = 2 * PI


# =============================================================================
# Exact Solution: Taylor-Green Vortex
# =============================================================================

def exact_solution(t, x, y, nu):
    """Taylor-Green vortex exact solution."""
    decay = torch.exp(-2 * nu * t)
    u = -torch.cos(x) * torch.sin(y) * decay
    v = torch.sin(x) * torch.cos(y) * decay
    p = -0.25 * (torch.cos(2*x) + torch.cos(2*y)) * torch.exp(-4 * nu * t)
    return u, v, p


# =============================================================================
# CUDA V3 MLP: CUDA Forward + PyTorch Backward (3D: t, x, y)
# =============================================================================

class HermiteLayerFunction3D_V3(torch.autograd.Function):
    """CUDA forward + PyTorch backward for Hermite propagation (3D: t, x, y)."""

    @staticmethod
    def forward(ctx, h, dh_dt, dh_dx, dh_dy, d2h_dtt, d2h_dxx, d2h_dyy, weight, bias, omega, apply_activation):
        outputs = hermite_mlp_cuda_3d_v2.forward(
            h.contiguous(), dh_dt.contiguous(), dh_dx.contiguous(), dh_dy.contiguous(),
            d2h_dtt.contiguous(), d2h_dxx.contiguous(), d2h_dyy.contiguous(),
            weight.contiguous(), bias.contiguous(),
            omega, apply_activation
        )
        out_h, out_dt, out_dx, out_dy, out_dtt, out_dxx, out_dyy = outputs[:7]
        save_z, save_dz_dt, save_dz_dx, save_dz_dy, save_d2z_dtt, save_d2z_dxx, save_d2z_dyy = outputs[7:]

        ctx.save_for_backward(
            h, dh_dt, dh_dx, dh_dy, d2h_dtt, d2h_dxx, d2h_dyy,
            weight,
            save_z, save_dz_dt, save_dz_dx, save_dz_dy
        )
        ctx.omega = omega
        ctx.apply_activation = apply_activation

        return out_h, out_dt, out_dx, out_dy, out_dtt, out_dxx, out_dyy

    @staticmethod
    def backward(ctx, grad_h, grad_dt, grad_dx, grad_dy, grad_dtt, grad_dxx, grad_dyy):
        h, dh_dt, dh_dx, dh_dy, d2h_dtt, d2h_dxx, d2h_dyy, weight, z, dz_dt, dz_dx, dz_dy = ctx.saved_tensors
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
            grad_z = grad_z + grad_dt * h_pp * dz_dt
            grad_z = grad_z + grad_dx * h_pp * dz_dx
            grad_z = grad_z + grad_dy * h_pp * dz_dy

            d2z_dtt = d2h_dtt @ weight.T
            grad_z = grad_z + grad_dtt * (h_ppp * dz_dt * dz_dt + h_pp * d2z_dtt)

            d2z_dxx = d2h_dxx @ weight.T
            grad_z = grad_z + grad_dxx * (h_ppp * dz_dx * dz_dx + h_pp * d2z_dxx)

            d2z_dyy = d2h_dyy @ weight.T
            grad_z = grad_z + grad_dyy * (h_ppp * dz_dy * dz_dy + h_pp * d2z_dyy)

            grad_dz_dt = grad_dt * h_p + grad_dtt * 2 * h_pp * dz_dt
            grad_dz_dx = grad_dx * h_p + grad_dxx * 2 * h_pp * dz_dx
            grad_dz_dy = grad_dy * h_p + grad_dyy * 2 * h_pp * dz_dy

            grad_d2z_dtt = grad_dtt * h_p
            grad_d2z_dxx = grad_dxx * h_p
            grad_d2z_dyy = grad_dyy * h_p
        else:
            grad_z = grad_h
            grad_dz_dt = grad_dt
            grad_dz_dx = grad_dx
            grad_dz_dy = grad_dy
            grad_d2z_dtt = grad_dtt
            grad_d2z_dxx = grad_dxx
            grad_d2z_dyy = grad_dyy

        grad_h_in = grad_z @ weight
        grad_dh_dt_in = grad_dz_dt @ weight
        grad_dh_dx_in = grad_dz_dx @ weight
        grad_dh_dy_in = grad_dz_dy @ weight
        grad_d2h_dtt_in = grad_d2z_dtt @ weight
        grad_d2h_dxx_in = grad_d2z_dxx @ weight
        grad_d2h_dyy_in = grad_d2z_dyy @ weight

        grad_weight = grad_z.T @ h
        grad_weight = grad_weight + grad_dz_dt.T @ dh_dt
        grad_weight = grad_weight + grad_dz_dx.T @ dh_dx
        grad_weight = grad_weight + grad_dz_dy.T @ dh_dy
        grad_weight = grad_weight + grad_d2z_dtt.T @ d2h_dtt
        grad_weight = grad_weight + grad_d2z_dxx.T @ d2h_dxx
        grad_weight = grad_weight + grad_d2z_dyy.T @ d2h_dyy

        grad_bias = grad_z.sum(dim=0)

        return (grad_h_in, grad_dh_dt_in, grad_dh_dx_in, grad_dh_dy_in,
                grad_d2h_dtt_in, grad_d2h_dxx_in, grad_d2h_dyy_in,
                grad_weight, grad_bias, None, None)


# =============================================================================
# SIREN MLP with 3 Outputs (u, v, p) and CUDA Hermite Propagation
# =============================================================================

class SIREN_NS(nn.Module):
    """SIREN MLP with 3 outputs (u, v, p) using CUDA Hermite propagation."""

    def __init__(self, input_dim, hidden_dim=256, n_layers=2, omega_0=0.5):
        super().__init__()
        self.omega_0 = omega_0
        self.n_layers = n_layers

        # Build layers
        self.layers = nn.ModuleList()
        dims = [input_dim] + [hidden_dim] * n_layers
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
        self.output_layer = nn.Linear(hidden_dim, 3)  # 3 outputs: u, v, p
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

    def forward_with_derivatives_cuda(self, enc, dt, dx, dy, dtt, dxx, dyy):
        """
        Forward with CUDA Hermite propagation for NS equations.

        Returns: out [N, 3], derivatives for each output
            out: [u, v, p]
            d_dt: [du/dt, dv/dt, dp/dt]
            d_dx: [du/dx, dv/dx, dp/dx]
            d_dy: [du/dy, dv/dy, dp/dy]
            d2_dxx: [d2u/dx2, d2v/dx2, d2p/dx2]
            d2_dyy: [d2u/dy2, d2v/dy2, d2p/dy2]
        """
        omega = self.omega_0
        h, dh_dt, dh_dx, dh_dy, d2h_dtt, d2h_dxx, d2h_dyy = enc, dt, dx, dy, dtt, dxx, dyy

        # Hidden layers with activation
        for layer in self.layers:
            h, dh_dt, dh_dx, dh_dy, d2h_dtt, d2h_dxx, d2h_dyy = HermiteLayerFunction3D_V3.apply(
                h, dh_dt, dh_dx, dh_dy, d2h_dtt, d2h_dxx, d2h_dyy,
                layer.weight, layer.bias, omega, True
            )

        # Output layer (no activation)
        out, d_dt, d_dx, d_dy, d2_dtt, d2_dxx, d2_dyy = HermiteLayerFunction3D_V3.apply(
            h, dh_dt, dh_dx, dh_dy, d2h_dtt, d2h_dxx, d2h_dyy,
            self.output_layer.weight, self.output_layer.bias, omega, False
        )

        return out, d_dt, d_dx, d_dy, d2_dxx, d2_dyy


# =============================================================================
# Hermite-NGP PINN Model for NS 2D+1
# =============================================================================

class HermiteNGP_NS2D(nn.Module):
    """
    PINN for 2D+1 Navier-Stokes using Hermite Hash Encoding.

    Input: [t, x, y] in normalized coords [0,1]^3
    Output: [u, v, p]
    """

    def __init__(self, config=None):
        super().__init__()

        config = config or {}
        self.n_levels = config.get('n_levels', 8)
        self.log2_hashmap_size_1 = config.get('log2_hashmap_size_1', 16)
        self.log2_hashmap_size_2 = config.get('log2_hashmap_size_2', 16)
        self.log2_hashmap_size_3 = config.get('log2_hashmap_size_3', 16)
        self.log2_hashmap_size_4 = config.get('log2_hashmap_size_4', 16)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.omega = config.get('omega', 0.5)
        self.bc_weight_cap = config.get('bc_weight_cap', 10000.0)
        self.nu = config.get('nu', 0.01)  # Viscosity

        # Domain scaling: [0,1] -> physical
        # t: [0, 1] -> [0, T_MAX]
        # x, y: [0, 1] -> [0, 2*pi]
        self.T_MAX = config.get('T_MAX', 1.0)
        self.SCALE_T = self.T_MAX
        self.SCALE_X = TWO_PI
        self.SCALE_Y = TWO_PI

        # Hermite Hash Encoding (CUDA) - 3D for (t, x, y)
        from hermite_ngp.encoding.hermite_encoding_cuda import HermiteHashEncodingCUDA_3D
        self.encoding = HermiteHashEncodingCUDA_3D(
            n_input_dims=3,
            n_levels=self.n_levels,
            n_features_per_level=2,
            log2_hashmap_size_1=self.log2_hashmap_size_1,
            log2_hashmap_size_2=self.log2_hashmap_size_2,
            log2_hashmap_size_3=self.log2_hashmap_size_3,
            log2_hashmap_size_4=self.log2_hashmap_size_4,
            base_resolution=4,
            per_level_scale=1.5,
        )

        # SIREN MLP with 3 outputs (multi-layer)
        encoding_dim = self.n_levels * 2
        self.n_mlp_layers = config.get('n_layers', 1)
        self.mlp = SIREN_NS(encoding_dim, self.hidden_dim, n_layers=self.n_mlp_layers, omega_0=self.omega)

        # Training state
        self.bc_weight = 100.0
        self.ic_weight = 100.0
        self.register_buffer('level_grad_mask', torch.ones(self.n_levels))

        # Curriculum phases (coarse-to-fine)
        self.phases = config.get('curriculum_phases', [
            (0, float('inf'), list(range(self.n_levels))),  # All levels by default
        ])

        self.to(device)

        # Pre-compute IC points (t=0)
        n_ic = config.get('num_ic', 5000)
        x_ic = torch.rand(n_ic, device=device)
        y_ic = torch.rand(n_ic, device=device)
        t_ic = torch.zeros(n_ic, device=device)
        self.fixed_ic_pts = torch.stack([t_ic, x_ic, y_ic], dim=-1)

        # IC values (physical coords)
        x_phys = x_ic * self.SCALE_X
        y_phys = y_ic * self.SCALE_Y
        t_phys = t_ic * self.SCALE_T
        u_ic, v_ic, p_ic = exact_solution(t_phys, x_phys, y_phys, self.nu)
        self.fixed_ic_u = u_ic
        self.fixed_ic_v = v_ic
        self.fixed_ic_p = p_ic

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

    def forward(self, x):
        """Forward pass returning [N, 3] for (u, v, p)."""
        enc = self.encoding(x)
        return self.mlp(enc)

    def forward_with_derivatives(self, x):
        """
        Forward with analytic derivatives using CUDA Hermite propagation.

        x: [N, 3] normalized coords [t_norm, x_norm, y_norm]

        Returns all necessary derivatives for NS equations.
        """
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA extension hermite_mlp_cuda_3d_v2 required. Run: python setup_mlp_cuda_v2.py install")

        # Get encoding with first and second derivatives
        enc, dt, dx, dy, dtt, dxx, dyy = self.encoding.forward_with_second_derivatives_cuda(x)

        # Propagate through MLP using CUDA Hermite
        # out: [N, 3] = [u, v, p]
        # d_dt, d_dx, d_dy: [N, 3] = derivatives of output w.r.t. t, x, y (in encoding space)
        # d2_dxx, d2_dyy: [N, 3] = second derivatives
        out, d_dt, d_dx, d_dy, d2_dxx, d2_dyy = self.mlp.forward_with_derivatives_cuda(
            enc, dt, dx, dy, dtt, dxx, dyy
        )

        # Extract per-variable derivatives
        # u derivatives (column 0)
        u_t = d_dt[:, 0:1] / self.SCALE_T
        u_x = d_dx[:, 0:1] / self.SCALE_X
        u_y = d_dy[:, 0:1] / self.SCALE_Y
        u_xx = d2_dxx[:, 0:1] / (self.SCALE_X ** 2)
        u_yy = d2_dyy[:, 0:1] / (self.SCALE_Y ** 2)

        # v derivatives (column 1)
        v_t = d_dt[:, 1:2] / self.SCALE_T
        v_x = d_dx[:, 1:2] / self.SCALE_X
        v_y = d_dy[:, 1:2] / self.SCALE_Y
        v_xx = d2_dxx[:, 1:2] / (self.SCALE_X ** 2)
        v_yy = d2_dyy[:, 1:2] / (self.SCALE_Y ** 2)

        # p derivatives (column 2)
        p_x = d_dx[:, 2:3] / self.SCALE_X
        p_y = d_dy[:, 2:3] / self.SCALE_Y

        return out, u_t, v_t, u_x, u_y, v_x, v_y, u_xx, u_yy, v_xx, v_yy, p_x, p_y

    def loss_pde(self, pts_norm):
        """
        PDE residual loss for NS equations.

        pts_norm: [N, 3] normalized coords
        """
        out, u_t, v_t, u_x, u_y, v_x, v_y, u_xx, u_yy, v_xx, v_yy, p_x, p_y = \
            self.forward_with_derivatives(pts_norm)

        u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]

        # Momentum equations:
        # u_t + u*u_x + v*u_y = -p_x + nu*(u_xx + u_yy)
        res_u = u_t + u * u_x + v * u_y + p_x - self.nu * (u_xx + u_yy)

        # v_t + u*v_x + v*v_y = -p_y + nu*(v_xx + v_yy)
        res_v = v_t + u * v_x + v * v_y + p_y - self.nu * (v_xx + v_yy)

        # Continuity: u_x + v_y = 0
        res_cont = u_x + v_y

        return (res_u**2).mean() + (res_v**2).mean() + 10.0 * (res_cont**2).mean()

    def loss_ic(self):
        """Initial condition loss at t=0."""
        out = self.forward(self.fixed_ic_pts)
        u_pred, v_pred, p_pred = out[:, 0], out[:, 1], out[:, 2]

        loss_u = ((u_pred - self.fixed_ic_u)**2).mean()
        loss_v = ((v_pred - self.fixed_ic_v)**2).mean()
        loss_p = ((p_pred - self.fixed_ic_p)**2).mean()

        return loss_u + loss_v + loss_p

    def loss_bc_periodic(self, n_points=1000):
        """
        Periodic BC loss: u(t,0,y) = u(t,2pi,y), etc.

        For Taylor-Green vortex, the solution is naturally periodic.
        """
        t = torch.rand(n_points, device=device)
        s = torch.rand(n_points, device=device)

        # x = 0 vs x = 1 (physical: 0 vs 2*pi)
        pts_x0 = torch.stack([t, torch.zeros_like(t), s], dim=-1)
        pts_x1 = torch.stack([t, torch.ones_like(t), s], dim=-1)
        out_x0 = self.forward(pts_x0)
        out_x1 = self.forward(pts_x1)
        loss_x = ((out_x0 - out_x1)**2).mean()

        # y = 0 vs y = 1 (physical: 0 vs 2*pi)
        pts_y0 = torch.stack([t, s, torch.zeros_like(t)], dim=-1)
        pts_y1 = torch.stack([t, s, torch.ones_like(t)], dim=-1)
        out_y0 = self.forward(pts_y0)
        out_y1 = self.forward(pts_y1)
        loss_y = ((out_y0 - out_y1)**2).mean()

        return loss_x + loss_y

    def apply_level_mask(self):
        """Apply gradient mask to hash tables."""
        mask = self.level_grad_mask.view(-1, 1, 1)
        for ht in [self.encoding.hash_table_1, self.encoding.hash_table_2,
                   self.encoding.hash_table_3, self.encoding.hash_table_4]:
            if ht.grad is not None:
                ht.grad *= mask

    def evaluate(self, t_eval=0.5, resolution=100):
        """
        Evaluate at a specific time slice.

        Returns L2 errors for u, v, p and predictions.
        """
        with torch.no_grad():
            g = torch.linspace(0, 1, resolution, device=device)
            X, Y = torch.meshgrid(g, g, indexing='ij')
            T = torch.full_like(X, t_eval / self.T_MAX)  # Normalize time

            pts_norm = torch.stack([T.flatten(), X.flatten(), Y.flatten()], dim=1)
            out = self.forward(pts_norm)

            u_pred = out[:, 0].reshape(resolution, resolution)
            v_pred = out[:, 1].reshape(resolution, resolution)
            p_pred = out[:, 2].reshape(resolution, resolution)

            # Exact solution in physical coords
            x_phys = X * self.SCALE_X
            y_phys = Y * self.SCALE_Y
            t_phys = torch.full_like(X, t_eval)
            u_exact, v_exact, p_exact = exact_solution(t_phys, x_phys, y_phys, self.nu)

            # L2 errors
            l2_u = torch.sqrt(((u_pred - u_exact)**2).mean()).item()
            l2_v = torch.sqrt(((v_pred - v_exact)**2).mean()).item()
            l2_p = torch.sqrt(((p_pred - p_exact)**2).mean()).item()

            # Combined velocity error
            U_pred = torch.sqrt(u_pred**2 + v_pred**2)
            U_exact = torch.sqrt(u_exact**2 + v_exact**2)
            l2_U = torch.sqrt(((U_pred - U_exact)**2).mean()).item()

            return {
                'l2_u': l2_u,
                'l2_v': l2_v,
                'l2_p': l2_p,
                'l2_U': l2_U,
                'u_pred': u_pred.cpu().numpy(),
                'v_pred': v_pred.cpu().numpy(),
                'p_pred': p_pred.cpu().numpy(),
                'u_exact': u_exact.cpu().numpy(),
                'v_exact': v_exact.cpu().numpy(),
                'p_exact': p_exact.cpu().numpy(),
                'X': (X * self.SCALE_X).cpu().numpy(),
                'Y': (Y * self.SCALE_Y).cpu().numpy(),
            }


# =============================================================================
# EMA
# =============================================================================

class EMA:
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
    seed = config.get('seed', 42)
    lr = config.get('lr', 1e-3)
    num_collocation = config.get('num_collocation', 20000)
    eval_interval = config.get('eval_interval', 1000)
    save_plots = config.get('save_plots', True)
    nu = config.get('nu', 0.01)

    # Adaptive LR config
    use_adaptive_lr = config.get('use_adaptive_lr', True)
    patience = config.get('lr_patience', 1000)
    min_lr = config.get('min_lr', 1e-7)

    # Cosine scheduler config
    use_cosine_scheduler = config.get('use_cosine_scheduler', True)
    use_warm_restart = config.get('use_warm_restart', False)
    restart_period = config.get('restart_period', 10000)
    restart_mult = config.get('restart_mult', 2)

    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print("Navier-Stokes 2D+1 (Taylor-Green Vortex) - V2")
    print("=" * 70)
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Epochs: {n_epochs}, Seed: {seed}")
    print(f"Viscosity (nu): {nu}")
    print(f"Domain: t in [0, {config.get('T_MAX', 1.0)}], x,y in [0, 2*pi]")
    print(f"Hidden: {config.get('hidden_dim', 256)}, MLP Layers: {config.get('n_layers', 1)}, Hash Levels: {config.get('n_levels', 8)}")
    print(f"Collocation: {num_collocation}, IC: {config.get('num_ic', 5000)}")
    if use_warm_restart:
        print(f"Scheduler: CosineAnnealingWarmRestarts (T_0={restart_period}, T_mult={restart_mult})")
    elif use_cosine_scheduler:
        print(f"Scheduler: CosineAnnealingLR (T_max={n_epochs}, eta_min={min_lr:.0e})")
    else:
        print(f"Scheduler: StepLR (step=10000, gamma=0.5)")
    if use_adaptive_lr:
        print(f"Adaptive LR: ON (patience={patience}, min_lr={min_lr:.0e})")
    print("=" * 70)

    # Create model
    model = HermiteNGP_NS2D(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Create scheduler based on config
    if use_warm_restart:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=restart_period, T_mult=restart_mult, eta_min=min_lr
        )
    elif use_cosine_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=min_lr
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.5)

    # EMA
    ema = EMA(model, decay=0.999)

    epochs_since_improvement = 0

    # Warmup
    print("\nWarmup...")
    for _ in range(20):
        pts = torch.rand(num_collocation, 3, device=device)
        loss = model.loss_pde(pts) + model.ic_weight * model.loss_ic() + model.bc_weight * model.loss_bc_periodic()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.update(model)

    # Training loop
    best_l2 = float('inf')
    best_epoch = 0
    best_state = None
    history = []

    print("\nTraining...")
    t0 = time.perf_counter()

    for epoch in range(n_epochs):
        # Curriculum: get active levels
        active_levels = model.get_active_levels(epoch)
        frozen_levels = [l for l in range(model.n_levels) if l not in active_levels]
        model.freeze_levels(frozen_levels)

        # Sample collocation points
        pts = torch.rand(num_collocation, 3, device=device)

        # GradNorm (every 100 epochs)
        if (epoch + 1) % 100 == 0:
            l_pde = model.loss_pde(pts)
            l_ic = model.loss_ic()
            l_bc = model.loss_bc_periodic()

            optimizer.zero_grad()
            l_pde.backward(retain_graph=True)
            model.apply_level_mask()
            grad_pde = get_grad_norm(model)

            optimizer.zero_grad()
            (l_ic + l_bc).backward(retain_graph=True)
            model.apply_level_mask()
            grad_bc = get_grad_norm(model)

            if grad_bc > 1e-8:
                ratio = (grad_pde / grad_bc).item()
                model.ic_weight = 0.9 * model.ic_weight + 0.1 * ratio
                model.bc_weight = 0.9 * model.bc_weight + 0.1 * ratio
                model.ic_weight = max(1.0, min(model.bc_weight_cap, model.ic_weight))
                model.bc_weight = max(1.0, min(model.bc_weight_cap, model.bc_weight))

            loss = l_pde + model.ic_weight * l_ic + model.bc_weight * l_bc
        else:
            loss = model.loss_pde(pts) + model.ic_weight * model.loss_ic() + model.bc_weight * model.loss_bc_periodic()

        optimizer.zero_grad()
        loss.backward()
        model.apply_level_mask()
        optimizer.step()
        scheduler.step()
        ema.update(model)

        # Evaluation
        if (epoch + 1) % eval_interval == 0:
            ema.apply_shadow(model)
            results = model.evaluate(t_eval=0.5)
            ema.restore(model)

            l2 = results['l2_U']
            if l2 < best_l2:
                best_l2 = l2
                best_epoch = epoch + 1
                best_state = {k: v.clone() for k, v in ema.shadow.items()}
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += eval_interval

            # Adaptive LR reduction
            lr_current = optimizer.param_groups[0]['lr']
            if use_adaptive_lr and epochs_since_improvement >= patience and lr_current > min_lr:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = max(param_group['lr'] * 0.5, min_lr)
                lr_current = optimizer.param_groups[0]['lr']
                print(f"  >> LR reduced to {lr_current:.2e} (no improvement for {patience} epochs)")
                epochs_since_improvement = 0

            elapsed = time.perf_counter() - t0
            history.append({
                'epoch': epoch + 1,
                'l2_U': l2,
                'l2_u': results['l2_u'],
                'l2_v': results['l2_v'],
                'l2_p': results['l2_p'],
                'best_l2': best_l2,
                'lr': lr_current,
                'ic_weight': model.ic_weight,
                'bc_weight': model.bc_weight,
            })

            n_active = len(active_levels)
            print(f"  Epoch {epoch+1:6d}: L2_U={l2:.4e} (u={results['l2_u']:.4e}, v={results['l2_v']:.4e}, p={results['l2_p']:.4e}), "
                  f"best={best_l2:.4e} @{best_epoch}, ic_w={model.ic_weight:.1f}, lvls={n_active}, lr={lr_current:.2e}, time={elapsed:.0f}s")

    # Restore best
    if best_state is not None:
        for name, param in model.named_parameters():
            if name in best_state:
                param.data = best_state[name]

    # Final evaluation
    results = model.evaluate(t_eval=0.5, resolution=200)
    elapsed_total = time.perf_counter() - t0
    ms_per_epoch = elapsed_total / n_epochs * 1000

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Best L2 Velocity Error: {best_l2:.4e} @ epoch {best_epoch}")
    print(f"  L2(u): {results['l2_u']:.4e}")
    print(f"  L2(v): {results['l2_v']:.4e}")
    print(f"  L2(p): {results['l2_p']:.4e}")
    print(f"Total Time: {elapsed_total:.1f}s ({ms_per_epoch:.2f} ms/epoch)")

    # Collect final results
    final_results = {
        'best_l2': best_l2,
        'best_epoch': best_epoch,
        'l2_u': results['l2_u'],
        'l2_v': results['l2_v'],
        'l2_p': results['l2_p'],
        'total_time': elapsed_total,
        'ms_per_epoch': ms_per_epoch,
        'n_params': n_params,
        'history': history,
        'u_pred': results['u_pred'],
        'v_pred': results['v_pred'],
        'p_pred': results['p_pred'],
        'u_exact': results['u_exact'],
        'v_exact': results['v_exact'],
        'p_exact': results['p_exact'],
        'X': results['X'],
        'Y': results['Y'],
    }

    if save_plots:
        save_visualization(results, best_l2, best_epoch, config)

    # Save NPZ results
    if config.get('save_npz', True):
        save_results_npz(model, config, final_results, history)

    # Save final fields
    if config.get('save_fields', True):
        save_final_fields(model, config)

    return best_l2, best_epoch, history


def save_results_npz(model, config, results, history):
    """Save comprehensive training results to NPZ file."""
    output_dir = os.path.dirname(os.path.abspath(__file__))
    nu = config.get('nu', 0.01)
    output_path = os.path.join(output_dir, f'ns2d_time_v2_nu{nu}_results.npz')

    # Extract history arrays
    history_epochs = np.array([h['epoch'] for h in history])
    history_l2_U = np.array([h['l2_U'] for h in history])
    history_l2_u = np.array([h['l2_u'] for h in history])
    history_l2_v = np.array([h['l2_v'] for h in history])
    history_l2_p = np.array([h['l2_p'] for h in history])
    history_best_l2 = np.array([h['best_l2'] for h in history])
    history_lr = np.array([h['lr'] for h in history])

    save_dict = {
        # Config
        'config_n_epochs': config.get('n_epochs', 0),
        'config_seed': config.get('seed', 0),
        'config_lr': config.get('lr', 0),
        'config_nu': nu,
        'config_hidden_dim': config.get('hidden_dim', 0),
        'config_n_layers': config.get('n_layers', 1),
        'config_n_levels': config.get('n_levels', 0),
        'config_num_collocation': config.get('num_collocation', 0),
        'config_use_cosine_scheduler': config.get('use_cosine_scheduler', True),
        'config_use_warm_restart': config.get('use_warm_restart', False),
        # Results
        'best_l2': results['best_l2'],
        'best_epoch': results['best_epoch'],
        'final_l2_u': results['l2_u'],
        'final_l2_v': results['l2_v'],
        'final_l2_p': results['l2_p'],
        'total_time': results['total_time'],
        'ms_per_epoch': results['ms_per_epoch'],
        'n_params': results['n_params'],
        # History
        'history_epochs': history_epochs,
        'history_l2_U': history_l2_U,
        'history_l2_u': history_l2_u,
        'history_l2_v': history_l2_v,
        'history_l2_p': history_l2_p,
        'history_best_l2': history_best_l2,
        'history_lr': history_lr,
        # Predictions (at t=0.5)
        'u_pred': results['u_pred'],
        'v_pred': results['v_pred'],
        'p_pred': results['p_pred'],
        'u_exact': results['u_exact'],
        'v_exact': results['v_exact'],
        'p_exact': results['p_exact'],
        'X': results['X'],
        'Y': results['Y'],
    }

    np.savez_compressed(output_path, **save_dict)
    print(f"Results saved: {output_path}")

    # Also save model state
    params_path = os.path.join(output_dir, f'ns2d_time_v2_nu{nu}_params.pth')
    torch.save(model.state_dict(), params_path)
    print(f"Model params saved: {params_path}")


def save_final_fields(model, config):
    """Save full 3D velocity and pressure fields (t, x, y)."""
    output_dir = os.path.dirname(os.path.abspath(__file__))
    T_MAX = config.get('T_MAX', 1.0)
    nu = config.get('nu', 0.01)

    nt, nx, ny = 50, 100, 100

    with torch.no_grad():
        t_grid = torch.linspace(0, 1, nt, device=device)
        x_grid = torch.linspace(0, 1, nx, device=device)
        y_grid = torch.linspace(0, 1, ny, device=device)

        u_3d = np.zeros((nt, nx, ny))
        v_3d = np.zeros((nt, nx, ny))
        p_3d = np.zeros((nt, nx, ny))

        for i, t_val in enumerate(t_grid):
            X, Y = torch.meshgrid(x_grid, y_grid, indexing='ij')
            T = torch.full_like(X, t_val.item())
            pts = torch.stack([T.flatten(), X.flatten(), Y.flatten()], dim=1)
            out = model.forward(pts)
            u_3d[i] = out[:, 0].reshape(nx, ny).cpu().numpy()
            v_3d[i] = out[:, 1].reshape(nx, ny).cpu().numpy()
            p_3d[i] = out[:, 2].reshape(nx, ny).cpu().numpy()

    t_phys = np.linspace(0, T_MAX, nt)
    x_phys = np.linspace(0, TWO_PI, nx)
    y_phys = np.linspace(0, TWO_PI, ny)

    np.savez(os.path.join(output_dir, f'ns2d_time_v2_nu{nu}_fields.npz'),
             u=u_3d, v=v_3d, p=p_3d, t=t_phys, x=x_phys, y=y_phys, nu=nu)
    print(f"Final 3D fields saved: ns2d_time_v2_nu{nu}_fields.npz (shape: {u_3d.shape})")


def save_visualization(results, best_l2, best_epoch, config):
    """Save visualization of results."""
    X, Y = results['X'], results['Y']

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    # Row 1: u velocity
    im00 = axes[0, 0].contourf(X, Y, results['u_exact'], levels=50, cmap='RdBu_r')
    axes[0, 0].set_title('u (Exact)')
    plt.colorbar(im00, ax=axes[0, 0])

    im01 = axes[0, 1].contourf(X, Y, results['u_pred'], levels=50, cmap='RdBu_r')
    axes[0, 1].set_title('u (Prediction)')
    plt.colorbar(im01, ax=axes[0, 1])

    u_err = np.abs(results['u_pred'] - results['u_exact'])
    im02 = axes[0, 2].contourf(X, Y, u_err, levels=50, cmap='hot')
    axes[0, 2].set_title(f'|u Error| (L2={results["l2_u"]:.2e})')
    plt.colorbar(im02, ax=axes[0, 2])

    # Row 2: v velocity
    im10 = axes[1, 0].contourf(X, Y, results['v_exact'], levels=50, cmap='RdBu_r')
    axes[1, 0].set_title('v (Exact)')
    plt.colorbar(im10, ax=axes[1, 0])

    im11 = axes[1, 1].contourf(X, Y, results['v_pred'], levels=50, cmap='RdBu_r')
    axes[1, 1].set_title('v (Prediction)')
    plt.colorbar(im11, ax=axes[1, 1])

    v_err = np.abs(results['v_pred'] - results['v_exact'])
    im12 = axes[1, 2].contourf(X, Y, v_err, levels=50, cmap='hot')
    axes[1, 2].set_title(f'|v Error| (L2={results["l2_v"]:.2e})')
    plt.colorbar(im12, ax=axes[1, 2])

    # Row 3: pressure
    im20 = axes[2, 0].contourf(X, Y, results['p_exact'], levels=50, cmap='RdBu_r')
    axes[2, 0].set_title('p (Exact)')
    plt.colorbar(im20, ax=axes[2, 0])

    im21 = axes[2, 1].contourf(X, Y, results['p_pred'], levels=50, cmap='RdBu_r')
    axes[2, 1].set_title('p (Prediction)')
    plt.colorbar(im21, ax=axes[2, 1])

    p_err = np.abs(results['p_pred'] - results['p_exact'])
    im22 = axes[2, 2].contourf(X, Y, p_err, levels=50, cmap='hot')
    axes[2, 2].set_title(f'|p Error| (L2={results["l2_p"]:.2e})')
    plt.colorbar(im22, ax=axes[2, 2])

    for ax in axes.flat:
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    nu = config.get('nu', 0.01)
    plt.suptitle(f'NS 2D+1 Taylor-Green V2 (nu={nu}): L2_U={best_l2:.4e} @ epoch {best_epoch}, t=0.5', fontsize=12)
    plt.tight_layout()

    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, f'ns2d_time_v2_nu{nu}_result.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nVisualization saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Navier-Stokes 2D+1 (Taylor-Green Vortex) - V2')
    parser.add_argument('--epochs', type=int, default=100000, help='Number of epochs')
    parser.add_argument('--seed', type=int, default=456, help='Random seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--nu', type=float, default=0.01, help='Viscosity')
    parser.add_argument('--omega', type=float, default=0.5, help='SIREN omega')
    parser.add_argument('--hidden', type=int, default=128, help='MLP hidden dim')
    parser.add_argument('--layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--no-plots', action='store_true', help='Disable plot saving')
    parser.add_argument('--no-save-fields', action='store_true', help='Disable saving final fields')
    parser.add_argument('--no-save-npz', action='store_true', help='Disable NPZ saving')
    # Adaptive LR
    parser.add_argument('--lr-patience', type=int, default=20000000, help='Epochs without improvement before LR reduction')
    parser.add_argument('--min-lr', type=float, default=1e-7, help='Minimum learning rate')
    parser.add_argument('--no-adaptive-lr', action='store_true', help='Disable adaptive LR reduction')
    # Cosine scheduler options
    parser.add_argument('--cosine-scheduler', action='store_true', default=True, help='Use cosine annealing scheduler (default: True)')
    parser.add_argument('--no-cosine-scheduler', action='store_true', help='Disable cosine annealing scheduler (use StepLR)')
    parser.add_argument('--warm-restart', action='store_true', help='Use cosine annealing with warm restarts')
    parser.add_argument('--restart-period', type=int, default=5000, help='Initial restart period T_0 for warm restart')
    parser.add_argument('--restart-mult', type=int, default=2, help='Period multiplier T_mult for warm restart')
    args = parser.parse_args()

    config = {
        'n_epochs': args.epochs,
        'seed': args.seed,
        'lr': args.lr,
        'nu': args.nu,
        'omega': args.omega,
        'hidden_dim': args.hidden,
        'n_layers': args.layers,
        'save_plots': not args.no_plots,
        'save_fields': not args.no_save_fields,
        'save_npz': not args.no_save_npz,
        'lr_patience': args.lr_patience,
        'min_lr': args.min_lr,
        'use_adaptive_lr': not args.no_adaptive_lr,
        # Cosine scheduler config
        'use_cosine_scheduler': args.cosine_scheduler and not args.no_cosine_scheduler,
        'use_warm_restart': args.warm_restart,
        'restart_period': args.restart_period,
        'restart_mult': args.restart_mult,

        # Fixed config
        'T_MAX': 1.0,
        'n_levels': 6,
        'log2_hashmap_size_1': 15,
        'log2_hashmap_size_2': 15,
        'log2_hashmap_size_3': 15,
        'log2_hashmap_size_4': 15,
        'bc_weight_cap': 50000.0,
        'num_collocation': 50000,
        'num_ic': 10000,
        'eval_interval': 500,
    }

    train(config)
