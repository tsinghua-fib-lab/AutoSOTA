"""
Helmholtz 3D PINN (low-frequency variant) with multi-layer SIREN MLP and
Hermite-NGP encoding. Coarse-to-fine curriculum + adaptive LR reduction.

Usage:
    python examples/helmholtz3d_a3.py
    python examples/helmholtz3d_a3.py --a1 5 --a2 5 --a3 5 --epochs 100000
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
# Problem Setup: Helmholtz 3D
# =============================================================================
# PDE: Laplacian(u) + k^2 * u = f
# Domain: [0, 1]^3
# Exact: u = sin(a1*pi*x) * sin(a2*pi*y) * sin(a3*pi*z)
# Source: f = (k^2 - (a1^2 + a2^2 + a3^2)*pi^2) * sin(...)

K = 1.0
PI = np.pi

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Import CUDA extension
try:
    import hermite_mlp_cuda_3d_v2
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("WARNING: hermite_mlp_cuda_3d_v2 not available. Run: python setup_mlp_cuda_v2.py install")


def exact_solution(x, y, z, a1, a2, a3):
    return torch.sin(a1 * PI * x) * torch.sin(a2 * PI * y) * torch.sin(a3 * PI * z)


def source_term(x, y, z, a1, a2, a3):
    coeff = K**2 - (a1**2 + a2**2 + a3**2) * PI**2
    return coeff * torch.sin(a1 * PI * x) * torch.sin(a2 * PI * y) * torch.sin(a3 * PI * z)


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

    def forward_with_laplacian_cuda(self, enc, dx, dy, dz, dxx, dyy, dzz):
        """Forward with full Hermite propagation using CUDA V3 for 3D."""
        omega = self.omega_0
        h, dh_dx, dh_dy, dh_dz, d2h_dxx, d2h_dyy, d2h_dzz = enc, dx, dy, dz, dxx, dyy, dzz

        for layer in self.layers:
            h, dh_dx, dh_dy, dh_dz, d2h_dxx, d2h_dyy, d2h_dzz = HermiteLayerFunction3D_V3.apply(
                h, dh_dx, dh_dy, dh_dz, d2h_dxx, d2h_dyy, d2h_dzz,
                layer.weight, layer.bias, omega, True
            )

        u, du_dx, du_dy, du_dz, d2u_dxx, d2u_dyy, d2u_dzz = HermiteLayerFunction3D_V3.apply(
            h, dh_dx, dh_dy, dh_dz, d2h_dxx, d2h_dyy, d2h_dzz,
            self.output_layer.weight, self.output_layer.bias, omega, False
        )

        laplacian = d2u_dxx + d2u_dyy + d2u_dzz
        return u, laplacian


# =============================================================================
# Model
# =============================================================================

class HermiteNGP_PINN_CUDA_3D(nn.Module):
    """PINN with Hermite encoding + CUDA V3 MLP for 3D."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # PDE parameters
        self.a1 = config.get('a1', 3)
        self.a2 = config.get('a2', 3)
        self.a3 = config.get('a3', 3)

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
        hidden_dim = config.get('hidden_dim', 128)
        n_layers = config.get('n_layers', 4)
        omega = config.get('omega', 0.5)
        self.mlp = SIREN_CUDA_3D_V3(input_dim, hidden_dim, n_layers, omega).to(device)

        self.n_levels = n_levels
        self.bc_weight = config.get('bc_weight_init', 1.0)
        self.bc_weight_cap = config.get('bc_weight_cap', 10000.0)

        # BC sampling config
        self.num_bc_per_face = config.get('num_bc_per_face', 5000)

        # Level mask for curriculum
        self.register_buffer('level_grad_mask', torch.ones(n_levels, device=device))

        # Curriculum phases (coarse-to-fine)
        self.phases = [
            # (0, 10000, [0, 1, 2, 3]),              # First 10k: coarse levels only
            # (10000, 30000, [0, 1, 2, 3, 4, 5]),    # 10k-30k: add medium levels
            (0, float('inf'), list(range(n_levels))),  # 30k+: all levels
        ]

    def generate_bc_points(self, n_per_face=None):
        """Generate random boundary points on all 6 faces of [0,1]^3."""
        if n_per_face is None:
            n_per_face = self.num_bc_per_face

        pts_list = []
        # x = 0 face
        t1, t2 = torch.rand(n_per_face, device=device), torch.rand(n_per_face, device=device)
        pts_list.append(torch.stack([torch.zeros(n_per_face, device=device), t1, t2], dim=1))
        # x = 1 face
        t1, t2 = torch.rand(n_per_face, device=device), torch.rand(n_per_face, device=device)
        pts_list.append(torch.stack([torch.ones(n_per_face, device=device), t1, t2], dim=1))
        # y = 0 face
        t1, t2 = torch.rand(n_per_face, device=device), torch.rand(n_per_face, device=device)
        pts_list.append(torch.stack([t1, torch.zeros(n_per_face, device=device), t2], dim=1))
        # y = 1 face
        t1, t2 = torch.rand(n_per_face, device=device), torch.rand(n_per_face, device=device)
        pts_list.append(torch.stack([t1, torch.ones(n_per_face, device=device), t2], dim=1))
        # z = 0 face
        t1, t2 = torch.rand(n_per_face, device=device), torch.rand(n_per_face, device=device)
        pts_list.append(torch.stack([t1, t2, torch.zeros(n_per_face, device=device)], dim=1))
        # z = 1 face
        t1, t2 = torch.rand(n_per_face, device=device), torch.rand(n_per_face, device=device)
        pts_list.append(torch.stack([t1, t2, torch.ones(n_per_face, device=device)], dim=1))

        pts = torch.cat(pts_list, dim=0)
        vals = exact_solution(pts[:, 0], pts[:, 1], pts[:, 2], self.a1, self.a2, self.a3)
        return pts, vals

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
        """Apply gradient mask to hash tables."""
        mask = self.level_grad_mask.view(-1, 1, 1)
        for ht in [self.encoding.hash_table_1, self.encoding.hash_table_2,
                   self.encoding.hash_table_3, self.encoding.hash_table_4]:
            if ht.grad is not None:
                ht.grad *= mask

    def forward(self, x):
        enc = self.encoding(x)
        return self.mlp(enc)

    def forward_with_laplacian(self, x):
        enc, dx, dy, dz, dxx, dyy, dzz = self.encoding.forward_with_second_derivatives_cuda(x)
        u, laplacian = self.mlp.forward_with_laplacian_cuda(enc, dx, dy, dz, dxx, dyy, dzz)
        return u, laplacian

    def loss_pde(self, pts):
        u, lap = self.forward_with_laplacian(pts)
        f = source_term(pts[:, 0], pts[:, 1], pts[:, 2], self.a1, self.a2, self.a3).unsqueeze(-1)
        residual = lap + K**2 * u - f
        return (residual**2).mean()

    def loss_bc(self, bc_pts, bc_vals):
        u = self.forward(bc_pts)
        return ((u.squeeze() - bc_vals)**2).mean()

    def evaluate(self, resolution=50, z_slice=0.5):
        """Evaluate L2 error on full 3D volume."""
        with torch.no_grad():
            # Full 3D evaluation
            res_3d = min(resolution, 34)
            x = torch.linspace(0, 1, res_3d, device=device)
            y = torch.linspace(0, 1, res_3d, device=device)
            z = torch.linspace(0, 1, res_3d, device=device)
            X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
            pts_3d = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)

            u_pred_3d = self.forward(pts_3d).squeeze()
            u_exact_3d = exact_solution(pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2],
                                        self.a1, self.a2, self.a3)

            l2_error = torch.sqrt(((u_pred_3d - u_exact_3d)**2).sum()) / torch.sqrt((u_exact_3d**2).sum() + 1e-8)
            max_error = (u_pred_3d - u_exact_3d).abs().max()

            # 2D slice for visualization
            x2d = torch.linspace(0, 1, resolution, device=device)
            y2d = torch.linspace(0, 1, resolution, device=device)
            X2d, Y2d = torch.meshgrid(x2d, y2d, indexing='ij')
            Z2d = torch.full_like(X2d, z_slice)
            pts_2d = torch.stack([X2d.flatten(), Y2d.flatten(), Z2d.flatten()], dim=1)

            u_pred_2d = self.forward(pts_2d).reshape(resolution, resolution)
            u_exact_2d = exact_solution(X2d, Y2d, Z2d, self.a1, self.a2, self.a3)

        return l2_error.item(), max_error.item(), u_pred_2d.cpu().numpy(), u_exact_2d.cpu().numpy()


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
    n_epochs = config.get('n_epochs', 60000)
    seed = config.get('seed', 42)
    lr = config.get('lr', 1e-3)
    num_collocation = config.get('num_collocation', 40000)
    eval_interval = config.get('eval_interval', 1000)
    save_plots = config.get('save_plots', True)

    # Adaptive LR config
    use_adaptive_lr = config.get('use_adaptive_lr', True)
    lr_patience = config.get('lr_patience', 3000)
    min_lr = config.get('min_lr', 1e-6)

    torch.manual_seed(seed)
    np.random.seed(seed)

    a1, a2, a3 = config.get('a1', 3), config.get('a2', 3), config.get('a3', 3)

    print("=" * 70)
    print("Helmholtz 3D PINN")
    print("=" * 70)
    print(f"PDE: Laplacian(u) + k^2*u = f, k={K}")
    print(f"Exact: u = sin({a1}*pi*x) * sin({a2}*pi*y) * sin({a3}*pi*z)")
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Epochs: {n_epochs}, Seed: {seed}")
    print(f"Hidden: {config.get('hidden_dim', 128)}, Layers: {config.get('n_layers', 4)}")
    print(f"Collocation: {num_collocation}, BC/face: {config.get('num_bc_per_face', 5000)}")
    if use_adaptive_lr:
        print(f"Adaptive LR: ON (patience={lr_patience}, min_lr={min_lr:.0e})")
    print("=" * 70)

    if not CUDA_AVAILABLE:
        print("\nERROR: CUDA extension not available!")
        print("Please run: python setup_mlp_cuda_v2.py install")
        return None

    model = HermiteNGP_PINN_CUDA_3D(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15000, gamma=0.5)

    ema = EMA(model, decay=0.999)

    # Warmup
    print("\nWarmup...")
    for _ in range(20):
        pts = torch.rand(num_collocation, 3, device=device)
        bc_pts, bc_vals = model.generate_bc_points()
        loss_pde = model.loss_pde(pts)
        loss_bc = model.loss_bc(bc_pts, bc_vals)
        loss = loss_pde + model.bc_weight * loss_bc
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

        # Sample points
        pts = torch.rand(num_collocation, 3, device=device)
        bc_pts, bc_vals = model.generate_bc_points()

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

        optimizer.zero_grad()
        loss.backward()
        model.apply_level_mask()
        optimizer.step()
        scheduler.step()
        ema.update(model)

        # Evaluate
        if (epoch + 1) % actual_eval_interval == 0 or epoch == 0:
            ema.apply_shadow(model)
            l2, max_err, u_pred, u_exact = model.evaluate()
            ema.restore(model)

            if l2 < best_l2:
                best_l2 = l2
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
                print(f"  >> LR reduced to {lr_current:.2e} (no improvement for {lr_patience} epochs)")
                epochs_since_improvement = 0

            elapsed = time.perf_counter() - t0
            history.append((epoch + 1, l2, max_err, best_l2))

            n_active = len(active_levels)
            print(f"  Epoch {epoch+1:6d}: L2={l2:.4e}, best={best_l2:.4e} @{best_epoch}, "
                  f"bc_w={model.bc_weight:.1f}, lvls={n_active}, lr={lr_current:.2e}, time={elapsed:.0f}s")

    # Restore best model
    if best_state is not None:
        for name, param in model.named_parameters():
            if name in best_state:
                param.data = best_state[name]

    # Final evaluation
    l2_final, max_err_final, u_pred, u_exact = model.evaluate(resolution=100)
    elapsed_total = time.perf_counter() - t0
    ms_per_epoch = elapsed_total / n_epochs * 1000

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"a1={a1}, a2={a2}, a3={a3}")
    print(f"Best L2 Error: {best_l2:.4e} @ epoch {best_epoch}")
    print(f"Final L2 Error: {l2_final:.4e}")
    print(f"Total Time: {elapsed_total:.1f}s ({ms_per_epoch:.2f} ms/epoch)")

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
        'a1': a1, 'a2': a2, 'a3': a3,
    }

    if save_plots:
        save_visualization(u_pred, u_exact, best_l2, best_epoch, config)

    if config.get('save_npz', True):
        save_results_npz(model, config, results, config.get('output_path'))

    return results


def save_results_npz(model, config, results, output_path=None):
    """Save training results to NPZ file."""
    if output_path is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
        a1, a2, a3 = config.get('a1', 3), config.get('a2', 3), config.get('a3', 3)
        output_path = os.path.join(output_dir, f'helmholtz3d_v3_a{a1}_{a2}_{a3}_results.npz')

    save_dict = {
        'config_n_epochs': config.get('n_epochs', 0),
        'config_seed': config.get('seed', 0),
        'config_lr': config.get('lr', 0),
        'config_a1': config.get('a1', 3),
        'config_a2': config.get('a2', 3),
        'config_a3': config.get('a3', 3),
        'config_hidden_dim': config.get('hidden_dim', 0),
        'config_n_layers': config.get('n_layers', 0),
        'config_num_collocation': config.get('num_collocation', 0),
        'best_l2': results.get('best_l2', 0),
        'best_epoch': results.get('best_epoch', 0),
        'final_l2': results.get('final_l2', 0),
        'total_time': results.get('total_time', 0),
        'ms_per_epoch': results.get('ms_per_epoch', 0),
        'n_params': results.get('n_params', 0),
        'history_epochs': np.array([h[0] for h in results.get('history', [])]),
        'history_l2': np.array([h[1] for h in results.get('history', [])]),
        'history_best_l2': np.array([h[3] for h in results.get('history', [])]),
        'u_pred': results.get('u_pred', np.array([])),
        'u_exact': results.get('u_exact', np.array([])),
    }

    np.savez_compressed(output_path, **save_dict)
    print(f"Results saved: {output_path}")

    # Save model state for reuse
    params_path = output_path.replace('.npz', '_params.pth')
    torch.save(model.state_dict(), params_path)
    print(f"Model params saved: {params_path}")


def save_visualization(u_pred, u_exact, best_l2, best_epoch, config):
    """Save visualization of results (2D slice at z=0.5)."""
    res = u_pred.shape[0]
    x = np.linspace(0, 1, res)
    y = np.linspace(0, 1, res)
    X, Y = np.meshgrid(x, y, indexing='ij')

    error = np.abs(u_pred - u_exact)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    im0 = axes[0].contourf(X, Y, u_exact, levels=50, cmap='RdBu_r')
    axes[0].set_title('Exact (z=0.5)')
    axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].contourf(X, Y, u_pred, levels=50, cmap='RdBu_r')
    axes[1].set_title(f'Prediction (L2={best_l2:.2e})')
    axes[1].set_xlabel('x'); axes[1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].contourf(X, Y, error, levels=50, cmap='hot')
    axes[2].set_title(f'|Error| (max={error.max():.2e})')
    axes[2].set_xlabel('x'); axes[2].set_ylabel('y')
    plt.colorbar(im2, ax=axes[2])

    mid = res // 2
    axes[3].plot(x, u_exact[:, mid], 'b-', label='Exact', linewidth=2)
    axes[3].plot(x, u_pred[:, mid], 'r--', label='Prediction', linewidth=2)
    axes[3].set_title('Cross-section y=0.5, z=0.5')
    axes[3].set_xlabel('x'); axes[3].set_ylabel('u')
    axes[3].legend(); axes[3].grid(True)

    a1, a2, a3 = config.get('a1', 3), config.get('a2', 3), config.get('a3', 3)
    plt.suptitle(f'Helmholtz 3D V3 (a={a1},{a2},{a3}): Best L2={best_l2:.4e} @ epoch {best_epoch}', fontsize=14)
    plt.tight_layout()

    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, f'helmholtz3d_v3_a{a1}_{a2}_{a3}_result.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nVisualization saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Helmholtz 3D - CUDA MLP V3')
    parser.add_argument('--epochs', type=int, default=100000, help='Number of epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--a1', type=int, default=3, help='Frequency a1')
    parser.add_argument('--a2', type=int, default=3, help='Frequency a2')
    parser.add_argument('--a3', type=int, default=3, help='Frequency a3')
    parser.add_argument('--omega', type=float, default=0.5, help='SIREN omega')
    parser.add_argument('--hidden', type=int, default=128, help='MLP hidden dim')
    parser.add_argument('--layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--collocation', type=int, default=40000, help='Number of collocation points')
    parser.add_argument('--bc-per-face', type=int, default=5000, help='BC points per face')
    parser.add_argument('--no-plots', action='store_true', help='Disable plot saving')
    parser.add_argument('--no-save', action='store_true', help='Disable NPZ saving')
    parser.add_argument('--output', type=str, default=None, help='Output path')
    parser.add_argument('--lr-patience', type=int, default=3000000, help='LR patience')
    parser.add_argument('--min-lr', type=float, default=1e-6, help='Min LR')
    parser.add_argument('--no-adaptive-lr', action='store_true', help='Disable adaptive LR')
    args = parser.parse_args()

    config = {
        'n_epochs': args.epochs,
        'seed': args.seed,
        'lr': args.lr,
        'a1': args.a1,
        'a2': args.a2,
        'a3': args.a3,
        'omega': args.omega,
        'hidden_dim': args.hidden,
        'n_layers': args.layers,
        'num_collocation': args.collocation,
        'num_bc_per_face': args.bc_per_face,
        'save_plots': not args.no_plots,
        'save_npz': not args.no_save,
        'output_path': args.output,
        'lr_patience': args.lr_patience,
        'min_lr': args.min_lr,
        'use_adaptive_lr': not args.no_adaptive_lr,

        # Fixed config (paper a=3 setting: 6 levels, hash 2^13)
        'n_levels': 6,
        'log2_hashmap_size': 13,
        'bc_weight_init': 5000.0,
        'bc_weight_cap': 50000.0,
        'eval_interval': 500,
    }

    train(config)
