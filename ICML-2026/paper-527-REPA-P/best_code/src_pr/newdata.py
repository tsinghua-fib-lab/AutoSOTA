"""
Physics-Informed Loss for Turbulent Channel Flow (Re_τ = 180)
=============================================================

Data source:
    2D x-y slice from 3D incompressible channel flow DNS.
    File: ch_2Dxysec.pickle, shape (10000, 128, 48, 1)
        - Axis 0: time snapshots (10000 frames)
        - Axis 1: x — streamwise direction (128 points)
        - Axis 2: y — wall-normal direction (48 points)
        - Axis 3: channel dim (1 — streamwise velocity fluctuation u')

Data analysis findings (verified on real data):
    1. Mean U(y) ≈ 0 at all y  →  data stores u' (fluctuation), not total u
    2. y=0:  std=0.017, max|u'|=0.124  →  BOTTOM WALL (no-slip holds)
    3. y=47: std=0.082, max|u'|=0.481  →  CHANNEL CENTERLINE (not a wall!)
    4. |u(x=0)-u(x=127)| = 0.175  ≫  |u(x=1)-u(x=0)| = 0.020
       → x=0 and x=127 are NOT periodic neighbours; grid does not
         repeat endpoint & may not span exactly one period.
    5. The 48 y-points cover the LOWER HALF of the channel only
       (wall → centerline), NOT wall-to-wall.

Governing equations (of the underlying 3D DNS):
    ∂u/∂t + (u·∇)u = -(1/ρ)∇p + ν∇²u + f ,   ∇·u = 0
    Ref: Kim, Moin & Moser (1987), JFM 177

Why NO PDE residual can be written:
    The observed field q(x,y,t) = u'(x, y, z₀, t) is a single velocity
    fluctuation component on a single 2D slice.  The streamwise momentum
    equation requires v', w', p', ∂u'/∂z, ∂²u'/∂z², none of which are
    available.  No closed PDE exists for q alone.

Strictly valid constraints:
    ✓  Wall no-slip at y=0 ONLY:  u'(x, 0, t) = 0
    ✗  No-slip at y=47:  WRONG — it is the centerline, not a wall
    ✗  u(x=0)=u(x=127):  WRONG — they are not periodic neighbours
    ✗  ∂_t u = ν∇²u:     WRONG — not the governing equation

Regularisers (valid, but NOT physics / NOT PDE residuals):
    ✓  Smoothness:  ||∇²u'||²   (penalises high-frequency noise)
    ✓  Gradient:    ||∇u'||²     (penalises excessive spatial variation)
    NOTE: y-direction FD must use SLICING (interior only), not roll,
          because y is wall-bounded, not periodic.

Final loss:
    L = L_data  +  λ_wall · L_wall  +  λ_smooth · L_smooth  [+ λ_grad · L_grad]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PhysicsLoss(nn.Module):
    """
    Physics-informed loss for turbulent channel flow reconstruction.

    Only enforces constraints that are STRICTLY valid for the available
    data: streamwise velocity fluctuation u' on a 2D (x, y) slice
    covering the lower half of a channel (wall at y=0, centerline at y=-1).

    Strict physical constraint:
        L_wall  — No-slip at bottom wall (y=0) ONLY.

    Regularisers (NOT PDE residuals):
        L_smooth   — Laplacian smoothness  (interior points only)
        L_gradient — Gradient magnitude    (interior points only)

    Data layout:
        4-D:  (B, X=128, Y=48, 1)
        5-D:  (B, T, X=128, Y=48, 1)
        y=0          → bottom wall    (no-slip)
        y=1 … y=46  → interior
        y=47         → channel centre (NO no-slip)
    """

    def __init__(self,
                 dx=1.0, dy=1.0,
                 lambda_wall=0.1,
                 lambda_smooth=0.01,
                 lambda_gradient=0.0,
                 near_wall_rows=3,
                 lambda_near_wall=0.0,
                 device='cuda'):
        """
        Args:
            dx, dy:            Grid spacing (normalised, typically 1.0)
            lambda_wall:       Weight for wall no-slip (y=0 only)
            lambda_smooth:     Weight for smoothness regulariser
            lambda_gradient:   Weight for gradient regulariser (0 = off)
            near_wall_rows:    How many rows near the wall to include in
                               the optional near-wall penalty (default 3)
            lambda_near_wall:  Weight for near-wall small-value penalty
                               (soft; 0 = off)
            device:            Compute device
        """
        super().__init__()
        self.dx = dx
        self.dy = dy
        self.lambda_wall = lambda_wall
        self.lambda_smooth = lambda_smooth
        self.lambda_gradient = lambda_gradient
        self.near_wall_rows = near_wall_rows
        self.lambda_near_wall = lambda_near_wall
        self.device = device

    # ----------------------------------------------------------------
    #  Finite-difference helpers  (wall-bounded in y, roll in x)
    # ----------------------------------------------------------------

    @staticmethod
    def _is_channels_first(u):
        return u.dim() in (4, 5) and u.size(1) == 1

    @staticmethod
    def _x_dim(u):
        """Return the x-axis dimension index."""
        if PhysicsLoss._is_channels_first(u):
            return 3 if u.dim() == 5 else 2
        return 2 if u.dim() == 5 else 1

    @staticmethod
    def _y_dim(u):
        """Return the y-axis dimension index."""
        if PhysicsLoss._is_channels_first(u):
            return 4 if u.dim() == 5 else 3
        return 3 if u.dim() == 5 else 2

    def _second_diff_x_interior(self, u):
        """Second derivative in x on interior points only."""
        xd = self._x_dim(u)
        yd = self._y_dim(u)
        NX = u.size(xd)
        NY = u.size(yd)
        if NX < 3 or NY < 3:
            raise ValueError('Need at least 3 grid points per dimension.')

        u_xp1 = u.narrow(xd, 2, NX - 2).narrow(yd, 1, NY - 2)
        u_x = u.narrow(xd, 1, NX - 2).narrow(yd, 1, NY - 2)
        u_xm1 = u.narrow(xd, 0, NX - 2).narrow(yd, 1, NY - 2)
        return (u_xp1 - 2.0 * u_x + u_xm1) / (self.dx ** 2)

    def _gradient_x(self, u):
        """∂u/∂x on interior x-points only, using slicing (non-periodic)."""
        xd = self._x_dim(u)
        yd = self._y_dim(u)
        NX = u.size(xd)
        NY = u.size(yd)
        if NX < 3 or NY < 3:
            raise ValueError('Need at least 3 grid points per dimension.')

        u_xp1 = u.narrow(xd, 2, NX - 2).narrow(yd, 1, NY - 2)
        u_xm1 = u.narrow(xd, 0, NX - 2).narrow(yd, 1, NY - 2)
        return (u_xp1 - u_xm1) / (2.0 * self.dx)

    def _gradient_y_interior(self, u):
        """∂u/∂y on INTERIOR y-points only (y=1 … y=N-2).

        Uses slicing, NOT roll, because y is wall-bounded.
        Returns a tensor shorter in both spatial dimensions because
        x/y derivatives are evaluated only on the shared interior stencil.
        """
        xd = self._x_dim(u)
        yd = self._y_dim(u)
        NX = u.size(xd)
        NY = u.size(yd)
        if NX < 3 or NY < 3:
            raise ValueError('Need at least 3 grid points per dimension.')

        # u_yp1: y+1,  u_ym1: y-1
        u_yp1 = u.narrow(yd, 2, NY - 2).narrow(xd, 1, NX - 2)     # y=2 … y=N-1
        u_ym1 = u.narrow(yd, 0, NY - 2).narrow(xd, 1, NX - 2)     # y=0 … y=N-3
        return (u_yp1 - u_ym1) / (2.0 * self.dy)

    def _laplacian_interior(self, u):
        """∇²u on INTERIOR y-points only (y=1 … y=N-2).

        Both x and y directions use interior slicing only.

        Returns tensor 2 shorter in both spatial dimensions.
        """
        yd = self._y_dim(u)
        NX = u.size(self._x_dim(u))
        NY = u.size(yd)
        if NX < 3 or NY < 3:
            raise ValueError('Need at least 3 grid points per dimension.')

        # --- ∂²u/∂x²  (interior only, no x-wrap) ---
        d2x = self._second_diff_x_interior(u)

        # --- ∂²u/∂y²  (slicing, interior only) ---
        xd = self._x_dim(u)
        u_yp1 = u.narrow(yd, 2, NY - 2).narrow(xd, 1, NX - 2)      # y=2 … N-1
        u_y   = u.narrow(yd, 1, NY - 2).narrow(xd, 1, NX - 2)      # y=1 … N-2
        u_ym1 = u.narrow(yd, 0, NY - 2).narrow(xd, 1, NX - 2)      # y=0 … N-3
        d2y = (u_yp1 - 2 * u_y + u_ym1) / (self.dy ** 2)

        return d2x + d2y

    def wall_residual_map(self, u_pred):
        """Return a residual map that is non-zero only on the bottom wall y=0."""
        residual = torch.zeros_like(u_pred)
        yd = self._y_dim(u_pred)
        residual.narrow(yd, 0, 1).copy_(u_pred.narrow(yd, 0, 1))
        return residual

    def smoothness_residual_map(self, u_pred):
        """Embed interior Laplacian residuals back to full image size."""
        residual = torch.zeros_like(u_pred)
        xd = self._x_dim(u_pred)
        yd = self._y_dim(u_pred)
        lap = self._laplacian_interior(u_pred)
        residual.narrow(xd, 1, u_pred.size(xd) - 2).narrow(yd, 1, u_pred.size(yd) - 2).copy_(lap)
        return residual

    def gradient_residual_map(self, u_pred):
        """Return squared-gradient residual map on shared interior points."""
        residual = torch.zeros_like(u_pred)
        xd = self._x_dim(u_pred)
        yd = self._y_dim(u_pred)
        dudx = self._gradient_x(u_pred)
        dudy = self._gradient_y_interior(u_pred)
        grad_mag = torch.sqrt(torch.clamp(dudx ** 2 + dudy ** 2, min=0.0))
        residual.narrow(xd, 1, u_pred.size(xd) - 2).narrow(yd, 1, u_pred.size(yd) - 2).copy_(grad_mag)
        return residual

    # ----------------------------------------------------------------
    #  Strictly valid physical constraint
    # ----------------------------------------------------------------

    def wall_noslip_loss(self, u_pred):
        """
        No-slip at the BOTTOM WALL only (y = 0).

        Physics:  u'(x, y=0, t) = 0
        Verified: data shows  y=0  mean|u'| = 0.014, max|u'| = 0.124

        y=47 is the channel CENTERLINE (std=0.082, max|u'|=0.481),
        NOT a wall — no-slip is NOT applied there.

        L_wall = mean[ u'(x, y=0, t)² ]
        """
        yd = self._y_dim(u_pred)
        # Extract y=0 slice
        u_wall = u_pred.narrow(yd, 0, 1)
        return torch.mean(u_wall ** 2)

    # ----------------------------------------------------------------
    #  Optional: near-wall penalty  (soft, not strict)
    # ----------------------------------------------------------------

    def near_wall_loss(self, u_pred):
        """
        Soft penalty: values in the first few rows near the wall should
        be small (physically, u' grows away from wall but is still
        smaller near it than in the bulk).

        L_near = mean[ u'(x, y=1..k, t)² ]

        This is NOT a strict BC — it is a soft regulariser.
        """
        yd = self._y_dim(u_pred)
        k = min(self.near_wall_rows, u_pred.size(yd) - 1)
        if k <= 0:
            return torch.tensor(0.0, device=u_pred.device)
        u_near = u_pred.narrow(yd, 1, k)  # y=1 … y=k
        return torch.mean(u_near ** 2)

    # ----------------------------------------------------------------
    #  Regularisers  (NOT PDE residuals)
    # ----------------------------------------------------------------

    def smoothness_regularizer(self, u_pred):
        """
        Laplacian smoothness on INTERIOR points only.

        L_smooth = mean[ (∇²u')² ]

        x: periodic roll.  y: slicing (no wall-to-centerline wrap).
        This is a REGULARISER, not a PDE residual.
        """
        lap = self._laplacian_interior(u_pred)
        return torch.mean(lap ** 2)

    def gradient_regularizer(self, u_pred):
        """
        Gradient magnitude on INTERIOR y-points only.

        L_grad = mean[ (∂u'/∂x)² + (∂u'/∂y)² ]

        This is a REGULARISER, not a PDE residual.
        """
        yd = self._y_dim(u_pred)
        NY = u_pred.size(yd)

        # ∂u/∂x everywhere, then trim to interior y
        dudx_full = self._gradient_x(u_pred)
        dudx = dudx_full.narrow(yd, 1, NY - 2)

        # ∂u/∂y interior (already trimmed)
        dudy = self._gradient_y_interior(u_pred)

        return torch.mean(dudx ** 2 + dudy ** 2)

    # ----------------------------------------------------------------
    #  Combined forward
    # ----------------------------------------------------------------

    def forward(self, u_pred, u_true=None, return_components=False):
        """
        Total physics-informed loss.

        L = L_data
            + λ_wall      · L_wall          (strict: y=0 no-slip)
            + λ_smooth    · L_smooth_reg    (regulariser)
           [+ λ_grad      · L_grad_reg]     (regulariser, optional)
           [+ λ_near_wall · L_near_wall]    (soft, optional)

        Args:
            u_pred:  (B, 128, 48, 1) or (B, T, 128, 48, 1)
            u_true:  same shape (optional, for data MSE)
            return_components:  return loss dict alongside total

        Returns:
            loss_total, [loss_dict]
        """
        loss_dict = {}

        # --- Data loss ---
        if u_true is not None:
            loss_data = F.mse_loss(u_pred, u_true)
        else:
            loss_data = torch.tensor(0.0, device=u_pred.device)
        loss_dict['data'] = loss_data

        # --- Strict physics ---
        loss_wall = self.wall_noslip_loss(u_pred)
        loss_dict['wall_noslip'] = loss_wall

        # --- Regularisers ---
        loss_smooth = self.smoothness_regularizer(u_pred)
        loss_dict['smooth_reg'] = loss_smooth

        loss_total = (loss_data
                      + self.lambda_wall * loss_wall
                      + self.lambda_smooth * loss_smooth)

        if self.lambda_gradient > 0:
            loss_grad = self.gradient_regularizer(u_pred)
            loss_dict['gradient_reg'] = loss_grad
            loss_total = loss_total + self.lambda_gradient * loss_grad

        if self.lambda_near_wall > 0:
            loss_nw = self.near_wall_loss(u_pred)
            loss_dict['near_wall_reg'] = loss_nw
            loss_total = loss_total + self.lambda_near_wall * loss_nw

        if return_components:
            return loss_total, loss_dict
        return loss_total


# ============================================================================
#  Validation
# ============================================================================

def validate_physics_loss():
    """Validate on real turbulent channel flow data."""
    import pickle

    print("=" * 64)
    print("🔬  Physics Loss Validation  (ch_2Dxysec, Re_τ=180)")
    print("=" * 64)

    path = "/home/limx/benchmarks/senseiver/Data/Turbulent/ch_2Dxysec.pickle"
    with open(path, "rb") as f:
        data = pickle.load(f)
    data = np.array(data, dtype=np.float32)
    T, NX, NY, C = data.shape
    nf = float(np.abs(data).max())
    dn = data / nf

    print(f"\nShape : {data.shape}")
    print(f"Range : [{data.min():.4f}, {data.max():.4f}]")
    print(f"Norm  : {nf:.4f}")
    print(f"Layout: T={T}, NX={NX}(x), NY={NY}(y), C={C}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    loss_fn = PhysicsLoss(
        dx=1.0, dy=1.0,
        lambda_wall=0.1,
        lambda_smooth=0.01,
        lambda_gradient=0.0,
        device=device
    )

    u0 = torch.tensor(dn[0:1], dtype=torch.float32, device=device)

    # --- Test 1: perfect prediction ---
    print("【Test 1】Perfect prediction")
    _, ld = loss_fn(u_pred=u0, u_true=u0, return_components=True)
    for k, v in ld.items():
        print(f"    L_{k:15s} = {v.item():.6f}")

    # --- Test 2: noisy prediction ---
    print("\n【Test 2】Noisy prediction (σ=0.1)")
    u_noisy = u0 + 0.1 * torch.randn_like(u0)
    _, ld = loss_fn(u_pred=u_noisy, u_true=u0, return_components=True)
    for k, v in ld.items():
        print(f"    L_{k:15s} = {v.item():.6f}")

    # --- Test 3: verify wall constraint ---
    print("\n【Test 3】Wall no-slip (y=0 only)")
    print(f"    y=0  (wall)   : mean|u'| = {u0.narrow(loss_fn._y_dim(u0), 0, 1).abs().mean().item():.6f}")
    print(f"    y=47 (center) : mean|u'| = {u0.narrow(loss_fn._y_dim(u0), 47, 1).abs().mean().item():.6f}")
    print(f"    L_wall (clean): {loss_fn.wall_noslip_loss(u0).item():.6f}")
    print(f"    ↑ Penalises y=0 ONLY; y=47 is centerline, NOT penalised.")

    # --- Test 4: interior-only FD ---
    print("\n【Test 4】Interior-only Laplacian (no y-wrap)")
    lap = loss_fn._laplacian_interior(u0)
    yd = loss_fn._y_dim(u0)
    print(f"    Input  y-size : {u0.size(yd)}")
    print(f"    Output y-size : {lap.size(yd)}  (interior only, 2 fewer)")
    print(f"    L_smooth (clean) = {loss_fn.smoothness_regularizer(u0).item():.6f}")
    print(f"    L_smooth (noisy) = {loss_fn.smoothness_regularizer(u_noisy).item():.6f}")

    print("\n" + "=" * 64)
    print("✅  Validation complete")
    print("=" * 64)

    return loss_fn


if __name__ == "__main__":
    loss_fn = validate_physics_loss()

    print("\n" + "=" * 64)
    print("📋  Recommended configuration")
    print("=" * 64)
    print("""
# Physics loss for ch_2Dxysec turbulent channel flow
# ---------------------------------------------------
# Data: u' (velocity fluctuation), shape (B, 128, 48, 1)
#   y=0  → bottom wall   (no-slip: u'=0)
#   y=47 → channel centre (NOT a wall)
#
# STRICT constraint:
#   Wall no-slip at y=0 only
#
# REGULARISERS (not PDE residuals):
#   Smoothness (interior-only Laplacian, no y-wrap)
#   Gradient   (interior-only, optional)
#
# There is NO closed PDE for a single velocity component
# from a 3D turbulent flow — do NOT use PDE residual loss.

loss_fn = PhysicsLoss(
    dx=1.0, dy=1.0,
    lambda_wall=0.1,        # strict: y=0 no-slip
    lambda_smooth=0.01,     # regulariser
    lambda_gradient=0.0,    # regulariser (off by default)
    device='cuda'
)
""")