import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence, Tuple

# NOTE: THIS centralized controller wont be used for now, keeping it just in case.  
# class KS2DControlNet(nn.Module):
#     """
#     Centralized Controller for 2D Kuramoto-Sivashinsky on Periodic Boundary.
    
#     Structure:
#     - Branch: CNN processing of the full 2D error field (u - u_target).
#     - Trunk: Fourier encoding of normalized actuator positions.
#     - Output: Forcing intensity (u) only.
#     """
#     features: Sequence[int] = (16, 32)
#     domain_size: Tuple[float, float] = (32.0, 32.0) # (Lx, Ly)
#     u_max: float = 10.0  # KS often requires lower forcing magnitude than Heat
    
#     def setup(self):
#         self.frequencies = jnp.array([1.0, 2.0, 4.0, 8.0])

#     def branch_net(self, error, error_grad_x, error_grad_y):
#         """
#         CNN branch processes the global spatial error state.
#         """
#         # (N, N, 3) input
#         x = jnp.stack([error, error_grad_x, error_grad_y], axis=-1)

#         # 2D Convolutions with Periodic Padding is tricky in standard Conv,
#         # but since this is global context, standard 'SAME' padding is 
#         # acceptable. Ideally, one would manually pad with 'wrap' before Conv.
#         for feat in self.features:
#             x = nn.Conv(feat, kernel_size=(3, 3), padding='SAME')(x)
#             x = nn.relu(x)

#         # Flatten and global normalization
#         x = x.reshape(-1)
#         # x = x / (jnp.linalg.norm(x) + 1.0) 
#         x = nn.Dense(64)(x)
#         x = nn.tanh(x)
#         return x

#     def trunk_net(self, xi_norm):
#         """
#         Fourier encoding for normalized 2D positions.
#         """
#         angle_x = xi_norm[:, 0, None] * self.frequencies * jnp.pi
#         angle_y = xi_norm[:, 1, None] * self.frequencies * jnp.pi

#         encoded = jnp.concatenate([
#             jnp.sin(angle_x), jnp.cos(angle_x),
#             jnp.sin(angle_y), jnp.cos(angle_y)
#         ], axis=-1)

#         for feat in [64, 64]:
#             encoded = nn.Dense(feat)(encoded)
#             encoded = nn.tanh(encoded)
#         return encoded

#     @nn.compact
#     def __call__(self, u_curr, u_target, xi_fixed):
#         """
#         Args:
#             u_curr: (N, N) field
#             u_target: (N, N) target
#             xi_fixed: (M, 2) actuator positions in domain units
#         """
#         # 1. Normalize Coordinates [0, L] -> [0, 1]
#         Lx, Ly = self.domain_size
#         xi_norm = jnp.stack([
#             xi_fixed[:, 0] / Lx,
#             xi_fixed[:, 1] / Ly
#         ], axis=-1)

#         # 2. Compute Error & Gradient
#         error = u_curr - u_target
#         # Gradient returns [grad_y, grad_x]
#         grads = jnp.gradient(error)
#         error_grad_y, error_grad_x = grads[0], grads[1]

#         # 3. Branch & Trunk
#         branch_out = self.branch_net(error, error_grad_x, error_grad_y)
#         trunk_out = self.trunk_net(xi_norm)

#         # 4. Fusion
#         branch_repeated = jnp.tile(branch_out, (xi_fixed.shape[0], 1))
#         combined = jnp.concatenate([branch_repeated, trunk_out], axis=-1)

#         # 5. Output Head (Intensity only)
#         h = nn.Dense(64)(combined)
#         h = nn.tanh(h)
#         u_raw = nn.Dense(1)(h).squeeze(-1)

#         return self.u_max * jnp.tanh(u_raw)
    
    
class DecentralizedKS2DControlNet(nn.Module):
    """
    Decentralized Controller for 2D KS.
    Fixes applied:
    1. Coordinate Alignment: Maps x->Row, y->Col to match Solver 'ij' indexing.
    2. Soft Normalization: Uses +1.0 to allow proportional control.
    """
    features: Sequence[int] = (16, 32)
    domain_size: Tuple[float, float] = (32.0, 32.0)
    u_max: float = 10.0
    patch_size: int = 12 

    def setup(self):
        self.frequencies = jnp.array([1.0, 2.0, 4.0, 8.0])

    def extract_local_patch(self, field, xi_norm, n_grid):
        """
        Extracts patch with Periodic Boundary Conditions.
        """
        # --- CRITICAL FIX: COORDINATE ALIGNMENT ---
        # Solver uses indexing='ij' (Axis 0 is X, Axis 1 is Y).
        # Therefore:
        # Index i (Axis 0) must come from x (xi_norm[0])
        # Index j (Axis 1) must come from y (xi_norm[1])
        
        i = (xi_norm[0] * n_grid).astype(int)  # Changed from index 1 to 0
        j = (xi_norm[1] * n_grid).astype(int)  # Changed from index 0 to 1
        # ------------------------------------------
        
        half_patch = self.patch_size // 2

        # Use 'wrap' for periodic boundaries
        padded_field = jnp.pad(field, (
            (half_patch, half_patch),
            (half_patch, half_patch)
        ), mode='wrap')

        start_i = i 
        start_j = j 

        patch = jax.lax.dynamic_slice(
            padded_field,
            (start_i, start_j),
            (self.patch_size, self.patch_size)
        )
        return patch

    def branch_net(self, local_patch):
        x = local_patch
        for feat in self.features:
            x = nn.Conv(feat, kernel_size=(3, 3), padding='SAME')(x)
            x = nn.relu(x)

        x = x.reshape(-1)
        
        # --- CRITICAL FIX: SOFT NORMALIZATION ---
        # +1.0 prevents "Bang-Bang" behavior on small errors
        # +1e-6 caused instability
        x = x / (jnp.linalg.norm(x) + 1.0) 
        
        x = nn.Dense(32)(x)
        x = nn.tanh(x)
        return x

    def trunk_net(self, xi_norm):
        angle_x = xi_norm[:, 0, None] * self.frequencies * jnp.pi
        angle_y = xi_norm[:, 1, None] * self.frequencies * jnp.pi

        encoded = jnp.concatenate([
            jnp.sin(angle_x), jnp.cos(angle_x),
            jnp.sin(angle_y), jnp.cos(angle_y)
        ], axis=-1)

        for feat in [32, 32]:
            encoded = nn.Dense(feat)(encoded)
            encoded = nn.tanh(encoded)
        return encoded

    @nn.compact
    def __call__(self, u_curr, u_target, xi_fixed):
        Lx, Ly = self.domain_size
        
        # Normalize to [0, 1]
        xi_norm = jnp.stack([
            xi_fixed[:, 0] / Lx,
            xi_fixed[:, 1] / Ly
        ], axis=-1)

        error = u_curr - u_target
        grads = jnp.gradient(error)
        
        # JAX gradient on (N, N) returns (grad_axis0, grad_axis1)
        # Since Axis 0 is X, Axis 1 is Y:
        grad_x, grad_y = grads[0], grads[1]

        n_grid = u_curr.shape[0]

        def get_local_obs(xi_single):
            p_err = self.extract_local_patch(error, xi_single, n_grid)
            p_gx = self.extract_local_patch(grad_x, xi_single, n_grid)
            p_gy = self.extract_local_patch(grad_y, xi_single, n_grid)
            return jnp.stack([p_err, p_gx, p_gy], axis=-1)

        local_patches = jax.vmap(get_local_obs)(xi_norm)

        branch_outs = jax.vmap(self.branch_net)(local_patches)
        trunk_outs = self.trunk_net(xi_norm)

        combined = jnp.concatenate([branch_outs, trunk_outs], axis=-1)
        
        h = nn.Dense(64)(combined)
        h = nn.tanh(h)
        u_raw = nn.Dense(1)(h).squeeze(-1)

        return self.u_max * jnp.tanh(u_raw)
    