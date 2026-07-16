import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence, Tuple

# NOTE: THIS centralized controller wont be used for now, keeping it just in case.
# class TurbulenceControlNet(nn.Module):
#     """
#     Centralized Controller for 2D Decaying Turbulence (Vorticity Formulation).
    
#     Structure:
#     - Branch: CNN processing of the full Vorticity field (w).
#     - Trunk: Fourier encoding of actuator positions.
#     - Output: Forcing intensity.
#     """
#     features: Sequence[int] = (16, 32)
#     domain_size: Tuple[float, float] = (1.0, 1.0) # Matches Solver L=1.0
#     u_max: float = 75.0  # Limits max forcing to prevent instability
    
#     def setup(self):
#         # Frequencies for Fourier Feature mapping
#         self.frequencies = jnp.array([1.0, 2.0, 4.0, 8.0])

#     def branch_net(self, w_field, w_grad_x, w_grad_y):
#         """
#         CNN branch processes the global vorticity state.
#         Input: Vorticity (w) and its spatial gradients (dissipation/shear).
#         """
#         # (N, N, 3) input
#         x = jnp.stack([w_field, w_grad_x, w_grad_y], axis=-1)

#         for feat in self.features:
#             x = nn.Conv(feat, kernel_size=(3, 3), padding='SAME')(x)
#             x = nn.relu(x)

#         # Flatten and global normalization
#         x = x.reshape(-1)
        
#         # Soft Normalization (Critical for preventing saturation near zero state)
#         x = x / (jnp.linalg.norm(x) + 1.0)
        
#         x = nn.Dense(64)(x)
#         x = nn.tanh(x)
#         return x

#     def trunk_net(self, xi_norm):
#         """Fourier encoding for normalized 2D actuator positions."""
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
#     def __call__(self, params, obs):
#         """
#         Args:
#             params: (Ignored in Flax functional calls, kept for API compatibility)
#             obs: (1, N, N) Vorticity field [Batch dim handled by vmap in training]
        
#         Note: The centralized net implicitly needs actuator positions. 
#         In this implementation, we assume they are fixed and stored in the class 
#         or passed via a slightly different API. For the Decentralized version below,
#         it is explicit.
#         """
#         # Remove batch dim if present (1, N, N) -> (N, N)
#         w_curr = obs.squeeze()
        
#         # Calculate gradients (Enstrophy density related)
#         grads = jnp.gradient(w_curr)
#         grad_y, grad_x = grads[0], grads[1] # Axis 0 is Y in 'xy' indexing

#         # Branch processing
#         # Note: Centralized net doesn't use Trunk in the same way as DeepONet 
#         # unless querying specific points. Here we simplify to a global CNN.
#         # Ideally, you would inject actuator positions here if they move.
        
#         # For compatibility with the dual-structure:
#         x = jnp.stack([w_curr, grad_x, grad_y], axis=-1)
#         for feat in self.features:
#             x = nn.Conv(feat, kernel_size=(3, 3), padding='SAME')(x)
#             x = nn.relu(x)
#         x = x.reshape(-1)
#         x = nn.Dense(128)(x)
#         x = nn.relu(x)
        
#         # Output head for N actuators
#         return x 

class DecentralizedTurbulenceNet(nn.Module):
    """
    Decentralized Controller for 2D Turbulence.
    Each agent (actuator) observes a local patch of vorticity around itself.
    """
    features: Sequence[int] = (16, 32)
    domain_size: Tuple[float, float] = (1.0, 1.0)
    u_max: float = 75.0
    patch_size: int = 20 # Larger patch for turbulence structures

    def setup(self):
        self.frequencies = jnp.array([1.0, 2.0, 4.0, 8.0])

    def extract_local_patch(self, field, xi_norm, n_grid):
        """
        Extracts patch with Periodic Boundary Conditions.
        """
        # --- COORDINATE ALIGNMENT (Matches JAX Solver) ---
        # Solver uses indexing='xy' (Standard Matrix/Image convention).
        # Axis 0 = Row = Y coordinate
        # Axis 1 = Col = X coordinate
        
        # Therefore:
        # i (Row index) comes from Y position (xi_norm[1])
        # j (Col index) comes from X position (xi_norm[0])
        
        i = (xi_norm[1] * n_grid).astype(int) 
        j = (xi_norm[0] * n_grid).astype(int)
        # -------------------------------------------------
        
        half_patch = self.patch_size // 2

        # Use 'wrap' for periodic boundaries (Turbulence is periodic)
        padded_field = jnp.pad(field, (
            (half_patch, half_patch),
            (half_patch, half_patch)
        ), mode='wrap')

        # Shift indices because of padding
        start_i = i 
        start_j = j 

        patch = jax.lax.dynamic_slice(
            padded_field,
            (start_i, start_j),
            (self.patch_size, self.patch_size)
        )
        return patch

    def branch_net(self, local_patch):
        """
        Processes the local observation (Patch).
        """
        x = local_patch # (H, W, Channels)
        
        for feat in self.features:
            x = nn.Conv(feat, kernel_size=(3, 3), padding='SAME')(x)
            x = nn.relu(x)

        x = x.reshape(-1)
        
        # --- SOFT NORMALIZATION ---
        # Critical for Decay tasks: As vorticity -> 0, gradients -> 0.
        # Standard normalization would amplify noise. 
        # +1.0 ensures that near zero state, the network outputs near zero.
        x = x / (jnp.linalg.norm(x) + 1.0) 
        
        x = nn.Dense(32)(x)
        x = nn.tanh(x)
        return x

    def trunk_net(self, xi_norm):
        """Encodes absolute position (Actuators stay fixed, but good context)."""
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
    def __call__(self, xi_fixed, obs):
        """
        Args:
            xi_fixed: (M, 2) Fixed actuator positions [0, L].
            obs: (1, N, N) Global Vorticity field [passed from Environment].
                 The network handles chopping this into local patches.
        """
        # 1. Setup
        Lx, Ly = self.domain_size
        w_curr = obs.squeeze() # (N, N)
        
        # Normalize positions to [0, 1]
        xi_norm = jnp.stack([
            xi_fixed[:, 0] / Lx,
            xi_fixed[:, 1] / Ly
        ], axis=-1)

        # 2. Compute Features (Vorticity + Gradients)
        # Gradients are important for sensing dissipation rate
        grads = jnp.gradient(w_curr)
        grad_y, grad_x = grads[0], grads[1] # Match indexing='xy'

        n_grid = w_curr.shape[0]

        # 3. Vectorized Patch Extraction (Per Agent)
        def get_local_obs(xi_single):
            p_w  = self.extract_local_patch(w_curr, xi_single, n_grid)
            p_gx = self.extract_local_patch(grad_x, xi_single, n_grid)
            p_gy = self.extract_local_patch(grad_y, xi_single, n_grid)
            return jnp.stack([p_w, p_gx, p_gy], axis=-1)

        # Extract patches for all agents in parallel
        local_patches = jax.vmap(get_local_obs)(xi_norm)

        # 4. Neural Processing
        branch_outs = jax.vmap(self.branch_net)(local_patches)
        trunk_outs = self.trunk_net(xi_norm)

        # 5. Fusion
        combined = jnp.concatenate([branch_outs, trunk_outs], axis=-1)
        
        h = nn.Dense(64)(combined)
        h = nn.tanh(h)
        u_raw = nn.Dense(1)(h).squeeze(-1)

        # Output: Control Intensity
        return self.u_max * jnp.tanh(u_raw)