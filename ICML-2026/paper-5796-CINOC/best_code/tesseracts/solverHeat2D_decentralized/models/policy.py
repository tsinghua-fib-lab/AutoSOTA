import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence

class Heat2DControlNet(nn.Module):
    """
    2D Heat Equation Controller (Centralized).

    Architecture:
    - Branch: CNN to process 2D error field
    - Trunk: Fourier encoding of 2D actuator positions
    - Fusion: Broadcast + concatenate
    - Heads: Separate outputs for u (forcing) and v (velocity)
    """
    features: Sequence[int] = (16, 32)  # CNN feature channels
    u_max: float = 40.0  # Max forcing intensity
    v_max: float = 5.0   # Max velocity (increased for better coverage)

    def setup(self):
        self.frequencies = jnp.array([1.0, 2.0, 4.0, 8.0])

    def branch_net(self, error, error_grad_x, error_grad_y):
        """
        CNN branch for 2D spatial error processing.

        Args:
            error: (N, N) - pointwise error
            error_grad_x: (N, N) - x-gradient
            error_grad_y: (N, N) - y-gradient

        Returns:
            Global context vector (feature_dim,)
        """
        # Stack into 3-channel input: [error, ∂error/∂x, ∂error/∂y]
        x = jnp.stack([error, error_grad_x, error_grad_y], axis=-1)  # (N, N, 3)

        # Convolutional layers (NO POOLING - preserves spatial resolution)
        for feat in self.features:
            x = nn.Conv(feat, kernel_size=(3, 3), padding='SAME')(x)
            x = nn.relu(x)

        # Flatten and normalize (analogous to decentralized)
        x = x.reshape(-1)
        x = x / (jnp.linalg.norm(x) + 1.0)  # L2 normalization like decentralized
        x = nn.Dense(64)(x)
        x = nn.tanh(x)

        return x

    def trunk_net(self, xi):
        """
        Fourier encoding for 2D actuator positions.

        Args:
            xi: (M, 2) - actuator positions [x, y]

        Returns:
            (M, trunk_dim) encoded positions
        """
        # Fourier features for each coordinate
        # xi[:, 0] = x, xi[:, 1] = y
        angle_x = xi[:, 0, None] * self.frequencies * jnp.pi
        angle_y = xi[:, 1, None] * self.frequencies * jnp.pi

        # Concatenate sin/cos for both dimensions
        encoded = jnp.concatenate([
            jnp.sin(angle_x), jnp.cos(angle_x),
            jnp.sin(angle_y), jnp.cos(angle_y)
        ], axis=-1)  # (M, 32) = (M, 4*4*2)

        # Process through MLP
        for feat in [64, 64]:
            encoded = nn.Dense(feat)(encoded)
            encoded = nn.tanh(encoded)

        return encoded

    @nn.compact
    def __call__(self, z_curr, z_target, xi_curr):
        """
        Forward pass.

        Args:
            z_curr: (N, N) current state
            z_target: (N, N) target state
            xi_curr: (M, 2) actuator positions

        Returns:
            u: (M,) forcing intensities
            v: (M, 2) actuator velocities
        """
        # Clip positions to domain
        xi_curr = jnp.clip(xi_curr, 0.0, 1.0)

        # Compute error and gradient
        error = z_curr - z_target
        error_grad = jnp.gradient(error)  # Returns (grad_y, grad_x) due to indexing='ij'
        # Swap to (grad_x, grad_y) for consistency
        error_grad_x = error_grad[1]
        error_grad_y = error_grad[0]

        # Branch: global context
        branch_out = self.branch_net(error, error_grad_x, error_grad_y)

        # Trunk: position encoding
        trunk_out = self.trunk_net(xi_curr)

        # Fusion: broadcast branch to all agents
        branch_repeated = jnp.tile(branch_out, (xi_curr.shape[0], 1))
        combined = jnp.concatenate([branch_repeated, trunk_out], axis=-1)

        # Shared hidden layer
        h = nn.Dense(64)(combined)
        h = nn.tanh(h)

        # Output heads
        u_raw = nn.Dense(1)(h).squeeze(-1)  # (M,) scalar forcing
        v_raw = nn.Dense(2)(h)               # (M, 2) 2D velocity

        u = self.u_max * jnp.tanh(u_raw)
        v = self.v_max * jnp.tanh(v_raw)

        return u, v


class DecentralizedHeat2DControlNet(nn.Module):
    """
    Decentralized 2D Heat Controller.

    Each agent perceives a local patch around its position.
    """
    features: Sequence[int] = (16, 32)
    u_max: float = 40.0
    v_max: float = 5.0  # Max velocity (increased for better coverage)
    patch_size: int = 12  # Local window size (12×12 patch)

    def setup(self):
        self.frequencies = jnp.array([1.0, 2.0, 4.0, 8.0])

    def extract_local_patch(self, field, xi, n_grid):
        """
        Extract local patch around actuator position.

        Args:
            field: (N, N) full field
            xi: (2,) single actuator position [x, y]
            n_grid: Grid size N

        Returns:
            patch: (patch_size, patch_size) local observation
        """
        # Convert position to grid index
        i = jnp.clip((xi[0] * (n_grid - 1)).astype(int), 0, n_grid-1)
        j = jnp.clip((xi[1] * (n_grid - 1)).astype(int), 0, n_grid-1)

        half_patch = self.patch_size // 2

        # Pad field to handle boundaries
        padded_field = jnp.pad(field, ((half_patch, half_patch),
                                        (half_patch, half_patch)),
                                mode='edge')

        # Extract patch using dynamic_slice
        # Indices are already offset by padding
        start_i = i
        start_j = j

        patch = jax.lax.dynamic_slice(
            padded_field,
            (start_i, start_j),
            (self.patch_size, self.patch_size)
        )

        return patch

    def branch_net(self, local_patch):
        """Process local 3-channel patch."""
        x = local_patch  # (patch_size, patch_size, 3)

        for feat in self.features:
            x = nn.Conv(feat, kernel_size=(3, 3), padding='SAME')(x)
            x = nn.relu(x)

        x = x.reshape(-1)
        x = x / (jnp.linalg.norm(x) + 1.0)  # Local normalization
        x = nn.Dense(32)(x)
        x = nn.tanh(x)

        return x

    def trunk_net(self, xi):
        """Same as centralized."""
        angle_x = xi[:, 0, None] * self.frequencies * jnp.pi
        angle_y = xi[:, 1, None] * self.frequencies * jnp.pi

        encoded = jnp.concatenate([
            jnp.sin(angle_x), jnp.cos(angle_x),
            jnp.sin(angle_y), jnp.cos(angle_y)
        ], axis=-1)

        for feat in [32, 32]:
            encoded = nn.Dense(feat)(encoded)
            encoded = nn.tanh(encoded)

        return encoded

    @nn.compact
    def __call__(self, z_curr, z_target, xi_curr):
        error = z_curr - z_target
        error_grad = jnp.gradient(error)
        error_grad_x = error_grad[1]  # Swap to x, y
        error_grad_y = error_grad[0]

        n_grid = z_curr.shape[0]

        # Extract local patches for each agent
        def get_local_obs(xi_single):
            patch_error = self.extract_local_patch(error, xi_single, n_grid)
            patch_grad_x = self.extract_local_patch(error_grad_x, xi_single, n_grid)
            patch_grad_y = self.extract_local_patch(error_grad_y, xi_single, n_grid)

            # Stack channels
            return jnp.stack([patch_error, patch_grad_x, patch_grad_y], axis=-1)

        local_patches = jax.vmap(get_local_obs)(xi_curr)  # (M, patch, patch, 3)

        # Process each patch independently
        branch_outs = jax.vmap(self.branch_net)(local_patches)

        # Trunk encoding
        trunk_outs = self.trunk_net(xi_curr)

        # Fusion and output
        combined = jnp.concatenate([branch_outs, trunk_outs], axis=-1)
        h = nn.Dense(64)(combined)
        h = nn.tanh(h)

        u_raw = nn.Dense(1)(h).squeeze(-1)
        v_raw = nn.Dense(2)(h)

        return self.u_max * jnp.tanh(u_raw), self.v_max * jnp.tanh(v_raw)