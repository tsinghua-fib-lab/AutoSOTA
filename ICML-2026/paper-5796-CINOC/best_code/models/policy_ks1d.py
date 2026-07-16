import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence

class ControlNet(nn.Module):
    """
    DeepONet-based Controller for KS (Static Actuators).
    
    Adaptations:
    - Output: Returns only intensity (u), no velocity.
    - Domain: Normalizes inputs assuming L=32.0
    """
    features: Sequence[int]
    L_domain: float
    u_max: float = 1.0 
    
    def setup(self):
        # Fourier frequencies for the Trunk
        self.frequencies = jnp.array([1.0, 2.0, 4.0, 8.0])

    def trunk_net(self, xi_norm):
        """Processes normalized actuator coordinates [0,1]."""
        angle = xi_norm[:, None] * self.frequencies * jnp.pi
        encoded = jnp.concatenate([jnp.sin(angle), jnp.cos(angle)], axis=-1)
        
        for feat in [32, 32]:
            encoded = nn.Dense(feat)(encoded)
            encoded = nn.tanh(encoded)
        return encoded 

    def branch_net(self, error_field, error_grad):
        combined = jnp.concatenate([error_field, error_grad], axis=-1)
        x = combined
        for feat in self.features:
            x = nn.Dense(feat)(x)
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)
        return x 

    @nn.compact
    def __call__(self, u_curr, u_target, xi_fixed):
        # 1. Normalize Coordinates
        # KS solver uses L=32.0. We normalize to [0, 1] for the network.
        xi_norm = xi_fixed / self.L_domain
        
        # 2. Calculate Error Field and Gradient
        error = u_curr - u_target
        error_grad = jnp.gradient(error) 

        # 3. Branch: Global Error Context
        branch_out = self.branch_net(error, error_grad)

        # 4. Trunk: Spatial Agent Context
        trunk_out = self.trunk_net(xi_norm)

        # 5. Fusion (DeepONet style)
        # Broadcast global context to all agents
        branch_repeated = jnp.tile(branch_out, (xi_fixed.shape[0], 1))
        combined = jnp.concatenate([branch_repeated, trunk_out], axis=-1)

        # 6. Control Head (Intensity Only)
        x = combined
        for feat in [32]:
            x = nn.Dense(feat)(x)
            x = nn.tanh(x)
        
        # Output 1 per agent
        u_raw = nn.Dense(1)(x).squeeze(-1) 
        
        # Scale to max intensity
        return self.u_max * jnp.tanh(u_raw)


class DecentralizedControlNet(nn.Module):
    """
    Local Controller for KS with Periodic Boundaries.
    """
    features: Sequence[int]
    L_domain: float
    u_max: float = 1.0
    window_size: int = 4


    def setup(self):
        self.frequencies = jnp.array([1.0, 2.0, 4.0, 8.0])

    def branch_net(self, local_patch):
        x = local_patch
        for feat in self.features:
            x = nn.Dense(feat)(x)
            # Simple normalization to keep gradients stable
            x = x / (jnp.linalg.norm(x) + 1.0) 
            x = nn.tanh(x)
        return x

    def trunk_net(self, xi_norm):
        angle = xi_norm[:, None] * self.frequencies * jnp.pi
        encoded = jnp.concatenate([jnp.sin(angle), jnp.cos(angle)], axis=-1)
        for feat in [32, 32]:
            encoded = nn.Dense(feat)(encoded)
            encoded = nn.tanh(encoded)
        return encoded 

    @nn.compact
    def __call__(self, u_curr, u_target, xi_fixed):
        xi_norm = xi_fixed / self.L_domain
        
        error = u_curr - u_target
        n_pde = u_curr.shape[0]
        
        # 1. Gradient
        error_grad = jnp.gradient(error)

        window_size = self.window_size 
        half_window = window_size // 2

        # 2. PAD with 'wrap' for PERIODIC BCs
        # This ensures agents at x=32 can 'see' neighbors at x=0
        padded_error = jnp.pad(error, (half_window, half_window), mode='wrap')
        padded_grad = jnp.pad(error_grad, (half_window, half_window), mode='wrap')

        def get_local_obs(xi_n):
            # Map normalized position back to grid index
            center_idx = jax.lax.stop_gradient((xi_n * (n_pde - 1)).astype(int)) + half_window
            start = center_idx - half_window
            
            # Slice the periodic (wrapped) fields
            p_err = jax.lax.dynamic_slice(padded_error, (start,), (window_size,))
            p_grad = jax.lax.dynamic_slice(padded_grad, (start,), (window_size,))
            
            # Resize for consistent input size
            p_err = jax.image.resize(p_err, (20,), method='bilinear')
            p_grad = jax.image.resize(p_grad, (20,), method='bilinear')
            
            return jnp.concatenate([p_err, p_grad])

        # 3. Process
        local_patches = jax.vmap(get_local_obs)(xi_norm)
        branch_outs = jax.vmap(self.branch_net)(local_patches)
        trunk_outs = self.trunk_net(xi_norm)

        combined = jnp.concatenate([branch_outs, trunk_outs], axis=-1)
        x = nn.Dense(32)(combined)
        x = nn.tanh(x)
        
        # 4. Output Head (Intensity Only)
        u_raw = nn.Dense(1)(x).squeeze(-1)

        return self.u_max * jnp.tanh(u_raw)