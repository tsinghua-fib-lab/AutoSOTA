import jax
import jax.numpy as jnp
import flax.linen as nn

# Action scaling constraints from NS2D baseline
V_MAX = 0.8  

class CentralizedMLPControlNetNS2D(nn.Module):
    """
    Single-agent (Centralized) MLP Controller for NS2D Density Control.
    Maps [rho_grid, target_grid, agent_positions] directly to all [vx, vy] push commands.
    """
    hidden_dim: int = 256
    n_agents: int = 9

    @nn.compact
    def __call__(self, rho, rho_target, xi):
        # Flatten the 2D spatial grids: (..., Nx, Ny) -> (..., Nx*Ny)
        rho_flat = rho.reshape((*rho.shape[:-2], -1))
        target_flat = rho_target.reshape((*rho_target.shape[:-2], -1))
        
        # Flatten the 2D agent positions: (..., n_agents, 2) -> (..., n_agents*2)
        xi_flat = xi.reshape((*xi.shape[:-2], -1))
        
        # Concatenate all global information into a single 1D vector per batch
        x = jnp.concatenate([rho_flat, target_flat, xi_flat], axis=-1)
        
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        
        # Safe Normalization to prevent NaN if ReLU outputs all zeros
        x_norm = jnp.sqrt(jnp.sum(x**2, axis=-1, keepdims=True) + 1e-8)
        x = x / (x_norm + 1.0)
        
        # --- THE FIX ---
        # 1. Tiny variance scaling for the 2D weight matrices (Kernels)
        kernel_tiny = jax.nn.initializers.variance_scaling(scale=1e-4, mode="fan_in", distribution="uniform")
        
        # 2. Tiny constant for the 1D bias arrays
        bias_tiny = jax.nn.initializers.constant(1e-4)
        
        vx_raw = nn.Dense(self.n_agents, kernel_init=kernel_tiny, bias_init=bias_tiny)(x)
        vy_raw = nn.Dense(self.n_agents, kernel_init=kernel_tiny, bias_init=bias_tiny)(x)
        
        vx_out = V_MAX * jnp.tanh(vx_raw)
        vy_out = V_MAX * jnp.tanh(vy_raw)
        
        # Stack to form output shape: (..., n_agents, 2)
        return jnp.stack([vx_out, vy_out], axis=-1)