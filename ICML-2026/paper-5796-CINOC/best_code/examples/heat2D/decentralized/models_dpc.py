import jax
import jax.numpy as jnp
import flax.linen as nn

class CentralizedMLPControlNet2D(nn.Module):
    """
    Single-agent (Centralized) MLP Controller for 2D Heat Equation.
    Maps [z_curr, z_target, xi_curr] directly to all [u, v] commands.
    """
    hidden_dim: int = 256
    n_agents: int = 16
    u_max: float = 40.0
    v_max: float = 5.0

    @nn.compact
    def __call__(self, z, z_target, xi):
        # Flatten the spatial grids and coordinates (handles both batched and unbatched)
        z_flat = z.reshape((*z.shape[:-2], -1))
        zt_flat = z_target.reshape((*z_target.shape[:-2], -1))
        xi_flat = xi.reshape((*xi.shape[:-2], -1))
        
        # Concatenate global information
        x = jnp.concatenate([z_flat, zt_flat, xi_flat], axis=-1)
        
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        
        # Stability normalization
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1.0)
        
        # Tri-Heads for Forcing (u) and Velocity (vx, vy) for ALL agents simultaneously
        u_raw = nn.Dense(self.n_agents)(x)
        vx_raw = nn.Dense(self.n_agents)(x)
        vy_raw = nn.Dense(self.n_agents)(x)
        
        u_out = self.u_max * jnp.tanh(u_raw)
        vx_out = self.v_max * jnp.tanh(vx_raw)
        vy_out = self.v_max * jnp.tanh(vy_raw)
        
        # Stack velocity to form output shape: (..., n_agents, 2)
        v_out = jnp.stack([vx_out, vy_out], axis=-1)
        
        # Returning as a tuple to remain compatible with PDEDynamics.unroll_controlled
        return u_out, v_out