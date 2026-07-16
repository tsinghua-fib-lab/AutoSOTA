import jax
import jax.numpy as jnp
import flax.linen as nn

class CentralizedMLPControlNet1D_KS(nn.Module):
    """
    Single-agent (Centralized) MLP Controller for 1D Kuramoto-Sivashinsky.
    Maps [u_curr, u_target, xi_curr] directly to forcing commands [u_ctrl].
    Actuators are fixed, so no velocity is output.
    """
    hidden_dim: int = 256
    n_agents: int = 8
    u_max: float = 1.0

    @nn.compact
    def __call__(self, u, u_target, xi):
        # Flatten the spatial grids and coordinates (handles batched and unbatched)
        u_flat = u.reshape((*u.shape[:-1], -1))
        ut_flat = u_target.reshape((*u_target.shape[:-1], -1))
        xi_flat = xi.reshape((*xi.shape[:-1], -1))
        
        # Concatenate global information
        x = jnp.concatenate([u_flat, ut_flat, xi_flat], axis=-1)
        
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        
        # Stability normalization
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1.0)
        
        # Zero-initialize the final layer so the policy starts exactly as the uncontrolled baseline.
        # This prevents exploding gradients in the chaotic KS BPTT rollout.
        zero_init = nn.initializers.zeros
        
        # Single Head for Forcing (u) for ALL agents simultaneously
        u_raw = nn.Dense(self.n_agents, kernel_init=zero_init, bias_init=zero_init)(x)
        
        u_out = self.u_max * jnp.tanh(u_raw)
        
        # Return only the forcing (no velocity tuple) for the fixed KS dynamics
        return u_out