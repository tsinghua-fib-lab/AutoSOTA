import jax
import jax.numpy as jnp
from flax import linen as nn

class CentralizedMLPControlNet(nn.Module):
    """
    Single-agent (Centralized) MLP Controller for FKPP Trajectory Tracking.
    Maps [current_state, target_state, agent_positions] directly to all [forcing, velocity] commands.
    Based on standard TD3 Actor architecture.
    """
    hidden_dim: int = 256
    n_agents: int = 20
    u_max: float = 40.0
    v_max: float = 2.0

    @nn.compact
    def __call__(self, z, z_target, xi):
        # Concatenate global information
        x = jnp.concatenate([z, z_target, xi], axis=-1)
        
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        
        # Stability normalization
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1.0)
        
        # Dual Heads for Forcing (u) and Velocity (v) for ALL agents simultaneously
        u_raw = nn.Dense(self.n_agents)(x)
        v_raw = nn.Dense(self.n_agents)(x)
        
        u_out = self.u_max * jnp.tanh(u_raw)
        v_out = self.v_max * jnp.tanh(v_raw)
        
        # Returning as a tuple to remain compatible with PDEDynamics.unroll_controlled
        return u_out, v_out