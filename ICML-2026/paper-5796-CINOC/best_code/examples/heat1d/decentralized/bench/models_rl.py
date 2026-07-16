import jax
import jax.numpy as jnp
import flax.linen as nn

# Action scaling constraints from DPC
U_MAX = 40.0
V_MAX = 2.0

class CentralizedActor(nn.Module):
    """
    Single-agent Actor for FKPP Trajectory Tracking.
    Maps the [full_state, target_state, agent_positions] to all [forcing, velocity] commands.
    """
    hidden_dim: int = 256
    n_agents: int = 20

    @nn.compact
    def __call__(self, z, z_target, xi):
        # Concatenate global information
        x = jnp.concatenate([z, z_target, xi], axis=-1)
        
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1.0)
        
        # Dual Heads for Forcing (u) and Velocity (v) for ALL agents simultaneously
        u_raw = nn.Dense(self.n_agents)(x)
        v_raw = nn.Dense(self.n_agents)(x)
        
        u_out = U_MAX * jnp.tanh(u_raw)
        v_out = V_MAX * jnp.tanh(v_raw)
        
        # Stack to form output shape: (..., n_agents, 2)
        return jnp.stack([u_out, v_out], axis=-1)

class CentralizedCritic(nn.Module):
    """
    Single-agent Critic.
    Maps [full_state, target_state, agent_positions, full_actions] to a single global Q-value.
    """
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, z, z_target, xi, actions):
        # actions is shape (..., n_agents, 2). Flatten it for the centralized critic.
        batch_shape = actions.shape[:-2]
        actions_flat = actions.reshape((*batch_shape, -1))
        
        xu = jnp.concatenate([z, z_target, xi, actions_flat], axis=-1)
        
        # Q1
        q1 = nn.Dense(self.hidden_dim)(xu)
        q1 = nn.relu(q1)
        q1 = nn.Dense(self.hidden_dim)(q1)
        q1 = nn.relu(q1)
        q1 = nn.Dense(1)(q1)

        # Q2
        q2 = nn.Dense(self.hidden_dim)(xu)
        q2 = nn.relu(q2)
        q2 = nn.Dense(self.hidden_dim)(q2)
        q2 = nn.relu(q2)
        q2 = nn.Dense(1)(q2)
        
        return q1, q2