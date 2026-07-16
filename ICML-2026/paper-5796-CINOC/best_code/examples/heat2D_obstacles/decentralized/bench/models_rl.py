import jax
import jax.numpy as jnp
import flax.linen as nn

# Action scaling constraints (Matches DPC 2D Heat config)
U_MAX = 40.0
V_MAX = 5.0  

class CentralizedActor2D(nn.Module):
    """
    Single-agent Actor for 2D Heat Equation Trajectory Tracking.
    Maps the [full_state_2d, target_state_2d, agent_positions_2d] to all [u, vx, vy] commands.
    """
    hidden_dim: int = 256
    n_agents: int = 16

    @nn.compact
    def __call__(self, z, z_target, xi):
        # Flatten the spatial grids and coordinates so they can pass into the MLP
        z_flat = z.reshape((*z.shape[:-2], -1))
        zt_flat = z_target.reshape((*z_target.shape[:-2], -1))
        xi_flat = xi.reshape((*xi.shape[:-2], -1))
        
        # Concatenate global information
        x = jnp.concatenate([z_flat, zt_flat, xi_flat], axis=-1)
        
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        
        # Normalization trick for stability
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1.0)
        
        # Tri-Heads for Forcing (u) and Velocity (vx, vy) for ALL agents simultaneously
        u_raw = nn.Dense(self.n_agents)(x)
        vx_raw = nn.Dense(self.n_agents)(x)
        vy_raw = nn.Dense(self.n_agents)(x)
        
        u_out = U_MAX * jnp.tanh(u_raw)
        vx_out = V_MAX * jnp.tanh(vx_raw)
        vy_out = V_MAX * jnp.tanh(vy_raw)
        
        # Stack to form output shape: (..., n_agents, 3)
        return jnp.stack([u_out, vx_out, vy_out], axis=-1)

class CentralizedCritic2D(nn.Module):
    """
    Single-agent Critic.
    Maps [full_state_2d, target_state_2d, agent_positions_2d, full_actions] to a single Q-value.
    """
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, z, z_target, xi, actions):
        # Flatten all inputs
        z_flat = z.reshape((*z.shape[:-2], -1))
        zt_flat = z_target.reshape((*z_target.shape[:-2], -1))
        xi_flat = xi.reshape((*xi.shape[:-2], -1))
        
        # actions is shape (..., n_agents, 3). Flatten it.
        batch_shape = actions.shape[:-2]
        actions_flat = actions.reshape((*batch_shape, -1))
        
        xu = jnp.concatenate([z_flat, zt_flat, xi_flat, actions_flat], axis=-1)
        
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