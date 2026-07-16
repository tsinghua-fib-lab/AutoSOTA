import jax
import jax.numpy as jnp
import flax.linen as nn

U_MAX = 40.0
V_MAX = 5.0

class MAPPOActor2D(nn.Module):
    """Decentralized Actor for MAPPO in 2D. Outputs 3D action distribution parameters."""
    n_agents: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, obs):
        # obs shape: (batch, n_agents, obs_dim)
        x = nn.Dense(self.hidden_dim)(obs)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1.0)
        
        # Mean heads for u and v (output per agent)
        u_mean_raw = nn.Dense(1)(x)
        v_mean_raw = nn.Dense(2)(x) # 2D Velocity Output (vx, vy)
        
        u_mean = U_MAX * jnp.tanh(u_mean_raw.squeeze(-1))
        v_mean = V_MAX * jnp.tanh(v_mean_raw)
        
        # State-independent learned standard deviation
        u_log_std = self.param('u_log_std', nn.initializers.zeros, (self.n_agents,))
        vx_log_std = self.param('vx_log_std', nn.initializers.zeros, (self.n_agents,))
        vy_log_std = self.param('vy_log_std', nn.initializers.zeros, (self.n_agents,))
        
        # Shape: (..., n_agents, 3)
        mean = jnp.concatenate([u_mean[..., None], v_mean], axis=-1)
        
        # Broadcast log_std to match the batch dimensions of mean
        batch_shape = mean.shape[:-2]
        
        u_log_std_b = jnp.broadcast_to(u_log_std, (*batch_shape, self.n_agents))
        vx_log_std_b = jnp.broadcast_to(vx_log_std, (*batch_shape, self.n_agents))
        vy_log_std_b = jnp.broadcast_to(vy_log_std, (*batch_shape, self.n_agents))
        
        log_std = jnp.stack([u_log_std_b, vx_log_std_b, vy_log_std_b], axis=-1)
        
        return mean, log_std

class MAPPOCritic2D(nn.Module):
    """Centralized State-Value Critic for 2D MAPPO. Outputs V(s) per agent."""
    n_agents: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, global_z, global_target, global_xi):
        # Flatten the global 2D grids and 2D coordinates for the critic
        # global_z: (batch, N, N), global_target: (batch, N, N), global_xi: (batch, n_agents, 2)
        z_flat = global_z.reshape(global_z.shape[:-2] + (-1,))
        zt_flat = global_target.reshape(global_target.shape[:-2] + (-1,))
        xi_flat = global_xi.reshape(global_xi.shape[:-2] + (-1,))
        
        x = jnp.concatenate([z_flat, zt_flat, xi_flat], axis=-1)
        
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        
        # Output one value per agent based on the global state
        v = nn.Dense(self.n_agents)(x) 
        return v