import jax
import jax.numpy as jnp
import flax.linen as nn

U_MAX = 40.0
V_MAX = 2.0

class MAPPOActor(nn.Module):
    """Decentralized Actor for MAPPO. Outputs action distribution parameters."""
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
        v_mean_raw = nn.Dense(1)(x)
        
        u_mean = U_MAX * jnp.tanh(u_mean_raw.squeeze(-1))
        v_mean = V_MAX * jnp.tanh(v_mean_raw.squeeze(-1))
        
        # State-independent learned standard deviation
        u_log_std = self.param('u_log_std', nn.initializers.zeros, (self.n_agents,))
        v_log_std = self.param('v_log_std', nn.initializers.zeros, (self.n_agents,))
        
        mean = jnp.stack([u_mean, v_mean], axis=-1)
        
        # Broadcast log_std to match the batch dimensions of mean
        batch_shape = mean.shape[:-2]
        u_log_std_b = jnp.broadcast_to(u_log_std, (*batch_shape, self.n_agents))
        v_log_std_b = jnp.broadcast_to(v_log_std, (*batch_shape, self.n_agents))
        log_std = jnp.stack([u_log_std_b, v_log_std_b], axis=-1)
        
        return mean, log_std

class MAPPOCritic(nn.Module):
    """Centralized State-Value Critic for MAPPO. Outputs V(s) per agent."""
    n_agents: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, global_z, global_target, global_xi):
        # Flatten the global information for the critic
        # global_z: (batch, N_GRID), global_target: (batch, N_GRID), global_xi: (batch, N_AGENTS)
        x = jnp.concatenate([global_z, global_target, global_xi], axis=-1)
        
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        
        # Output one value per agent based on the global state
        v = nn.Dense(self.n_agents)(x) 
        return v