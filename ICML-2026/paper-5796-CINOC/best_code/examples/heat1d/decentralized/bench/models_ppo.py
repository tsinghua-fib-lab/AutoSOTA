# models.py
import jax
import jax.numpy as jnp
import flax.linen as nn

U_MAX = 40.0
V_MAX = 2.0

class PPOActor(nn.Module):
    """Stochastic Actor for PPO. Outputs action distribution parameters."""
    n_agents: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, z, z_target, xi):
        x = jnp.concatenate([z, z_target, xi], axis=-1)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1.0)
        
        # Mean heads for u and v
        u_mean_raw = nn.Dense(self.n_agents)(x)
        v_mean_raw = nn.Dense(self.n_agents)(x)
        
        u_mean = U_MAX * jnp.tanh(u_mean_raw)
        v_mean = V_MAX * jnp.tanh(v_mean_raw)
        
        # State-independent learned standard deviation (Standard practice for continuous PPO)
        u_log_std = self.param('u_log_std', nn.initializers.zeros, (self.n_agents,))
        v_log_std = self.param('v_log_std', nn.initializers.zeros, (self.n_agents,))
        
        mean = jnp.stack([u_mean, v_mean], axis=-1)
        
        # Broadcast log_std to match the batch dimensions of mean
        batch_shape = mean.shape[:-2]
        u_log_std_b = jnp.broadcast_to(u_log_std, (*batch_shape, self.n_agents))
        v_log_std_b = jnp.broadcast_to(v_log_std, (*batch_shape, self.n_agents))
        log_std = jnp.stack([u_log_std_b, v_log_std_b], axis=-1)
        
        return mean, log_std

class PPOCritic(nn.Module):
    """State-Value Critic for PPO. Outputs V(s)."""
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, z, z_target, xi):
        x = jnp.concatenate([z, z_target, xi], axis=-1)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        
        # Single scalar value output per environment
        v = nn.Dense(1)(x)
        return v