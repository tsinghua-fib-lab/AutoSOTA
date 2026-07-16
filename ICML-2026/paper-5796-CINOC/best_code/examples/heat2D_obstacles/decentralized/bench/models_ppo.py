import jax
import jax.numpy as jnp
import flax.linen as nn

U_MAX = 40.0
V_MAX = 5.0

class PPOActor2D(nn.Module):
    """
    Centralized Stochastic Actor for 2D PPO. 
    Observes the full 2D grid and outputs distribution params for ALL agents.
    """
    n_agents: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, z, z_target, xi):
        # z and z_target are 2D grids (N, N). We must flatten them.
        z_flat = z.reshape(z.shape[:-2] + (-1,))
        zt_flat = z_target.reshape(z_target.shape[:-2] + (-1,))
        xi_flat = xi.reshape(xi.shape[:-2] + (-1,)) # xi is (n_agents, 2)
        
        x = jnp.concatenate([z_flat, zt_flat, xi_flat], axis=-1)
        
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1.0)
        
        # Mean heads for u, vx, vy for ALL agents
        u_mean_raw = nn.Dense(self.n_agents)(x)
        vx_mean_raw = nn.Dense(self.n_agents)(x)
        vy_mean_raw = nn.Dense(self.n_agents)(x)
        
        u_mean = U_MAX * jnp.tanh(u_mean_raw)
        vx_mean = V_MAX * jnp.tanh(vx_mean_raw)
        vy_mean = V_MAX * jnp.tanh(vy_mean_raw)
        
        # Shape: (..., n_agents, 3)
        mean = jnp.stack([u_mean, vx_mean, vy_mean], axis=-1)
        
        # State-independent learned standard deviation
        u_log_std = self.param('u_log_std', nn.initializers.zeros, (self.n_agents,))
        vx_log_std = self.param('vx_log_std', nn.initializers.zeros, (self.n_agents,))
        vy_log_std = self.param('vy_log_std', nn.initializers.zeros, (self.n_agents,))
        
        log_std_raw = jnp.stack([u_log_std, vx_log_std, vy_log_std], axis=-1) # (n_agents, 3)
        
        # Broadcast to match batch size
        batch_shape = mean.shape[:-2]
        log_std = jnp.broadcast_to(log_std_raw, (*batch_shape, self.n_agents, 3))
        
        return mean, log_std

class PPOCritic2D(nn.Module):
    """Centralized State-Value Critic for 2D PPO. Outputs V(s)."""
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, z, z_target, xi):
        z_flat = z.reshape(z.shape[:-2] + (-1,))
        zt_flat = z_target.reshape(z_target.shape[:-2] + (-1,))
        xi_flat = xi.reshape(xi.shape[:-2] + (-1,))
        
        x = jnp.concatenate([z_flat, zt_flat, xi_flat], axis=-1)
        
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        
        # Single scalar value output per environment
        v = nn.Dense(1)(x)
        return v