import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import orthogonal, constant
import numpy as np

class MAPPOActorNS2D(nn.Module):
    n_agents: int
    push_max: float
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, obs):
        # obs shape is expected to be (..., n_agents, obs_dim)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        
        # Soft Normalization trick to prevent exploding activations in decentralized patches
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1.0)
        
        # 2 Actions per agent (vx, vy)
        mean_raw = nn.Dense(2, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        mean = jnp.tanh(mean_raw) * self.push_max
        
        # Initialize std dev tightly to prevent chaotic drifts (-0.5 init)
        log_std = self.param('log_std', lambda rng, shape: jnp.full(shape, -0.5), (self.n_agents, 2))
        
        # Clamp log_std to prevent Entropy Collapse (-20 to 2 is standard)
        log_std = jnp.clip(log_std, -20.0, 2.0)
        
        # Broadcast to match batch size
        batch_shape = mean.shape[:-2]
        log_std_b = jnp.broadcast_to(log_std, (*batch_shape, self.n_agents, 2))
        
        return mean, log_std_b


class MAPPOCriticNS2D(nn.Module):
    n_agents: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, rho, target, xi):
        # 1. Flatten the 2D spatial grids for Centralized view
        rho_flat = rho.reshape((*rho.shape[:-2], -1))
        target_flat = target.reshape((*target.shape[:-2], -1))
        
        # 2. Flatten all agent positions
        xi_flat = xi.reshape((*xi.shape[:-2], -1))
        
        x = jnp.concatenate([rho_flat, target_flat, xi_flat], axis=-1)
        
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        
        # Output a unique value for each specific agent based on the global state
        v = nn.Dense(self.n_agents, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return v