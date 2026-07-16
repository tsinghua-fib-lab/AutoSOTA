import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import orthogonal, constant
import numpy as np

class CentralizedPPOActor(nn.Module):
    n_agents: int
    push_max: float
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, rho, target, xi):
        # 1. Flatten the 2D spatial grids
        rho_flat = rho.reshape((*rho.shape[:-2], -1))
        target_flat = target.reshape((*target.shape[:-2], -1))
        
        # 2. Flatten the 2D agent positions
        xi_flat = xi.reshape((*xi.shape[:-2], -1))
        
        # 3. Concatenate global information
        x = jnp.concatenate([rho_flat, target_flat, xi_flat], axis=-1)
        
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        
        # 4. DPC-style Soft Normalization trick for chaotic stability
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1.0)
        
        # 5. Bounded mean for physical constraints (vx, vy per agent)
        mean_raw = nn.Dense(self.n_agents * 2, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        mean = jnp.tanh(mean_raw) * self.push_max
        
        # Reshape to (..., n_agents, 2)
        mean = mean.reshape((*mean.shape[:-1], self.n_agents, 2))
        
        # 6. State-independent learned standard deviation
        log_std = self.param('log_std', lambda rng, shape: jnp.full(shape, -0.5), (self.n_agents, 2))
        
        # Clamp log_std to prevent Entropy Collapse (-20 to 2 is standard)
        log_std = jnp.clip(log_std, -20.0, 2.0)
        
        # Broadcast to match batch size
        batch_shape = mean.shape[:-2]
        log_std = jnp.broadcast_to(log_std, (*batch_shape, self.n_agents, 2))
        
        return mean, log_std


class CentralizedPPOCritic(nn.Module):
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, rho, target, xi):
        # 1. Flatten the 2D spatial grids
        rho_flat = rho.reshape((*rho.shape[:-2], -1))
        target_flat = target.reshape((*target.shape[:-2], -1))
        
        # 2. Flatten agent positions
        xi_flat = xi.reshape((*xi.shape[:-2], -1))
        
        x = jnp.concatenate([rho_flat, target_flat, xi_flat], axis=-1)
        
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        
        # 3. Value output initialized with 1.0 to preserve variance
        v = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        
        return v