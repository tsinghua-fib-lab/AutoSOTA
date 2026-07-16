import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import orthogonal, constant
import numpy as np

U_MAX = 5.0

class PPOActor2DKS(nn.Module):
    n_agents: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, u):
        # 1. Flatten the 2D grid
        x = u.reshape((*u.shape[:-2], -1)) 
        
        # 2. Dense representation
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        
        # 3. DPC-style Soft Normalization trick for chaotic stability
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1.0)
        
        # 4. Bounded mean for physical constraints
        mean_raw = nn.Dense(self.n_agents, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        mean = jnp.tanh(mean_raw) * U_MAX
        
        # 5. State-independent learned standard deviation (initialized to -0.5 for tight sampling)
        log_std = self.param('log_std', lambda rng, shape: jnp.full(shape, -0.5), (self.n_agents,))
        
        # Broadcast to match batch size
        batch_shape = mean.shape[:-1]
        log_std = jnp.broadcast_to(log_std, (*batch_shape, self.n_agents))
        
        return mean, log_std

class PPOCritic2DKS(nn.Module):
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, u):
        # 1. Flatten the 2D grid
        x = u.reshape((*u.shape[:-2], -1))
        
        # 2. Dense representation
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        
        # 3. Value output initialized with 1.0 to preserve variance
        v = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        
        return v