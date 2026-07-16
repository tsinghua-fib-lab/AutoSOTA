import jax
import jax.numpy as jnp
import flax.linen as nn

U_MAX = 5.0

class MAPPOActor2DKS(nn.Module):
    n_agents: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(self.hidden_dim)(obs)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        
        # Soft Normalization trick to prevent exploding activations
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1.0)
        
        mean_raw = nn.Dense(1)(x)
        # Bound the mean to prevent the center of the distribution from drifting into chaos
        mean = jnp.tanh(mean_raw) * U_MAX 
        
        # Initialize std dev to be small (-0.5) so samples stay tight around the bounded mean
        log_std = self.param('log_std', lambda rng, shape: jnp.full(shape, -0.5), (self.n_agents, 1))
        
        batch_shape = mean.shape[:-2]
        log_std_b = jnp.broadcast_to(log_std, (*batch_shape, self.n_agents, 1))
        
        return mean, log_std_b

class MAPPOCritic2DKS(nn.Module):
    n_agents: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, global_u):
        x = global_u.reshape((*global_u.shape[:-2], -1))
        
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        
        v = nn.Dense(self.n_agents)(x) 
        return v