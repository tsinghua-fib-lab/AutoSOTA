import jax
import jax.numpy as jnp
import flax.linen as nn

U_MAX = 40.0
V_MAX = 5.0

class PPOActor1D(nn.Module):
    n_agents: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, u):
        x = nn.Dense(self.hidden_dim)(u)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        
        # Mean bounded to [-1, 1] for KS1D control
        mean_raw = nn.Dense(self.n_agents)(x)
        mean = jnp.tanh(mean_raw)
        
        # State-independent learned standard deviation
        log_std = self.param('log_std', nn.initializers.zeros, (self.n_agents,))
        
        # Broadcast to match batch size
        batch_shape = mean.shape[:-1]
        log_std = jnp.broadcast_to(log_std, (*batch_shape, self.n_agents))
        
        return mean, log_std

class PPOCritic1D(nn.Module):
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, u):
        x = nn.Dense(self.hidden_dim)(u)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        
        v = nn.Dense(1)(x)
        return v