import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, variance_scaling

class FCNActorPPO(nn.Module):
    n_agents: int = 64
    u_max: float = 75.0

    @nn.compact
    def __call__(self, z):
        # State Feature Extraction (Downsample 64x64 -> 8x8)
        x = jnp.expand_dims(z, -1) 
        
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        
        x = nn.Conv(features=128, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        
        # Mean action projection (Tiny initialization to start near 0.0)
        init_fn = variance_scaling(scale=0.01, mode="fan_in", distribution="uniform")
        mean_grid = nn.Conv(features=1, kernel_size=(3, 3), strides=(1, 1), padding='SAME', kernel_init=init_fn)(x)
        
        # Flatten the spatial 8x8x1 grid into the flat 64 action vector
        mean_flat = mean_grid.reshape((*mean_grid.shape[:-3], self.n_agents))
        
        # Squash strictly into [-U_MAX, U_MAX]
        mean = jnp.tanh(mean_flat) * self.u_max 
        
        # Learned standard deviation (log space for stability)
        log_std = self.param('log_std', constant(0.0), (self.n_agents,))
        
        # Prevent Entropy Collapse
        log_std = jnp.clip(log_std, -20.0, 2.0)
        
        return mean, log_std

class FCNCriticPPO(nn.Module):
    @nn.compact
    def __call__(self, z):
        # PPO Critic only evaluates the state: V(s)
        x = jnp.expand_dims(z, -1)
        
        # State Feature Extraction
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        
        x = nn.Conv(features=128, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x) 
        
        # Value Processing Network
        v = nn.Conv(features=256, kernel_size=(3, 3), padding='SAME')(x)
        v = nn.LayerNorm()(v)
        v = nn.relu(v)
        
        v = nn.Conv(features=256, kernel_size=(3, 3), padding='SAME')(v)
        v = nn.LayerNorm()(v)
        v = nn.relu(v)
        
        # Map down to exactly 1 Value per grid cell (Shape: 8x8x1)
        v = nn.Conv(features=1, kernel_size=(3, 3), padding='SAME')(v)
        
        # Global Average Pooling: Average the local values into one Global Value
        return jnp.mean(v, axis=(-3, -2))