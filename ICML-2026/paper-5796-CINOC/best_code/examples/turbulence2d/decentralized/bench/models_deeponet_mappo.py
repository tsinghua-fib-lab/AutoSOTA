import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence

class DeepONetBranch(nn.Module):
    """Processes local vorticity patches with Soft Normalization."""
    features: Sequence[int] = (16, 32)
    
    @nn.compact
    def __call__(self, x):
        # x expected shape: (..., patch_size, patch_size, 3)
        for feat in self.features:
            x = nn.Conv(feat, kernel_size=(3, 3), padding='SAME')(x)
            x = nn.relu(x)

        # Flatten spatial dimensions
        x = x.reshape((*x.shape[:-3], -1))
        
        # Soft Normalization (Critical for decay tasks near zero state)
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1.0) 
        
        x = nn.Dense(32)(x)
        x = nn.tanh(x)
        return x

class DeepONetTrunk(nn.Module):
    """Encodes absolute normalized position using fixed Fourier frequencies."""
    @nn.compact
    def __call__(self, xi_norm):
        # xi_norm expected shape: (..., 2)
        frequencies = jnp.array([1.0, 2.0, 4.0, 8.0])
        
        angle_x = xi_norm[..., 0, None] * frequencies * jnp.pi
        angle_y = xi_norm[..., 1, None] * frequencies * jnp.pi

        encoded = jnp.concatenate([
            jnp.sin(angle_x), jnp.cos(angle_x),
            jnp.sin(angle_y), jnp.cos(angle_y)
        ], axis=-1)

        for feat in [32, 32]:
            encoded = nn.Dense(feat)(encoded)
            encoded = nn.tanh(encoded)
        return encoded

class DeepONetMAPPOActor(nn.Module):
    """MAPPO Actor using DeepONet fusion."""
    n_agents: int
    u_max: float = 75.0
    
    @nn.compact
    def __call__(self, patches, xi_norm):
        branch_out = DeepONetBranch()(patches)
        trunk_out = DeepONetTrunk()(xi_norm)
        
        combined = jnp.concatenate([branch_out, trunk_out], axis=-1)
        
        h = nn.Dense(64)(combined)
        h = nn.tanh(h)
        
        # Output unbounded raw mean (we squash it in the main script)
        mean_raw = nn.Dense(1)(h)
        
        # Learnable state-independent log standard deviation
        log_std = self.param('log_std', lambda rng, shape: jnp.zeros(shape), (1, 1))
        
        # Broadcast to match mean_raw shape: (..., N_AGENTS, 1)
        batch_shape = mean_raw.shape[:-2]
        log_std_b = jnp.broadcast_to(log_std, (*batch_shape, self.n_agents, 1))
        
        return mean_raw, log_std_b

class DeepONetMAPPOCritic(nn.Module):
    """MAPPO Critic using the same DeepONet representation, outputting V."""
    n_agents: int
    
    @nn.compact
    def __call__(self, patches, xi_norm):
        branch_out = DeepONetBranch()(patches)
        trunk_out = DeepONetTrunk()(xi_norm)
        
        combined = jnp.concatenate([branch_out, trunk_out], axis=-1)
        
        # V-network Architecture
        v = nn.Dense(256)(combined)
        v = nn.relu(v)
        v = nn.Dense(256)(v)
        v = nn.relu(v)
        v = nn.Dense(1)(v)
        
        return v