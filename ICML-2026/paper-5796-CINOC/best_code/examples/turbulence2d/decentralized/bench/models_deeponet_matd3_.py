import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence

class DeepONetBranch(nn.Module):
    """Processes local vorticity patches with Soft Normalization."""
    features: Sequence[int] = (16, 32)
    
    @nn.compact
    def __call__(self, x):
        # x expected shape: (Batch*Agents, patch_size, patch_size, 3)
        for feat in self.features:
            x = nn.Conv(feat, kernel_size=(3, 3), padding='SAME')(x)
            x = nn.relu(x)

        # The Trick: Dynamically flatten spatial dims, preserving the leading batch/agent dim
        x = x.reshape((*x.shape[:-3], -1))
        
        # Soft Normalization (Critical for decay tasks near zero state)
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1.0) 
        
        x = nn.Dense(32)(x)
        x = nn.tanh(x)
        return x

class DeepONetTrunk(nn.Module):
    """
    Restored High-Frequency, Static Positional Encoding.
    CRITICAL for Off-Policy TD3 Stability (Prevents Representation Drift).
    """
    d_per_dim: int = 64     
    n_base: float = 1000.0  
    
    @nn.compact
    def __call__(self, xi_norm):
        # xi_norm expected shape: (Batch*Agents, 2)
        
        # 1. Generate high-frequency progression
        exponent = jnp.arange(0, self.d_per_dim, 2, dtype=jnp.float32) / self.d_per_dim
        inv_freq = 1.0 / (self.n_base ** exponent)
        
        pos_x = xi_norm[..., 0:1]
        pos_y = xi_norm[..., 1:2]
        
        angle_x = pos_x * inv_freq
        angle_y = pos_y * inv_freq
        
        # 2. Pure mathematical encoding (128 dimensions)
        # NO Dense layers here. The Critic needs a rigid coordinate frame.
        encoded = jnp.concatenate([
            jnp.sin(angle_x), jnp.cos(angle_x),
            jnp.sin(angle_y), jnp.cos(angle_y)
        ], axis=-1)
        
        return encoded

class DeepONetActor(nn.Module):
    """MATD3 Actor using MAPPO-style DeepONet fusion."""
    u_max: float = 40.0

    @nn.compact
    def __call__(self, patches, xi_norm):
        branch_out = DeepONetBranch()(patches)
        trunk_out = DeepONetTrunk()(xi_norm)
        
        combined = jnp.concatenate([branch_out, trunk_out], axis=-1)
        
        h = nn.Dense(64)(combined)
        h = nn.tanh(h)
        u_raw = nn.Dense(1)(h)
        
        # MATD3 requires deterministic scaled actions
        return self.u_max * jnp.tanh(u_raw)

class DeepONetCritic(nn.Module):
    """MATD3 Critic using MAPPO-style DeepONet fusion, outputting Q1/Q2."""
    @nn.compact
    def __call__(self, patches, xi_norm, u):
        
        branch_out = DeepONetBranch()(patches)
        trunk_out = DeepONetTrunk()(xi_norm)
        
        # Concatenate Branch, Trunk, and Action u
        xu = jnp.concatenate([branch_out, trunk_out, u], axis=-1)
        
        # Q1 Architecture
        q1 = nn.Dense(256)(xu)
        q1 = nn.relu(q1)
        q1 = nn.Dense(256)(q1)
        q1 = nn.relu(q1)
        q1 = nn.Dense(1)(q1)

        # Q2 Architecture
        q2 = nn.Dense(256)(xu)
        q2 = nn.relu(q2)
        q2 = nn.Dense(256)(q2)
        q2 = nn.relu(q2)
        q2 = nn.Dense(1)(q2)
        
        return q1, q2