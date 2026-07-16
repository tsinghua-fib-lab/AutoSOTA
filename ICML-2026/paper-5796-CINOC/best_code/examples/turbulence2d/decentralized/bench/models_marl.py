import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence

class CNNFeatureExtractor(nn.Module):
    """Matches the branch_net from DecentralizedTurbulenceNet"""
    features: Sequence[int] = (16, 32)
    
    @nn.compact
    def __call__(self, x):
        # x expected shape: (..., 20, 20, 3)
        for feat in self.features:
            x = nn.Conv(feat, kernel_size=(3, 3), padding='SAME')(x)
            x = nn.relu(x)

        # Flatten spatial dimensions
        x = x.reshape((*x.shape[:-3], -1))
        
        # Soft Normalization (from DPC)
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1.0) 
        
        x = nn.Dense(32)(x)
        x = nn.tanh(x)
        return x

class MARLActor2DKS(nn.Module):
    u_max: float = 40.0

    @nn.compact
    def __call__(self, patches, pos_enc):
        """
        patches: (..., 20, 20, 3) 
        pos_enc: (..., pe_dim)
        """
        branch_out = CNNFeatureExtractor()(patches)
        combined = jnp.concatenate([branch_out, pos_enc], axis=-1)
        
        h = nn.Dense(64)(combined)
        h = nn.tanh(h)
        u_raw = nn.Dense(1)(h)
        
        return self.u_max * jnp.tanh(u_raw)

class MARLCritic2DKS(nn.Module):
    @nn.compact
    def __call__(self, patches, pos_enc, u):
        if u.ndim == patches.ndim - 3:
            u = jnp.expand_dims(u, axis=-1)

        branch_out = CNNFeatureExtractor()(patches)
        xu = jnp.concatenate([branch_out, pos_enc, u], axis=-1)
        
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