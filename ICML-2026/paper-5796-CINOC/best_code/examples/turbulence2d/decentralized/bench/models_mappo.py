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

class MAPPOActorTurb(nn.Module):
    n_agents: int
    u_max: float = 75.0
    
    @nn.compact
    def __call__(self, patches, pos_enc):
        """
        patches: (..., N_AGENTS, 20, 20, 3) 
        pos_enc: (..., N_AGENTS, pe_dim)
        """
        branch_out = CNNFeatureExtractor()(patches)
        combined = jnp.concatenate([branch_out, pos_enc], axis=-1)
        
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

class MAPPOCriticTurb(nn.Module):
    n_agents: int
    
    @nn.compact
    def __call__(self, patches, pos_enc):
        """
        patches: (..., N_AGENTS, 20, 20, 3) 
        pos_enc: (..., N_AGENTS, pe_dim)
        
        Decentralized Value network to match MATD3's Q-network logic.
        """
        branch_out = CNNFeatureExtractor()(patches)
        combined = jnp.concatenate([branch_out, pos_enc], axis=-1)
        
        # V-network Architecture
        v = nn.Dense(256)(combined)
        v = nn.relu(v)
        v = nn.Dense(256)(v)
        v = nn.relu(v)
        v = nn.Dense(1)(v)
        
        return v