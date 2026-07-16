import jax.numpy as jnp
import flax.linen as nn

# Action scaling constraint (Matches DecentralizedKS2DControlNet config)
U_MAX = 5.0  

class MARLActor2DKS(nn.Module):
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        
        # DPC-style Soft Normalization trick for stability (+ 1.0)
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1.0)
        
        u_raw = nn.Dense(1)(x)
        u_out = U_MAX * jnp.tanh(u_raw)
        return u_out

class MARLCritic2DKS(nn.Module):
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, x, u):
        if u.ndim == x.ndim - 1:
            u = jnp.expand_dims(u, axis=-1)

        xu = jnp.concatenate([x, u], axis=-1)
        
        q1 = nn.Dense(self.hidden_dim)(xu)
        q1 = nn.relu(q1)
        q1 = nn.Dense(self.hidden_dim)(q1)
        q1 = nn.relu(q1)
        q1 = nn.Dense(1)(q1)

        q2 = nn.Dense(self.hidden_dim)(xu)
        q2 = nn.relu(q2)
        q2 = nn.Dense(self.hidden_dim)(q2)
        q2 = nn.relu(q2)
        q2 = nn.Dense(1)(q2)
        
        return q1, q2