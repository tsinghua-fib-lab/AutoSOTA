import jax
import jax.numpy as jnp
import flax.linen as nn

class MARLActor(nn.Module):
    """
    Standard Decentralized Actor.
    Maps concatenated [y_i, PE(p_i), mu] directly to action u_i.
    """
    hidden_dim: int = 256
    action_dim: int = 1
    u_max: float = 1.0

    @nn.compact
    def __call__(self, x):
        # x is the concatenated vector [patch, mu, pe]
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        
        # DPC-style Normalization trick for stability
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1.0)
        
        out = nn.Dense(self.action_dim)(x)
        return jnp.tanh(out) * self.u_max

class MARLCritic(nn.Module):
    """
    Standard Decentralized Critic (Twin Q-networks).
    Maps [y_i, PE(p_i), mu, u_i] to Q-value.
    """
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, x, u):
        xu = jnp.concatenate([x, u], axis=-1)
        
        # Q1 Architecture
        q1 = nn.Dense(self.hidden_dim)(xu)
        q1 = nn.relu(q1)
        q1 = nn.Dense(self.hidden_dim)(q1)
        q1 = nn.relu(q1)
        q1 = nn.Dense(1)(q1)

        # Q2 Architecture
        q2 = nn.Dense(self.hidden_dim)(xu)
        q2 = nn.relu(q2)
        q2 = nn.Dense(self.hidden_dim)(q2)
        q2 = nn.relu(q2)
        q2 = nn.Dense(1)(q2)
        
        return q1, q2