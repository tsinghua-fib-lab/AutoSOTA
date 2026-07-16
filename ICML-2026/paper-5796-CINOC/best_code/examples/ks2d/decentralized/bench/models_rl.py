import jax.numpy as jnp
import flax.linen as nn

U_MAX = 5.0  

class CentralizedActorKS2D(nn.Module):
    hidden_dim: int = 256
    n_agents: int = 100

    @nn.compact
    def __call__(self, z):
        # Flatten the spatial grid (N_grid, N_grid) -> (N_grid * N_grid)
        z_flat = z.reshape((*z.shape[:-2], -1))
        
        x = nn.Dense(self.hidden_dim)(z_flat)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        
        # Output directly for all agents, shape: (..., n_agents)
        out = nn.Dense(self.n_agents)(x)
        return jnp.tanh(out) * U_MAX

class CentralizedCriticKS2D(nn.Module):
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, z, actions):
        # Flatten the state
        z_flat = z.reshape((*z.shape[:-2], -1))
        
        # Concat flattened state and actions directly
        xu = jnp.concatenate([z_flat, actions], axis=-1)
        
        # Q1
        q1 = nn.Dense(self.hidden_dim)(xu)
        q1 = nn.relu(q1)
        q1 = nn.Dense(self.hidden_dim)(q1)
        q1 = nn.relu(q1)
        q1 = nn.Dense(1)(q1)

        # Q2
        q2 = nn.Dense(self.hidden_dim)(xu)
        q2 = nn.relu(q2)
        q2 = nn.Dense(self.hidden_dim)(q2)
        q2 = nn.relu(q2)
        q2 = nn.Dense(1)(q2)
        
        return q1, q2