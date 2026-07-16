import jax
import jax.numpy as jnp
import flax.linen as nn

class CentralizedActor(nn.Module):
    """
    Single-agent Actor.
    Maps the full PDE field (N_grid) to all actuator intensities (N_agents).
    """
    hidden_dim: int = 256
    n_agents: int = 8
    u_max: float = 1.0

    @nn.compact
    def __call__(self, u_field):
        # u_field shape: (N_grid,)
        x = nn.Dense(self.hidden_dim)(u_field)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        
        # Output all 8 control signals at once
        out = nn.Dense(self.n_agents)(x)
        return jnp.tanh(out) * self.u_max

class CentralizedCritic(nn.Module):
    """
    Single-agent Critic.
    Maps [full_field, full_action] to a single global Q-value.
    """
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, u_field, actions):
        xu = jnp.concatenate([u_field, actions], axis=-1)
        
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