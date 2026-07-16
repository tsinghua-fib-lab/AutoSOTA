import jax.numpy as jnp
import flax.linen as nn

# Action scaling constraints (Matches DPC 2D Heat config)
U_MAX = 40.0
V_MAX = 5.0  

class MARLActor2D(nn.Module):
    """
    Standard Decentralized Actor adapted for 2D spaces and 3D outputs.
    Maps concatenated [y_local, mu, PE_2d(x, y)] directly to action [u, vx, vy].
    """
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        
        # DPC-style Normalization trick for stability
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1.0)
        
        # Dual Heads for Forcing (u) and Velocity (v)
        u_raw = nn.Dense(1)(x)
        v_raw = nn.Dense(2)(x) # 2D Velocity Output (vx, vy)
        
        u_out = U_MAX * jnp.tanh(u_raw)
        v_out = V_MAX * jnp.tanh(v_raw)
        
        # Shape: (..., 3) -> [u, vx, vy]
        return jnp.concatenate([u_out, v_out], axis=-1)


class MATD3Critic2D(nn.Module):
    """
    Centralized Critic network for MATD3 estimating Q-values.
    Takes joint observations and joint actions, outputs individual Q-values per agent.
    """
    n_agents: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, joint_x, joint_u):
        # joint_x: (batch, N_AGENTS * obs_dim)
        # joint_u: (batch, N_AGENTS * act_dim)
        xu = jnp.concatenate([joint_x, joint_u], axis=-1)
        
        # Q1 Architecture
        q1 = nn.Dense(self.hidden_dim)(xu)
        q1 = nn.relu(q1)
        q1 = nn.Dense(self.hidden_dim)(q1)
        q1 = nn.relu(q1)
        q1 = nn.Dense(self.n_agents)(q1)

        # Q2 Architecture
        q2 = nn.Dense(self.hidden_dim)(xu)
        q2 = nn.relu(q2)
        q2 = nn.Dense(self.hidden_dim)(q2)
        q2 = nn.relu(q2)
        q2 = nn.Dense(self.n_agents)(q2)
        
        # Shape becomes (batch, N_AGENTS, 1) to perfectly match reward shape
        return jnp.expand_dims(q1, -1), jnp.expand_dims(q2, -1)