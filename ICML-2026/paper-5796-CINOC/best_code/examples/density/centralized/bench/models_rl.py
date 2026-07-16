import jax
import jax.numpy as jnp
import flax.linen as nn

# Action scaling constraints from NS2D baseline
V_MAX = 0.8  

class CentralizedActor(nn.Module):
    """
    Single-agent Actor for NS2D Density Control.
    Maps the [rho_grid, target_grid, agent_positions] to all [vx, vy] push commands.
    """
    hidden_dim: int = 256
    n_agents: int = 9

    @nn.compact
    def __call__(self, rho, rho_target, xi):
        # Flatten the 2D spatial grids: (..., Nx, Ny) -> (..., Nx*Ny)
        rho_flat = rho.reshape((*rho.shape[:-2], -1))
        target_flat = rho_target.reshape((*rho_target.shape[:-2], -1))
        
        # Flatten the 2D agent positions: (..., n_agents, 2) -> (..., n_agents*2)
        xi_flat = xi.reshape((*xi.shape[:-2], -1))
        
        # Concatenate all global information into a single 1D vector per batch
        x = jnp.concatenate([rho_flat, target_flat, xi_flat], axis=-1)
        
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        
        # Normalization trick for stability
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1.0)
        
        # Dual Heads for Push Velocity (vx, vy) for ALL agents simultaneously
        vx_raw = nn.Dense(self.n_agents)(x)
        vy_raw = nn.Dense(self.n_agents)(x)
        
        vx_out = V_MAX * jnp.tanh(vx_raw)
        vy_out = V_MAX * jnp.tanh(vy_raw)
        
        # Stack to form output shape: (..., n_agents, 2)
        return jnp.stack([vx_out, vy_out], axis=-1)


class CentralizedCritic(nn.Module):
    """
    Single-agent Critic.
    Maps [rho_grid, target_grid, agent_positions, actions] to a single global Q-value.
    """
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, rho, rho_target, xi, actions):
        # Flatten the 2D spatial grids
        rho_flat = rho.reshape((*rho.shape[:-2], -1))
        target_flat = rho_target.reshape((*rho_target.shape[:-2], -1))
        
        # Flatten agent positions and actions
        xi_flat = xi.reshape((*xi.shape[:-2], -1))
        actions_flat = actions.reshape((*actions.shape[:-2], -1))
        
        xu = jnp.concatenate([rho_flat, target_flat, xi_flat, actions_flat], axis=-1)
        
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