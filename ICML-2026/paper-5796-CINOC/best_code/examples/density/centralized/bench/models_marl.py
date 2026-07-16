import jax
import jax.numpy as jnp
import flax.linen as nn

PUSH_MAX = 0.8

class MARLActor(nn.Module):
    """
    MATD3 Shared Decentralized Actor with Full FOV.
    Every agent sees the flattened global grid + agent positions, 
    concatenated with its own one-hot ID to distinguish itself.
    """
    n_agents: int
    hidden_dim: int = 256
    
    @nn.compact
    def __call__(self, rho, target, xi):
        # 1. Flatten the 2D spatial grids & positions
        rho_flat = rho.reshape((*rho.shape[:-2], -1))
        target_flat = target.reshape((*target.shape[:-2], -1))
        xi_flat = xi.reshape((*xi.shape[:-2], -1))
        
        # Global state shared by all agents
        global_s = jnp.concatenate([rho_flat, target_flat, xi_flat], axis=-1)
        
        # 2. Expand global state for each agent: (..., N_AGENTS, global_dim)
        global_s_expanded = jnp.repeat(jnp.expand_dims(global_s, -2), self.n_agents, axis=-2)
        
        # 3. Create one-hot Agent IDs: (..., N_AGENTS, N_AGENTS)
        batch_shape = global_s.shape[:-1]
        agent_ids = jnp.broadcast_to(jnp.eye(self.n_agents), (*batch_shape, self.n_agents, self.n_agents))
        
        # 4. Concatenate global state + ID: (..., N_AGENTS, global_dim + N_AGENTS)
        x = jnp.concatenate([global_s_expanded, agent_ids], axis=-1)
        
        # 5. Shared MLP applied across the N_AGENTS dimension natively
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        
        # Normalization trick for stability (from DPC/ks1d)
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1.0)
        
        # Dual Heads for Push Velocity (vx, vy) 
        vx_raw = nn.Dense(1)(x)
        vy_raw = nn.Dense(1)(x)
        
        vx_out = PUSH_MAX * jnp.tanh(vx_raw)
        vy_out = PUSH_MAX * jnp.tanh(vy_raw)
        
        # Stack to form output shape: (..., n_agents, 2)
        return jnp.concatenate([vx_out, vy_out], axis=-1)

class MARLCritic(nn.Module):
    """
    MATD3 Centralized Critic.
    Takes the flattened global state + ALL actions to evaluate the joint policy.
    """
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, rho, target, xi, actions):
        # Flatten all states and actions for centralized evaluation
        rho_flat = rho.reshape((*rho.shape[:-2], -1))
        target_flat = target.reshape((*target.shape[:-2], -1))
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