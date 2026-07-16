import jax
import jax.numpy as jnp
import numpy as np

def get_sinusoidal_encoding(p, d=2048, n=1000.0):
    """
    Computes the sinusoidal positional encoding for the agents' relative positions.
    
    Args:
        p: Array of agent positions of shape (N_agents,).
        d: Dimension of the positional embedding vector.
        n: Constant scaling value.
        
    Returns:
        pe: Positional encoding matrix of shape (N_agents, d).
    """
    # Create an array of j values: 1 to d/2
    j_vals = jnp.arange(1, (d // 2) + 1)
    
    # Calculate omega_j = n^(2j/d)
    omega_j = jnp.power(n, 2 * j_vals / d)
    
    # Expand dimensions for broadcasting: (N_agents, 1) and (1, d/2)
    p_expanded = p[:, None]
    omega_expanded = omega_j[None, :]
    
    # Calculate arguments for sin and cos
    args = p_expanded / omega_expanded
    
    # Calculate sin and cos components
    sin_enc = jnp.sin(args)
    cos_enc = jnp.cos(args)
    
    # Interleave sin and cos: [sin1, cos1, sin2, cos2, ...]
    pe = jnp.stack([sin_enc, cos_enc], axis=-1).reshape(p.shape[0], d)
    return pe

class DecentralizedReplayBuffer:
    """
    A replay buffer tailored for Decentralized MARL and MB-HypeMARL.
    Stores flat transitions for local agents: (obs, action, reward, next_obs, mu).
    """
    def __init__(self, max_size, obs_dim, action_dim, n_mu):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        # Local state y_{i,t} + mu
        self.obs = np.zeros((max_size, obs_dim))
        self.next_obs = np.zeros((max_size, obs_dim))
        self.actions = np.zeros((max_size, action_dim))
        self.rewards = np.zeros((max_size, 1))
        # Storing mu separately is useful for the Model-Based surrogate training
        self.mu = np.zeros((max_size, n_mu)) 

    def add(self, obs_batch, action_batch, reward_batch, next_obs_batch, mu_batch):
        """
        Adds a batch of N_agents experiences to the buffer.
        """
        batch_size = obs_batch.shape[0]
        
        # Handle wrap-around if adding the batch exceeds max_size
        end_idx = self.ptr + batch_size
        if end_idx > self.max_size:
            # Simple approach: reset pointer if it doesn't fit exactly
            # (In production, you'd wrap around precisely, but this keeps it clean)
            self.ptr = 0
            end_idx = batch_size
            
        self.obs[self.ptr:end_idx] = obs_batch
        self.actions[self.ptr:end_idx] = action_batch
        self.rewards[self.ptr:end_idx] = reward_batch
        self.next_obs[self.ptr:end_idx] = next_obs_batch
        self.mu[self.ptr:end_idx] = mu_batch
        
        self.ptr = (self.ptr + batch_size) % self.max_size
        self.size = min(self.size + batch_size, self.max_size)

    def sample(self, batch_size):
        """
        Samples a batch of local transitions and returns them as JAX arrays.
        """
        ind = np.random.randint(0, self.size, size=batch_size)
        
        return (
            jnp.array(self.obs[ind]),
            jnp.array(self.actions[ind]),
            jnp.array(self.rewards[ind]),
            jnp.array(self.next_obs[ind]),
            jnp.array(self.mu[ind])
        )