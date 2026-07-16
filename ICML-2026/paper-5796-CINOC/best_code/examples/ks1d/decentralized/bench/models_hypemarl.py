import jax
import jax.numpy as jnp
import flax.linen as nn

class HyperActor(nn.Module):
    """
    Hypernetwork-based Actor.
    Maps z = [PE(p_i), mu] to the parameters of a local policy network.
    The local policy then maps local state y_i to action u_i.
    """
    hidden_dim: int = 256
    action_dim: int = 1
    u_max: float = 1.0  # MATCHING DPC u_max bounds
    
    @nn.compact
    def __call__(self, z, y):
        # Calculate total parameters needed for a 1-hidden-layer MLP
        y_dim = y.shape[-1]
        w1_size = y_dim * self.hidden_dim
        b1_size = self.hidden_dim
        w2_size = self.hidden_dim * self.action_dim
        b2_size = self.action_dim
        total_params = w1_size + b1_size + w2_size + b2_size
        
        # Hypernetwork forward pass (predicts primary network weights)
        h_out = nn.Dense(total_params, kernel_init=nn.initializers.xavier_uniform())(z)
        
        # Unpack weights for the batch
        idx = 0
        w1 = h_out[:, idx : idx+w1_size].reshape(-1, y_dim, self.hidden_dim)
        idx += w1_size
        b1 = h_out[:, idx : idx+b1_size].reshape(-1, 1, self.hidden_dim)
        idx += b1_size
        w2 = h_out[:, idx : idx+w2_size].reshape(-1, self.hidden_dim, self.action_dim)
        idx += w2_size
        b2 = h_out[:, idx : idx+b2_size].reshape(-1, 1, self.action_dim)
        
        # Primary local network forward pass 
        y_exp = jnp.expand_dims(y, 1) 
        
        # DPC Normalization trick to keep gradients stable over chaotic PDEs
        y_exp = y_exp / (jnp.linalg.norm(y_exp, axis=-1, keepdims=True) + 1.0)
        
        hidden = nn.relu(jnp.matmul(y_exp, w1) + b1)
        out = jnp.matmul(hidden, w2) + b2
        
        # Actions bounded to [-u_max, u_max]
        return jnp.squeeze(jnp.tanh(out), axis=1) * self.u_max


class HyperCritic(nn.Module):
    """
    Hypernetwork-based Critic (TD3 uses two of these)
    Maps z = [PE(p_i), mu] to parameters of a local Q-network.
    """
    hidden_dim: int = 256
    
    @nn.compact
    def __call__(self, z, y, u):
        yu = jnp.concatenate([y, u], axis=-1)
        yu_dim = yu.shape[-1]
        
        w1_size = yu_dim * self.hidden_dim
        b1_size = self.hidden_dim
        w2_size = self.hidden_dim * 1
        b2_size = 1
        total_params = w1_size + b1_size + w2_size + b2_size
        
        # Hypernetwork
        h_out = nn.Dense(total_params, kernel_init=nn.initializers.xavier_uniform())(z)
        
        # Unpack
        idx = 0
        w1 = h_out[:, idx : idx+w1_size].reshape(-1, yu_dim, self.hidden_dim)
        idx += w1_size
        b1 = h_out[:, idx : idx+b1_size].reshape(-1, 1, self.hidden_dim)
        idx += b1_size
        w2 = h_out[:, idx : idx+w2_size].reshape(-1, self.hidden_dim, 1)
        idx += w2_size
        b2 = h_out[:, idx : idx+b2_size].reshape(-1, 1, 1)
        
        # Primary network
        yu_exp = jnp.expand_dims(yu, 1)
        hidden = nn.relu(jnp.matmul(yu_exp, w1) + b1)
        q_val = jnp.matmul(hidden, w2) + b2
        
        return jnp.squeeze(q_val, axis=1)


class SurrogateModel(nn.Module):
    """
    Shallow NN surrogate model for MB-HypeMARL
    Approximates local dynamics: y_{i, t+1} = F_tilde(y_{i,t}, u_{i,t}, mu)
    """
    hidden_dim: int = 256
    
    @nn.compact
    def __call__(self, y, u, mu):
        # Concatenate local state, action, and system parameters
        x = jnp.concatenate([y, u, mu], axis=-1)
        
        # Shallow architecture
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        next_y_pred = nn.Dense(y.shape[-1])(x)
        return next_y_pred