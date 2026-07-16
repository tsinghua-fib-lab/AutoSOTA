import gymnasium as gym
from gymnasium import spaces
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import sys
from pathlib import Path

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(script_dir))

from examples.fkpp1d.decentralized.data_utils import generate_grf

@partial(jax.jit, static_argnames=['window_size'])
def extract_patches_jit(full_state, target_state, xi, window_size):
    """JIT-compiled pure function for DPC local patch extraction (FKPP version)."""
    error = full_state - target_state
    error_grad = jnp.gradient(error)
    n_pde = full_state.shape[0]
    half_window = window_size // 2

    # Pad with 'constant' (0.0) for ZERO BOUNDARY CONDITIONS used in FKPP
    padded_error = jnp.pad(error, (half_window, half_window), mode='constant', constant_values=0.0)
    padded_grad = jnp.pad(error_grad, (half_window, half_window), mode='constant', constant_values=0.0)

    def get_local_obs(pos):
        # Map normalized position [0, 1] back to grid index
        center_idx = jax.lax.stop_gradient((pos * (n_pde - 1)).astype(int)) + half_window
        start = center_idx - half_window
        
        # Slice the fields
        p_err = jax.lax.dynamic_slice(padded_error, (start,), (window_size,))
        p_grad = jax.lax.dynamic_slice(padded_grad, (start,), (window_size,))
        
        # Resize for consistent neural network input size
        p_err = jax.image.resize(p_err, (20,), method='bilinear')
        p_grad = jax.image.resize(p_grad, (20,), method='bilinear')
        return jnp.concatenate([p_err, p_grad])

    return jax.vmap(get_local_obs)(xi)


class FKPPHypeMARLEnv(gym.Env):
    def __init__(self, pde_dynamics, n_agents=20, N_grid=100, L=1.0, dt=0.05, max_steps=300):
        super().__init__()
        self.pde = pde_dynamics
        self.n_agents = n_agents
        self.N_grid = N_grid
        self.L = L
        self.dt = dt
        self.max_steps = max_steps
        
        # System parameters (nu: diffusion, rho: growth)
        self.mu = np.array([0.005, 3.0], dtype=np.float32)
        self.n_mu = len(self.mu)
        
        self.window_size = 4
        self.resized_dim = 20
        self.local_y_dim = self.resized_dim * 2  # 20 for error, 20 for grad = 40
        self.local_obs_dim = self.local_y_dim + self.n_mu
        
        # Action space is now 2D per agent: [forcing_intensity, velocity]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_agents, 2), dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.n_agents, self.local_obs_dim), 
            dtype=np.float32
        )
        
        self.current_state = None
        self.target_state = None
        self.agent_positions = None 
        self.timestep = 0
        self.rng = jax.random.PRNGKey(0)

    def _get_local_observations(self, full_state, target_state, xi):
        # Call the external JIT function
        y_local = np.array(extract_patches_jit(
            full_state, target_state, xi, self.window_size
        ))
        
        # Broadcast mu and concatenate
        mu_broadcast = np.tile(self.mu, (self.n_agents, 1))
        return np.concatenate([y_local, mu_broadcast], axis=1)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = jax.random.PRNGKey(seed)
            
        self.timestep = 0
        self.rng, key_init, key_target = jax.random.split(self.rng, 3)
        
        # Generate initial PDE state and Target state using GRF
        _, z_init = generate_grf(key_init, n_points=self.N_grid, length_scale=0.2)
        _, z_target = generate_grf(key_target, n_points=self.N_grid, length_scale=0.4)
        
        self.current_state = np.array(z_init)
        self.target_state = np.array(z_target)
        
        # Reset agents to evenly spaced positions
        self.agent_positions = np.linspace(0.2, 0.8, self.n_agents, dtype=np.float32)
        
        return self._get_local_observations(self.current_state, self.target_state, self.agent_positions)

    def step(self, joint_action):
        self.timestep += 1
        self.rng, subkey = jax.random.split(self.rng)
        
        # Unroll FKPP dynamics (returns trajectories for z, xi, u, and v)
        z_traj, xi_traj, u_traj, v_traj = self.pde.unroll_controlled(
            z_init=jnp.array(self.current_state),
            xi_init=jnp.array(self.agent_positions),
            z_target=jnp.array(self.target_state),
            params=jnp.array(joint_action),
            t_steps=1,
            key=subkey,
            nu=self.mu[0],
            rho=self.mu[1]
        )
        
        # Extract the state at the end of the 1-step unroll
        new_full_state = z_traj[-1]
        new_xi = xi_traj[-1]
        
        # --- PHYSICS BLOW-UP KILL SWITCH ---
        if np.isnan(new_full_state).any() or np.isinf(new_full_state).any():
            print(f" [Env Warning] PDE Diverged at step {self.timestep}. Terminating episode.")
            dummy_obs = np.zeros((self.n_agents, self.local_obs_dim))
            penalty_rewards = np.full((self.n_agents, 1), -100.0) 
            info = {"global_reward": -100.0, "global_state": np.zeros(self.N_grid)}
            return dummy_obs, penalty_rewards, True, info
        # ----------------------------------------

        self.current_state = np.array(new_full_state)
        self.agent_positions = np.array(new_xi)
        
        obs = self._get_local_observations(self.current_state, self.target_state, self.agent_positions)
        
        # Calculate Local Rewards (Tracking error at center of observation)
        y_local = obs[:, :self.local_y_dim]
        center_errors = y_local[:, self.resized_dim // 2] 
        
        # Penalty for moving out of bounds [0, 1]
        margin = 0.02
        out_of_bounds_penalty = np.maximum(0, margin - self.agent_positions)**2 + np.maximum(0, self.agent_positions - (1.0 - margin))**2
        
        # Final local reward: negative tracking error minus boundary penalty
        rewards = -np.square(center_errors) - 10.0 * out_of_bounds_penalty
        rewards = rewards.reshape(-1, 1)
        
        # Global Logging Metrics
        global_mse = np.mean(np.square(self.current_state - self.target_state))
        info = {
            "global_reward": -global_mse, 
            "global_state": self.current_state,
            "agent_positions": self.agent_positions
        }
        
        done = self.timestep >= self.max_steps
        
        return obs, rewards, done, info