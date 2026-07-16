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

from examples.ks1d.decentralized.data_utils import evolve_to_attractor

@partial(jax.jit, static_argnames=['window_size'])
def extract_patches_jit(full_state, target_state, xi_norm, window_size):
    """JIT-compiled pure function for DPC local patch extraction."""
    error = full_state - target_state
    error_grad = jnp.gradient(error)
    n_pde = full_state.shape[0]
    half_window = window_size // 2

    # Pad with 'wrap' for PERIODIC BCs
    padded_error = jnp.pad(error, (half_window, half_window), mode='wrap')
    padded_grad = jnp.pad(error_grad, (half_window, half_window), mode='wrap')

    def get_local_obs(xi_n):
        # Map normalized position back to grid index
        center_idx = jax.lax.stop_gradient((xi_n * (n_pde - 1)).astype(int)) + half_window
        start = center_idx - half_window
        
        # Slice the periodic (wrapped) fields
        p_err = jax.lax.dynamic_slice(padded_error, (start,), (window_size,))
        p_grad = jax.lax.dynamic_slice(padded_grad, (start,), (window_size,))
        
        # Resize for consistent input size
        p_err = jax.image.resize(p_err, (20,), method='bilinear')
        p_grad = jax.image.resize(p_grad, (20,), method='bilinear')
        return jnp.concatenate([p_err, p_grad])

    return jax.vmap(get_local_obs)(xi_norm)


class KSHypeMARLEnv(gym.Env):
    def __init__(self, pde_dynamics, n_agents=8, N_grid=128, L=22.0, dt=0.05, max_steps=200):
        super().__init__()
        self.pde = pde_dynamics
        self.n_agents = n_agents
        self.N_grid = N_grid
        self.L = L
        self.dt = dt
        self.max_steps = max_steps
        
        # System parameters
        self.mu = np.array([L, dt], dtype=np.float32)
        self.n_mu = len(self.mu)
        
        # Actuator positions
        self.agent_positions = np.linspace(0.0, L, n_agents, endpoint=False) + (L/n_agents)/2
        self.xi_norm = jnp.array(self.agent_positions / self.L) # Cast to JAX array once
        
        # Match DPC Observation Dimensions
        self.window_size = 4
        self.resized_dim = 20
        self.local_y_dim = self.resized_dim * 2  # 20 for error, 20 for grad = 40
        self.local_obs_dim = self.local_y_dim + self.n_mu
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_agents,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.n_agents, self.local_obs_dim), 
            dtype=np.float32
        )
        
        self.current_state = None
        self.target_state = jnp.zeros(self.N_grid) 
        self.timestep = 0
        self.rng = jax.random.PRNGKey(0)

    def _get_local_observations(self, full_state):
        # Call the external JIT function
        y_local = np.array(extract_patches_jit(
            full_state, self.target_state, self.xi_norm, self.window_size
        ))
        
        # Broadcast mu and concatenate
        mu_broadcast = np.tile(self.mu, (self.n_agents, 1))
        return np.concatenate([y_local, mu_broadcast], axis=1)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = jax.random.PRNGKey(seed)
            
        self.timestep = 0
        self.rng, subkey = jax.random.split(self.rng)
        
        # Evolve to Chaotic Attractor
        # Instead of raw noise, we spin up the PDE so the agent 
        # starts from fully developed turbulence.
        self.current_state = np.array(evolve_to_attractor(
            subkey, 
            self.N_grid, 
            self.L, 
            warmup_time=100.0, 
            dt=self.dt
        ))
        
        return self._get_local_observations(self.current_state)

    def step(self, joint_action):
        self.timestep += 1
        self.rng, subkey = jax.random.split(self.rng)
        
        traj = self.pde.unroll_controlled(
            u_init=self.current_state,
            xi_fixed=jnp.array(self.agent_positions),
            u_target=self.target_state,
            params=jnp.array(joint_action),
            t_steps=1,
            N_grid=self.N_grid,
            L=self.L,
            dt=self.dt,
            key=subkey
        )
        
        new_full_state = traj[0][-1]
        
        # --- PHYSICS BLOW-UP KILL SWITCH ---
        # If the PDE solver diverges (NAN), penalize heavily and terminate the episode.
        if np.isnan(new_full_state).any() or np.isinf(new_full_state).any():
            print(f" [Env Warning] PDE Diverged at step {self.timestep}. Terminating episode.")
            
            # Create dummy observations to return safely
            dummy_obs = np.zeros((self.n_agents, self.local_obs_dim))
            
            # Massive penalty for blowing up the system
            penalty_rewards = np.full((self.n_agents, 1), -100.0) 
            
            info = {"global_reward": -100.0, "global_state": np.zeros(self.N_grid)}
            
            # Return done = True immediately
            return dummy_obs, penalty_rewards, True, info
        # ----------------------------------------

        self.current_state = new_full_state
        
        obs = self._get_local_observations(new_full_state)
        
        # Calculate Local Rewards
        y_local = obs[:, :self.local_y_dim]
        center_errors = y_local[:, 10] 
        rewards = -np.square(center_errors).reshape(-1, 1)
        
        # Global Logging Metrics
        global_energy = np.mean(np.square(new_full_state))
        info = {"global_reward": -global_energy, "global_state": np.array(new_full_state)}
        
        done = self.timestep >= self.max_steps
        
        return obs, rewards, done, info