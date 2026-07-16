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

from examples.heat1d.decentralized.data_utils import generate_grf

@partial(jax.jit, static_argnames=['window_size'])
def extract_patches_heat_jit(full_state, target_state, xi, window_size):
    """JIT-compiled pure function for DPC local patch extraction (Heat version)."""
    error = full_state - target_state
    error_grad = jnp.gradient(error)
    n_pde = full_state.shape[0]
    half_window = window_size // 2

    # Heat Equation uses Zero (Dirichlet) Boundary Conditions
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


class HeatHypeMARLEnv(gym.Env):
    def __init__(self, pde_dynamics, n_agents=8, N_grid=100, L=1.0, max_steps=300):
        super().__init__()
        self.pde = pde_dynamics
        self.n_agents = n_agents
        self.N_grid = N_grid
        self.L = L
        self.max_steps = max_steps
        
        # System parameters (nu: diffusion)
        # Using a default diffusion constant for heat. Adjust as needed.
        self.mu = np.array([0.01], dtype=np.float32) 
        self.n_mu = len(self.mu)
        
        self.window_size = 4
        self.resized_dim = 20
        self.local_y_dim = self.resized_dim * 2  # 20 for error, 20 for grad = 40
        self.local_obs_dim = self.local_y_dim + self.n_mu
        
        # Action space is now 2D per agent: [forcing_intensity, velocity]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_agents, 2), dtype=np.float32)
        
        # Observation space includes patch + mu
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.n_agents, self.local_obs_dim), 
            dtype=np.float32
        )
        
        self.current_state = None
        self.target_state = None
        self.agent_positions = None
        self.prev_v = None # Track previous velocity for acceleration penalty
        self.timestep = 0
        self.rng = jax.random.PRNGKey(0)

    def _get_local_observations(self, full_state, target_state, xi):
        # Call the external JIT function
        y_local = np.array(extract_patches_heat_jit(
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
        
        # Reset agents to evenly spaced positions within the [0.2, 0.8] domain
        self.agent_positions = np.linspace(0.2, 0.8, self.n_agents, dtype=np.float32)
        self.prev_v = np.zeros(self.n_agents, dtype=np.float32)
        
        return self._get_local_observations(self.current_state, self.target_state, self.agent_positions)

    def step(self, joint_action):
        self.timestep += 1
        
        # Unroll Heat dynamics (returns trajectories for z, xi, u, and v)
        # Note: Depending on your exact Heat PDEDynamics class, you might need to pass key/nu here
        z_traj, xi_traj, u_traj, v_traj = self.pde.unroll_controlled(
            z_init=jnp.array(self.current_state),
            xi_init=jnp.array(self.agent_positions),
            z_target=jnp.array(self.target_state),
            params=jnp.array(joint_action),
            t_steps=1
        )
        
        # Extract the state at the end of the 1-step unroll
        new_full_state = np.array(z_traj[-1])
        new_positions = np.array(xi_traj[-1])
        u_act = np.array(u_traj[-1])
        v_act = np.array(v_traj[-1])
        
        # --- PHYSICS BLOW-UP KILL SWITCH ---
        if np.isnan(new_full_state).any() or np.isinf(new_full_state).any() or np.isnan(new_positions).any():
            print(f" [Env Warning] Heat PDE Diverged at step {self.timestep}. Terminating episode.")
            dummy_obs = np.zeros((self.n_agents, self.local_obs_dim))
            penalty_rewards = np.full((self.n_agents, 1), -100.0) 
            info = {"global_reward": -100.0, "global_state": np.zeros(self.N_grid)}
            return dummy_obs, penalty_rewards, True, info
        # ----------------------------------------

        self.current_state = new_full_state
        self.agent_positions = new_positions
        
        obs = self._get_local_observations(self.current_state, self.target_state, self.agent_positions)
        
        # --- REWARD CALCULATION (Exact mirror of DPC loss_fn) ---
        
        # 1. Tracking Reward (Center error index: 10 out of 20 resized patch)
        y_local = obs[:, :self.local_y_dim]
        center_errors = y_local[:, self.resized_dim // 2]
        r_track = -5.0 * np.square(center_errors)
        
        # 2. Effort Penalty
        r_effort = -0.001 * (np.square(u_act) + 0.1 * np.square(v_act))
        
        # 3. Boundary Penalty
        margin = 0.02
        r_bound = -100.0 * (np.maximum(0, margin - self.agent_positions)**2 + 
                            np.maximum(0, self.agent_positions - (1.0 - margin))**2)
        
        # 4. Collision Avoidance
        R_safe = 0.05
        dists = np.abs(self.agent_positions[:, None] - self.agent_positions[None, :])
        mask = np.eye(self.n_agents)
        r_coll = -1.0 * np.sum(np.maximum(0, R_safe - (dists + mask * 1.0)) ** 2, axis=1)
        
        # 5. Damping (Acceleration) Penalty
        r_accel = -0.1 * np.square(v_act - self.prev_v)
        self.prev_v = v_act # Update velocity tracker for next step
        
        # Sum up individual agent rewards
        rewards = (r_track + r_effort + r_bound + r_coll + r_accel).reshape(-1, 1)
        
        # Global Logging Metrics
        global_mse = np.mean(np.square(self.current_state - self.target_state))
        info = {
            "global_reward": -global_mse, 
            "global_state": self.current_state,
            "agent_positions": self.agent_positions,
            "reward_breakdown": {
                "track": np.mean(r_track),
                "effort": np.mean(r_effort),
                "bound": np.mean(r_bound),
                "coll": np.mean(r_coll),
                "accel": np.mean(r_accel)
            }
        }
        
        done = self.timestep >= self.max_steps
        
        return obs, rewards, done, info