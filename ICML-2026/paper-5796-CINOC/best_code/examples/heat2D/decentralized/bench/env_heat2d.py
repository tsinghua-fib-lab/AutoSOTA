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

# Note: generate_grf is kept for reset() generation during evaluation,
# but training will use the pre-generated dataset.
from examples.heat1d.decentralized.data_utils import generate_grf

@partial(jax.jit, static_argnames=['window_size', 'resized_dim'])
def extract_patches_heat2d_jit(full_state, target_state, xi, window_size, resized_dim):
    """JIT-compiled 2D DPC local patch extraction."""
    error = full_state - target_state
    grad_x, grad_y = jnp.gradient(error)
    n_pde = full_state.shape[0]
    half_window = window_size // 2

    # Zero (Dirichlet) Boundary Conditions in 2D
    pad_width = ((half_window, half_window), (half_window, half_window))
    padded_error = jnp.pad(error, pad_width, mode='constant', constant_values=0.0)
    padded_gx = jnp.pad(grad_x, pad_width, mode='constant', constant_values=0.0)
    padded_gy = jnp.pad(grad_y, pad_width, mode='constant', constant_values=0.0)

    def get_local_obs(pos):
        # Map normalized [0, 1] 2D position back to grid indices
        idx_x = jax.lax.stop_gradient((pos[0] * (n_pde - 1)).astype(int)) + half_window
        idx_y = jax.lax.stop_gradient((pos[1] * (n_pde - 1)).astype(int)) + half_window
        start = (idx_x - half_window, idx_y - half_window)
        
        # Slice 2D fields
        slice_shape = (window_size, window_size)
        p_err = jax.lax.dynamic_slice(padded_error, start, slice_shape)
        p_gx = jax.lax.dynamic_slice(padded_gx, start, slice_shape)
        p_gy = jax.lax.dynamic_slice(padded_gy, start, slice_shape)
        
        # Resize to fixed dimension for neural network
        target_shape = (resized_dim, resized_dim)
        p_err = jax.image.resize(p_err, target_shape, method='bilinear').flatten()
        p_gx = jax.image.resize(p_gx, target_shape, method='bilinear').flatten()
        p_gy = jax.image.resize(p_gy, target_shape, method='bilinear').flatten()
        
        return jnp.concatenate([p_err, p_gx, p_gy])

    return jax.vmap(get_local_obs)(xi)

class Heat2DHypeMARLEnv(gym.Env):
    def __init__(self, pde_dynamics, n_agents=16, N_grid=32, L=1.0, max_steps=300):
        super().__init__()
        self.pde = pde_dynamics
        self.n_agents = n_agents
        self.N_grid = N_grid
        self.L = L
        self.max_steps = max_steps
        
        # System parameters (nu: diffusion)
        self.mu = np.array([0.01], dtype=np.float32) 
        self.n_mu = len(self.mu)
        
        self.window_size = 6  # 6x6 patch in 32x32 grid
        self.resized_dim = 10 # Resized to 10x10
        # 3 channels (err, gx, gy) * 100 pixels
        self.local_y_dim = (self.resized_dim ** 2) * 3 
        self.local_obs_dim = self.local_y_dim + self.n_mu
        
        # Action space: 3D per agent: [forcing_u, velocity_vx, velocity_vy]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_agents, 3), dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.n_agents, self.local_obs_dim), 
            dtype=np.float32
        )
        
        self.current_state = None
        self.target_state = None
        self.agent_positions = None
        self.prev_v = None 
        self.timestep = 0
        self.rng = jax.random.PRNGKey(0)

    def _get_local_observations(self, full_state, target_state, xi):
        y_local = np.array(extract_patches_heat2d_jit(
            full_state, target_state, xi, self.window_size, self.resized_dim
        ))
        mu_broadcast = np.tile(self.mu, (self.n_agents, 1))
        return np.concatenate([y_local, mu_broadcast], axis=1)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = jax.random.PRNGKey(seed)
            
        self.timestep = 0
        self.rng, key_init, key_target = jax.random.split(self.rng, 3)
        
        # Used for fallback/evaluation. Training loop will override this using the dataset.
        _, z_init = generate_grf(key_init, n_points=self.N_grid, length_scale=0.2)
        _, z_target = generate_grf(key_target, n_points=self.N_grid, length_scale=0.4)
        
        self.current_state = np.array(z_init)
        self.target_state = np.array(z_target)
        
        # 4x4 Grid initialization for 16 agents
        n_side = int(np.sqrt(self.n_agents))
        pos_1d = np.linspace(0.2, 0.8, n_side)
        X, Y = np.meshgrid(pos_1d, pos_1d)
        self.agent_positions = np.stack([X.flatten(), Y.flatten()], axis=-1).astype(np.float32)
        
        self.prev_v = np.zeros((self.n_agents, 2), dtype=np.float32)
        
        return self._get_local_observations(self.current_state, self.target_state, self.agent_positions)

    def step(self, joint_action):
        self.timestep += 1
        
        # joint_action shape: (n_agents, 3) -> [u, vx, vy]
        z_traj, xi_traj, u_traj, v_traj = self.pde.unroll_controlled(
            z_init=jnp.array(self.current_state),
            xi_init=jnp.array(self.agent_positions),
            z_target=jnp.array(self.target_state),
            params=jnp.array(joint_action),
            t_steps=1
        )
        
        new_full_state = np.array(z_traj[-1])
        new_positions = np.array(xi_traj[-1])
        u_act = np.array(u_traj[-1])
        v_act = np.array(v_traj[-1]) # shape (n_agents, 2)
        
        # --- PHYSICS BLOW-UP KILL SWITCH ---
        if np.isnan(new_full_state).any() or np.isinf(new_full_state).any() or np.isnan(new_positions).any():
            print(f" [Env Warning] 2D Heat PDE Diverged at step {self.timestep}. Terminating.")
            dummy_obs = np.zeros((self.n_agents, self.local_obs_dim))
            penalty_rewards = np.full((self.n_agents, 1), -100.0) 
            info = {"global_reward": -100.0, "global_state": np.zeros((self.N_grid, self.N_grid))}
            return dummy_obs, penalty_rewards, True, info

        self.current_state = new_full_state
        self.agent_positions = new_positions
        obs = self._get_local_observations(self.current_state, self.target_state, self.agent_positions)
        
        # --- 2D REWARD CALCULATION ---
        
        # 1. Tracking Reward (Center pixel of the flattened 10x10 error patch)
        # Center of 10x10 is approx index 55 (row 5, col 5 -> 5*10+5)
        center_idx = (self.resized_dim // 2) * self.resized_dim + (self.resized_dim // 2)
        center_errors = obs[:, center_idx]
        r_track = -5.0 * np.square(center_errors)
        
        # 2. Effort Penalty
        r_effort = -0.001 * (np.square(u_act) + 0.1 * np.sum(np.square(v_act), axis=-1))
        
        # 3. Boundary Penalty (2D margins)
        margin = 0.02
        x_pen = np.maximum(0, margin - self.agent_positions[:, 0])**2 + np.maximum(0, self.agent_positions[:, 0] - (1.0 - margin))**2
        y_pen = np.maximum(0, margin - self.agent_positions[:, 1])**2 + np.maximum(0, self.agent_positions[:, 1] - (1.0 - margin))**2
        r_bound = -100.0 * (x_pen + y_pen)
        
        # 4. Collision Avoidance
        R_safe = 0.08
        diff = self.agent_positions[:, None, :] - self.agent_positions[None, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1) + 1e-8)
        mask = np.eye(self.n_agents)
        r_coll = -20.0 * np.sum(np.maximum(0, R_safe - (dists + mask * 10.0)) ** 2, axis=1)
        
        # 5. Damping (Acceleration) Penalty
        r_accel = -0.1 * np.sum(np.square(v_act - self.prev_v), axis=-1)
        self.prev_v = v_act 
        
        rewards = (r_track + r_effort + r_bound + r_coll + r_accel).reshape(-1, 1)
        
        global_mse = np.mean(np.square(self.current_state - self.target_state))
        info = {
            "global_reward": -global_mse, 
            "global_state": self.current_state,
            "agent_positions": self.agent_positions
        }
        done = self.timestep >= self.max_steps
        return obs, rewards, done, info