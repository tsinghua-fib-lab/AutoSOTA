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

@partial(jax.jit, static_argnames=['window_size', 'resized_dim'])
def extract_patches_ns2d_jit(full_rho, target_rho, xi, window_size, resized_dim):
    """JIT-compiled 2D DPC local patch extraction for NS2D Density Control."""
    error = full_rho - target_rho
    grad_x, grad_y = jnp.gradient(error)
    n_grid = full_rho.shape[0]
    half_window = window_size // 2

    # Zero (Dirichlet) Boundary Conditions in 2D
    pad_width = ((half_window, half_window), (half_window, half_window))
    padded_error = jnp.pad(error, pad_width, mode='constant', constant_values=0.0)
    padded_gx = jnp.pad(grad_x, pad_width, mode='constant', constant_values=0.0)
    padded_gy = jnp.pad(grad_y, pad_width, mode='constant', constant_values=0.0)

    def get_local_obs(pos):
        # Map normalized [0, 1] 2D position back to grid indices
        idx_x = jax.lax.stop_gradient((pos[0] * (n_grid - 1)).astype(int)) + half_window
        idx_y = jax.lax.stop_gradient((pos[1] * (n_grid - 1)).astype(int)) + half_window
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

class NS2DDensityMARLEnv(gym.Env):
    def __init__(self, pde_dynamics, n_agents=9, Nx=64, Ny=64, max_steps=150):
        super().__init__()
        self.pde = pde_dynamics
        self.n_agents = n_agents
        self.Nx = Nx
        self.Ny = Ny
        self.max_steps = max_steps
        
        # NS2D physics parameters
        self.buoyancy = np.array([0.0], dtype=np.float32)
        self.n_phys = len(self.buoyancy)
        
        self.window_size = 12  # Slightly larger window for fluid dynamics
        self.resized_dim = 10  # Resized to 10x10
        # 3 channels (density err, gx, gy) * 100 pixels
        self.local_y_dim = (self.resized_dim ** 2) * 3 
        self.local_obs_dim = self.local_y_dim + self.n_phys
        
        # Action space: 2D per agent: [velocity_vx, velocity_vy] (push control)
        # Limits correspond to push_max = 0.8 from centralized config
        self.action_space = spaces.Box(low=-0.8, high=0.8, shape=(self.n_agents, 2), dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.n_agents, self.local_obs_dim), 
            dtype=np.float32
        )
        
        self.current_rho = None
        self.target_rho = None
        self.agent_positions = None
        self.prev_v = None 
        self.timestep = 0
        self.rng = jax.random.PRNGKey(42)

    def _get_local_observations(self, current_rho, target_rho, xi):
        y_local = np.array(extract_patches_ns2d_jit(
            current_rho, target_rho, xi, self.window_size, self.resized_dim
        ))
        phys_broadcast = np.tile(self.buoyancy, (self.n_agents, 1))
        return np.concatenate([y_local, phys_broadcast], axis=1)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = jax.random.PRNGKey(seed)
            
        self.timestep = 0
        self.rng, key_init, key_target = jax.random.split(self.rng, 3)
        
        # Placeholder for dataset initialization: In practice, load from `train_data.npz`
        # Here we use dummy zeros for structural setup
        self.current_rho = np.zeros((self.Nx, self.Ny), dtype=np.float32)
        self.target_rho = np.zeros((self.Nx, self.Ny), dtype=np.float32)
        
        # 3x3 Grid initialization based on centralized baseline bounds
        n_side = int(np.sqrt(self.n_agents))
        X, Y = np.meshgrid(
            np.linspace(0.15, 0.85, n_side),
            np.linspace(0.15, 1.0, n_side)
        )
        self.agent_positions = np.stack([X.flatten(), Y.flatten()], axis=-1).astype(np.float32)
        
        self.prev_v = np.zeros((self.n_agents, 2), dtype=np.float32)
        
        return self._get_local_observations(self.current_rho, self.target_rho, self.agent_positions)

    def step(self, joint_action):
        self.timestep += 1
        
        # joint_action shape: (n_agents, 2) -> [vx, vy]
        # Assuming unroll_controlled signature returns rho, xi, and velocities
        rho_traj, xi_traj, v_traj = self.pde.unroll_controlled(
            rho_init=jnp.array(self.current_rho),
            xi_init=jnp.array(self.agent_positions),
            rho_target=jnp.array(self.target_rho),
            params=jnp.array(joint_action),
            t_steps=1
        )
        
        new_rho = np.array(rho_traj[-1])
        new_positions = np.array(xi_traj[-1])
        v_act = np.array(v_traj[-1]) # shape (n_agents, 2)
        
        # --- PHYSICS BLOW-UP KILL SWITCH ---
        if np.isnan(new_rho).any() or np.isinf(new_rho).any() or np.isnan(new_positions).any():
            print(f" [Env Warning] NS2D PDE Diverged at step {self.timestep}. Terminating.")
            dummy_obs = np.zeros((self.n_agents, self.local_obs_dim))
            penalty_rewards = np.full((self.n_agents, 1), -100.0) 
            info = {"global_reward": -100.0, "global_state": np.zeros((self.Nx, self.Ny))}
            return dummy_obs, penalty_rewards, True, info

        self.current_rho = new_rho
        self.agent_positions = new_positions
        obs = self._get_local_observations(self.current_rho, self.target_rho, self.agent_positions)
        
        # --- NS2D REWARD CALCULATION ---
        # Matching loss weights: Track: 10.0, Effort: 0.001, Bound: 20.0, Coll: 10.0, Accel: 0.05
        
        # 1. Tracking Reward (Center pixel of the flattened 10x10 error patch)
        center_idx = (self.resized_dim // 2) * self.resized_dim + (self.resized_dim // 2)
        center_errors = obs[:, center_idx]
        r_track = -10.0 * np.square(center_errors)
        
        # 2. Effort Penalty (Only velocity applied for movable injectors)
        r_effort = -0.001 * np.sum(np.square(v_act), axis=-1)
        
        # 3. Boundary Penalty
        margin = 0.02
        x_pen = np.maximum(0, margin - self.agent_positions[:, 0])**2 + \
                np.maximum(0, self.agent_positions[:, 0] - (1.0 - margin))**2
        y_pen = np.maximum(0, margin - self.agent_positions[:, 1])**2 + \
                np.maximum(0, self.agent_positions[:, 1] - (1.0 - margin))**2
        r_bound = -20.0 * (x_pen + y_pen)
        
        # 4. Collision Avoidance 
        R_safe = 0.15
        diff = self.agent_positions[:, None, :] - self.agent_positions[None, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1) + 1e-8)
        mask = np.eye(self.n_agents)
        r_coll = -10.0 * np.sum(np.maximum(0, R_safe - (dists + mask * 10.0)) ** 2, axis=1)
        
        # 5. Damping (Acceleration) Penalty
        r_accel = -0.05 * np.sum(np.square(v_act - self.prev_v), axis=-1)
        self.prev_v = v_act 
        
        rewards = (r_track + r_effort + r_bound + r_coll + r_accel).reshape(-1, 1)
        
        global_mse = np.mean(np.square(self.current_rho - self.target_rho))
        info = {
            "global_reward": -global_mse, 
            "global_state": self.current_rho,
            "agent_positions": self.agent_positions
        }
        
        done = self.timestep >= self.max_steps
        
        return obs, rewards, done, info