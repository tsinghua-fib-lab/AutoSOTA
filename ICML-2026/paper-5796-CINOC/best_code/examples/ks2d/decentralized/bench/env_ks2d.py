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

@partial(jax.jit, static_argnames=['patch_size', 'n_grid'])
def extract_patches_2d_jit(full_state, target_state, xi_norm, patch_size, n_grid):
    """
    JIT-compiled pure function for 2D local patch extraction.
    Uses 'wrap' padding for the periodic boundary conditions of the KS equation.
    """
    error = full_state - target_state
    grad_x, grad_y = jnp.gradient(error)
    half_patch = patch_size // 2

    # Pad with 'wrap' for PERIODIC boundaries
    pad_width = ((half_patch, half_patch), (half_patch, half_patch))
    padded_err = jnp.pad(error, pad_width, mode='wrap')
    padded_gx = jnp.pad(grad_x, pad_width, mode='wrap')
    padded_gy = jnp.pad(grad_y, pad_width, mode='wrap')

    def get_local_obs(xi):
        # Map normalized coordinates [0, 1] to original grid indices.
        # Since we padded by half_patch on both sides, the original grid index 'i'
        # acts as the start index for a slice of size 'patch_size' centered at 'i'.
        i = jax.lax.stop_gradient((xi[0] * n_grid).astype(int))
        j = jax.lax.stop_gradient((xi[1] * n_grid).astype(int))
        
        # Slice patches directly
        p_err = jax.lax.dynamic_slice(padded_err, (i, j), (patch_size, patch_size))
        p_gx = jax.lax.dynamic_slice(padded_gx, (i, j), (patch_size, patch_size))
        p_gy = jax.lax.dynamic_slice(padded_gy, (i, j), (patch_size, patch_size))
        
        # Flatten and concatenate (Shape: 3 * patch_size^2)
        return jnp.concatenate([p_err.flatten(), p_gx.flatten(), p_gy.flatten()])

    return jax.vmap(get_local_obs)(xi_norm)


class KS2DMARLEnv(gym.Env):
    def __init__(self, pde_dynamics, initial_conditions, n_agents=100, N_grid=64, L=32.0, dt=0.005, substeps=20, max_steps=50):
        super().__init__()
        self.pde = pde_dynamics
        self.u_pool = initial_conditions
        self.n_agents = n_agents
        self.N_grid = N_grid
        self.L = L
        self.dt = dt
        self.substeps = substeps
        self.max_steps = max_steps
        
        # System parameters (Injected into observation)
        self.mu = np.array([L, dt], dtype=np.float32)
        self.n_mu = len(self.mu)
        
        # 2D Actuator Grid Setup (Fixed Positions)
        grid_dim = int(np.sqrt(n_agents))
        x_lin = np.linspace(0, L, grid_dim, endpoint=False) + (L/grid_dim)/2
        xv, yv = np.meshgrid(x_lin, x_lin)
        self.agent_positions = np.stack([xv.flatten(), yv.flatten()], axis=-1)
        self.xi_norm = jnp.array(self.agent_positions / self.L)
        
        # Observation Config
        self.patch_size = 12 
        self.local_y_dim = 3 * (self.patch_size ** 2) # err, grad_x, grad_y
        self.local_obs_dim = self.local_y_dim + self.n_mu
        
        # Action space: 1D per agent (forcing u only, agents don't move in this task)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_agents,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.n_agents, self.local_obs_dim), 
            dtype=np.float32
        )
        
        self.current_state = None
        self.target_state = jnp.zeros((self.N_grid, self.N_grid)) 
        self.timestep = 0
        self.rng = jax.random.PRNGKey(0)

    def _get_local_observations(self, full_state):
        y_local = np.array(extract_patches_2d_jit(
            full_state, self.target_state, self.xi_norm, self.patch_size, self.N_grid
        ))
        mu_broadcast = np.tile(self.mu, (self.n_agents, 1))
        return np.concatenate([y_local, mu_broadcast], axis=1)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = jax.random.PRNGKey(seed)
            
        self.timestep = 0
        self.rng, subkey = jax.random.split(self.rng)
        
        # Sample directly from the pre-generated fully-developed turbulence pool
        idx = jax.random.randint(subkey, (), 0, self.u_pool.shape[0])
        self.current_state = np.array(self.u_pool[idx])
        
        return self._get_local_observations(self.current_state)

    def step(self, joint_action):
        self.timestep += 1
        
        # Unroll 1 control step (applies 'substeps' physics integration steps internally)
        traj = self.pde.unroll_controlled(
            u_init=jnp.array(self.current_state),
            xi_fixed=jnp.array(self.agent_positions),
            u_target=jnp.array(self.target_state),
            params=jnp.array(joint_action),
            t_steps=1, 
            substeps=self.substeps,
            N_grid=self.N_grid,
            L=self.L,
            dt=self.dt,
            sigma=1.2 # Matches DPC config
        )
        
        # Trajectory returns: (u_traj, t_traj, u_ctrl_traj, xi_traj)
        new_full_state = np.array(traj[0][-1])
        
        # --- PHYSICS BLOW-UP KILL SWITCH ---
        if np.isnan(new_full_state).any() or np.isinf(new_full_state).any():
            print(f" [Env] KS2D PDE Diverged at step {self.timestep}.")
            dummy_obs = np.zeros((self.n_agents, self.local_obs_dim))
            penalty_rewards = np.full((self.n_agents, 1), -100.0)
            return dummy_obs, penalty_rewards, True, {"global_reward": -100.0}

        self.current_state = new_full_state
        obs = self._get_local_observations(new_full_state)
        
        # --- REWARD CALCULATION ---
        global_energy = np.mean(np.square(new_full_state))
        
        # Local reward: negative mean squared error of the agent's specific patch
        y_local_err = obs[:, :self.patch_size**2] 
        local_rewards = -np.mean(np.square(y_local_err), axis=-1, keepdims=True)
        
        # Combined reward formulation
        rewards = 0.5 * local_rewards + 0.5 * (-global_energy)
        
        info = {"global_reward": -global_energy, "global_state": new_full_state}
        done = self.timestep >= self.max_steps
        
        return obs, rewards, done, info