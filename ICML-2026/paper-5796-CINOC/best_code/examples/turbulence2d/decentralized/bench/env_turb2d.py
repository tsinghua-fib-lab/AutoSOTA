import gymnasium as gym
from gymnasium import spaces
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

# Import the core solver directly
import tesseracts.turbulence2d.solver as solver

@partial(jax.jit, static_argnames=['patch_size', 'n_grid'])
def extract_patches_2d_jit(w_state, w_target, xi_norm, patch_size, n_grid):
    # For turbulence stabilization, target is 0 vorticity
    error = w_state - w_target
    grad_x, grad_y = jnp.gradient(error)
    half_patch = patch_size // 2

    # Pad with 'wrap' for PERIODIC boundaries
    pad_width = ((half_patch, half_patch), (half_patch, half_patch))
    padded_err = jnp.pad(error, pad_width, mode='wrap')
    padded_gx = jnp.pad(grad_x, pad_width, mode='wrap')
    padded_gy = jnp.pad(grad_y, pad_width, mode='wrap')

    def get_local_obs(xi):
        i = jax.lax.stop_gradient((xi[0] * n_grid).astype(int))
        j = jax.lax.stop_gradient((xi[1] * n_grid).astype(int))
        
        p_err = jax.lax.dynamic_slice(padded_err, (i, j), (patch_size, patch_size))
        p_gx = jax.lax.dynamic_slice(padded_gx, (i, j), (patch_size, patch_size))
        p_gy = jax.lax.dynamic_slice(padded_gy, (i, j), (patch_size, patch_size))
        
        return jnp.concatenate([p_err.flatten(), p_gx.flatten(), p_gy.flatten()])

    return jax.vmap(get_local_obs)(xi_norm)


class Turbulence2DMARLEnv(gym.Env):
    def __init__(self, initial_conditions, n_agents=64, N_grid=64, L=1.0, dt=0.01, viscosity=5e-4, substeps=5, max_steps=150, sigma=0.05):
        super().__init__()
        # Removed pde_dynamics from arguments
        self.w_pool = initial_conditions 
        self.n_agents = n_agents
        self.N_grid = N_grid
        self.L = L
        self.dt = dt
        self.dt_phys = dt / substeps # Calculate physical timestep
        self.viscosity = viscosity
        self.substeps = substeps
        self.max_steps = max_steps
        
        # System parameters (Injected into observation)
        self.mu = np.array([L, dt, viscosity], dtype=np.float32)
        self.n_mu = len(self.mu)
        
        # 2D Actuator Grid Setup (8x8 Fixed Positions)
        grid_dim = int(np.sqrt(n_agents))
        self.grid_shape = (grid_dim, grid_dim)
        x_lin = np.linspace(0, L, grid_dim, endpoint=False) + (L/grid_dim)/2
        xv, yv = np.meshgrid(x_lin, x_lin)
        self.agent_positions = np.stack([xv.flatten(), yv.flatten()], axis=-1)
        self.xi_norm = jnp.array(self.agent_positions / self.L)
        
        # --- Precompute Spectral Solver Constants ---
        self.kx, self.ky, self.k_sq, self.k_inv = solver.get_spectral_grid(self.N_grid, self.L)
        centers_x = jnp.array(self.agent_positions[:, 0])
        centers_y = jnp.array(self.agent_positions[:, 1])
        self.forcing_hat = solver.compute_forcing_profile(centers_x, centers_y, self.N_grid, self.L, sigma)
        
        # JIT-compile the internal RK4 sub-stepping loop for speed
        @jax.jit
        def _physics_loop(w_hat_in, action_array):
            def rk4_loop(i, w):
                return solver.rk4_step(
                    w, self.dt_phys, self.kx, self.ky, self.k_sq, self.k_inv,
                    self.viscosity, self.forcing_hat, action_array
                )
            return jax.lax.fori_loop(0, self.substeps, rk4_loop, w_hat_in)
        
        self.physics_loop = _physics_loop
        
        # Observation Config
        self.patch_size = 16 
        self.local_y_dim = 3 * (self.patch_size ** 2) 
        self.local_obs_dim = self.local_y_dim + self.n_mu
        
        # Action space: 1D per agent, scaled to u_max=40.0 (or 75.0 depending on your config)
        self.u_max = 75.0 
        self.action_space = spaces.Box(low=-self.u_max, high=self.u_max, shape=(self.n_agents,), dtype=np.float32)
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
        
        idx = jax.random.randint(subkey, (), 0, self.w_pool.shape[0])
        self.current_state = np.array(self.w_pool[idx])
        
        return self._get_local_observations(self.current_state)

    def step(self, joint_action):
        self.timestep += 1
        
        # 1. Convert physical state to spectral space
        w_hat = jnp.fft.fft2(jnp.array(self.current_state))
        actions_jnp = jnp.array(joint_action)
        
        # 2. Run the fast JIT-compiled RK4 loop
        w_hat_next = self.physics_loop(w_hat, actions_jnp)
        
        # 3. Convert back to physical space for the environment state
        new_full_state = np.array(jnp.fft.ifft2(w_hat_next).real)
        
        # --- PHYSICS BLOW-UP KILL SWITCH ---
        if np.isnan(new_full_state).any() or np.isinf(new_full_state).any() or np.max(np.abs(new_full_state)) > 1000.0:
            print(f" [Env] Turbulence PDE Diverged at step {self.timestep}.")
            dummy_obs = np.zeros((self.n_agents, self.local_obs_dim))
            penalty_rewards = np.full((self.n_agents, 1), -500.0)
            return dummy_obs, penalty_rewards, True, {"global_reward": -500.0, "enstrophy": 500.0}

        self.current_state = new_full_state
        obs = self._get_local_observations(new_full_state)
        
        # --- REWARD CALCULATION ---
        global_enstrophy = np.mean(np.square(new_full_state))
        y_local_err = obs[:, :self.patch_size**2] 
        local_rewards = -np.mean(np.square(y_local_err), axis=-1, keepdims=True)
        
        rewards = 0.5 * local_rewards + 0.5 * (-global_enstrophy)
        
        info = {"global_reward": -global_enstrophy, "enstrophy": global_enstrophy, "global_state": new_full_state}
        done = self.timestep >= self.max_steps
        
        return obs, rewards, done, info