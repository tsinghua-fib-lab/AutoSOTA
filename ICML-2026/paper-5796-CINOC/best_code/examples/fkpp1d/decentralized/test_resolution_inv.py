import jax
import jax.numpy as jnp
from jax import jit, lax
from flax import linen as nn
import flax.serialization
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from functools import partial
from tqdm import tqdm

# --- 1. Physics Solver (Resolution Agnostic) ---
def forcing_fn_1d(xi, u, N, sigma=0.05):
    x_coords = jnp.linspace(0, 1, N)
    def single_actuator(pos, intensity):
        dist_sq = (x_coords - pos)**2
        return intensity * jnp.exp(-dist_sq / (2 * sigma**2))
    forcings = jax.vmap(single_actuator)(xi, u)
    return jnp.sum(forcings, axis=0)

def solve_tridiagonal_diffusion(z_explicit, r, N):
    d = jnp.ones(N) * (1 + 2 * r)
    d = d.at[0].set(1.0).at[-1].set(1.0)
    ld = jnp.ones(N) * (-r); ld = ld.at[0].set(0.0)
    ud = jnp.ones(N) * (-r); ud = ud.at[-1].set(0.0)
    rhs = z_explicit.at[0].set(0.0).at[-1].set(0.0)[:, None]
    return jax.lax.linalg.tridiagonal_solve(ld, d, ud, rhs).ravel()

@jit
def fkpp_step_1d(z, xi, u, v, dt, dx, nu, rho):
    N = z.shape[0]
    # Physics parameters now depend on passed dx/dt
    f_t = forcing_fn_1d(xi, u, N)
    reaction = rho * z * (1.0 - z)
    z_explicit = z + dt * (reaction + f_t)
    
    r = nu * dt / (dx**2)
    z_next = solve_tridiagonal_diffusion(z_explicit, r, N)
    
    z_next = jnp.clip(z_next, 0.0, 1.0)
    xi_next = jnp.clip(xi + dt * v, 0.0, 1.0)
    return z_next, xi_next

# --- 2. Policy Definition ---
class DecentralizedControlNet(nn.Module):
    features: tuple = (64, 64)
    u_max: float = 40.0
    v_max: float = 2.0
    sensor_range: float = 0.08 
    
    def setup(self):
        self.frequencies = jnp.array([1.0, 2.0, 4.0, 8.0])

    def branch_net(self, local_patch):
        x = local_patch
        for feat in self.features:
            x = nn.Dense(feat)(x)
            x = x / (jnp.linalg.norm(x) + 1.0) 
            x = nn.tanh(x)
        return x

    def trunk_net(self, xi):
        angle = xi[:, None] * self.frequencies * jnp.pi
        encoded = jnp.concatenate([jnp.sin(angle), jnp.cos(angle)], axis=-1)
        for feat in [32, 32]:
            encoded = nn.Dense(feat)(encoded)
            encoded = nn.tanh(encoded)
        return encoded 

    @nn.compact
    def __call__(self, z_curr, z_target, xi_curr):
        n_pde = z_curr.shape[0]
        error = z_curr - z_target
        error_grad = jnp.gradient(error)

        # Scale window size by resolution to maintain physical sensor width
        window_size = int(self.sensor_range * n_pde)
        half_window = window_size // 2

        padded_error = jnp.pad(error, (half_window, half_window), mode='edge')
        padded_grad = jnp.pad(error_grad, (half_window, half_window), mode='edge')

        def get_local_obs(xi):
            xi = jnp.clip(xi, 0.0, 1.0)
            center_idx = jax.lax.stop_gradient((xi * (n_pde - 1)).astype(int)) + half_window
            start = center_idx - half_window
            
            p_err = jax.lax.dynamic_slice(padded_error, (start,), (window_size,))
            p_grad = jax.lax.dynamic_slice(padded_grad, (start,), (window_size,))
            
            # The Magic: Resize to fixed dimension for Zero-Shot
            p_err = jax.image.resize(p_err, (20,), method='bilinear')
            p_grad = jax.image.resize(p_grad, (20,), method='bilinear')
            
            return jnp.concatenate([p_err, p_grad])

        local_patches = jax.vmap(get_local_obs)(xi_curr)
        branch_outs = jax.vmap(self.branch_net)(local_patches)
        trunk_outs = self.trunk_net(xi_curr)
        
        combined = jnp.concatenate([branch_outs, trunk_outs], axis=-1)
        x = nn.Dense(32)(combined); x = nn.tanh(x)
        return self.u_max * jnp.tanh(nn.Dense(1)(x).squeeze(-1)), \
               self.v_max * jnp.tanh(nn.Dense(1)(x).squeeze(-1))

# --- 3. Evaluation Logic ---

def generate_consistent_conditions(key, base_res=1000):
    """
    Generates conditions at high res, returns function to downsample.
    Ensures different resolutions solve the EXACT same physical problem.
    """
    k1, k2 = jax.random.split(key)
    
    # Simple GRF generation
    x = jnp.linspace(0, 1, base_res)
    def get_grf(k):
        # Random Fourier features
        freqs = jnp.arange(1, 11)
        amps = jax.random.normal(k, (10,)) * jnp.exp(-0.5 * freqs)
        phases = jax.random.uniform(k, (10,)) * 2 * jnp.pi
        signal = jnp.sum(amps[:, None] * jnp.sin(2*jnp.pi*freqs[:, None]*x + phases[:, None]), axis=0)
        # Normalize to [0, 1] roughly
        signal = (signal - signal.min()) / (signal.max() - signal.min())
        return signal

    z_init_high = get_grf(k1) * 0.5  # Start low
    z_target_high = get_grf(k2)
    
    def sampler(target_res):
        x_target = jnp.linspace(0, 1, target_res)
        x_base = jnp.linspace(0, 1, base_res)
        z_i = jnp.interp(x_target, x_base, z_init_high)
        z_t = jnp.interp(x_target, x_base, z_target_high)
        return z_i, z_t

    return sampler

def run_simulation(params, model, z_init, xi_init, z_target, t_steps=300):
    dt = 0.001
    L = 1.0
    N = z_init.shape[0]
    dx = L / N

    def step_fn(carry, _):
        z, xi = carry
        # Zero-shot inference: works because model resizes inputs internally
        u, v = model.apply(params, z, z_target, xi)
        z_next, xi_next = fkpp_step_1d(z, xi, u, v, dt, dx, nu=0.005, rho=3.0)
        return (z_next, xi_next), z_next

    (z_final, xi_final), traj = jax.lax.scan(step_fn, (z_init, xi_init), None, length=t_steps)
    return z_final, traj

# --- 4. Main Execution ---

def main():
    # Setup
    key = jax.random.PRNGKey(42)
    model = DecentralizedControlNet()
    
    # LOAD PARAMETERS
    try:
        with open('decentralized_params.msgpack', 'rb') as f:
            params_bytes = f.read()
        
        # Init dummy params to get structure
        dummy_params = model.init(key, jnp.zeros((100,)), jnp.zeros((100,)), jnp.zeros((10,)))
        params = flax.serialization.from_bytes(dummy_params, params_bytes)
        print("Loaded trained parameters.")
    except FileNotFoundError:
        print("Warning: Parameter file not found. using random init for demonstration.")
        params = model.init(key, jnp.zeros((100,)), jnp.zeros((100,)), jnp.zeros((10,)))

    # Configuration
    resolutions = [50, 100, 200, 400, 800]
    agent_counts = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140]
    results = []

    print("Starting Zero-Shot Resolution Evaluation...")
        
    for n_agents in tqdm(agent_counts, desc="Agents"):
        for res in resolutions:
            # Generate consistent problem instance for this resolution
            sampler = generate_consistent_conditions(jax.random.PRNGKey(0)) # Fixed seed = same physics
            z_init, z_target = sampler(res)
            
            # Init agents uniformly
            xi_init = jnp.linspace(0.1, 0.9, n_agents)
            
            run_jit = jax.jit(partial(run_simulation, params, model, t_steps=300))
            
            z_final, _ = run_jit(z_init, xi_init, z_target)
            
            # Compute Metric (MSE)
            mse = jnp.mean((z_final - z_target)**2)
            
            results.append({
                "Resolution": res,
                "Agents": n_agents,
                "MSE": float(mse)
            })

    # --- 5. Plotting ---
    df = pd.DataFrame(results)
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Agents", y="MSE", hue="Resolution", marker="o", palette="viridis")
    
    plt.title("Zero-Shot Policy Performance Across Resolutions\n(Same Weights, Different Grid Sizes)")
    plt.ylabel("Final MSE (Tracking Error)")
    plt.xlabel("Number of Agents")
    plt.yscale("log")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig("resolution_generalization.png")
    print("Saved plot to resolution_generalization.png")

if __name__ == "__main__":
    main()