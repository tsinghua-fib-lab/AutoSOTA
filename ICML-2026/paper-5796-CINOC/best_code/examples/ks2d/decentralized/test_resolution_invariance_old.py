import jax
import jax.numpy as jnp
from flax import linen as nn
import flax.serialization
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np 
from functools import partial
from tqdm import tqdm
import sys
from pathlib import Path

# --- Setup Imports ---
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from tesseracts.ks2d.solver import ks_spectral_step_etdrk4, precompute_etdrk4_coeffs
from models.policy_ks2d import DecentralizedKS2DControlNet 

jax.config.update("jax_enable_x64", True)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. THE RESOLUTION ADAPTER
# ═══════════════════════════════════════════════════════════════════════════════
class ScaleInvariantPolicy(DecentralizedKS2DControlNet):
    training_res: int = 64

    def extract_local_patch(self, field, xi_norm, n_grid):
        scale = n_grid / self.training_res
        extract_size = int(self.patch_size * scale)
        half_extract = extract_size // 2

        i = (xi_norm[0] * n_grid).astype(int)
        j = (xi_norm[1] * n_grid).astype(int)

        padded_field = jnp.pad(field, (
            (half_extract, half_extract),
            (half_extract, half_extract)
        ), mode='wrap')

        patch = jax.lax.dynamic_slice(
            padded_field,
            (i, j),
            (extract_size, extract_size)
        )
        
        patch_resized = jax.image.resize(
            patch, 
            (self.patch_size, self.patch_size), 
            method='linear'
        )
        return patch_resized

# ═══════════════════════════════════════════════════════════════════════════════
# 2. CHAOTIC INITIAL CONDITIONS (With Warmup)
# ═══════════════════════════════════════════════════════════════════════════════

def warmup_high_res_chaos(key, N, L, warmup_time=200.0, dt=0.005):
    """
    1. Generates noise.
    2. Evolves it for T=200s to create REAL CHAOS (matching training data).
    3. Returns the chaotic field at high resolution.
    """
    # A. Generate Random Noise (Same as data_utils.py)
    x = jnp.linspace(0, L, N, endpoint=False)
    X, Y = jnp.meshgrid(x, x)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    
    # Mode 1: Base
    phase1_x = jax.random.uniform(k1, minval=0, maxval=2*jnp.pi)
    phase1_y = jax.random.uniform(k2, minval=0, maxval=2*jnp.pi)
    u = jnp.sin(2*jnp.pi*X/L + phase1_x) * jnp.cos(2*jnp.pi*Y/L + phase1_y)
    
    # Mode 2: Perturbation
    phase2 = jax.random.uniform(k3, minval=0, maxval=2*jnp.pi)
    u += 0.5 * jnp.sin(4*jnp.pi*X/L + phase2)
    u += 0.05 * jax.random.normal(k4, shape=(N, N))
    
    # Normalize
    u = (u - jnp.mean(u)) 
    u = u / (jnp.std(u) + 1e-6)

    # B. Setup Solver for Warmup
    u_hat = jnp.fft.rfftn(u)
    dx = L / N
    kx_vec = 2 * jnp.pi * jnp.fft.fftfreq(N, d=dx)
    ky_vec = 2 * jnp.pi * jnp.fft.rfftfreq(N, d=dx)
    KX, KY = jnp.meshgrid(kx_vec, ky_vec, indexing='ij')
    q_sq = KX**2 + KY**2
    L_linear = q_sq - q_sq**2
    
    k_max_x = jnp.max(jnp.abs(kx_vec))
    k_max_y = jnp.max(jnp.abs(ky_vec))
    mask_x = jnp.abs(KX) < (2.0/3.0 * k_max_x)
    mask_y = jnp.abs(KY) < (2.0/3.0 * k_max_y)
    dealias_mask = (mask_x & mask_y).astype(jnp.float32)
    
    etdrk4_coeffs = precompute_etdrk4_coeffs(L_linear, dt)
    
    # C. Run Warmup Loop (No Control)
    steps = int(warmup_time / dt)
    xi_dummy = jnp.zeros((1, 2)) 
    u_control_dummy = jnp.zeros(1)

    def warmup_step(carry, _):
        uh, uc = carry
        uh_next, uc_next = ks_spectral_step_etdrk4(
            uh, uc, xi_dummy, u_control_dummy,
            KX, KY, etdrk4_coeffs, dealias_mask,
            N=N, L=L, dt=dt, sigma=1.2
        )
        return (uh_next, uc_next), None

    print(f"  [Warmup] Evolving high-res chaos for {warmup_time}s ({steps} steps)...")
    # We use scan but don't save trajectory to save VRAM
    (u_hat_final, u_final), _ = jax.lax.scan(
        warmup_step, (u_hat, u), None, length=steps
    )
    
    return u_final

# ═══════════════════════════════════════════════════════════════════════════════
# 3. SIMULATION LOOP (Runs on GPU)
# ═══════════════════════════════════════════════════════════════════════════════

def run_eval_episode(params, model, u_init, xi_fixed, u_target, config):
    N = config['N']
    L = config['L']
    dt = config['dt']
    steps = config['steps']
    substeps = config['substeps']
    
    # Setup Physics
    dx = L / N
    kx_vec = 2 * jnp.pi * jnp.fft.fftfreq(N, d=dx)
    ky_vec = 2 * jnp.pi * jnp.fft.rfftfreq(N, d=dx)
    KX, KY = jnp.meshgrid(kx_vec, ky_vec, indexing='ij')
    q_sq = KX**2 + KY**2
    L_linear = q_sq - q_sq**2
    
    k_max_x = jnp.max(jnp.abs(kx_vec))
    k_max_y = jnp.max(jnp.abs(ky_vec))
    mask_x = jnp.abs(KX) < (2.0/3.0 * k_max_x)
    mask_y = jnp.abs(KY) < (2.0/3.0 * k_max_y)
    dealias_mask = (mask_x & mask_y).astype(jnp.float32)
    
    etdrk4_coeffs = precompute_etdrk4_coeffs(L_linear, dt)

    def step_fn(carry, _):
        u_hat, u_curr = carry
        u_control = model.apply(params, u_curr, u_target, xi_fixed)
        
        def inner_physics(c_inner, _):
            uh, uc = c_inner
            uh_next, uc_next = ks_spectral_step_etdrk4(
                uh, uc, xi_fixed, u_control,
                KX, KY, etdrk4_coeffs, dealias_mask,
                N=N, L=L, dt=dt, sigma=1.2
            )
            return (uh_next, uc_next), None

        (u_hat_next, u_next), _ = jax.lax.scan(
            inner_physics, (u_hat, u_curr), None, length=substeps
        )
        return (u_hat_next, u_next), u_next

    u_hat_init = jnp.fft.rfftn(u_init)
    (u_hat_final, u_final), traj = jax.lax.scan(
        step_fn, (u_hat_init, u_init), None, length=steps
    )
    return u_final, traj

# ═══════════════════════════════════════════════════════════════════════════════
# 4. QUALITATIVE VISUALIZATION (GPU Compute -> CPU Plot)
# ═══════════════════════════════════════════════════════════════════════════════

def visualize_comparison(params, model, key, L_domain=32.0):
    print("\n--- Running Qualitative Comparison (64 vs 512) ---")
    
    resolutions = [64, 512]
    snapshots = []
    
    # 1. Generate HIGH-RES CHAOS (The Ground Truth)
    # Using N=512 for warmup ensures fine-scale turbulence is present
    u_chaos_high = warmup_high_res_chaos(key, 512, L_domain)
    u_target_high = jnp.zeros_like(u_chaos_high)
    
    n_agents = 100
    grid_dim = int(jnp.sqrt(n_agents))
    x_lin = jnp.linspace(0, L_domain, grid_dim, endpoint=False) + (L_domain/grid_dim)/2
    xv, yv = jnp.meshgrid(x_lin, x_lin)
    xi_fixed = jnp.stack([xv.flatten(), yv.flatten()], axis=-1)

    for res in resolutions:
        print(f"  > Simulating N={res} on {jax.devices()[0].platform.upper()}...")
        
        dt = 0.005 * (64 / res)
        substeps = int(0.1 / dt)
        
        if res == 512:
            u_init = u_chaos_high
            u_target = u_target_high
        else:
            # Downsample the High-Res Chaos to Low-Res Grid
            u_init = jax.image.resize(u_chaos_high, (res, res), method='cubic')
            u_target = jnp.zeros_like(u_init)

        config = {'N': res, 'L': L_domain, 'dt': dt, 'steps': 50, 'substeps': substeps}
        run_jit = jax.jit(partial(run_eval_episode, params, model, config=config))
        
        # 1. RUN ON GPU
        _, traj_gpu = run_jit(u_init, xi_fixed, u_target)
        
        # 2. TRANSFER TO CPU
        traj_cpu = np.array(traj_gpu)
        
        frames = [traj_cpu[0], traj_cpu[25], traj_cpu[49]]
        snapshots.append(frames)

    # 4. Plotting
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    times = ["t=0.0s", "t=2.5s", "t=5.0s"]
    vmin, vmax = -2.0, 2.0 
    
    for i, res in enumerate(resolutions):
        for j, time_label in enumerate(times):
            ax = axes[i, j]
            field = snapshots[i][j]
            im = ax.imshow(field, origin='lower', cmap='RdBu_r', 
                           extent=[0, L_domain, 0, L_domain],
                           vmin=vmin, vmax=vmax)
            if i == 0: ax.set_title(f"{time_label}", fontsize=12)
            if j == 0: ax.set_ylabel(f"Resolution N={res}", fontsize=12, fontweight='bold')
            ax.set_xticks([]); ax.set_yticks([])

    cbar = fig.colorbar(im, ax=axes[:, 2], shrink=0.6)
    cbar.set_label("Vorticity u(x,y)")
    plt.suptitle("Discretization Quality: Zero-Shot Transfer (Chaotic I.C.)", fontsize=16)
    plt.savefig("ks2d_qualitative_chaos.png")
    print("Saved comparison plot.")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. QUANTITATIVE METRICS (GPU Compute -> CPU Float)
# ═══════════════════════════════════════════════════════════════════════════════

def run_quantitative_metrics(params, model, key, L_domain=32.0):
    print("\n--- Running Quantitative Metrics ---")
    resolutions = [64, 128, 256, 512] 
    agent_counts = [64, 100, 121, 144] 
    
    dt_base = 0.005; substeps_base = 20; N_base = 64
    results = []
    
    # Pre-calculate ONE chaotic state to use across all resolutions
    print("Pre-calculating chaotic state...")
    u_chaos_high = warmup_high_res_chaos(key, 512, L_domain)
    
    for res in resolutions:
        print(f"\n--- Testing Resolution: {res}x{res} ---")
        scaling_factor = res / N_base
        dt_scaled = dt_base / scaling_factor
        substeps_scaled = int(dt_base * substeps_base / dt_scaled)
        
        config = {'N': res, 'L': L_domain, 'dt': dt_scaled, 'steps': 50, 'substeps': substeps_scaled}
        run_jit = jax.jit(partial(run_eval_episode, params, model, config=config))
        
        # Resize chaos for this resolution
        u_init = jax.image.resize(u_chaos_high, (res, res), method='cubic')
        u_target = jnp.zeros_like(u_init)

        for n_agents in tqdm(agent_counts, desc="Agents"):
            grid_dim = int(jnp.sqrt(n_agents))
            x_lin = jnp.linspace(0, L_domain, grid_dim, endpoint=False) + (L_domain/grid_dim)/2
            xv, yv = jnp.meshgrid(x_lin, x_lin)
            xi_fixed = jnp.stack([xv.flatten(), yv.flatten()], axis=-1)
            
            # RUN
            u_final_gpu, _ = run_jit(u_init, xi_fixed, u_target)
            mse_cpu = float(jnp.mean((u_final_gpu - u_target)**2))
            
            results.append({
                "Resolution": res,
                "Agents": n_agents,
                "MSE": mse_cpu,
                "dt": dt_scaled
            })

    df = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Agents", y="MSE", hue="Resolution", marker="o", palette="viridis")
    plt.yscale('log')
    plt.title("Zero-Shot Resolution Generalization (Chaotic I.C.)")
    plt.ylabel("Final MSE")
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig("ks2d_resolution_scaling_chaos.png")
    print("Saved quantitative plot.")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    key = jax.random.PRNGKey(55) # New Seed
    model = ScaleInvariantPolicy(
        training_res=64,
        features=(64, 128),  
        u_max=5.0,           
        domain_size=(32.0, 32.0)
    )

    try:
        with open('ks2d_centralized_params.msgpack', 'rb') as f:
            params = flax.serialization.from_bytes(
                model.init(key, jnp.zeros((64, 64)), jnp.zeros((64, 64)), jnp.zeros((10, 2))),
                f.read()
            )
        print("Params loaded successfully.")
    except FileNotFoundError:
        print("Warning: Params not found. Using random initialization.")
        params = model.init(key, jnp.zeros((64, 64)), jnp.zeros((64, 64)), jnp.zeros((10, 2)))

    # run_quantitative_metrics(params, model, key)
    visualize_comparison(params, model, key)