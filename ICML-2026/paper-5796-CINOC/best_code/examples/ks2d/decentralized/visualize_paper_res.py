"""
Paper-Quality Visualization for KS-2D Resolution Invariance Test.

Creates a publication-ready figure demonstrating discretization invariance:
- Row 1: Resolution 64×64 (Training Resolution)
- Row 2: Resolution 512×512 (High-Fidelity)
- 5 Time Steps: t=0.00s, t=1.25s, t=2.50s, t=3.75s, t=5.00s

Requires pre-computed trajectory data from test_resolution_invariance.py.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import sys
import flax.serialization
from pathlib import Path
from functools import partial

# Use default device (GPU if available)
jax.config.update("jax_enable_x64", True)

# --- Path Setup ---
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

try:
    from tesseracts.ks2d.solver import ks_spectral_step_etdrk4, precompute_etdrk4_coeffs
    from models.policy_ks2d import DecentralizedKS2DControlNet
except ImportError:
    print("Warning: Could not import tesseracts or models. Ensure paths are correct.")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    'L_domain': 32.0,
    'resolutions': [64, 512],           # Training vs High-Fidelity
    'snapshot_times': [0.0, 1.25, 2.5, 3.75, 5.0],  # 5 time steps
    'n_agents': 100,
    'v_lim': 2.0,                        # Color limits
    'params_file': 'ks2d_centralized_params.msgpack'
}


# ═══════════════════════════════════════════════════════════════════════════════
# PAPER STYLE
# ═══════════════════════════════════════════════════════════════════════════════

def setup_paper_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


# ═══════════════════════════════════════════════════════════════════════════════
# RESOLUTION ADAPTER (from test_resolution_invariance.py)
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
# WARMUP & SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def warmup_high_res_chaos(key, N, L, warmup_time=200.0, dt=0.005):
    """Generate chaotic initial condition via warmup evolution."""
    x = jnp.linspace(0, L, N, endpoint=False)
    X, Y = jnp.meshgrid(x, x)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    phase1_x = jax.random.uniform(k1, minval=0, maxval=2*jnp.pi)
    phase1_y = jax.random.uniform(k2, minval=0, maxval=2*jnp.pi)
    u = jnp.sin(2*jnp.pi*X/L + phase1_x) * jnp.cos(2*jnp.pi*Y/L + phase1_y)

    phase2 = jax.random.uniform(k3, minval=0, maxval=2*jnp.pi)
    u += 0.5 * jnp.sin(4*jnp.pi*X/L + phase2)
    u += 0.05 * jax.random.normal(k4, shape=(N, N))
    u = (u - jnp.mean(u)) / (jnp.std(u) + 1e-6)

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
    (u_hat_final, u_final), _ = jax.lax.scan(
        warmup_step, (u_hat, u), None, length=steps
    )

    return u_final


def run_eval_episode(params, model, u_init, xi_fixed, u_target, config):
    """Run controlled simulation and return trajectory."""
    N = config['N']
    L = config['L']
    dt = config['dt']
    steps = config['steps']
    substeps = config['substeps']

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
# PAPER-QUALITY FIGURE
# ═══════════════════════════════════════════════════════════════════════════════

def create_resolution_figure(snapshots_dict, save_name="ks2d_resolution_invariance_paper"):
    """
    Create the paper-quality resolution invariance figure.

    Args:
        snapshots_dict: {resolution: list of 5 snapshot arrays}
    """
    setup_paper_style()

    resolutions = CONFIG['resolutions']
    times = CONFIG['snapshot_times']
    n_cols = len(times)
    n_rows = len(resolutions)
    v_lim = CONFIG['v_lim']
    L = CONFIG['L_domain']

    # Figure size optimized for paper (wider aspect)
    fig = plt.figure(figsize=(12, 5))

    # GridSpec: 2 rows x 5 cols + colorbar
    gs = gridspec.GridSpec(n_rows, n_cols + 1, width_ratios=[1]*n_cols + [0.05],
                           hspace=0.08, wspace=0.03)

    cf = None
    for row_idx, res in enumerate(resolutions):
        frames = snapshots_dict[res]
        for col_idx, (t_val, frame) in enumerate(zip(times, frames)):
            ax = fig.add_subplot(gs[row_idx, col_idx])

            # Plot field
            cf = ax.imshow(
                frame, origin='lower', cmap='RdBu_r',
                extent=[0, L, 0, L],
                vmin=-v_lim, vmax=v_lim,
                aspect='equal'
            )

            # Column titles (time)
            if row_idx == 0:
                ax.set_title(f"t={t_val:.2f}s", fontsize=14, pad=8)

            # Row labels (resolution)
            if col_idx == 0:
                ax.set_ylabel(f"Resolution N={res}", fontsize=14, fontweight='bold')

            # Clean axes
            ax.set_xticks([])
            ax.set_yticks([])

            # Border styling
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(0.8)

    # Colorbar on the right
    cax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(cf, cax=cax)
    # cbar.set_label(r"$u(x, y)$", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    # Suptitle
    fig.suptitle("Discretization Quality: Zero-Shot Transfer (5 Time Steps)", 
                 fontsize=16, fontweight='bold', y=1.02)

    # Save
    pdf_path = f"{save_name}.pdf"
    png_path = f"{save_name}.png"
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    print(f"✓ Saved: {pdf_path}")
    print(f"✓ Saved: {png_path}")

    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def run_and_visualize():
    """Run simulations at both resolutions and create paper figure."""
    print("=" * 60)
    print("KS-2D Resolution Invariance Paper Figure")
    print("=" * 60)

    # Create output directory
    output_dir = Path("figures/images/resolution")
    output_dir.mkdir(parents=True, exist_ok=True)

    key = jax.random.PRNGKey(55)
    L_domain = CONFIG['L_domain']

    # Initialize model
    model = ScaleInvariantPolicy(
        training_res=64,
        features=(64, 128),
        u_max=5.0,
        domain_size=(L_domain, L_domain)
    )

    # Load parameters
    params_path = Path(__file__).parent / CONFIG['params_file']
    try:
        with open(params_path, 'rb') as f:
            params = flax.serialization.from_bytes(
                model.init(key, jnp.zeros((64, 64)), jnp.zeros((64, 64)), jnp.zeros((10, 2))),
                f.read()
            )
        print(f"✓ Parameters loaded from {params_path}")
    except FileNotFoundError:
        print(f"Error: {params_path} not found. Using random params.")
        params = model.init(key, jnp.zeros((64, 64)), jnp.zeros((64, 64)), jnp.zeros((10, 2)))

    # Generate chaotic IC at high resolution
    print("\nGenerating chaotic initial condition...")
    u_chaos_high = warmup_high_res_chaos(key, 512, L_domain)
    u_target_high = jnp.zeros_like(u_chaos_high)

    # Fixed actuator grid
    n_agents = CONFIG['n_agents']
    grid_dim = int(jnp.sqrt(n_agents))
    x_lin = jnp.linspace(0, L_domain, grid_dim, endpoint=False) + (L_domain/grid_dim)/2
    xv, yv = jnp.meshgrid(x_lin, x_lin)
    xi_fixed = jnp.stack([xv.flatten(), yv.flatten()], axis=-1)

    snapshots_dict = {}

    for res in CONFIG['resolutions']:
        print(f"\n--- Simulating Resolution N={res} ---")

        # Time parameters (scale dt with resolution)
        dt = 0.005 * (64 / res)
        substeps = int(0.1 / dt)  # 0.1s per control step
        total_time = 5.0
        steps = int(total_time / 0.1)  # Number of control steps

        # Prepare initial condition
        if res == 512:
            u_init = u_chaos_high
            u_target = u_target_high
        else:
            u_init = jax.image.resize(u_chaos_high, (res, res), method='cubic')
            u_target = jnp.zeros_like(u_init)

        config = {'N': res, 'L': L_domain, 'dt': dt, 'steps': steps, 'substeps': substeps}
        run_jit = jax.jit(partial(run_eval_episode, params, model, config=config))

        print(f"  Running controlled simulation (T=5.0s)...")
        _, traj_gpu = run_jit(u_init, xi_fixed, u_target)
        traj_cpu = np.array(traj_gpu)

        # Extract snapshots at specified times
        # traj has shape (steps,) where each step = 0.1s
        # times = [0.0, 1.25, 2.5, 3.75, 5.0]
        # indices = [0, 12.5, 25, 37.5, 50] -> round to [0, 12, 25, 37, 49]
        snap_indices = [int(t / 0.1) for t in CONFIG['snapshot_times']]
        snap_indices = [min(i, steps-1) for i in snap_indices]  # Clamp to valid range
        
        frames = [traj_cpu[i] for i in snap_indices]
        snapshots_dict[res] = frames
        print(f"  ✓ Extracted {len(frames)} snapshots at indices {snap_indices}")

    # Create paper figure
    print("\nCreating paper figure...")
    create_resolution_figure(snapshots_dict, save_name="figures/images/resolution/ks2d_resolution_invariance_paper")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    run_and_visualize()
