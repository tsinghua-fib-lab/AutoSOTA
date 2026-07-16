"""
Evaluation Script: KS-1D Stabilization Benchmark
Compares Differentiable Predictive Control (DPC), HypeMARL, and Uncontrolled Chaos.
"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import flax.serialization
from flax.serialization import msgpack_restore, from_state_dict
import sys
from pathlib import Path

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

# Import KS specific modules
from examples.ks1d.decentralized.dynamics_dual import PDEDynamics 
from models.policy_ks1d import DecentralizedControlNet
from examples.ks1d.decentralized.data_utils import get_batch_initial_conditions

# Import HypeMARL specific modules
from models_hypemarl import HyperActor
from examples.ks1d.decentralized.bench.utils_hypemarl import get_sinusoidal_encoding
from examples.ks1d.decentralized.bench.env_ks import extract_patches_jit

# --- 1. Configuration ---
N_grid = 128
L_domain = 22.0
n_agents = 8
T_steps = 200
N_eval = 100

# Initialize DPC Model Structure
dpc_model = DecentralizedControlNet(features=(64, 64), L_domain=L_domain)

# Initialize HypeMARL Actor Structure
hyper_actor = HyperActor()
local_y_dim = 40  # 20 error + 20 grad
n_mu = 2          # [L, dt]
pe_dim = 2048

# --- 2. Helpers ---
def zero_policy_apply(params, u_obs, u_target, xi_fixed):
    """Dummy policy for Uncontrolled baseline."""
    n_agents = xi_fixed.shape[0]
    return jnp.zeros((n_agents,))

# Precompute static HypeMARL embeddings (z) to save time during evaluation
hm_mu = jnp.array([L_domain, 0.05])
xi_fixed_single = jnp.linspace(0.0, L_domain, n_agents, endpoint=False) + (L_domain/n_agents)/2
hm_pe = get_sinusoidal_encoding(xi_fixed_single, d=pe_dim)
hm_z = jnp.concatenate([hm_pe, jnp.tile(hm_mu, (n_agents, 1))], axis=-1)

def hypemarl_policy_apply(hm_params, u_obs, u_target, xi_fixed):
    """Wraps the global PDE state into local patches for the HyperActor."""
    xi_norm = xi_fixed / L_domain
    
    # Extract the same 40-dim local patches used during training
    y_local = extract_patches_jit(u_obs, u_target, xi_norm, window_size=4)
    
    # Forward pass through Hypernetwork -> Primary Network
    actions = hyper_actor.apply(hm_params, hm_z, y_local)
    return actions

# --- 3. Data Generation ---
print(f"Generating {N_eval} Chaotic Initial Conditions...")
key = jax.random.PRNGKey(1234) # Validation Seed

key, subkey = jax.random.split(key)
u_init_batch = get_batch_initial_conditions(subkey, N_eval, N_grid, L_domain)
u_target_batch = jnp.zeros_like(u_init_batch) 
xi_fixed_batch = jnp.tile(xi_fixed_single, (N_eval, 1))

# --- 4. Load Models ---
print("Loading trained parameters...")

# 4a. Load DPC Params
try:
    with open('ks_centralized_params.msgpack', 'rb') as f:
        dpc_bytes = f.read()
except FileNotFoundError:
    print("Error: 'ks_centralized_params.msgpack' not found.")
    sys.exit(1)

dummy_u = jnp.zeros((N_grid,))
dummy_dpc_params = dpc_model.init(jax.random.PRNGKey(0), dummy_u, dummy_u, xi_fixed_single)
dpc_params = flax.serialization.from_bytes(dummy_dpc_params, dpc_bytes)

# 4b. Load HypeMARL Params
try:
    with open('hypemarl_params.msgpack', 'rb') as f:
        hm_bytes = f.read()
except FileNotFoundError:
    print("Error: 'hypemarl_params.msgpack' not found.")
    sys.exit(1)

dummy_y = jnp.zeros((n_agents, local_y_dim))
dummy_hm_params = hyper_actor.init(jax.random.PRNGKey(0), hm_z, dummy_y)

# The training script saved a dict of {'actor': ..., 'critic1': ...}
# We restore the dictionary and extract just the actor weights
parsed_hm_dict = msgpack_restore(hm_bytes)
hm_params = from_state_dict(dummy_hm_params, parsed_hm_dict['actor'])

# --- 5. Evaluation Loop ---
dynamics_dpc = PDEDynamics(policy_apply_fn=dpc_model.apply)
dynamics_hm = PDEDynamics(policy_apply_fn=hypemarl_policy_apply)
dynamics_unc = PDEDynamics(policy_apply_fn=zero_policy_apply)

print("Running simulations...")

@jax.jit
def run_comparison(u_init, xi_fixed, u_target):
    # DPC (Centralized)
    u_c_dpc, _, _, _ = dynamics_dpc.unroll_controlled(
        u_init, xi_fixed, u_target, dpc_params, T_steps, N_grid, L_domain, dt=0.05, sigma=1.0
    )
    # HypeMARL (Decentralized)
    u_c_hm, _, _, _ = dynamics_hm.unroll_controlled(
        u_init, xi_fixed, u_target, hm_params, T_steps, N_grid, L_domain, dt=0.05, sigma=1.0
    )
    # Uncontrolled
    u_u, _, _, _ = dynamics_unc.unroll_controlled(
        u_init, xi_fixed, u_target, None, T_steps, N_grid, L_domain, dt=0.05, sigma=1.0
    )
    return u_c_dpc, u_c_hm, u_u

# Batch execution
u_dpc_all, u_hm_all, u_unc_all = jax.vmap(run_comparison)(u_init_batch, xi_fixed_batch, u_target_batch)

# --- 6. Metrics ---
print("Calculating metrics...")

energy_dpc = jnp.mean(u_dpc_all**2, axis=2) 
energy_hm = jnp.mean(u_hm_all**2, axis=2)
energy_unc = jnp.mean(u_unc_all**2, axis=2)

final_energy_dpc = energy_dpc[:, -1]
final_energy_hm = energy_hm[:, -1]
final_energy_unc = energy_unc[:, -1]

print(f"Mean Final Energy (DPC Baseline): {jnp.mean(final_energy_dpc):.6f}")
print(f"Mean Final Energy (HypeMARL):     {jnp.mean(final_energy_hm):.6f}")
print(f"Mean Final Energy (Uncontrolled): {jnp.mean(final_energy_unc):.6f}")

# --- 7. Plotting ---
plt.figure(figsize=(18, 12))

# 1. Energy Distribution (Boxplot)
plt.subplot(2, 3, 1)
plt.boxplot(
    [final_energy_dpc, final_energy_hm, final_energy_unc], 
    labels=['DPC', 'HypeMARL', 'Uncontrolled']
)
plt.yscale('log')
plt.title(f'Final System Energy (Log Scale, N={N_eval})')
plt.ylabel('Mean Squared Field Value')
plt.grid(True, alpha=0.3)

# 2. Energy Evolution (Mean over batch)
plt.subplot(2, 3, 2)
time_axis = jnp.arange(T_steps) * 0.05
# Uncontrolled
plt.plot(time_axis, jnp.mean(energy_unc, axis=0), 'r-', label='Uncontrolled', linewidth=2)
plt.fill_between(time_axis, jnp.percentile(energy_unc, 25, axis=0), jnp.percentile(energy_unc, 75, axis=0), color='red', alpha=0.1)
# DPC
plt.plot(time_axis, jnp.mean(energy_dpc, axis=0), 'b-', label='DPC', linewidth=2)
plt.fill_between(time_axis, jnp.percentile(energy_dpc, 25, axis=0), jnp.percentile(energy_dpc, 75, axis=0), color='blue', alpha=0.1)
# HypeMARL
plt.plot(time_axis, jnp.mean(energy_hm, axis=0), 'g-', label='HypeMARL', linewidth=2)
plt.fill_between(time_axis, jnp.percentile(energy_hm, 25, axis=0), jnp.percentile(energy_hm, 75, axis=0), color='green', alpha=0.1)

plt.title('System Energy Evolution')
plt.xlabel('Time (s)')
plt.ylabel('Energy (L2)')
plt.legend()
plt.grid(True)

# 3. Space-Time Heatmap (Uncontrolled)
sample_idx = 0
plt.subplot(2, 3, 3)
plt.imshow(u_unc_all[sample_idx].T, aspect='auto', origin='lower', 
           extent=[0, T_steps*0.05, 0, L_domain], cmap='RdBu_r', vmin=-3, vmax=3)
plt.colorbar(label='u(x,t)')
plt.title('Uncontrolled Chaos')
plt.xlabel('Time (s)')

# 4. Space-Time Heatmap (DPC)
plt.subplot(2, 3, 4)
plt.imshow(u_dpc_all[sample_idx].T, aspect='auto', origin='lower', 
           extent=[0, T_steps*0.05, 0, L_domain], cmap='RdBu_r', vmin=-3, vmax=3)
plt.colorbar(label='u(x,t)')
plt.title('DPC (Centralized)')
plt.xlabel('Time (s)')
plt.ylabel('Domain (x)')

# 5. Space-Time Heatmap (HypeMARL)
plt.subplot(2, 3, 5)
plt.imshow(u_hm_all[sample_idx].T, aspect='auto', origin='lower', 
           extent=[0, T_steps*0.05, 0, L_domain], cmap='RdBu_r', vmin=-3, vmax=3)
plt.colorbar(label='u(x,t)')
plt.title('HypeMARL (Decentralized)')
plt.xlabel('Time (s)')

# 6. Blank (to keep the grid clean)
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

plt.tight_layout()
save_path = Path("figures/images/bench") / "benchmark_comparison_results.png"
save_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(save_path)
print(f"Comparison plot saved to '{save_path}'")