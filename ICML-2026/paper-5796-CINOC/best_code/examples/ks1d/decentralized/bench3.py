import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import flax.linen as nn
import flax.serialization
from flax.serialization import msgpack_restore, from_state_dict
import sys
import os
from pathlib import Path

# --- 1. Model Definitions & Paths ---
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

output_dir = Path("figures/images/bench_new")
output_dir.mkdir(parents=True, exist_ok=True)

bench_models_dir = Path("bench/models")
bench_models_dir.mkdir(parents=True, exist_ok=True) 

from examples.ks1d.decentralized.dynamics_dual import PDEDynamics 
from models.policy_ks1d import DecentralizedControlNet
from examples.ks1d.decentralized.data_utils import get_batch_initial_conditions
from examples.ks1d.decentralized.bench.env_ks import extract_patches_jit
from examples.ks1d.decentralized.bench.utils_hypemarl import get_sinusoidal_encoding
from examples.ks1d.decentralized.bench.models_marl import MARLActor
from examples.ks1d.decentralized.bench.models_rl import CentralizedActor
from examples.ks1d.decentralized.bench.models_ppo import PPOActor1D
from examples.ks1d.decentralized.bench.models_mappo import MAPPOActor1D
from examples.ks1d.decentralized.bench.models_dpc import CentralizedMLPControlNet1D_KS

# KS Specific Settings
N_grid, L_domain, n_agents = 128, 22.0, 8
n_sensors = 8
T_steps, N_eval, dt = 200, 100, 0.05
STATE_NORM_FACTOR = 5.0  # Used by PPO and MAPPO

bench_registry = {}

# --- 2. Loading Logic ---
def load_params(filename, model, dummy_input):
    if not os.path.exists(filename):
        print(f"[-] {filename} not found. Skipping benchmark.")
        return None
        
    with open(filename, 'rb') as f:
        bytes_data = f.read()
        
    variables = model.init(jax.random.PRNGKey(0), *dummy_input)
    
    try:
        state_dict = msgpack_restore(bytes_data)
    except Exception as e:
        print(f"[-] Could not parse msgpack for {filename}: {e}")
        return None

    if 'actor' in state_dict: 
        state_dict = state_dict['actor']
        
    if 'params' in variables and 'params' not in state_dict:
        state_dict = {'params': state_dict}
    elif 'params' not in variables and 'params' in state_dict:
        state_dict = state_dict['params']

    try:
        return from_state_dict(variables, state_dict)
    except Exception as e:
        print(f"[-] Failed to load weights for {filename}. Mismatch error: {e}")
        try:
            return flax.serialization.from_bytes(variables, bytes_data)
        except Exception as e2:
            print(f"[-] Fallback also failed: {e2}")
            return None

print("Loading Models...")

xi_single = jnp.linspace(0.0, L_domain, n_agents, endpoint=False) + (L_domain/n_agents)/2

# 1. CINOC (Centralized Training Output)
CINOC_model = DecentralizedControlNet(features=(64, 64), L_domain=L_domain)
CINOC_p = load_params('ks_centralized_params.msgpack', CINOC_model, (jnp.zeros(N_grid), jnp.zeros(N_grid), xi_single))
if CINOC_p:
    bench_registry['CINOC'] = {'apply': CINOC_model.apply, 'params': CINOC_p, 'color': 'blue'}

# 2. MARL 
marl_model = MARLActor()
marl_mu_val = jnp.array([L_domain, 0.05]) 
marl_pe_val = get_sinusoidal_encoding(xi_single, d=64)
marl_p = load_params(bench_models_dir / 'marl_standard_params.msgpack', marl_model, (jnp.zeros((n_agents, 106)),))

if marl_p:
    def marl_apply(p, u, target, xi):
        y = extract_patches_jit(u, target, xi/L_domain, window_size=4)
        mu_context = jnp.tile(marl_mu_val, (n_agents, 1))
        full_input = jnp.concatenate([y, mu_context, marl_pe_val], axis=-1)
        return marl_model.apply(p, full_input).flatten()
    bench_registry['MARL'] = {'apply': marl_apply, 'params': marl_p, 'color': 'orange'}

# 3. RL 
rl_model = CentralizedActor()
rl_p = load_params(bench_models_dir / 'rl_centralized_params.msgpack', rl_model, (jnp.zeros(N_grid),))

if rl_p:
    bench_registry['RL'] = {'apply': lambda p, u, t, xi: rl_model.apply(p, u), 'params': rl_p, 'color': 'purple'}

# 4. PPO 
ppo_model = PPOActor1D(n_agents=n_agents)
ppo_p = load_params(bench_models_dir / 'ppo_ks1d_params.msgpack', ppo_model, (jnp.zeros(N_grid),))

if ppo_p:
    def ppo_apply(p, u, target, xi):
        u_norm = u / STATE_NORM_FACTOR
        mean, _ = ppo_model.apply(p, u_norm)
        return mean
    bench_registry['PPO'] = {'apply': ppo_apply, 'params': ppo_p, 'color': 'cyan'}

# 5. MAPPO
mappo_model = MAPPOActor1D(n_agents=n_agents)
dummy_mappo_obs = jnp.zeros((n_agents, N_grid + 1))
mappo_p = load_params(bench_models_dir / 'mappo_ks1d_params.msgpack', mappo_model, (dummy_mappo_obs,))

if mappo_p:
    def mappo_apply(p, u, target, xi):
        u_norm = u / STATE_NORM_FACTOR
        u_tiled = jnp.tile(u_norm, (n_agents, 1))
        xi_tiled = jnp.expand_dims(xi / L_domain, axis=-1)
        
        obs = jnp.concatenate([u_tiled, xi_tiled], axis=-1)
        mean, _ = mappo_model.apply(p, obs)
        return mean.squeeze(-1)
    bench_registry['MAPPO'] = {'apply': mappo_apply, 'params': mappo_p, 'color': 'brown'}

# 6. Centralized DPC (MLP)
dpc_model = CentralizedMLPControlNet1D_KS(hidden_dim=256, n_agents=n_agents, u_max=1.0)
dpc_dummy_inputs = (jnp.zeros(N_grid), jnp.zeros(N_grid), xi_single)
dpc_path = bench_models_dir / 'dpc_ks1d_params.msgpack'

dpc_p = None
if os.path.exists(dpc_path):
    with open(dpc_path, 'rb') as f:
        dpc_bytes = f.read()
    
    # 1. Restore the raw dictionary
    raw_dict = msgpack_restore(dpc_bytes)
    
    # 2. Peel back the nested wrappers to get to the raw Dense layers
    state_dict = raw_dict
    if 'params' in state_dict:
        state_dict = state_dict['params']
    if 'params' in state_dict:  # Catch the double-nesting trap!
        state_dict = state_dict['params']
        
    # 3. Initialize variables to get the target structure
    variables = dpc_model.init(jax.random.PRNGKey(0), *dpc_dummy_inputs)
    
    # 4. Safely load the unwrapped dictionary into the initialized structure
    try:
        # Re-wrap with exactly one 'params' key so Flax is happy
        dpc_p = from_state_dict(variables, {'params': state_dict})
        print("[+] Successfully loaded DPC")
    except Exception as e:
        print(f"[-] Failed to load DPC: {e}")
else:
    print(f"[-] {dpc_path} not found.")

if dpc_p:
    def dpc_apply(p, u, target, xi):
        # DPC Model returns u_out directly for the fixed actuators
        return dpc_model.apply(p, u, target, xi)
    bench_registry['DPC'] = {'apply': dpc_apply, 'params': dpc_p, 'color': 'magenta'}

    
# Uncontrolled
bench_registry['Uncontrolled'] = {
    'apply': lambda p, u, t, xi: jnp.zeros((n_agents,)), 
    'params': None, 'color': 'red'
}

# --- 3. Simulation & Adapted Evaluation ---
print(f"Generating Data & Running Simulations for {list(bench_registry.keys())}...")
key = jax.random.PRNGKey(1234)
u_init_batch = get_batch_initial_conditions(jax.random.split(key)[1], N_eval, N_grid, L_domain)
xi_batch = jnp.tile(xi_single, (N_eval, 1))
u_target_batch = jnp.zeros_like(u_init_batch)

@jax.jit(static_argnames=['name'])
def run_sim(name, u_i, xi_i, u_t):
    dyn = PDEDynamics(policy_apply_fn=bench_registry[name]['apply'])
    u_traj, xi_traj, u_ctrl_traj, v_traj = dyn.unroll_controlled(
        u_i, xi_i, u_t, 
        bench_registry[name]['params'], 
        T_steps, N_grid, L_domain, dt=dt, sigma=1.0
    )
    return u_traj, u_ctrl_traj

for name in bench_registry:
    print(f"Running {name} unrolls...")
    u_traj_data, u_ctrl_data = jax.vmap(lambda u, x, t: run_sim(name, u, x, t))(u_init_batch, xi_batch, u_target_batch)
    bench_registry[name]['u_traj'] = u_traj_data
    bench_registry[name]['u_ctrl'] = u_ctrl_data

# --- 4. Metrics & Results Printing ---
print("\n" + "="*85)
print(f"{'Method':<15} | {'Tracking Error (l_track)':<25} | {'Control Effort (l_effort)':<25}")
print("-" * 85)

for name in bench_registry:
    u_traj = bench_registry[name]['u_traj']
    u_ctrl = bench_registry[name]['u_ctrl']
    
    l_track_batch = jnp.mean((u_traj - u_target_batch[:, None, :]) ** 2, axis=(1, 2))
    mean_track, std_track = jnp.mean(l_track_batch), jnp.std(l_track_batch)
    
    l_effort_batch = jnp.mean(u_ctrl ** 2, axis=(1, 2))
    mean_effort, std_effort = jnp.mean(l_effort_batch), jnp.std(l_effort_batch)
    
    print(f"{name:<15} | {mean_track:.6f} ±{2*std_track:.6f} | {mean_effort:.6f} ±{2*std_effort:.6f}")

print("="*85)

# --- 5. Plotting & PDF Export ---
print("Saving individual state plots to PDF...")
for name in bench_registry:
    plt.figure(figsize=(8, 5))
    plt.imshow(bench_registry[name]['u_traj'][0].T, aspect='auto', origin='lower', 
               extent=[0, T_steps*dt, 0, L_domain], cmap='RdBu_r', vmin=-3, vmax=3)
    plt.colorbar(label='u(x,t)')
    plt.title(f'Controlled State: {name}')
    plt.xlabel('Time (s)')
    plt.ylabel('Space (x)')
    plt.tight_layout()
    plt.savefig(output_dir / f"state_{name.lower().replace(' ', '_')}.pdf")
    plt.close()

plt.figure(figsize=(18, 8))

# 1. Boxplot (Tracking Error)
plt.subplot(1, 2, 1)
data_boxplot = [jnp.mean((bench_registry[n]['u_traj'] - u_target_batch[:, None, :])**2, axis=(1, 2)) for n in bench_registry]
plt.boxplot(data_boxplot, labels=list(bench_registry.keys()))
plt.yscale('log')
plt.title('Tracking Error (Log Scale)')
plt.ylabel('MSE vs Target')
plt.grid(True, alpha=0.3)

# 2. Tracking Error Evolution 
plt.subplot(1, 2, 2)
time_axis = jnp.arange(T_steps) * dt
for name in bench_registry:
    if name == 'Uncontrolled': continue 
    
    evol = jnp.mean((bench_registry[name]['u_traj'] - u_target_batch[:, None, :])**2, axis=(0, 2))
    plt.plot(time_axis, evol, label=name, color=bench_registry[name]['color'], lw=2.5)

plt.yscale('log')
plt.title('Stabilization Error Evolution (Controlled Only)')
plt.xlabel('Time (s)')
plt.ylabel('Mean Tracking Error (Log)')
plt.legend()
plt.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "ks_resilient_results.png")
print(f"\nSummary results saved to {output_dir}/ks_resilient_results.png")