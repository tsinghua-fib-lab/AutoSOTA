import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import flax.serialization
from flax.serialization import msgpack_restore, from_state_dict
import sys
import os
from pathlib import Path
from typing import Sequence
from functools import partial

# --- 1. Configuration & Paths ---
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

output_dir = Path("figures/bench_heat")
output_dir.mkdir(parents=True, exist_ok=True)

bench_models_dir = Path("bench/models")
bench_models_dir.mkdir(parents=True, exist_ok=True) 

# Base Environment & Utils
from examples.heat1d.decentralized.dynamics_dual import PDEDynamics 
from examples.heat1d.decentralized.data_utils import generate_grf
from examples.heat1d.decentralized.bench.utils_hypemarl import get_sinusoidal_encoding

from models.policy import DecentralizedControlNet
from examples.heat1d.decentralized.bench.models_marl import MARLActor
from examples.heat1d.decentralized.bench.models_rl import CentralizedActor
from examples.heat1d.decentralized.bench.models_ppo import PPOActor
from examples.heat1d.decentralized.bench.models_mappo import MAPPOActor
from examples.heat1d.decentralized.bench.models_dpc import CentralizedMLPControlNet

N_grid, L_domain, n_agents = 100, 1.0, 8
T_steps, N_eval = 300, 50
ENV_MU = jnp.array([0.01]) 

# --- Shared Patch Extractor ---
@partial(jax.jit, static_argnames=['window_size'])
def extract_patches_jit(full_state, target_state, xi_norm, window_size):
    error = full_state - target_state
    error_grad = jnp.gradient(error)
    half_window = window_size // 2
    
    padded_error = jnp.pad(error, (half_window, half_window), mode='constant', constant_values=0.0)
    padded_grad = jnp.pad(error_grad, (half_window, half_window), mode='constant', constant_values=0.0)
    
    def get_local_obs(xi_n):
        center_idx = jax.lax.stop_gradient((xi_n * (N_grid - 1)).astype(int)) + half_window
        start = center_idx - half_window
        p_err = jax.lax.dynamic_slice(padded_error, (start,), (window_size,))
        p_grad = jax.lax.dynamic_slice(padded_grad, (start,), (window_size,))
        p_err = jax.image.resize(p_err, (20,), method='bilinear')
        p_grad = jax.image.resize(p_grad, (20,), method='bilinear')
        return jnp.concatenate([p_err, p_grad])
        
    return jax.vmap(get_local_obs)(xi_norm)

bench_registry = {}

# --- 2. Loading Logic ---
def load_params(filename, model, dummy_input):
    if not os.path.exists(filename):
        print(f"[-] {filename} not found.")
        return None
    with open(filename, 'rb') as f: bytes_data = f.read()
    variables = model.init(jax.random.PRNGKey(0), *dummy_input)
    try:
        state_dict = msgpack_restore(bytes_data)
        if 'actor' in state_dict: state_dict = state_dict['actor']
        if 'params' in variables and 'params' not in state_dict: state_dict = {'params': state_dict}
        elif 'params' not in variables and 'params' in state_dict: state_dict = state_dict['params']
        return from_state_dict(variables, state_dict)
    except Exception as e:
        print(f"[-] Failed to load {filename}: {e}")
        return None

print("Loading Models...")
xi_init = jnp.linspace(0.2, 0.8, n_agents)

# 1. CINOC
CINOC_model = DecentralizedControlNet(features=(64, 64))
CINOC_p = load_params('decentralized_params.msgpack', CINOC_model, (jnp.zeros(N_grid), jnp.zeros(N_grid), xi_init))
if CINOC_p:
    bench_registry['CINOC'] = {'apply': CINOC_model.apply, 'params': CINOC_p, 'color': 'blue'}


# 2. MARL (TD3)
marl_model = MARLActor()
marl_dummy_input = (jnp.zeros((n_agents, 2089)),)
marl_p = load_params(bench_models_dir / 'marl_heat_params.msgpack', marl_model, marl_dummy_input)

if marl_p:
    def marl_apply(p, z, target, xi):
        y = extract_patches_jit(z, target, xi/L_domain, window_size=8)
        pe = get_sinusoidal_encoding(xi, d=128)
        obs = jnp.concatenate([y, jnp.tile(ENV_MU, (n_agents, 1)), pe], axis=-1)
        action = marl_model.apply(p, obs)
        return action[:, 0], action[:, 1]
    
    bench_registry['MARL (TD3)'] = {'apply': marl_apply, 'params': marl_p, 'color': 'orange'}


# 3. PPO (Centralized global state inputs: z, target, xi)
ppo_model = PPOActor(n_agents=n_agents)
ppo_dummy_input = (jnp.zeros(N_grid), jnp.zeros(N_grid), jnp.zeros(n_agents))
ppo_p = load_params(bench_models_dir / 'ppo_heat1d_params.msgpack', ppo_model, ppo_dummy_input)

if ppo_p:
    def ppo_apply(p, z, target, xi):
        # Extract mean, ignore standard deviation for eval
        mean, _ = ppo_model.apply(p, z, target, xi)
        return mean[:, 0], mean[:, 1]
        
    bench_registry['PPO'] = {'apply': ppo_apply, 'params': ppo_p, 'color': 'magenta'}


# 4. MAPPO (Decentralized patch inputs: obs)
mappo_model = MAPPOActor(n_agents=n_agents)
mappo_dummy_input = (jnp.zeros((n_agents, 128 + 41)),)
mappo_p = load_params(bench_models_dir / 'mappo_heat_params_new.msgpack', mappo_model, mappo_dummy_input)

if mappo_p:
    def mappo_apply(p, z, target, xi):
        y = extract_patches_jit(z, target, xi/L_domain, window_size=8)
        pe = get_sinusoidal_encoding(xi, d=128)
        obs = jnp.concatenate([y, jnp.tile(ENV_MU, (n_agents, 1)), pe], axis=-1)
        
        # Extract mean, ignore standard deviation for eval
        mean, _ = mappo_model.apply(p, obs) 
        return mean[:, 0], mean[:, 1]
        
    bench_registry['MAPPO'] = {'apply': mappo_apply, 'params': mappo_p, 'color': 'cyan'}


# 5. RL Centralized (Centralized global state inputs: z, target, xi)
rl_model = CentralizedActor(n_agents=n_agents)
rl_dummy_input = (jnp.zeros(N_grid), jnp.zeros(N_grid), jnp.zeros(n_agents))
rl_p = load_params(bench_models_dir / 'rl_heat_params.msgpack', rl_model, rl_dummy_input)

if rl_p:
    def rl_apply(p, z, target, xi):
        action = rl_model.apply(p, z, target, xi)
        return action[:, 0], action[:, 1]
    bench_registry['Centralized RL'] = {'apply': rl_apply, 'params': rl_p, 'color': 'purple'}


# 6. Centralized DPC (MLP)
dpc_model = CentralizedMLPControlNet(hidden_dim=256, n_agents=n_agents)
dpc_dummy_inputs = (jnp.zeros(N_grid), jnp.zeros(N_grid), xi_init)
dpc_path = bench_models_dir / 'dpc_heat_params.msgpack'

dpc_p = None
if os.path.exists(dpc_path):
    with open(dpc_path, 'rb') as f:
        dpc_bytes = f.read()
    
    # Safely restore and map keys to handle dict wrappers
    raw_dict = msgpack_restore(dpc_bytes)
    state_dict = raw_dict['params'] if 'params' in raw_dict else raw_dict
    state_dict = state_dict['dcp'] if 'dcp' in state_dict else state_dict
        
    variables = dpc_model.init(jax.random.PRNGKey(0), *dpc_dummy_inputs)
    
    try:
        dpc_p = from_state_dict(variables, state_dict)
        print("[+] Successfully loaded DPC")
    except Exception as e:
        print(f"[-] Failed to load DPC: {e}")
else:
    print(f"[-] {dpc_path} not found.")

if dpc_p:
    def dpc_apply(p, z, target, xi):
        return dpc_model.apply(p, z, target, xi)
    
    bench_registry['DPC'] = {'apply': dpc_apply, 'params': dpc_p, 'color': 'brown'}


# 7. Uncontrolled Baseline
bench_registry['Uncontrolled'] = {
    'apply': lambda p, z, t, xi: (jnp.zeros(n_agents), jnp.zeros(n_agents)), 
    'params': None, 'color': 'red'
}

# --- 3. Simulation ---
print(f"Running Simulations for {list(bench_registry.keys())}...")
key = jax.random.PRNGKey(42)
keys_init = jax.random.split(key, N_eval)
keys_target = jax.random.split(jax.random.PRNGKey(100), N_eval)

_, z_init_batch = jax.vmap(partial(generate_grf, n_points=N_grid, length_scale=0.2))(keys_init)
_, z_target_batch = jax.vmap(partial(generate_grf, n_points=N_grid, length_scale=0.4))(keys_target)
xi_batch = jnp.tile(xi_init, (N_eval, 1))

@jax.jit(static_argnames=['name'])
def run_sim(name, z_i, target_i, xi_i):
    dyn = PDEDynamics(policy_apply_fn=bench_registry[name]['apply'])
    z_traj, xi_traj, u_traj, v_traj = dyn.unroll_controlled(
        z_init=z_i, xi_init=xi_i, z_target=target_i, 
        params=bench_registry[name]['params'], t_steps=T_steps
    )
    return z_traj, xi_traj

for name in bench_registry:
    print(f"Running {name} unrolls...")
    z_res, xi_res = jax.vmap(lambda z, t, x: run_sim(name, z, t, x))(z_init_batch, z_target_batch, xi_batch)
    bench_registry[name]['z_data'] = z_res
    bench_registry[name]['xi_data'] = xi_res

# --- 4. Metrics & Results Printing ---
print("\n" + "="*70)
print(f"{'Method':<15} | {'Mean Track Error':<20} | {'2-Sigma':<20}")
print("-" * 70)

for name in bench_registry:
    final_err = jnp.mean((bench_registry[name]['z_data'][:, -1] - z_target_batch)**2, axis=1)
    mean_val, std_val = jnp.mean(final_err), jnp.std(final_err)
    print(f"{name:<15} | {mean_val:.6f}             | ±{2*std_val:.6f}")
print("="*70)

# --- 5. Individual Field Plots (PDF Export) ---
print("Saving individual field plots to PDF...")
for name in bench_registry:
    plt.figure(figsize=(8, 5))
    field_data = bench_registry[name]['z_data'][0]
    plt.imshow(field_data.T, aspect='auto', origin='lower', 
               extent=[0, T_steps, 0, L_domain], 
               cmap='hot', vmin=float(jnp.min(z_target_batch)), vmax=float(jnp.max(z_target_batch)))
    
    plt.colorbar(label='Temperature')
    plt.title(f'Heat Tracking Field: {name}')
    plt.xlabel('Time Step')
    plt.ylabel('Space (x)')
    plt.tight_layout()
    
    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").lower()
    plt.savefig(output_dir / f"field_{safe_name}.pdf")
    plt.close()

# --- 6. Plotting ---
plt.figure(figsize=(18, 8))

# 1. Boxplot of Tracking Error
plt.subplot(1, 2, 1)
data_boxplot = [jnp.mean((bench_registry[n]['z_data'][:, -1] - z_target_batch)**2, axis=1) for n in bench_registry]
plt.boxplot(data_boxplot, labels=list(bench_registry.keys()), tick_labels=list(bench_registry.keys()))
plt.yscale('log')
plt.title('Final Tracking Error (MSE)')
plt.ylabel('Mean Squared Error')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# 2. Error Evolution
plt.subplot(1, 2, 2)
time_axis = jnp.arange(T_steps)
for name in bench_registry:
    evol = jnp.mean(jnp.mean((bench_registry[name]['z_data'] - z_target_batch[:, None, :])**2, axis=2), axis=0)
    plt.plot(time_axis, evol, label=name, color=bench_registry[name]['color'], lw=2.5)

plt.yscale('log')
plt.title('Tracking Error Evolution')
plt.xlabel('Time Step')
plt.ylabel('MSE (Log)')
plt.legend()
plt.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "heat_tracking_results.png")
print(f"\nSummary results saved to {output_dir}/heat_tracking_results.png")