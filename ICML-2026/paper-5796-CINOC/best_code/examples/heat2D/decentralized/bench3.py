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
from functools import partial

# --- Configuration & Paths ---
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

output_dir = Path("figures/bench_heat2d")
output_dir.mkdir(parents=True, exist_ok=True)

bench_models_dir = Path("bench/models")
bench_models_dir.mkdir(parents=True, exist_ok=True) 

# Heat 2D Imports
from examples.heat2D.decentralized.dynamics_dual import PDEDynamics 
from models.policy import DecentralizedHeat2DControlNet
from examples.heat2D.decentralized.data_utils import get_training_data
from examples.heat2D.decentralized.bench.env_heat2d import extract_patches_heat2d_jit
from examples.heat2D.decentralized.bench.utils_hypemarl import get_sinusoidal_encoding
from examples.heat2D.decentralized.bench.models_marl import MARLActor2D, U_MAX, V_MAX
from examples.heat2D.decentralized.bench.models_ppo import PPOActor2D
from examples.heat2D.decentralized.bench.models_mappo import MAPPOActor2D
from examples.heat2D.decentralized.bench.models_rl import CentralizedActor2D

# Added DPC Import
from examples.heat2D.decentralized.bench.models_dpc import CentralizedMLPControlNet2D

# 2D Specific Configuration
N_grid = 32
L_domain = 1.0
n_agents = 16
T_steps = 100 # Match test evaluation steps
N_eval = 50
ENV_MU = jnp.array([0.01]) 

def get_2d_sinusoidal_encoding(p_2d, d=64, n=1000.0):
    pe_x = get_sinusoidal_encoding(p_2d[:, 0], d=d, n=n)
    pe_y = get_sinusoidal_encoding(p_2d[:, 1], d=d, n=n)
    return jnp.concatenate([pe_x, pe_y], axis=-1)

@jax.jit
def get_poly_features_jax(x):
    is_1d = x.ndim == 1
    x_2d = jnp.atleast_2d(x)
    n_feat = x_2d.shape[-1]
    _r, _c = np.triu_indices(n_feat)
    def poly_single(feat):
        bias = jnp.ones((1,))
        quad = jnp.outer(feat, feat)[_r, _c] 
        return jnp.concatenate([bias, feat, quad])
    res = jax.vmap(poly_single)(x_2d)
    return res[0] if is_1d else res

bench_registry = {}

# --- Loading Logic ---
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
# Initialize 4x4 grid of agents
n_side = int(np.sqrt(n_agents))
pos_1d = np.linspace(0.2, 0.8, n_side)
X, Y = np.meshgrid(pos_1d, pos_1d)
xi_init = jnp.stack([X.flatten(), Y.flatten()], axis=-1).astype(np.float32)

# 1. CINOC (Centralized / DeepONet)
CINOC_model = DecentralizedHeat2DControlNet(features=(16, 32))
CINOC_p = load_params('decentralized_params_heat2d.msgpack', CINOC_model, (jnp.zeros((N_grid, N_grid)), jnp.zeros((N_grid, N_grid)), xi_init))
if CINOC_p:
    bench_registry['CINOC'] = {'apply': CINOC_model.apply, 'params': CINOC_p, 'color': 'blue'}

# 2. MARL (Decentralized Multi-Agent DDPG/TD3)
marl_model = MARLActor2D()
marl_dummy_input = jnp.zeros((n_agents, 429))
marl_p = load_params(bench_models_dir / 'marl_heat2d_params.msgpack', marl_model, (marl_dummy_input,))

if marl_p:
    def marl_apply(p, z, target, xi):
        y = extract_patches_heat2d_jit(z, target, xi/L_domain, window_size=6, resized_dim=10)
        mu_broadcast = jnp.tile(ENV_MU, (n_agents, 1))
        pe = get_2d_sinusoidal_encoding(xi/L_domain, d=64) 
        
        obs = jnp.concatenate([y, mu_broadcast, pe], axis=-1)
        action = marl_model.apply(p, obs)
        return action[:, 0], action[:, 1:3]
    
    bench_registry['MARL'] = {'apply': marl_apply, 'params': marl_p, 'color': 'orange'}

# 3. RL Centralized (DDPG/TD3)
rl_model = CentralizedActor2D(n_agents=n_agents)
rl_dummy_z = jnp.zeros((N_grid, N_grid))
rl_dummy_xi = jnp.zeros((n_agents, 2))
rl_p = load_params(bench_models_dir / 'rl_heat2d_params.msgpack', rl_model, (rl_dummy_z, rl_dummy_z, rl_dummy_xi))

if rl_p:
    def rl_apply(p, z, target, xi):
        action = rl_model.apply(p, z, target, xi)
        return action[:, 0], action[:, 1:3]
    bench_registry['RL'] = {'apply': rl_apply, 'params': rl_p, 'color': 'purple'}

# 4. PPO (Centralized)
ppo_model = PPOActor2D(n_agents=n_agents)
ppo_dummy_z = jnp.zeros((1, N_grid, N_grid))
ppo_dummy_xi = jnp.zeros((1, n_agents, 2))
ppo_p = load_params(bench_models_dir / 'ppo_heat2d_params.msgpack', ppo_model, (ppo_dummy_z, ppo_dummy_z, ppo_dummy_xi))
if ppo_p:
    def ppo_apply(p, z, target, xi):
        mean, _ = ppo_model.apply(p, z[None, ...], target[None, ...], xi[None, ...])
        action = mean[0]
        return action[:, 0], action[:, 1:3]
    bench_registry['PPO'] = {'apply': ppo_apply, 'params': ppo_p, 'color': 'green'}

# 5. MAPPO (Decentralized Multi-Agent PPO)
mappo_model = MAPPOActor2D(n_agents=n_agents)
mappo_dummy_input = jnp.zeros((1, n_agents, 557))
mappo_p = load_params(bench_models_dir / 'mappo_heat2d_params.msgpack', mappo_model, (mappo_dummy_input,))

if mappo_p:
    def mappo_apply(p, z, target, xi):
        y = extract_patches_heat2d_jit(z, target, xi/L_domain, window_size=6, resized_dim=10)
        mu_broadcast = jnp.tile(ENV_MU, (n_agents, 1))
        pe = get_2d_sinusoidal_encoding(xi/L_domain, d=128) 
        
        obs = jnp.concatenate([y, mu_broadcast, pe], axis=-1)
        mean, _ = mappo_model.apply(p, obs[None, ...])
        action = mean[0]
        return action[:, 0], action[:, 1:3]
    
    bench_registry['MAPPO'] = {'apply': mappo_apply, 'params': mappo_p, 'color': 'cyan'}


# 6. Centralized DPC (MLP)
dpc_model = CentralizedMLPControlNet2D(hidden_dim=256, n_agents=n_agents)
dpc_dummy_inputs = (jnp.zeros((N_grid, N_grid)), jnp.zeros((N_grid, N_grid)), jnp.zeros((n_agents, 2)))
dpc_path = bench_models_dir / 'dpc_heat2d_params.msgpack'

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
        # DPC Model returns (u, v) natively
        return dpc_model.apply(p, z, target, xi)
    
    bench_registry['DPC'] = {'apply': dpc_apply, 'params': dpc_p, 'color': 'magenta'}


# 7. Uncontrolled Baseline
bench_registry['Uncontrolled'] = {
    'apply': lambda p, z, t, xi: (jnp.zeros(n_agents), jnp.zeros((n_agents, 2))), 
    'params': None, 'color': 'red'
}

# --- 3. Simulation ---
print(f"Loading 2D dataset and Running Simulations for {list(bench_registry.keys())}...")
z_init_all, z_target_all, _ = get_training_data(n_samples=N_eval, n_grid=N_grid, dataset_dir='../data')
z_init_batch = jnp.array(z_init_all[:N_eval])
z_target_batch = jnp.array(z_target_all[:N_eval])
xi_batch = jnp.tile(xi_init, (N_eval, 1, 1))

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
    final_err = jnp.mean((bench_registry[name]['z_data'][:, -1] - z_target_batch)**2, axis=(1, 2))
    mean_val, std_val = jnp.mean(final_err), jnp.std(final_err)
    print(f"{name:<15} | {mean_val:.6f}             | ±{2*std_val:.6f}")
print("="*70)

# --- 5. Individual Field Plots (PDF Export) ---
print("Saving individual field plots to PDF...")
for name in bench_registry:
    plt.figure(figsize=(15, 5))
    
    final_state = bench_registry[name]['z_data'][0, -1]
    target_state = z_target_batch[0]
    initial_state = z_init_batch[0]
    
    vmin = float(jnp.min(z_target_batch))
    vmax = float(jnp.max(z_target_batch))
    
    plt.subplot(1, 3, 1)
    plt.imshow(initial_state, aspect='auto', origin='lower', cmap='hot', vmin=vmin, vmax=vmax)
    plt.title('Initial State')
    plt.colorbar(label='Temperature')
    
    plt.subplot(1, 3, 2)
    plt.imshow(target_state, aspect='auto', origin='lower', cmap='hot', vmin=vmin, vmax=vmax)
    plt.title('Target State')
    plt.colorbar(label='Temperature')

    plt.subplot(1, 3, 3)
    plt.imshow(final_state, aspect='auto', origin='lower', cmap='hot', vmin=vmin, vmax=vmax)
    plt.title(f'Final Controlled State: {name}')
    plt.colorbar(label='Temperature')
    
    plt.tight_layout()
    plt.savefig(output_dir / f"field_{name.lower().replace(' ', '_')}.pdf")
    plt.close()

# --- 6. Plotting Trendlines ---
plt.figure(figsize=(18, 8))

plt.subplot(1, 2, 1)
data_boxplot = [jnp.mean((bench_registry[n]['z_data'][:, -1] - z_target_batch)**2, axis=(1, 2)) for n in bench_registry]
plt.boxplot(data_boxplot, labels=list(bench_registry.keys()))
plt.yscale('log')
plt.title('Final Tracking Error (MSE)')
plt.ylabel('Mean Squared Error')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
time_axis = jnp.arange(T_steps)
for name in bench_registry:
    evol = jnp.mean(jnp.mean((bench_registry[name]['z_data'] - z_target_batch[:, None, :, :])**2, axis=(2, 3)), axis=0)
    plt.plot(time_axis, evol, label=name, color=bench_registry[name]['color'], lw=2.5)

plt.yscale('log')
plt.title('Tracking Error Evolution')
plt.xlabel('Time Step')
plt.ylabel('MSE (Log)')
plt.legend()
plt.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "heat2d_tracking_results.png")
print(f"\nSummary results saved to {output_dir}/heat2d_tracking_results.png")