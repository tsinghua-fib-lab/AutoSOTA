import os
os.environ["JAX_ENABLE_X64"] = "False"
import jax
jax.config.update("jax_enable_x64", False)
jax.config.update("jax_disable_jit", False) 

import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import matplotlib.pyplot as plt
import flax.serialization
from flax.serialization import msgpack_restore, from_state_dict
from flax import struct
import sys
from pathlib import Path
from functools import partial
from typing import Callable, Tuple

# --- 1. Model Definitions & Paths ---
script_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(script_dir))

output_dir = Path("figures/images/bench_ns2d")
output_dir.mkdir(parents=True, exist_ok=True)

# Imports specific to NS2D
from dynamics import ns2d_step_jax
from models.policy_ns2d import NS2DControlNet

# Baseline Imports
from bench.models_rl import CentralizedActor
from bench.models_marl import MARLActor
from bench.models_ppo import CentralizedPPOActor
from bench.models_mappo import MAPPOActorNS2D
from bench.models_dpc import CentralizedMLPControlNetNS2D

# --- 2. NS2D Config & Physics Constants ---
data_dir = Path(__file__).parent.parent / 'data'
config = np.load(data_dir / 'config.npz')
Nx = int(config['Nx'])
Ny = int(config['Ny'])
dt = float(config['dt'])

N_AGENTS = 9
MAX_ENV_STEPS = 150
N_EVAL = 50  # Number of episodes to evaluate for the benchmark

BUOYANCY = 0.0
SIGMA_PUSH = 0.2
PUSH_MAX = 0.8
R_SAFE = 0.15

bench_registry = {}

# Load Data Banks
train_data = np.load(data_dir / 'train_data.npz')
rho_init_bank = jnp.array(train_data['rho_init'], dtype=jnp.float32)
rho_target_bank = jnp.array(train_data['rho_target'], dtype=jnp.float32)

# Set up evaluation batch
key = jax.random.PRNGKey(1234)
key, subkey = jax.random.split(key)
idx = jax.random.randint(subkey, (N_EVAL,), 0, len(rho_init_bank))
rho_init_batch = rho_init_bank[idx]
rho_target_batch = rho_target_bank[idx]

# Agent starting grid (3x3)
n_side = int(np.sqrt(N_AGENTS))
X, Y = jnp.meshgrid(
    jnp.linspace(0.15, 0.85, n_side),
    jnp.linspace(0.15, 1.0, n_side)
)
xi_single = jnp.stack([X.flatten(), Y.flatten()], axis=-1).astype(np.float32)
xi_batch = jnp.tile(xi_single, (N_EVAL, 1, 1))

# --- 3. Observation Builders ---
@jax.jit
def build_marl_obs_single(rho, target, xi):
    """Constructs the decentralized MAPPO observation for a single environment step."""
    rho_flat = rho.flatten()
    tgt_flat = target.flatten()
    N = xi.shape[0]
    
    global_context = jnp.concatenate([rho_flat, tgt_flat], axis=-1)
    global_context_exp = jnp.tile(global_context[None, :], (N, 1))
    
    obs = jnp.concatenate([global_context_exp, xi], axis=-1)
    return obs

# --- 4. Dynamics Wrapper (JAX-Safe) ---
@struct.dataclass
class DensityDynamicsWrapper:
    policy_apply_fn: Callable = struct.field(pytree_node=False)

    @partial(jax.jit, static_argnames=['t_steps', 'Nx', 'Ny'])
    def unroll_controlled(
        self, rho_init: jnp.ndarray, xi_init: jnp.ndarray, rho_target: jnp.ndarray,
        params, t_steps: int, Nx: int = 64, Ny: int = 80, dt: float = 1.0,
        buoyancy: float = 0.0, sigma_push: float = 0.2, push_max: float = 0.8
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        
        def step_fn(carry, _):
            smoke, xi = carry
            push_vel = self.policy_apply_fn(params, smoke, rho_target, xi)
            
            push_norm = jnp.linalg.norm(push_vel, axis=-1, keepdims=True)
            push_vel = jnp.where(push_norm > push_max, push_vel * push_max / (push_norm + 1e-8), push_vel)
            
            smoke_new = ns2d_step_jax(smoke, xi, push_vel, dt=dt, buoyancy=buoyancy, sigma_push=sigma_push, Nx=Nx, Ny=Ny)
            
            xi_new = xi + dt * push_vel * 0.01
            xi_new = jnp.clip(xi_new, 0.1, jnp.array([0.9, 1.15], dtype=xi.dtype)) 
            
            smoke_new = smoke_new.astype(smoke.dtype)
            xi_new = xi_new.astype(xi.dtype)
            push_vel = push_vel.astype(jnp.float32)

            return (smoke_new, xi_new), (smoke_new, xi_new, push_vel)
        
        _, trajectory = jax.lax.scan(step_fn, (rho_init, xi_init), None, length=t_steps)
        return trajectory

# --- 5. Loading Logic ---
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
        return None

print("\nLoading Models...")

# 1. CINOC Model
CINOC_model = NS2DControlNet(features=(16, 32), v_max=PUSH_MAX)
CINOC_p = load_params('ns2d_params.msgpack', CINOC_model, (jnp.zeros((Nx, Ny)), jnp.zeros((Nx, Ny)), xi_single))
if CINOC_p:
    bench_registry['CINOC'] = {
        'apply': lambda p, smoke, target, xi: CINOC_model.apply(p, smoke, target, xi), 
        'params': CINOC_p, 'color': 'blue'
    }

# 2. RL (Centralized TD3) Model
rl_model = CentralizedActor(n_agents=N_AGENTS)
rl_dummy_in = (jnp.zeros((Nx, Ny)), jnp.zeros((Nx, Ny)), jnp.zeros((N_AGENTS, 2)))
rl_p = load_params('bench/models/rl_ns2d_centralized_params.msgpack', rl_model, rl_dummy_in)

if rl_p:
    bench_registry['RL (Centralized TD3)'] = {
        'apply': lambda p, smoke, target, xi: rl_model.apply(p, smoke, target, xi), 
        'params': rl_p, 'color': 'purple'
    }

# 3. MARL (MATD3) Model
marl_model = MARLActor(n_agents=N_AGENTS)
marl_p = load_params('bench/models/marl_matd3_ns2d_params_new.msgpack', marl_model, rl_dummy_in)

if marl_p:
    bench_registry['MARL (MATD3)'] = {
        'apply': lambda p, smoke, target, xi: marl_model.apply(p, smoke, target, xi), 
        'params': marl_p, 'color': 'green'
    }

# 4. PPO (Centralized) Model
ppo_model = CentralizedPPOActor(n_agents=N_AGENTS, push_max=PUSH_MAX)
ppo_p = load_params('bench/models/ppo_ns2d_centralized_params.msgpack', ppo_model, rl_dummy_in)

if ppo_p:
    def ppo_apply(p, smoke, target, xi):
        mean, _ = ppo_model.apply(p, smoke, target, xi)
        return mean
        
    bench_registry['PPO (Centralized)'] = {
        'apply': ppo_apply, 
        'params': ppo_p, 'color': 'orange'
    }

# 5. MAPPO (Decentralized) Model
mappo_model = MAPPOActorNS2D(n_agents=N_AGENTS, push_max=PUSH_MAX)
mappo_dummy_obs = build_marl_obs_single(jnp.zeros((Nx, Ny)), jnp.zeros((Nx, Ny)), jnp.zeros((N_AGENTS, 2)))
mappo_p = load_params('bench/models/mappo_ns2d_params.msgpack', mappo_model, (mappo_dummy_obs,))

if mappo_p:
    def mappo_apply(p, smoke, target, xi):
        obs = build_marl_obs_single(smoke, target, xi)
        mean, _ = mappo_model.apply(p, obs)
        return mean
        
    bench_registry['MAPPO (CTDE)'] = {
        'apply': mappo_apply, 
        'params': mappo_p, 'color': 'cyan'
    }


# 6. Centralized DPC (MLP) Model
dpc_model = CentralizedMLPControlNetNS2D(hidden_dim=256, n_agents=N_AGENTS)
dpc_dummy_in = (jnp.zeros((Nx, Ny)), jnp.zeros((Nx, Ny)), jnp.zeros((N_AGENTS, 2)))
dpc_path = 'bench/models/dpc_ns2d_params.msgpack'

dpc_p = None
if os.path.exists(dpc_path):
    with open(dpc_path, 'rb') as f:
        dpc_bytes = f.read()
    
    # Safely unpack the nested dictionary trap
    raw_dict = msgpack_restore(dpc_bytes)
    state_dict = raw_dict
    if 'params' in state_dict:
        state_dict = state_dict['params']
    if 'params' in state_dict:
        state_dict = state_dict['params']
        
    variables = dpc_model.init(jax.random.PRNGKey(0), *dpc_dummy_in)
    
    try:
        # Re-wrap properly for loading
        dpc_p = from_state_dict(variables, {'params': state_dict})
        print("[+] Successfully loaded DPC")
    except Exception as e:
        print(f"[-] Failed to load DPC: {e}")
else:
    print(f"[-] {dpc_path} not found.")

if dpc_p:
    bench_registry['DPC (Centralized MLP)'] = {
        'apply': lambda p, smoke, target, xi: dpc_model.apply(p, smoke, target, xi), 
        'params': dpc_p, 'color': 'magenta'
    }

# Uncontrolled Baseline
bench_registry['Uncontrolled'] = {
    'apply': lambda p, smoke, t, xi: jnp.zeros((N_AGENTS, 2)), 
    'params': None, 'color': 'red'
}

# --- 6. Simulation & Evaluation ---
print(f"\nGenerating Data & Running Simulations for {list(bench_registry.keys())}...")

@jax.jit(static_argnames=['name'])
def run_sim(name, rho_i, xi_i, rho_t):
    dyn = DensityDynamicsWrapper(policy_apply_fn=bench_registry[name]['apply'])
    traj = dyn.unroll_controlled(
        rho_init=rho_i, xi_init=xi_i, rho_target=rho_t, 
        params=bench_registry[name]['params'], 
        t_steps=MAX_ENV_STEPS, Nx=Nx, Ny=Ny, dt=dt, 
        buoyancy=BUOYANCY, sigma_push=SIGMA_PUSH, push_max=PUSH_MAX
    )
    # Return smoke_traj and push_vel_traj
    return traj[0], traj[2]

for name in bench_registry:
    print(f"Running {name} unrolls...")
    smoke_traj_data, ctrl_traj_data = jax.vmap(lambda r, x, t: run_sim(name, r, x, t))(rho_init_batch, xi_batch, rho_target_batch)
    bench_registry[name]['smoke_traj'] = smoke_traj_data
    bench_registry[name]['ctrl_traj'] = ctrl_traj_data

# --- 7. Metrics & Results Printing ---
print("\n" + "="*85)
print(f"{'Method':<20} | {'Tracking Error (l_track)':<25} | {'Control Effort (l_effort)':<25}")
print("-" * 85)

for name in bench_registry:
    s_traj = bench_registry[name]['smoke_traj']
    c_traj = bench_registry[name]['ctrl_traj']
    
    # NS2D Tracking Error: MSE against target density map
    l_track_batch = jnp.mean((s_traj[:, -1, :, :] - rho_target_batch) ** 2, axis=(1, 2))
    mean_track, std_track = jnp.mean(l_track_batch), jnp.std(l_track_batch)
    
    # NS2D Control Effort: Squared push velocity
    l_effort_batch = jnp.mean(c_traj ** 2, axis=(1, 2, 3))
    mean_effort, std_effort = jnp.mean(l_effort_batch), jnp.std(l_effort_batch)
    
    print(f"{name:<20} | {mean_track:.6f} ±{2*std_track:.6f} | {mean_effort:.6f} ±{2*std_effort:.6f}")

print("="*85)

# --- 8. Plotting & Export ---
print("\nSaving final state plots to PDF...")
for name in bench_registry:
    plt.figure(figsize=(6, 5))
    # Grab the final state of the first evaluation sample
    final_state = bench_registry[name]['smoke_traj'][0, -1]
    plt.imshow(final_state.T, origin='lower', cmap='magma', vmin=0, vmax=2.0)
    plt.colorbar(label='Density')
    plt.title(f'Final Density State: {name}')
    plt.tight_layout()
    plt.savefig(output_dir / f"final_state_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.pdf")
    plt.close()

# Main Summary Plot (Boxplots & Evolutions)
plt.figure(figsize=(18, 8))

# 1. Boxplot (Tracking Error)
plt.subplot(1, 2, 1)
# Note: This is using the Final MSE calculation as discussed!
data_boxplot = [jnp.mean((bench_registry[n]['smoke_traj'][:, -1, :, :] - rho_target_batch)**2, axis=(1, 2)) for n in bench_registry]
plt.boxplot(data_boxplot, labels=list(bench_registry.keys()))
plt.yscale('log')
plt.title('Final Tracking Error (Log Scale)')
plt.ylabel('Final MSE vs Target')
plt.grid(True, alpha=0.3)

# 2. Tracking Error Evolution Over Time
plt.subplot(1, 2, 2)
time_axis = jnp.arange(MAX_ENV_STEPS) * dt
for name in bench_registry:
    # Mean error across the batch, plotted over time steps
    evol = jnp.mean((bench_registry[name]['smoke_traj'] - rho_target_batch[:, None, :, :])**2, axis=(0, 2, 3))
    
    # Make the uncontrolled line dashed so it stands out as a baseline
    line_style = '--' if name == 'Uncontrolled' else '-'
    
    plt.plot(time_axis, evol, label=name, color=bench_registry[name]['color'], lw=2.5, linestyle=line_style)

plt.yscale('log')
plt.title('Stabilization Error Evolution')
plt.xlabel('Time Step')
plt.ylabel('Mean Tracking Error (Log)')
plt.legend()
plt.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "ns2d_bench_results.png")
print(f"\nSummary results saved to {output_dir}/ns2d_bench_results.png")