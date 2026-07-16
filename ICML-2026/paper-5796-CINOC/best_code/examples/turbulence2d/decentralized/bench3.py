import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import flax.linen as nn
import flax.serialization
from flax.serialization import msgpack_restore, from_state_dict
import sys
import os
import pickle
from pathlib import Path
from functools import partial

# Enable x64 for Spectral Solvers
jax.config.update("jax_enable_x64", True)

# --- Configuration & Paths ---
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

output_dir = Path("figures/bench_turb")
output_dir.mkdir(parents=True, exist_ok=True)

bench_models_dir = Path("bench/models")
bench_models_dir.mkdir(parents=True, exist_ok=True) 

# Turbulence Imports
from examples.turbulence2d.decentralized.dynamics_dual import PDEDynamics2D 
from models.policy_turb import DecentralizedTurbulenceNet
from examples.turbulence2d.decentralized.data_utils import get_batch_initial_conditions

# MARL & RL Model Imports
from examples.turbulence2d.decentralized.bench.env_turb2d import extract_patches_2d_jit
from examples.turbulence2d.decentralized.bench.utils_hypemarl import get_sinusoidal_encoding
from examples.turbulence2d.decentralized.bench.models_marl import MARLActor2DKS
from examples.turbulence2d.decentralized.bench.models_mappo import MAPPOActorTurb
from examples.turbulence2d.decentralized.bench.models_rl import FCNActor
from examples.turbulence2d.decentralized.bench.models_ppo import FCNActorPPO
from examples.turbulence2d.decentralized.bench.models_dpc import CentralizedFCNControlNet2D_Turb

# DeepONet Imports
from examples.turbulence2d.decentralized.bench.models_deeponet_matd3_ import DeepONetActor
from examples.turbulence2d.decentralized.bench.models_deeponet_mappo import DeepONetMAPPOActor

# 2D Specific Configuration
N_grid = 64
L_domain = 1.0
n_agents = 64 # 8x8 Actuator Grid
T_steps = 150 
substeps = 5
dt = 0.01
viscosity = 5e-4
N_eval = 20 # Evaluation batch size
ENV_MU = jnp.array([L_domain, dt, viscosity])

# Scaling Constants
STATE_NORM_FACTOR = 50.0 
U_MAX_RL = 75.0          

def get_2d_sinusoidal_encoding(p_2d, d=64, n=1000.0):
    pe_x = get_sinusoidal_encoding(p_2d[:, 0], d=d, n=n)
    pe_y = get_sinusoidal_encoding(p_2d[:, 1], d=d, n=n)
    return jnp.concatenate([pe_x, pe_y], axis=-1)

bench_registry = {}

# --- 1. Loading Logic ---
def load_params(filename, model, dummy_input):
    if not os.path.exists(filename):
        print(f"[-] {filename} not found.")
        return None
    with open(filename, 'rb') as f: bytes_data = f.read()
    
    if isinstance(dummy_input, tuple):
        variables = model.init(jax.random.PRNGKey(0), *dummy_input)
    else:
        variables = model.init(jax.random.PRNGKey(0), dummy_input)
        
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
grid_dim = int(np.sqrt(n_agents))
x_lin = np.linspace(0, L_domain, grid_dim, endpoint=False) + (L_domain/grid_dim)/2
xv, yv = np.meshgrid(x_lin, x_lin)
xi_init = jnp.stack([xv.flatten(), yv.flatten()], axis=-1).astype(np.float32)
target_state = jnp.zeros((N_grid, N_grid))

# 1. CINOC
CINOC_model = DecentralizedTurbulenceNet(features=(32, 64), patch_size=16, domain_size=(L_domain, L_domain), u_max=40.0)
CINOC_p = load_params('turbulence_params.msgpack', CINOC_model, (xi_init, jnp.zeros((1, N_grid, N_grid))))
if CINOC_p:
    def CINOC_apply_wrapped(p, xi_fixed, obs):
        return CINOC_model.apply(p, xi_fixed, obs)
    bench_registry['CINOC'] = {'apply': CINOC_apply_wrapped, 'params': CINOC_p, 'color': 'blue'}

# --- MARL Patch Extraction Helpers ---
@partial(jax.jit, static_argnames=['n_grid', 'p_size'])
def extract_local_patch(field, xi_n, n_grid, p_size):
    i = (xi_n[1] * n_grid).astype(jnp.int32) 
    j = (xi_n[0] * n_grid).astype(jnp.int32)
    half_patch = p_size // 2

    padded_field = jnp.pad(field, ((half_patch, half_patch), (half_patch, half_patch)), mode='wrap')
    patch = jax.lax.dynamic_slice(padded_field, (i, j), (p_size, p_size))
    return patch

def get_marl_obs(w_curr, xi_norm):
    grads = jnp.gradient(w_curr)
    grad_y, grad_x = grads[0], grads[1]

    def get_local_obs(xi_single):
        p_w  = extract_local_patch(w_curr, xi_single, N_grid, 20)
        p_gx = extract_local_patch(grad_x, xi_single, N_grid, 20)
        p_gy = extract_local_patch(grad_y, xi_single, N_grid, 20)
        return jnp.stack([p_w, p_gx, p_gy], axis=-1)

    local_patches = jax.vmap(get_local_obs)(xi_norm)
    return (local_patches / 50.0).astype(jnp.float32)

# 2. MARL 
marl_model = MARLActor2DKS(u_max=75.0) 
marl_dummy_patches = jnp.zeros((n_agents, 20, 20, 3), dtype=jnp.float32)
marl_dummy_pe = jnp.zeros((n_agents, 128), dtype=jnp.float32)
marl_dummy_input = (marl_dummy_patches, marl_dummy_pe)

marl_p = load_params(bench_models_dir / 'marl_turb_params.msgpack', marl_model, marl_dummy_input)

if marl_p:
    def marl_apply(p, xi_fixed, obs):
        w_phys = obs.squeeze()
        xi_norm = xi_fixed / L_domain
        patches = get_marl_obs(w_phys, xi_norm)
        pe = get_2d_sinusoidal_encoding(xi_norm, d=64).astype(jnp.float32) 
        action = marl_model.apply(p, patches, pe)
        return action.squeeze(-1)
    
    bench_registry['MARL'] = {'apply': marl_apply, 'params': marl_p, 'color': 'orange'}


# --- 3. TD3 ---
rl_model = FCNActor()
rl_dummy_input = jnp.zeros((1, N_grid, N_grid), dtype=jnp.float32) 

rl_p = load_params(bench_models_dir / 'rl_turb_params.msgpack', rl_model, rl_dummy_input)
if not rl_p:
    rl_p = load_params(Path('models/rl_turb_params.msgpack'), rl_model, rl_dummy_input)

if rl_p:
    def rl_apply(p, xi_fixed, obs):
        obs_squeeze = obs.squeeze()
        obs_norm = jnp.clip(obs_squeeze / STATE_NORM_FACTOR, -5.0, 5.0).astype(jnp.float32)
        
        act_norm = rl_model.apply(p, obs_norm[None, ...])
        action = (act_norm * U_MAX_RL).squeeze().astype(jnp.float64) 
        
        return action
    
    bench_registry['RL (TD3)'] = {'apply': rl_apply, 'params': rl_p, 'color': 'green'}


# --- 4. PPO ---
ppo_model = FCNActorPPO(n_agents=n_agents, u_max=U_MAX_RL)
ppo_dummy_input = jnp.zeros((1, N_grid, N_grid), dtype=jnp.float32)

ppo_p = load_params(bench_models_dir / 'ppo_turb_params.msgpack', ppo_model, ppo_dummy_input)
if not ppo_p:
    ppo_p = load_params(Path('models/ppo_turb_params.msgpack'), ppo_model, ppo_dummy_input)

if ppo_p:
    def ppo_apply(p, xi_fixed, obs):
        obs_squeeze = obs.squeeze()
        obs_norm = jnp.clip(obs_squeeze / STATE_NORM_FACTOR, -5.0, 5.0).astype(jnp.float32)
        
        mean, _ = ppo_model.apply(p, obs_norm[None, ...])
        action = mean.squeeze(0).astype(jnp.float64) 
        
        return action
    
    bench_registry['PPO'] = {'apply': ppo_apply, 'params': ppo_p, 'color': 'magenta'}


# --- 5. MAPPO (Turbulence-Specific CTDE) ---
mappo_model = MAPPOActorTurb(n_agents=n_agents, u_max=75.0)
mappo_dummy_patches = jnp.zeros((1, n_agents, 20, 20, 3), dtype=jnp.float32)
mappo_dummy_pe = jnp.zeros((1, n_agents, 128), dtype=jnp.float32)
mappo_dummy_input = (mappo_dummy_patches, mappo_dummy_pe)

mappo_p = load_params('bench/models/mappo_turb_params.msgpack', mappo_model, mappo_dummy_input)
if not mappo_p:
    mappo_p = load_params('models/mappo_turb_params.msgpack', mappo_model, mappo_dummy_input)

if mappo_p:
    @jax.jit
    def mappo_apply(p, xi_fixed, obs):
        w_phys = obs.squeeze()
        xi_norm = xi_fixed / L_domain
        patches = get_marl_obs(w_phys, xi_norm)
        pe = get_2d_sinusoidal_encoding(xi_norm, d=64).astype(jnp.float32)
        
        mean_raw, _ = mappo_model.apply(p, patches[None, ...], pe[None, ...])
        env_action = jnp.tanh(mean_raw) * 75.0
        return env_action.squeeze().astype(jnp.float64)

    bench_registry['MAPPO'] = {'apply': mappo_apply, 'params': mappo_p, 'color': 'cyan'}


# --- 6. Centralized DPC (FCN) ---
dpc_model = CentralizedFCNControlNet2D_Turb(u_max=75.0)
dpc_dummy_xi = jnp.zeros((n_agents, 2))
dpc_dummy_obs = jnp.zeros((N_grid, N_grid))
dpc_path = Path('bench/models/dpc_turb_params.msgpack')

dpc_p = None
if dpc_path.exists():
    with open(dpc_path, 'rb') as f:
        dpc_bytes = f.read()
    variables = dpc_model.init(jax.random.PRNGKey(0), dpc_dummy_xi, dpc_dummy_obs)
    raw_dict = msgpack_restore(dpc_bytes)
    
    def find_model_root(d):
        if isinstance(d, dict):
            if 'Conv_3' in d and 'LayerNorm_2' in d:
                return d
            for k, v in d.items():
                res = find_model_root(v)
                if res is not None:
                    return res
        return None
    
    model_root = find_model_root(raw_dict)
    if model_root is not None:
        try:
            dpc_p = from_state_dict(variables, {'params': model_root})
        except Exception as e:
            pass

if dpc_p:
    def dpc_apply(p, xi_fixed, obs):
        action = dpc_model.apply(p, xi_fixed, obs)
        return action
    
    bench_registry['DPC'] = {'apply': dpc_apply, 'params': dpc_p, 'color': 'brown'}

# --- 7. DeepONet MATD3 ---
d_matd3_model = DeepONetActor(u_max=40.0)
d_matd3_dummy_patches = jnp.zeros((n_agents, 20, 20, 3), dtype=jnp.float32)
d_matd3_dummy_xi = jnp.zeros((n_agents, 2), dtype=jnp.float32)

d_matd3_p = load_params(bench_models_dir / 'deeponet_matd3_params.msgpack', d_matd3_model, (d_matd3_dummy_patches, d_matd3_dummy_xi))
if not d_matd3_p:
    d_matd3_p = load_params(Path('models/deeponet_matd3_params.msgpack'), d_matd3_model, (d_matd3_dummy_patches, d_matd3_dummy_xi))

if d_matd3_p:
    @jax.jit
    def d_matd3_apply(p, xi_fixed, obs):
        w_phys = obs.squeeze()
        xi_norm = (xi_fixed / L_domain).astype(jnp.float32)
        patches = get_marl_obs(w_phys, xi_norm)
        action = d_matd3_model.apply(p, patches, xi_norm)
        return action.squeeze(-1).astype(jnp.float64)
    
    bench_registry['D-MATD3'] = {'apply': d_matd3_apply, 'params': d_matd3_p, 'color': 'purple'}

# --- 8. DeepONet MAPPO ---
d_mappo_model = DeepONetMAPPOActor(n_agents=n_agents, u_max=75.0)
d_mappo_dummy_patches = jnp.zeros((1, n_agents, 20, 20, 3), dtype=jnp.float32)
d_mappo_dummy_xi = jnp.zeros((1, n_agents, 2), dtype=jnp.float32)

d_mappo_p = load_params(bench_models_dir / 'deeponet_mappo_turb_params.msgpack', d_mappo_model, (d_mappo_dummy_patches, d_mappo_dummy_xi))
if not d_mappo_p:
    d_mappo_p = load_params(Path('models/deeponet_mappo_turb_params.msgpack'), d_mappo_model, (d_mappo_dummy_patches, d_mappo_dummy_xi))

if d_mappo_p:
    @jax.jit
    def d_mappo_apply(p, xi_fixed, obs):
        w_phys = obs.squeeze()
        xi_norm = (xi_fixed / L_domain).astype(jnp.float32)
        patches = get_marl_obs(w_phys, xi_norm)
        
        # MAPPO requires an explicit batch dimension for both inputs
        mean_raw, _ = d_mappo_model.apply(p, patches[None, ...], xi_norm[None, ...])
        
        env_action = jnp.tanh(mean_raw) * 75.0
        return env_action.squeeze().astype(jnp.float64)

    bench_registry['D-MAPPO'] = {'apply': d_mappo_apply, 'params': d_mappo_p, 'color': 'teal'}

# --- 9. Uncontrolled Baseline ---
bench_registry['Uncontrolled'] = {
    'apply': lambda p, xi_fixed, obs: jnp.zeros(n_agents), 
    'params': None, 'color': 'red'
}


# --- Simulation ---
print(f"Loading Turbulence Spectral Data and Running Simulations...")

data_dir = Path('../../data')
file_path = data_dir / 'turbulence_chaotic_ics_64_more.pkl'
if file_path.exists():
    with open(file_path, 'rb') as f:
        w_hat_pool = jnp.array(pickle.load(f)[:N_eval])
else:
    print("Generating ICs on the fly...")
    key = jax.random.PRNGKey(1234)
    w_hat_pool = get_batch_initial_conditions(key, N_eval, N_grid, L_domain, viscosity=5e-4)

xi_batch = jnp.tile(xi_init, (N_eval, 1, 1))

def run_sim(name, w_hat_init, xi_i):
    apply_fn = bench_registry[name]['apply']
    params = bench_registry[name]['params']
    dyn = PDEDynamics2D(policy_apply_fn=apply_fn)
    
    w_phys_traj, u_ctrl_traj = dyn.unroll_controlled(
        w_hat_init=w_hat_init, 
        xi_fixed=xi_i, 
        params=params, 
        t_steps=T_steps,
        substeps=substeps,
        N_grid=N_grid,
        L=L_domain,
        dt=dt,
        viscosity=viscosity,
        actuator_grid_shape=(8, 8) 
    )
    return w_phys_traj 

for name in bench_registry:
    print(f"Running {name} unrolls...")
    
    @jax.jit
    def batched_sim(w_batch, x_batch):
        return jax.vmap(lambda w, x: run_sim(name, w, x))(w_batch, x_batch)
        
    w_phys_res = batched_sim(w_hat_pool, xi_batch)
    bench_registry[name]['data'] = w_phys_res

# --- Metrics & Results Printing ---
print("\n" + "="*70)
print(f"{'Method':<15} | {'Final Enstrophy':<20} | {'2-Sigma':<20}")
print("-" * 70)

for name in bench_registry:
    final_err = jnp.mean(bench_registry[name]['data'][:, -1]**2, axis=(1, 2))
    mean_val, std_val = jnp.mean(final_err), jnp.std(final_err)
    print(f"{name:<15} | {mean_val:.6f}             | ±{2*std_val:.6f}")
print("="*70)

# --- Individual Field Plots (PDF Export) ---
print("Saving individual state plots to PDF...")
for name in bench_registry:
    fig = plt.figure(figsize=(10, 5))
    
    initial_state = jnp.fft.ifft2(w_hat_pool[0]).real
    final_state = bench_registry[name]['data'][0, -1]
    
    vmin, vmax = float(jnp.min(initial_state)), float(jnp.max(initial_state))
    
    ax1 = plt.subplot(1, 2, 1)
    im1 = ax1.imshow(initial_state, aspect='auto', origin='lower', extent=[0, L_domain, 0, L_domain], cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.title('Initial Vorticity')
    plt.colorbar(im1, label='ω(x,y)')
    
    ax2 = plt.subplot(1, 2, 2)
    im2 = ax2.imshow(final_state, aspect='auto', origin='lower', extent=[0, L_domain, 0, L_domain], cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.title(f'Final Controlled State: {name}')
    plt.colorbar(im2, label='ω(x,y)')
    
    plt.tight_layout()
    plt.savefig(output_dir / f"state_{name.replace(' ', '_').lower()}.pdf")
    plt.close()

# --- Plotting Trendlines ---
plt.figure(figsize=(18, 8))

plt.subplot(1, 2, 1)
data_boxplot = [jnp.mean((bench_registry[n]['data'][:, -1])**2, axis=(1, 2)) for n in bench_registry]
plt.boxplot(data_boxplot, labels=list(bench_registry.keys()))
plt.yscale('log')
plt.title('Final System Enstrophy (Log Scale)')
plt.ylabel('Mean L2 Vorticity')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45) # Added rotation for cleaner labels

plt.subplot(1, 2, 2)
time_axis = jnp.arange(T_steps) * substeps * dt
for name in bench_registry:
    evol = jnp.mean(jnp.mean(bench_registry[name]['data']**2, axis=(2, 3)), axis=0)
    plt.plot(time_axis, evol, label=name, color=bench_registry[name]['color'], lw=2.5)

plt.yscale('log')
plt.title('Stabilization Enstrophy Evolution')
plt.xlabel('Time (s)')
plt.ylabel('Mean Enstrophy (Log)')
plt.legend()
plt.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "turbulence_stabilization_results.png")
print(f"\nSummary results saved to {output_dir}/turbulence_stabilization_results.png")