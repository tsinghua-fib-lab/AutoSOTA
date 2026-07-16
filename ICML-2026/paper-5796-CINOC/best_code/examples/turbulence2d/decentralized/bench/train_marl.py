import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.80'

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
import flax.serialization
from flax import struct
import numpy as np
import time
import pickle
from pathlib import Path
import sys
from functools import partial
from tqdm import trange

# Enable x64 globally (Crucial for Spectral Stability)
jax.config.update("jax_enable_x64", True)

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(script_dir))

# Project imports
from env_turb2d import Turbulence2DMARLEnv
from utils_hypemarl import get_sinusoidal_encoding
from examples.turbulence2d.decentralized.data_utils import get_batch_initial_conditions
import tesseracts.turbulence2d.solver as solver

from examples.turbulence2d.decentralized.bench.models_marl import MARLActor2DKS, MARLCritic2DKS

# ==========================================
# 1. CONFIGURATIONS
# ==========================================
N_AGENTS = 64          
L_DOMAIN = 1.0         
N_GRID = 64
U_MAX = 75.0           
PATCH_SIZE = 20        

ENV_BATCH_SIZE = 512 
EVAL_INT = 50          
POLICY_DELAY = 2 

MAX_ENV_STEPS = 150    
SUBSTEPS = 5           
DT = 0.01              
VISCOSITY = 5e-4       
SIGMA = 0.05           

NUM_PARALLEL_ENVS = 64 
TOTAL_UPDATES = 50000  
WARMUP_UPDATES = 500

key = jax.random.PRNGKey(42)

# ==========================================
# 2. GLOBAL PRECOMPUTATION
# ==========================================
kx, ky, k_sq, k_inv = solver.get_spectral_grid(N_GRID, L_DOMAIN)
dt_phys = DT / SUBSTEPS

grid_dim = int(np.sqrt(N_AGENTS))
x_c = jnp.linspace(0, L_DOMAIN, grid_dim, endpoint=False) + L_DOMAIN/(2*grid_dim)
y_c = jnp.linspace(0, L_DOMAIN, grid_dim, endpoint=False) + L_DOMAIN/(2*grid_dim)
xv, yv = jnp.meshgrid(x_c, y_c)
centers_flat = jnp.stack([xv.flatten(), yv.flatten()], axis=1)

forcing_hat = solver.compute_forcing_profile(
    centers_flat[:, 0], centers_flat[:, 1], N_GRID, L_DOMAIN, SIGMA
)

# ==========================================
# 3. UTILS & ENVIRONMENT SETUP
# ==========================================
def get_2d_sinusoidal_encoding(p_2d, d=64, n=1000.0):
    pe_x = get_sinusoidal_encoding(p_2d[:, 0], d=d, n=n)
    pe_y = get_sinusoidal_encoding(p_2d[:, 1], d=d, n=n)
    return jnp.concatenate([pe_x, pe_y], axis=-1)

dummy_pool = jnp.zeros((1, N_GRID, N_GRID), dtype=jnp.float64)

env = Turbulence2DMARLEnv(
    initial_conditions=dummy_pool, n_agents=N_AGENTS, 
    N_grid=N_GRID, L=L_DOMAIN, dt=DT, viscosity=VISCOSITY, 
    substeps=SUBSTEPS, max_steps=MAX_ENV_STEPS, sigma=SIGMA
)

pe_dim = 128 
xi_norm = jnp.array(env.xi_norm)
pe_jax = jnp.array(get_2d_sinusoidal_encoding(xi_norm, d=64), dtype=jnp.float32)

actor = MARLActor2DKS(u_max=U_MAX)
critic = MARLCritic2DKS()

key, *subkeys = jax.random.split(key, 4)

# Network Initialization
dummy_patches = jnp.zeros((ENV_BATCH_SIZE, N_AGENTS, PATCH_SIZE, PATCH_SIZE, 3), dtype=jnp.float32)
dummy_pe = jnp.zeros((ENV_BATCH_SIZE, N_AGENTS, pe_dim), dtype=jnp.float32)
dummy_u = jnp.zeros((ENV_BATCH_SIZE, N_AGENTS, 1), dtype=jnp.float32)

actor_params = actor.init(subkeys[0], dummy_patches, dummy_pe)
critic_params = critic.init(subkeys[1], dummy_patches, dummy_pe, dummy_u)

target_actor_params = jax.tree_util.tree_map(jnp.copy, actor_params)
target_critic_params = jax.tree_util.tree_map(jnp.copy, critic_params)

tx_actor = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-5))
tx_critic = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-5))
opt_actor = tx_actor.init(actor_params)
opt_critic = tx_critic.init(critic_params)

# ==========================================
# 4. HYBRID REPLAY BUFFER (Smaller & Global)
# ==========================================
@struct.dataclass
class DeviceReplayBuffer:
    w: jnp.ndarray  # Global Vorticity Field
    a: jnp.ndarray
    r: jnp.ndarray
    nw: jnp.ndarray # Global Next Vorticity Field
    d: jnp.ndarray
    ptr: jnp.int32
    size: jnp.int32
    max_size: int = struct.field(pytree_node=False)

    @classmethod
    def create(cls, max_size, n_grid, a_dim):
        return cls(
            w=jnp.zeros((max_size, n_grid, n_grid), dtype=jnp.float32),
            a=jnp.zeros((max_size, N_AGENTS, a_dim), dtype=jnp.float32),
            r=jnp.zeros((max_size, N_AGENTS, 1), dtype=jnp.float32),
            nw=jnp.zeros((max_size, n_grid, n_grid), dtype=jnp.float32),
            d=jnp.zeros((max_size, N_AGENTS, 1), dtype=jnp.float32),
            ptr=jnp.int32(0),
            size=jnp.int32(0),
            max_size=max_size
        )

@jax.jit
def add_batch_to_buffer(buffer, w_b, a_b, r_b, nw_b, d_b):
    batch_size = w_b.shape[0]
    indices = (buffer.ptr + jnp.arange(batch_size)) % buffer.max_size
    
    new_w = buffer.w.at[indices].set(w_b.astype(jnp.float32))
    new_a = buffer.a.at[indices].set(a_b.astype(jnp.float32))
    new_r = buffer.r.at[indices].set(r_b.astype(jnp.float32))
    new_nw = buffer.nw.at[indices].set(nw_b.astype(jnp.float32))
    new_d = buffer.d.at[indices].set(d_b.astype(jnp.float32))
    
    new_ptr = (buffer.ptr + batch_size) % buffer.max_size
    new_size = jnp.minimum(buffer.size + batch_size, buffer.max_size)
    
    return buffer.replace(w=new_w, a=new_a, r=new_r, nw=new_nw, d=new_d, ptr=new_ptr, size=new_size)

@partial(jax.jit, static_argnames=['batch_size'])
def sample_buffer(buffer, batch_size, key):
    valid_range = jnp.minimum(buffer.size, buffer.max_size)
    indices = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=valid_range)
    return buffer.w[indices], buffer.a[indices], buffer.r[indices], buffer.nw[indices], buffer.d[indices]

buffer = DeviceReplayBuffer.create(100_000, N_GRID, 1)

# ==========================================
# 5. DPC PATCH EXTRACTION LOGIC
# ==========================================
@partial(jax.jit, static_argnames=['n_grid', 'p_size'])
def extract_local_patch(field, xi_n, n_grid, p_size):
    i = (xi_n[1] * n_grid).astype(jnp.int32) 
    j = (xi_n[0] * n_grid).astype(jnp.int32)
    half_patch = p_size // 2

    padded_field = jnp.pad(field, ((half_patch, half_patch), (half_patch, half_patch)), mode='wrap')
    
    patch = jax.lax.dynamic_slice(padded_field, (i, j), (p_size, p_size))
    return patch

@jax.jit
def build_marl_obs_batch(full_state_batch):
    def single_env_obs(w_curr):
        grads = jnp.gradient(w_curr)
        grad_y, grad_x = grads[0], grads[1]

        def get_local_obs(xi_single):
            p_w  = extract_local_patch(w_curr, xi_single, N_GRID, PATCH_SIZE)
            p_gx = extract_local_patch(grad_x, xi_single, N_GRID, PATCH_SIZE)
            p_gy = extract_local_patch(grad_y, xi_single, N_GRID, PATCH_SIZE)
            return jnp.stack([p_w, p_gx, p_gy], axis=-1)

        local_patches = jax.vmap(get_local_obs)(xi_norm)
        return (local_patches / 50.0).astype(jnp.float32)
        
    return jax.vmap(single_env_obs)(full_state_batch)

# ==========================================
# 6. TRAINING LOGIC
# ==========================================
@jax.jit
def update_critic(c_p, ta_p, tc_p, opt_c, x_patches, pe, u, r, nx_patches, d, key):
    key, noise_key = jax.random.split(key)
    
    # Decoupled from U_MAX: Fixed std dev of 5.0, strictly clamped to +/- 15.0
    noise = jax.random.normal(noise_key, u.shape, dtype=jnp.float32) * 2.0
    noise = jnp.clip(noise, -6.0, 6.0)
    
    next_u = jnp.clip(actor.apply(ta_p, nx_patches, pe) + noise, -U_MAX, U_MAX)
    
    t_q1, t_q2 = critic.apply(tc_p, nx_patches, pe, next_u)
    target_q = r + 0.99 * (1.0 - d) * jnp.minimum(t_q1, t_q2)
    
    def c_loss_fn(p):
        q1, q2 = critic.apply(p, x_patches, pe, u)
        return jnp.mean((q1 - target_q)**2 + (q2 - target_q)**2)
    
    l_c, grads_c = jax.value_and_grad(c_loss_fn)(c_p)
    up_c, opt_c = tx_critic.update(grads_c, opt_c)
    return optax.apply_updates(c_p, up_c), opt_c

@jax.jit
def update_actor_and_targets(a_p, c_p, ta_p, tc_p, opt_a, x_patches, pe):
    def a_loss_fn(p):
        return -jnp.mean(critic.apply(c_p, x_patches, pe, actor.apply(p, x_patches, pe))[0])
    
    l_a, grads_a = jax.value_and_grad(a_loss_fn)(a_p)
    up_a, opt_a = tx_actor.update(grads_a, opt_a)
    a_p = optax.apply_updates(a_p, up_a)
    
    tau = 0.005
    new_ta = jax.tree_util.tree_map(lambda new, old: tau*new + (1-tau)*old, a_p, ta_p)
    new_tc = jax.tree_util.tree_map(lambda new, old: tau*new + (1-tau)*old, c_p, tc_p)
    return a_p, new_ta, new_tc, opt_a

@partial(jax.jit, static_argnames=['add_noise'])
def get_batch_actions(a_p, obs_batch_patches, step_idx=0, key=None, add_noise=True):
    pe_expanded = jnp.tile(pe_jax[None, :, :], (obs_batch_patches.shape[0], 1, 1))
    
    actions = actor.apply(a_p, obs_batch_patches, pe_expanded)
    
    if add_noise:
        # Decoupled from U_MAX: Fixed std dev of 5.0, strictly clamped to +/- 15.0
        noise = jax.random.normal(key, actions.shape, dtype=jnp.float32) * 5.0
        noise = jnp.clip(noise, -15.0, 15.0)
        
        actions = jnp.clip(actions + noise, -U_MAX, U_MAX)
    return actions

@jax.jit
def parallel_marl_physics_step(w_batch, actions):
    acts_flat = actions.squeeze(-1).astype(jnp.float64)
    
    def single_physics_step(w_single, act_single):
        w_hat = jnp.fft.fft2(w_single)
        def rk4_loop(i, w):
            return solver.rk4_step(w, dt_phys, kx, ky, k_sq, k_inv, VISCOSITY, forcing_hat, act_single)
        w_hat_next = jax.lax.fori_loop(0, SUBSTEPS, rk4_loop, w_hat)
        return jnp.fft.ifft2(w_hat_next).real
    
    next_w_batch = jax.vmap(single_physics_step)(w_batch, acts_flat)
    
    is_invalid = jnp.logical_not(jnp.isfinite(next_w_batch).all(axis=(1, 2)))
    is_exploding = jnp.max(jnp.abs(next_w_batch), axis=(1, 2)) > 1000.0
    dones_batch = jnp.logical_or(is_invalid, is_exploding)
    
    safe_w = jnp.where(dones_batch[:, None, None], jnp.zeros_like(next_w_batch), next_w_batch)
        
    # Calculate global enstrophy (The true physics objective)
    global_enstrophy = jnp.mean(jnp.square(safe_w), axis=(1, 2))
    
    # Calculate global enstrophy
    global_enstrophy = jnp.mean(jnp.square(safe_w), axis=(1, 2))
    global_reward = -jnp.log(global_enstrophy + 1.0)
    
    # Local Credit Assignment
    grid_dim = int(np.sqrt(N_AGENTS)) # 8
    cell_size = N_GRID // grid_dim    # 8
    
    # Reshape field to isolate agent-specific grid cells (B, 8, 8, 8, 8)
    w_blocks = safe_w.reshape(-1, grid_dim, cell_size, grid_dim, cell_size)
    w_blocks = w_blocks.swapaxes(2, 3) 
    
    # Calculate Enstrophy strictly inside each agent's zone
    local_enstrophy = jnp.mean(jnp.square(w_blocks), axis=(3, 4)).reshape(-1, N_AGENTS)
    local_reward = -jnp.log(local_enstrophy + 1.0)
    
    action_penalty = -1e-3 * jnp.mean(jnp.square(actions / U_MAX), axis=-1)

    # Blend: 30% Team Goal, 70% Individual Accountability
    mixed_reward = (0.3 * global_reward[:, None]) + (0.7 * local_reward) + action_penalty
    
    rewards_batch = mixed_reward[..., None].astype(jnp.float32) 
    dones_batch = dones_batch.astype(jnp.float32)
    
    return safe_w, rewards_batch, dones_batch

@partial(jax.jit, static_argnames=['max_steps'])
def fast_eval_episode(actor_params, init_state, max_steps):
    def step_fn(state, _):
        obs_patches = build_marl_obs_batch(state[None, ...]) 
        act = get_batch_actions(actor_params, obs_patches, step_idx=0, key=None, add_noise=False)
        act_flat = act[0, :, 0].astype(jnp.float64)
        
        w_hat = jnp.fft.fft2(state)
        def rk4_loop(i, w):
            return solver.rk4_step(w, dt_phys, kx, ky, k_sq, k_inv, VISCOSITY, forcing_hat, act_flat)
        w_hat_next = jax.lax.fori_loop(0, SUBSTEPS, rk4_loop, w_hat)
        next_state = jnp.fft.ifft2(w_hat_next).real
        
        enstrophy = jnp.mean(next_state**2)
        crashed = jnp.isnan(next_state).any() | jnp.isinf(next_state).any() | (jnp.max(jnp.abs(next_state)) > 1000.0)
        
        return next_state, (enstrophy, crashed)

    _, (enstrophies, crashes) = jax.lax.scan(step_fn, init_state, None, length=max_steps)
    
    # Return both the mean across the episode and the absolute final value
    return jnp.mean(enstrophies), enstrophies[-1], jnp.any(crashes)

@jax.jit(donate_argnums=(0,))
def train_chunk(carry, step_indices, state_bank):
    def scan_step(carry, step_idx):
        buf, a_p, c_p, ta_p, tc_p, o_a, o_c, w, steps, rng = carry
        rng, act_k, res_k, samp_k, net_k = jax.random.split(rng, 5)
        
        # Build patches for the current step on the fly
        curr_patches = build_marl_obs_batch(w)
        
        def warmup_actions(_):
            # Scale down warmup chaos to 50% of U_MAX to prevent poisoning the initial replay buffer
            safe_u_max = U_MAX * 0.5 
            return jax.random.uniform(act_k, (NUM_PARALLEL_ENVS, N_AGENTS, 1), minval=-safe_u_max, maxval=safe_u_max, dtype=jnp.float32)
        
        def policy_actions(_):
            return get_batch_actions(a_p, curr_patches, step_idx, act_k, add_noise=True)
            
        actions = jax.lax.cond(step_idx < WARMUP_UPDATES, warmup_actions, policy_actions, None)
        
        next_w, rewards, dones = parallel_marl_physics_step(w, actions)
        steps += 1
        truncs = steps >= MAX_ENV_STEPS
        needs_reset = jnp.logical_or(dones.flatten(), truncs)
        
        safe_rewards = jnp.where(dones[:, None, None], -5000.0, rewards).astype(jnp.float32)
        dones_expanded = jnp.tile(dones[:, None, None], (1, N_AGENTS, 1)).astype(jnp.float32)
        
        # Store Global State `w` instead of patches
        new_buf = add_batch_to_buffer(buf, w, actions, safe_rewards, next_w, dones_expanded)
        
        fresh_states = jax.random.choice(res_k, state_bank, shape=(NUM_PARALLEL_ENVS,))
        w_next = jnp.where(needs_reset[:, None, None], fresh_states, next_w)
        steps_next = jnp.where(needs_reset, 0, steps)

        def do_network_updates(net_state):
            c_p, a_p, ta_p, tc_p, o_c, o_a = net_state
            
            # Sample global states
            bw, bu, br, bnw, bd = sample_buffer(new_buf, ENV_BATCH_SIZE, samp_k)
            
            # Extract patches dynamically right before the network sees them
            bx = build_marl_obs_batch(bw)
            bnx = build_marl_obs_batch(bnw)
            
            bx_patches = bx.reshape(-1, PATCH_SIZE, PATCH_SIZE, 3)
            bu_flat = bu.reshape(-1, 1)
            br_flat = br.reshape(-1, 1)
            bnx_patches = bnx.reshape(-1, PATCH_SIZE, PATCH_SIZE, 3)
            bd_flat = bd.reshape(-1, 1)
            
            pe_tiled = jnp.tile(pe_jax, (ENV_BATCH_SIZE, 1))
            
            new_c_p, new_o_c = update_critic(
                c_p, ta_p, tc_p, o_c, bx_patches, pe_tiled, bu_flat, br_flat, bnx_patches, bd_flat, net_k
            )
            
            def do_actor_update(_):
                return update_actor_and_targets(a_p, new_c_p, ta_p, tc_p, o_a, bx_patches, pe_tiled)
            def skip_actor_update(_):
                return a_p, ta_p, tc_p, o_a
                
            new_a_p, new_ta_p, new_tc_p, new_o_a = jax.lax.cond(
                step_idx % POLICY_DELAY == 0, do_actor_update, skip_actor_update, None
            )
            
            return new_c_p, new_a_p, new_ta_p, new_tc_p, new_o_c, new_o_a

        def skip_network_updates(net_state):
            return net_state

        net_state = (c_p, a_p, ta_p, tc_p, o_c, o_a)
        
        c_p, a_p, ta_p, tc_p, o_c, o_a = jax.lax.cond(
            new_buf.size >= ENV_BATCH_SIZE, do_network_updates, skip_network_updates, net_state
        )

        new_carry = (new_buf, a_p, c_p, ta_p, tc_p, o_a, o_c, w_next, steps_next, rng)
        return new_carry, None

    return jax.lax.scan(scan_step, carry, step_indices)

# ==========================================
# 7. EXECUTION LOOP
# ==========================================
print("Loading 2D Turbulence Initial Conditions...")
data_dir = Path('../../data')
data_dir.mkdir(parents=True, exist_ok=True)
file_path = data_dir / 'turbulence_chaotic_ics_64_more.pkl'

if file_path.exists():
    with open(file_path, 'rb') as f:
        state_bank = jnp.array(pickle.load(f))
    print(f"Loaded {len(state_bank)} ICs from {file_path}")
else:
    print("Generating ICs (this may take a few minutes)...")
    state_bank = get_batch_initial_conditions(key, 500, N_GRID, L_DOMAIN, viscosity=5e-4)
    with open(file_path, 'wb') as f:
        pickle.dump(np.array(state_bank), f)

if jnp.iscomplexobj(state_bank):
    print("Converting spectral initial conditions to physical space...")
    state_bank = jnp.fft.ifft2(state_bank).real.astype(jnp.float64)
else:
    state_bank = state_bank.astype(jnp.float64)

key, subkey = jax.random.split(key)
w_batch = jax.random.choice(subkey, state_bank, shape=(NUM_PARALLEL_ENVS,))
env_step_counts = jnp.zeros(NUM_PARALLEL_ENVS, dtype=jnp.int32)

carry = (
    buffer, actor_params, critic_params, target_actor_params, target_critic_params,
    opt_actor, opt_critic, w_batch, env_step_counts, key
)

print("Starting Massively Parallel MARL Training (Chunked & JITed 2D Turbulence)...")
start_time = time.time()

num_chunks = TOTAL_UPDATES // EVAL_INT

for chunk_idx in trange(num_chunks):
    start_step = chunk_idx * EVAL_INT
    step_indices = jnp.arange(start_step, start_step + EVAL_INT)
    
    carry, _ = train_chunk(carry, step_indices, state_bank)
    current_actor_params = carry[1] 
    
    eval_w = state_bank[0] 
    # Unpack the new final enstropy value
    eval_e_mean, eval_e_final, crashed = fast_eval_episode(current_actor_params, eval_w, MAX_ENV_STEPS)
    
    current_total_step = start_step + EVAL_INT
    episode_num = current_total_step // MAX_ENV_STEPS
    
    if crashed:
        print(f"\nUpdate {current_total_step:05d} | Episode {episode_num} | Eval Mean: [CRASHED] | Eval Final: [CRASHED] | Time: {time.time()-start_time:.1f}s")
    else:
        print(f"\nUpdate {current_total_step:05d} | Episode {episode_num} | Eval Mean: {eval_e_mean:.4f} | Eval Final: {eval_e_final:.4f} | Time: {time.time()-start_time:.1f}s")

final_actor_params = carry[1]
models_dir = Path('models')
models_dir.mkdir(exist_ok=True)
with open(models_dir / 'marl_turb_params.msgpack', 'wb') as f:
    f.write(flax.serialization.to_bytes({'actor': final_actor_params}))
print(f"Training finished in {time.time()-start_time:.1f}s. Weights saved.")