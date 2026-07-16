import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.80'

import jax
jax.config.update("jax_enable_x64", True)

# --- Keep compilation fast with JAX Caching ---
jax.config.update("jax_disable_jit", False)
cache_dir = os.path.join(os.path.dirname(__file__), ".jax_cache")
os.makedirs(cache_dir, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", cache_dir)
# ----------------------------------------------

import jax.numpy as jnp
import optax
import flax.serialization
from flax import struct
import numpy as np
import time
import pickle
from pathlib import Path
import sys
from functools import partial
from tqdm import trange

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(script_dir))

# --- UPDATED: Import the FCN Models ---
from models_rl import FCNActor, FCNCritic
from examples.turbulence2d.decentralized.data_utils import get_batch_initial_conditions
import tesseracts.turbulence2d.solver as solver

# --- Configurations ---
N_AGENTS = 64          # 8x8 grid
L_DOMAIN = 1.0         # Domain size
N_GRID = 64
U_MAX = 75.0

ENV_BATCH_SIZE = 128 
EVAL_INT = 50
POLICY_DELAY = 2 

MAX_ENV_STEPS = 150    
SUBSTEPS = 5           
DT = 0.01              
VISCOSITY = 5e-4       

NUM_PARALLEL_ENVS = 64
TOTAL_UPDATES = 100000 
WARMUP_UPDATES = 500

STATE_NORM_FACTOR = 50.0  

key = jax.random.PRNGKey(42)

# --- Precompute Spectral Grid & Forcing Profiles Globally (FP64) ---
kx, ky, k_sq, k_inv = solver.get_spectral_grid(N_GRID, L_DOMAIN)
dt_phys = DT / SUBSTEPS

grid_dim = int(np.sqrt(N_AGENTS))
x_c = jnp.linspace(0, L_DOMAIN, grid_dim, endpoint=False) + L_DOMAIN/(2*grid_dim)
y_c = jnp.linspace(0, L_DOMAIN, grid_dim, endpoint=False) + L_DOMAIN/(2*grid_dim)
xv, yv = jnp.meshgrid(x_c, y_c)
centers_flat = jnp.stack([xv.flatten(), yv.flatten()], axis=1)

forcing_hat = solver.compute_forcing_profile(
    centers_flat[:, 0], centers_flat[:, 1], N_GRID, L_DOMAIN, 0.05
)

# --- UPDATED: Initialize FCN Models ---
actor = FCNActor()
critic = FCNCritic()

key, *subkeys = jax.random.split(key, 4)

dummy_z = jnp.zeros((ENV_BATCH_SIZE, N_GRID, N_GRID), dtype=jnp.float32)
dummy_action = jnp.zeros((ENV_BATCH_SIZE, N_AGENTS), dtype=jnp.float32) 

actor_params = actor.init(subkeys[0], dummy_z)
critic_params = critic.init(subkeys[1], dummy_z, dummy_action)

target_actor_params = actor_params
target_critic_params = critic_params

# --- NEW: Cosine Decay Learning Rate Schedule ---
lr_schedule = optax.cosine_decay_schedule(
    init_value=1e-4, 
    decay_steps=TOTAL_UPDATES, 
    alpha=0.1  # Decays to 1e-4 * 0.1 = 1e-5 at the end of training
)

tx_actor = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=lr_schedule))
tx_critic = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=lr_schedule))
opt_actor = tx_actor.init(actor_params)
opt_critic = tx_critic.init(critic_params)

# --- 1. ON-DEVICE REPLAY BUFFER (Forced FP32) ---
@struct.dataclass
class DeviceReplayBuffer:
    z: jnp.ndarray
    a: jnp.ndarray
    r: jnp.ndarray
    nz: jnp.ndarray
    d: jnp.ndarray
    ptr: jnp.int32
    size: jnp.int32
    max_size: int = struct.field(pytree_node=False)

    @classmethod
    def create(cls, max_size, n_grid, n_agents):
        return cls(
            z=jnp.zeros((max_size, n_grid, n_grid), dtype=jnp.float32),
            a=jnp.zeros((max_size, n_agents), dtype=jnp.float32), 
            r=jnp.zeros((max_size, 1), dtype=jnp.float32),
            nz=jnp.zeros((max_size, n_grid, n_grid), dtype=jnp.float32),
            d=jnp.zeros((max_size, 1), dtype=jnp.float32),
            ptr=jnp.int32(0),
            size=jnp.int32(0),
            max_size=max_size
        )

@jax.jit
def add_batch_to_buffer(buffer, z_b, a_b, r_b, nz_b, d_b):
    batch_size = z_b.shape[0]
    indices = (buffer.ptr + jnp.arange(batch_size)) % buffer.max_size
    
    new_z = buffer.z.at[indices].set(z_b.astype(jnp.float32))
    new_a = buffer.a.at[indices].set(a_b.astype(jnp.float32))
    new_r = buffer.r.at[indices].set(r_b.astype(jnp.float32))
    new_nz = buffer.nz.at[indices].set(nz_b.astype(jnp.float32))
    new_d = buffer.d.at[indices].set(d_b.astype(jnp.float32))
    
    new_ptr = (buffer.ptr + batch_size) % buffer.max_size
    new_size = jnp.minimum(buffer.size + batch_size, buffer.max_size)
    
    return buffer.replace(z=new_z, a=new_a, r=new_r, nz=new_nz, d=new_d, ptr=new_ptr, size=new_size)

@partial(jax.jit, static_argnames=['batch_size'])
def sample_buffer(buffer, batch_size, key):
    valid_range = jnp.minimum(buffer.size, buffer.max_size)
    indices = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=valid_range)
    return buffer.z[indices], buffer.a[indices], buffer.r[indices], buffer.nz[indices], buffer.d[indices]

buffer = DeviceReplayBuffer.create(100_000, N_GRID, N_AGENTS)

# --- 2. JIT TRAINING & ROLLOUT FUNCTIONS ---
@jax.jit
def update_critic(c_p, ta_p, tc_p, opt_c, z, a, r, nz, d, key): 
    key, noise_key = jax.random.split(key)
    
    # FIX: Hard-clip states to protect CNN from explosive gradients
    z_norm = jnp.clip(z / STATE_NORM_FACTOR, -5.0, 5.0).astype(jnp.float32)
    nz_norm = jnp.clip(nz / STATE_NORM_FACTOR, -5.0, 5.0).astype(jnp.float32)
    
    a_norm = a / U_MAX
    next_a_norm = actor.apply(ta_p, nz_norm)
    
    noise_std = 2.0 / U_MAX
    noise_clip = 6.0 / U_MAX
    noise = jax.random.normal(noise_key, next_a_norm.shape, dtype=jnp.float32) * noise_std
    noise = jnp.clip(noise, -noise_clip, noise_clip)
    next_a_norm = jnp.clip(next_a_norm + noise, -1.0, 1.0)
    
    t_q1, t_q2 = critic.apply(tc_p, nz_norm, next_a_norm)
    target_q = r + 0.99 * (1.0 - d) * jnp.minimum(t_q1, t_q2)
    
    def c_loss_fn(p):
        q1, q2 = critic.apply(p, z_norm, a_norm)
        return jnp.mean((q1 - target_q)**2 + (q2 - target_q)**2)
    
    l_c, grads_c = jax.value_and_grad(c_loss_fn)(c_p)
    up_c, opt_c = tx_critic.update(grads_c, opt_c)
    return optax.apply_updates(c_p, up_c), opt_c

@jax.jit
def update_actor_and_targets(a_p, c_p, ta_p, tc_p, opt_a, z):
    # FIX: Hard-clip states to protect CNN from explosive gradients
    z_norm = jnp.clip(z / STATE_NORM_FACTOR, -5.0, 5.0).astype(jnp.float32)
    
    def a_loss_fn(p):
        norm_actions = actor.apply(p, z_norm)
        return -jnp.mean(critic.apply(c_p, z_norm, norm_actions)[0])
    
    l_a, grads_a = jax.value_and_grad(a_loss_fn)(a_p)
    up_a, opt_a = tx_actor.update(grads_a, opt_a)
    a_p = optax.apply_updates(a_p, up_a)
    
    tau = 0.005
    new_ta = jax.tree_util.tree_map(lambda new, old: tau*new + (1-tau)*old, a_p, ta_p)
    new_tc = jax.tree_util.tree_map(lambda new, old: tau*new + (1-tau)*old, c_p, tc_p)
    return a_p, new_ta, new_tc, opt_a

@partial(jax.jit, static_argnames=['add_noise'])
def get_batch_actions(a_p, z_batch, step_idx=0, key=None, add_noise=True):
    # FIX: Hard-clip states to protect CNN from explosive gradients
    z_norm = jnp.clip(z_batch / STATE_NORM_FACTOR, -5.0, 5.0).astype(jnp.float32)
    
    a_norm = actor.apply(a_p, z_norm) 
    
    if add_noise:
        noise_std = 5.0 / U_MAX
        noise_clip = 15.0 / U_MAX
        
        noise = jax.random.normal(key, a_norm.shape, dtype=jnp.float32) * noise_std
        noise = jnp.clip(noise, -noise_clip, noise_clip)
        a_norm = jnp.clip(a_norm + noise, -1.0, 1.0)
        
    return a_norm * U_MAX

@jax.jit
def parallel_physics_step(w_init_batch, actions):
    actions_64 = actions.astype(jnp.float64)
    
    def single_physics_step(w_single, act_single):
        w_hat = jnp.fft.fft2(w_single)
        def rk4_loop(i, w):
            return solver.rk4_step(
                w, dt_phys, kx, ky, k_sq, k_inv, VISCOSITY, forcing_hat, act_single
            )
        w_hat_next = jax.lax.fori_loop(0, SUBSTEPS, rk4_loop, w_hat)
        return jnp.fft.ifft2(w_hat_next).real
    
    next_w_batch = jax.vmap(single_physics_step)(w_init_batch, actions_64)
    
    is_invalid = jnp.logical_not(jnp.isfinite(next_w_batch).all(axis=(1, 2)))
    is_exploding = jnp.max(jnp.abs(next_w_batch), axis=(1, 2)) > 1000.0 
    dones_batch = jnp.logical_or(is_invalid, is_exploding)[:, None]
    
    safe_w = jnp.where(dones_batch[:, :, None], jnp.zeros_like(next_w_batch), next_w_batch)
    
    global_enstrophy = jnp.mean(jnp.square(safe_w), axis=(1, 2))[:, None]
    r_track = -jnp.log(global_enstrophy + 1.0)
    
    normalized_actions = actions / U_MAX
    effort = jnp.mean(jnp.square(normalized_actions), axis=1)[:, None]

    # Heavily penalize actions so the baseline becomes the preferred default
    r_effort = -(effort * .1) 
    
    rewards_batch = r_track + r_effort
    
    return safe_w, rewards_batch.astype(jnp.float32), dones_batch

# --- 3. FAST JIT-COMPILED EVALUATION ---
@partial(jax.jit, static_argnames=['max_steps'])
def fast_eval_episode(actor_params, init_state, max_steps):
    def step_fn(state, _):
        # FIX: Hard-clip states for evaluation stability too
        state_norm = jnp.clip(state / STATE_NORM_FACTOR, -5.0, 5.0).astype(jnp.float32)
        act = actor.apply(actor_params, state_norm[None, ...])
        
        act_flat = (act.squeeze(0) * U_MAX).astype(jnp.float64)
        w_hat = jnp.fft.fft2(state)
        
        def rk4_loop(i, w):
            return solver.rk4_step(
                w, dt_phys, kx, ky, k_sq, k_inv, VISCOSITY, forcing_hat, act_flat
            )
        w_hat_next = jax.lax.fori_loop(0, SUBSTEPS, rk4_loop, w_hat)
        next_state = jnp.fft.ifft2(w_hat_next).real
        
        enstrophy = jnp.mean(next_state**2)
        crashed = jnp.isnan(next_state).any() | jnp.isinf(next_state).any() | (jnp.max(jnp.abs(next_state)) > 1000.0)
        
        return next_state, (enstrophy, crashed)

    _, (enstrophies, crashes) = jax.lax.scan(step_fn, init_state, None, length=max_steps)
    
    return jnp.mean(enstrophies), enstrophies[-1], jnp.any(crashes)

# --- 4. THE SCAN-COMPILED TRAINING CHUNK ---
@jax.jit
def train_chunk(carry, step_indices, state_bank):
    def scan_step(carry, step_idx):
        buf, a_p, c_p, ta_p, tc_p, o_a, o_c, w, steps, rng = carry
        rng, act_k, res_k, samp_k, net_k = jax.random.split(rng, 5)
        
        def warmup_actions(_):
            # FIX: Gaussian zero-centered warmup to stay near natural enstrophy
            noise_std = U_MAX * 0.05
            noise = jax.random.normal(act_k, (NUM_PARALLEL_ENVS, N_AGENTS), dtype=jnp.float32) * noise_std
            return jnp.clip(noise, -U_MAX * 0.15, U_MAX * 0.15)
            
        def policy_actions(_):
            return get_batch_actions(a_p, w, step_idx, act_k, add_noise=True)
            
        actions = jax.lax.cond(step_idx < WARMUP_UPDATES, warmup_actions, policy_actions, None)
        
        next_w, rewards, dones = parallel_physics_step(w, actions)
        steps += 1
        truncs = steps >= MAX_ENV_STEPS
        needs_reset = jnp.logical_or(dones.flatten(), truncs)
        
        safe_next_w = jnp.where(dones[:, :, None], jnp.zeros_like(next_w), next_w)
        
        # FIX: -2000.0 Crash penalty
        safe_rewards = jnp.where(dones, -2000.0, rewards)
        
        new_buf = add_batch_to_buffer(buf, w, actions, safe_rewards, safe_next_w, dones)
        
        fresh_states = jax.random.choice(res_k, state_bank, shape=(NUM_PARALLEL_ENVS,))
        w_next = jnp.where(needs_reset[:, None, None], fresh_states, safe_next_w)
        steps_next = jnp.where(needs_reset, 0, steps)

        def do_network_updates(net_state):
            c_p, a_p, ta_p, tc_p, o_c, o_a = net_state
            
            bs, ba, br, bns, bd = sample_buffer(new_buf, ENV_BATCH_SIZE, samp_k)
            new_c_p, new_o_c = update_critic(c_p, ta_p, tc_p, o_c, bs, ba, br, bns, bd, net_k)
            
            def do_actor_update(_):
                return update_actor_and_targets(a_p, new_c_p, ta_p, tc_p, o_a, bs)
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

# --- Vectorized Training Loop ---
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

print("Starting Massively Parallel RL Training (Chunked & JITed 2D Turbulence)...")
start_time = time.time()

num_chunks = TOTAL_UPDATES // EVAL_INT

for chunk_idx in trange(num_chunks):
    start_step = chunk_idx * EVAL_INT
    step_indices = jnp.arange(start_step, start_step + EVAL_INT)
    
    carry, _ = train_chunk(carry, step_indices, state_bank)
    current_actor_params = carry[1] 
    
    eval_w = state_bank[0] 
    
    eval_e_mean, eval_e_final, crashed = fast_eval_episode(current_actor_params, eval_w, MAX_ENV_STEPS)
    
    current_total_step = start_step + EVAL_INT
    episode_num = current_total_step // MAX_ENV_STEPS
    
    if crashed:
        print(f"\nUpdate {current_total_step:05d} | Episode {episode_num} | Eval Mean: [CRASHED] | Eval Final: [CRASHED] | Time: {time.time()-start_time:.1f}s")
    else:
        print(f"\nUpdate {current_total_step:05d} | Episode {episode_num} | Eval Mean: {eval_e_mean:.4f} | Eval Final: {eval_e_final:.4f} | Time: {time.time()-start_time:.1f}s")

final_actor_params = carry[1]

os.makedirs('models', exist_ok=True)

with open('models/rl_turb_params.msgpack', 'wb') as f:
    f.write(flax.serialization.to_bytes({'actor': final_actor_params}))
print(f"Training finished in {time.time()-start_time:.1f}s. Weights saved.")