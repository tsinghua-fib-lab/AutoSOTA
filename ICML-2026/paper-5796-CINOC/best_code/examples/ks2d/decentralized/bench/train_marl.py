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

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(script_dir))

# Project imports
from env_ks2d import KS2DMARLEnv, extract_patches_2d_jit
from utils_hypemarl import get_sinusoidal_encoding
from examples.ks2d.decentralized.data_utils import get_batch_initial_conditions
from examples.ks2d.decentralized.dynamics_dual import PDEDynamics2D 
from examples.ks2d.decentralized.bench.models_marl import MARLActor2DKS, MARLCritic2DKS

# ==========================================
# 1. CONFIGURATIONS
# ==========================================
N_AGENTS = 100
L_DOMAIN = 32.0
N_GRID = 64

ENV_BATCH_SIZE = 128 
EVAL_INT = 50 
POLICY_DELAY = 2 

# KS2D Specific Control Timing
MAX_ENV_STEPS = 50     
SUBSTEPS = 10          
DT = 0.005             

# Vectorization Configs
NUM_PARALLEL_ENVS = 64
TOTAL_UPDATES = 100000 
WARMUP_UPDATES = 500
U_MAX = 5.0  

key = jax.random.PRNGKey(42)

# ==========================================
# 3. UTILS & ENVIRONMENT SETUP
# ==========================================
def get_2d_sinusoidal_encoding(p_2d, d=64, n=1000.0):
    pe_x = get_sinusoidal_encoding(p_2d[:, 0], d=d, n=n)
    pe_y = get_sinusoidal_encoding(p_2d[:, 1], d=d, n=n)
    return jnp.concatenate([pe_x, pe_y], axis=-1)

def direct_control_policy(action_params, u_obs, u_target, xi_fixed):
    return action_params

dynamics = PDEDynamics2D(policy_apply_fn=direct_control_policy)

dummy_pool = jnp.zeros((1, N_GRID, N_GRID))
env = KS2DMARLEnv(
    dynamics, initial_conditions=dummy_pool, n_agents=N_AGENTS, 
    N_grid=N_GRID, L=L_DOMAIN, dt=DT, substeps=SUBSTEPS, max_steps=MAX_ENV_STEPS
)

patch_size = env.patch_size 
local_y_dim = env.local_y_dim 
n_mu = env.n_mu 
pe_dim = 128 

stored_obs_dim = local_y_dim + n_mu 
total_input_dim = stored_obs_dim + pe_dim

xi_fixed = jnp.array(env.agent_positions)
xi_norm = jnp.array(env.xi_norm)
mu_jax = jnp.array(env.mu)
pe_jax = jnp.array(get_2d_sinusoidal_encoding(xi_norm, d=64))
target_state = jnp.zeros((N_GRID, N_GRID))

actor = MARLActor2DKS()
critic = MARLCritic2DKS()

key, *subkeys = jax.random.split(key, 4)
dummy_input = jnp.zeros((ENV_BATCH_SIZE, total_input_dim))
dummy_u = jnp.zeros((ENV_BATCH_SIZE, 1))

actor_params = actor.init(subkeys[0], dummy_input)
critic_params = critic.init(subkeys[1], dummy_input, dummy_u)

target_actor_params = jax.tree_util.tree_map(jnp.copy, actor_params)
target_critic_params = jax.tree_util.tree_map(jnp.copy, critic_params)

tx_actor = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-6))
tx_critic = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(5e-5))
opt_actor = tx_actor.init(actor_params)
opt_critic = tx_critic.init(critic_params)

# ==========================================
# 4. REPLAY BUFFER
# ==========================================
@struct.dataclass
class DeviceReplayBuffer:
    s: jnp.ndarray
    a: jnp.ndarray
    r: jnp.ndarray
    ns: jnp.ndarray
    d: jnp.ndarray
    ptr: jnp.int32
    size: jnp.int32
    max_size: int = struct.field(pytree_node=False)

    @classmethod
    def create(cls, max_size, s_dim, a_dim):
        return cls(
            s=jnp.zeros((max_size, N_AGENTS, s_dim), dtype=jnp.float32),
            a=jnp.zeros((max_size, N_AGENTS, a_dim), dtype=jnp.float32),
            r=jnp.zeros((max_size, N_AGENTS, 1), dtype=jnp.float32),
            ns=jnp.zeros((max_size, N_AGENTS, s_dim), dtype=jnp.float32),
            d=jnp.zeros((max_size, N_AGENTS, 1), dtype=jnp.float32),
            ptr=jnp.int32(0),
            size=jnp.int32(0),
            max_size=max_size
        )

@jax.jit
def add_batch_to_buffer(buffer, s_batch, a_batch, r_batch, ns_batch, d_batch):
    batch_size = s_batch.shape[0]
    indices = (buffer.ptr + jnp.arange(batch_size)) % buffer.max_size
    
    new_s = buffer.s.at[indices].set(s_batch.astype(jnp.float32))
    new_a = buffer.a.at[indices].set(a_batch.astype(jnp.float32))
    new_r = buffer.r.at[indices].set(r_batch.astype(jnp.float32))
    new_ns = buffer.ns.at[indices].set(ns_batch.astype(jnp.float32))
    new_d = buffer.d.at[indices].set(d_batch.astype(jnp.float32))
    
    new_ptr = (buffer.ptr + batch_size) % buffer.max_size
    new_size = jnp.minimum(buffer.size + batch_size, buffer.max_size)
    
    return buffer.replace(s=new_s, a=new_a, r=new_r, ns=new_ns, d=new_d, ptr=new_ptr, size=new_size)

@partial(jax.jit, static_argnames=['batch_size'])
def sample_buffer(buffer, batch_size, key):
    valid_range = jnp.minimum(buffer.size, buffer.max_size)
    indices = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=valid_range)
    return buffer.s[indices], buffer.a[indices], buffer.r[indices], buffer.ns[indices], buffer.d[indices]

buffer = DeviceReplayBuffer.create(12_500, stored_obs_dim, 1)

# ==========================================
# 5. TRAINING LOGIC
# ==========================================
@jax.jit
def build_marl_obs_batch(full_state_batch):
    def single_env_obs(state):
        y_local = extract_patches_2d_jit(state, target_state, xi_norm, patch_size, N_GRID)
        mu_broadcast = jnp.tile(mu_jax, (N_AGENTS, 1))
        return jnp.concatenate([y_local, mu_broadcast], axis=-1)
    return jax.vmap(single_env_obs)(full_state_batch)

@jax.jit
def update_critic(c_p, ta_p, tc_p, opt_c, x, u, r, nx, d, key):
    key, noise_key = jax.random.split(key)
    
    noise = jnp.clip(jax.random.normal(noise_key, u.shape) * (U_MAX * 0.1), -U_MAX * 0.5, U_MAX * 0.5)
    next_u = jnp.clip(actor.apply(ta_p, nx) + noise, -U_MAX, U_MAX)
    
    t_q1, t_q2 = critic.apply(tc_p, nx, next_u)
    target_q = r + 0.99 * (1.0 - d) * jnp.minimum(t_q1, t_q2)
    
    def c_loss_fn(p):
        q1, q2 = critic.apply(p, x, u)
        return jnp.mean((q1 - target_q)**2 + (q2 - target_q)**2)
    
    l_c, grads_c = jax.value_and_grad(c_loss_fn)(c_p)
    up_c, opt_c = tx_critic.update(grads_c, opt_c)
    return optax.apply_updates(c_p, up_c), opt_c

@jax.jit
def update_actor_and_targets(a_p, c_p, ta_p, tc_p, opt_a, x):
    def a_loss_fn(p):
        return -jnp.mean(critic.apply(c_p, x, actor.apply(p, x))[0])
    
    l_a, grads_a = jax.value_and_grad(a_loss_fn)(a_p)
    up_a, opt_a = tx_actor.update(grads_a, opt_a)
    a_p = optax.apply_updates(a_p, up_a)
    
    tau = 0.005
    new_ta = jax.tree_util.tree_map(lambda new, old: tau*new + (1-tau)*old, a_p, ta_p)
    new_tc = jax.tree_util.tree_map(lambda new, old: tau*new + (1-tau)*old, c_p, tc_p)
    return a_p, new_ta, new_tc, opt_a

@partial(jax.jit, static_argnames=['add_noise'])
def get_batch_actions(a_p, obs_batch_no_pe, key=None, add_noise=True):
    pe_expanded = jnp.tile(pe_jax[None, :, :], (obs_batch_no_pe.shape[0], 1, 1))
    full_obs = jnp.concatenate([obs_batch_no_pe, pe_expanded], axis=-1)
    
    # ACCELERATION: Removed double vmap! nn.Dense natively processes N_AGENTS.
    actions = actor.apply(a_p, full_obs)
    
    if add_noise:
        noise = jax.random.normal(key, actions.shape) * (U_MAX * 0.1)
        actions = jnp.clip(actions + noise, -U_MAX, U_MAX)
    return actions

@jax.jit
def parallel_marl_physics_step(u_batch, actions):
    acts_flat = actions.squeeze(-1) 
    
    def single_physics_step(u_single, act_single):
        traj = dynamics.unroll_controlled(
            u_init=u_single, xi_fixed=xi_fixed, u_target=target_state, params=act_single, 
            t_steps=1, substeps=SUBSTEPS, N_grid=N_GRID, L=L_DOMAIN, dt=DT, sigma=1.2
        )
        return traj[0][-1]
    
    next_u_batch = jax.vmap(single_physics_step)(u_batch, acts_flat)
    
    is_invalid = jnp.logical_not(jnp.isfinite(next_u_batch).all(axis=(1, 2)))
    is_exploding = jnp.max(jnp.abs(next_u_batch), axis=(1, 2)) > 100.0
    dones_batch = jnp.logical_or(is_invalid, is_exploding)
    
    safe_u = jnp.where(dones_batch[:, None, None], jnp.zeros_like(next_u_batch), next_u_batch)
    next_obs_batch_no_pe = build_marl_obs_batch(safe_u)
    
    # ACCURACY FIX: Scale down massive 2D energy so it doesn't break MARL local rewards
    global_energy = jnp.mean(jnp.square(safe_u), axis=(1, 2))
    y_local_err = next_obs_batch_no_pe[..., :patch_size**2] 
    local_rewards = -jnp.mean(jnp.square(y_local_err), axis=-1)
    
    rewards_batch = 0.5 * local_rewards + 0.5 * (-global_energy[:, None] / 10.0)
    rewards_batch = rewards_batch[..., None] 
    
    return safe_u, next_obs_batch_no_pe, rewards_batch, dones_batch

@partial(jax.jit, static_argnames=['max_steps'])
def fast_eval_episode(actor_params, init_state, max_steps):
    def step_fn(state, _):
        obs_no_pe = build_marl_obs_batch(state[None, ...]) 
        act = get_batch_actions(actor_params, obs_no_pe, None, add_noise=False)
        act_flat = act.squeeze() 
        
        traj = dynamics.unroll_controlled(
            u_init=state, xi_fixed=xi_fixed, u_target=target_state, params=act_flat, 
            t_steps=1, substeps=SUBSTEPS, N_grid=N_GRID, L=L_DOMAIN, dt=DT, sigma=1.2
        )
        next_state = traj[0][-1]
        
        energy = jnp.mean(next_state**2)
        crashed = jnp.isnan(next_state).any() | jnp.isinf(next_state).any() | (jnp.max(jnp.abs(next_state)) > 100.0)
        
        return next_state, (energy, crashed)

    _, (energies, crashes) = jax.lax.scan(step_fn, init_state, None, length=max_steps)
    return jnp.mean(energies), jnp.any(crashes)

# ACCELERATION: Memory donation prevents constant buffer reallocation in VRAM
@jax.jit(donate_argnums=(0,))
def train_chunk(carry, step_indices, state_bank):
    def scan_step(carry, step_idx):
        buf, a_p, c_p, ta_p, tc_p, o_a, o_c, u, obs, steps, rng = carry
        rng, act_k, res_k, samp_k, net_k = jax.random.split(rng, 5)
        
        def warmup_actions(_):
            return jax.random.uniform(act_k, (NUM_PARALLEL_ENVS, N_AGENTS, 1), minval=-U_MAX, maxval=U_MAX)
        def policy_actions(_):
            return get_batch_actions(a_p, obs, act_k, add_noise=True)
            
        actions = jax.lax.cond(step_idx < WARMUP_UPDATES, warmup_actions, policy_actions, None)
        
        next_u, next_obs, rewards, dones = parallel_marl_physics_step(u, actions)
        steps += 1
        truncs = steps >= MAX_ENV_STEPS
        needs_reset = jnp.logical_or(dones.flatten(), truncs)
        
        safe_rewards = jnp.where(dones[:, None, None], -100.0, rewards)
        dones_expanded = jnp.tile(dones[:, None, None], (1, N_AGENTS, 1))
        
        new_buf = add_batch_to_buffer(buf, obs, actions, safe_rewards, next_obs, dones_expanded)
        
        fresh_states = jax.random.choice(res_k, state_bank, shape=(NUM_PARALLEL_ENVS,))
        u_next = jnp.where(needs_reset[:, None, None], fresh_states, next_u)
        obs_next = build_marl_obs_batch(u_next) 
        steps_next = jnp.where(needs_reset, 0, steps)

        def do_network_updates(net_state):
            c_p, a_p, ta_p, tc_p, o_c, o_a = net_state
            
            bx, bu, br, bnx, bd = sample_buffer(new_buf, ENV_BATCH_SIZE, samp_k)
            
            bx_flat = bx.reshape(-1, stored_obs_dim)
            bu_flat = bu.reshape(-1, 1)
            br_flat = br.reshape(-1, 1)
            bnx_flat = bnx.reshape(-1, stored_obs_dim)
            bd_flat = bd.reshape(-1, 1)
            
            pe_tiled = jnp.tile(pe_jax, (ENV_BATCH_SIZE, 1))
            
            bx_full = jnp.concatenate([bx_flat, pe_tiled], axis=-1)
            bnx_full = jnp.concatenate([bnx_flat, pe_tiled], axis=-1)
            
            new_c_p, new_o_c = update_critic(c_p, ta_p, tc_p, o_c, bx_full, bu_flat, br_flat, bnx_full, bd_flat, net_k)
            
            def do_actor_update(_):
                return update_actor_and_targets(a_p, new_c_p, ta_p, tc_p, o_a, bx_full)
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

        new_carry = (new_buf, a_p, c_p, ta_p, tc_p, o_a, o_c, u_next, obs_next, steps_next, rng)
        return new_carry, None

    return jax.lax.scan(scan_step, carry, step_indices)

# ==========================================
# 6. EXECUTION LOOP
# ==========================================
print("Loading 2D KS Initial Conditions...")
data_dir = Path('../../data')
data_dir.mkdir(parents=True, exist_ok=True)
file_path = data_dir / 'ks2d_chaotic_ics_64.pkl'

if file_path.exists():
    with open(file_path, 'rb') as f:
        state_bank = jnp.array(pickle.load(f))
    print(f"Loaded {len(state_bank)} ICs from {file_path}")
else:
    print("Generating ICs (this may take a few minutes for KS2D)...")
    state_bank = get_batch_initial_conditions(key, 500, N_GRID, L_DOMAIN)
    with open(file_path, 'wb') as f:
        pickle.dump(np.array(state_bank), f)

key, subkey = jax.random.split(key)
u_batch = jax.random.choice(subkey, state_bank, shape=(NUM_PARALLEL_ENVS,))
obs_batch = build_marl_obs_batch(u_batch)
env_step_counts = jnp.zeros(NUM_PARALLEL_ENVS)

carry = (
    buffer, actor_params, critic_params, target_actor_params, target_critic_params,
    opt_actor, opt_critic, u_batch, obs_batch, env_step_counts, key
)

print("Starting Massively Parallel MARL Training (Chunked & JITed 2D KS)...")
start_time = time.time()

num_chunks = TOTAL_UPDATES // EVAL_INT

for chunk_idx in trange(num_chunks):
    start_step = chunk_idx * EVAL_INT
    step_indices = jnp.arange(start_step, start_step + EVAL_INT)
    
    carry, _ = train_chunk(carry, step_indices, state_bank)
    current_actor_params = carry[1] 
    
    eval_u = state_bank[0] 
    eval_e, crashed = fast_eval_episode(current_actor_params, eval_u, MAX_ENV_STEPS)
    
    current_total_step = start_step + EVAL_INT
    episode_num = current_total_step // MAX_ENV_STEPS
    
    if crashed:
        print(f"\nUpdate {current_total_step:05d} | Episode {episode_num} | Eval Energy: [CRASHED] | Time: {time.time()-start_time:.1f}s")
    else:
        print(f"\nUpdate {current_total_step:05d} | Episode {episode_num} | Eval Energy: {eval_e:.6f} | Time: {time.time()-start_time:.1f}s")

final_actor_params = carry[1]
models_dir = Path('models')
models_dir.mkdir(exist_ok=True)
with open(models_dir / 'marl_ks2d_params.msgpack', 'wb') as f:
    f.write(flax.serialization.to_bytes({'actor': final_actor_params}))
print(f"Training finished in {time.time()-start_time:.1f}s. Weights saved.")