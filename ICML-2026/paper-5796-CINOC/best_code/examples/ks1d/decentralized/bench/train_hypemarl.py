import jax
import jax.numpy as jnp
import optax
import flax.serialization
from flax import struct
import numpy as np
import time
from pathlib import Path
import sys
from functools import partial

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(script_dir))

# Project imports
from examples.ks1d.decentralized.bench.env_ks import KSHypeMARLEnv
from models_hypemarl import HyperActor, HyperCritic, SurrogateModel
from utils_hypemarl import get_sinusoidal_encoding
from examples.ks1d.decentralized.data_utils import evolve_to_attractor
from examples.ks1d.decentralized.dynamics_dual import PDEDynamics 

# --- Configurations ---
USE_MB_HYPEMARL = False  
N_AGENTS = 8
L_DOMAIN = 22.0
N_GRID = 128
ENV_BATCH_SIZE = 64
EVAL_INT = 500
POLICY_DELAY = 2 
MAX_ENV_STEPS = 200

# Vectorization Configs
NUM_PARALLEL_ENVS = 256
TOTAL_UPDATES = 50000 
WARMUP_UPDATES = 500

# MB-HypeMARL specific
IMAGINARY_RATIO = 10  
IMAGINATION_HORIZON = 50  

# --- Initialization ---
key = jax.random.PRNGKey(42)

def direct_control_policy(action_params, u_obs, u_target, xi_fixed):
    return action_params

dynamics = PDEDynamics(policy_apply_fn=direct_control_policy)
env = KSHypeMARLEnv(dynamics, n_agents=N_AGENTS, N_grid=N_GRID, L=L_DOMAIN, max_steps=MAX_ENV_STEPS)

local_y_dim = 40 
n_mu = env.n_mu
pe_dim = 2048
z_dim = pe_dim + n_mu

# Static JAX Arrays
xi_fixed = jnp.array(env.agent_positions)
xi_norm = jnp.array(env.xi_norm)
mu_jax = jnp.array(env.mu)
pe_jax = jnp.array(get_sinusoidal_encoding(xi_fixed, d=pe_dim))
target_state = jnp.zeros(N_GRID)
window_size = env.window_size

# Models
actor = HyperActor()
critic = HyperCritic()
surrogate = SurrogateModel()

key, *subkeys = jax.random.split(key, 6)
dummy_z = jnp.zeros((ENV_BATCH_SIZE, z_dim))
dummy_y = jnp.zeros((ENV_BATCH_SIZE, local_y_dim))
dummy_u = jnp.zeros((ENV_BATCH_SIZE, 1))
dummy_mu = jnp.zeros((ENV_BATCH_SIZE, n_mu))

actor_params = actor.init(subkeys[0], dummy_z, dummy_y)
critic1_params = critic.init(subkeys[1], dummy_z, dummy_y, dummy_u)
critic2_params = critic.init(subkeys[2], dummy_z, dummy_y, dummy_u)
surrogate_params = surrogate.init(subkeys[3], dummy_y, dummy_u, dummy_mu)

target_actor_params = actor_params
target_critic1_params = critic1_params
target_critic2_params = critic2_params

tx_actor = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-6))
tx_critic = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(5e-5))
tx_surrogate = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-4))

opt_actor = tx_actor.init(actor_params)
opt_critic1 = tx_critic.init(critic1_params)
opt_critic2 = tx_critic.init(critic2_params)
opt_surrogate = tx_surrogate.init(surrogate_params)

# --- 1. MEMORY-OPTIMIZED DUAL BUFFERS ---
@struct.dataclass
class DualDeviceReplayBuffer:
    y: jnp.ndarray
    mu: jnp.ndarray # We store mu, but completely omit PE to save VRAM
    a: jnp.ndarray
    r: jnp.ndarray
    ny: jnp.ndarray
    d: jnp.ndarray
    ptr: jnp.int32
    size: jnp.int32
    max_size: int = struct.field(pytree_node=False)

    @classmethod
    def create(cls, max_size, y_dim, mu_dim, a_dim):
        return cls(
            y=jnp.zeros((max_size, N_AGENTS, y_dim), dtype=jnp.float32),
            mu=jnp.zeros((max_size, N_AGENTS, mu_dim), dtype=jnp.float32),
            a=jnp.zeros((max_size, N_AGENTS, a_dim), dtype=jnp.float32),
            r=jnp.zeros((max_size, N_AGENTS, 1), dtype=jnp.float32),
            ny=jnp.zeros((max_size, N_AGENTS, y_dim), dtype=jnp.float32),
            d=jnp.zeros((max_size, N_AGENTS, 1), dtype=jnp.float32),
            ptr=jnp.int32(0),
            size=jnp.int32(0),
            max_size=max_size
        )

@jax.jit
def add_batch_to_buffer(buffer, y_b, mu_b, a_b, r_b, ny_b, d_b):
    batch_size = y_b.shape[0]
    indices = (buffer.ptr + jnp.arange(batch_size)) % buffer.max_size
    
    new_y = buffer.y.at[indices].set(y_b)
    new_mu = buffer.mu.at[indices].set(mu_b)
    new_a = buffer.a.at[indices].set(a_b)
    new_r = buffer.r.at[indices].set(r_b)
    new_ny = buffer.ny.at[indices].set(ny_b)
    new_d = buffer.d.at[indices].set(d_b)
    
    new_ptr = (buffer.ptr + batch_size) % buffer.max_size
    new_size = jnp.minimum(buffer.size + batch_size, buffer.max_size)
    
    return buffer.replace(y=new_y, mu=new_mu, a=new_a, r=new_r, ny=new_ny, d=new_d, ptr=new_ptr, size=new_size)

@partial(jax.jit, static_argnames=['batch_size'])
def sample_buffer(buffer, batch_size, key):
    valid_range = jnp.minimum(buffer.size, buffer.max_size)
    indices = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=valid_range)
    return buffer.y[indices], buffer.mu[indices], buffer.a[indices], buffer.r[indices], buffer.ny[indices], buffer.d[indices]

# Reduced buffer size to 100,000 for lower memory footprint
agent_buffer = DualDeviceReplayBuffer.create(100_000, local_y_dim, n_mu, 1)
rom_buffer = DualDeviceReplayBuffer.create(100_000, local_y_dim, n_mu, 1)

# --- 2. PURE JAX OBSERVATIONS ---
@partial(jax.jit, static_argnames=['window_size'])
def extract_patches_jax(full_state, target_st, xi_n, window_size):
    error = full_state - target_st
    error_grad = jnp.gradient(error)
    half_window = window_size // 2
    padded_error = jnp.pad(error, (half_window, half_window), mode='wrap')
    padded_grad = jnp.pad(error_grad, (half_window, half_window), mode='wrap')

    def get_local_obs(xi):
        center_idx = jax.lax.stop_gradient((xi * (full_state.shape[0] - 1)).astype(int)) + half_window
        start = center_idx - half_window
        p_err = jax.image.resize(jax.lax.dynamic_slice(padded_error, (start,), (window_size,)), (20,), method='bilinear')
        p_grad = jax.image.resize(jax.lax.dynamic_slice(padded_grad, (start,), (window_size,)), (20,), method='bilinear')
        return jnp.concatenate([p_err, p_grad])
    return jax.vmap(get_local_obs)(xi_n)

@jax.jit
def build_marl_obs_batch(full_state_batch):
    def single_env_obs(state):
        y_local = extract_patches_jax(state, target_state, xi_norm, window_size)
        mu_broadcast = jnp.tile(mu_jax, (N_AGENTS, 1))
        return y_local, mu_broadcast
    return jax.vmap(single_env_obs)(full_state_batch)

# --- 3. JIT TRAINING & ROLLOUT ---
@jax.jit
def train_surrogate_step(s_params, opt_s, y, u, mu, next_y):
    def loss_fn(p):
        pred_y = surrogate.apply(p, y, u, mu)
        return jnp.mean((pred_y - next_y)**2)
    loss, grads = jax.value_and_grad(loss_fn)(s_params)
    updates, opt_s = tx_surrogate.update(grads, opt_s)
    return optax.apply_updates(s_params, updates), opt_s

@jax.jit
def update_critics(c1_p, c2_p, ta_p, tc1_p, tc2_p, opt_c1, opt_c2, y, mu, u, r, ny, d, key):
    key, noise_key = jax.random.split(key)
    
    # Dynamically build Z to save buffer memory
    pe_tiled = jnp.tile(pe_jax, (y.shape[0] // N_AGENTS, 1))
    z = jnp.concatenate([pe_tiled, mu], axis=-1)
    
    noise = jnp.clip(jax.random.normal(noise_key, u.shape) * 0.2, -0.5, 0.5)
    next_u = jnp.clip(actor.apply(ta_p, z, ny) + noise, -5.0, 5.0)
    
    q1_target = critic.apply(tc1_p, z, ny, next_u)
    q2_target = critic.apply(tc2_p, z, ny, next_u)
    
    # Clip Target using Batch Extrema (HypeMARL trick)
    min_q = jnp.minimum(q1_target, q2_target)
    q_batch_min, q_batch_max = jnp.min(min_q), jnp.max(min_q)
    clipped_min_q = jnp.clip(min_q, q_batch_min, q_batch_max) 
    
    target_q = jax.lax.stop_gradient(r + 0.99 * (1.0 - d) * clipped_min_q)
    
    # Use Huber Loss for Critics
    def q_loss_fn(p):
        q_pred = critic.apply(p, z, y, u)
        return jnp.mean(optax.huber_loss(q_pred, target_q))
    
    loss_c1, grads_c1 = jax.value_and_grad(q_loss_fn)(c1_p)
    loss_c2, grads_c2 = jax.value_and_grad(q_loss_fn)(c2_p)
    
    up_c1, opt_c1 = tx_critic.update(grads_c1, opt_c1)
    up_c2, opt_c2 = tx_critic.update(grads_c2, opt_c2)
    
    return optax.apply_updates(c1_p, up_c1), optax.apply_updates(c2_p, up_c2), opt_c1, opt_c2

@jax.jit
def update_actor_and_targets(a_p, c1_p, c2_p, ta_p, tc1_p, tc2_p, opt_a, y, mu):
    pe_tiled = jnp.tile(pe_jax, (y.shape[0] // N_AGENTS, 1))
    z = jnp.concatenate([pe_tiled, mu], axis=-1)
    
    # HypeMARL Actor Loss averages the two critic networks
    def a_loss_fn(p):
        act = actor.apply(p, z, y)
        q1 = critic.apply(c1_p, z, y, act)
        q2 = critic.apply(c2_p, z, y, act)
        return -jnp.mean(0.5 * (q1 + q2))
    
    loss_a, grads_a = jax.value_and_grad(a_loss_fn)(a_p)
    up_a, opt_a = tx_actor.update(grads_a, opt_a)
    a_p = optax.apply_updates(a_p, up_a)
    
    tau = 0.005
    ta_p = jax.tree_util.tree_map(lambda n, o: tau*n + (1-tau)*o, a_p, ta_p)
    tc1_p = jax.tree_util.tree_map(lambda n, o: tau*n + (1-tau)*o, c1_p, tc1_p)
    tc2_p = jax.tree_util.tree_map(lambda n, o: tau*n + (1-tau)*o, c2_p, tc2_p)
    
    return a_p, ta_p, tc1_p, tc2_p, opt_a

@partial(jax.jit, static_argnames=['add_noise'])
def get_batch_actions(a_p, y_batch, mu_batch, key, step, add_noise=True):
    num_envs = y_batch.shape[0]
    
    pe_expanded = jnp.tile(pe_jax[None, :, :], (num_envs, 1, 1))
    z_batch = jnp.concatenate([pe_expanded, mu_batch], axis=-1)
    
    y_flat = y_batch.reshape(-1, local_y_dim)
    z_flat = z_batch.reshape(-1, z_dim)
    
    actions_flat = actor.apply(a_p, z_flat, y_flat)
    actions = actions_flat.reshape(num_envs, N_AGENTS, 1)
    
    # Implemented Linear Decay for Exploration Noise
    if add_noise:
        noise_scale = jnp.maximum(0.05, 0.5 - (step / 50000.0) * (0.5 - 0.05))
        noise = jax.random.normal(key, actions.shape) * noise_scale
        actions = jnp.clip(actions + noise, -5.0, 5.0)
        
    return actions

@jax.jit
def parallel_marl_physics_step(u_batch, actions):
    acts_flat = actions.squeeze(-1) 
    def single_physics_step(u_single, act_single):
        traj = dynamics.unroll_controlled(u_single, xi_fixed, jnp.zeros(N_GRID), act_single, 1, N_grid=N_GRID, L=L_DOMAIN)
        return traj[0][-1]
    
    next_u_batch = jax.vmap(single_physics_step)(u_batch, acts_flat)
    
    is_invalid = jnp.logical_not(jnp.isfinite(next_u_batch).all(axis=-1, keepdims=True))
    is_exploding = jnp.max(jnp.abs(next_u_batch), axis=-1, keepdims=True) > 100.0
    dones_batch = jnp.logical_or(is_invalid, is_exploding)
    
    safe_u = jnp.where(dones_batch, jnp.zeros_like(next_u_batch), next_u_batch)
    next_y_batch, next_mu_batch = build_marl_obs_batch(safe_u)
    
    center_errors = next_y_batch[:, :, 10]
    rewards_batch = -jnp.square(center_errors)[..., None] 
    
    return safe_u, next_y_batch, next_mu_batch, rewards_batch, dones_batch

@jax.jit
def fast_imagination_step(s_params, a_p, y, mu, key, step):
    act = get_batch_actions(a_p, y, mu, key, step, add_noise=True)
    
    y_flat = y.reshape(-1, local_y_dim)
    act_flat = act.reshape(-1, 1)
    mu_flat = mu.reshape(-1, n_mu)
    
    ny_flat = surrogate.apply(s_params, y_flat, act_flat, mu_flat)
    ny = ny_flat.reshape(y.shape)
    
    is_invalid = jnp.logical_not(jnp.isfinite(ny).all(axis=-1, keepdims=True))
    safe_ny = jnp.where(is_invalid, jnp.zeros_like(ny), ny)
    
    rewards = jnp.where(is_invalid, -100.0, -jnp.square(safe_ny[:, :, 10])[..., None])
    dones = is_invalid 
    
    return safe_ny, act, rewards, dones

@partial(jax.jit, static_argnames=['max_steps'])
def fast_eval_episode(a_p, init_state, max_steps):
    def step_fn(state, _):
        y, mu = build_marl_obs_batch(state[None, ...]) 
        act = get_batch_actions(a_p, y, mu, None, step=0, add_noise=False)
        act_flat = act.squeeze() 
        
        traj = dynamics.unroll_controlled(state, xi_fixed, jnp.zeros(N_GRID), act_flat, 1, N_grid=N_GRID, L=L_DOMAIN)
        next_state = traj[0][-1]
        
        energy = jnp.mean(next_state**2)
        crashed = jnp.isnan(next_state).any() | jnp.isinf(next_state).any() | (jnp.max(jnp.abs(next_state)) > 100.0)
        return next_state, (energy, crashed)

    _, (energies, crashes) = jax.lax.scan(step_fn, init_state, None, length=max_steps)
    return jnp.mean(energies), jnp.any(crashes)


# --- Vectorized Training Loop ---
print("Pre-generating starting state bank (Vectorized)...")
bank_keys = jax.random.split(key, 1000)
state_bank = jax.vmap(lambda k: evolve_to_attractor(k, N_GRID, L_DOMAIN))(bank_keys)

key, subkey = jax.random.split(key)
u_batch = jax.random.choice(subkey, state_bank, shape=(NUM_PARALLEL_ENVS,))
y_batch, mu_batch = build_marl_obs_batch(u_batch)
env_step_counts = jnp.zeros(NUM_PARALLEL_ENVS)

python_agent_buffer_size = 0
python_rom_buffer_size = 0

print(f"Starting Massively Parallel Training: {'MB-HypeMARL' if USE_MB_HYPEMARL else 'HypeMARL'}")
start_time = time.time()

for update_step in range(TOTAL_UPDATES):
    
    if update_step % EVAL_INT == 0:
        eval_u = state_bank[0] 
        eval_e, crashed = fast_eval_episode(actor_params, eval_u, MAX_ENV_STEPS)
        episode_num = update_step // MAX_ENV_STEPS
        
        if crashed:
            print(f"Update {update_step:06d} | Episode {episode_num} | Eval Energy: [CRASHED] | Time: {time.time()-start_time:.1f}s")
        else:
            print(f"Update {update_step:06d} | Episode {episode_num} | Eval Energy: {eval_e:.6f} | Time: {time.time()-start_time:.1f}s")

    # 1. Real World Rollout
    key, act_key, reset_key = jax.random.split(key, 3)
    
    if update_step < WARMUP_UPDATES:
        actions = jax.random.uniform(act_key, (NUM_PARALLEL_ENVS, N_AGENTS, 1), minval=-5.0, maxval=5.0)
    else:
        actions = get_batch_actions(actor_params, y_batch, mu_batch, act_key, step=update_step, add_noise=True)
        
    next_u_batch, next_y_batch, next_mu_batch, rewards_batch, dones_batch = parallel_marl_physics_step(u_batch, actions)
    
    env_step_counts += 1
    truncations_batch = env_step_counts >= MAX_ENV_STEPS
    safe_rewards = jnp.where(dones_batch[:, None], -100.0, rewards_batch)
    dones_expanded = jnp.tile(dones_batch[:, None, :], (1, N_AGENTS, 1))

    # Push to both buffers
    agent_buffer = add_batch_to_buffer(agent_buffer, y_batch, mu_batch, actions, safe_rewards, next_y_batch, dones_expanded)
    rom_buffer = add_batch_to_buffer(rom_buffer, y_batch, mu_batch, actions, safe_rewards, next_y_batch, dones_expanded)
    
    # Capped python buffer size counters to 100,000
    python_agent_buffer_size = min(python_agent_buffer_size + NUM_PARALLEL_ENVS, 100_000)
    python_rom_buffer_size = min(python_rom_buffer_size + NUM_PARALLEL_ENVS, 100_000)

    # 2. Handle Resets (No Sync)
    needs_reset = jnp.logical_or(dones_batch.flatten(), truncations_batch)
    fresh_states = jax.random.choice(reset_key, state_bank, shape=(NUM_PARALLEL_ENVS,))
    u_batch = jnp.where(needs_reset[:, None], fresh_states, next_u_batch)
    
    y_batch, mu_batch = build_marl_obs_batch(u_batch)
    env_step_counts = jnp.where(needs_reset, 0, env_step_counts)
        
    # 3. MB-HypeMARL Surrogate & Imagination
    if USE_MB_HYPEMARL and python_rom_buffer_size > ENV_BATCH_SIZE:
        b_y, b_mu, b_act, _, b_ny, _ = sample_buffer(rom_buffer, ENV_BATCH_SIZE, subkey)
        
        surrogate_params, opt_surrogate = train_surrogate_step(
            surrogate_params, opt_surrogate, 
            b_y.reshape(-1, local_y_dim), b_act.reshape(-1, 1), b_mu.reshape(-1, n_mu), b_ny.reshape(-1, local_y_dim)
        )
        
        if update_step >= WARMUP_UPDATES:
            for _ in range(IMAGINARY_RATIO):
                img_y, img_mu, _, _, _, _ = sample_buffer(rom_buffer, ENV_BATCH_SIZE, subkey)
                
                for _ in range(IMAGINATION_HORIZON):
                    key, subkey = jax.random.split(key)
                    img_ny, img_act, img_rew, img_done = fast_imagination_step(surrogate_params, actor_params, img_y, img_mu, subkey, update_step)
                    
                    agent_buffer = add_batch_to_buffer(agent_buffer, img_y, img_mu, img_act, img_rew, img_ny, img_done)
                    python_agent_buffer_size = min(python_agent_buffer_size + ENV_BATCH_SIZE, 100_000)
                    img_y = img_ny

    # 4. TD3 Updates 
    if python_agent_buffer_size > ENV_BATCH_SIZE:
        bx, bmu, bu, br, bnx, bd = sample_buffer(agent_buffer, ENV_BATCH_SIZE, subkey) 
        key, subkey = jax.random.split(key)
        
        bx_flat = bx.reshape(-1, local_y_dim)
        bmu_flat = bmu.reshape(-1, n_mu)
        bu_flat = bu.reshape(-1, 1)
        br_flat = br.reshape(-1, 1)
        bnx_flat = bnx.reshape(-1, local_y_dim)
        bd_flat = bd.reshape(-1, 1)
        
        critic1_params, critic2_params, opt_critic1, opt_critic2 = update_critics(
            critic1_params, critic2_params, target_actor_params, target_critic1_params, target_critic2_params,
            opt_critic1, opt_critic2, bx_flat, bmu_flat, bu_flat, br_flat, bnx_flat, bd_flat, subkey
        )
        
        if update_step % POLICY_DELAY == 0:
            actor_params, target_actor_params, target_critic1_params, target_critic2_params, opt_actor = update_actor_and_targets(
                actor_params, critic1_params, critic2_params, target_actor_params, target_critic1_params, target_critic2_params,
                opt_actor, bx_flat, bmu_flat
            )

# Save
with open('models/hypemarl_params.msgpack', 'wb') as f:
    f.write(flax.serialization.to_bytes({
        'actor': actor_params, 'critic1': critic1_params, 'critic2': critic2_params, 'surrogate': surrogate_params
    }))
print(f"Training finished in {time.time()-start_time:.1f}s. Weights saved.")