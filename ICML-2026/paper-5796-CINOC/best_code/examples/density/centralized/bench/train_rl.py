import os
os.environ["JAX_ENABLE_X64"] = "False"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
jax.config.update("jax_enable_x64", False)
jax.config.update("jax_disable_jit", False)
cache_dir = os.path.join(os.path.dirname(__file__), ".jax_cache")
os.makedirs(cache_dir, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", cache_dir)

import jax.numpy as jnp
import optax
import flax.serialization
from flax import struct
import flax.linen as nn
import numpy as np
import time
from pathlib import Path
import sys
from functools import partial
from tqdm import trange
from typing import Callable, Tuple

# Add project root
script_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(script_dir))

from examples.density.centralized.dynamics import ns2d_step_jax

# =============================================================================
# Load Data & Config
# =============================================================================
print("Loading NS2D starting state & target banks from dataset...")
data_dir = Path(__file__).parent.parent.parent / 'data'

config = np.load(data_dir / 'config.npz')
Nx = int(config['Nx'])
Ny = int(config['Ny'])
dt = float(config['dt'])

train_data = np.load(data_dir / 'train_data.npz')
rho_init_bank = jnp.array(train_data['rho_init'], dtype=jnp.float32)
rho_target_bank = jnp.array(train_data['rho_target'], dtype=jnp.float32)

# =============================================================================
# Configurations
# =============================================================================
N_AGENTS = 9 

ENV_BATCH_SIZE = 256 
EVAL_INT = 500
POLICY_DELAY = 2 
MAX_ENV_STEPS = 150 

NUM_PARALLEL_ENVS = 128
TOTAL_UPDATES = 100000 
WARMUP_UPDATES = 500

# Physics Constants 
BUOYANCY = 0.0
SIGMA_PUSH = 0.2
PUSH_MAX = 0.8
R_SAFE = 0.15


# =============================================================================
# Centralized Models (Dense MLP based for Flattened 2D Grids)
# =============================================================================
class CentralizedActor(nn.Module):
    n_agents: int
    hidden_dim: int = 256
    
    @nn.compact
    def __call__(self, rho, target, xi):
        # Flatten the 2D spatial grids: (..., Nx, Ny) -> (..., Nx*Ny)
        rho_flat = rho.reshape((*rho.shape[:-2], -1))
        target_flat = target.reshape((*target.shape[:-2], -1))
        
        # Flatten the 2D agent positions: (..., n_agents, 2) -> (..., n_agents*2)
        xi_flat = xi.reshape((*xi.shape[:-2], -1))
        
        # Concatenate all global information into a single 1D vector per batch
        x = jnp.concatenate([rho_flat, target_flat, xi_flat], axis=-1)
        
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        
        # Normalization trick for stability
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1.0)
        
        # Dual Heads for Push Velocity (vx, vy) for ALL agents simultaneously
        vx_raw = nn.Dense(self.n_agents)(x)
        vy_raw = nn.Dense(self.n_agents)(x)
        
        vx_out = PUSH_MAX * jnp.tanh(vx_raw)
        vy_out = PUSH_MAX * jnp.tanh(vy_raw)
        
        # Stack to form output shape: (..., n_agents, 2)
        return jnp.stack([vx_out, vy_out], axis=-1)

class CentralizedCritic(nn.Module):
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, rho, target, xi, actions):
        # Flatten the 2D spatial grids
        rho_flat = rho.reshape((*rho.shape[:-2], -1))
        target_flat = target.reshape((*target.shape[:-2], -1))
        
        # Flatten agent positions and actions
        xi_flat = xi.reshape((*xi.shape[:-2], -1))
        actions_flat = actions.reshape((*actions.shape[:-2], -1))
        
        xu = jnp.concatenate([rho_flat, target_flat, xi_flat, actions_flat], axis=-1)
        
        # Q1
        q1 = nn.Dense(self.hidden_dim)(xu)
        q1 = nn.relu(q1)
        q1 = nn.Dense(self.hidden_dim)(q1)
        q1 = nn.relu(q1)
        q1 = nn.Dense(1)(q1)

        # Q2
        q2 = nn.Dense(self.hidden_dim)(xu)
        q2 = nn.relu(q2)
        q2 = nn.Dense(self.hidden_dim)(q2)
        q2 = nn.relu(q2)
        q2 = nn.Dense(1)(q2)
        
        return q1, q2

# =============================================================================
# Dynamics Wrapper
# =============================================================================
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

# --- Initialization ---
key = jax.random.PRNGKey(42)

def direct_control_policy(action_params, rho_obs, rho_target, xi_fixed):
    return action_params.astype(jnp.float32)

dynamics = DensityDynamicsWrapper(policy_apply_fn=direct_control_policy)

actor = CentralizedActor(n_agents=N_AGENTS)
critic = CentralizedCritic()

key, *subkeys = jax.random.split(key, 4)

# Native 2D inputs
dummy_rho = jnp.zeros((ENV_BATCH_SIZE, Nx, Ny))
dummy_target = jnp.zeros((ENV_BATCH_SIZE, Nx, Ny))
dummy_xi = jnp.zeros((ENV_BATCH_SIZE, N_AGENTS, 2))
dummy_act = jnp.zeros((ENV_BATCH_SIZE, N_AGENTS, 2))

actor_params = actor.init(subkeys[0], dummy_rho, dummy_target, dummy_xi)
critic_params = critic.init(subkeys[1], dummy_rho, dummy_target, dummy_xi, dummy_act)

target_actor_params = jax.tree_util.tree_map(jnp.copy, actor_params)
target_critic_params = jax.tree_util.tree_map(jnp.copy, critic_params)

tx_actor = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-4)) 
tx_critic = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(5e-4))
opt_actor = tx_actor.init(actor_params)
opt_critic = tx_critic.init(critic_params)

# --- 1. DYNAMIC REPLAY BUFFER ---
@struct.dataclass
class DeviceReplayBuffer:
    rho: jnp.ndarray
    target: jnp.ndarray
    xi: jnp.ndarray
    a: jnp.ndarray
    r: jnp.ndarray
    nrho: jnp.ndarray
    nxi: jnp.ndarray
    d: jnp.ndarray
    ptr: jnp.int32
    size: jnp.int32
    max_size: int = struct.field(pytree_node=False)

    @classmethod
    def create(cls, max_size):
        return cls(
            rho=jnp.zeros((max_size, Nx, Ny), dtype=jnp.float32),
            target=jnp.zeros((max_size, Nx, Ny), dtype=jnp.float32),
            xi=jnp.zeros((max_size, N_AGENTS, 2), dtype=jnp.float32),
            a=jnp.zeros((max_size, N_AGENTS, 2), dtype=jnp.float32),
            r=jnp.zeros((max_size, 1), dtype=jnp.float32),
            nrho=jnp.zeros((max_size, Nx, Ny), dtype=jnp.float32),
            nxi=jnp.zeros((max_size, N_AGENTS, 2), dtype=jnp.float32),
            d=jnp.zeros((max_size, 1), dtype=jnp.float32),
            ptr=jnp.int32(0), size=jnp.int32(0), max_size=max_size
        )

@jax.jit
def add_batch_to_buffer(buffer, rho_b, t_b, xi_b, a_b, r_b, nrho_b, nxi_b, d_b):
    batch_size = rho_b.shape[0]
    indices = (buffer.ptr + jnp.arange(batch_size)) % buffer.max_size
    
    new_rho = buffer.rho.at[indices].set(rho_b)
    new_t = buffer.target.at[indices].set(t_b)
    new_xi = buffer.xi.at[indices].set(xi_b)
    new_a = buffer.a.at[indices].set(a_b)
    new_r = buffer.r.at[indices].set(r_b)
    new_nrho = buffer.nrho.at[indices].set(nrho_b)
    new_nxi = buffer.nxi.at[indices].set(nxi_b)
    new_d = buffer.d.at[indices].set(d_b)
    
    new_ptr = (buffer.ptr + batch_size) % buffer.max_size
    new_size = jnp.minimum(buffer.size + batch_size, buffer.max_size)
    
    return buffer.replace(rho=new_rho, target=new_t, xi=new_xi, a=new_a, r=new_r, 
                          nrho=new_nrho, nxi=new_nxi, d=new_d, ptr=new_ptr, size=new_size)

@partial(jax.jit, static_argnames=['batch_size'])
def sample_buffer(buffer, batch_size, key):
    valid_range = jnp.minimum(buffer.size, buffer.max_size)
    indices = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=valid_range)
    return (buffer.rho[indices], buffer.target[indices], buffer.xi[indices], 
            buffer.a[indices], buffer.r[indices], buffer.nrho[indices], 
            buffer.nxi[indices], buffer.d[indices])

buffer = DeviceReplayBuffer.create(50_000) 

# --- 3. JIT TRAINING & ROLLOUT ---
@jax.jit
def update_critic(c_p, ta_p, tc_p, opt_c, rho, target, xi, a, r, nrho, nxi, d, key):
    key, noise_key = jax.random.split(key)
    
    noise_scale = jnp.array([PUSH_MAX, PUSH_MAX]) * 0.1
    noise = jnp.clip(jax.random.normal(noise_key, a.shape) * noise_scale, -0.5 * noise_scale, 0.5 * noise_scale)
    
    raw_next_a = actor.apply(ta_p, nrho, target, nxi) + noise
    next_a = jnp.clip(raw_next_a, jnp.array([-PUSH_MAX, -PUSH_MAX]), jnp.array([PUSH_MAX, PUSH_MAX]))

    t_q1, t_q2 = critic.apply(tc_p, nrho, target, nxi, next_a)
    target_q = r + 0.99 * (1.0 - d) * jnp.minimum(t_q1, t_q2)
    
    def c_loss_fn(p):
        q1, q2 = critic.apply(p, rho, target, xi, a)
        return jnp.mean((q1 - target_q)**2 + (q2 - target_q)**2)
    
    l_c, grads_c = jax.value_and_grad(c_loss_fn)(c_p)
    up_c, opt_c = tx_critic.update(grads_c, opt_c)
    return optax.apply_updates(c_p, up_c), opt_c

@jax.jit
def update_actor_and_targets(a_p, c_p, ta_p, tc_p, opt_a, rho, target, xi):
    def a_loss_fn(p):
        curr_a = actor.apply(p, rho, target, xi)
        q_vals, _ = critic.apply(c_p, rho, target, xi, curr_a)
        return -jnp.mean(q_vals)
    
    l_a, grads_a = jax.value_and_grad(a_loss_fn)(a_p)
    up_a, opt_a = tx_actor.update(grads_a, opt_a)
    a_p = optax.apply_updates(a_p, up_a)
    
    tau = 0.005
    new_ta = jax.tree_util.tree_map(lambda new, old: tau*new + (1-tau)*old, a_p, ta_p)
    new_tc = jax.tree_util.tree_map(lambda new, old: tau*new + (1-tau)*old, c_p, tc_p)
    return a_p, new_ta, new_tc, opt_a

@partial(jax.jit, static_argnames=['add_noise'])
def get_batch_actions(a_p, rho_batch, target_batch, xi_batch, key, add_noise=True):
    actions = actor.apply(a_p, rho_batch, target_batch, xi_batch)
    if add_noise:
        noise_scale = jnp.array([PUSH_MAX, PUSH_MAX]) * 0.1
        noise = jax.random.normal(key, actions.shape) * noise_scale
        actions = jnp.clip(actions + noise, jnp.array([-PUSH_MAX, -PUSH_MAX]), jnp.array([PUSH_MAX, PUSH_MAX]))
    return actions

@jax.jit
def parallel_physics_step(rho_batch, xi_batch, target_batch, actions, key):
    keys = jax.random.split(key, rho_batch.shape[0])
    
    def single_physics_step(rho_s, xi_s, target_s, act_s, k_s):
        traj = dynamics.unroll_controlled(
            rho_init=rho_s, xi_init=xi_s, rho_target=target_s, params=act_s, 
            t_steps=1, dt=dt, buoyancy=BUOYANCY, sigma_push=SIGMA_PUSH, push_max=PUSH_MAX, Nx=Nx, Ny=Ny
        )
        return traj[0][-1], traj[1][-1]
    
    next_rho_batch, next_xi_batch = jax.vmap(single_physics_step)(rho_batch, xi_batch, target_batch, actions, keys)
    
    # Catch Invalid States
    is_invalid = jnp.logical_not(jnp.isfinite(next_rho_batch).all(axis=(1, 2)))
    dones_batch = is_invalid
    
    safe_rho = jnp.where(dones_batch[:, None, None], jnp.zeros_like(next_rho_batch), next_rho_batch)
    safe_xi = jnp.where(dones_batch[:, None, None], xi_batch, next_xi_batch)
    
    # --- REWARDS (Centralized mapping) ---
    mse = jnp.mean(jnp.square(safe_rho - target_batch), axis=(1, 2))
    
    effort = 0.001 * jnp.mean(jnp.sum(jnp.square(actions), axis=-1), axis=1)
    
    margin = 0.02
    x_pen = jnp.maximum(0.0, margin - safe_xi[..., 0])**2 + jnp.maximum(0.0, safe_xi[..., 0] - (1.0 - margin))**2
    y_pen = jnp.maximum(0.0, margin - safe_xi[..., 1])**2 + jnp.maximum(0.0, safe_xi[..., 1] - (1.0 - margin))**2
    bound_pen = 20.0 * jnp.mean(x_pen + y_pen, axis=1)
    
    diff = safe_xi[:, :, None, :] - safe_xi[:, None, :, :]
    dists = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-8)
    mask = jnp.eye(N_AGENTS)[None, :, :]
    coll_pen = 10.0 * jnp.mean(jnp.sum(jnp.maximum(0.0, R_SAFE - (dists + mask * 10.0)) ** 2, axis=2), axis=1)

    rewards_batch = -10.0 * mse - effort - bound_pen - coll_pen
    
    return safe_rho, safe_xi, rewards_batch[..., None], dones_batch[..., None]

# --- 4. THE SCAN-COMPILED TRAINING CHUNK ---
@jax.jit(donate_argnums=(0,))
def train_chunk(carry, step_indices, rho_init_bank, rho_target_bank, xi_init_single):
    def scan_step(carry, step_idx):
        buf, a_p, c_p, ta_p, tc_p, o_a, o_c, rho, target, xi, steps, rng = carry
        rng, act_k, phys_k, res_k, samp_k, net_k = jax.random.split(rng, 6)
        
        def warmup_actions(_):
            return jax.random.uniform(act_k, (NUM_PARALLEL_ENVS, N_AGENTS, 2), 
                                      minval=jnp.array([-PUSH_MAX, -PUSH_MAX], dtype=jnp.float32), 
                                      maxval=jnp.array([PUSH_MAX, PUSH_MAX], dtype=jnp.float32))
        def policy_actions(_):
            return get_batch_actions(a_p, rho, target, xi, act_k, add_noise=True)
            
        actions = jax.lax.cond(step_idx < WARMUP_UPDATES, warmup_actions, policy_actions, None)
        
        nrho, nxi, rew, dones = parallel_physics_step(rho, xi, target, actions, phys_k)
        steps += 1
        truncs = steps >= MAX_ENV_STEPS
        needs_reset = jnp.logical_or(dones.flatten(), truncs)
        
        safe_rew = jnp.where(dones, jnp.array(-100.0, dtype=jnp.float32), rew.astype(jnp.float32))
        
        # Everything goes into buffer natively without PE
        new_buf = add_batch_to_buffer(
            buf, rho.astype(jnp.float32), target.astype(jnp.float32), xi.astype(jnp.float32), 
            actions.astype(jnp.float32), safe_rew, nrho.astype(jnp.float32), 
            nxi.astype(jnp.float32), dones.astype(jnp.float32)
        )
        
        idx_reset = jax.random.randint(res_k, (NUM_PARALLEL_ENVS,), 0, len(rho_init_bank))
        fresh_rho = rho_init_bank[idx_reset]
        fresh_target = rho_target_bank[idx_reset]
        fresh_xi = jnp.tile(xi_init_single, (NUM_PARALLEL_ENVS, 1, 1))
        
        rho_next = jnp.where(needs_reset[:, None, None], fresh_rho, nrho)
        target_next = jnp.where(needs_reset[:, None, None], fresh_target, target)
        xi_next = jnp.where(needs_reset[:, None, None], fresh_xi, nxi)
        steps_next = jnp.where(needs_reset, 0, steps)

        def do_network_updates(net_state):
            c_p, a_p, ta_p, tc_p, o_c, o_a = net_state
            
            brho, btarget, bxi, ba, br, bnrho, bnxi, bd = sample_buffer(new_buf, ENV_BATCH_SIZE, samp_k)
            
            new_c_p, new_o_c = update_critic(c_p, ta_p, tc_p, o_c, brho, btarget, bxi, ba, br, bnrho, bnxi, bd, net_k)
            
            def do_actor_update(_):
                return update_actor_and_targets(a_p, new_c_p, ta_p, tc_p, o_a, brho, btarget, bxi)
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

        new_carry = (new_buf, a_p, c_p, ta_p, tc_p, o_a, o_c, rho_next, target_next, xi_next, steps_next, rng)
        return new_carry, None

    return jax.lax.scan(scan_step, carry, step_indices)

# --- 5. FAST EVALUATION ---
@partial(jax.jit, static_argnames=['max_steps'])
def fast_eval_episode(actor_params, init_rho, init_xi, target_rho, max_steps, key):
    def step_fn(state_tuple, _):
        rho_curr, xi_curr, k = state_tuple
        k, subk = jax.random.split(k)
        
        act = get_batch_actions(actor_params, rho_curr[None, ...], target_rho[None, ...], xi_curr[None, ...], None, add_noise=False)
        act_flat = act.squeeze(0)
        
        traj = dynamics.unroll_controlled(
            rho_init=rho_curr, xi_init=xi_curr, rho_target=target_rho, params=act_flat, 
            t_steps=1, dt=dt, buoyancy=BUOYANCY, sigma_push=SIGMA_PUSH, push_max=PUSH_MAX, Nx=Nx, Ny=Ny
        )
        next_rho, next_xi = traj[0][-1], traj[1][-1]
        
        mse = jnp.mean((next_rho - target_rho)**2)
        crashed = jnp.isnan(next_rho).any() | jnp.isinf(next_rho).any()
        
        return (next_rho, next_xi, k), (mse, crashed)

    _, (mses, crashes) = jax.lax.scan(step_fn, (init_rho, init_xi, key), None, length=max_steps)
    return jnp.mean(mses), jnp.any(crashes)

# --- Vectorized Training Loop ---
n_side = int(np.sqrt(N_AGENTS))
X, Y = jnp.meshgrid(
    jnp.linspace(0.15, 0.85, n_side),
    jnp.linspace(0.15, 1.0, n_side)
)
xi_init_single = jnp.stack([X.flatten(), Y.flatten()], axis=-1).astype(np.float32)

key, subkey = jax.random.split(key)
idx = jax.random.randint(subkey, (NUM_PARALLEL_ENVS,), 0, len(rho_init_bank))
rho_batch = rho_init_bank[idx]
target_batch = rho_target_bank[idx]
xi_batch = jnp.tile(xi_init_single, (NUM_PARALLEL_ENVS, 1, 1))
env_step_counts = jnp.zeros(NUM_PARALLEL_ENVS, dtype=jnp.int32)

carry = (
    buffer, actor_params, critic_params, target_actor_params, target_critic_params,
    opt_actor, opt_critic, rho_batch, target_batch, xi_batch, env_step_counts, key
)

print(f"Starting Massively Parallel MARL Training (Chunked & JITed NS2D Density Control)...")
start_time = time.time()

num_chunks = TOTAL_UPDATES // EVAL_INT

for chunk_idx in trange(num_chunks):
    start_step = chunk_idx * EVAL_INT
    step_indices = jnp.arange(start_step, start_step + EVAL_INT)
    
    carry, _ = train_chunk(carry, step_indices, rho_init_bank, rho_target_bank, xi_init_single)
    
    current_actor_params = carry[1] 
    
    eval_rho = rho_init_bank[0] 
    eval_target = rho_target_bank[0]
    key, eval_key = jax.random.split(key)
    
    eval_e, crashed = fast_eval_episode(current_actor_params, eval_rho, xi_init_single, eval_target, MAX_ENV_STEPS, eval_key)
    
    current_total_step = start_step + EVAL_INT
    episode_num = current_total_step // MAX_ENV_STEPS
    
    if crashed:
        print(f"\nUpdate {current_total_step:06d} | Episode {episode_num} | Eval Tracking MSE: [CRASHED] | Time: {time.time()-start_time:.1f}s")
    else:
        print(f"\nUpdate {current_total_step:06d} | Episode {episode_num} | Eval Tracking MSE: {eval_e:.6f} | Time: {time.time()-start_time:.1f}s")

final_actor_params = carry[1]
with open('models/rl_ns2d_centralized_params.msgpack', 'wb') as f:
    f.write(flax.serialization.to_bytes({'actor': final_actor_params}))
print(f"Training finished in {time.time()-start_time:.1f}s. Weights saved.")