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
from tqdm import trange

# Add project root
script_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(script_dir))

from env_heat2d import Heat2DHypeMARLEnv, extract_patches_heat2d_jit
from utils_hypemarl import get_sinusoidal_encoding
from examples.heat2D.decentralized.data_utils import get_training_data
from examples.heat2D.decentralized.dynamics_dual import PDEDynamics 

from models_marl import MARLActor2D, MATD3Critic2D, U_MAX, V_MAX

# --- Configurations ---
N_AGENTS = 16 
L_DOMAIN = 1.0
N_GRID = 32

ENV_BATCH_SIZE = 256 
EVAL_INT = 500
POLICY_DELAY = 2 
MAX_ENV_STEPS = 100 

NUM_PARALLEL_ENVS = 128
TOTAL_UPDATES = 100000 
WARMUP_UPDATES = 500

# --- Positional Encoding (128 Dim Total) ---
PE_D = 64  # 64 features per axis -> 128 total PE features

def get_2d_sinusoidal_encoding(p_2d, d=PE_D, n=1000.0):
    """Combines two 1D positional encodings for 2D coordinates."""
    pe_x = get_sinusoidal_encoding(p_2d[:, 0], d=d, n=n)
    pe_y = get_sinusoidal_encoding(p_2d[:, 1], d=d, n=n)
    return jnp.concatenate([pe_x, pe_y], axis=-1)

# --- Initialization ---
key = jax.random.PRNGKey(42)

def direct_control_policy(action_params, u_obs, u_target, xi_fixed):
    u = action_params[:, 0]
    v = action_params[:, 1:3]
    return u, v

dynamics = PDEDynamics(policy_apply_fn=direct_control_policy)
env = Heat2DHypeMARLEnv(dynamics, n_agents=N_AGENTS, N_grid=N_GRID, L=L_DOMAIN, max_steps=MAX_ENV_STEPS)

local_y_dim = env.local_y_dim
n_mu = env.n_mu
pe_dim = PE_D * 2 # 128

stored_obs_dim = local_y_dim + n_mu 
total_input_dim = stored_obs_dim + pe_dim

mu_jax = jnp.array(env.mu)
window_size = env.window_size
resized_dim = env.resized_dim

actor = MARLActor2D()
critic = MATD3Critic2D(n_agents=N_AGENTS)

key, *subkeys = jax.random.split(key, 4)

# FIX: Add the N_AGENTS dimension to dummy inputs
dummy_input = jnp.zeros((ENV_BATCH_SIZE, N_AGENTS, total_input_dim))
dummy_act = jnp.zeros((ENV_BATCH_SIZE, N_AGENTS, 3)) 

# Critic dummies must be joint (flattened across agents)
dummy_joint_input = dummy_input.reshape((ENV_BATCH_SIZE, -1))
dummy_joint_act = dummy_act.reshape((ENV_BATCH_SIZE, -1))

actor_params = actor.init(subkeys[0], dummy_input)
critic_params = critic.init(subkeys[1], dummy_joint_input, dummy_joint_act)

target_actor_params = actor_params
target_critic_params = critic_params

tx_actor = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-4)) 
tx_critic = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(5e-4))
opt_actor = tx_actor.init(actor_params)
opt_critic = tx_critic.init(critic_params)

# --- 1. DYNAMIC REPLAY BUFFER ---
@struct.dataclass
class DeviceReplayBuffer:
    s: jnp.ndarray
    xi: jnp.ndarray
    a: jnp.ndarray
    r: jnp.ndarray
    ns: jnp.ndarray
    nxi: jnp.ndarray
    d: jnp.ndarray
    ptr: jnp.int32
    size: jnp.int32
    max_size: int = struct.field(pytree_node=False)

    @classmethod
    def create(cls, max_size, s_dim, a_dim):
        return cls(
            s=jnp.zeros((max_size, N_AGENTS, s_dim), dtype=jnp.float32),
            xi=jnp.zeros((max_size, N_AGENTS, 2), dtype=jnp.float32), 
            a=jnp.zeros((max_size, N_AGENTS, a_dim), dtype=jnp.float32),
            r=jnp.zeros((max_size, N_AGENTS, 1), dtype=jnp.float32),
            ns=jnp.zeros((max_size, N_AGENTS, s_dim), dtype=jnp.float32),
            nxi=jnp.zeros((max_size, N_AGENTS, 2), dtype=jnp.float32),
            d=jnp.zeros((max_size, N_AGENTS, 1), dtype=jnp.float32),
            ptr=jnp.int32(0),
            size=jnp.int32(0),
            max_size=max_size
        )

@jax.jit
def add_batch_to_buffer(buffer, s_b, xi_b, a_b, r_b, ns_b, nxi_b, d_b):
    batch_size = s_b.shape[0]
    indices = (buffer.ptr + jnp.arange(batch_size)) % buffer.max_size
    
    new_s = buffer.s.at[indices].set(s_b)
    new_xi = buffer.xi.at[indices].set(xi_b)
    new_a = buffer.a.at[indices].set(a_b)
    new_r = buffer.r.at[indices].set(r_b)
    new_ns = buffer.ns.at[indices].set(ns_b)
    new_nxi = buffer.nxi.at[indices].set(nxi_b)
    new_d = buffer.d.at[indices].set(d_b)
    
    new_ptr = (buffer.ptr + batch_size) % buffer.max_size
    new_size = jnp.minimum(buffer.size + batch_size, buffer.max_size)
    
    return buffer.replace(s=new_s, xi=new_xi, a=new_a, r=new_r, ns=new_ns, nxi=new_nxi, d=new_d, ptr=new_ptr, size=new_size)

@partial(jax.jit, static_argnames=['batch_size'])
def sample_buffer(buffer, batch_size, key):
    valid_range = jnp.minimum(buffer.size, buffer.max_size)
    indices = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=valid_range)
    return buffer.s[indices], buffer.xi[indices], buffer.a[indices], buffer.r[indices], buffer.ns[indices], buffer.nxi[indices], buffer.d[indices]

buffer = DeviceReplayBuffer.create(125_000, stored_obs_dim, 3) 

# --- 2. JAX OBSERVATION BUILDER ---
@jax.jit
def build_marl_obs_batch(z_batch, target_batch, xi_batch):
    def single_env_obs(state, target, xi):
        y_local = extract_patches_heat2d_jit(state, target, xi, window_size, resized_dim)
        mu_broadcast = jnp.tile(mu_jax, (N_AGENTS, 1))
        return jnp.concatenate([y_local, mu_broadcast], axis=-1)
    return jax.vmap(single_env_obs)(z_batch, target_batch, xi_batch)

# --- 3. JIT TRAINING & ROLLOUT ---
@jax.jit
def update_critic(c_p, ta_p, tc_p, opt_c, x, u, r, nx, d, key):
    batch_size = x.shape[0]
    key, noise_key = jax.random.split(key)
    
    noise_scale = jnp.array([U_MAX, V_MAX, V_MAX]) * 0.1
    noise = jnp.clip(jax.random.normal(noise_key, u.shape) * noise_scale, -0.5 * noise_scale, 0.5 * noise_scale)
    
    raw_next_u = actor.apply(ta_p, nx) + noise
    next_u = jnp.clip(raw_next_u, jnp.array([-U_MAX, -V_MAX, -V_MAX]), jnp.array([U_MAX, V_MAX, V_MAX]))
    
    # Flatten across agents to form Joint State/Action for MATD3
    joint_x = x.reshape(batch_size, -1)
    joint_u = u.reshape(batch_size, -1)
    joint_nx = nx.reshape(batch_size, -1)
    joint_next_u = next_u.reshape(batch_size, -1)

    t_q1, t_q2 = critic.apply(tc_p, joint_nx, joint_next_u)
    target_q = r + 0.99 * (1.0 - d) * jnp.minimum(t_q1, t_q2)
    
    def c_loss_fn(p):
        q1, q2 = critic.apply(p, joint_x, joint_u)
        return jnp.mean((q1 - target_q)**2 + (q2 - target_q)**2)
    
    l_c, grads_c = jax.value_and_grad(c_loss_fn)(c_p)
    up_c, opt_c = tx_critic.update(grads_c, opt_c)
    return optax.apply_updates(c_p, up_c), opt_c

@jax.jit
def update_actor_and_targets(a_p, c_p, ta_p, tc_p, opt_a, x):
    batch_size = x.shape[0]
    joint_x = x.reshape(batch_size, -1)

    def a_loss_fn(p):
        curr_u = actor.apply(p, x)
        joint_curr_u = curr_u.reshape(batch_size, -1)
        q_vals, _ = critic.apply(c_p, joint_x, joint_curr_u)
        return -jnp.mean(q_vals)
    
    l_a, grads_a = jax.value_and_grad(a_loss_fn)(a_p)
    up_a, opt_a = tx_actor.update(grads_a, opt_a)
    a_p = optax.apply_updates(a_p, up_a)
    
    tau = 0.005
    new_ta = jax.tree_util.tree_map(lambda new, old: tau*new + (1-tau)*old, a_p, ta_p)
    new_tc = jax.tree_util.tree_map(lambda new, old: tau*new + (1-tau)*old, c_p, tc_p)
    return a_p, new_ta, new_tc, opt_a

@partial(jax.jit, static_argnames=['add_noise'])
def get_batch_actions(a_p, obs_batch_no_pe, xi_batch, key, add_noise=True):
    pe_batch = jax.vmap(lambda xi: get_2d_sinusoidal_encoding(xi))(xi_batch)
    full_obs = jnp.concatenate([obs_batch_no_pe, pe_batch], axis=-1)
    
    actions = actor.apply(a_p, full_obs)
    
    if add_noise:
        noise_scale = jnp.array([U_MAX, V_MAX, V_MAX]) * 0.1
        noise = jax.random.normal(key, actions.shape) * noise_scale
        actions = jnp.clip(actions + noise, jnp.array([-U_MAX, -V_MAX, -V_MAX]), jnp.array([U_MAX, V_MAX, V_MAX]))
    return actions

@jax.jit
def parallel_marl_physics_step(z_batch, xi_batch, target_batch, actions, prev_v_batch, key):
    keys = jax.random.split(key, z_batch.shape[0])
    
    def single_physics_step(z_s, xi_s, target_s, act_s, k_s):
        traj = dynamics.unroll_controlled(
            z_init=z_s, xi_init=xi_s, z_target=target_s, params=act_s, 
            t_steps=1,
        )
        return traj[0][-1], traj[1][-1]
    
    next_z_batch, next_xi_batch = jax.vmap(single_physics_step)(z_batch, xi_batch, target_batch, actions, keys)
    
    is_invalid = jnp.logical_not(jnp.isfinite(next_z_batch).all(axis=(1, 2)))
    dones_batch = is_invalid
    
    safe_z = jnp.where(dones_batch[:, None, None], jnp.zeros_like(next_z_batch), next_z_batch)
    safe_xi = jnp.where(dones_batch[:, None, None], xi_batch, next_xi_batch)
    
    next_obs_batch_no_pe = build_marl_obs_batch(safe_z, target_batch, safe_xi)
    
    u_batch = actions[..., 0]
    v_batch = actions[..., 1:3]
    
    # --- COMBINED LOCAL + GLOBAL TRACKING REWARD ---
    x_idx = jnp.clip((safe_xi[..., 0] * (N_GRID - 1)).astype(jnp.int32), 0, N_GRID - 1)
    y_idx = jnp.clip((safe_xi[..., 1] * (N_GRID - 1)).astype(jnp.int32), 0, N_GRID - 1)
    batch_indices = jnp.arange(safe_z.shape[0])[:, None]
    
    local_z = safe_z[batch_indices, x_idx, y_idx]
    local_target = target_batch[batch_indices, x_idx, y_idx]
    local_mse = jnp.square(local_z - local_target)
    
    global_mse = jnp.mean(jnp.square(safe_z - target_batch), axis=(1, 2))[:, None]
    
    r_track_combined = -10.0 * local_mse - 10.0 * global_mse 
    # -----------------------------------------------
    
    # 2. Effort Penalty
    r_effort = -0.001 * (jnp.square(u_batch) + 0.1 * jnp.sum(jnp.square(v_batch), axis=-1))
    
    # 3. Boundary Penalty 
    margin = 0.02
    x_pen = jnp.maximum(0.0, margin - safe_xi[..., 0])**2 + jnp.maximum(0.0, safe_xi[..., 0] - (1.0 - margin))**2
    y_pen = jnp.maximum(0.0, margin - safe_xi[..., 1])**2 + jnp.maximum(0.0, safe_xi[..., 1] - (1.0 - margin))**2
    r_bound = -100.0 * (x_pen + y_pen)
    
    # 4. Agent-Agent Collision Penalty 
    R_safe = 0.08
    diff = safe_xi[:, :, None, :] - safe_xi[:, None, :, :]
    dists = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-8)
    mask = jnp.eye(N_AGENTS)[None, :, :]
    r_coll = -20.0 * jnp.sum(jnp.maximum(0.0, R_safe - (dists + mask * 10.0)) ** 2, axis=2)

    # 5. Acceleration Penalty
    r_accel = -0.1 * jnp.sum(jnp.square(v_batch - prev_v_batch), axis=-1)
    
    # Combine into a unified reward signal (without obstacles)
    rewards_batch = r_track_combined + r_effort + r_bound + r_coll + r_accel
    rewards_batch = rewards_batch[..., None]
    
    return safe_z, safe_xi, next_obs_batch_no_pe, rewards_batch, dones_batch, v_batch

# --- 4. THE SCAN-COMPILED TRAINING CHUNK ---
@jax.jit
def train_chunk(carry, step_indices, z_init_bank, z_target_bank, xi_init_single):
    def scan_step(carry, step_idx):
        buf, a_p, c_p, ta_p, tc_p, o_a, o_c, z, target, xi, pv, obs, steps, rng = carry
        rng, act_k, phys_k, res_k, samp_k, net_k = jax.random.split(rng, 6)
        
        # 1. Action Selection (Warmup vs Policy)
        def warmup_actions(_):
            return jax.random.uniform(act_k, (NUM_PARALLEL_ENVS, N_AGENTS, 3), 
                                      minval=jnp.array([-U_MAX, -V_MAX, -V_MAX]), 
                                      maxval=jnp.array([U_MAX, V_MAX, V_MAX]))
        def policy_actions(_):
            return get_batch_actions(a_p, obs, xi, act_k, add_noise=True)
            
        actions = jax.lax.cond(step_idx < WARMUP_UPDATES, warmup_actions, policy_actions, None)
        
        # 2. Physics Step
        nz, nxi, nobs, rew, dones, npv = parallel_marl_physics_step(z, xi, target, actions, pv, phys_k)
        steps += 1
        truncs = steps >= MAX_ENV_STEPS
        needs_reset = jnp.logical_or(dones.flatten(), truncs)
        
        # 3. Update Buffer
        safe_rew = jnp.where(dones[:, None, None], -100.0, rew)
        dones_exp = jnp.tile(dones[:, None, None], (1, N_AGENTS, 1))
        new_buf = add_batch_to_buffer(buf, obs, xi, actions, safe_rew, nobs, nxi, dones_exp)
        
        # 4. Handle Resets
        idx_reset = jax.random.randint(res_k, (NUM_PARALLEL_ENVS,), 0, len(z_init_bank))
        fresh_z = z_init_bank[idx_reset]
        fresh_target = z_target_bank[idx_reset]
        fresh_xi = jnp.tile(xi_init_single, (NUM_PARALLEL_ENVS, 1, 1))
        fresh_pv = jnp.zeros((NUM_PARALLEL_ENVS, N_AGENTS, 2))
        
        z_next = jnp.where(needs_reset[:, None, None], fresh_z, nz)
        target_next = jnp.where(needs_reset[:, None, None], fresh_target, target)
        xi_next = jnp.where(needs_reset[:, None, None], fresh_xi, nxi)
        pv_next = jnp.where(needs_reset[:, None, None], fresh_pv, npv)
        obs_next = build_marl_obs_batch(z_next, target_next, xi_next)
        steps_next = jnp.where(needs_reset, 0, steps)

        # 5. Network Updates (Conditional on Buffer Size)
        def do_network_updates(net_state):
            c_p, a_p, ta_p, tc_p, o_c, o_a = net_state
            
            bx, bxi, bu, br, bnx, bnxi, bd = sample_buffer(new_buf, ENV_BATCH_SIZE, samp_k)
            bpe = jax.vmap(lambda x_i: get_2d_sinusoidal_encoding(x_i))(bxi)
            bnpe = jax.vmap(lambda x_i: get_2d_sinusoidal_encoding(x_i))(bnxi)
            
            bx_f = jnp.concatenate([bx, bpe], axis=-1)
            bnx_f = jnp.concatenate([bnx, bnpe], axis=-1)
            
            # Critic Update
            new_c_p, new_o_c = update_critic(c_p, ta_p, tc_p, o_c, bx_f, bu, br, bnx_f, bd, net_k)
            
            # Policy Delayed Actor Update
            def do_actor_update(_):
                return update_actor_and_targets(a_p, new_c_p, ta_p, tc_p, o_a, bx_f)
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

        new_carry = (new_buf, a_p, c_p, ta_p, tc_p, o_a, o_c, z_next, target_next, xi_next, pv_next, obs_next, steps_next, rng)
        return new_carry, None

    return jax.lax.scan(scan_step, carry, step_indices)

# --- 5. FAST EVALUATION ---
@partial(jax.jit, static_argnames=['max_steps'])
def fast_eval_episode(actor_params, init_z, init_xi, target_z, max_steps, key):
    def step_fn(state_tuple, _):
        z_curr, xi_curr, k = state_tuple
        k, subk = jax.random.split(k)
        
        obs_no_pe = build_marl_obs_batch(z_curr[None, ...], target_z[None, ...], xi_curr[None, ...]) 
        act = get_batch_actions(actor_params, obs_no_pe, xi_curr[None, ...], None, add_noise=False)
        act_flat = act.squeeze(0)
        
        traj = dynamics.unroll_controlled(
            z_init=z_curr, xi_init=xi_curr, z_target=target_z, params=act_flat, 
            t_steps=1,
        )
        next_z, next_xi = traj[0][-1], traj[1][-1]
        
        mse = jnp.mean((next_z - target_z)**2)
        crashed = jnp.isnan(next_z).any() | jnp.isinf(next_z).any()
        
        return (next_z, next_xi, k), (mse, crashed)

    _, (mses, crashes) = jax.lax.scan(step_fn, (init_z, init_xi, key), None, length=max_steps)
    return jnp.mean(mses), jnp.any(crashes)

# --- Vectorized Training Loop ---
print("Loading 2D starting state & target banks from dataset...")
z_init_all, z_target_all, _ = get_training_data(n_samples=5000, n_grid=N_GRID, dataset_dir='../../data')
z_init_bank = jnp.array(z_init_all)
z_target_bank = jnp.array(z_target_all)

# 4x4 Agent Template
n_side = int(np.sqrt(N_AGENTS))
pos_1d = np.linspace(0.2, 0.8, n_side)
X, Y = np.meshgrid(pos_1d, pos_1d)
xi_init_single = jnp.stack([X.flatten(), Y.flatten()], axis=-1).astype(np.float32)

key, subkey = jax.random.split(key)
idx = jax.random.randint(subkey, (NUM_PARALLEL_ENVS,), 0, len(z_init_bank))
z_batch = z_init_bank[idx]
target_batch = z_target_bank[idx]
xi_batch = jnp.tile(xi_init_single, (NUM_PARALLEL_ENVS, 1, 1))
prev_v_batch = jnp.zeros((NUM_PARALLEL_ENVS, N_AGENTS, 2))

obs_batch = build_marl_obs_batch(z_batch, target_batch, xi_batch)
env_step_counts = jnp.zeros(NUM_PARALLEL_ENVS)

# Pack everything into the initial carry state
carry = (
    buffer, actor_params, critic_params, target_actor_params, target_critic_params,
    opt_actor, opt_critic, z_batch, target_batch, xi_batch, prev_v_batch, obs_batch, env_step_counts, key
)

print(f"Starting Massively Parallel MARL Training (Chunked & JITed 2D Heat Equation)...")
start_time = time.time()

num_chunks = TOTAL_UPDATES // EVAL_INT

for chunk_idx in trange(num_chunks):
    start_step = chunk_idx * EVAL_INT
    step_indices = jnp.arange(start_step, start_step + EVAL_INT)
    
    # Run the compiled chunk
    carry, _ = train_chunk(carry, step_indices, z_init_bank, z_target_bank, xi_init_single)
    
    # Unpack the current actor for evaluation
    current_actor_params = carry[1] 
    
    # Evaluation Logic
    eval_z = z_init_bank[0] 
    eval_target = z_target_bank[0]
    key, eval_key = jax.random.split(key)
    
    eval_e, crashed = fast_eval_episode(current_actor_params, eval_z, xi_init_single, eval_target, MAX_ENV_STEPS, eval_key)
    
    current_total_step = start_step + EVAL_INT
    episode_num = current_total_step // MAX_ENV_STEPS
    
    if crashed:
        print(f"\nUpdate {current_total_step:06d} | Episode {episode_num} | Eval Tracking MSE: [CRASHED] | Time: {time.time()-start_time:.1f}s")
    else:
        print(f"\nUpdate {current_total_step:06d} | Episode {episode_num} | Eval Tracking MSE: {eval_e:.6f} | Time: {time.time()-start_time:.1f}s")

# Extract final weights and save
final_actor_params = carry[1]
with open('models/marl_heat2d_params.msgpack', 'wb') as f:
    f.write(flax.serialization.to_bytes({'actor': final_actor_params}))
print(f"Training finished in {time.time()-start_time:.1f}s. Weights saved.")