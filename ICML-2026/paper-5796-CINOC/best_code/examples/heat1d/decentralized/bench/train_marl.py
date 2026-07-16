import sys
import time
from functools import partial
from pathlib import Path

import flax.linen as nn
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from tqdm import trange

# --- Project Path Setup ---
script_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(script_dir))

from env_he import HeatHypeMARLEnv 
from utils_hypemarl import get_sinusoidal_encoding
from examples.heat1d.decentralized.data_utils import generate_grf
from examples.heat1d.decentralized.dynamics_dual import PDEDynamics
from models_marl import MARLActor, MARLCritic, U_MAX, V_MAX

# --- Configurations ---
N_AGENTS = 8 
L_DOMAIN = 1.0
N_GRID = 100

ENV_BATCH_SIZE = 256
EVAL_INT = 500
POLICY_DELAY = 2
MAX_ENV_STEPS = 300

# Vectorization Configs
NUM_PARALLEL_ENVS = 256
TOTAL_UPDATES = 100000
WARMUP_UPDATES = 500
PE_DIM = 128

# --- Initialization ---
key = jax.random.PRNGKey(42)

def direct_control_policy(action_params, u_obs, u_target, xi_fixed):
    u = action_params[:, 0]
    v = action_params[:, 1]
    return u, v

dynamics = PDEDynamics(policy_apply_fn=direct_control_policy)
env = HeatHypeMARLEnv(
    dynamics, n_agents=N_AGENTS, N_grid=N_GRID, L=L_DOMAIN, max_steps=MAX_ENV_STEPS
)

local_y_dim = env.local_y_dim
n_mu = env.n_mu
stored_obs_dim = local_y_dim + n_mu
total_input_dim = stored_obs_dim + PE_DIM

mu_jax = jnp.array(env.mu)
window_size = env.window_size

actor = MARLActor()
critic = MARLCritic()

key, *subkeys = jax.random.split(key, 4)

dummy_input = jnp.zeros((ENV_BATCH_SIZE, total_input_dim))
dummy_joint_input = jnp.zeros((ENV_BATCH_SIZE, N_AGENTS * total_input_dim))
dummy_joint_act = jnp.zeros((ENV_BATCH_SIZE, N_AGENTS * 2))

actor_params = actor.init(subkeys[0], dummy_input)
critic_params = critic.init(subkeys[1], dummy_joint_input, dummy_joint_act)

target_actor_params = actor_params
target_critic_params = critic_params

# --- NEW: Linear Learning Rate Schedules ---
actor_lr_schedule = optax.linear_schedule(init_value=1e-4, end_value=0.0, transition_steps=TOTAL_UPDATES)
critic_lr_schedule = optax.linear_schedule(init_value=5e-4, end_value=0.0, transition_steps=TOTAL_UPDATES)

tx_actor = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(actor_lr_schedule))
tx_critic = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(critic_lr_schedule))
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
            xi=jnp.zeros((max_size, N_AGENTS), dtype=jnp.float32),
            a=jnp.zeros((max_size, N_AGENTS, a_dim), dtype=jnp.float32),
            r=jnp.zeros((max_size, N_AGENTS, 1), dtype=jnp.float32),
            ns=jnp.zeros((max_size, N_AGENTS, s_dim), dtype=jnp.float32),
            nxi=jnp.zeros((max_size, N_AGENTS), dtype=jnp.float32),
            d=jnp.zeros((max_size, N_AGENTS, 1), dtype=jnp.float32),
            ptr=jnp.int32(0),
            size=jnp.int32(0),
            max_size=max_size,
        )


@jax.jit
def add_batch_to_buffer(buffer, s_b, xi_b, a_b, r_b, ns_b, nxi_b, d_b):
    batch_size = s_b.shape[0]
    indices = (buffer.ptr + jnp.arange(batch_size)) % buffer.max_size
    new_ptr = (buffer.ptr + batch_size) % buffer.max_size
    new_size = jnp.minimum(buffer.size + batch_size, buffer.max_size)

    return buffer.replace(
        s=buffer.s.at[indices].set(s_b), xi=buffer.xi.at[indices].set(xi_b),
        a=buffer.a.at[indices].set(a_b), r=buffer.r.at[indices].set(r_b),
        ns=buffer.ns.at[indices].set(ns_b), nxi=buffer.nxi.at[indices].set(nxi_b),
        d=buffer.d.at[indices].set(d_b), ptr=new_ptr, size=new_size
    )


@partial(jax.jit, static_argnames=['batch_size'])
def sample_buffer(buffer, batch_size, key):
    valid_range = jnp.minimum(buffer.size, buffer.max_size)
    indices = jax.random.randint(key, (batch_size,), 0, valid_range)
    return (
        buffer.s[indices], buffer.xi[indices], buffer.a[indices], 
        buffer.r[indices], buffer.ns[indices], buffer.nxi[indices], buffer.d[indices]
    )


# --- 2. OBSERVATION BUILDER ---
@partial(jax.jit, static_argnames=['window_size'])
def extract_patches_jax(full_state, target_st, xi_n, window_size):
    error = full_state - target_st
    error_grad = jnp.gradient(error)
    n_pde = full_state.shape[0]
    half_window = window_size // 2
    padded_error = jnp.pad(error, (half_window, half_window), mode='constant')
    padded_grad = jnp.pad(error_grad, (half_window, half_window), mode='constant')

    def get_local_obs(xi):
        center_idx = jax.lax.stop_gradient((xi * (n_pde - 1)).astype(int)) + half_window
        start = center_idx - half_window
        p_err = jax.lax.dynamic_slice(padded_error, (start,), (window_size,))
        p_grad = jax.lax.dynamic_slice(padded_grad, (start,), (window_size,))
        p_err = jax.image.resize(p_err, (20,), method='bilinear')
        p_grad = jax.image.resize(p_grad, (20,), method='bilinear')
        return jnp.concatenate([p_err, p_grad])

    return jax.vmap(get_local_obs)(xi_n)


@jax.jit
def build_marl_obs_batch(z_batch, target_batch, xi_batch):
    def single_env_obs(state, target, xi):
        y_local = extract_patches_jax(state, target, xi, window_size)
        mu_broadcast = jnp.tile(mu_jax, (N_AGENTS, 1))
        return jnp.concatenate([y_local, mu_broadcast], axis=-1)
    return jax.vmap(single_env_obs)(z_batch, target_batch, xi_batch)


# --- 3. MATD3 TRAINING COMPONENTS ---
@jax.jit
def update_critic(c_p, ta_p, tc_p, opt_c, x, u, r, nx, d, key):
    batch_size = x.shape[0]
    key, noise_key = jax.random.split(key)
    noise_scale = jnp.array([U_MAX, V_MAX]) * 0.1
    noise = jnp.clip(jax.random.normal(noise_key, u.shape) * noise_scale, -0.5 * noise_scale, 0.5 * noise_scale)
    
    actor_apply_fn = jax.vmap(jax.vmap(actor.apply, in_axes=(None, 0)), in_axes=(None, 0))
    raw_next_u = actor_apply_fn(ta_p, nx) + noise
    next_u = jnp.clip(raw_next_u, jnp.array([-U_MAX, -V_MAX]), jnp.array([U_MAX, V_MAX]))
    
    joint_x, joint_u = x.reshape(batch_size, -1), u.reshape(batch_size, -1)
    joint_nx, joint_next_u = nx.reshape(batch_size, -1), next_u.reshape(batch_size, -1)
    
    joint_r = jnp.mean(r, axis=1)
    joint_d = jnp.max(d, axis=1)  
    
    t_q1, t_q2 = critic.apply(tc_p, joint_nx, joint_next_u)
    target_q = joint_r + 0.99 * (1.0 - joint_d) * jnp.minimum(t_q1, t_q2)
    
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
        actor_apply_fn = jax.vmap(jax.vmap(actor.apply, in_axes=(None, 0)), in_axes=(None, 0))
        u_pred = actor_apply_fn(p, x)
        q1, _ = critic.apply(c_p, joint_x, u_pred.reshape(batch_size, -1))
        return -jnp.mean(q1)
    
    l_a, grads_a = jax.value_and_grad(a_loss_fn)(a_p)
    up_a, opt_a = tx_actor.update(grads_a, opt_a)
    a_p = optax.apply_updates(a_p, up_a)
    
    tau = 0.005
    new_ta = jax.tree_util.tree_map(lambda n, o: tau*n + (1-tau)*o, a_p, ta_p)
    new_tc = jax.tree_util.tree_map(lambda n, o: tau*n + (1-tau)*o, c_p, tc_p)
    return a_p, new_ta, new_tc, opt_a


@partial(jax.jit, static_argnames=['add_noise'])
def get_batch_actions(a_p, obs_batch_no_pe, xi_batch, key, add_noise=True):
    pe_batch = jax.vmap(lambda xi: get_sinusoidal_encoding(xi, d=PE_DIM))(xi_batch)
    full_obs = jnp.concatenate([obs_batch_no_pe, pe_batch], axis=-1)
    actions = jax.vmap(jax.vmap(actor.apply, in_axes=(None, 0)), in_axes=(None, 0))(a_p, full_obs)
    if add_noise:
        noise = jax.random.normal(key, actions.shape) * (jnp.array([U_MAX, V_MAX]) * 0.1)
        actions = jnp.clip(actions + noise, jnp.array([-U_MAX, -V_MAX]), jnp.array([U_MAX, V_MAX]))
    return actions


@jax.jit
def parallel_marl_physics_step(z_batch, xi_batch, target_batch, actions, key):
    keys = jax.random.split(key, z_batch.shape[0])
    def single_physics_step(z_s, xi_s, target_s, act_s, k_s):
        traj = dynamics.unroll_controlled(
            z_init=z_s, xi_init=xi_s, z_target=target_s, params=act_s, t_steps=1
        )
        return traj[0][-1], traj[1][-1]
    
    nz_b, nxi_b = jax.vmap(single_physics_step)(z_batch, xi_batch, target_batch, actions, keys)
    dones_b = jnp.logical_not(jnp.isfinite(nz_b).all(axis=-1, keepdims=True))
    safe_z, safe_xi = jnp.where(dones_b, jnp.zeros_like(nz_b), nz_b), jnp.where(dones_b, xi_batch, nxi_b)
    
    # --- ENHANCED: Blended Reward Logic ---
    agent_idx = jnp.clip((safe_xi * (N_GRID - 1)).astype(jnp.int32), 0, N_GRID - 1)
    batch_idx = jnp.arange(safe_z.shape[0])[:, None]
    local_mse = jnp.square(safe_z[batch_idx, agent_idx] - target_batch[batch_idx, agent_idx])
    global_mse = jnp.mean(jnp.square(safe_z - target_batch), axis=-1, keepdims=True)
    r_track = -10.0 * local_mse - 10.0 * global_mse
    # --------------------------------------

    r_effort = -0.001 * (jnp.square(actions[..., 0]) + 0.1 * jnp.square(actions[..., 1]))
    r_bound = -100.0 * (jnp.maximum(0.0, 0.02 - safe_xi)**2 + jnp.maximum(0.0, safe_xi - 0.98)**2)
    dists = jnp.abs(safe_xi[:, :, None] - safe_xi[:, None, :])
    r_coll = -1.0 * jnp.sum(jnp.maximum(0.0, 0.02 - (dists + jnp.eye(N_AGENTS)[None, :, :] * 1.0)) ** 2, axis=2)
    
    rewards_batch = (r_track + r_effort + r_bound + r_coll)[..., None] * 0.05 
    return safe_z, safe_xi, build_marl_obs_batch(safe_z, target_batch, safe_xi), rewards_batch, dones_b


# --- 4. FAST EVALUATION ---
@partial(jax.jit, static_argnames=['max_steps'])
def fast_eval_episode(actor_params, init_z, init_xi, target_z, max_steps, key):
    def step_fn(state_tuple, _):
        z_curr, xi_curr, k = state_tuple
        k, subk = jax.random.split(k)
        obs = build_marl_obs_batch(z_curr[None, ...], target_z[None, ...], xi_curr[None, ...])
        act = get_batch_actions(actor_params, obs, xi_curr[None, ...], None, add_noise=False)
        traj = dynamics.unroll_controlled(
            z_init=z_curr, xi_init=xi_curr, z_target=target_z, params=act.squeeze(0), t_steps=1
        )
        mse = jnp.mean((traj[0][-1] - target_z)**2)
        return (traj[0][-1], traj[1][-1], k), (mse, jnp.isnan(traj[0][-1]).any())

    _, (mses, crashes) = jax.lax.scan(step_fn, (init_z, init_xi, key), None, length=max_steps)
    return jnp.mean(mses), jnp.any(crashes)


# --- 5. SCAN-COMPILED TRAINING CHUNK ---
@jax.jit
def train_chunk(carry, step_indices, z_init_bank, z_target_bank):
    def scan_step(carry, step_idx):
        buf, a_p, c_p, ta_p, tc_p, o_a, o_c, z, target, xi, obs, steps, rng = carry
        rng, act_k, phys_k, res_k, samp_k, net_k = jax.random.split(rng, 6)
        
        actions = jax.lax.cond(
            step_idx < WARMUP_UPDATES,
            lambda _: jax.random.uniform(act_k, (NUM_PARALLEL_ENVS, N_AGENTS, 2), minval=-1.0, maxval=1.0),
            lambda _: get_batch_actions(a_p, obs, xi, act_k, True), None
        )
        
        nz, nxi, nobs, rew, dones = parallel_marl_physics_step(z, xi, target, actions, phys_k)
        steps += 1
        needs_reset = jnp.logical_or(dones.flatten(), steps >= MAX_ENV_STEPS)
        new_buf = add_batch_to_buffer(buf, obs, xi, actions, jnp.where(dones[:, None], -100.0, rew), nobs, nxi, jnp.tile(dones[:, None, :], (1, N_AGENTS, 1)))
        
        idx = jax.random.randint(res_k, (NUM_PARALLEL_ENVS,), 0, 1000)
        z_next = jnp.where(needs_reset[:, None], z_init_bank[idx], nz)
        xi_next = jnp.where(needs_reset[:, None], jnp.tile(jnp.linspace(0.2, 0.8, N_AGENTS), (NUM_PARALLEL_ENVS, 1)), nxi)
        
        def do_updates(st):
            cp, ap, tap, tcp, oc, oa = st
            bx, bxi, bu, br, bnx, bnxi, bd = sample_buffer(new_buf, ENV_BATCH_SIZE, samp_k)
            bpe = jax.vmap(lambda x: get_sinusoidal_encoding(x, PE_DIM))(bxi)
            bnpe = jax.vmap(lambda x: get_sinusoidal_encoding(x, PE_DIM))(bnxi)
            bx_f, bnx_f = jnp.concatenate([bx, bpe], axis=-1), jnp.concatenate([bnx, bnpe], axis=-1)
            new_cp, new_oc = update_critic(cp, tap, tcp, oc, bx_f, bu, br, bnx_f, bd, net_k)
            new_ap, new_tap, new_tcp, new_oa = jax.lax.cond(step_idx % POLICY_DELAY == 0,
                lambda _: update_actor_and_targets(ap, new_cp, tap, tcp, oa, bx_f), lambda _: (ap, tap, tcp, oa), None)
            return new_cp, new_ap, new_tap, new_tcp, new_oc, new_oa

        c_p, a_p, ta_p, tc_p, o_c, o_a = jax.lax.cond(new_buf.size >= ENV_BATCH_SIZE, do_updates, lambda x: x, (c_p, a_p, ta_p, tc_p, o_c, o_a))
        return (new_buf, a_p, c_p, ta_p, tc_p, o_a, o_c, z_next, target, xi_next, build_marl_obs_batch(z_next, target, xi_next), jnp.where(needs_reset, 0, steps), rng), None

    return jax.lax.scan(scan_step, carry, step_indices)

# --- 6. EXECUTION ---
bank_keys = jax.random.split(key, 1000)
_, z_init_bank = jax.vmap(partial(generate_grf, n_points=100, length_scale=0.2))(bank_keys)
_, z_target_bank = jax.vmap(partial(generate_grf, n_points=100, length_scale=0.4))(bank_keys)
xi_init_single = jnp.linspace(0.2, 0.8, N_AGENTS)

idx = jax.random.randint(jax.random.split(key)[0], (NUM_PARALLEL_ENVS,), 0, 1000)
carry = (DeviceReplayBuffer.create(125_000, stored_obs_dim, 2), actor_params, critic_params, target_actor_params, target_critic_params, opt_actor, opt_critic, z_init_bank[idx], z_target_bank[idx], jnp.tile(xi_init_single, (NUM_PARALLEL_ENVS, 1)), build_marl_obs_batch(z_init_bank[idx], z_target_bank[idx], jnp.tile(xi_init_single, (NUM_PARALLEL_ENVS, 1))), jnp.zeros(NUM_PARALLEL_ENVS), key)

for chunk in trange(TOTAL_UPDATES // EVAL_INT):
    carry, _ = train_chunk(carry, jnp.arange(chunk*EVAL_INT, (chunk+1)*EVAL_INT), z_init_bank, z_target_bank)
    eval_mse, crashed = fast_eval_episode(carry[1], z_init_bank[0], xi_init_single, z_target_bank[0], MAX_ENV_STEPS, key)
    print(f"Update {(chunk+1)*EVAL_INT:06d} | Eval MSE: {'[CRASH]' if crashed else f'{eval_mse:.6f}'}")

with open('models/mappo_heat1d_params.msgpack', 'wb') as f:
    f.write(flax.serialization.to_bytes({'actor': carry[1]}))