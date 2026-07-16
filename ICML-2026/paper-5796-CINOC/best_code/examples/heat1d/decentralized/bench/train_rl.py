import jax
import jax.numpy as jnp
import optax
import flax.serialization
from flax import struct
import time
from pathlib import Path
import sys
from functools import partial
from tqdm import trange

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(script_dir))

# Project imports for Heat Equation
from models_rl import CentralizedActor, CentralizedCritic, U_MAX, V_MAX
from examples.heat1d.decentralized.data_utils import generate_grf
from examples.heat1d.decentralized.dynamics_dual import PDEDynamics 

# --- Configurations ---
N_AGENTS = 8  
L_DOMAIN = 1.0
N_GRID = 100
BATCH_SIZE = 256
EVAL_INT = 500
POLICY_DELAY = 2
MAX_ENV_STEPS = 300 

# Vectorization Configs
NUM_PARALLEL_ENVS = 256
TOTAL_UPDATES = 100000 
WARMUP_UPDATES = 500

# --- Initialization ---
key = jax.random.PRNGKey(42)

def direct_control_policy(action_params, u_obs, u_target, xi_fixed):
    u = action_params[:, 0]
    v = action_params[:, 1]
    return u, v

dynamics = PDEDynamics(policy_apply_fn=direct_control_policy)

actor = CentralizedActor(n_agents=N_AGENTS)
critic = CentralizedCritic()

key, *subkeys = jax.random.split(key, 4)
dummy_z = jnp.zeros((BATCH_SIZE, N_GRID))
dummy_target = jnp.zeros((BATCH_SIZE, N_GRID))
dummy_xi = jnp.zeros((BATCH_SIZE, N_AGENTS))
dummy_action = jnp.zeros((BATCH_SIZE, N_AGENTS, 2))

actor_params = actor.init(subkeys[0], dummy_z, dummy_target, dummy_xi)
critic_params = critic.init(subkeys[1], dummy_z, dummy_target, dummy_xi, dummy_action)
target_actor_params, target_critic_params = actor_params, critic_params

tx_actor = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-4))
tx_critic = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(5e-4))
opt_actor = tx_actor.init(actor_params)
opt_critic = tx_critic.init(critic_params)

# --- 1. ON-DEVICE REPLAY BUFFER ---
@struct.dataclass
class DeviceReplayBuffer:
    z: jnp.ndarray
    zt: jnp.ndarray
    xi: jnp.ndarray
    a: jnp.ndarray
    r: jnp.ndarray
    nz: jnp.ndarray
    nxi: jnp.ndarray
    d: jnp.ndarray
    ptr: jnp.int32
    size: jnp.int32
    max_size: int = struct.field(pytree_node=False)

    @classmethod
    def create(cls, max_size, s_dim, a_dim):
        return cls(
            z=jnp.zeros((max_size, s_dim), dtype=jnp.float32),
            zt=jnp.zeros((max_size, s_dim), dtype=jnp.float32),
            xi=jnp.zeros((max_size, N_AGENTS), dtype=jnp.float32),
            a=jnp.zeros((max_size, N_AGENTS, a_dim), dtype=jnp.float32),
            r=jnp.zeros((max_size, 1), dtype=jnp.float32),
            nz=jnp.zeros((max_size, s_dim), dtype=jnp.float32),
            nxi=jnp.zeros((max_size, N_AGENTS), dtype=jnp.float32),
            d=jnp.zeros((max_size, 1), dtype=jnp.float32),
            ptr=jnp.int32(0),
            size=jnp.int32(0),
            max_size=max_size
        )

@jax.jit
def add_batch_to_buffer(buffer, z_b, zt_b, xi_b, a_b, r_b, nz_b, nxi_b, d_b):
    batch_size = z_b.shape[0]
    indices = (buffer.ptr + jnp.arange(batch_size)) % buffer.max_size
    
    new_z = buffer.z.at[indices].set(z_b)
    new_zt = buffer.zt.at[indices].set(zt_b)
    new_xi = buffer.xi.at[indices].set(xi_b)
    new_a = buffer.a.at[indices].set(a_b)
    new_r = buffer.r.at[indices].set(r_b)
    new_nz = buffer.nz.at[indices].set(nz_b)
    new_nxi = buffer.nxi.at[indices].set(nxi_b)
    new_d = buffer.d.at[indices].set(d_b)
    
    new_ptr = (buffer.ptr + batch_size) % buffer.max_size
    new_size = jnp.minimum(buffer.size + batch_size, buffer.max_size)
    
    return buffer.replace(z=new_z, zt=new_zt, xi=new_xi, a=new_a, r=new_r, nz=new_nz, nxi=new_nxi, d=new_d, ptr=new_ptr, size=new_size)

@partial(jax.jit, static_argnames=['batch_size'])
def sample_buffer(buffer, batch_size, key):
    valid_range = jnp.minimum(buffer.size, buffer.max_size)
    indices = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=valid_range)
    return buffer.z[indices], buffer.zt[indices], buffer.xi[indices], buffer.a[indices], buffer.r[indices], buffer.nz[indices], buffer.nxi[indices], buffer.d[indices]

# --- JIT Training & Rollout Functions ---
@jax.jit
def update_critic(c_p, ta_p, tc_p, opt_c, z, zt, xi, a, r, nz, nxi, d, key): 
    key, noise_key = jax.random.split(key)
    
    noise_scale = jnp.array([U_MAX, V_MAX]) * 0.1
    noise = jnp.clip(jax.random.normal(noise_key, a.shape) * noise_scale, -0.5 * noise_scale, 0.5 * noise_scale)
    
    raw_next_a = actor.apply(ta_p, nz, zt, nxi) + noise
    next_a = jnp.clip(raw_next_a, jnp.array([-U_MAX, -V_MAX]), jnp.array([U_MAX, V_MAX]))
    
    t_q1, t_q2 = critic.apply(tc_p, nz, zt, nxi, next_a)
    target_q = r + 0.99 * (1.0 - d) * jnp.minimum(t_q1, t_q2)
    
    def c_loss_fn(p):
        q1, q2 = critic.apply(p, z, zt, xi, a)
        return jnp.mean((q1 - target_q)**2 + (q2 - target_q)**2)
    
    l_c, grads_c = jax.value_and_grad(c_loss_fn)(c_p)
    up_c, opt_c = tx_critic.update(grads_c, opt_c)
    return optax.apply_updates(c_p, up_c), opt_c

@jax.jit
def update_actor_and_targets(a_p, c_p, ta_p, tc_p, opt_a, z, zt, xi):
    def a_loss_fn(p):
        return -jnp.mean(critic.apply(c_p, z, zt, xi, actor.apply(p, z, zt, xi))[0])
    
    l_a, grads_a = jax.value_and_grad(a_loss_fn)(a_p)
    up_a, opt_a = tx_actor.update(grads_a, opt_a)
    a_p = optax.apply_updates(a_p, up_a)
    
    tau = 0.005
    new_ta = jax.tree_util.tree_map(lambda new, old: tau*new + (1-tau)*old, a_p, ta_p)
    new_tc = jax.tree_util.tree_map(lambda new, old: tau*new + (1-tau)*old, c_p, tc_p)
    return a_p, new_ta, new_tc, opt_a

@partial(jax.jit, static_argnames=['add_noise'])
def get_batch_actions(a_p, z_batch, z_target_batch, xi_batch, key, add_noise=True):
    actions = jax.vmap(actor.apply, in_axes=(None, 0, 0, 0))(a_p, z_batch, z_target_batch, xi_batch)
    if add_noise:
        noise_scale = jnp.array([U_MAX, V_MAX]) * 0.1
        noise = jax.random.normal(key, actions.shape) * noise_scale
        actions = jnp.clip(actions + noise, jnp.array([-U_MAX, -V_MAX]), jnp.array([U_MAX, V_MAX]))
    return actions

@jax.jit
def parallel_physics_step(z_batch, xi_batch, target_batch, actions, key):
    keys = jax.random.split(key, z_batch.shape[0])
    
    def single_physics_step(z_s, xi_s, target_s, act_s, k_s):
        traj = dynamics.unroll_controlled(
            z_init=z_s, xi_init=xi_s, z_target=target_s, params=act_s, 
            t_steps=1
        )
        return traj[0][-1], traj[1][-1]
    
    next_z_batch, next_xi_batch = jax.vmap(single_physics_step)(z_batch, xi_batch, target_batch, actions, keys)
    
    is_invalid = jnp.logical_not(jnp.isfinite(next_z_batch).all(axis=-1, keepdims=True))
    dones_batch = is_invalid
    
    safe_z = jnp.where(dones_batch, jnp.zeros_like(next_z_batch), next_z_batch)
    safe_xi = jnp.where(dones_batch, xi_batch, next_xi_batch)
    
    u_batch = actions[..., 0]
    v_batch = actions[..., 1]
    
    # Global Tracking MSE
    mse = jnp.mean((safe_z - target_batch)**2, axis=-1, keepdims=True)
    # Effort Penalty
    effort = jnp.mean(0.001 * (jnp.square(u_batch) + 0.1 * jnp.square(v_batch)), axis=-1, keepdims=True)
    # Global margin penalty
    margin = 0.02
    oob_penalty = 100.0 * (jnp.maximum(0.0, margin - safe_xi)**2 + jnp.maximum(0.0, safe_xi - (1.0 - margin))**2)
    mean_oob_penalty = jnp.mean(oob_penalty, axis=-1, keepdims=True)
    # Collision Penalty
    R_safe = 0.05
    dists = jnp.abs(safe_xi[:, :, None] - safe_xi[:, None, :])
    mask = jnp.eye(N_AGENTS)[None, :, :]
    coll_penalty = 1.0 * jnp.sum(jnp.maximum(0.0, R_safe - (dists + mask * 1.0)) ** 2, axis=2)
    mean_coll_penalty = jnp.mean(coll_penalty, axis=-1, keepdims=True)
    
    rewards_batch = -mse - effort - mean_oob_penalty - mean_coll_penalty
    return safe_z, safe_xi, rewards_batch, dones_batch

# --- FAST JIT-COMPILED EVALUATION ---
@partial(jax.jit, static_argnames=['max_steps'])
def fast_eval_episode(actor_params, init_z, init_xi, target_z, max_steps, key):
    def step_fn(state_tuple, _):
        z_curr, xi_curr, k = state_tuple
        k, subk = jax.random.split(k)
        
        act = actor.apply(actor_params, z_curr[None, ...], target_z[None, ...], xi_curr[None, ...])
        act_flat = act.squeeze(0)
        
        traj = dynamics.unroll_controlled(
            z_init=z_curr, xi_init=xi_curr, z_target=target_z, params=act_flat, 
            t_steps=1
        )
        next_z, next_xi = traj[0][-1], traj[1][-1]
        
        mse = jnp.mean((next_z - target_z)**2)
        crashed = jnp.isnan(next_z).any() | jnp.isinf(next_z).any()
        
        return (next_z, next_xi, k), (mse, crashed)

    _, (mses, crashes) = jax.lax.scan(step_fn, (init_z, init_xi, key), None, length=max_steps)
    return jnp.mean(mses), jnp.any(crashes)

# --- THE SCAN-COMPILED TRAINING CHUNK ---
@jax.jit
def train_chunk(carry, step_indices, z_init_bank, z_target_bank):
    """Executes a chunk of TD3 updates entirely on the GPU."""
    def scan_step(carry, step_idx):
        buf, a_p, c_p, ta_p, tc_p, o_a, o_c, z, target, xi, steps, rng = carry
        rng, act_k, phys_k, res_k, samp_k, net_k = jax.random.split(rng, 6)
        
        # 1. Action Selection (Warmup vs Policy)
        def warmup_actions(_):
            return jax.random.uniform(act_k, (NUM_PARALLEL_ENVS, N_AGENTS, 2), 
                                      minval=jnp.array([-U_MAX, -V_MAX]), 
                                      maxval=jnp.array([U_MAX, V_MAX]))
        def policy_actions(_):
            return get_batch_actions(a_p, z, target, xi, act_k, add_noise=True)
            
        actions = jax.lax.cond(step_idx < WARMUP_UPDATES, warmup_actions, policy_actions, None)
        
        # 2. Physics Step
        nz, nxi, rew, dones = parallel_physics_step(z, xi, target, actions, phys_k)
        steps += 1
        truncs = steps >= MAX_ENV_STEPS
        needs_reset = jnp.logical_or(dones.flatten(), truncs)
        
        # 3. Update Buffer
        safe_rew = jnp.where(dones, -100.0, rew)
        new_buf = add_batch_to_buffer(buf, z, target, xi, actions, safe_rew, nz, nxi, dones)
        
        # 4. Handle Resets
        idx_reset = jax.random.randint(res_k, (NUM_PARALLEL_ENVS,), 0, 1000)
        fresh_z = z_init_bank[idx_reset]
        fresh_target = z_target_bank[idx_reset]
        fresh_xi = jnp.tile(jnp.linspace(0.2, 0.8, N_AGENTS, dtype=jnp.float32), (NUM_PARALLEL_ENVS, 1))
        
        z_next = jnp.where(needs_reset[:, None], fresh_z, nz)
        target_next = jnp.where(needs_reset[:, None], fresh_target, target)
        xi_next = jnp.where(needs_reset[:, None], fresh_xi, nxi)
        steps_next = jnp.where(needs_reset, 0, steps)

        # 5. Network Updates (Conditional on Buffer Size)
        def do_network_updates(net_state):
            c_p, a_p, ta_p, tc_p, o_c, o_a = net_state
            bz, bzt, bxi, ba, br, bnz, bnxi, bd = sample_buffer(new_buf, BATCH_SIZE, samp_k)
            
            # Critic Update
            new_c_p, new_o_c = update_critic(c_p, ta_p, tc_p, o_c, bz, bzt, bxi, ba, br, bnz, bnxi, bd, net_k)
            
            # Policy Delayed Actor Update
            def do_actor_update(_):
                return update_actor_and_targets(a_p, new_c_p, ta_p, tc_p, o_a, bz, bzt, bxi)
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
            new_buf.size >= BATCH_SIZE, do_network_updates, skip_network_updates, net_state
        )

        new_carry = (new_buf, a_p, c_p, ta_p, tc_p, o_a, o_c, z_next, target_next, xi_next, steps_next, rng)
        return new_carry, None

    return jax.lax.scan(scan_step, carry, step_indices)

# --- Main Execution Loop ---
print("Pre-generating starting state & target banks (Vectorized)...")
bank_keys = jax.random.split(key, 1000)
_, z_init_bank = jax.vmap(partial(generate_grf, n_points=N_GRID, length_scale=0.2))(bank_keys)
_, z_target_bank = jax.vmap(partial(generate_grf, n_points=N_GRID, length_scale=0.4))(bank_keys)
xi_init_single = jnp.linspace(0.2, 0.8, N_AGENTS, dtype=jnp.float32)

key, subkey = jax.random.split(key)
idx = jax.random.randint(subkey, (NUM_PARALLEL_ENVS,), 0, 1000)
z_init = z_init_bank[idx]
target_init = z_target_bank[idx]
xi_init = jnp.tile(xi_init_single, (NUM_PARALLEL_ENVS, 1))
steps_init = jnp.zeros(NUM_PARALLEL_ENVS)
buffer = DeviceReplayBuffer.create(125_000, N_GRID, 2)

# Pack everything into the initial carry state
carry = (
    buffer, actor_params, critic_params, target_actor_params, target_critic_params,
    opt_actor, opt_critic, z_init, target_init, xi_init, steps_init, key
)

print(f"Starting Massively Parallel Centralized RL Training (Chunked & JITed)...")
start_time = time.time()

num_chunks = TOTAL_UPDATES // EVAL_INT

for chunk_idx in trange(num_chunks):
    start_step = chunk_idx * EVAL_INT
    step_indices = jnp.arange(start_step, start_step + EVAL_INT)
    
    # Run the compiled chunk
    carry, _ = train_chunk(carry, step_indices, z_init_bank, z_target_bank)
    
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
        print(f"Update {current_total_step:06d} | Episode {episode_num} | Eval Tracking MSE: [CRASHED] | Time: {time.time()-start_time:.1f}s")
    else:
        print(f"Update {current_total_step:06d} | Episode {episode_num} | Eval Tracking MSE: {eval_e:.6f} | Time: {time.time()-start_time:.1f}s")

# Extract final weights and save
final_actor_params = carry[1]
with open('models/rl_heat_params.msgpack', 'wb') as f:
    f.write(flax.serialization.to_bytes({'actor': final_actor_params}))
print(f"Training finished in {time.time()-start_time:.1f}s. Weights saved.")