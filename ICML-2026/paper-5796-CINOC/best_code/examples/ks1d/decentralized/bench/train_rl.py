import jax
import jax.numpy as jnp
import optax
import flax.serialization
from flax import struct
import numpy as np
import time
from pathlib import Path
import jax.tree_util
import sys
from functools import partial
from tqdm import trange

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(script_dir))

# Project imports
from models_rl import CentralizedActor, CentralizedCritic
from examples.ks1d.decentralized.data_utils import evolve_to_attractor
from examples.ks1d.decentralized.dynamics_dual import PDEDynamics 

# --- Configurations ---
N_AGENTS = 8
L_DOMAIN = 22.0
N_GRID = 128
BATCH_SIZE = 512
EVAL_INT = 500
POLICY_DELAY = 2
MAX_ENV_STEPS = 200 
STATE_NORM_FACTOR = 5.0 

# Vectorization Configs
NUM_PARALLEL_ENVS = 256
TOTAL_UPDATES = 100000 
WARMUP_UPDATES = 500

# --- Initialization ---
key = jax.random.PRNGKey(42)

def direct_control_policy(action_params, u_obs, u_target, xi_fixed):
    return action_params

dynamics = PDEDynamics(policy_apply_fn=direct_control_policy)
xi_fixed = jnp.linspace(0.0, L_DOMAIN, N_AGENTS, endpoint=False) + (L_DOMAIN/N_AGENTS)/2

actor = CentralizedActor(n_agents=N_AGENTS)
critic = CentralizedCritic()

key, *subkeys = jax.random.split(key, 4)
dummy_state = jnp.zeros((BATCH_SIZE, N_GRID))
dummy_action = jnp.zeros((BATCH_SIZE, N_AGENTS))

actor_params = actor.init(subkeys[0], dummy_state)
critic_params = critic.init(subkeys[1], dummy_state, dummy_action)
target_actor_params, target_critic_params = actor_params, critic_params

tx_actor = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-6))
tx_critic = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(5e-5))
opt_actor = tx_actor.init(actor_params)
opt_critic = tx_critic.init(critic_params)

# --- 1. ON-DEVICE REPLAY BUFFER ---
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
            s=jnp.zeros((max_size, s_dim), dtype=jnp.float32),
            a=jnp.zeros((max_size, a_dim), dtype=jnp.float32),
            r=jnp.zeros((max_size, 1), dtype=jnp.float32),
            ns=jnp.zeros((max_size, s_dim), dtype=jnp.float32),
            d=jnp.zeros((max_size, 1), dtype=jnp.float32),
            ptr=jnp.int32(0),
            size=jnp.int32(0),
            max_size=max_size
        )

@jax.jit
def add_batch_to_buffer(buffer, s_batch, a_batch, r_batch, ns_batch, d_batch):
    batch_size = s_batch.shape[0]
    indices = (buffer.ptr + jnp.arange(batch_size)) % buffer.max_size
    
    new_s = buffer.s.at[indices].set(s_batch)
    new_a = buffer.a.at[indices].set(a_batch)
    new_r = buffer.r.at[indices].set(r_batch)
    new_ns = buffer.ns.at[indices].set(ns_batch)
    new_d = buffer.d.at[indices].set(d_batch)
    
    new_ptr = (buffer.ptr + batch_size) % buffer.max_size
    new_size = jnp.minimum(buffer.size + batch_size, buffer.max_size)
    
    return buffer.replace(s=new_s, a=new_a, r=new_r, ns=new_ns, d=new_d, ptr=new_ptr, size=new_size)

@partial(jax.jit, static_argnames=['batch_size'])
def sample_buffer(buffer, batch_size, key):
    valid_range = jnp.minimum(buffer.size, buffer.max_size)
    indices = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=valid_range)
    return buffer.s[indices], buffer.a[indices], buffer.r[indices], buffer.ns[indices], buffer.d[indices]

buffer = DeviceReplayBuffer.create(200_000, N_GRID, N_AGENTS)

# --- JIT Training & Rollout Functions ---
@jax.jit
def update_critic(c_p, ta_p, tc_p, opt_c, s, a, r, ns, d, key): 
    key, noise_key = jax.random.split(key)
    
    s_norm = s / STATE_NORM_FACTOR
    ns_norm = ns / STATE_NORM_FACTOR
    
    noise = jnp.clip(jax.random.normal(noise_key, a.shape) * 0.2, -0.5, 0.5)
    next_a = jnp.clip(actor.apply(ta_p, ns_norm) + noise, -1.0, 1.0)
    
    t_q1, t_q2 = critic.apply(tc_p, ns_norm, next_a)
    target_q = r + 0.99 * (1.0 - d) * jnp.minimum(t_q1, t_q2)
    
    def c_loss_fn(p):
        q1, q2 = critic.apply(p, s_norm, a)
        return jnp.mean((q1 - target_q)**2 + (q2 - target_q)**2)
    
    l_c, grads_c = jax.value_and_grad(c_loss_fn)(c_p)
    up_c, opt_c = tx_critic.update(grads_c, opt_c)
    return optax.apply_updates(c_p, up_c), opt_c

@jax.jit
def update_actor_and_targets(a_p, c_p, ta_p, tc_p, opt_a, s):
    s_norm = s / STATE_NORM_FACTOR
    
    def a_loss_fn(p):
        return -jnp.mean(critic.apply(c_p, s_norm, actor.apply(p, s_norm))[0])
    
    l_a, grads_a = jax.value_and_grad(a_loss_fn)(a_p)
    up_a, opt_a = tx_actor.update(grads_a, opt_a)
    a_p = optax.apply_updates(a_p, up_a)
    
    tau = 0.005
    new_ta = jax.tree_util.tree_map(lambda new, old: tau*new + (1-tau)*old, a_p, ta_p)
    new_tc = jax.tree_util.tree_map(lambda new, old: tau*new + (1-tau)*old, c_p, tc_p)
    return a_p, new_ta, new_tc, opt_a

@partial(jax.jit, static_argnames=['add_noise'])
def get_batch_actions(a_p, u_batch, key, add_noise=True):
    u_batch_norm = u_batch / STATE_NORM_FACTOR
    actions = jax.vmap(actor.apply, in_axes=(None, 0))(a_p, u_batch_norm)
    if add_noise:
        noise = jax.random.normal(key, actions.shape) * 0.1
        actions = jnp.clip(actions + noise, -1.0, 1.0)
    return actions

@jax.jit
def parallel_physics_step(u_batch, actions, xi_fixed):
    def single_physics_step(u_single, act_single):
        traj = dynamics.unroll_controlled(
            u_single, xi_fixed, jnp.zeros(N_GRID), act_single, 
            1, N_grid=N_GRID, L=L_DOMAIN
        )
        return traj[0][-1]
    
    next_u_batch = jax.vmap(single_physics_step)(u_batch, actions)
    
    is_invalid = jnp.logical_not(jnp.isfinite(next_u_batch).all(axis=-1, keepdims=True))
    is_exploding = jnp.max(jnp.abs(next_u_batch), axis=-1, keepdims=True) > 100.0
    dones_batch = jnp.logical_or(is_invalid, is_exploding)
    
    safe_u = jnp.where(dones_batch, jnp.zeros_like(next_u_batch), next_u_batch)
    rewards_batch = -(jnp.mean(safe_u**2, axis=-1, keepdims=True) / 10.0)
    
    return next_u_batch, rewards_batch, dones_batch

# --- 2. FAST JIT-COMPILED EVALUATION ---
@partial(jax.jit, static_argnames=['max_steps'])
def fast_eval_episode(actor_params, init_state, xi_fixed, max_steps):
    def step_fn(state, _):
        state_norm = state / STATE_NORM_FACTOR
        act = actor.apply(actor_params, state_norm)
        
        traj = dynamics.unroll_controlled(
            state, xi_fixed, jnp.zeros(N_GRID), act, 
            1, N_grid=N_GRID, L=L_DOMAIN
        )
        next_state = traj[0][-1]
        
        energy = jnp.mean(next_state**2)
        crashed = jnp.isnan(next_state).any() | jnp.isinf(next_state).any() | (jnp.max(jnp.abs(next_state)) > 100.0)
        
        return next_state, (energy, crashed)

    _, (energies, crashes) = jax.lax.scan(step_fn, init_state, None, length=max_steps)
    return jnp.mean(energies), jnp.any(crashes)


# --- 3. THE SCAN-COMPILED TRAINING CHUNK ---
@jax.jit
def train_chunk(carry, step_indices, state_bank, xi_fixed):
    def scan_step(carry, step_idx):
        buf, a_p, c_p, ta_p, tc_p, o_a, o_c, u, steps, rng = carry
        rng, act_k, res_k, samp_k, net_k = jax.random.split(rng, 5)
        
        # 1. Action Selection (Warmup vs Policy)
        def warmup_actions(_):
            return jax.random.uniform(act_k, (NUM_PARALLEL_ENVS, N_AGENTS), minval=-1.0, maxval=1.0)
        def policy_actions(_):
            return get_batch_actions(a_p, u, act_k, add_noise=True)
            
        actions = jax.lax.cond(step_idx < WARMUP_UPDATES, warmup_actions, policy_actions, None)
        
        # 2. Physics Step
        next_u, rewards, dones = parallel_physics_step(u, actions, xi_fixed)
        steps += 1
        truncs = steps >= MAX_ENV_STEPS
        needs_reset = jnp.logical_or(dones.flatten(), truncs)
        
        # 3. Update Buffer
        safe_next_u = jnp.where(dones, jnp.zeros_like(next_u), next_u)
        safe_rewards = jnp.where(dones, -1000.0, rewards)
        new_buf = add_batch_to_buffer(buf, u, actions, safe_rewards, safe_next_u, dones)
        
        # 4. Handle Resets
        fresh_states = jax.random.choice(res_k, state_bank, shape=(NUM_PARALLEL_ENVS,))
        u_next = jnp.where(needs_reset[:, None], fresh_states, safe_next_u)
        steps_next = jnp.where(needs_reset, 0, steps)

        # 5. Network Updates (Conditional on Buffer Size)
        def do_network_updates(net_state):
            c_p, a_p, ta_p, tc_p, o_c, o_a = net_state
            
            bs, ba, br, bns, bd = sample_buffer(new_buf, BATCH_SIZE, samp_k)
            
            # Critic Update
            new_c_p, new_o_c = update_critic(c_p, ta_p, tc_p, o_c, bs, ba, br, bns, bd, net_k)
            
            # Policy Delayed Actor Update
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
        
        # We replace python_buffer_size with a native JAX size check
        c_p, a_p, ta_p, tc_p, o_c, o_a = jax.lax.cond(
            new_buf.size >= BATCH_SIZE, do_network_updates, skip_network_updates, net_state
        )

        new_carry = (new_buf, a_p, c_p, ta_p, tc_p, o_a, o_c, u_next, steps_next, rng)
        return new_carry, None

    return jax.lax.scan(scan_step, carry, step_indices)

# --- Vectorized Training Loop ---
print("Pre-generating starting state bank (Vectorized)...")
bank_keys = jax.random.split(key, 1000)
state_bank = jax.vmap(lambda k: evolve_to_attractor(k, N_GRID, L_DOMAIN))(bank_keys)

key, subkey = jax.random.split(key)
u_batch = jax.random.choice(subkey, state_bank, shape=(NUM_PARALLEL_ENVS,))
env_step_counts = jnp.zeros(NUM_PARALLEL_ENVS)

# Pack everything into the initial carry state
carry = (
    buffer, actor_params, critic_params, target_actor_params, target_critic_params,
    opt_actor, opt_critic, u_batch, env_step_counts, key
)

print("Starting Massively Parallel RL Training (Chunked & JITed 1D KS Equation)...")
start_time = time.time()

num_chunks = TOTAL_UPDATES // EVAL_INT

for chunk_idx in trange(num_chunks):
    start_step = chunk_idx * EVAL_INT
    step_indices = jnp.arange(start_step, start_step + EVAL_INT)
    
    # Run the compiled chunk
    carry, _ = train_chunk(carry, step_indices, state_bank, xi_fixed)
    
    # Unpack the current actor for evaluation
    current_actor_params = carry[1] 
    
    # Fast Evaluation
    eval_u = state_bank[0] 
    eval_e, crashed = fast_eval_episode(current_actor_params, eval_u, xi_fixed, MAX_ENV_STEPS)
    
    current_total_step = start_step + EVAL_INT
    episode_num = current_total_step // MAX_ENV_STEPS
    
    if crashed:
        print(f"\nUpdate {current_total_step:05d} | Episode {episode_num} | Eval Energy: [CRASHED] | Time: {time.time()-start_time:.1f}s")
    else:
        print(f"\nUpdate {current_total_step:05d} | Episode {episode_num} | Eval Energy: {eval_e:.6f} | Time: {time.time()-start_time:.1f}s")

# Extract final weights and save
final_actor_params = carry[1]
with open('models/rl_centralized_params.msgpack', 'wb') as f:
    f.write(flax.serialization.to_bytes({'actor': final_actor_params}))
print(f"Training finished in {time.time()-start_time:.1f}s. Weights saved.")