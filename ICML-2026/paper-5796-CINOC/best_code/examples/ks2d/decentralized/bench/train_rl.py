import jax
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

# Project imports
from models_rl import CentralizedActorKS2D, CentralizedCriticKS2D, U_MAX
from examples.ks2d.decentralized.data_utils import get_batch_initial_conditions
from examples.ks2d.decentralized.dynamics_dual import PDEDynamics2D 

# --- Configurations ---
N_AGENTS = 100
L_DOMAIN = 32.0
N_GRID = 64

ENV_BATCH_SIZE = 128 
EVAL_INT = 50  # Increased to make chunks larger and faster
POLICY_DELAY = 2 

# KS2D Specific Control Timing
MAX_ENV_STEPS = 50     # Control steps (T_steps)
SUBSTEPS = 10          # Physics steps per control step
DT = 0.005             # Physics dt

# Vectorization Configs
NUM_PARALLEL_ENVS = 64
TOTAL_UPDATES = 10000 
WARMUP_UPDATES = 500

STATE_NORM_FACTOR = 5.0

# --- Initialization ---
key = jax.random.PRNGKey(42)

def direct_control_policy(action_params, u_obs, u_target, xi_fixed):
    return action_params

dynamics = PDEDynamics2D(policy_apply_fn=direct_control_policy)

# 2D Actuator Grid Setup (Fixed Positions)
grid_dim = int(np.sqrt(N_AGENTS))
x_lin = np.linspace(0, L_DOMAIN, grid_dim, endpoint=False) + (L_DOMAIN/grid_dim)/2
xv, yv = jnp.meshgrid(x_lin, x_lin) # Use jnp instead of np for consistency
agent_positions = jnp.stack([xv.flatten(), yv.flatten()], axis=-1)
xi_fixed = jnp.array(agent_positions)
target_state = jnp.zeros((N_GRID, N_GRID))

actor = CentralizedActorKS2D(n_agents=N_AGENTS)
critic = CentralizedCriticKS2D()

key, *subkeys = jax.random.split(key, 4)

# Dummy inputs for initialization
dummy_z = jnp.zeros((ENV_BATCH_SIZE, N_GRID, N_GRID))
dummy_action = jnp.zeros((ENV_BATCH_SIZE, N_AGENTS))

actor_params = actor.init(subkeys[0], dummy_z)
critic_params = critic.init(subkeys[1], dummy_z, dummy_action)

target_actor_params = actor_params
target_critic_params = critic_params

tx_actor = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-6))
tx_critic = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(5e-5))
opt_actor = tx_actor.init(actor_params)
opt_critic = tx_critic.init(critic_params)

# --- 1. ON-DEVICE REPLAY BUFFER ---
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
    
    new_z = buffer.z.at[indices].set(z_b)
    new_a = buffer.a.at[indices].set(a_b)
    new_r = buffer.r.at[indices].set(r_b)
    new_nz = buffer.nz.at[indices].set(nz_b)
    new_d = buffer.d.at[indices].set(d_b)
    
    new_ptr = (buffer.ptr + batch_size) % buffer.max_size
    new_size = jnp.minimum(buffer.size + batch_size, buffer.max_size)
    
    return buffer.replace(z=new_z, a=new_a, r=new_r, nz=new_nz, d=new_d, ptr=new_ptr, size=new_size)

@partial(jax.jit, static_argnames=['batch_size'])
def sample_buffer(buffer, batch_size, key):
    valid_range = jnp.minimum(buffer.size, buffer.max_size)
    indices = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=valid_range)
    return buffer.z[indices], buffer.a[indices], buffer.r[indices], buffer.nz[indices], buffer.d[indices]

buffer = DeviceReplayBuffer.create(12_500, N_GRID, N_AGENTS)

# --- JIT TRAINING & ROLLOUT FUNCTIONS ---
@jax.jit
def update_critic(c_p, ta_p, tc_p, opt_c, z, a, r, nz, d, key): 
    key, noise_key = jax.random.split(key)
    
    z_norm = z / STATE_NORM_FACTOR
    nz_norm = nz / STATE_NORM_FACTOR
    
    noise = jnp.clip(jax.random.normal(noise_key, a.shape) * (U_MAX * 0.1), -U_MAX * 0.5, U_MAX * 0.5)
    next_a = jnp.clip(actor.apply(ta_p, nz_norm) + noise, -U_MAX, U_MAX)
    
    t_q1, t_q2 = critic.apply(tc_p, nz_norm, next_a)
    target_q = r + 0.99 * (1.0 - d) * jnp.minimum(t_q1, t_q2)
    
    def c_loss_fn(p):
        q1, q2 = critic.apply(p, z_norm, a)
        return jnp.mean((q1 - target_q)**2 + (q2 - target_q)**2)
    
    l_c, grads_c = jax.value_and_grad(c_loss_fn)(c_p)
    up_c, opt_c = tx_critic.update(grads_c, opt_c)
    return optax.apply_updates(c_p, up_c), opt_c

@jax.jit
def update_actor_and_targets(a_p, c_p, ta_p, tc_p, opt_a, z):
    z_norm = z / STATE_NORM_FACTOR
    
    def a_loss_fn(p):
        return -jnp.mean(critic.apply(c_p, z_norm, actor.apply(p, z_norm))[0])
    
    l_a, grads_a = jax.value_and_grad(a_loss_fn)(a_p)
    up_a, opt_a = tx_actor.update(grads_a, opt_a)
    a_p = optax.apply_updates(a_p, up_a)
    
    tau = 0.005
    new_ta = jax.tree_util.tree_map(lambda new, old: tau*new + (1-tau)*old, a_p, ta_p)
    new_tc = jax.tree_util.tree_map(lambda new, old: tau*new + (1-tau)*old, c_p, tc_p)
    return a_p, new_ta, new_tc, opt_a

@partial(jax.jit, static_argnames=['add_noise'])
def get_batch_actions(a_p, z_batch, key, add_noise=True):
    actions = jax.vmap(actor.apply, in_axes=(None, 0))(a_p, z_batch)
    
    if add_noise:
        noise = jax.random.normal(key, actions.shape) * (U_MAX * 0.1)
        actions = jnp.clip(actions + noise, -U_MAX, U_MAX)
    return actions

@jax.jit
def parallel_physics_step(u_batch, actions, xi_fixed):
    
    def single_physics_step(u_single, act_single):
        traj = dynamics.unroll_controlled(
            u_init=u_single, xi_fixed=xi_fixed, u_target=target_state, params=act_single, # Pass it here
            t_steps=1, substeps=SUBSTEPS, N_grid=N_GRID, L=L_DOMAIN, dt=DT, sigma=1.2
        )
        return traj[0][-1]
    
    next_u_batch = jax.vmap(single_physics_step)(u_batch, actions)
    
    is_invalid = jnp.logical_not(jnp.isfinite(next_u_batch).all(axis=(1, 2)))
    is_exploding = jnp.max(jnp.abs(next_u_batch), axis=(1, 2)) > 100.0
    dones_batch = jnp.logical_or(is_invalid, is_exploding)[:, None]
    
    safe_u = jnp.where(dones_batch[:, :, None], jnp.zeros_like(next_u_batch), next_u_batch)
    
    # Scale the reward!
    global_energy = jnp.mean(jnp.square(safe_u), axis=(1, 2))[:, None]
    rewards_batch = -(global_energy / 10.0) 
    
    return safe_u, rewards_batch, dones_batch

# --- 2. FAST JIT-COMPILED EVALUATION ---
@partial(jax.jit, static_argnames=['max_steps'])
def fast_eval_episode(actor_params, init_state, max_steps):
    def step_fn(state, _):
        act = actor.apply(actor_params, state[None, ...])
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

# --- 3. THE SCAN-COMPILED TRAINING CHUNK ---
@jax.jit
def train_chunk(carry, step_indices, state_bank, xi_fixed):
    def scan_step(carry, step_idx):
        buf, a_p, c_p, ta_p, tc_p, o_a, o_c, u, steps, rng = carry
        rng, act_k, res_k, samp_k, net_k = jax.random.split(rng, 5)
        
        # 1. Action Selection (Warmup vs Policy)
        def warmup_actions(_):
            return jax.random.uniform(act_k, (NUM_PARALLEL_ENVS, N_AGENTS), minval=-U_MAX, maxval=U_MAX)
        def policy_actions(_):
            return get_batch_actions(a_p, u, act_k, add_noise=True)
            
        actions = jax.lax.cond(step_idx < WARMUP_UPDATES, warmup_actions, policy_actions, None)
        
        # 2. Physics Step
        next_u, rewards, dones = parallel_physics_step(u, actions, xi_fixed)
        steps += 1
        truncs = steps >= MAX_ENV_STEPS
        needs_reset = jnp.logical_or(dones.flatten(), truncs)
        
        # 3. Update Buffer
        safe_next_u = jnp.where(dones[:, :, None], jnp.zeros_like(next_u), next_u)
        safe_rewards = jnp.where(dones, -100.0, rewards)
        new_buf = add_batch_to_buffer(buf, u, actions, safe_rewards, safe_next_u, dones)
        
        # 4. Handle Resets
        fresh_states = jax.random.choice(res_k, state_bank, shape=(NUM_PARALLEL_ENVS,))
        u_next = jnp.where(needs_reset[:, None, None], fresh_states, safe_next_u)
        steps_next = jnp.where(needs_reset, 0, steps)

        # 5. Network Updates (Conditional on Buffer Size)
        def do_network_updates(net_state):
            c_p, a_p, ta_p, tc_p, o_c, o_a = net_state
            
            bs, ba, br, bns, bd = sample_buffer(new_buf, ENV_BATCH_SIZE, samp_k)
            
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
        
        c_p, a_p, ta_p, tc_p, o_c, o_a = jax.lax.cond(
            new_buf.size >= ENV_BATCH_SIZE, do_network_updates, skip_network_updates, net_state
        )

        new_carry = (new_buf, a_p, c_p, ta_p, tc_p, o_a, o_c, u_next, steps_next, rng)
        return new_carry, None

    return jax.lax.scan(scan_step, carry, step_indices)

# --- Vectorized Training Loop ---
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
env_step_counts = jnp.zeros(NUM_PARALLEL_ENVS)

# Pack everything into the initial carry state
carry = (
    buffer, actor_params, critic_params, target_actor_params, target_critic_params,
    opt_actor, opt_critic, u_batch, env_step_counts, key
)

print("Starting Massively Parallel RL Training (Chunked & JITed 2D KS Equation)...")
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
    eval_e, crashed = fast_eval_episode(current_actor_params, eval_u, MAX_ENV_STEPS)
    
    current_total_step = start_step + EVAL_INT
    episode_num = current_total_step // MAX_ENV_STEPS
    
    if crashed:
        print(f"\nUpdate {current_total_step:05d} | Episode {episode_num} | Eval Energy: [CRASHED] | Time: {time.time()-start_time:.1f}s")
    else:
        print(f"\nUpdate {current_total_step:05d} | Episode {episode_num} | Eval Energy: {eval_e:.6f} | Time: {time.time()-start_time:.1f}s")

# Extract final weights and save
final_actor_params = carry[1]
with open('models/rl_ks2d_params.msgpack', 'wb') as f:
    f.write(flax.serialization.to_bytes({'actor': final_actor_params}))
print(f"Training finished in {time.time()-start_time:.1f}s. Weights saved.")