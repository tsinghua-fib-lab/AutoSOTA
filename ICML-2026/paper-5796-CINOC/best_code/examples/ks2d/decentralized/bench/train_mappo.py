import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
import flax.serialization
from flax.training.train_state import TrainState
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

from env_ks2d import extract_patches_2d_jit
from utils_hypemarl import get_sinusoidal_encoding
from examples.ks2d.decentralized.data_utils import get_batch_initial_conditions
from examples.ks2d.decentralized.dynamics_dual import PDEDynamics2D 
from examples.ks2d.decentralized.bench.models_mappo import MAPPOActor2DKS, MAPPOCritic2DKS

# ==========================================
# 1. CONFIGURATIONS
# ==========================================
N_AGENTS = 100
L_DOMAIN = 32.0
N_GRID = 64
PATCH_SIZE = 12

STATE_NORM_FACTOR = 5.0 
U_MAX = 5.0  

MAX_ENV_STEPS = 50     
SUBSTEPS = 10          
DT = 0.005             
NUM_PARALLEL_ENVS = 64

ROLLOUT_STEPS = 50 
PPO_EPOCHS = 4
MINIBATCH_SIZE = 1600 
TOTAL_UPDATES = 1000
EVAL_INT = 50 

GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.01
LR = 3e-4

key = jax.random.PRNGKey(42)

# ==========================================
# 2. UTILS & ENVIRONMENT SETUP
# ==========================================
def direct_control_policy(action_params, u_obs, u_target, xi_fixed):
    return action_params

dynamics = PDEDynamics2D(policy_apply_fn=direct_control_policy)

grid_dim = int(np.sqrt(N_AGENTS))
x_lin = jnp.linspace(0, L_DOMAIN, grid_dim, endpoint=False) + (L_DOMAIN/grid_dim)/2
xv, yv = jnp.meshgrid(x_lin, x_lin)
agent_positions = jnp.stack([xv.flatten(), yv.flatten()], axis=-1)
xi_fixed = jnp.array(agent_positions)
xi_norm = xi_fixed / L_DOMAIN
target_state = jnp.zeros((N_GRID, N_GRID))

def get_2d_sinusoidal_encoding(p_2d, d=64, n=1000.0):
    pe_x = get_sinusoidal_encoding(p_2d[:, 0], d=d, n=n)
    pe_y = get_sinusoidal_encoding(p_2d[:, 1], d=d, n=n)
    return jnp.concatenate([pe_x, pe_y], axis=-1)

# Pre-compute static positional encodings
pe_jax = get_2d_sinusoidal_encoding(xi_norm, d=64)

@jax.jit
def build_marl_obs_batch(u_batch):
    def single_env_obs(state):
        y_local = extract_patches_2d_jit(state, target_state, xi_norm, PATCH_SIZE, N_GRID)
        y_local = y_local / STATE_NORM_FACTOR
        return jnp.concatenate([y_local, pe_jax], axis=-1)
    return jax.vmap(single_env_obs)(u_batch)

def parallel_physics_step(u_batch, actions):
    def single_physics_step(u_s, act_s):
        traj = dynamics.unroll_controlled(
            u_init=u_s, xi_fixed=xi_fixed, u_target=target_state, params=act_s, 
            t_steps=1, substeps=SUBSTEPS, N_grid=N_GRID, L=L_DOMAIN, dt=DT, sigma=1.2
        )
        return traj[0][-1]
    
    next_u_batch = jax.vmap(single_physics_step)(u_batch, actions)
    
    is_invalid = jnp.logical_not(jnp.isfinite(next_u_batch).all(axis=(1, 2)))
    is_exploding = jnp.max(jnp.abs(next_u_batch), axis=(1, 2)) > 100.0
    dones_batch = jnp.logical_or(is_invalid, is_exploding)
    
    safe_u = jnp.where(dones_batch[:, None, None], jnp.zeros_like(next_u_batch), next_u_batch)
    
    # Local + Global Reward (Matches MATD3 logic)
    batched_extract = jax.vmap(extract_patches_2d_jit, in_axes=(0, None, None, None, None))
    y_local = batched_extract(safe_u, target_state, xi_norm, PATCH_SIZE, N_GRID)
    
    local_rewards = -jnp.mean(jnp.square(y_local), axis=-1)
    
    global_energy = jnp.mean(jnp.square(safe_u), axis=(1, 2))
    base_reward = jnp.where(dones_batch, -100.0, -(global_energy / 10.0))
    
    # Combine localized and centralized penalties
    rewards_batch = 0.5 * local_rewards + 0.5 * base_reward[:, None]
    dones_batch_exp = jnp.tile(dones_batch[:, None], (1, N_AGENTS))
    
    return safe_u, rewards_batch, dones_batch_exp

# ==========================================
# 3. PPO CORE FUNCTIONS (CTDE)
# ==========================================
def get_action_and_value(actor_params, critic_params, obs, global_u, key):
    # ACTOR: Uses decentralized local patch observations
    mean, log_std = actor.apply(actor_params, obs)
    std = jnp.exp(log_std)
    action = mean + std * jax.random.normal(key, mean.shape)
    log_prob = -0.5 * jnp.sum(jnp.square((action - mean) / std) + 2 * log_std + jnp.log(2 * jnp.pi), axis=-1)
    
    # CRITIC: Uses centralized global grid state
    val = critic.apply(critic_params, global_u / STATE_NORM_FACTOR)
    return action, log_prob, val

def compute_gae_jax(rewards, values, dones, true_next_values, last_val, gamma=0.99, lam=0.95):
    def scan_fn(carry, transition):
        r, v, d, true_next_v = transition
        gae, _ = carry
        nextnonterminal = 1.0 - d
        delta = r + gamma * true_next_v * nextnonterminal - v
        gae = delta + gamma * lam * nextnonterminal * gae
        return (gae, v), gae
    
    _, advantages = jax.lax.scan(
        scan_fn, 
        (jnp.zeros_like(last_val), last_val), 
        (rewards, values, dones, true_next_values), 
        reverse=True
    )
    returns = advantages + values
    return advantages, returns

def ppo_update_epoch(actor_state, critic_state, batch):
    obs, global_u, actions, old_log_probs, advantages, returns, old_values = batch    
    
    def actor_loss_fn(params):
        mean, log_std = actor.apply(params, obs) 
        std = jnp.exp(log_std)
        
        log_probs = -0.5 * jnp.sum(jnp.square((actions - mean) / std) + 2 * log_std + jnp.log(2 * jnp.pi), axis=-1)
        entropy = jnp.sum(0.5 + 0.5 * jnp.log(2 * jnp.pi) + log_std, axis=-1)
        ratio = jnp.exp(log_probs - old_log_probs)
        
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * jnp.clip(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()
        
        entropy_loss = entropy.mean()
        return pg_loss - ENTROPY_COEF * entropy_loss, (pg_loss, entropy_loss)
        
    def value_loss_fn(params):
        v = critic.apply(params, global_u / STATE_NORM_FACTOR) 
        
        v_clipped = old_values + jnp.clip(v - old_values, -CLIP_EPS, CLIP_EPS)
        v_loss_unclipped = jnp.square(v - returns)
        v_loss_clipped = jnp.square(v_clipped - returns)
        
        return 0.5 * jnp.maximum(v_loss_unclipped, v_loss_clipped).mean()
        
    (a_loss, aux), a_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_state.params)
    v_loss, v_grads = jax.value_and_grad(value_loss_fn)(critic_state.params)
    
    actor_state = actor_state.apply_gradients(grads=a_grads)
    critic_state = critic_state.apply_gradients(grads=v_grads)
    
    return actor_state, critic_state, a_loss, v_loss, aux[1]

# ==========================================
# 4. TRAINING LOGIC
# ==========================================
@jax.jit
def train_step(runner_state, state_bank):
    actor_state, critic_state, u_batch, obs_batch, env_counts, rng = runner_state
    
    # 1. Rollout Phase
    def _env_step(carry, _):
        u, obs, counts, k = carry
        k, act_k, reset_k = jax.random.split(k, 3)
        
        action, log_prob, val = get_action_and_value(actor_state.params, critic_state.params, obs, u, act_k)
        env_action = jnp.clip(action.squeeze(-1), -U_MAX, U_MAX)
        next_u, rewards, crashes = parallel_physics_step(u, env_action)
        
        counts += 1
        truncs = counts >= MAX_ENV_STEPS
        needs_reset = jnp.logical_or(crashes[:, 0], truncs)
        
        fresh_states = jax.random.choice(reset_k, state_bank, shape=(NUM_PARALLEL_ENVS,))
        next_u_final = jnp.where(needs_reset[:, None, None], fresh_states, next_u)
        next_counts = jnp.where(needs_reset, 0, counts)
        
        next_obs_final = build_marl_obs_batch(next_u_final)
        true_next_v = critic.apply(critic_state.params, next_u_final / STATE_NORM_FACTOR)
        
        transition = (obs, u, action, log_prob, rewards, val, crashes, true_next_v)
        return (next_u_final, next_obs_final, next_counts, k), transition

    carry = (u_batch, obs_batch, env_counts, rng)
    carry, transitions = jax.lax.scan(_env_step, carry, None, length=ROLLOUT_STEPS)
    (next_u_batch, next_obs_batch, next_env_counts, rng) = carry
    t_obs, t_u, t_a, t_logp, t_r, t_v, t_d, t_true_next_v = transitions
    
    # 2. GAE Phase
    last_val = critic.apply(critic_state.params, next_u_batch / STATE_NORM_FACTOR)
    adv, ret = compute_gae_jax(t_r, t_v, t_d, t_true_next_v, last_val, GAMMA, GAE_LAMBDA)
    
    # BATCH-WIDE ADVANTAGE NORMALIZATION GOES HERE
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    
    f_obs = t_obs.reshape(-1, N_AGENTS, t_obs.shape[-1])
    f_u = t_u.reshape(-1, N_GRID, N_GRID) 
    f_a = t_a.reshape(-1, N_AGENTS, 1)
    f_logp = t_logp.reshape(-1, N_AGENTS)
    f_ret = ret.reshape(-1, N_AGENTS)
    f_adv = adv.reshape(-1, N_AGENTS)
    f_v = t_v.reshape(-1, N_AGENTS)

    dataset_size = f_u.shape[0]
    num_minibatches = dataset_size // MINIBATCH_SIZE

    # 3. Optimization Phase
    def _update_epoch(epoch_carry, _):
        a_state, c_state, k = epoch_carry
        k, subk = jax.random.split(k)
        
        indices = jax.random.permutation(subk, dataset_size)
        s_obs = f_obs[indices]
        s_u = f_u[indices]
        s_a = f_a[indices]
        s_logp = f_logp[indices]
        s_ret = f_ret[indices]
        s_adv = f_adv[indices]
        s_v = f_v[indices]
        
        mb_obs = s_obs.reshape((num_minibatches, MINIBATCH_SIZE, *s_obs.shape[1:]))
        mb_u = s_u.reshape((num_minibatches, MINIBATCH_SIZE, *s_u.shape[1:]))
        mb_a = s_a.reshape((num_minibatches, MINIBATCH_SIZE, *s_a.shape[1:]))
        mb_logp = s_logp.reshape((num_minibatches, MINIBATCH_SIZE, *s_logp.shape[1:]))
        mb_ret = s_ret.reshape((num_minibatches, MINIBATCH_SIZE, *s_ret.shape[1:]))
        mb_adv = s_adv.reshape((num_minibatches, MINIBATCH_SIZE, *s_adv.shape[1:]))
        mb_v = s_v.reshape((num_minibatches, MINIBATCH_SIZE, *s_v.shape[1:]))

        def _update_minibatch(mb_carry, mb_data):
            a_state_, c_state_ = mb_carry
            mb_obs_, mb_u_, mb_a_, mb_logp_, mb_ret_, mb_adv_, mb_v_ = mb_data
                        
            mb_batch = (mb_obs_, mb_u_, mb_a_, mb_logp_, mb_adv_, mb_ret_, mb_v_)
            a_state_n, c_state_n, al, vl, ent = ppo_update_epoch(a_state_, c_state_, mb_batch)
            return (a_state_n, c_state_n), jnp.stack([al, vl, ent])
            
        (a_state, c_state), epoch_metrics = jax.lax.scan(
            _update_minibatch, (a_state, c_state), 
            (mb_obs, mb_u, mb_a, mb_logp, mb_ret, mb_adv, mb_v) 
        )
        return (a_state, c_state, k), epoch_metrics

    epoch_carry = (actor_state, critic_state, rng)
    epoch_carry, ppo_metrics = jax.lax.scan(_update_epoch, epoch_carry, None, length=PPO_EPOCHS)
    (actor_state, critic_state, rng) = epoch_carry

    new_runner_state = (actor_state, critic_state, next_u_batch, next_obs_batch, next_env_counts, rng)
    
    metrics = {
        "mean_return": t_r.sum(axis=0).mean(),
        "actor_loss": ppo_metrics[..., 0].mean(),
        "critic_loss": ppo_metrics[..., 1].mean(),
        "entropy": ppo_metrics[..., 2].mean()
    }
    
    return new_runner_state, metrics

# ACCELERATION: Chunk training on the device to minimize host-device sync
@partial(jax.jit, donate_argnums=(0,))
def train_chunk(runner_state, state_bank):
    def scan_step(carry, _):
        new_state, metrics = train_step(carry, state_bank)
        return new_state, metrics
    return jax.lax.scan(scan_step, runner_state, None, length=EVAL_INT)

# ==========================================
# 5. FAST EVALUATION
# ==========================================
@partial(jax.jit, static_argnames=['max_steps'])
def fast_eval_episode(actor_params, init_u, max_steps):
    def step_fn(state, _):
        obs = build_marl_obs_batch(state[None, ...])
        mean, _ = actor.apply(actor_params, obs)
        act_flat = mean.squeeze(0).squeeze(-1) 
        
        traj = dynamics.unroll_controlled(
            u_init=state, xi_fixed=xi_fixed, u_target=target_state, params=act_flat, 
            t_steps=1, substeps=SUBSTEPS, N_grid=N_GRID, L=L_DOMAIN, dt=DT, sigma=1.2
        )
        next_u = traj[0][-1]
        
        energy = jnp.mean(next_u**2)
        crashed = jnp.isnan(next_u).any() | jnp.isinf(next_u).any() | (jnp.max(jnp.abs(next_u)) > 100.0)
        return next_u, (energy, crashed)

    _, (energies, crashes) = jax.lax.scan(step_fn, init_u, None, length=max_steps)
    return jnp.mean(energies), jnp.any(crashes)


# ==========================================
# 6. EXECUTION LOOP
# ==========================================
actor = MAPPOActor2DKS(n_agents=N_AGENTS)
critic = MAPPOCritic2DKS(n_agents=N_AGENTS)

key, act_k, val_k = jax.random.split(key, 3)

# Initializer Shapes Enforce CTDE
dummy_u_global = jnp.zeros((1, N_GRID, N_GRID))
dummy_obs_local = build_marl_obs_batch(dummy_u_global)

total_rollout_steps = NUM_PARALLEL_ENVS * ROLLOUT_STEPS
optax_steps_per_update = (total_rollout_steps // MINIBATCH_SIZE) * PPO_EPOCHS
total_optax_steps = TOTAL_UPDATES * optax_steps_per_update

lr_schedule = optax.linear_schedule(init_value=LR, end_value=0.0, transition_steps=total_optax_steps)

actor_state = TrainState.create(
    apply_fn=actor.apply,
    params=actor.init(act_k, dummy_obs_local), # ACTOR -> Local
    tx=optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr_schedule, eps=1e-5))
)

critic_state = TrainState.create(
    apply_fn=critic.apply,
    params=critic.init(val_k, dummy_u_global), # CRITIC -> Global
    tx=optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr_schedule, eps=1e-5))
)

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

initial_runner_state = (
    actor_state, critic_state, u_batch, obs_batch,
    jnp.zeros(NUM_PARALLEL_ENVS), key
)

print(f"Starting Massively Parallel CTDE MAPPO Training (Chunked & JITed)...")
start_time = time.time()

runner_state = initial_runner_state
num_chunks = TOTAL_UPDATES // EVAL_INT

for chunk_idx in trange(num_chunks):
    # Evaluate BEFORE the chunk
    eval_u = state_bank[0]
    eval_energy, crashed = fast_eval_episode(runner_state[0].params, eval_u, MAX_ENV_STEPS)
    
    current_update = chunk_idx * EVAL_INT
    status = "[CRASHED]" if crashed else f"{eval_energy:.6f}"
    print(f"\nUpdate {current_update:04d} | Eval Energy: {status} | Time: {time.time()-start_time:.1f}s")
    
    # Run EVAL_INT updates instantly on the GPU
    runner_state, chunk_metrics = train_chunk(runner_state, state_bank)

# Final evaluation after all chunks finish
eval_u = state_bank[0]
eval_energy, crashed = fast_eval_episode(runner_state[0].params, eval_u, MAX_ENV_STEPS)
status = "[CRASHED]" if crashed else f"{eval_energy:.6f}"
print(f"\nUpdate {TOTAL_UPDATES:04d} | Eval Energy: {status} | Time: {time.time()-start_time:.1f}s")

actor_state_final, critic_state_final = runner_state[0], runner_state[1]
models_dir = Path('models')
models_dir.mkdir(exist_ok=True)
with open(models_dir / 'mappo_ks2d_params.msgpack', 'wb') as f:
    f.write(flax.serialization.to_bytes({'actor': actor_state_final.params, 'critic': critic_state_final.params}))
print(f"Training finished in {time.time()-start_time:.1f}s. Weights saved.")