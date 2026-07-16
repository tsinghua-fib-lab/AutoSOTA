import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
import flax.serialization
import numpy as np
import time
from pathlib import Path
import sys
from functools import partial
from tqdm import trange

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(script_dir))

# Project imports
from examples.ks1d.decentralized.data_utils import evolve_to_attractor
from examples.ks1d.decentralized.dynamics_dual import PDEDynamics 
from examples.ks1d.decentralized.bench.models_ppo import PPOActor1D, PPOCritic1D

# --- Configurations ---
N_AGENTS = 8
L_DOMAIN = 22.0
N_GRID = 128
STATE_NORM_FACTOR = 5.0 
MAX_ENV_STEPS = 200
NUM_PARALLEL_ENVS = 256

# PPO Specific Configs
ROLLOUT_STEPS = 128
PPO_EPOCHS = 4
MINIBATCH_SIZE = 1024
TOTAL_UPDATES = 1000
EVAL_INT = 10 

key = jax.random.PRNGKey(42)

def direct_control_policy(action_params, u_obs, u_target, xi_fixed):
    return action_params

dynamics = PDEDynamics(policy_apply_fn=direct_control_policy)
xi_fixed = jnp.linspace(0.0, L_DOMAIN, N_AGENTS, endpoint=False) + (L_DOMAIN/N_AGENTS)/2

# --- Gaussian Action Utils ---
def get_logprob_and_action(mean, log_std, key=None, action=None):
    std = jnp.exp(log_std)
    if action is None:
        noise = jax.random.normal(key, mean.shape)
        action = mean + noise * std
        action = jnp.clip(action, -1.0, 1.0)
    
    var = std ** 2
    log_prob = -0.5 * ((action - mean) ** 2) / var - log_std - 0.5 * jnp.log(2 * jnp.pi)
    return action, jnp.sum(log_prob, axis=-1)

# --- Initialization ---
actor = PPOActor1D(n_agents=N_AGENTS)
critic = PPOCritic1D()

key, *subkeys = jax.random.split(key, 4)
dummy_u = jnp.zeros((1, N_GRID))

actor_params = actor.init(subkeys[0], dummy_u)
critic_params = critic.init(subkeys[1], dummy_u)

tx_actor = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(3e-4))
tx_critic = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(1e-3))
opt_actor = tx_actor.init(actor_params)
opt_critic = tx_critic.init(critic_params)

# --- JAX-Native GAE ---
@jax.jit
def compute_gae_jax(rewards, values, dones, true_next_values, last_val, gamma=0.99, lam=0.95):
    def scan_fn(carry, transition):
        r, v, d, true_next_v = transition
        gae, _ = carry
        delta = r + gamma * true_next_v * (1.0 - d) - v
        gae = delta + gamma * lam * (1.0 - d) * gae
        return (gae, v), gae
    
    _, advantages = jax.lax.scan(
        scan_fn, 
        (jnp.zeros_like(last_val), last_val), 
        (rewards, values, dones, true_next_values), 
        reverse=True
    )
    returns = advantages + values
    return advantages, returns

# --- Parallel Physics Step (1D) ---
def parallel_physics_step(u_batch, actions, xi_fixed):
    def single_physics_step(u_s, act_s):
        traj = dynamics.unroll_controlled(
            u_s, xi_fixed, jnp.zeros(N_GRID), act_s, 
            1, N_grid=N_GRID, L=L_DOMAIN
        )
        return traj[0][-1]
    
    next_u_batch = jax.vmap(single_physics_step)(u_batch, actions)
    
    is_invalid = jnp.logical_not(jnp.isfinite(next_u_batch).all(axis=-1, keepdims=True))
    is_exploding = jnp.max(jnp.abs(next_u_batch), axis=-1, keepdims=True) > 100.0
    dones_batch = jnp.logical_or(is_invalid, is_exploding)
    
    safe_u = jnp.where(dones_batch, jnp.zeros_like(next_u_batch), next_u_batch)
    rewards_batch = -(jnp.mean(safe_u**2, axis=-1, keepdims=True) / 10.0)
    
    return safe_u, rewards_batch, dones_batch

# --- Minibatch Update Logic ---
def update_ppo_minibatch(a_params, c_params, opt_a, opt_c, b_u, b_a, b_logp, b_ret, b_adv):
    b_u_norm = b_u / STATE_NORM_FACTOR

    def loss_fn(ap, cp):
        mean, log_std = actor.apply(ap, b_u_norm)
        _, new_logp = get_logprob_and_action(mean, log_std, action=b_a)
        
        entropy = jnp.sum(log_std + 0.5 + 0.5 * jnp.log(2 * jnp.pi), axis=-1).mean()
        
        ratio = jnp.exp(new_logp - b_logp)
        pg_loss1 = -b_adv * ratio
        pg_loss2 = -b_adv * jnp.clip(ratio, 1.0 - 0.2, 1.0 + 0.2)
        actor_loss = jnp.maximum(pg_loss1, pg_loss2).mean() - 0.01 * entropy
        
        values = critic.apply(cp, b_u_norm).squeeze(-1)
        critic_loss = 0.5 * jnp.mean((b_ret - values) ** 2)
        
        return actor_loss + 0.5 * critic_loss, (actor_loss, critic_loss, entropy)

    (total_loss, metrics), grads = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(a_params, c_params)
    
    up_a, opt_a = tx_actor.update(grads[0], opt_a)
    up_c, opt_c = tx_critic.update(grads[1], opt_c)
    
    return optax.apply_updates(a_params, up_a), optax.apply_updates(c_params, up_c), opt_a, opt_c, metrics

# --- THE PURE JAX PPO TRAIN STEP ---
def train_step(runner_state, state_bank):
    a_params, c_params, opt_a, opt_c, u_batch, env_counts, rng = runner_state
    
    # 1. Rollout Phase
    def _env_step(carry, _):
        u, counts, k = carry
        k, act_k, reset_k = jax.random.split(k, 3)
        
        u_norm = u / STATE_NORM_FACTOR
        mean, log_std = actor.apply(a_params, u_norm)
        actions, log_probs = get_logprob_and_action(mean, log_std, key=act_k)
        values = critic.apply(c_params, u_norm).squeeze(-1)
        
        next_u, rewards, dones = parallel_physics_step(u, actions, xi_fixed)
        
        counts += 1
        truncs = counts >= MAX_ENV_STEPS
        needs_reset = jnp.logical_or(dones.flatten(), truncs)
        
        fresh_states = jax.random.choice(reset_k, state_bank, shape=(NUM_PARALLEL_ENVS,))
        
        # Get the value of the true next state BEFORE overwriting it with resets
        next_u_norm = next_u / STATE_NORM_FACTOR
        true_next_v = critic.apply(c_params, next_u_norm).squeeze(-1)

        next_u_final = jnp.where(needs_reset[:, None], fresh_states, next_u)
        next_counts = jnp.where(needs_reset, 0, counts)
        
        transition = (u, actions, rewards.squeeze(-1), values, log_probs, dones.squeeze(-1), true_next_v)
        return (next_u_final, next_counts, k), transition

    carry = (u_batch, env_counts, rng)
    carry, transitions = jax.lax.scan(_env_step, carry, None, length=ROLLOUT_STEPS)
    (next_u_batch, next_env_counts, rng) = carry
    t_u, t_a, t_r, t_v, t_logp, t_d, t_true_next_v = transitions
    
    # 2. GAE Phase
    last_val = critic.apply(c_params, next_u_batch / STATE_NORM_FACTOR).squeeze(-1)
    adv, ret = compute_gae_jax(t_r, t_v, t_d, t_true_next_v, last_val)
    
    # Flatten across time and envs
    f_u = t_u.reshape(-1, N_GRID)
    f_a = t_a.reshape(-1, N_AGENTS)
    f_logp = t_logp.reshape(-1) 
    f_ret = ret.reshape(-1)
    f_adv = adv.reshape(-1)
    
    # Normalize advantages across the ENTIRE batch here
    f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)
    dataset_size = f_u.shape[0]

    # 3. Optimization Phase
    def _update_epoch(epoch_carry, _):
        ap, cp, oa, oc, k = epoch_carry
        k, subk = jax.random.split(k)
        indices = jax.random.permutation(subk, dataset_size)
        
        def _update_minibatch(mb_carry, start_idx):
            ap_, cp_, oa_, oc_ = mb_carry
            batch_idx = jax.lax.dynamic_slice(indices, (start_idx,), (MINIBATCH_SIZE,))
            
            ap_n, cp_n, oa_n, oc_n, metrics = update_ppo_minibatch(
                ap_, cp_, oa_, oc_, 
                f_u[batch_idx], f_a[batch_idx], f_logp[batch_idx], 
                f_ret[batch_idx], f_adv[batch_idx]
            )
            return (ap_n, cp_n, oa_n, oc_n), metrics
            
        mb_starts = jnp.arange(0, dataset_size, MINIBATCH_SIZE)
        (ap, cp, oa, oc), epoch_metrics = jax.lax.scan(_update_minibatch, (ap, cp, oa, oc), mb_starts)
        return (ap, cp, oa, oc, k), epoch_metrics

    epoch_carry = (a_params, c_params, opt_a, opt_c, rng)
    epoch_carry, ppo_metrics = jax.lax.scan(_update_epoch, epoch_carry, None, length=PPO_EPOCHS)
    (a_params, c_params, opt_a, opt_c, rng) = epoch_carry

    new_runner_state = (a_params, c_params, opt_a, opt_c, next_u_batch, next_env_counts, rng)
    
    metrics = {
        "mean_return": t_r.sum(axis=0).mean(),
        "actor_loss": ppo_metrics[0].mean(),
        "critic_loss": ppo_metrics[1].mean()
    }
    
    return new_runner_state, metrics

# --- SCAN-COMPILED TRAINING CHUNK ---
@jax.jit
def train_chunk(runner_state, state_bank):
    def scan_step(carry, _):
        new_state, metrics = train_step(carry, state_bank)
        return new_state, metrics
    
    return jax.lax.scan(scan_step, runner_state, None, length=EVAL_INT)

# --- Fast Evaluation ---
@partial(jax.jit, static_argnames=['max_steps'])
def fast_eval_episode(a_params, init_u, max_steps):
    def step_fn(state, _):
        u_curr = state
        u_norm = u_curr / STATE_NORM_FACTOR
        
        mean, _ = actor.apply(a_params, u_norm[None, ...])
        act_flat = mean.squeeze(0)
        
        # FIX: Pass arguments positionally just like in parallel_physics_step
        traj = dynamics.unroll_controlled(
            u_curr, xi_fixed, jnp.zeros(N_GRID), act_flat, 
            1, N_grid=N_GRID, L=L_DOMAIN
        )
        next_u = traj[0][-1]
        
        energy = jnp.mean(next_u**2)
        crashed = jnp.isnan(next_u).any() | jnp.isinf(next_u).any() | (jnp.max(jnp.abs(next_u)) > 100.0)
        return next_u, (energy, crashed)

    _, (energies, crashes) = jax.lax.scan(step_fn, init_u, None, length=max_steps)
    return jnp.mean(energies), jnp.any(crashes)

# --- Python Execution Loop ---
print("Pre-generating starting state bank (Vectorized)...")
bank_keys = jax.random.split(key, 1000)
state_bank = jax.vmap(lambda k: evolve_to_attractor(k, N_GRID, L_DOMAIN))(bank_keys)

key, subkey = jax.random.split(key)
u_batch = jax.random.choice(subkey, state_bank, shape=(NUM_PARALLEL_ENVS,))

initial_runner_state = (
    actor_params, critic_params, opt_actor, opt_critic, 
    u_batch, jnp.zeros(NUM_PARALLEL_ENVS), key
)

print("Starting Massively Parallel Pure JAX PPO Training (Chunked 1D)...")
start_time = time.time()

runner_state = initial_runner_state
num_chunks = TOTAL_UPDATES // EVAL_INT

for chunk in trange(num_chunks):
    current_update = chunk * EVAL_INT
    
    # Evaluate at the start of the chunk
    eval_u = state_bank[0]
    eval_energy, crashed = fast_eval_episode(runner_state[0], eval_u, MAX_ENV_STEPS)
    
    status = "[CRASHED]" if crashed else f"{eval_energy:.6f}"
    print(f"Update {current_update:04d} | Eval Energy: {status} | Time: {time.time()-start_time:.1f}s")

    # Run the compiled chunk
    runner_state, batch_metrics = train_chunk(runner_state, state_bank)

# Save output
actor_params_final = runner_state[0]
with open('models/ppo_ks1d_params.msgpack', 'wb') as f:
    f.write(flax.serialization.to_bytes({'actor': actor_params_final}))
print(f"Training finished in {time.time()-start_time:.1f}s. Weights saved.")