import jax
import jax.numpy as jnp
import optax
import flax.serialization
from flax.training.train_state import TrainState
import numpy as np
import time
from pathlib import Path
import sys
from functools import partial
from tqdm import trange

# Add project root
script_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(script_dir))

from examples.ks1d.decentralized.data_utils import evolve_to_attractor
from examples.ks1d.decentralized.dynamics_dual import PDEDynamics 
from models_mappo import MAPPOActor1D, MAPPOCritic1D

# --- Configurations ---
N_AGENTS = 8
L_DOMAIN = 22.0
N_GRID = 128
STATE_NORM_FACTOR = 5.0 

NUM_PARALLEL_ENVS = 256
ROLLOUT_STEPS = 128
PPO_EPOCHS = 4
MINIBATCH_SIZE = 1024
TOTAL_TIMESTEPS = 50_000_000

# PPO Hyperparameters
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.01
LR = 3e-4

# --- Initialization ---
key = jax.random.PRNGKey(42)

def direct_control_policy(action_params, u_obs, u_target, xi_fixed):
    return action_params

dynamics = PDEDynamics(policy_apply_fn=direct_control_policy)
xi_fixed = jnp.linspace(0.0, L_DOMAIN, N_AGENTS, endpoint=False) + (L_DOMAIN/N_AGENTS)/2

# --- Environment Handlers ---
@jax.jit
def build_marl_obs_batch(u_batch, xi_fixed):
    """Creates a decentralized observation: normalized global state + agent's normalized position."""
    batch_size = u_batch.shape[0]
    u_norm = u_batch / STATE_NORM_FACTOR
    
    # Tile the global state for each agent
    u_tiled = jnp.tile(u_norm[:, None, :], (1, N_AGENTS, 1))
    
    # Tile the positions for each batch and normalize them
    xi_tiled = jnp.tile(xi_fixed[None, :, None], (batch_size, 1, 1)) / L_DOMAIN
    
    return jnp.concatenate([u_tiled, xi_tiled], axis=-1)

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
    
    # Calculate global tracking reward and broadcast to all agents
    global_reward = -(jnp.mean(safe_u**2, axis=-1, keepdims=True) / 10.0)
    rewards_batch = jnp.tile(global_reward, (1, N_AGENTS))
    dones_batch_exp = jnp.tile(dones_batch, (1, N_AGENTS))
    
    return safe_u, rewards_batch, dones_batch_exp

# --- PPO Core Functions ---
def get_action_and_value(actor_params, critic_params, obs, global_u, key):
    mean, log_std = actor.apply(actor_params, obs)
    
    std = jnp.exp(log_std)
    action = mean + std * jax.random.normal(key, mean.shape)
    log_prob = -0.5 * jnp.sum(jnp.square((action - mean) / std) + 2 * log_std + jnp.log(2 * jnp.pi), axis=-1)
    
    val = critic.apply(critic_params, global_u / STATE_NORM_FACTOR)
    return action, log_prob, val

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

# --- THE PURE JAX MAPPO TRAIN STEP ---
@jax.jit
def train_step(runner_state, state_bank):
    actor_state, critic_state, u_batch, obs_batch, env_counts, rng = runner_state
    
    # 1. Rollout Phase
    def _env_step(carry, _):
        u, obs, counts, k = carry
        k, act_k, reset_k = jax.random.split(k, 3)
        
        action, log_prob, val = get_action_and_value(actor_state.params, critic_state.params, obs, u, act_k)
        
        # Squeeze action for physics step (batch, n_agents, 1) -> (batch, n_agents)
        env_action = jnp.clip(action.squeeze(-1), -1.0, 1.0)
        
        next_u, rewards, crashes = parallel_physics_step(u, env_action, xi_fixed)
        
        counts += 1
        truncs = counts >= ROLLOUT_STEPS
        
        # Collapse crashes back to (batch,) for resetting
        needs_reset = jnp.logical_or(crashes[:, 0], truncs)
        
        fresh_states = jax.random.choice(reset_k, state_bank, shape=(NUM_PARALLEL_ENVS,))
        
        next_u_final = jnp.where(needs_reset[:, None], fresh_states, next_u)
        next_counts = jnp.where(needs_reset, 0, counts)
        obs_new = build_marl_obs_batch(next_u_final, xi_fixed)
        
        true_next_v = critic.apply(critic_state.params, next_u / STATE_NORM_FACTOR)
        
        transition = (obs, u, action, log_prob, rewards, val, crashes, true_next_v)
        return (next_u_final, obs_new, next_counts, k), transition

    carry = (u_batch, obs_batch, env_counts, rng)
    carry, transitions = jax.lax.scan(_env_step, carry, None, length=ROLLOUT_STEPS)
    (next_u_batch, next_obs_batch, next_env_counts, rng) = carry
    t_obs, t_u, t_a, t_logp, t_r, t_v, t_d, t_true_next_v = transitions
    
    # 2. GAE Phase
    last_val = critic.apply(critic_state.params, next_u_batch / STATE_NORM_FACTOR)
    
    def gae_scan_fn(carry, transition):
        r, v, d_exp, true_next_v = transition
        gae, _ = carry
        nextnonterminal = 1.0 - d_exp
        delta = r + GAMMA * true_next_v * nextnonterminal - v
        gae = delta + GAMMA * GAE_LAMBDA * nextnonterminal * gae
        return (gae, v), gae
        
    _, adv = jax.lax.scan(
        gae_scan_fn, 
        (jnp.zeros((NUM_PARALLEL_ENVS, N_AGENTS)), last_val), 
        (t_r, t_v, t_d, t_true_next_v), 
        reverse=True
    )
    ret = adv + t_v
    
    obs_dim = t_obs.shape[-1]
    f_obs = t_obs.reshape(-1, N_AGENTS, obs_dim)  
    f_u = t_u.reshape(-1, N_GRID)
    f_a = t_a.reshape(-1, N_AGENTS, 1)
    f_logp = t_logp.reshape(-1, N_AGENTS)
    f_ret = ret.reshape(-1, N_AGENTS)
    f_adv = adv.reshape(-1, N_AGENTS)
    f_v = t_v.reshape(-1, N_AGENTS)

    # Global normalization of adv (Shared across all agents)
    f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)
    dataset_size = f_u.shape[0]

    # 3. Optimization Phase
    def _update_epoch(epoch_carry, _):
        a_state, c_state, k = epoch_carry
        k, subk = jax.random.split(k)
        indices = jax.random.permutation(subk, dataset_size)
        
        def _update_minibatch(mb_carry, start_idx):
            a_state_, c_state_ = mb_carry
            batch_idx = jax.lax.dynamic_slice(indices, (start_idx,), (MINIBATCH_SIZE,))
            
            mb_batch = (
                f_obs[batch_idx], f_u[batch_idx], f_a[batch_idx], 
                f_logp[batch_idx], f_adv[batch_idx], f_ret[batch_idx], f_v[batch_idx]
            )
            
            a_state_n, c_state_n, al, vl, ent = ppo_update_epoch(a_state_, c_state_, mb_batch)
            return (a_state_n, c_state_n), jnp.stack([al, vl, ent])
            
        mb_starts = jnp.arange(0, dataset_size, MINIBATCH_SIZE)
        (a_state, c_state), epoch_metrics = jax.lax.scan(_update_minibatch, (a_state, c_state), mb_starts)
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

# --- Fast Evaluation ---
@partial(jax.jit, static_argnames=['max_steps'])
def fast_eval_episode(actor_params, init_u, max_steps):
    def step_fn(state, _):
        u_curr = state
        obs = build_marl_obs_batch(u_curr[None, ...], xi_fixed)
        
        mean, _ = actor.apply(actor_params, obs)
        act_flat = mean.squeeze(0).squeeze(-1) # (1, n_agents, 1) -> (n_agents,)
        
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


# --- Setup & Training Loop ---
actor = MAPPOActor1D(n_agents=N_AGENTS)
critic = MAPPOCritic1D(n_agents=N_AGENTS)

key, act_k, val_k = jax.random.split(key, 3)

dummy_obs_full = jnp.zeros((1, N_AGENTS, N_GRID + 1))
dummy_u = jnp.zeros((1, N_GRID))

total_rollout_steps = NUM_PARALLEL_ENVS * ROLLOUT_STEPS
num_updates = TOTAL_TIMESTEPS // total_rollout_steps
optax_steps_per_update = (total_rollout_steps // MINIBATCH_SIZE) * PPO_EPOCHS
total_optax_steps = num_updates * optax_steps_per_update

lr_schedule = optax.linear_schedule(init_value=LR, end_value=0.0, transition_steps=total_optax_steps)

actor_state = TrainState.create(
    apply_fn=actor.apply,
    params=actor.init(act_k, dummy_obs_full),
    tx=optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr_schedule, eps=1e-5))
)

critic_state = TrainState.create(
    apply_fn=critic.apply,
    params=critic.init(val_k, dummy_u),
    tx=optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr_schedule, eps=1e-5))
)

print("Pre-generating starting state bank (Vectorized)...")
bank_keys = jax.random.split(key, 1000)
state_bank = jax.vmap(lambda k: evolve_to_attractor(k, N_GRID, L_DOMAIN))(bank_keys)

key, subkey = jax.random.split(key)
u_batch = jax.random.choice(subkey, state_bank, shape=(NUM_PARALLEL_ENVS,))

initial_runner_state = (
    actor_state, critic_state, u_batch, build_marl_obs_batch(u_batch, xi_fixed),
    jnp.zeros(NUM_PARALLEL_ENVS), key
)

print(f"Starting Pure JAX MAPPO 1D Training for {num_updates} Updates...")
start_time = time.time()

runner_state = initial_runner_state

for update in trange(num_updates):
    if update % 10 == 0:
        eval_u = state_bank[0]
        # max_steps argument is passed positionally here as well
        eval_energy, crashed = fast_eval_episode(runner_state[0].params, eval_u, ROLLOUT_STEPS)
        
        status = "[CRASHED]" if crashed else f"{eval_energy:.6f}"
        print(f"\nUpdate {update:04d} | Eval Energy: {status} | Time: {time.time()-start_time:.1f}s")
        if update > 0:
            print(f"Metrics -> Ret: {metrics['mean_return']:.2f} | Val L: {metrics['critic_loss']:.4f} | Act L: {metrics['actor_loss']:.4f} | Ent: {metrics['entropy']:.4f}")

    runner_state, metrics = train_step(runner_state, state_bank)

# Save output
actor_state_final, critic_state_final = runner_state[0], runner_state[1]
with open('models/mappo_ks1d_params.msgpack', 'wb') as f:
    f.write(flax.serialization.to_bytes({'actor': actor_state_final.params, 'critic': critic_state_final.params}))
print(f"Training finished in {time.time()-start_time:.1f}s. Weights saved.")