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

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(script_dir))

# Updated Imports for FKPP
from env_fkpp import FKPPHypeMARLEnv 
from utils_hypemarl import get_sinusoidal_encoding
from examples.fkpp1d.decentralized.data_utils import generate_grf
from examples.fkpp1d.decentralized.dynamics_dual import PDEDynamics 
from models_mappo import MAPPOActor, MAPPOCritic, V_MAX, U_MAX

# --- PPO Configurations ---
N_AGENTS = 20 
L_DOMAIN = 1.0
N_GRID = 100

NUM_PARALLEL_ENVS = 256
ROLLOUT_STEPS = 300      # Match episode length
PPO_EPOCHS = 4
MINIBATCH_SIZE = 1024
TOTAL_TIMESTEPS = 50_000_000

# FKPP Constants
NU = 0.005
RHO = 3.0

# PPO Hyperparameters
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.01
VF_COEF = 0.5
LR = 3e-4

# --- Initialization ---
key = jax.random.PRNGKey(42)

def direct_control_policy(action_params, u_obs, u_target, xi_fixed):
    u = action_params[:, 0]
    v = action_params[:, 1]
    return u, v

dynamics = PDEDynamics(policy_apply_fn=direct_control_policy)
env = FKPPHypeMARLEnv(dynamics, n_agents=N_AGENTS, N_grid=N_GRID, L=L_DOMAIN, max_steps=ROLLOUT_STEPS)

pe_dim = 128
stored_obs_dim = env.local_y_dim + env.n_mu 
total_input_dim = stored_obs_dim + pe_dim
mu_jax = jnp.array(env.mu)
window_size = env.window_size

# --- Helper: On-The-Fly Positional Encoding ---
@jax.jit
def attach_pe_batch(obs_no_pe, xi_batch):
    def single_env_pe(obs_env, xi_env):
        pe = get_sinusoidal_encoding(xi_env, d=pe_dim)
        return jnp.concatenate([obs_env, pe], axis=-1)
    return jax.vmap(single_env_pe)(obs_no_pe, xi_batch)

# --- PPO Core Functions ---
def get_action_and_value(actor_params, critic_params, obs_no_pe, z, target, xi, key):
    full_obs = attach_pe_batch(obs_no_pe, xi)
    
    mean, log_std = actor.apply(actor_params, full_obs)
    std = jnp.exp(log_std)
    
    action = mean + std * jax.random.normal(key, mean.shape)
    log_prob = -0.5 * jnp.sum(jnp.square((action - mean) / std) + 2 * log_std + jnp.log(2 * jnp.pi), axis=-1)
    
    val = critic.apply(critic_params, z, target, xi)
    return action, log_prob, val

def get_value(critic_params, z, target, xi):
    return critic.apply(critic_params, z, target, xi)

# --- Updated: True Next Value GAE Calculation ---
def compute_gae_jax(rewards, values, dones, true_next_values, last_val, gamma=0.99, lam=0.95):
    def scan_fn(carry, transition):
        r, v, d, true_next_v = transition
        
        # Changes shape from (256,) to (256, 1) to broadcast across agents
        d = d[:, None] 
        
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

# --- Updated: Value Clipping included ---
def ppo_update_epoch(actor_state, critic_state, batch):
    obs_no_pe, z, target, xi, actions, old_log_probs, advantages, returns, old_values = batch
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    def actor_loss_fn(params):
        full_obs = attach_pe_batch(obs_no_pe, xi)
        mean, log_std = actor.apply(params, full_obs)
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
        v = critic.apply(params, z, target, xi)
        
        # Value clipping limits how much the value function can change per epoch
        v_clipped = old_values + jnp.clip(v - old_values, -CLIP_EPS, CLIP_EPS)
        v_loss_unclipped = jnp.square(v - returns)
        v_loss_clipped = jnp.square(v_clipped - returns)
        
        return 0.5 * jnp.maximum(v_loss_unclipped, v_loss_clipped).mean()
        
    (a_loss, aux), a_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_state.params)
    v_loss, v_grads = jax.value_and_grad(value_loss_fn)(critic_state.params)
    
    actor_state = actor_state.apply_gradients(grads=a_grads)
    critic_state = critic_state.apply_gradients(grads=v_grads)
    
    return actor_state, critic_state, a_loss, v_loss, aux[1]

# --- Environment Handlers ---
@partial(jax.jit, static_argnames=['window_size'])
def extract_patches_jax(full_state, target_st, xi_n, window_size):
    error = full_state - target_st
    error_grad = jnp.gradient(error)
    n_pde = full_state.shape[0]
    half_window = window_size // 2

    padded_error = jnp.pad(error, (half_window, half_window), mode='constant', constant_values=0.0)
    padded_grad = jnp.pad(error_grad, (half_window, half_window), mode='constant', constant_values=0.0)

    def get_local_obs(xi):
        center_idx = jax.lax.stop_gradient((xi * (n_pde - 1)).astype(int)) + half_window
        start = center_idx - half_window
        p_err = jax.lax.dynamic_slice(padded_error, (start,), (window_size,))
        p_grad = jax.lax.dynamic_slice(padded_grad, (start,), (window_size,))
        p_err = jax.image.resize(p_err, (20,), method='bilinear')
        p_grad = jax.image.resize(p_grad, (20,), method='bilinear')
        return jnp.concatenate([p_err, p_grad])

    return jax.vmap(get_local_obs)(xi_n)

def build_marl_obs_batch(z_batch, target_batch, xi_batch):
    def single_env_obs(state, target, xi):
        y_local = extract_patches_jax(state, target, xi, window_size)
        mu_broadcast = jnp.tile(mu_jax, (N_AGENTS, 1))
        return jnp.concatenate([y_local, mu_broadcast], axis=-1)
    return jax.vmap(single_env_obs)(z_batch, target_batch, xi_batch)

def parallel_marl_physics_step(z_batch, xi_batch, target_batch, actions, key):
    keys = jax.random.split(key, z_batch.shape[0])
    
    def single_physics_step(z_s, xi_s, target_s, act_s, k_s):
        traj = dynamics.unroll_controlled(
            z_init=z_s, xi_init=xi_s, z_target=target_s, params=act_s, 
            t_steps=1, key=k_s, nu=NU, rho=RHO
        )
        return traj[0][-1], traj[1][-1]
    
    next_z_batch, next_xi_batch = jax.vmap(single_physics_step)(z_batch, xi_batch, target_batch, actions, keys)
    
    is_invalid = jnp.logical_not(jnp.isfinite(next_z_batch).all(axis=-1, keepdims=True))
    dones_batch = is_invalid.squeeze(-1)
    
    safe_z = jnp.where(is_invalid, jnp.zeros_like(next_z_batch), next_z_batch)
    safe_xi = jnp.where(is_invalid, xi_batch, next_xi_batch)
    
    next_obs_batch_no_pe = build_marl_obs_batch(safe_z, target_batch, safe_xi)
    
    u_batch = actions[..., 0]
    v_batch = actions[..., 1]
    
    # 1. Combined Local + Global Tracking Reward
    agent_indices = jnp.clip((safe_xi * (N_GRID - 1)).astype(jnp.int32), 0, N_GRID - 1)
    batch_indices = jnp.arange(safe_z.shape[0])[:, None]
    
    local_z = safe_z[batch_indices, agent_indices]
    local_target = target_batch[batch_indices, agent_indices]
    
    local_mse = jnp.square(local_z - local_target)
    global_mse = jnp.mean(jnp.square(safe_z - target_batch), axis=1)[:, None]
    
    r_track_combined = -50.0 * local_mse - 10.0 * global_mse 

    # 2. Effort Penalty
    r_effort = -0.001 * (jnp.square(u_batch) + 0.1 * jnp.square(v_batch))
    
    # 3. Boundary Margin Penalty
    margin = 0.02
    r_bound = -10.0 * (jnp.maximum(0.0, margin - safe_xi)**2 + jnp.maximum(0.0, safe_xi - (1.0 - margin))**2)
    
    # 4. Collision Penalty
    R_safe = 0.02 
    dists = jnp.abs(safe_xi[:, :, None] - safe_xi[:, None, :])
    mask = jnp.eye(N_AGENTS)[None, :, :]
    r_coll = -1.0 * jnp.sum(jnp.maximum(0.0, R_safe - (dists + mask * 1.0)) ** 2, axis=2)
    
    # Combine the local tracking reward with local penalties
    rewards_batch = r_track_combined + r_effort + r_bound + r_coll
    
    # Apply massive penalty for crashing
    rewards_batch = jnp.where(dones_batch[:, None], -100.0, rewards_batch)
    
    return safe_z, safe_xi, next_obs_batch_no_pe, rewards_batch, dones_batch

# --- THE PURE JAX MAPPO TRAIN STEP ---
@jax.jit
def train_step(runner_state):
    """Executes a full rollout AND all optimization epochs entirely on the GPU."""
    actor_state, critic_state, z_batch, target_batch, xi_batch, obs_batch, env_counts, rng = runner_state
    
    # 1. Rollout Phase (Compiled loop)
    def _env_step(carry, _):
        z, zt, xi, obs, counts, k = carry
        k, act_k, phys_k, reset_k = jax.random.split(k, 4)
        
        action, log_prob, val = get_action_and_value(actor_state.params, critic_state.params, obs, z, zt, xi, act_k)

        env_action_u = jnp.clip(action[..., 0], -U_MAX, U_MAX)
        env_action_v = jnp.clip(action[..., 1], -V_MAX, V_MAX)
        env_action = jnp.stack([env_action_u, env_action_v], axis=-1)
        
        next_z, next_xi, next_obs_no_pe, rewards, crashes = parallel_marl_physics_step(
            z, xi, zt, env_action, phys_k
        )
        
        counts += 1
        truncs = counts >= ROLLOUT_STEPS
        needs_reset = jnp.logical_or(crashes, truncs)
        
        idx_reset = jax.random.randint(reset_k, (NUM_PARALLEL_ENVS,), 0, 1000)
        fresh_z = z_init_bank[idx_reset]
        fresh_target = z_target_bank[idx_reset]
        fresh_xi = jnp.tile(xi_init_single, (NUM_PARALLEL_ENVS, 1))
        
        next_z = jnp.where(needs_reset[:, None], fresh_z, next_z)
        next_zt = jnp.where(needs_reset[:, None], fresh_target, zt)
        next_xi = jnp.where(needs_reset[:, None], fresh_xi, next_xi)
        next_counts = jnp.where(needs_reset, 0, counts)
        
        # Build new obs if the environment was reset
        obs_new = jnp.where(
            needs_reset[:, None, None], 
            build_marl_obs_batch(next_z, next_zt, next_xi), 
            next_obs_no_pe
        )
        
        # Grab true next value BEFORE the next state gets overwritten
        true_next_v = get_value(critic_state.params, next_z, zt, next_xi)
        
        transition = (obs, z, zt, xi, action, log_prob, rewards, val, crashes, true_next_v)
        return (next_z, next_zt, next_xi, obs_new, next_counts, k), transition

    carry = (z_batch, target_batch, xi_batch, obs_batch, env_counts, rng)
    carry, transitions = jax.lax.scan(_env_step, carry, None, length=ROLLOUT_STEPS)
    (next_z_batch, next_target_batch, next_xi_batch, next_obs_batch, next_env_counts, rng) = carry
    t_obs, t_z, t_zt, t_xi, t_a, t_logp, t_r, t_v, t_d, t_true_next_v = transitions
    
    # 2. GAE Phase
    last_val = get_value(critic_state.params, next_z_batch, next_target_batch, next_xi_batch)
    adv, ret = compute_gae_jax(t_r, t_v, t_d, t_true_next_v, last_val, GAMMA, GAE_LAMBDA)
    
    # Flatten across time and envs
    f_obs = t_obs.reshape(-1, N_AGENTS, stored_obs_dim)  
    f_z = t_z.reshape(-1, N_GRID)
    f_zt = t_zt.reshape(-1, N_GRID)
    f_xi = t_xi.reshape(-1, N_AGENTS)
    f_a = t_a.reshape(-1, N_AGENTS, 2)
    f_logp = t_logp.reshape(-1, N_AGENTS)
    f_ret = ret.reshape(-1, N_AGENTS)
    f_adv = adv.reshape(-1, N_AGENTS)
    f_v = t_v.reshape(-1, N_AGENTS)  # Flatten old values for clipping

    # Global normalization of adv
    f_adv = (f_adv - f_adv.mean(axis=0)) / (f_adv.std(axis=0) + 1e-8)
    
    dataset_size = f_z.shape[0]

    # 3. Optimization Phase (Compiled nested loops)
    def _update_epoch(epoch_carry, _):
        a_state, c_state, k = epoch_carry
        k, subk = jax.random.split(k)
        indices = jax.random.permutation(subk, dataset_size)
        
        def _update_minibatch(mb_carry, start_idx):
            a_state_, c_state_ = mb_carry
            batch_idx = jax.lax.dynamic_slice(indices, (start_idx,), (MINIBATCH_SIZE,))
            
            mb_batch = (
                f_obs[batch_idx], f_z[batch_idx], f_zt[batch_idx], f_xi[batch_idx],
                f_a[batch_idx], f_logp[batch_idx], f_adv[batch_idx], f_ret[batch_idx], f_v[batch_idx]
            )
            
            a_state_n, c_state_n, al, vl, ent = ppo_update_epoch(a_state_, c_state_, mb_batch)
            return (a_state_n, c_state_n), jnp.stack([al, vl, ent])
            
        mb_starts = jnp.arange(0, dataset_size, MINIBATCH_SIZE)
        (a_state, c_state), epoch_metrics = jax.lax.scan(_update_minibatch, (a_state, c_state), mb_starts)
        return (a_state, c_state, k), epoch_metrics

    epoch_carry = (actor_state, critic_state, rng)
    epoch_carry, ppo_metrics = jax.lax.scan(_update_epoch, epoch_carry, None, length=PPO_EPOCHS)
    (actor_state, critic_state, rng) = epoch_carry

    new_runner_state = (actor_state, critic_state, next_z_batch, next_target_batch, next_xi_batch, next_obs_batch, next_env_counts, rng)
    
    metrics = {
        "mean_return": t_r.sum(axis=0).mean(),
        "actor_loss": ppo_metrics[..., 0].mean(),
        "critic_loss": ppo_metrics[..., 1].mean(),
        "entropy": ppo_metrics[..., 2].mean()
    }
    
    return new_runner_state, metrics

# --- Fast Evaluation ---
@jax.jit
def get_eval_action(actor_params, obs_no_pe, xi_batch):
    full_obs = attach_pe_batch(obs_no_pe, xi_batch)
    mean, _ = actor.apply(actor_params, full_obs)
    return mean

@partial(jax.jit, static_argnames=['max_steps'])
def fast_eval_episode(actor_params, init_z, init_xi, target_z, max_steps, key):
    def step_fn(state_tuple, _):
        z_curr, xi_curr, k = state_tuple
        k, subk = jax.random.split(k)
        obs_no_pe = build_marl_obs_batch(z_curr[None, ...], target_z[None, ...], xi_curr[None, ...]) 
        
        act = get_eval_action(actor_params, obs_no_pe, xi_curr[None, ...])
        act_flat = act.squeeze(0)
        
        traj = dynamics.unroll_controlled(
            z_init=z_curr, xi_init=xi_curr, z_target=target_z, params=act_flat, 
            t_steps=1, key=subk, nu=NU, rho=RHO
        )
        next_z, next_xi = traj[0][-1], traj[1][-1]
        
        mse = jnp.mean((next_z - target_z)**2)
        crashed = jnp.isnan(next_z).any() | jnp.isinf(next_z).any()
        return (next_z, next_xi, k), (mse, crashed)

    _, (mses, crashes) = jax.lax.scan(step_fn, (init_z, init_xi, key), None, length=max_steps)
    return jnp.mean(mses), jnp.any(crashes)


# --- Setup & Training Loop ---
actor = MAPPOActor(n_agents=N_AGENTS)
critic = MAPPOCritic(n_agents=N_AGENTS)

key, act_k, val_k = jax.random.split(key, 3)

dummy_obs_full = jnp.zeros((1, N_AGENTS, total_input_dim))
dummy_z = jnp.zeros((1, N_GRID))
dummy_target = jnp.zeros((1, N_GRID))
dummy_xi = jnp.zeros((1, N_AGENTS))

# --- Updated: Linear Learning Rate Schedule ---
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
    params=critic.init(val_k, dummy_z, dummy_target, dummy_xi),
    tx=optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr_schedule, eps=1e-5))
)

# --- Pre-generate Starting State Banks globally for JIT access ---
print("Pre-generating starting state & target banks...")
bank_keys = jax.random.split(key, 1000)
_, z_init_bank = jax.vmap(partial(generate_grf, n_points=N_GRID, length_scale=0.2))(bank_keys)
_, z_target_bank = jax.vmap(partial(generate_grf, n_points=N_GRID, length_scale=0.4))(bank_keys)
xi_init_single = jnp.linspace(0.2, 0.8, N_AGENTS, dtype=jnp.float32)

key, subkey = jax.random.split(key)
idx = jax.random.randint(subkey, (NUM_PARALLEL_ENVS,), 0, 1000)

initial_runner_state = (
    actor_state, critic_state, 
    z_init_bank[idx], z_target_bank[idx], jnp.tile(xi_init_single, (NUM_PARALLEL_ENVS, 1)), 
    build_marl_obs_batch(z_init_bank[idx], z_target_bank[idx], jnp.tile(xi_init_single, (NUM_PARALLEL_ENVS, 1))),
    jnp.zeros(NUM_PARALLEL_ENVS), key
)

print(f"Starting Pure JAX MAPPO Training (Static FKPP) for {num_updates} Updates...")
start_time = time.time()

runner_state = initial_runner_state

for update in trange(num_updates):
    if update % 10 == 0:
        eval_z, eval_target, eval_xi = z_init_bank[0], z_target_bank[0], xi_init_single
        
        # Ensure we pass an RNG key explicitly to fast_eval_episode for FKPP physics steps
        key, eval_key = jax.random.split(key)
        eval_mse, crashed = fast_eval_episode(runner_state[0].params, eval_z, eval_xi, eval_target, ROLLOUT_STEPS, eval_key)
        
        status = "[CRASHED]" if crashed else f"{eval_mse:.6f}"
        print(f"\nUpdate {update:04d} | Eval MSE: {status} | Time: {time.time()-start_time:.1f}s")
        if update > 0:
            print(f"Metrics -> Ret: {metrics['mean_return']:.2f} | Val L: {metrics['critic_loss']:.4f} | Act L: {metrics['actor_loss']:.4f} | Ent: {metrics['entropy']:.4f}")

    runner_state, metrics = train_step(runner_state)

# Save output
actor_state_final, critic_state_final = runner_state[0], runner_state[1]
with open('models/mappo_fkpp_params2.msgpack', 'wb') as f:
    f.write(flax.serialization.to_bytes({'actor': actor_state_final.params, 'critic': critic_state_final.params}))
print(f"Training finished in {time.time()-start_time:.1f}s. Weights saved.")