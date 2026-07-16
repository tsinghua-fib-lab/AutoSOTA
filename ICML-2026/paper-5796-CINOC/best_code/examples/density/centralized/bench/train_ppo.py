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

# Project Imports
from examples.density.centralized.dynamics import ns2d_step_jax
from models_ppo import CentralizedPPOActor, CentralizedPPOCritic

# ==========================================
# 1. LOAD DATA & CONFIGURATIONS
# ==========================================
print("Loading NS2D starting state & target banks from dataset...")
data_dir = Path(__file__).parent.parent.parent / 'data'

config = np.load(data_dir / 'config.npz')
Nx = int(config['Nx'])
Ny = int(config['Ny'])
dt = float(config['dt'])

train_data = np.load(data_dir / 'train_data.npz')
rho_init_bank = jnp.array(train_data['rho_init'], dtype=jnp.float32)
rho_target_bank = jnp.array(train_data['rho_target'], dtype=jnp.float32)

# Environment / Physics Constants
N_AGENTS = 9 
MAX_ENV_STEPS = 150 
NUM_PARALLEL_ENVS = 128
BUOYANCY = 0.0
SIGMA_PUSH = 0.2
PUSH_MAX = 0.8
R_SAFE = 0.15

# PPO Specific Configs
ROLLOUT_STEPS = 50     
PPO_EPOCHS = 4
MINIBATCH_SIZE = 1600 
TOTAL_UPDATES = 20000
EVAL_INT = 50        
TARGET_KL = 0.05        # Early stopping threshold

key = jax.random.PRNGKey(42)

# ==========================================
# 2. DYNAMICS & REWARD WRAPPER
# ==========================================
@struct.dataclass
class DensityDynamicsWrapper:
    policy_apply_fn: Callable = struct.field(pytree_node=False)

    @partial(jax.jit, static_argnames=['t_steps', 'Nx', 'Ny'])
    def unroll_controlled(
        self, rho_init, xi_init, rho_target, params, t_steps: int, 
        Nx: int = 64, Ny: int = 80, dt: float = 1.0, buoyancy: float = 0.0, 
        sigma_push: float = 0.2, push_max: float = 0.8
    ):
        def step_fn(carry, _):
            smoke, xi = carry
            push_vel = self.policy_apply_fn(params, smoke, rho_target, xi)
            
            # Enforce max push velocity constraint
            push_norm = jnp.linalg.norm(push_vel, axis=-1, keepdims=True)
            push_vel = jnp.where(push_norm > push_max, push_vel * push_max / (push_norm + 1e-8), push_vel)
            
            smoke_new = ns2d_step_jax(smoke, xi, push_vel, dt=dt, buoyancy=buoyancy, sigma_push=sigma_push, Nx=Nx, Ny=Ny)
            
            xi_new = xi + dt * push_vel * 0.01
            xi_new = jnp.clip(xi_new, 0.1, jnp.array([0.9, 1.15], dtype=xi.dtype)) 
            
            # Explicit float32 casting to avoid XLA float64 accumulation
            smoke_new = smoke_new.astype(smoke.dtype)
            xi_new = xi_new.astype(xi.dtype)
            push_vel = push_vel.astype(jnp.float32)
            return (smoke_new, xi_new), (smoke_new, xi_new, push_vel)
        
        _, trajectory = jax.lax.scan(step_fn, (rho_init, xi_init), None, length=t_steps)
        return trajectory

def direct_control_policy(action_params, rho_obs, rho_target, xi_fixed):
    return action_params.astype(jnp.float32)

dynamics = DensityDynamicsWrapper(policy_apply_fn=direct_control_policy)

def get_logprob_and_action(mean, log_std, key=None, action=None):
    std = jnp.exp(log_std)
    if action is None:
        noise = jax.random.normal(key, mean.shape)
        raw_action = mean + noise * std
    else:
        raw_action = action
    
    var = std ** 2
    log_prob = -0.5 * ((raw_action - mean) ** 2) / var - log_std - 0.5 * jnp.log(2 * jnp.pi)
    env_action = jnp.clip(raw_action, -PUSH_MAX, PUSH_MAX)
    
    # Sum log probabilities over both agents and spatial dimensions (last 2 axes)
    return env_action, raw_action, jnp.sum(log_prob, axis=(-2, -1))

# ==========================================
# 3. INITIALIZATION
# ==========================================
actor = CentralizedPPOActor(n_agents=N_AGENTS, push_max=PUSH_MAX)
critic = CentralizedPPOCritic()

key, *subkeys = jax.random.split(key, 4)
dummy_rho = jnp.zeros((NUM_PARALLEL_ENVS, Nx, Ny), dtype=jnp.float32)
dummy_target = jnp.zeros((NUM_PARALLEL_ENVS, Nx, Ny), dtype=jnp.float32)
dummy_xi = jnp.zeros((NUM_PARALLEL_ENVS, N_AGENTS, 2), dtype=jnp.float32)

actor_params = actor.init(subkeys[0], dummy_rho, dummy_target, dummy_xi)
critic_params = critic.init(subkeys[1], dummy_rho, dummy_target, dummy_xi)

# Calculate the exact number of gradient applications for Optax
dataset_size = NUM_PARALLEL_ENVS * ROLLOUT_STEPS
num_minibatches = dataset_size // MINIBATCH_SIZE
updates_per_rollout = num_minibatches * PPO_EPOCHS
actual_optax_steps = TOTAL_UPDATES * updates_per_rollout

lr_schedule_actor = optax.linear_schedule(
    init_value=1e-4, end_value=0.0, transition_steps=actual_optax_steps
)
lr_schedule_critic = optax.linear_schedule(
    init_value=5e-4, end_value=0.0, transition_steps=actual_optax_steps
)

tx_actor = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr_schedule_actor))
tx_critic = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr_schedule_critic))

opt_actor = tx_actor.init(actor_params)
opt_critic = tx_critic.init(critic_params)

# ==========================================
# 4. TRAINING LOGIC
# ==========================================
@jax.jit
def compute_gae_jax(rewards, values, dones, needs_reset, true_next_values, gamma=0.99, lam=0.95):
    def scan_fn(carry, transition):
        r, v, d, reset, true_next_v = transition
        gae = carry
        delta = r + gamma * true_next_v * (1.0 - d) - v
        gae = delta + gamma * lam * (1.0 - reset) * gae
        return gae, gae
    
    _, advantages = jax.lax.scan(
        scan_fn, 
        jnp.zeros_like(values[0]), 
        (rewards, values, dones, needs_reset, true_next_values), 
        reverse=True
    )
    returns = advantages + values
    return advantages, returns

def parallel_physics_step(rho_batch, xi_batch, target_batch, actions):
    def single_physics_step(rho_s, xi_s, target_s, act_s):
        traj = dynamics.unroll_controlled(
            rho_init=rho_s, xi_init=xi_s, rho_target=target_s, params=act_s, 
            t_steps=1, dt=dt, buoyancy=BUOYANCY, sigma_push=SIGMA_PUSH, push_max=PUSH_MAX, Nx=Nx, Ny=Ny
        )
        return traj[0][-1], traj[1][-1]
    
    next_rho, next_xi = jax.vmap(single_physics_step)(rho_batch, xi_batch, target_batch, actions)
    
    # Check bounds and invalids
    is_invalid = jnp.logical_not(jnp.isfinite(next_rho).all(axis=(1, 2)))
    dones_batch = is_invalid
    
    safe_rho = jnp.where(dones_batch[:, None, None], jnp.zeros_like(next_rho), next_rho)
    safe_xi = jnp.where(dones_batch[:, None, None], xi_batch, next_xi)
    
    # Exact TD3 Reward Logic
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
    
    return safe_rho, safe_xi, rewards_batch.astype(jnp.float32), dones_batch

def train_step(runner_state, rho_init_bank, rho_target_bank, xi_init_single):
    a_params, c_params, opt_a, opt_c, rho_b, target_b, xi_b, env_counts, rng = runner_state
    
    # --- Rollout Phase ---
    def _env_step(carry, _):
        rho, target, xi, counts, k = carry
        k, act_k, reset_k = jax.random.split(k, 3)
        
        mean, log_std = actor.apply(a_params, rho, target, xi)
        env_actions, raw_actions, log_probs = get_logprob_and_action(mean, log_std, key=act_k)
        values = critic.apply(c_params, rho, target, xi).squeeze(-1)
        
        next_rho, next_xi, rewards, dones = parallel_physics_step(rho, xi, target, env_actions)
        
        counts += 1
        truncs = counts >= MAX_ENV_STEPS
        needs_reset = jnp.logical_or(dones, truncs)
        
        true_next_v = critic.apply(c_params, next_rho, target, next_xi).squeeze(-1)
        
        idx_reset = jax.random.randint(reset_k, (NUM_PARALLEL_ENVS,), 0, len(rho_init_bank))
        fresh_rho = rho_init_bank[idx_reset]
        fresh_target = rho_target_bank[idx_reset]
        fresh_xi = jnp.tile(xi_init_single, (NUM_PARALLEL_ENVS, 1, 1))
        
        next_rho_final = jnp.where(needs_reset[:, None, None], fresh_rho, next_rho)
        next_target_final = jnp.where(needs_reset[:, None, None], fresh_target, target)
        next_xi_final = jnp.where(needs_reset[:, None, None], fresh_xi, next_xi)
        
        # Keep counts int32 to match input carry typing
        next_counts = jnp.where(needs_reset, jnp.int32(0), counts)
        
        transition = (rho, target, xi, raw_actions, rewards, values, log_probs, dones, needs_reset, true_next_v)
        return (next_rho_final, next_target_final, next_xi_final, next_counts, k), transition

    carry = (rho_b, target_b, xi_b, env_counts, rng)
    carry, transitions = jax.lax.scan(_env_step, carry, None, length=ROLLOUT_STEPS)
    (next_rho_b, next_target_b, next_xi_b, next_env_counts, rng) = carry
    t_rho, t_target, t_xi, t_raw_a, t_r, t_v, t_logp, t_d, t_reset, t_true_next_v = transitions
    
    adv, ret = compute_gae_jax(t_r, t_v, t_d, t_reset, t_true_next_v)
    
    # Flatten rollout data for minibatching
    f_rho = t_rho.reshape(-1, Nx, Ny)
    f_target = t_target.reshape(-1, Nx, Ny)
    f_xi = t_xi.reshape(-1, N_AGENTS, 2)
    f_a = t_raw_a.reshape(-1, N_AGENTS, 2)
    f_logp = t_logp.reshape(-1) 
    f_ret = ret.reshape(-1)
    f_adv = adv.reshape(-1)
    f_v = t_v.reshape(-1)
    
    # Global advantage normalization
    f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)
    dataset_size = f_rho.shape[0]
    num_minibatches = dataset_size // MINIBATCH_SIZE

    # --- Optimization Phase ---
    def _update_epoch(epoch_carry, _):
        ap, cp, oa, oc, update_actor_flag, k = epoch_carry
        k, subk = jax.random.split(k)
        
        indices = jax.random.permutation(subk, dataset_size)
        s_rho, s_target, s_xi, s_a = f_rho[indices], f_target[indices], f_xi[indices], f_a[indices]
        s_logp, s_ret, s_adv, s_v = f_logp[indices], f_ret[indices], f_adv[indices], f_v[indices]
        
        mb_rho = s_rho.reshape((num_minibatches, MINIBATCH_SIZE, *s_rho.shape[1:]))
        mb_target = s_target.reshape((num_minibatches, MINIBATCH_SIZE, *s_target.shape[1:]))
        mb_xi = s_xi.reshape((num_minibatches, MINIBATCH_SIZE, *s_xi.shape[1:]))
        mb_a = s_a.reshape((num_minibatches, MINIBATCH_SIZE, *s_a.shape[1:]))
        mb_logp = s_logp.reshape((num_minibatches, MINIBATCH_SIZE))
        mb_ret = s_ret.reshape((num_minibatches, MINIBATCH_SIZE))
        mb_adv = s_adv.reshape((num_minibatches, MINIBATCH_SIZE))
        mb_v = s_v.reshape((num_minibatches, MINIBATCH_SIZE)) 
        
        def _update_minibatch(mb_carry, mb_data):
            ap_, cp_, oa_, oc_, is_actor_updating = mb_carry
            mb_rho_, mb_tgt_, mb_xi_, mb_a_, mb_logp_, mb_ret_, mb_adv_, mb_v_ = mb_data
            
            # Minibatch advantage normalization
            mb_adv_ = (mb_adv_ - mb_adv_.mean()) / (mb_adv_.std() + 1e-8)
            
            def loss_fn(ap_tgt, cp_tgt):
                mean, log_std = actor.apply(ap_tgt, mb_rho_, mb_tgt_, mb_xi_) 
                _, _, new_logp = get_logprob_and_action(mean, log_std, action=mb_a_)
                
                entropy = jnp.sum(log_std + 0.5 + 0.5 * jnp.log(2 * jnp.pi), axis=(-2, -1)).mean()
                log_ratio = new_logp - mb_logp_
                log_ratio_safe = jnp.clip(log_ratio, -5.0, 2.0)
                ratio = jnp.exp(log_ratio_safe)
                approx_kl = jnp.mean((ratio - 1.0) - log_ratio)
                
                pg_loss1 = -mb_adv_ * ratio
                pg_loss2 = -mb_adv_ * jnp.clip(ratio, 1.0 - 0.2, 1.0 + 0.2)
                actor_loss = jnp.maximum(pg_loss1, pg_loss2).mean() - 0.001 * entropy
                
                values = critic.apply(cp_tgt, mb_rho_, mb_tgt_, mb_xi_).squeeze(-1) 
                v_clipped = mb_v_ + jnp.clip(values - mb_v_, -0.2, 0.2)
                v_loss_unclipped = (values - mb_ret_) ** 2
                v_loss_clipped = (v_clipped - mb_ret_) ** 2
                critic_loss = 0.5 * jnp.mean(jnp.maximum(v_loss_unclipped, v_loss_clipped))
            
                return actor_loss + 0.5 * critic_loss, jnp.stack([actor_loss, critic_loss, entropy, approx_kl])

            (total_loss, metrics), grads = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(ap_, cp_)
            current_kl = metrics[3]
            
            should_update_actor = jnp.logical_and(is_actor_updating, current_kl < TARGET_KL)
            
            def do_update():
                u_a, new_oa = tx_actor.update(grads[0], oa_)
                return optax.apply_updates(ap_, u_a), new_oa
            def skip_update():
                return ap_, oa_
            
            ap_n, oa_n = jax.lax.cond(should_update_actor, do_update, skip_update)
            
            up_c, oc_n = tx_critic.update(grads[1], oc_)
            cp_n = optax.apply_updates(cp_, up_c)
            
            return (ap_n, cp_n, oa_n, oc_n, should_update_actor), metrics
            
        (ap, cp, oa, oc, update_actor_flag), epoch_metrics = jax.lax.scan(
            _update_minibatch, (ap, cp, oa, oc, update_actor_flag), 
            (mb_rho, mb_target, mb_xi, mb_a, mb_logp, mb_ret, mb_adv, mb_v)
        )
        return (ap, cp, oa, oc, update_actor_flag, k), epoch_metrics

    epoch_carry = (a_params, c_params, opt_a, opt_c, jnp.bool_(True), rng)
    epoch_carry, ppo_metrics = jax.lax.scan(_update_epoch, epoch_carry, None, length=PPO_EPOCHS)
    (a_params, c_params, opt_a, opt_c, _, rng) = epoch_carry

    new_runner_state = (a_params, c_params, opt_a, opt_c, next_rho_b, next_target_b, next_xi_b, next_env_counts, rng)
    
    metrics = {
        "mean_return": t_r.sum(axis=0).mean(),
        "actor_loss": ppo_metrics[..., 0].mean(),
        "critic_loss": ppo_metrics[..., 1].mean(),
        "entropy": ppo_metrics[..., 2].mean(),
        "approx_kl": ppo_metrics[..., 3].mean()
    }
    return new_runner_state, metrics

@jax.jit(donate_argnums=(0,))
def train_chunk(runner_state, rho_init_bank, rho_target_bank, xi_init_single): 
    def scan_step(carry, _):
        new_state, metrics = train_step(carry, rho_init_bank, rho_target_bank, xi_init_single) 
        return new_state, metrics
    return jax.lax.scan(scan_step, runner_state, None, length=EVAL_INT)

# ==========================================
# 5. FAST EVALUATION
# ==========================================
@partial(jax.jit, static_argnames=['max_steps'])
def fast_eval_episode(a_params, init_rho, init_xi, target_rho, max_steps):
    def step_fn(state, _):
        rho_curr, xi_curr = state
        
        mean, _ = actor.apply(a_params, rho_curr[None, ...], target_rho[None, ...], xi_curr[None, ...])
        act_flat = mean.squeeze(0)
        
        traj = dynamics.unroll_controlled(
            rho_init=rho_curr, xi_init=xi_curr, rho_target=target_rho, params=act_flat, 
            t_steps=1, dt=dt, buoyancy=BUOYANCY, sigma_push=SIGMA_PUSH, push_max=PUSH_MAX, Nx=Nx, Ny=Ny
        )
        next_rho, next_xi = traj[0][-1], traj[1][-1]
        
        mse = jnp.mean((next_rho - target_rho)**2)
        crashed = jnp.isnan(next_rho).any() | jnp.isinf(next_rho).any()
        
        return (next_rho, next_xi), (mse, crashed)

    _, (mses, crashes) = jax.lax.scan(step_fn, (init_rho, init_xi), None, length=max_steps)
    return jnp.mean(mses), jnp.any(crashes)

# ==========================================
# 6. EXECUTION LOOP
# ==========================================
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

# Initialize counts as explicitly int32 to fix the jax.lax.scan type discrepancy
initial_runner_state = (
    actor_params, critic_params, opt_actor, opt_critic, 
    rho_batch, target_batch, xi_batch, jnp.zeros(NUM_PARALLEL_ENVS, dtype=jnp.int32), key
)

print("Starting Massively Parallel Pure JAX PPO Training (NS2D)...")
start_time = time.time()

runner_state = initial_runner_state
num_chunks = TOTAL_UPDATES // EVAL_INT

for chunk in trange(num_chunks):
    eval_rho = rho_init_bank[0]
    eval_target = rho_target_bank[0]
    eval_mse, crashed = fast_eval_episode(runner_state[0], eval_rho, xi_init_single, eval_target, MAX_ENV_STEPS)
    
    current_update = chunk * EVAL_INT
    status = "[CRASHED]" if crashed else f"{eval_mse:.6f}"
    
    runner_state, batch_metrics = train_chunk(runner_state, rho_init_bank, rho_target_bank, xi_init_single)
    current_kl = jnp.mean(batch_metrics["approx_kl"]) 
    print(f"Update {current_update:04d} | Eval MSE: {status} | KL: {current_kl:.4f} | Time: {time.time()-start_time:.1f}s")

# Final Eval
eval_mse, crashed = fast_eval_episode(runner_state[0], eval_rho, xi_init_single, eval_target, MAX_ENV_STEPS)
status = "[CRASHED]" if crashed else f"{eval_mse:.6f}"
print(f"\nUpdate {TOTAL_UPDATES:04d} | Final Eval MSE: {status} | Time: {time.time()-start_time:.1f}s")

actor_params_final = runner_state[0]
models_dir = Path('models')
models_dir.mkdir(exist_ok=True)
with open(models_dir / 'ppo_ns2d_centralized_params.msgpack', 'wb') as f:
    f.write(flax.serialization.to_bytes({'actor': actor_params_final}))
print(f"Training finished. Weights saved.")