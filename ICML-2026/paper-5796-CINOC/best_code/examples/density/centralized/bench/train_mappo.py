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
from flax.training.train_state import TrainState
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
from models_mappo import MAPPOActorNS2D, MAPPOCriticNS2D

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

# MAPPO Specific Configs
ROLLOUT_STEPS = 50 
PPO_EPOCHS = 4
MINIBATCH_SIZE = 1600 
TOTAL_UPDATES = 20000
EVAL_INT = 50 

GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.001
LR = 3e-4

key = jax.random.PRNGKey(42)

# ==========================================
# 2. DYNAMICS, OBS, & REWARDS (CTDE)
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
            
            push_norm = jnp.linalg.norm(push_vel, axis=-1, keepdims=True)
            push_vel = jnp.where(push_norm > push_max, push_vel * push_max / (push_norm + 1e-8), push_vel)
            
            smoke_new = ns2d_step_jax(smoke, xi, push_vel, dt=dt, buoyancy=buoyancy, sigma_push=sigma_push, Nx=Nx, Ny=Ny)
            
            xi_new = xi + dt * push_vel * 0.01
            xi_new = jnp.clip(xi_new, 0.1, jnp.array([0.9, 1.15], dtype=xi.dtype)) 
            
            # Prevent XLA Float64 casting
            smoke_new = smoke_new.astype(smoke.dtype)
            xi_new = xi_new.astype(xi.dtype)
            push_vel = push_vel.astype(jnp.float32)

            return (smoke_new, xi_new), (smoke_new, xi_new, push_vel)
        
        _, trajectory = jax.lax.scan(step_fn, (rho_init, xi_init), None, length=t_steps)
        return trajectory

def direct_control_policy(action_params, rho_obs, rho_target, xi_fixed):
    return action_params.astype(jnp.float32)

dynamics = DensityDynamicsWrapper(policy_apply_fn=direct_control_policy)

@jax.jit
def build_marl_obs_batch(rho_batch, target_batch, xi_batch):
    # CTDE: Extract localized observations per agent from the global field.
    # To maintain comparability with centralized without writing a custom Nd grid slicer,
    # we provide the flattened grids but append the specific agent's coordinates.
    rho_flat = rho_batch.reshape(rho_batch.shape[0], -1)
    tgt_flat = target_batch.reshape(target_batch.shape[0], -1)
    
    B, N, _ = xi_batch.shape
    global_context = jnp.concatenate([rho_flat, tgt_flat], axis=-1)
    global_context_exp = jnp.tile(global_context[:, None, :], (1, N, 1))
    
    # Obs Shape: (Batch, N_Agents, ObsDim)
    obs = jnp.concatenate([global_context_exp, xi_batch], axis=-1)
    return obs

def parallel_physics_step(rho_batch, xi_batch, target_batch, actions):
    def single_physics_step(rho_s, xi_s, target_s, act_s):
        traj = dynamics.unroll_controlled(
            rho_init=rho_s, xi_init=xi_s, rho_target=target_s, params=act_s, 
            t_steps=1, dt=dt, buoyancy=BUOYANCY, sigma_push=SIGMA_PUSH, push_max=PUSH_MAX, Nx=Nx, Ny=Ny
        )
        return traj[0][-1], traj[1][-1]
    
    next_rho, next_xi = jax.vmap(single_physics_step)(rho_batch, xi_batch, target_batch, actions)
    
    is_invalid = jnp.logical_not(jnp.isfinite(next_rho).all(axis=(1, 2)))
    dones_batch = is_invalid
    
    safe_rho = jnp.where(dones_batch[:, None, None], jnp.zeros_like(next_rho), next_rho)
    safe_xi = jnp.where(dones_batch[:, None, None], xi_batch, next_xi)
    
    # MAPPO Per-Agent Reward Decomposition
    # Global task success shared equally
    mse = jnp.mean(jnp.square(safe_rho - target_batch), axis=(1, 2))
    
    # Local penalties attributed to specific agents
    effort = 0.001 * jnp.sum(jnp.square(actions), axis=-1)
    
    margin = 0.02
    x_pen = jnp.maximum(0.0, margin - safe_xi[..., 0])**2 + jnp.maximum(0.0, safe_xi[..., 0] - (1.0 - margin))**2
    y_pen = jnp.maximum(0.0, margin - safe_xi[..., 1])**2 + jnp.maximum(0.0, safe_xi[..., 1] - (1.0 - margin))**2
    bound_pen = 20.0 * (x_pen + y_pen)
    
    diff = safe_xi[:, :, None, :] - safe_xi[:, None, :, :]
    dists = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-8)
    mask = jnp.eye(N_AGENTS)[None, :, :]
    coll_pen = 10.0 * jnp.sum(jnp.maximum(0.0, R_SAFE - (dists + mask * 10.0)) ** 2, axis=2)

    # rewards_batch shape is (BATCH, N_AGENTS) allowing decentralized advantage calculations
    rewards_batch = -10.0 * mse[:, None] - effort - bound_pen - coll_pen
    
    dones_batch_exp = jnp.tile(dones_batch[:, None], (1, N_AGENTS))
    return safe_rho, safe_xi, rewards_batch.astype(jnp.float32), dones_batch_exp

# ==========================================
# 3. MAPPO CORE FUNCTIONS
# ==========================================
def get_action_and_value(actor_params, critic_params, obs, rho, target, xi, key):
    # ACTOR: Uses decentralized local patch observations
    mean, log_std = actor.apply(actor_params, obs)
    std = jnp.exp(log_std)
    action = mean + std * jax.random.normal(key, mean.shape)
    
    # Log prob summed over the action dimensions (vx, vy)
    log_prob = -0.5 * jnp.sum(jnp.square((action - mean) / std) + 2 * log_std + jnp.log(2 * jnp.pi), axis=-1)
    
    # CRITIC: Uses centralized global grids and all positions
    val = critic.apply(critic_params, rho, target, xi)
    return action, log_prob, val

def compute_gae_jax(rewards, values, dones, true_next_values, last_val, gamma=0.99, lam=0.95):
    def scan_fn(carry, transition):
        r, v, d, true_next_v = transition
        gae = carry
        delta = r + gamma * true_next_v * (1.0 - d) - v
        gae = delta + gamma * lam * (1.0 - d) * gae
        return gae, gae
    
    _, advantages = jax.lax.scan(
        scan_fn, 
        jnp.zeros_like(last_val), 
        (rewards, values, dones, true_next_values), 
        reverse=True
    )
    returns = advantages + values
    return advantages, returns

def mappo_update_epoch(actor_state, critic_state, batch):
    obs, rho, target, xi, actions, old_log_probs, advantages, returns, old_values = batch    
    
    def actor_loss_fn(params):
        mean, log_std = actor.apply(params, obs) 
        std = jnp.exp(log_std)
        
        log_probs = -0.5 * jnp.sum(jnp.square((actions - mean) / std) + 2 * log_std + jnp.log(2 * jnp.pi), axis=-1)
        entropy = jnp.sum(log_std + 0.5 + 0.5 * jnp.log(2 * jnp.pi), axis=-1)
        ratio = jnp.exp(log_probs - old_log_probs)
        
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * jnp.clip(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()
        
        entropy_loss = entropy.mean()
        return pg_loss - ENTROPY_COEF * entropy_loss, (pg_loss, entropy_loss)
        
    def value_loss_fn(params):
        v = critic.apply(params, rho, target, xi) 
        
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
def train_step(runner_state, rho_init_bank, rho_target_bank, xi_init_single):
    actor_state, critic_state, rho_b, target_b, xi_b, obs_b, env_counts, rng = runner_state
    
    # 1. Rollout Phase
    def _env_step(carry, _):
        rho, target, xi, obs, counts, k = carry
        k, act_k, reset_k = jax.random.split(k, 3)
        
        action, log_prob, val = get_action_and_value(actor_state.params, critic_state.params, obs, rho, target, xi, act_k)
        env_action = jnp.clip(action, -PUSH_MAX, PUSH_MAX)
        next_rho, next_xi, rewards, dones_exp = parallel_physics_step(rho, xi, target, env_action)
        
        counts += 1
        truncs = counts >= MAX_ENV_STEPS
        needs_reset = jnp.logical_or(dones_exp[:, 0], truncs)
        
        idx_reset = jax.random.randint(reset_k, (NUM_PARALLEL_ENVS,), 0, len(rho_init_bank))
        fresh_rho = rho_init_bank[idx_reset]
        fresh_target = rho_target_bank[idx_reset]
        fresh_xi = jnp.tile(xi_init_single, (NUM_PARALLEL_ENVS, 1, 1))
        
        next_rho_final = jnp.where(needs_reset[:, None, None], fresh_rho, next_rho)
        next_target_final = jnp.where(needs_reset[:, None, None], fresh_target, target)
        next_xi_final = jnp.where(needs_reset[:, None, None], fresh_xi, next_xi)
        next_counts = jnp.where(needs_reset, jnp.int32(0), counts)
        
        next_obs_final = build_marl_obs_batch(next_rho_final, next_target_final, next_xi_final)
        true_next_v = critic.apply(critic_state.params, next_rho_final, next_target_final, next_xi_final)
        
        transition = (obs, rho, target, xi, action, log_prob, rewards, val, dones_exp, true_next_v)
        return (next_rho_final, next_target_final, next_xi_final, next_obs_final, next_counts, k), transition

    carry = (rho_b, target_b, xi_b, obs_b, env_counts, rng)
    carry, transitions = jax.lax.scan(_env_step, carry, None, length=ROLLOUT_STEPS)
    (next_rho_b, next_target_b, next_xi_b, next_obs_b, next_env_counts, rng) = carry
    t_obs, t_rho, t_target, t_xi, t_a, t_logp, t_r, t_v, t_d_exp, t_true_next_v = transitions
    
    # 2. GAE Phase
    last_val = critic.apply(critic_state.params, next_rho_b, next_target_b, next_xi_b)
    adv, ret = compute_gae_jax(t_r, t_v, t_d_exp, t_true_next_v, last_val, GAMMA, GAE_LAMBDA)
    
    # Global Advantage Normalization
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    
    f_obs = t_obs.reshape(-1, N_AGENTS, t_obs.shape[-1])
    f_rho = t_rho.reshape(-1, Nx, Ny)
    f_target = t_target.reshape(-1, Nx, Ny)
    f_xi = t_xi.reshape(-1, N_AGENTS, 2)
    f_a = t_a.reshape(-1, N_AGENTS, 2)
    f_logp = t_logp.reshape(-1, N_AGENTS)
    f_ret = ret.reshape(-1, N_AGENTS)
    f_adv = adv.reshape(-1, N_AGENTS)
    f_v = t_v.reshape(-1, N_AGENTS)

    dataset_size = f_rho.shape[0]
    num_minibatches = dataset_size // MINIBATCH_SIZE

    # 3. Optimization Phase
    def _update_epoch(epoch_carry, _):
        a_state, c_state, k = epoch_carry
        k, subk = jax.random.split(k)
        
        indices = jax.random.permutation(subk, dataset_size)
        s_obs, s_rho, s_target = f_obs[indices], f_rho[indices], f_target[indices]
        s_xi, s_a, s_logp = f_xi[indices], f_a[indices], f_logp[indices]
        s_ret, s_adv, s_v = f_ret[indices], f_adv[indices], f_v[indices]
        
        mb_obs = s_obs.reshape((num_minibatches, MINIBATCH_SIZE, *s_obs.shape[1:]))
        mb_rho = s_rho.reshape((num_minibatches, MINIBATCH_SIZE, *s_rho.shape[1:]))
        mb_target = s_target.reshape((num_minibatches, MINIBATCH_SIZE, *s_target.shape[1:]))
        mb_xi = s_xi.reshape((num_minibatches, MINIBATCH_SIZE, *s_xi.shape[1:]))
        mb_a = s_a.reshape((num_minibatches, MINIBATCH_SIZE, *s_a.shape[1:]))
        mb_logp = s_logp.reshape((num_minibatches, MINIBATCH_SIZE, *s_logp.shape[1:]))
        mb_ret = s_ret.reshape((num_minibatches, MINIBATCH_SIZE, *s_ret.shape[1:]))
        mb_adv = s_adv.reshape((num_minibatches, MINIBATCH_SIZE, *s_adv.shape[1:]))
        mb_v = s_v.reshape((num_minibatches, MINIBATCH_SIZE, *s_v.shape[1:]))

        def _update_minibatch(mb_carry, mb_data):
            a_state_, c_state_ = mb_carry
            mb_obs_, mb_rho_, mb_tgt_, mb_xi_, mb_a_, mb_logp_, mb_ret_, mb_adv_, mb_v_ = mb_data
                        
            mb_batch = (mb_obs_, mb_rho_, mb_tgt_, mb_xi_, mb_a_, mb_logp_, mb_adv_, mb_ret_, mb_v_)
            a_state_n, c_state_n, al, vl, ent = mappo_update_epoch(a_state_, c_state_, mb_batch)
            return (a_state_n, c_state_n), jnp.stack([al, vl, ent])
            
        (a_state, c_state), epoch_metrics = jax.lax.scan(
            _update_minibatch, (a_state, c_state), 
            (mb_obs, mb_rho, mb_target, mb_xi, mb_a, mb_logp, mb_ret, mb_adv, mb_v) 
        )
        return (a_state, c_state, k), epoch_metrics

    epoch_carry = (actor_state, critic_state, rng)
    epoch_carry, ppo_metrics = jax.lax.scan(_update_epoch, epoch_carry, None, length=PPO_EPOCHS)
    (actor_state, critic_state, rng) = epoch_carry

    new_runner_state = (actor_state, critic_state, next_rho_b, next_target_b, next_xi_b, next_obs_b, next_env_counts, rng)
    
    metrics = {
        "mean_return": t_r.sum(axis=0).mean(),
        "actor_loss": ppo_metrics[..., 0].mean(),
        "critic_loss": ppo_metrics[..., 1].mean(),
        "entropy": ppo_metrics[..., 2].mean()
    }
    
    return new_runner_state, metrics

@partial(jax.jit, donate_argnums=(0,))
def train_chunk(runner_state, rho_init_bank, rho_target_bank, xi_init_single):
    def scan_step(carry, _):
        new_state, metrics = train_step(carry, rho_init_bank, rho_target_bank, xi_init_single)
        return new_state, metrics
    return jax.lax.scan(scan_step, runner_state, None, length=EVAL_INT)

# ==========================================
# 5. FAST EVALUATION
# ==========================================
@partial(jax.jit, static_argnames=['max_steps'])
def fast_eval_episode(actor_params, init_rho, init_xi, target_rho, max_steps):
    def step_fn(state, _):
        rho_curr, xi_curr = state
        
        obs = build_marl_obs_batch(rho_curr[None, ...], target_rho[None, ...], xi_curr[None, ...])
        mean, _ = actor.apply(actor_params, obs)
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
actor = MAPPOActorNS2D(n_agents=N_AGENTS, push_max=PUSH_MAX)
critic = MAPPOCriticNS2D(n_agents=N_AGENTS)

key, act_k, val_k = jax.random.split(key, 3)

# Initializer Shapes Enforce CTDE
dummy_rho_global = jnp.zeros((1, Nx, Ny))
dummy_tgt_global = jnp.zeros((1, Nx, Ny))
dummy_xi_global = jnp.zeros((1, N_AGENTS, 2))
dummy_obs_local = build_marl_obs_batch(dummy_rho_global, dummy_tgt_global, dummy_xi_global)

# Exact Learning Rate Fix
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
    params=critic.init(val_k, dummy_rho_global, dummy_tgt_global, dummy_xi_global), # CRITIC -> Global
    tx=optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr_schedule, eps=1e-5))
)

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
obs_batch = build_marl_obs_batch(rho_batch, target_batch, xi_batch)

initial_runner_state = (
    actor_state, critic_state, rho_batch, target_batch, xi_batch, obs_batch,
    jnp.zeros(NUM_PARALLEL_ENVS, dtype=jnp.int32), key
)

print(f"Starting Massively Parallel CTDE MAPPO Training (NS2D)...")
start_time = time.time()

runner_state = initial_runner_state
num_chunks = TOTAL_UPDATES // EVAL_INT

for chunk_idx in trange(num_chunks):
    eval_rho = rho_init_bank[0]
    eval_target = rho_target_bank[0]
    eval_energy, crashed = fast_eval_episode(runner_state[0].params, eval_rho, xi_init_single, eval_target, MAX_ENV_STEPS)
    
    current_update = chunk_idx * EVAL_INT
    status = "[CRASHED]" if crashed else f"{eval_energy:.6f}"
    print(f"\nUpdate {current_update:04d} | Eval MSE: {status} | Time: {time.time()-start_time:.1f}s")
    
    runner_state, chunk_metrics = train_chunk(runner_state, rho_init_bank, rho_target_bank, xi_init_single)

# Final Eval
eval_energy, crashed = fast_eval_episode(runner_state[0].params, eval_rho, xi_init_single, eval_target, MAX_ENV_STEPS)
status = "[CRASHED]" if crashed else f"{eval_energy:.6f}"
print(f"\nUpdate {TOTAL_UPDATES:04d} | Eval MSE: {status} | Time: {time.time()-start_time:.1f}s")

actor_state_final, critic_state_final = runner_state[0], runner_state[1]
models_dir = Path('models')
models_dir.mkdir(exist_ok=True)
with open(models_dir / 'mappo_ns2d_params.msgpack', 'wb') as f:
    f.write(flax.serialization.to_bytes({'actor': actor_state_final.params, 'critic': critic_state_final.params}))
print(f"Training finished in {time.time()-start_time:.1f}s. Weights saved.")