import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.80'

import jax
jax.config.update("jax_enable_x64", True)

# --- Keep compilation fast with JAX Caching ---
jax.config.update("jax_disable_jit", False)
cache_dir = os.path.join(os.path.dirname(__file__), ".jax_cache")
os.makedirs(cache_dir, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", cache_dir)
# ----------------------------------------------

import jax.numpy as jnp
import optax
import flax.serialization
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

# IMPORT THE NEW PPO MODELS
from models_ppo import FCNActorPPO, FCNCriticPPO
from examples.turbulence2d.decentralized.data_utils import get_batch_initial_conditions
import tesseracts.turbulence2d.solver as solver

# ==========================================
# 1. CONFIGURATIONS
# ==========================================
N_AGENTS = 64
L_DOMAIN = 1.0
N_GRID = 64
U_MAX = 75.0
EVAL_INT = 20

# Turbulence Specific Control Timing & Physics
MAX_ENV_STEPS = 150
SUBSTEPS = 5
DT = 0.01
VISCOSITY = 5e-4
NUM_PARALLEL_ENVS = 64

# --- PPO Configs ---
ROLLOUT_STEPS = 150
PPO_EPOCHS = 4
MINIBATCH_SIZE = 1600
TOTAL_UPDATES = 1000 
TARGET_KL = 0.05 

STATE_NORM_FACTOR = 50.0

key = jax.random.PRNGKey(42)

# ==========================================
# 2. GLOBAL PRECOMPUTATION & INIT
# ==========================================
kx, ky, k_sq, k_inv = solver.get_spectral_grid(N_GRID, L_DOMAIN)
dt_phys = DT / SUBSTEPS

grid_dim = int(np.sqrt(N_AGENTS))
x_c = jnp.linspace(0, L_DOMAIN, grid_dim, endpoint=False) + L_DOMAIN/(2*grid_dim)
y_c = jnp.linspace(0, L_DOMAIN, grid_dim, endpoint=False) + L_DOMAIN/(2*grid_dim)
xv, yv = jnp.meshgrid(x_c, y_c)
centers_flat = jnp.stack([xv.flatten(), yv.flatten()], axis=1)

forcing_hat = solver.compute_forcing_profile(
    centers_flat[:, 0], centers_flat[:, 1], N_GRID, L_DOMAIN, 0.05
)

actor = FCNActorPPO(n_agents=N_AGENTS, u_max=U_MAX)
critic = FCNCriticPPO()

key, *subkeys = jax.random.split(key, 4)

# Initialize networks in FP32
dummy_z = jnp.zeros((1, N_GRID, N_GRID), dtype=jnp.float32)
actor_params = actor.init(subkeys[0], dummy_z)
critic_params = critic.init(subkeys[1], dummy_z)

dataset_size = NUM_PARALLEL_ENVS * ROLLOUT_STEPS
num_minibatches = dataset_size // MINIBATCH_SIZE
total_optimizer_steps = TOTAL_UPDATES * PPO_EPOCHS * num_minibatches

lr_schedule_actor = optax.cosine_decay_schedule(init_value=1e-5, decay_steps=total_optimizer_steps, alpha=0.1)
lr_schedule_critic = optax.cosine_decay_schedule(init_value=1e-3, decay_steps=total_optimizer_steps, alpha=0.1)

tx_actor = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr_schedule_actor))
tx_critic = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr_schedule_critic))

opt_actor = tx_actor.init(actor_params)
opt_critic = tx_critic.init(critic_params)

# ==========================================
# 3. CORE LOGIC
# ==========================================
def get_logprob_and_action(mean, log_std, key=None, action=None):
    std = jnp.exp(log_std)
    if action is None:
        noise = jax.random.normal(key, mean.shape, dtype=jnp.float32)
        raw_action = mean + noise * std
    else:
        raw_action = action
    
    var = std ** 2
    log_prob = -0.5 * ((raw_action - mean) ** 2) / var - log_std - 0.5 * jnp.log(2 * jnp.pi)
    env_action = jnp.clip(raw_action, -U_MAX, U_MAX)
    
    return env_action, raw_action, jnp.sum(log_prob, axis=-1)

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

def parallel_physics_step(w_init_batch, actions):
    actions_64 = actions.astype(jnp.float64)
    
    def single_physics_step(w_single, act_single):
        w_hat = jnp.fft.fft2(w_single)
        def rk4_loop(i, w):
            return solver.rk4_step(w, dt_phys, kx, ky, k_sq, k_inv, VISCOSITY, forcing_hat, act_single)
        w_hat_next = jax.lax.fori_loop(0, SUBSTEPS, rk4_loop, w_hat)
        return jnp.fft.ifft2(w_hat_next).real
    
    next_w_batch = jax.vmap(single_physics_step)(w_init_batch, actions_64)
    
    is_invalid = jnp.logical_not(jnp.isfinite(next_w_batch).all(axis=(1, 2)))
    is_exploding = jnp.max(jnp.abs(next_w_batch), axis=(1, 2)) > 1000.0 
    dones_batch = jnp.logical_or(is_invalid, is_exploding)[:, None]
    
    safe_w = jnp.where(dones_batch[:, :, None], jnp.zeros_like(next_w_batch), next_w_batch)
    
    global_enstrophy = jnp.mean(jnp.square(safe_w), axis=(1, 2))[:, None]
    r_track = -jnp.log(global_enstrophy + 1.0)
    
    normalized_actions = actions / U_MAX
    effort = jnp.mean(jnp.square(normalized_actions), axis=1)[:, None]
    r_effort = -(effort * 1.0) 
    
    rewards_batch = jnp.where(dones_batch, -2000.0, r_track + r_effort).astype(jnp.float32)
    
    return safe_w, rewards_batch, dones_batch

def train_step(runner_state, state_bank):
    a_params, c_params, opt_a, opt_c, w_batch, env_counts, rng = runner_state
    
    def _env_step(carry, _):
        w, counts, k = carry
        k, act_k, reset_k = jax.random.split(k, 3)
        
        w_norm = jnp.clip(w / STATE_NORM_FACTOR, -5.0, 5.0).astype(jnp.float32)
        
        mean, log_std = actor.apply(a_params, w_norm)
        env_actions, raw_actions, log_probs = get_logprob_and_action(mean, log_std, key=act_k)
        values = critic.apply(c_params, w_norm)
        
        next_w, rewards, dones = parallel_physics_step(w, env_actions)
        
        counts += 1
        truncs = counts >= MAX_ENV_STEPS
        needs_reset = jnp.logical_or(dones.flatten(), truncs)
        
        true_next_w_norm = jnp.clip(next_w / STATE_NORM_FACTOR, -5.0, 5.0).astype(jnp.float32)
        true_next_v = critic.apply(c_params, true_next_w_norm)
        
        fresh_states = jax.random.choice(reset_k, state_bank, shape=(NUM_PARALLEL_ENVS,))
        next_w_final = jnp.where(needs_reset[:, None, None], fresh_states, next_w)
        next_counts = jnp.where(needs_reset, 0, counts)
        
        transition = (w, raw_actions, rewards, values, log_probs, dones, needs_reset[:, None], true_next_v)
        return (next_w_final, next_counts, k), transition

    carry = (w_batch, env_counts, rng)
    carry, transitions = jax.lax.scan(_env_step, carry, None, length=ROLLOUT_STEPS)
    (next_w_batch, next_env_counts, rng) = carry
    t_w, t_raw_a, t_r, t_v, t_logp, t_d, t_reset, t_true_next_v = transitions
    
    adv, ret = compute_gae_jax(t_r, t_v, t_d, t_reset, t_true_next_v)
    
    f_w = t_w.reshape(-1, N_GRID, N_GRID)
    f_a = t_raw_a.reshape(-1, N_AGENTS)
    f_logp = t_logp.reshape(-1) 
    f_ret = ret.reshape(-1)
    f_adv = adv.reshape(-1)
    f_v = t_v.reshape(-1)
    
    # Global Buffer Advantage Normalization (CORRECT)
    f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)
    f_adv = jnp.clip(f_adv, -5.0, 5.0)

    def _update_epoch(epoch_carry, _):
        ap, cp, oa, oc, update_actor_flag, k = epoch_carry
        k, subk = jax.random.split(k)
        
        indices = jax.random.permutation(subk, dataset_size)
        s_w, s_a = f_w[indices], f_a[indices]
        s_logp, s_ret, s_adv, s_v = f_logp[indices], f_ret[indices], f_adv[indices], f_v[indices]
        
        mb_w = s_w.reshape((num_minibatches, MINIBATCH_SIZE, *s_w.shape[1:]))
        mb_a = s_a.reshape((num_minibatches, MINIBATCH_SIZE, *s_a.shape[1:]))
        mb_logp = s_logp.reshape((num_minibatches, MINIBATCH_SIZE, *s_logp.shape[1:]))
        mb_ret = s_ret.reshape((num_minibatches, MINIBATCH_SIZE, *s_ret.shape[1:]))
        mb_adv = s_adv.reshape((num_minibatches, MINIBATCH_SIZE, *s_adv.shape[1:]))
        mb_v = s_v.reshape((num_minibatches, MINIBATCH_SIZE, *s_v.shape[1:])) 
        
        def _update_minibatch(mb_carry, mb_data):
            ap_, cp_, oa_, oc_, is_actor_updating = mb_carry
            mb_w_, mb_a_, mb_logp_, mb_ret_, mb_adv_, mb_v_ = mb_data
            
            b_w_norm = jnp.clip(mb_w_ / STATE_NORM_FACTOR, -5.0, 5.0).astype(jnp.float32)
            
            # (REMOVED MINIBATCH ADVANTAGE NORMALIZATION HERE)
            
            def loss_fn(ap_tgt, cp_tgt):
                mean, log_std = actor.apply(ap_tgt, b_w_norm) 
                _, _, new_logp = get_logprob_and_action(mean, log_std, action=mb_a_)
                
                entropy = jnp.sum(log_std + 0.5 + 0.5 * jnp.log(2 * jnp.pi), axis=-1).mean()
                log_ratio = new_logp - mb_logp_
                log_ratio_safe = jnp.clip(log_ratio, -5.0, 2.0)
                ratio = jnp.exp(log_ratio_safe)
                
                approx_kl = jnp.mean((ratio - 1.0) - log_ratio)
                
                pg_loss1 = -mb_adv_ * ratio
                pg_loss2 = -mb_adv_ * jnp.clip(ratio, 1.0 - 0.1, 1.0 + 0.1)
                actor_loss = jnp.maximum(pg_loss1, pg_loss2).mean() - 0.001 * entropy
                
                values = critic.apply(cp_tgt, b_w_norm).squeeze(-1) 
                v_clipped = mb_v_ + jnp.clip(values - mb_v_, -0.2, 0.2)
                v_loss_unclipped = (values - mb_ret_) ** 2
                v_loss_clipped = (v_clipped - mb_ret_) ** 2
                critic_loss = 0.5 * jnp.mean(jnp.maximum(v_loss_unclipped, v_loss_clipped))
            
                return actor_loss + 0.5 * critic_loss, jnp.stack([actor_loss, critic_loss, entropy, approx_kl])

            (total_loss, metrics), grads = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(ap_, cp_)
            current_kl = metrics[3]
            
            should_update_actor = jnp.logical_and(is_actor_updating, current_kl < TARGET_KL)
            
            # FIX: Always advance the optimizer state to keep the LR schedule in sync, 
            # but conditionally apply the parameter updates based on KL early stopping.
            u_a, oa_n = tx_actor.update(grads[0], oa_)
            
            def apply_update():
                return optax.apply_updates(ap_, u_a)
            def skip_update():
                return ap_
            
            ap_n = jax.lax.cond(should_update_actor, apply_update, skip_update)
            
            up_c, oc_n = tx_critic.update(grads[1], oc_)
            cp_n = optax.apply_updates(cp_, up_c)
            
            return (ap_n, cp_n, oa_n, oc_n, should_update_actor), metrics
            
        (ap, cp, oa, oc, update_actor_flag), epoch_metrics = jax.lax.scan(
            _update_minibatch, (ap, cp, oa, oc, update_actor_flag), 
            (mb_w, mb_a, mb_logp, mb_ret, mb_adv, mb_v)
        )
        return (ap, cp, oa, oc, update_actor_flag, k), epoch_metrics

    epoch_carry = (a_params, c_params, opt_a, opt_c, jnp.bool_(True), rng)
    epoch_carry, ppo_metrics = jax.lax.scan(_update_epoch, epoch_carry, None, length=PPO_EPOCHS)
    (a_params, c_params, opt_a, opt_c, _, rng) = epoch_carry

    new_runner_state = (a_params, c_params, opt_a, opt_c, next_w_batch, next_env_counts, rng)
    
    metrics = {
        "mean_return": t_r.sum(axis=0).mean(),
        "actor_loss": ppo_metrics[..., 0].mean(),
        "critic_loss": ppo_metrics[..., 1].mean(),
        "entropy": ppo_metrics[..., 2].mean(),
        "approx_kl": ppo_metrics[..., 3].mean()
    }
    
    return new_runner_state, metrics

@jax.jit
def train_chunk(runner_state, state_bank):
    def scan_step(carry, _):
        new_state, metrics = train_step(carry, state_bank)
        return new_state, metrics
    return jax.lax.scan(scan_step, runner_state, None, length=EVAL_INT)

@partial(jax.jit, static_argnames=['max_steps'])
def fast_eval_episode(a_params, init_w, max_steps):
    def step_fn(state, _):
        w_norm = jnp.clip(state / STATE_NORM_FACTOR, -5.0, 5.0).astype(jnp.float32)
        mean, _ = actor.apply(a_params, w_norm[None, ...])
        act_flat = (mean.squeeze(0)).astype(jnp.float64)
        
        w_hat = jnp.fft.fft2(state)
        def rk4_loop(i, w):
            return solver.rk4_step(w, dt_phys, kx, ky, k_sq, k_inv, VISCOSITY, forcing_hat, act_flat)
        w_hat_next = jax.lax.fori_loop(0, SUBSTEPS, rk4_loop, w_hat)
        next_state = jnp.fft.ifft2(w_hat_next).real
        
        energy = jnp.mean(next_state**2)
        crashed = jnp.isnan(next_state).any() | jnp.isinf(next_state).any() | (jnp.max(jnp.abs(next_state)) > 1000.0)
        return next_state, (energy, crashed)

    _, (energies, crashes) = jax.lax.scan(step_fn, init_w, None, length=max_steps)
    return jnp.mean(energies), energies[-1], jnp.any(crashes)

# ==========================================
# 4. EXECUTION LOOP
# ==========================================
print("Loading 2D Turbulence Initial Conditions...")
data_dir = Path('../../data')
data_dir.mkdir(parents=True, exist_ok=True)
file_path = data_dir / 'turbulence_chaotic_ics_64_more.pkl'

if file_path.exists():
    with open(file_path, 'rb') as f:
        state_bank = jnp.array(pickle.load(f))
    print(f"Loaded {len(state_bank)} ICs from {file_path}")
else:
    print("Generating ICs (this may take a few minutes)...")
    state_bank = get_batch_initial_conditions(key, 500, N_GRID, L_DOMAIN, viscosity=5e-4)
    with open(file_path, 'wb') as f:
        pickle.dump(np.array(state_bank), f)

if jnp.iscomplexobj(state_bank):
    state_bank = jnp.fft.ifft2(state_bank).real.astype(jnp.float64)
else:
    state_bank = state_bank.astype(jnp.float64)

key, subkey = jax.random.split(key)
w_batch = jax.random.choice(subkey, state_bank, shape=(NUM_PARALLEL_ENVS,))

initial_runner_state = (
    actor_params, critic_params, opt_actor, opt_critic, 
    w_batch, jnp.zeros(NUM_PARALLEL_ENVS, dtype=jnp.int32), key
)

print(f"Starting Massively Parallel Pure JAX PPO Training...")
start_time = time.time()

runner_state = initial_runner_state
num_chunks = TOTAL_UPDATES // EVAL_INT

for chunk_idx in trange(num_chunks):
    runner_state, batch_metrics = train_chunk(runner_state, state_bank)
    
    current_update = (chunk_idx + 1) * EVAL_INT
    eval_w = state_bank[0]
    eval_e_mean, eval_e_final, crashed = fast_eval_episode(runner_state[0], eval_w, MAX_ENV_STEPS)
    
    current_kl = jnp.mean(batch_metrics["approx_kl"])
    if crashed:
        print(f"\nUpdate {current_update:04d} | Mean: [CRASHED] | Final: [CRASHED] | KL: {current_kl:.4f} | Time: {time.time()-start_time:.1f}s")
    else:
        print(f"\nUpdate {current_update:04d} | Mean: {eval_e_mean:.4f} | Final: {eval_e_final:.4f} | KL: {current_kl:.4f} | Time: {time.time()-start_time:.1f}s")

actor_params_final = runner_state[0]
models_dir = Path('models')
models_dir.mkdir(exist_ok=True)
with open(models_dir / 'ppo_turb_params.msgpack', 'wb') as f:
    f.write(flax.serialization.to_bytes({'actor': actor_params_final}))
print(f"Training finished in {time.time()-start_time:.1f}s. Weights saved.")