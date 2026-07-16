import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.80'

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

# Enable x64 globally (Crucial for Spectral Stability)
jax.config.update("jax_enable_x64", True)

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(script_dir))

# Project imports
from env_turb2d import Turbulence2DMARLEnv
from examples.turbulence2d.decentralized.data_utils import get_batch_initial_conditions
import tesseracts.turbulence2d.solver as solver

# Import the new DeepONet MAPPO models
from models_deeponet_mappo import DeepONetMAPPOActor, DeepONetMAPPOCritic

# ==========================================
# 1. CONFIGURATIONS
# ==========================================
N_AGENTS = 64          
L_DOMAIN = 1.0         
N_GRID = 64
U_MAX = 75.0           
PATCH_SIZE = 20        

MAX_ENV_STEPS = 150    
SUBSTEPS = 5           
DT = 0.01              
VISCOSITY = 5e-4       
SIGMA = 0.05           

NUM_PARALLEL_ENVS = 64 
ROLLOUT_STEPS = 150 
PPO_EPOCHS = 4
MINIBATCH_SIZE = 200 
TOTAL_UPDATES = 1000
EVAL_INT = 1

GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.01
LR = 3e-4

key = jax.random.PRNGKey(42)

# ==========================================
# 2. GLOBAL PRECOMPUTATION
# ==========================================
kx, ky, k_sq, k_inv = solver.get_spectral_grid(N_GRID, L_DOMAIN)
dt_phys = DT / SUBSTEPS

grid_dim = int(np.sqrt(N_AGENTS))
x_c = jnp.linspace(0, L_DOMAIN, grid_dim, endpoint=False) + L_DOMAIN/(2*grid_dim)
y_c = jnp.linspace(0, L_DOMAIN, grid_dim, endpoint=False) + L_DOMAIN/(2*grid_dim)
xv, yv = jnp.meshgrid(x_c, y_c)
centers_flat = jnp.stack([xv.flatten(), yv.flatten()], axis=1)

forcing_hat = solver.compute_forcing_profile(
    centers_flat[:, 0], centers_flat[:, 1], N_GRID, L_DOMAIN, SIGMA
)

dummy_pool = jnp.zeros((1, N_GRID, N_GRID), dtype=jnp.float64)
env = Turbulence2DMARLEnv(
    initial_conditions=dummy_pool, n_agents=N_AGENTS, 
    N_grid=N_GRID, L=L_DOMAIN, dt=DT, viscosity=VISCOSITY, 
    substeps=SUBSTEPS, max_steps=MAX_ENV_STEPS, sigma=SIGMA
)

# Extract and store normalized coordinates
xi_norm = jnp.array(env.xi_norm, dtype=jnp.float32)

# ==========================================
# 3. PATCH EXTRACTION & PHYSICS
# ==========================================
@jax.jit
def build_marl_obs_batch(full_state_batch):
    def single_env_obs(w_curr):
        grads = jnp.gradient(w_curr)
        grad_y, grad_x = grads[0], grads[1]

        half_patch = PATCH_SIZE // 2
        
        # MEMORY OPTIMIZATION: Pad the grids ONCE globally
        w_pad = jnp.pad(w_curr, ((half_patch, half_patch), (half_patch, half_patch)), mode='wrap')
        gx_pad = jnp.pad(grad_x, ((half_patch, half_patch), (half_patch, half_patch)), mode='wrap')
        gy_pad = jnp.pad(grad_y, ((half_patch, half_patch), (half_patch, half_patch)), mode='wrap')

        def get_local_obs(xi_single):
            i = (xi_single[1] * N_GRID).astype(jnp.int32) 
            j = (xi_single[0] * N_GRID).astype(jnp.int32)
            
            p_w  = jax.lax.dynamic_slice(w_pad, (i, j), (PATCH_SIZE, PATCH_SIZE))
            p_gx = jax.lax.dynamic_slice(gx_pad, (i, j), (PATCH_SIZE, PATCH_SIZE))
            p_gy = jax.lax.dynamic_slice(gy_pad, (i, j), (PATCH_SIZE, PATCH_SIZE))
            
            return jnp.stack([p_w, p_gx, p_gy], axis=-1)

        local_patches = jax.vmap(get_local_obs)(xi_norm)
        return (local_patches / 50.0).astype(jnp.float32)
        
    return jax.vmap(single_env_obs)(full_state_batch)

@jax.jit
def parallel_marl_physics_step(w_batch, actions):
    acts_flat = actions.squeeze(-1).astype(jnp.float64)
    
    def single_physics_step(w_single, act_single):
        w_hat = jnp.fft.fft2(w_single)
        def rk4_loop(i, w):
            return solver.rk4_step(w, dt_phys, kx, ky, k_sq, k_inv, VISCOSITY, forcing_hat, act_single)
        w_hat_next = jax.lax.fori_loop(0, SUBSTEPS, rk4_loop, w_hat)
        return jnp.fft.ifft2(w_hat_next).real
    
    next_w_batch = jax.vmap(single_physics_step)(w_batch, acts_flat)
    
    is_invalid = jnp.logical_not(jnp.isfinite(next_w_batch).all(axis=(1, 2)))
    is_exploding = jnp.max(jnp.abs(next_w_batch), axis=(1, 2)) > 1000.0
    dones_batch = jnp.logical_or(is_invalid, is_exploding)
    
    safe_w = jnp.where(dones_batch[:, None, None], jnp.zeros_like(next_w_batch), next_w_batch)
        
    global_enstrophy = jnp.mean(jnp.square(safe_w), axis=(1, 2))
    global_reward = -jnp.log(global_enstrophy + 1.0)
    
    grid_dim_sq = int(np.sqrt(N_AGENTS)) 
    cell_size = N_GRID // grid_dim_sq    
    
    w_blocks = safe_w.reshape(-1, grid_dim_sq, cell_size, grid_dim_sq, cell_size)
    w_blocks = w_blocks.swapaxes(2, 3) 
    
    local_enstrophy = jnp.mean(jnp.square(w_blocks), axis=(3, 4)).reshape(-1, N_AGENTS)
    local_reward = -jnp.log(local_enstrophy + 1.0)
    
    action_penalty = -1e-3 * jnp.mean(jnp.square(actions / U_MAX), axis=-1)

    mixed_reward = (0.3 * global_reward[:, None]) + (0.7 * local_reward) + action_penalty
    
    rewards_batch = mixed_reward[..., None].astype(jnp.float32) 
    dones_batch_exp = jnp.tile(dones_batch[:, None, None], (1, N_AGENTS, 1)).astype(jnp.float32)
    
    return safe_w, rewards_batch, dones_batch_exp

# ==========================================
# 4. PPO CORE FUNCTIONS
# ==========================================
def get_action_and_value(actor_params, critic_params, patches, xi_n, key):
    # ACTOR -> Squashed Gaussian
    mean_raw, log_std = actor.apply(actor_params, patches, xi_n)
    std = jnp.exp(log_std)
    
    unscaled_action = mean_raw + std * jax.random.normal(key, mean_raw.shape)
    env_action = jnp.tanh(unscaled_action) * U_MAX
    
    # Jacobian correction for Tanh Squashing
    log_prob_normal = -0.5 * jnp.sum(jnp.square((unscaled_action - mean_raw) / std) + 2 * log_std + jnp.log(2 * jnp.pi), axis=-1)
    log_prob_correction = jnp.sum(jnp.log(U_MAX * (1.0 - jnp.square(jnp.tanh(unscaled_action))) + 1e-6), axis=-1)
    
    log_prob = log_prob_normal - log_prob_correction
    
    # CRITIC -> Decentralized Value
    val = critic.apply(critic_params, patches, xi_n)
    return unscaled_action, env_action, log_prob, val

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
    patches, xi_n, unscaled_actions, old_log_probs, advantages, returns, old_values = batch    
    
    def actor_loss_fn(params):
        mean_raw, log_std = actor.apply(params, patches, xi_n) 
        std = jnp.exp(log_std)
        
        # Reconstruct Log Probs with updated parameters
        log_prob_normal = -0.5 * jnp.sum(jnp.square((unscaled_actions - mean_raw) / std) + 2 * log_std + jnp.log(2 * jnp.pi), axis=-1)
        log_prob_correction = jnp.sum(jnp.log(U_MAX * (1.0 - jnp.square(jnp.tanh(unscaled_actions))) + 1e-6), axis=-1)
        log_probs = log_prob_normal - log_prob_correction
        
        entropy = jnp.sum(0.5 + 0.5 * jnp.log(2 * jnp.pi) + log_std, axis=-1)
        ratio = jnp.exp(log_probs - old_log_probs)
        
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * jnp.clip(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()
        
        entropy_loss = entropy.mean()
        return pg_loss - ENTROPY_COEF * entropy_loss, (pg_loss, entropy_loss)
        
    def value_loss_fn(params):
        v = critic.apply(params, patches, xi_n) 
        
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
# 5. TRAINING LOGIC
# ==========================================
@jax.jit
def train_step(runner_state, state_bank):
    actor_state, critic_state, w_batch, obs_patches, env_counts, rng = runner_state
    
    xi_norm_tiled = jnp.tile(xi_norm[None, :, :], (NUM_PARALLEL_ENVS, 1, 1))

    # 1. Rollout Phase
    def _env_step(carry, _):
        w, patches, counts, k = carry
        k, act_k, reset_k = jax.random.split(k, 3)
        
        unscaled_action, env_action, log_prob, val = get_action_and_value(actor_state.params, critic_state.params, patches, xi_norm_tiled, act_k)
        
        next_w, rewards, crashes = parallel_marl_physics_step(w, env_action)
        
        next_patches = build_marl_obs_batch(next_w)
        true_next_v = critic.apply(critic_state.params, next_patches, xi_norm_tiled)
        
        counts += 1
        truncs = counts >= MAX_ENV_STEPS
        needs_reset = jnp.logical_or(crashes[:, 0, 0], truncs)
        
        fresh_states = jax.random.choice(reset_k, state_bank, shape=(NUM_PARALLEL_ENVS,))
        next_w_final = jnp.where(needs_reset[:, None, None], fresh_states, next_w)
        next_counts = jnp.where(needs_reset, 0, counts)
        
        next_patches_final = build_marl_obs_batch(next_w_final)
        
        # Store global 'w' (very lightweight) instead of patches
        transition = (w, unscaled_action, log_prob, rewards, val, crashes, true_next_v)
        return (next_w_final, next_patches_final, next_counts, k), transition

    carry = (w_batch, obs_patches, env_counts, rng)
    carry, transitions = jax.lax.scan(_env_step, carry, None, length=ROLLOUT_STEPS)
    (next_w_batch, next_patches_batch, next_env_counts, rng) = carry
    
    t_w, t_a_unscaled, t_logp, t_r, t_v, t_d, t_true_next_v = transitions
    
    # 2. GAE Phase
    last_val = critic.apply(critic_state.params, next_patches_batch, xi_norm_tiled)
    adv, ret = compute_gae_jax(t_r, t_v, t_d, t_true_next_v, last_val, GAMMA, GAE_LAMBDA)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    
    # 3. Optimization Phase Setup
    f_w = t_w.reshape(-1, N_GRID, N_GRID) 
    f_a = t_a_unscaled.reshape(-1, N_AGENTS, 1)
    f_logp = t_logp.reshape(-1, N_AGENTS)
    f_adv = adv.reshape(-1, N_AGENTS)
    f_ret = ret.reshape(-1, N_AGENTS, 1)
    f_v = t_v.reshape(-1, N_AGENTS, 1)

    dataset_size = f_w.shape[0]
    num_minibatches = dataset_size // MINIBATCH_SIZE
    xi_norm_tiled_mb = jnp.tile(xi_norm[None, :, :], (MINIBATCH_SIZE, 1, 1))

    # 4. Optimization Loop
    def _update_epoch(epoch_carry, _):
        a_state, c_state, k = epoch_carry
        k, subk = jax.random.split(k)
        
        indices = jax.random.permutation(subk, dataset_size)
        
        mb_w = f_w[indices].reshape((num_minibatches, MINIBATCH_SIZE, *f_w.shape[1:]))
        mb_a = f_a[indices].reshape((num_minibatches, MINIBATCH_SIZE, *f_a.shape[1:]))
        mb_logp = f_logp[indices].reshape((num_minibatches, MINIBATCH_SIZE, *f_logp.shape[1:]))
        mb_ret = f_ret[indices].reshape((num_minibatches, MINIBATCH_SIZE, *f_ret.shape[1:]))
        mb_adv = f_adv[indices].reshape((num_minibatches, MINIBATCH_SIZE, *f_adv.shape[1:]))
        mb_v = f_v[indices].reshape((num_minibatches, MINIBATCH_SIZE, *f_v.shape[1:]))

        def _update_minibatch(mb_carry, mb_data):
            a_state_, c_state_ = mb_carry
            mb_w_, mb_a_unscaled_, mb_logp_, mb_ret_, mb_adv_, mb_v_ = mb_data
            
            # Compute patches dynamically ON THE MINIBATCH
            mb_p_ = build_marl_obs_batch(mb_w_) 
                        
            mb_batch = (mb_p_, xi_norm_tiled_mb, mb_a_unscaled_, mb_logp_, mb_adv_, mb_ret_, mb_v_)
            a_state_n, c_state_n, al, vl, ent = ppo_update_epoch(a_state_, c_state_, mb_batch)
            return (a_state_n, c_state_n), jnp.stack([al, vl, ent])
            
        (a_state, c_state), epoch_metrics = jax.lax.scan(
            _update_minibatch, (a_state, c_state), 
            (mb_w, mb_a, mb_logp, mb_ret, mb_adv, mb_v) 
        )
        return (a_state, c_state, k), epoch_metrics

    epoch_carry = (actor_state, critic_state, rng)
    epoch_carry, ppo_metrics = jax.lax.scan(_update_epoch, epoch_carry, None, length=PPO_EPOCHS)
    (actor_state, critic_state, rng) = epoch_carry

    new_runner_state = (actor_state, critic_state, next_w_batch, next_patches_batch, next_env_counts, rng)
    
    metrics = {
        "mean_return": t_r.sum(axis=0).mean(),
        "actor_loss": ppo_metrics[..., 0].mean(),
        "critic_loss": ppo_metrics[..., 1].mean()
    }
    return new_runner_state, metrics

@partial(jax.jit, donate_argnums=(0,))
def train_chunk(runner_state, state_bank):
    def scan_step(carry, _):
        new_state, metrics = train_step(carry, state_bank)
        return new_state, metrics
    return jax.lax.scan(scan_step, runner_state, None, length=EVAL_INT)

# ==========================================
# 6. FAST EVALUATION
# ==========================================
@partial(jax.jit, static_argnames=['max_steps'])
def fast_eval_episode(actor_params, init_w, max_steps):
    def step_fn(state, _):
        obs_patches = build_marl_obs_batch(state[None, ...]) 
        xi_norm_expanded = jnp.tile(xi_norm[None, :, :], (obs_patches.shape[0], 1, 1))
        
        mean_raw, _ = actor.apply(actor_params, obs_patches, xi_norm_expanded)
        
        # Deterministic evaluation using bounded Tanh
        env_action = jnp.tanh(mean_raw) * U_MAX
        act_flat = env_action[0, :, 0].astype(jnp.float64)
        
        w_hat = jnp.fft.fft2(state)
        def rk4_loop(i, w):
            return solver.rk4_step(w, dt_phys, kx, ky, k_sq, k_inv, VISCOSITY, forcing_hat, act_flat)
        w_hat_next = jax.lax.fori_loop(0, SUBSTEPS, rk4_loop, w_hat)
        next_w = jnp.fft.ifft2(w_hat_next).real
        
        enstrophy = jnp.mean(next_w**2)
        crashed = jnp.isnan(next_w).any() | jnp.isinf(next_w).any() | (jnp.max(jnp.abs(next_w)) > 1000.0)
        
        return next_w, (enstrophy, crashed)

    _, (enstrophies, crashes) = jax.lax.scan(step_fn, init_w, None, length=max_steps)
    return jnp.mean(enstrophies), enstrophies[-1], jnp.any(crashes)

# ==========================================
# 7. EXECUTION LOOP
# ==========================================
actor = DeepONetMAPPOActor(n_agents=N_AGENTS, u_max=U_MAX)
critic = DeepONetMAPPOCritic(n_agents=N_AGENTS)

key, act_k, val_k = jax.random.split(key, 3)

# Initializer Shapes
dummy_patches = jnp.zeros((1, N_AGENTS, PATCH_SIZE, PATCH_SIZE, 3), dtype=jnp.float32)
dummy_xi = jnp.zeros((1, N_AGENTS, 2), dtype=jnp.float32)

total_rollout_steps = NUM_PARALLEL_ENVS * ROLLOUT_STEPS
optax_steps_per_update = (total_rollout_steps // MINIBATCH_SIZE) * PPO_EPOCHS
total_optax_steps = TOTAL_UPDATES * optax_steps_per_update

lr_schedule = optax.linear_schedule(init_value=LR, end_value=0.0, transition_steps=total_optax_steps)

actor_state = TrainState.create(
    apply_fn=actor.apply,
    params=actor.init(act_k, dummy_patches, dummy_xi),
    tx=optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr_schedule, eps=1e-5))
)

critic_state = TrainState.create(
    apply_fn=critic.apply,
    params=critic.init(val_k, dummy_patches, dummy_xi),
    tx=optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr_schedule, eps=1e-5))
)

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
obs_patches_batch = build_marl_obs_batch(w_batch)

initial_runner_state = (
    actor_state, critic_state, w_batch, obs_patches_batch,
    jnp.zeros(NUM_PARALLEL_ENVS), key
)

print(f"Starting Massively Parallel Decentralized DeepONet MAPPO Training for 2D Turbulence...")
start_time = time.time()

runner_state = initial_runner_state

# Evaluate every 5 PPO updates
EVAL_FREQ = 5 

for update_idx in trange(TOTAL_UPDATES):
    # Run a single PPO rollout & update
    runner_state, chunk_metrics = train_step(runner_state, state_bank)
    
    # Evaluate and print periodically
    if update_idx % EVAL_FREQ == 0:
        eval_w = state_bank[0]
        eval_e_mean, eval_e_final, crashed = fast_eval_episode(runner_state[0].params, eval_w, MAX_ENV_STEPS)
        
        if crashed:
            print(f"\nUpdate {update_idx:04d} | Eval Mean: [CRASHED] | Time: {time.time()-start_time:.1f}s")
        else:
            print(f"\nUpdate {update_idx:04d} | Eval Mean: {eval_e_mean:.4f} | Final: {eval_e_final:.4f} | Time: {time.time()-start_time:.1f}s")

final_actor_params = runner_state[0].params
final_critic_params = runner_state[1].params
models_dir = Path('models')
models_dir.mkdir(exist_ok=True)
with open(models_dir / 'deeponet_mappo_turb_params.msgpack', 'wb') as f:
    f.write(flax.serialization.to_bytes({'actor': final_actor_params, 'critic': final_critic_params}))
print(f"Training finished in {time.time()-start_time:.1f}s. Weights saved.")