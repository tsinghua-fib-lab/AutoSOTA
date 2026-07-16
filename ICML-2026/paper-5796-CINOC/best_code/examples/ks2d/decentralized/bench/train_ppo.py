import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.linen.initializers import orthogonal, constant
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

# Project imports
from examples.ks2d.decentralized.data_utils import get_batch_initial_conditions
from examples.ks2d.decentralized.dynamics_dual import PDEDynamics2D 
from examples.ks2d.decentralized.bench.models_ppo import PPOCritic2DKS # Only importing Critic now

# ==========================================
# 1. CONFIGURATIONS
# ==========================================
N_AGENTS = 100
L_DOMAIN = 32.0
N_GRID = 64

# KS2D Specific Control Timing
MAX_ENV_STEPS = 50     
SUBSTEPS = 10          
DT = 0.005             
NUM_PARALLEL_ENVS = 64

# PPO Specific Configs
ROLLOUT_STEPS = 50     
PPO_EPOCHS = 4
MINIBATCH_SIZE = 1600 
TOTAL_UPDATES = 10000
EVAL_INT = 50        
TARGET_KL = 0.05  # Early stopping threshold

STATE_NORM_FACTOR = 5.0
U_MAX = 5.0            

key = jax.random.PRNGKey(42)

# ==========================================
# 2. ROBUST ACTOR DEFINITION
# ==========================================
class RobustPPOActor2DKS(nn.Module):
    n_agents: int = 100

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)))(x)
        x = nn.relu(x)
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)))(x)
        x = nn.relu(x)
        
        mean = nn.Dense(self.n_agents, kernel_init=orthogonal(0.1))(x)
        
        # Squash the mean strictly into [-U_MAX, U_MAX]
        mean = nn.tanh(mean) * U_MAX 
        
        log_std = self.param('log_std', constant(0.0), (self.n_agents,))
        
        # Clamp log_std to prevent Entropy Collapse (-20 to 2 is standard)
        log_std = jnp.clip(log_std, -20.0, 2.0)
        
        return mean, log_std

# ==========================================
# 3. UTILS & ENVIRONMENT SETUP
# ==========================================
def direct_control_policy(action_params, u_obs, u_target, xi_fixed):
    return action_params

dynamics = PDEDynamics2D(policy_apply_fn=direct_control_policy)

grid_dim = int(np.sqrt(N_AGENTS))
x_lin = np.linspace(0, L_DOMAIN, grid_dim, endpoint=False) + (L_DOMAIN/grid_dim)/2
xv, yv = jnp.meshgrid(x_lin, x_lin)
agent_positions = jnp.stack([xv.flatten(), yv.flatten()], axis=-1)
xi_fixed = jnp.array(agent_positions)
target_state = jnp.zeros((N_GRID, N_GRID))

def get_logprob_and_action(mean, log_std, key=None, action=None):
    std = jnp.exp(log_std)
    if action is None:
        noise = jax.random.normal(key, mean.shape)
        raw_action = mean + noise * std
    else:
        raw_action = action
    
    var = std ** 2
    log_prob = -0.5 * ((raw_action - mean) ** 2) / var - log_std - 0.5 * jnp.log(2 * jnp.pi)
    env_action = jnp.clip(raw_action, -U_MAX, U_MAX)
    
    # Retaining the sum for pure single-agent joint probability
    return env_action, raw_action, jnp.sum(log_prob, axis=-1)

# --- Initialization & LR Annealing ---
actor = RobustPPOActor2DKS(n_agents=N_AGENTS)
critic = PPOCritic2DKS()

key, *subkeys = jax.random.split(key, 4)
dummy_u = jnp.zeros((1, N_GRID, N_GRID))

actor_params = actor.init(subkeys[0], dummy_u)
critic_params = critic.init(subkeys[1], dummy_u)

num_updates = TOTAL_UPDATES // EVAL_INT 
total_training_steps = num_updates * EVAL_INT 

# --- Calculate exact number of optimizer steps ---
dataset_size = NUM_PARALLEL_ENVS * ROLLOUT_STEPS
num_minibatches = dataset_size // MINIBATCH_SIZE
total_optimizer_steps = TOTAL_UPDATES * PPO_EPOCHS * num_minibatches

# FIX 3: Microscopic Actor Learning Rate (Corrected schedules)
lr_schedule_actor = optax.linear_schedule(
    init_value=1e-4, 
    end_value=0.0, 
    transition_steps=total_optimizer_steps
)
lr_schedule_critic = optax.linear_schedule(
    init_value=1e-3, 
    end_value=0.0, 
    transition_steps=total_optimizer_steps
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
    # FIX 4: Use true_next_values instead of overlapping arrays to solve the bootstrap bug
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

def parallel_physics_step(u_batch, actions, xi_fixed):
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
    global_energy = jnp.mean(jnp.square(safe_u), axis=(1, 2))
    
    rewards_batch = jnp.where(dones_batch, -100.0, -(global_energy / 10.0))
    return safe_u, rewards_batch, dones_batch

def train_step(runner_state, state_bank, xi_fixed_pts):
    a_params, c_params, opt_a, opt_c, u_batch, env_counts, rng = runner_state
    
    # --- Rollout Phase ---
    def _env_step(carry, _):
        u, counts, k = carry
        k, act_k, reset_k = jax.random.split(k, 3)
        u_norm = u / STATE_NORM_FACTOR
        
        mean, log_std = actor.apply(a_params, u_norm)
        env_actions, raw_actions, log_probs = get_logprob_and_action(mean, log_std, key=act_k)
        values = critic.apply(c_params, u_norm).squeeze(-1)
        
        next_u, rewards, dones = parallel_physics_step(u, env_actions, xi_fixed_pts)
        
        counts += 1
        truncs = counts >= MAX_ENV_STEPS
        needs_reset = jnp.logical_or(dones, truncs)
        
        # Calculate true next value BEFORE overriding with resets
        true_next_u_norm = next_u / STATE_NORM_FACTOR
        true_next_v = critic.apply(c_params, true_next_u_norm).squeeze(-1)
        
        fresh_states = jax.random.choice(reset_k, state_bank, shape=(NUM_PARALLEL_ENVS,))
        next_u_final = jnp.where(needs_reset[:, None, None], fresh_states, next_u)
        next_counts = jnp.where(needs_reset, 0, counts)
        
        transition = (u, raw_actions, rewards, values, log_probs, dones, needs_reset, true_next_v)
        return (next_u_final, next_counts, k), transition

    carry = (u_batch, env_counts, rng)
    carry, transitions = jax.lax.scan(_env_step, carry, None, length=ROLLOUT_STEPS)
    (next_u_batch, next_env_counts, rng) = carry
    t_u, t_raw_a, t_r, t_v, t_logp, t_d, t_reset, t_true_next_v = transitions
    
    adv, ret = compute_gae_jax(t_r, t_v, t_d, t_reset, t_true_next_v)
    
    f_u = t_u.reshape(-1, N_GRID, N_GRID)
    f_a = t_raw_a.reshape(-1, N_AGENTS)
    f_logp = t_logp.reshape(-1) 
    f_ret = ret.reshape(-1)
    f_adv = adv.reshape(-1)
    f_v = t_v.reshape(-1)
    
    f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)
    dataset_size = f_u.shape[0]
    num_minibatches = dataset_size // MINIBATCH_SIZE

    # --- Optimization Phase ---
    def _update_epoch(epoch_carry, _):
        ap, cp, oa, oc, update_actor_flag, k = epoch_carry
        k, subk = jax.random.split(k)
        
        indices = jax.random.permutation(subk, dataset_size)
        s_u, s_a = f_u[indices], f_a[indices]
        s_logp, s_ret, s_adv, s_v = f_logp[indices], f_ret[indices], f_adv[indices], f_v[indices]
        
        mb_u = s_u.reshape((num_minibatches, MINIBATCH_SIZE, *s_u.shape[1:]))
        mb_a = s_a.reshape((num_minibatches, MINIBATCH_SIZE, *s_a.shape[1:]))
        mb_logp = s_logp.reshape((num_minibatches, MINIBATCH_SIZE, *s_logp.shape[1:]))
        mb_ret = s_ret.reshape((num_minibatches, MINIBATCH_SIZE, *s_ret.shape[1:]))
        mb_adv = s_adv.reshape((num_minibatches, MINIBATCH_SIZE, *s_adv.shape[1:]))
        mb_v = s_v.reshape((num_minibatches, MINIBATCH_SIZE, *s_v.shape[1:])) 
        
        def _update_minibatch(mb_carry, mb_data):
            ap_, cp_, oa_, oc_, is_actor_updating = mb_carry
            mb_u_, mb_a_, mb_logp_, mb_ret_, mb_adv_, mb_v_ = mb_data
            
            b_u_norm = mb_u_ / STATE_NORM_FACTOR
            
            # Minibatch-level advantage normalization stabilizes the joint 100D updates
            mb_adv_ = (mb_adv_ - mb_adv_.mean()) / (mb_adv_.std() + 1e-8)
            
            def loss_fn(ap_tgt, cp_tgt):
                mean, log_std = actor.apply(ap_tgt, b_u_norm) 
                _, _, new_logp = get_logprob_and_action(mean, log_std, action=mb_a_)
                
                entropy = jnp.sum(log_std + 0.5 + 0.5 * jnp.log(2 * jnp.pi), axis=-1).mean()
                
                log_ratio = new_logp - mb_logp_
                
                # Clamp the log_ratio to prevent exp() from creating massive NaNs/Infs
                log_ratio_safe = jnp.clip(log_ratio, -5.0, 2.0)
                ratio = jnp.exp(log_ratio_safe)
                
                approx_kl = jnp.mean((ratio - 1.0) - log_ratio)
                
                pg_loss1 = -mb_adv_ * ratio
                pg_loss2 = -mb_adv_ * jnp.clip(ratio, 1.0 - 0.2, 1.0 + 0.2)
                
                # Entropy bonus to 0.001 to prevent the network from adding too much noise
                actor_loss = jnp.maximum(pg_loss1, pg_loss2).mean() - 0.001 * entropy
                
                values = critic.apply(cp_tgt, b_u_norm).squeeze(-1) 
                v_clipped = mb_v_ + jnp.clip(values - mb_v_, -0.2, 0.2)
                v_loss_unclipped = (values - mb_ret_) ** 2
                v_loss_clipped = (v_clipped - mb_ret_) ** 2
                critic_loss = 0.5 * jnp.mean(jnp.maximum(v_loss_unclipped, v_loss_clipped))
            
                return actor_loss + 0.5 * critic_loss, jnp.stack([actor_loss, critic_loss, entropy, approx_kl])

            (total_loss, metrics), grads = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(ap_, cp_)
            current_kl = metrics[3]
            
            # Determine if KL is still safe
            should_update_actor = jnp.logical_and(is_actor_updating, current_kl < TARGET_KL)
            
            # JAX conditional to either apply gradients or skip actor update
            def do_update():
                u_a, new_oa = tx_actor.update(grads[0], oa_)
                return optax.apply_updates(ap_, u_a), new_oa
            def skip_update():
                return ap_, oa_
            
            ap_n, oa_n = jax.lax.cond(should_update_actor, do_update, skip_update)
            
            # Critic always updates regardless of actor KL
            up_c, oc_n = tx_critic.update(grads[1], oc_)
            cp_n = optax.apply_updates(cp_, up_c)
            
            return (ap_n, cp_n, oa_n, oc_n, should_update_actor), metrics
            
        (ap, cp, oa, oc, update_actor_flag), epoch_metrics = jax.lax.scan(
            _update_minibatch, (ap, cp, oa, oc, update_actor_flag), 
            (mb_u, mb_a, mb_logp, mb_ret, mb_adv, mb_v)
        )
        return (ap, cp, oa, oc, update_actor_flag, k), epoch_metrics

    # Start every epoch with actor updating set to True
    epoch_carry = (a_params, c_params, opt_a, opt_c, jnp.bool_(True), rng)
    epoch_carry, ppo_metrics = jax.lax.scan(_update_epoch, epoch_carry, None, length=PPO_EPOCHS)
    (a_params, c_params, opt_a, opt_c, _, rng) = epoch_carry

    new_runner_state = (a_params, c_params, opt_a, opt_c, next_u_batch, next_env_counts, rng)
    
    metrics = {
        "mean_return": t_r.sum(axis=0).mean(),
        "actor_loss": ppo_metrics[..., 0].mean(),
        "critic_loss": ppo_metrics[..., 1].mean(),
        "entropy": ppo_metrics[..., 2].mean(),
        "approx_kl": ppo_metrics[..., 3].mean()
    }
    
    return new_runner_state, metrics

@jax.jit(donate_argnums=(0,))
def train_chunk(runner_state, state_bank, xi_fixed_pts): 
    def scan_step(carry, _):
        new_state, metrics = train_step(carry, state_bank, xi_fixed_pts) 
        return new_state, metrics
    return jax.lax.scan(scan_step, runner_state, None, length=EVAL_INT)

@partial(jax.jit, static_argnames=['max_steps'])
def fast_eval_episode(a_params, init_u, xi_fixed_pts, max_steps):
    def step_fn(state, _):
        u_norm = state / STATE_NORM_FACTOR
        mean, _ = actor.apply(a_params, u_norm[None, ...])
        act_flat = mean.squeeze(0)
        
        traj = dynamics.unroll_controlled(
            u_init=state, xi_fixed=xi_fixed_pts, u_target=target_state, params=act_flat, 
            t_steps=1, substeps=SUBSTEPS, N_grid=N_GRID, L=L_DOMAIN, dt=DT, sigma=1.2
        )
        next_state = traj[0][-1]
        energy = jnp.mean(next_state**2)
        crashed = jnp.isnan(next_state).any() | jnp.isinf(next_state).any() | (jnp.max(jnp.abs(next_state)) > 100.0)
        return next_state, (energy, crashed)

    _, (energies, crashes) = jax.lax.scan(step_fn, init_u, None, length=max_steps)
    return jnp.mean(energies), jnp.any(crashes)

# ==========================================
# 5. EXECUTION LOOP
# ==========================================
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

initial_runner_state = (
    actor_params, critic_params, opt_actor, opt_critic, 
    u_batch, jnp.zeros(NUM_PARALLEL_ENVS), key
)

print("Starting Massively Parallel Pure JAX PPO Training (Chunked 2D)...")
start_time = time.time()

runner_state = initial_runner_state
num_chunks = TOTAL_UPDATES // EVAL_INT

for chunk in trange(num_chunks):
    eval_u = state_bank[0]
    eval_energy, crashed = fast_eval_episode(runner_state[0], eval_u, xi_fixed, MAX_ENV_STEPS)
    
    current_update = chunk * EVAL_INT
    status = "[CRASHED]" if crashed else f"{eval_energy:.6f}"
    print(f"\nUpdate {current_update:04d} | Eval Energy: {status} | Time: {time.time()-start_time:.1f}s")

    runner_state, batch_metrics = train_chunk(runner_state, state_bank, xi_fixed)
    current_kl = jnp.mean(batch_metrics["approx_kl"]) 
    print(f"Update {current_update:04d} | Eval Energy: {status} | KL: {current_kl:.4f}")

eval_u = state_bank[0]
eval_energy, crashed = fast_eval_episode(runner_state[0], eval_u, xi_fixed, MAX_ENV_STEPS)
status = "[CRASHED]" if crashed else f"{eval_energy:.6f}"
print(f"\nUpdate {TOTAL_UPDATES:04d} | Eval Energy: {status} | Time: {time.time()-start_time:.1f}s")

actor_params_final = runner_state[0]
models_dir = Path('models')
models_dir.mkdir(exist_ok=True)
with open(models_dir / 'ppo_ks2d_params.msgpack', 'wb') as f:
    f.write(flax.serialization.to_bytes({'actor': actor_params_final}))
print(f"Training finished in {time.time()-start_time:.1f}s. Weights saved.")