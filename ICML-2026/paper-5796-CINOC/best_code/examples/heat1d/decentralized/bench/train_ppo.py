import jax
import jax.numpy as jnp
import optax
import flax.serialization
import time
from pathlib import Path
import sys
from functools import partial
from tqdm import trange

script_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(script_dir))

# --- Updated Heat1D Imports ---
from env_he import HeatHypeMARLEnv 
from models_ppo import PPOActor, PPOCritic, U_MAX, V_MAX
from ppo_utils import get_logprob_and_action
from examples.heat1d.decentralized.data_utils import generate_grf
from examples.heat1d.decentralized.dynamics_dual import PDEDynamics 

# --- Configurations ---
N_AGENTS = 8 # Adjusted to 8 for standard Heat1D setups, change to 20 if needed
N_GRID = 100
MAX_ENV_STEPS = 300
NUM_PARALLEL_ENVS = 256

# PPO Specific Configs
ROLLOUT_STEPS = 128
PPO_EPOCHS = 4
MINIBATCH_SIZE = 1024
TOTAL_UPDATES = 651
EVAL_INT = 10 
CLIP_COEF = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5

key = jax.random.PRNGKey(42)

def direct_control_policy(action_params, u_obs, u_target, xi_fixed):
    return action_params[:, 0], action_params[:, 1]

dynamics = PDEDynamics(policy_apply_fn=direct_control_policy)

# --- Initialization ---
actor = PPOActor(n_agents=N_AGENTS)
critic = PPOCritic()

key, *subkeys = jax.random.split(key, 4)
dummy_z = jnp.zeros((1, N_GRID))
dummy_target = jnp.zeros((1, N_GRID))
dummy_xi = jnp.zeros((1, N_AGENTS))

actor_params = actor.init(subkeys[0], dummy_z, dummy_target, dummy_xi)
critic_params = critic.init(subkeys[1], dummy_z, dummy_target, dummy_xi)

# --- Linear Learning Rate Schedule ---
total_rollout_steps = NUM_PARALLEL_ENVS * ROLLOUT_STEPS
optax_steps_per_update = (total_rollout_steps // MINIBATCH_SIZE) * PPO_EPOCHS
total_optax_steps = TOTAL_UPDATES * optax_steps_per_update

actor_lr_schedule = optax.linear_schedule(init_value=3e-4, end_value=0.0, transition_steps=total_optax_steps)
critic_lr_schedule = optax.linear_schedule(init_value=1e-3, end_value=0.0, transition_steps=total_optax_steps)

tx_actor = optax.chain(optax.clip_by_global_norm(1), optax.adam(actor_lr_schedule))
tx_critic = optax.chain(optax.clip_by_global_norm(1), optax.adam(critic_lr_schedule))
opt_actor = tx_actor.init(actor_params)
opt_critic = tx_critic.init(critic_params)

# --- Pre-generate Starting State Banks ---
print("Pre-generating starting state & target banks...")
bank_keys = jax.random.split(key, 1000)
_, z_init_bank = jax.vmap(partial(generate_grf, n_points=N_GRID, length_scale=0.2))(bank_keys)
_, z_target_bank = jax.vmap(partial(generate_grf, n_points=N_GRID, length_scale=0.4))(bank_keys)
xi_init_single = jnp.linspace(0.2, 0.8, N_AGENTS, dtype=jnp.float32)

# --- JAX-Native GAE ---
@jax.jit
def compute_gae_jax(rewards, values, dones, true_next_values, gamma=0.99, lam=0.95):
    def scan_fn(carry, transition):
        r, v, d, next_v = transition
        gae = carry
        delta = r + gamma * next_v * (1.0 - d) - v
        gae = delta + gamma * lam * (1.0 - d) * gae
        return gae, gae
    
    _, advantages = jax.lax.scan(
        scan_fn, 
        jnp.zeros_like(rewards[0]), 
        (rewards, values, dones, true_next_values), 
        reverse=True
    )
    returns = advantages + values
    return advantages, returns

# --- Parallel Physics Step ---
def parallel_physics_step(z_batch, xi_batch, target_batch, actions, phys_key):
    keys = jax.random.split(phys_key, z_batch.shape[0])
    actions = jnp.clip(actions, -1.0, 1.0) 
    
    def single_physics_step(z_s, xi_s, target_s, act_s, k_s):
        traj = dynamics.unroll_controlled(
            z_init=z_s, xi_init=xi_s, z_target=target_s, params=act_s, t_steps=1
        )
        return traj[0][-1], traj[1][-1]
    
    next_z, next_xi = jax.vmap(single_physics_step)(z_batch, xi_batch, target_batch, actions, keys)
    dones = jnp.logical_not(jnp.isfinite(next_z).all(axis=-1, keepdims=True))
    
    safe_z = jnp.where(dones, jnp.zeros_like(next_z), next_z)
    safe_xi = jnp.where(dones, xi_batch, next_xi)
    
    # --- Blended Tracking Reward ---
    batch_indices = jnp.arange(safe_z.shape[0])[:, None]
    agent_indices = jnp.clip((safe_xi * (N_GRID - 1)).astype(jnp.int32), 0, N_GRID - 1)
    local_z = safe_z[batch_indices, agent_indices]
    local_target = target_batch[batch_indices, agent_indices]
    
    mean_local_mse = jnp.mean(jnp.square(local_z - local_target), axis=-1, keepdims=True)
    global_mse = jnp.mean(jnp.square(safe_z - target_batch), axis=-1, keepdims=True)
    r_track = -10.0 * mean_local_mse - 10.0 * global_mse 
    
    u_act, v_act = actions[..., 0], actions[..., 1]
    r_effort = jnp.mean(-0.001 * (jnp.square(u_act) + 0.1 * jnp.square(v_act)), axis=-1, keepdims=True)
    
    margin = 0.02
    oob_penalty = 100.0 * (jnp.maximum(0.0, margin - safe_xi)**2 + jnp.maximum(0.0, safe_xi - (1.0 - margin))**2)
    r_bound = jnp.mean(-oob_penalty, axis=-1, keepdims=True)
    
    R_safe = 0.05
    dists = jnp.abs(safe_xi[:, :, None] - safe_xi[:, None, :])
    mask = jnp.eye(N_AGENTS)[None, :, :]
    r_coll = jnp.mean(-1.0 * jnp.sum(jnp.maximum(0.0, R_safe - (dists + mask * 10.0)) ** 2, axis=2), axis=-1, keepdims=True)
    
    rewards = (r_track + r_effort + r_bound + r_coll) * 0.05 
    return safe_z, safe_xi, rewards, dones

# --- Minibatch Update Logic ---
def update_ppo_minibatch(a_params, c_params, opt_a, opt_c, b_z, b_zt, b_xi, b_a, b_logp, b_ret, b_adv, b_v):
    def loss_fn(ap, cp):
        mean, log_std = actor.apply(ap, b_z, b_zt, b_xi)
        _, new_logp = get_logprob_and_action(mean, log_std, action=b_a)
        entropy = jnp.sum(log_std + 0.5 + 0.5 * jnp.log(2 * jnp.pi), axis=(-1, -2)).mean()
        
        ratio = jnp.exp(new_logp - b_logp)
        pg_loss1 = -b_adv * ratio
        pg_loss2 = -b_adv * jnp.clip(ratio, 1.0 - CLIP_COEF, 1.0 + CLIP_COEF)
        actor_loss = jnp.maximum(pg_loss1, pg_loss2).mean() - ENT_COEF * entropy
        
        values = critic.apply(cp, b_z, b_zt, b_xi).squeeze(-1)
        v_clipped = b_v + jnp.clip(values - b_v, -CLIP_COEF, CLIP_COEF)
        v_loss_unclipped = (values - b_ret) ** 2
        v_loss_clipped = (v_clipped - b_ret) ** 2
        critic_loss = VF_COEF * jnp.maximum(v_loss_unclipped, v_loss_clipped).mean()
        
        return actor_loss + critic_loss, (actor_loss, critic_loss, entropy)

    (total_loss, metrics), grads = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(a_params, c_params)
    up_a, opt_a = tx_actor.update(grads[0], opt_a)
    up_c, opt_c = tx_critic.update(grads[1], opt_c)
    return optax.apply_updates(a_params, up_a), optax.apply_updates(c_params, up_c), opt_a, opt_c, metrics

# --- THE PURE JAX PPO TRAIN STEP ---
def train_step(runner_state):
    a_params, c_params, opt_a, opt_c, z_batch, target_batch, xi_batch, env_counts, rng = runner_state
    
    def _env_step(carry, _):
        z, zt, xi, counts, k = carry
        k, act_k, phys_k, reset_k = jax.random.split(k, 4)
        
        mean, log_std = actor.apply(a_params, z, zt, xi)
        actions, log_probs = get_logprob_and_action(mean, log_std, key=act_k)
        values = critic.apply(c_params, z, zt, xi).squeeze(-1)
        
        next_z, next_xi, rewards, dones = parallel_physics_step(z, xi, zt, actions, phys_k)
        true_next_v = critic.apply(c_params, next_z, zt, next_xi).squeeze(-1)
        
        counts += 1
        truncs = counts >= MAX_ENV_STEPS
        needs_reset = jnp.logical_or(dones.flatten(), truncs)
        
        idx_reset = jax.random.randint(reset_k, (NUM_PARALLEL_ENVS,), 0, 1000)
        fresh_z, fresh_target = z_init_bank[idx_reset], z_target_bank[idx_reset]
        fresh_xi = jnp.tile(xi_init_single, (NUM_PARALLEL_ENVS, 1))
        
        next_z = jnp.where(needs_reset[:, None], fresh_z, next_z)
        next_zt = jnp.where(needs_reset[:, None], fresh_target, zt)
        next_xi = jnp.where(needs_reset[:, None], fresh_xi, next_xi)
        next_counts = jnp.where(needs_reset, 0, counts)
        
        transition = (z, zt, xi, actions, rewards.squeeze(-1), values, log_probs, dones.squeeze(-1), true_next_v)
        return (next_z, next_zt, next_xi, next_counts, k), transition

    carry = (z_batch, target_batch, xi_batch, env_counts, rng)
    carry, transitions = jax.lax.scan(_env_step, carry, None, length=ROLLOUT_STEPS)
    (next_z_batch, next_target_batch, next_xi_batch, next_env_counts, rng) = carry
    t_z, t_zt, t_xi, t_a, t_r, t_v, t_logp, t_d, t_true_next_v = transitions
    
    adv, ret = compute_gae_jax(t_r, t_v, t_d, t_true_next_v)
    
    f_z, f_zt, f_xi = t_z.reshape(-1, N_GRID), t_zt.reshape(-1, N_GRID), t_xi.reshape(-1, N_AGENTS)
    f_a, f_logp, f_ret, f_v = t_a.reshape(-1, N_AGENTS, 2), t_logp.reshape(-1), ret.reshape(-1), t_v.reshape(-1)
    f_adv = adv.reshape(-1)
    f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8) 
    
    dataset_size = f_z.shape[0]

    def _update_epoch(epoch_carry, _):
        ap, cp, oa, oc, k = epoch_carry
        k, subk = jax.random.split(k)
        indices = jax.random.permutation(subk, dataset_size)
        
        def _update_minibatch(mb_carry, start_idx):
            ap_, cp_, oa_, oc_ = mb_carry
            batch_idx = jax.lax.dynamic_slice(indices, (start_idx,), (MINIBATCH_SIZE,))
            ap_n, cp_n, oa_n, oc_n, metrics = update_ppo_minibatch(
                ap_, cp_, oa_, oc_, f_z[batch_idx], f_zt[batch_idx], f_xi[batch_idx], 
                f_a[batch_idx], f_logp[batch_idx], f_ret[batch_idx], f_adv[batch_idx], f_v[batch_idx]
            )
            return (ap_n, cp_n, oa_n, oc_n), metrics
            
        mb_starts = jnp.arange(0, dataset_size, MINIBATCH_SIZE)
        (ap, cp, oa, oc), epoch_metrics = jax.lax.scan(_update_minibatch, (ap, cp, oa, oc), mb_starts)
        return (ap, cp, oa, oc, k), epoch_metrics

    epoch_carry = (a_params, c_params, opt_a, opt_c, rng)
    epoch_carry, ppo_metrics = jax.lax.scan(_update_epoch, epoch_carry, None, length=PPO_EPOCHS)
    (a_params, c_params, opt_a, opt_c, rng) = epoch_carry

    return (a_params, c_params, opt_a, opt_c, next_z_batch, next_target_batch, next_xi_batch, next_env_counts, rng), {
        "mean_return": t_r.sum(axis=0).mean(),
        "actor_loss": ppo_metrics[0].mean(), "critic_loss": ppo_metrics[1].mean()
    }

@jax.jit
def train_chunk(runner_state):
    def scan_step(carry, _):
        return train_step(carry)
    return jax.lax.scan(scan_step, runner_state, None, length=EVAL_INT)

# --- Fast Evaluation ---
@partial(jax.jit, static_argnames=['max_steps'])
def fast_eval_episode(a_params, init_z, init_xi, target_z, max_steps, key):
    def step_fn(state, _):
        z_curr, xi_curr, k = state
        k, subk = jax.random.split(k)
        mean, _ = actor.apply(a_params, z_curr[None, ...], target_z[None, ...], xi_curr[None, ...])
        traj = dynamics.unroll_controlled(z_init=z_curr, xi_init=xi_curr, z_target=target_z, params=mean.squeeze(0), t_steps=1)
        next_z, next_xi = traj[0][-1], traj[1][-1]
        mse = jnp.mean((next_z - target_z)**2)
        crashes = jnp.isnan(next_z).any() | jnp.isinf(next_z).any()
        return (next_z, next_xi, k), (mse, crashes)

    _, (mses, crashes) = jax.lax.scan(step_fn, (init_z, init_xi, key), None, length=max_steps)
    return jnp.mean(mses), jnp.any(crashes)

# --- Python Execution Loop ---
idx = jax.random.randint(jax.random.PRNGKey(42), (NUM_PARALLEL_ENVS,), 0, 1000)
runner_state = (actor_params, critic_params, opt_actor, opt_critic, z_init_bank[idx], z_target_bank[idx], jnp.tile(xi_init_single, (NUM_PARALLEL_ENVS, 1)), jnp.zeros(NUM_PARALLEL_ENVS), key)

print("Starting Optimized Pure JAX Centralized PPO (Heat1D)...")
start_time = time.time()
for chunk in trange(TOTAL_UPDATES // EVAL_INT):
    eval_mse, crashed = fast_eval_episode(runner_state[0], z_init_bank[0], xi_init_single, z_target_bank[0], MAX_ENV_STEPS, jax.random.split(runner_state[-1])[0])
    print(f"Update {chunk*EVAL_INT:04d} | Eval MSE: {'[CRASHED]' if crashed else f'{eval_mse:.6f}'} | Time: {time.time()-start_time:.1f}s")
    runner_state, _ = train_chunk(runner_state)

with open('models/ppo_heat1d_params.msgpack', 'wb') as f:
    f.write(flax.serialization.to_bytes({'actor': runner_state[0]}))