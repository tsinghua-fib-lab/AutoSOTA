# ppo_utils.py
import jax
import jax.numpy as jnp
from functools import partial

@jax.jit
def get_logprob_and_action(mean, log_std, key=None, action=None):
    """Samples an action and computes its log probability."""
    std = jnp.exp(log_std)
    
    if action is None:
        noise = jax.random.normal(key, mean.shape)
        action = mean + noise * std
    
    var = std ** 2
    # Log probability density function of the Normal distribution
    log_prob = -((action - mean) ** 2) / (2 * var) - log_std - jnp.log(jnp.sqrt(2 * jnp.pi))
    
    # Sum over agents and action dimensions to get one scalar log_prob per environment
    total_log_prob = jnp.sum(log_prob, axis=(-1, -2))
    return action, total_log_prob

@partial(jax.jit, static_argnames=['gamma', 'lam'])
def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    """Calculates Generalized Advantage Estimation (GAE) backwards through the trajectory."""
    advantages = jnp.zeros_like(rewards)
    lastgaelam = 0.0
    
    # We use a standard Python loop here because JAX unrolls it during JIT for fixed trajectory lengths
    rollout_steps = rewards.shape[0]
    for t in reversed(range(rollout_steps)):
        if t == rollout_steps - 1:
            nextnonterminal = 1.0 - dones[t]
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t]
            nextvalues = values[t+1]
            
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        advantages = advantages.at[t].set(lastgaelam)
        
    returns = advantages + values
    return advantages, returns