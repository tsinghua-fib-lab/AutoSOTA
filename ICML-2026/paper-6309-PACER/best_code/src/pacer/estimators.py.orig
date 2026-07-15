import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map
import optax
import numpy as np
from collections import defaultdict
from .plackett_luce import PlackettLuce
from .objectives import calculate_interventional_loglikelihoods, expected_adjacency_regularizer, objective_analytic_gaussian, calculate_interventional_loglikelihoods_analytic
import functools

"""
REINFORCE estimator
"""

def compute_z(u, logits):
     log_probs = jax.nn.log_softmax(logits, axis=-1)
     return log_probs - jnp.log(-jnp.log(u))

d_logp_logits_fn = jax.grad(lambda x, b: PlackettLuce(x).log_prob(b), argnums=0)

def reinforce(objective_value_and_grad_fn, u, bernoulli_samples, objective_params, objective_args):
     """
     Returns gradient estimate ĝ, i.e. derivative of an objective function f
     with respect to the logits of the Plackett-Luce model.

     Parameters:
     * logits: Logits of the Plackett-Luce model, shape (dim,)
     * bernoulli_logits: Bernoulli logits, shape (dim, dim)
     * u: Uniform samples, shape (dim,)                  # * pl_samples: Samples from the Plackett-Luce distribution, shape (dim,)
     * bernoulli_samples: Samples from a Bernoulli distribution, shape (dim, dim)
     * objective_value_and_grad_fn: Value and gradient (wrt objective_params) objective function
     * objective_params: Parameters of the objective function
     * objective_args: Additional _static_ arguments of the objective function
     # * objective_fn: function taking a permutation and returning a scalar.
     """
     logits = objective_params['logits']
     bernoulli_logits = objective_params['bernoulli_logits']
     
     z = compute_z(u, logits)
     b = jnp.argsort(z, axis=-1, descending=True)
     fb, grad_params = objective_value_and_grad_fn(b,
                                                   bernoulli_samples,
                                                   objective_params,
                                                   objective_args)
     
     d_logp_logits = d_logp_logits_fn(logits, b)
     d_logbp_bernoulli_logits = bernoulli_samples - jax.nn.sigmoid(bernoulli_logits)
     out = {
          'gradient': fb*d_logp_logits,
          'd_logp_logits': d_logp_logits,
          'd_logbp_bernoulli_logits': d_logbp_bernoulli_logits,
          'fb': fb,
          'grad_params': grad_params,
     }
     return out


def fit_reinforce(key,
                  objective_value_and_grad_fn,
                  objective_args,
                  params,
                  objective_args_val=None,
                  n_steps=10000,
                  learning_rate=0.1,
                  n_mc_samples=30,
                  batch_size=64,
                  average_control_variate=True,
                  sample_batch_fn=None,
                  optimize_params=['logits', 'weight_premask', 'bias_premask', 'layer_weights', 'layer_biases', 'bernoulli_logits'],
                  dict_metric_fns=None,
                  metrics_freq = 200,
                  density='gaussian',
                  multigpu=False,
                  lambd=0
                  ):
     
     dim = params['logits'].shape[-1]
     optimizer = optax.adam(learning_rate)
     opt_state = optimizer.init(params)
     
     scores = defaultdict(list)
     step_scores = [] # train scores
     best_score_val = np.nan
     best_params = None
     dtype = params['logits'].dtype
     eps = jnp.finfo(dtype).eps
     
     # Prepare functions
     vmap_bernoulli_value = 0
     P_bernoulli = P('i', None, None)
     if 'bernoulli_logits' not in optimize_params:
          vmap_bernoulli_value = None
          P_bernoulli = P(None, None)
     reinforce_partial = functools.partial(reinforce, objective_value_and_grad_fn)
     vectorized_reinforce = jax.jit(jax.vmap(reinforce_partial, in_axes=(0, vmap_bernoulli_value, None, None)))
     devices = jax.devices()
     if multigpu and len(devices) > 1:
          m = Mesh(devices, ('i',))
          vectorized_reinforce = jax.shard_map(vectorized_reinforce, mesh=m, in_specs=(P('i', None), P_bernoulli, P(None), P(None)), out_specs=P('i'))
     expected_adjacency_regularizer_fn = jax.value_and_grad(expected_adjacency_regularizer, argnums=0)

     for i in range(n_steps):
          # Batch sampling
          if sample_batch_fn is None:
               step_objective_args = objective_args
          else:
               key, subkey = jax.random.split(key)
               step_objective_args = sample_batch_fn(objective_args, subkey, batch_size=batch_size)
          
          # Random samples for permutation and bernoulli probabilities
          key, u_key, bp_key = jax.random.split(key, num=3)
          if 'logits' in optimize_params:
               u = jax.random.uniform(u_key, (n_mc_samples, dim), minval=eps, maxval=1-eps, dtype=dtype)  # Shape: (dim,)
          else:
               u = jnp.ones((n_mc_samples, dim), dtype=dtype) * jnp.exp(-1)
          if 'bernoulli_logits' in optimize_params:
               bernoulli_samples = jax.random.bernoulli(bp_key, shape=(n_mc_samples, dim, dim), p=jax.nn.sigmoid(params['bernoulli_logits'])).astype(dtype)  # Shape: (n_mc_samples, dim, dim)
          else:
               bernoulli_samples = jnp.ones((dim, dim), dtype=dtype)
          out = vectorized_reinforce(u,
                    bernoulli_samples,
                    params,
                    step_objective_args)
          
          # Regularization term to penalize expected # edges
          reg_term, reg_grad = expected_adjacency_regularizer_fn(params, lambd)
          
          # Computing gradients
          logits_grad = jnp.mean((out['fb'])[..., None] * out['d_logp_logits'], axis=0)
          bernoulli_logits_grad = jnp.mean((out['fb'])[..., None, None] * out['d_logbp_bernoulli_logits'], axis=0)
  
          if average_control_variate:
               control_variate = jnp.mean(out['fb'])
               logits_grad = jnp.mean((out['fb'] - control_variate)[..., None] * out['d_logp_logits'], axis=0)
               bernoulli_logits_grad = jnp.mean((out['fb'] - control_variate)[..., None, None] * out['d_logbp_bernoulli_logits'], axis=0)
               
          grads = out['grad_params']
          grads = jax.tree.map(lambda x: jnp.mean(x, axis=0) if x is not None else 0, grads)
          grads['logits'] = logits_grad + reg_grad['logits']
          grads['bernoulli_logits'] = bernoulli_logits_grad + reg_grad['bernoulli_logits']
          for k, param_grad in grads.items():
               if k not in optimize_params:
                    grads[k] = jax.tree.map(lambda x: jnp.zeros_like(x) if x is not None else 0, param_grad)

          # Gradient update
          updates, opt_state = optimizer.update(grads, opt_state, params)
          params = optax.apply_updates(params, updates)
          
          score = np.array(np.mean(out['fb']) + reg_term)
          step_scores.append(score)
          scores['score'].append(score)
          
          # Printing scores
          if (i + 1) % metrics_freq == 0:
               # Train score
               score = np.array(step_scores).mean()
               print_str = f'Step {i}: Train {score:.3f}'
               # Printing training metrics
               if dict_metric_fns is not None:
                    metrics_str = ''          
                    for k, v in dict_metric_fns.items():
                         scores[k].append(v(params))
                         metrics_str += f' | {k}: {scores[k][-1]:.3f}'
                    print_str += f'{metrics_str}'
               step_scores = []
               
               # Validation score
               if objective_args_val is not None:
                    ills_out = calculate_interventional_loglikelihoods(params,
                                                                       objective_args_val,
                                                                       density)
                    ills = ills_out['unique_regimes_ills']
                    score_val = jnp.mean(-ills)
                    max_score = jnp.max(-ills)
                    scores['score_val'].append(score_val)
                    print_str += f' |  Validation {score_val:.3f}   Max: {max_score:.3f}'

                    if np.isnan(best_score_val) or score_val < best_score_val:
                         best_params = params.copy()
                         best_score_val = score_val
               print(print_str)
     if best_params is None:
          best_params = params.copy()

     return {
          'params': params,
          'scores': scores,
          'best_params': best_params
     }



"""
Analytic estimator
"""

def fit_analytic(key,
               objective_args,
               params,
               objective_args_val=None,
               n_steps=10000,
               learning_rate=0.1,
               batch_size=64,
               sample_batch_fn=None,
               optimize_params=['logits', 'weight_premask', 'bias_premask', 'bernoulli_logits'],
               dict_metric_fns=None,
               metrics_freq = 200,
               density='gaussian',
               covariance_var_fraction=1,
               lambd=0
               ):
     assert density == 'gaussian'
     optimizer = optax.adam(learning_rate)
     opt_state = optimizer.init(params)
     
     scores = defaultdict(list)
     step_scores = np.zeros(metrics_freq) # train scores
     best_score_val = np.nan
     best_params = None
     
     expected_adjacency_regularizer_fn = jax.value_and_grad(expected_adjacency_regularizer, argnums=0)
     expected_score_fn = jax.value_and_grad(objective_analytic_gaussian, argnums=0)

     num_genes = objective_args[0].shape[1]
     
     """unique, counts = np.unique(objective_args[2], return_counts=True)
     regime_size = dict(zip(unique, counts))
     num_groups = len(unique)
     p = np.array([1 / (num_groups * regime_size[r]) for r in objective_args[2]])
     """
     
     for i in range(n_steps):
          # Batch sampling
          if sample_batch_fn is None:
               step_objective_args = objective_args[0], objective_args[1] # remove regimes
          else:
               key, subkey = jax.random.split(key)
               # step_objective_args = sample_batch_fn(objective_args, subkey, p=p, batch_size=batch_size)
               step_objective_args = sample_batch_fn(objective_args, subkey, batch_size=batch_size)
               
          key, subkey = jax.random.split(key)
          gene_subset_covariances = jax.random.choice(subkey, num_genes, (int(num_genes**covariance_var_fraction),), replace=False)

          score, score_grad = expected_score_fn(params, step_objective_args, gene_subset_covariances)
          reg_term, reg_grad = expected_adjacency_regularizer_fn(params, lambd)
          grads = jax.tree_util.tree_map(lambda g1, g2: g1 + g2, score_grad, reg_grad)

          for k, param_grad in grads.items():
               if k not in optimize_params:
                    grads[k] = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x) if x is not None else 0, param_grad)
                    
          # Gradient update
          updates, opt_state = optimizer.update(grads, opt_state, params)
          params = optax.apply_updates(params, updates)
          step_scores[i % metrics_freq] = score + reg_term
          
          # Printing scores
          if (i + 1) % metrics_freq == 0:
               # Train score
               score = step_scores.mean()
               scores['score'].append(score)
               print_str = f'Step {i}: Train {score:.3f}'
               # Printing training metrics
               if dict_metric_fns is not None:
                    metrics_str = ''          
                    for k, v in dict_metric_fns.items():
                         scores[k].append(v(params))
                         metrics_str += f' | {k}: {scores[k][-1]:.3f}'
                    print_str += f'{metrics_str}'
               
               # Validation score
               if objective_args_val is not None:
                    ills_out = calculate_interventional_loglikelihoods_analytic(params,
                                                                       objective_args_val)
                    ills = ills_out['unique_regimes_ills']
                    score_val = jnp.mean(ills)
                    max_score = jnp.max(ills)
                    scores['score_val'].append(score_val)
                    print_str += f' |  Validation {score_val:.3f}   Max: {max_score:.3f}'

                    if np.isnan(best_score_val) or score_val < best_score_val:
                         best_params = params.copy()
                         best_score_val = score_val
               print(print_str)
               
     if best_params is None:
          best_params = params.copy()

     return {
          'params': params,
          'scores': scores,
          'best_params': best_params
     }
     

"""
Fine tune
"""

def fine_tune(key,
               objective_value_and_grad_fn,
               objective_args,
               params,
               objective_args_val=None,
               n_steps=10000,
               learning_rate=0.01,
               batch_size=64,
               sample_batch_fn=None,
               optimize_params=['weight_premask', 'bias_premask', 'layer_weights', 'layer_biases'],
               dict_metric_fns=None,
               metrics_freq=200,
               threshold=0.5,
               density='gaussian',
               ):
     
     optimizer = optax.adam(learning_rate)
     opt_state = optimizer.init(params)
     scores = defaultdict(list)
     step_scores = [] # train scores
     best_score_val = np.nan
     best_params = None
     permutation = jnp.argsort(params['logits'], axis=-1, descending=True)
     for i in range(n_steps):
          # Batch sampling
          if sample_batch_fn is None:
               step_objective_args = objective_args[0], objective_args[1]
          else:
               key, subkey = jax.random.split(key)
               step_objective_args = sample_batch_fn(objective_args, subkey, batch_size=batch_size)

          bernoulli_mask = (jax.nn.sigmoid(params['bernoulli_logits']) >= threshold).astype(jnp.float32)     
          fb, grads = objective_value_and_grad_fn(permutation,
                                                  bernoulli_mask,
                                                  params,
                                                  step_objective_args)
          
          grads = jax.tree.map(lambda x: jnp.mean(x, axis=0) if x is not None else 0, grads)
          
          # Update everything but logits and bernoulli_logits
          for k, param_grad in grads.items():
               if k not in optimize_params:
                    grads[k] = jax.tree.map(lambda x: jnp.zeros_like(x) if x is not None else 0, param_grad)

          # Gradient update
          updates, opt_state = optimizer.update(grads, opt_state, params)
          params = optax.apply_updates(params, updates)
          
          score = np.array(np.mean(fb))
          step_scores.append(score)
          scores['score'].append(score)
          
          # Printing scores
          if (i + 1) % metrics_freq == 0:
               # Train score
               score = np.array(step_scores).mean()
               print_str = f'Step {i}: Train {score:.3f}'
               # Printing training metrics
               if dict_metric_fns is not None:
                    metrics_str = ''          
                    for k, v in dict_metric_fns.items():
                         scores[k].append(v(params))
                         metrics_str += f"{k}: {scores[k][-1]:.3f} "
                    print_str += f' | {metrics_str}'    
               step_scores = []

               # Validation score
               if objective_args_val is not None:
                    ills_out = calculate_interventional_loglikelihoods(params,
                                                                       objective_args_val,
                                                                       density)
                    ills = ills_out['unique_regimes_ills']                    
                    score_val = jnp.mean(ills)
                    max_score = jnp.max(ills)
                    scores['score_val'].append(score_val)
                    print_str += f' |  Validation {score_val:.3f}   Max: {max_score:.3f}'

                    if np.isnan(best_score_val) or score_val < best_score_val:
                         best_params = params.copy()
                         best_score_val = score_val
               print(print_str)
     if best_params is None:
          best_params = params.copy()

     return {
          'params': params,
          'scores': scores,
          'best_params': best_params
     }