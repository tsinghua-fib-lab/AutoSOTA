import click
import importlib
import pandas as pd
import numpy as np
import seaborn as sns
from jax import random, vmap, jit
from jax.example_libraries import optimizers
import jax
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import jax.numpy as jnp
import jaxopt
import optax
import equinox as eqx

@click.command()
@click.option('--model', default='Golf', help='The model name, must match a class name under model/')
@click.option('--objective', default='VIBasic', help='The loss, must match a class name under objective/')
@click.option('--regularizer', default='VIBasic', help='The regularizer, must match a class name under objective/')
@click.option('--posterior', default='NormalizingFlow', help='The parametric family of posterior, must match a class under posterior/')
@click.option('--seed', default=0, help='The random seed')
@click.option('--lamb', default=0.0, help='Regularization strength')
@click.option('--alpha', default=0.0, help='Interval score parameter')
@click.option('--n', default=100,  help='Number of observations')
@click.option('--s', default=1, help='Number of draws for \\theta')
@click.option('--iterations', default=10000, help='Number of training iterations')
@click.option('--prediction_sample', default=10000, help='Number of samples for evaluation')
@click.option('--learning_rate', default=0.001, help='Learning rate')
def main(model, objective, regularizer, posterior, seed, lamb, alpha, n,  s, iterations,
         prediction_sample, learning_rate):
    module1 = importlib.import_module('model')
    m = getattr(module1, model)()
    rng_key = random.PRNGKey(seed)
    #data_key, rng_key = random.split(rng_key)
    y = m.data(random.PRNGKey(1))
    test_y = None
    if type(y) == tuple:
        y, test_y = y

    init_key, rng_key = random.split(rng_key)
    module2 = importlib.import_module('posterior')
    p = getattr(module2, posterior)(m.d, init_key)
    params = p.gen_params()

    module3 = importlib.import_module('objective')
    o = getattr(module3, objective)(m, p, y, s=s, alpha=alpha)
    r = getattr(module3, regularizer)(m, p, y, s=s)

    chain = []
    schedule = optax.warmup_cosine_decay_schedule(init_value=0.0,
                                                  peak_value=1.0,
                                                  warmup_steps=1000,
                                                  decay_steps=iterations,
                                                  end_value=0.1,
                                                  exponent=1.0)
    chain.append(optax.clip_by_global_norm(10.0))
    chain.append(optax.adam(learning_rate, ))
    chain.append(optax.scale_by_schedule(schedule))
    optimizer = optax.chain(*chain)
    opt_state = optimizer.init(eqx.filter(params, eqx.is_inexact_array)), params, rng_key


    if lamb != 0:
        def objective(flow, key):
            key1, key2 = random.split(key)
            v1 = -o.objective(key, flow) / n
            v2 = -r.objective(key2, flow) / n
            return v1 + lamb * v2
    else:
        def objective(flow, key):
            return -o.objective(key, flow) / n


    def step(step, opt_state):
        param, flow, rng_key = opt_state
        data_key, rng_key = random.split(rng_key)
        value, grads = eqx.filter_jit(eqx.filter_value_and_grad(objective))(flow, data_key)
        skip, dict = optax.skip_not_finite(grads, param, flow)
        #skip = False
        if not skip:
            updates, new_opt_state = optimizer.update(grads, param, flow)
            new_flow = eqx.apply_updates(flow, updates)
        else:
            print(step,'skipped')
            new_opt_state = param
            new_flow = flow
        return value, (new_opt_state, new_flow, rng_key)


    valid_key, rng_key = random.split(rng_key)

    data = []
    for i in tqdm(range(iterations)):
        value, opt_state = step(i, opt_state)
        _, flow, rng_key = opt_state
        if i % 1000 == 0:
            print(i, value,)

        data.append({'step':i, 'loss': float(value)})

    _, flow, rng_key = opt_state

    theta_sample, _ = eqx.filter_vmap(flow.sample_and_log_prob)(random.split(rng_key, prediction_sample), )

    result_dir = f'result/{model}/PVI_{seed}_{alpha}_{s}_{lamb}_{regularizer}/'
    os.makedirs(result_dir,exist_ok=True)
    np.savez_compressed(f'result/{model}/PVI_{seed}_{alpha}_{s}_{lamb}_{regularizer}/sample.npz', samples = np.array(
        theta_sample))
    print(f'Saved in result/{model}/PVI_{seed}_{alpha}_{s}_{lamb}_{regularizer}/sample.npz')



if __name__ == '__main__':
    main()