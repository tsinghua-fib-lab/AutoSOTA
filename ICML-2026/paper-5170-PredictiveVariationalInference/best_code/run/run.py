import click
import importlib
import pandas as pd
import numpy as np
import seaborn as sns
from jax import random, vmap
from jax.example_libraries import optimizers
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import scipy
import jax.numpy as jnp

@click.command()
@click.option('--model', default='Golf', help='The model name, must match a class name under model/')
@click.option('--objective', default='VIBasic', help='The loss, must match a class name under objective/')
@click.option('--regularizer', default='VIBasic', help='The regularizer, must match a class name under objective/')
@click.option('--posterior', default='Basic', help='The parametric family of posterior, must match a class under posterior/')
@click.option('--seed', default=0, help='The random seed')
@click.option('--lamb', default=0.0, help='Regularization strength')
@click.option('--alpha', default=0.0, help='Interval score parameter')
@click.option('--n', default=100,  help='Number of observations')
@click.option('--s', default=1, help='Number of draws for \\theta')
@click.option('--iterations', default=1000, help='Number of training iterations')
@click.option('--prediction_sample', default=10000, help='Number of samples for evaluation')
@click.option('--learning_rate', default=0.001, help='Learning rate')
@click.option('--optimizer', default = 'sgd', help='Optimizer, choice from [sgd, rmsprop, nesterov]')
def main(model, objective, regularizer, posterior, seed, lamb, alpha, n, s, iterations, prediction_sample,
         learning_rate, optimizer):
    module1 = importlib.import_module('model')
    m = getattr(module1, model)()
    rng_key = random.PRNGKey(seed)
    y = m.data(random.PRNGKey(1))
    test_y = None
    if type(y) == tuple:
        y, test_y = y

    module2 = importlib.import_module('posterior')
    p = getattr(module2, posterior)(m.d)
    params = p.gen_params()

    module3 = importlib.import_module('objective')
    o = getattr(module3, objective)(m, p, s=s, alpha=alpha)

    r = getattr(module3, regularizer)(m, p, s=s,)

    def scheduler(step):
        if step < iterations // 2:
            return learning_rate
        return learning_rate / 10


    if optimizer == 'sgd':
        opt_init, opt_update, get_params = optimizers.sgd(scheduler,)
    elif optimizer == 'nesterov':
        opt_init, opt_update, get_params = optimizers.nesterov(scheduler, 0.9)
    elif optimizer == 'rmsprop':
        opt_init, opt_update, get_params = optimizers.rmsprop_momentum(scheduler, )
    else:
        raise ValueError(optimizer)
    opt_state = opt_init(params), rng_key

    if lamb != 0:
        def regularized_objective(key, params):
            key1, key2 = random.split(key)
            v1, g1 = o.value_and_grad(key1, params)
            v2, g2 = r.value_and_grad(key2, params)
            return v1 + lamb * v2, g1 + lamb * g2
    else:
        def regularized_objective(key, params):
            key1, key2 = random.split(key)
            v1, g1 = o.value_and_grad(key1, params)
            return v1 , g1
    def step(step, opt_state):
        param, rng_key = opt_state
        data_key, rng_key = random.split(rng_key)
        value, grads = regularized_objective(data_key, get_params(param))
        norm = jnp.linalg.norm(grads)
        grads = grads / n
        if norm > 100:
            grads = grads / norm * 100
        if jnp.sum(jnp.isnan(grads)):
            updated_state = param
        else:
            updated_state = opt_update(step, -grads, param)
        return value, (updated_state, rng_key)


    data = []
    best_pred = -np.inf
    best_param = params
    for i in tqdm(range(iterations)):
        value, opt_state = step(i, opt_state)
        param, rng_key = opt_state
        if i % 1000 == 0:
            print(i, value, get_params(param))

            theta_sample = p.sample(rng_key, get_params(param), prediction_sample)
            log_likelihoods = vmap(m.valid_log_likelihoods)(theta_sample)
            predictive_ll = np.sum(
                np.array(scipy.special.logsumexp(log_likelihoods, axis=0)) - np.log(prediction_sample))
            if predictive_ll >= best_pred:
                best_pred = predictive_ll
                best_param = get_params(param)
            print(predictive_ll)

        data.append({'step':i, 'loss': float(value)})

    param, rng_key = opt_state
    param_list = get_params(param)
    theta_sample = p.sample(rng_key, param_list, prediction_sample)
    log_likelihoods = vmap(m.valid_log_likelihoods)(theta_sample,)
    predictive_ll = np.sum(np.array(scipy.special.logsumexp(log_likelihoods, axis=0)) - np.log(prediction_sample))

    print(best_param, param_list, predictive_ll)

    if test_y is not None:
        log_likelihoods = vmap(m.test_log_likelihoods, in_axes=(0, None))(theta_sample, test_y)
        test_ll = np.sum(np.array(scipy.special.logsumexp(log_likelihoods, axis=0)) - np.log(prediction_sample))
        predictive_ll = test_ll
        print('test set :', test_ll)

    os.makedirs(f"result/{model}_{alpha}_{n}_{posterior}_{objective}/", exist_ok=True)
    result_file = f'result/{model}_{alpha}_{n}_{posterior}_{objective}/{regularizer}_{lamb}_{seed}_{s}'
    print('writing to',result_file)

    with open(result_file, 'w') as f:
        params = best_param
        print(' '.join(map(str, params)), predictive_ll, file=f)

        if hasattr(m, 'valid_y'):
            log_likelihoods = vmap(m.valid_log_likelihoods,)(theta_sample,)
            valid_ll = np.sum(np.array(scipy.special.logsumexp(log_likelihoods, axis=0)) - np.log(prediction_sample))
            print('valid set :', valid_ll, )
            print('valid set :', valid_ll, file=f)

    data = pd.DataFrame(data)
    sns.lineplot(data = data, x='step', y = 'loss')
    plt.savefig(result_file+'.pdf')


if __name__ == '__main__':
    main()