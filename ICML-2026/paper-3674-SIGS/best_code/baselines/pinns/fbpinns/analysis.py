"""
Defines helper functions for loading and running saved FBPINN / PINN models
"""

import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from fbpinns.util.logger import logger
from fbpinns.util.other import DictToObj
from fbpinns.trainers import get_inputs, FBPINN_model, PINN_model
from fbpinns import networks


def load_model(run, i=None, rootdir="results/"):
    """load a model, its supplements and constants object from rootdir.
    If i is specified, load the model at that timestep, otherwise
    load the model at the largest timestep"""

    # get model_dir and summary_dir
    model_dir = rootdir+f"models/{run}/"
    summary_dir = rootdir+f"summaries/{run}/"

    # load constants dictionary, convert to object
    with open(summary_dir+f"constants_{run}.pickle", "rb") as f:
        c_dict = pickle.load(f)
    c = DictToObj(**c_dict, copy=True)

    # get last timestep to load
    if i is None:
        last_file = sorted(os.listdir(model_dir))[-1]
        i = int(os.path.splitext(last_file)[0].split("_")[1])

    # load model
    file = model_dir+f"model_{i:08d}.jax"
    logger.info(f"Loading model from:\n{file}")
    with open(file, "rb") as f:
        model = pickle.load(f)

    # convert np arrays to jax
    model = jax.tree_util.tree_map(lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x, model)

    return c, model


def FBPINN_solution(c, all_params, active, x_batch):
    "Runs trained FBPINN on a batch of points"

    problem, decomposition, network = c.problem, c.decomposition, c.network
    model_fns = (decomposition.norm_fn, network.network_fn, decomposition.unnorm_fn, decomposition.window_fn, problem.constraining_fn)
    takes, _, (_, _, _, cut_all, _) = get_inputs(x_batch, active, all_params, decomposition)
    all_params_cut = {"static":cut_all(all_params["static"]),
                      "trainable":cut_all(all_params["trainable"])}
    u, *_ = FBPINN_model(all_params_cut, x_batch, takes, model_fns)
    return u

def PINN_solution(c, all_params, x_batch):
    "Runs trained PINN on a batch of points"

    domain, problem, network = c.domain, c.problem, c.network

    # define unnorm function
    mu_, sd_ = c.decomposition_init_kwargs["unnorm"]
    unnorm_fn = lambda u: networks.unnorm(mu_, sd_, u)
    model_fns = (domain.norm_fn, network.network_fn, unnorm_fn, problem.constraining_fn)

    u, *_ = PINN_model(all_params, x_batch, model_fns)
    return u




if __name__ == "__main__":

    import matplotlib.pyplot as plt


    x_batch = jnp.linspace(0,1,100).reshape((-1,1))

    plt.figure(figsize=(12,4))

    c, model = load_model(run="FBPINN", i=10000, rootdir="../test/results/")
    i, all_params, all_opt_states, active, u_test_losses = model
    u = FBPINN_solution(c, all_params, active, x_batch)

    print(len(model))
    print(c)
    print(i)
    plt.subplot(1,3,1)
    plt.plot(u_test_losses[:,0], u_test_losses[:,-1], label=c.run)# steps
    plt.subplot(1,3,2)
    plt.plot(u_test_losses[:,2], u_test_losses[:,-1], label=c.run)# flops
    plt.subplot(1,3,3)
    plt.plot(x_batch, u)

    c, model = load_model(run="PINN", i=10000, rootdir="../test/results/")
    i, all_params, all_opt_states, u_test_losses = model
    u = PINN_solution(c, all_params, x_batch)

    print(len(model))
    print(c)
    print(i)
    plt.subplot(1,3,1)
    plt.plot(u_test_losses[:,0], u_test_losses[:,-1], label=c.run)# steps
    plt.subplot(1,3,2)
    plt.plot(u_test_losses[:,2], u_test_losses[:,-1], label=c.run)# flops
    plt.subplot(1,3,3)
    plt.plot(x_batch, u)

    plt.subplot(1,3,1)
    plt.yscale("log")
    plt.legend()
    plt.subplot(1,3,1)
    plt.yscale("log")
    plt.legend()
    plt.show()