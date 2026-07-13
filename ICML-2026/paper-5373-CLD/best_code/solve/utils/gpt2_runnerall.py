import jax
import numpy as np
import jax.numpy as jnp
from models.cvx_relu_mlp import CVX_ReLU_MLP
from models.two_layer_mlp import Two_Layer_ReLU_MLP
from optimizers.cronos import admm
from experiments.lr_exp_nojit import lr_random_search
import os
from os.path import dirname, join, abspath
import pickle
from utils.gpt2_runner import run
import random
import wandb


wandb.init(
    project="my-project",

)

# List of model names
model_names = [
    'gpt2_no_train',
    'gpt2_notrain_commune',
    'gpt2_notrain_meanpooled',
    'gpt2_notrain_medium',
    'gpt2_notrain_medium_meanpooled',
    'gpt2_lmhead_commune',
    'gpt2_notrain_commune_meanpooled',
    'gpt2lmhead',
    'gpt2lmhead_med',
    'gpt2seqhead',
    'gpt2seqhead_commu',
    'gpt2seqhead_med'
]

OUTPUT_DIR = '/home/--/--/RESULTS/search'

# adamW params
adamW_params = dict(optimizer='AdamW_nojit', gamma=10**-4, n_epoch=30, batch_size=1024)

# Seed for optimizers (change for each dataset)
opt_seed = 2024

# Seed for test train split (change for each dataset)
data_seed = random.randint(1, 10)

# Number of data batches to use
for MODEL in model_names:
    if MODEL in [
        'gpt2_notrain_medium',
        'gpt2_notrain_medium_meanpooled',
        'gpt2lmhead_med',
        'gpt2seqhead_med']:
        num_batches = 84
    else:
        num_batches = 9

    # Loop through combinations of parameters
    for rho in [10**(-x) for x in range(1, 7)]:  # rho from 0.1 to 0.000001
        for admm_iters in range(3, 21):  # admm_iters between 3 and 20
            for pcg_iters in range(10, 51):  # pcg_iters between 10 and 50
                wandb.config.update({
                    "rho": rho,
                    "admm_iters": admm_iters,
                    "pcg_iters": pcg_iters,
                    "model": MODEL
                })
                cronos_params = dict(rank=10, beta=0.001, rho=rho,
                                     gamma_ratio=1, admm_iters=admm_iters, pcg_iters=pcg_iters, check_opt=False)
                
                print(f"Running model {MODEL} with rho={rho}, admm_iters={admm_iters}, pcg_iters={pcg_iters}")
                run(MODEL, num_batches, cronos_params, adamW_params, opt_seed, data_seed, OUTPUT_DIR)
                wandb.log({"test cronos accu": test_peak, "train cronos accu": train_peak})


wandb.finish()