#!/usr/bin/env python3

import torch
from botorch.models import SingleTaskGP
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from botorch.utils.sampling import draw_sobol_samples
from botorch.sampling.pathwise import draw_kernel_feature_paths
from botorch.utils.sampling import optimize_posterior_samples
from botorch.acquisition import LogExpectedImprovement, UpperConfidenceBound
from pandora_automl.acquisition.gittins import GittinsIndex
from pandora_automl.acquisition.stable_gittins import StableGittinsIndex
from pandora_automl.acquisition.log_ei import LogExpectedImprovement
from pandora_automl.acquisition.log_ei_puc import LogExpectedImprovementWithCost
from scripts.bayesianoptimizer import BayesianOptimizer


import numpy as np
import matplotlib.pyplot as plt
import wandb
from wandb import Settings, init
import os
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# use a GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set default tensor type to float64
torch.set_default_dtype(torch.float64)

class Cost:
    def __init__(self, dim, cost_type='uniform', global_opt=None, alpha=2, beta=2, gamma=0):
        self.dim = dim
        self.cost_type = cost_type
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.global_optimum_point = global_opt 
        print("Using cost function:", cost_type)

    def __call__(self, x):
        if self.cost_type == 'uniform':
            # print("uniform cost")
            #print("x:", x)
            #print(torch.ones(x.shape[:-1]).squeeze(-1))
            return torch.ones(x.shape[:-1])
        elif self.cost_type == 'linear':
            linear_cost = (1.0 + 20*x.mean(dim=-1, keepdim=True))/ (1.0 + 20/2.0)
            # print("linear cost:", linear_cost)
            return linear_cost
        elif self.cost_type == 'periodic':
            # print("difference until optimum", x-self.global_optimum_point)
            normalizer = torch.i0(torch.tensor(self.alpha/self.dim))**self.dim
            periodic_cost = torch.exp(
                self.alpha*torch.cos(2 * np.pi * self.beta * (x-self.global_optimum_point + self.gamma)).sum(dim=-1, keepdim=True)/self.dim)/normalizer
            # print("periodic cost:", periodic_cost)
            return periodic_cost
        else:
            raise ValueError(f"Unknown cost type: {self.cost_type}")


def run_bayesopt_experiment(config):
    print(config)

    dim = config['dim']
    num_iterations = config['num_iterations']
    lengthscale = config['lengthscale']
    outputscale = config['amplitude']
    policy = config['policy']
    print("policy:", policy)
    maximize = True
    kernel = config['kernel']
    if kernel == 'Matern12':
        nu = 0.5
    elif kernel == 'Matern32':
        nu = 1.5 
    elif kernel == 'Matern52':
        nu = 2.5

    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed) 
    # Create the objective function
    if kernel == 'RBF':
        base_kernel = RBFKernel().double()
    else:
        base_kernel = MaternKernel(nu=nu).double()
    base_kernel.lengthscale = torch.tensor([[lengthscale]])
    base_kernel.raw_lengthscale.requires_grad = False
    scale_kernel = ScaleKernel(base_kernel).double()
    scale_kernel.outputscale = torch.tensor([[outputscale]])
    scale_kernel.raw_outputscale.requires_grad = False

    # Define Noise Level
    noise_level = 1e-6

    # Initialize Placeholder Data with Correct Dimensions
    num_samples = 1  # Replace with actual number of samples
    num_features = dim  # Replace with actual number of features
    train_X = torch.zeros(num_samples, num_features)  # Placeholder data
    train_Y = torch.zeros(num_samples, 1)             # Placeholder data
    Yvar = torch.ones(num_samples) * noise_level

    # Initialize Model
    model = SingleTaskGP(train_X, train_Y, likelihood = FixedNoiseGaussianLikelihood(noise=Yvar), covar_module=scale_kernel)

    # Draw a sample path
    sample_path = draw_kernel_feature_paths(model, sample_shape=torch.Size([1]))
    def objective_function(x):
        return sample_path(x).squeeze(0).detach()

    # Find the global optimum
    bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])
    if (dim == 1) or (('optimize_method' in config) and (config['optimize_method'] == 'grid')):
        granularity = 10001
        test_grids = [torch.linspace(0, 1, granularity) for _ in range(dim)]
        test_mesh = torch.meshgrid(*test_grids, indexing="ij")  # consistent indexing
        test_x = torch.stack([g.flatten() for g in test_mesh], dim=-1).unsqueeze(-2)  # shape (granularity^dim, dim)
    
        objective_vals = objective_function(test_x)
        global_optimum_value = torch.max(objective_vals)
        global_optimum_idx = torch.argmax(objective_vals)
        global_optimum_point = test_x[global_optimum_idx].unsqueeze(0).unsqueeze(1)   
    else:
        global_optimum_point, global_optimum_value = optimize_posterior_samples(paths=sample_path, bounds=bounds, raw_samples=1024*dim, num_restarts=20*dim, maximize=maximize)
    
    print("global optimum point:", global_optimum_point.detach().numpy())
    print("global optimum value:", global_optimum_value.item())

    # create the cost function, uniform as default
    if 'cost' not in config:
        cost_fn = Cost(dim, cost_type='uniform')
    else:
        alpha = config.get('alpha', 2)
        beta = config.get('beta', 2)
        gamma = config.get('gamma', 0)
        cost_fn = Cost(dim, cost_type=config['cost'],
                    global_opt=global_optimum_point,
                    alpha=alpha, beta=beta, gamma=gamma)

    # Test performance of different policies
    draw_initial_method = config['draw_initial_method']
    if draw_initial_method == 'sobol':
        bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])
        init_x = draw_sobol_samples(bounds=bounds, n=1, q=2*(dim+1)).squeeze(0)
        # print("initial points:", init_x)
    output_standardize = config['output_standardize']

    # print("cost of initial points: ", cost_fn(torch.linspace(0, 1, 50).unsqueeze(1)))
    Optimizer = BayesianOptimizer(
        dim=dim, 
        maximize=maximize, 
        initial_points=init_x,
        objective=objective_function, 
        kernel=scale_kernel,
        output_standardize=output_standardize,
        cost=cost_fn
    )

    if policy == 'Stable_Gittins_Lambda_1':
        Optimizer.run(
            num_iterations=num_iterations, 
            acquisition_function_class = StableGittinsIndex,
            lmbda = 0.1
        )
    elif policy == 'Stable_Gittins_Lambda_01':
        Optimizer.run(
            num_iterations=num_iterations, 
            acquisition_function_class = StableGittinsIndex,
            lmbda = 0.01,
        )
    elif policy == 'Stable_Gittins_Lambda_001':
        Optimizer.run(
            num_iterations=num_iterations, 
            acquisition_function_class = StableGittinsIndex,
            lmbda = 0.001
        )
    elif policy == 'LogEI':
        Optimizer.run(
            num_iterations=num_iterations, 
            acquisition_function_class = LogExpectedImprovement
        )
    elif policy == 'LogEIWithCost':
        Optimizer.run(
            num_iterations=num_iterations, 
            acquisition_function_class = LogExpectedImprovementWithCost
        )
    elif policy == 'ThompsonSampling':
        Optimizer.run(
            num_iterations=num_iterations, 
            acquisition_function_class = "ThompsonSampling"
        )
    elif policy == "UpperConfidenceBound":
        Optimizer.run(
            num_iterations=num_iterations, 
            acquisition_function_class = UpperConfidenceBound,
            heuristic = True
        )

    cost_history = Optimizer.get_cost_history()
    best_history = Optimizer.get_best_history()
    regret_history = Optimizer.get_regret_history(global_optimum_value.item())
    acq_history = Optimizer.get_acq_history()
    stopping_history = Optimizer.get_stopping_history()
    lmbda_history = Optimizer.get_lmbda_history()

    print("Cost history:", cost_history)
    print("Best history:", best_history)
    print("Regret history:", regret_history)
    print("Acquisition history:", acq_history)
    print("Stopping history:", stopping_history)
    print("Lambda history:", lmbda_history)

    print()
    
    return (global_optimum_value.item(), cost_history, best_history, regret_history, acq_history, stopping_history, lmbda_history)

# Initialize wandb

try:
    os.environ["WANDB_MODE"] = "online"
    run = wandb.init(sync_tensorboard=False, settings=wandb.Settings(_disable_stats=True))
except wandb.errors.UsageError:
    print("Falling back to offline mode due to WANDB error")
    os.environ["WANDB_MODE"] = "offline"
    run = wandb.init(sync_tensorboard=False, settings=wandb.Settings(_disable_stats=True))

config = run.config
print(config)

(global_optimum_value, cost_history, best_history, regret_history, acq_history, stopping_history, lmbda_history) = run_bayesopt_experiment(run.config)


run.log({"global optimum value": global_optimum_value})

lmbdas = [0.1, 0.01, 0.001]
if 'include_prb' not in config:
    include_prb = (config['dim'] == 1)
else:
    include_prb = config['include_prb']

if 'include_time_log' not in config:
    include_time_log = False
else:
    include_time_log = config['include_time_log']

# Log full info
for idx in range(len(cost_history)):
    log_dict = {
        "cumulative cost": cost_history[idx],
        "best observed": best_history[idx],
        "regret": regret_history[idx],
        "lg(regret)": np.log10(regret_history[idx]),
        "StablePBGI(0.1)": stopping_history['StablePBGI(0.1)'][idx],
        "StablePBGI(0.01)": stopping_history['StablePBGI(0.01)'][idx],
        "StablePBGI(0.001)": stopping_history['StablePBGI(0.001)'][idx], 
        "LogEIC acq": stopping_history['LogEIC'][idx],
        "UCB-LCB acq": stopping_history['UCB-LCB'][idx],
        "Regret-Gap acq": stopping_history['Expected-Min-Regret-Gap'][idx]
    }
    if (include_prb):
        log_dict["PRB_0.1"] = stopping_history['PRB_0.1'][idx] 
    if (include_time_log):
        log_dict["StablePBGI(0.1)_time"] = stopping_history['StablePBGI(0.1)_time'][idx]
        log_dict["StablePBGI(0.01)_time"] = stopping_history['StablePBGI(0.01)_time'][idx] 
        log_dict["StablePBGI(0.001)_time"] = stopping_history['StablePBGI(0.001)_time'][idx]
        log_dict["LogEIC_time"] = stopping_history['LogEIC_time'][idx]
        log_dict["UCB-LCB_time"] = stopping_history['UCB-LCB_time'][idx]
        log_dict["Regret-Gap_time"] = stopping_history['Expected_Min_Regret_Gap_time'][idx] 
        if (include_prb): 
            log_dict["PRB_0.1_time"] = stopping_history['PRB_0.1_time'][idx]  
    run.log(log_dict)
    time.sleep(0.1)  
run.finish()
