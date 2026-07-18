#!/usr/bin/env python3

from pandora_automl.test_functions.lunar_lander import LunarLanderProblem
from pandora_automl.test_functions.pest_control import PestControl, pest_control_price
from pandora_automl.test_functions.robot_pushing.robot_pushing import robot_pushing_4d, robot_pushing_14d

import torch
from botorch.utils.sampling import draw_sobol_samples
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
from pandora_automl.acquisition.log_ei_puc import LogExpectedImprovementWithCost
from pandora_automl.acquisition.stable_gittins import StableGittinsIndex
from bayesianoptimizer import BayesianOptimizer
import numpy as np
import wandb
import time
from scipy.interpolate import interp1d


# # use a GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set default tensor type to float64
torch.set_default_dtype(torch.float64)

def run_bayesopt_experiment(config):
    print(config)

    problem = config['problem']
    num_iterations = config['num_iteration']

    if problem == "LunarLander":
        dim = 12
        def objective_cost_function(X):
            return LunarLanderProblem()(X)

    if problem == "PestControl":
        dim = 25
        def objective_function(X):
            choice_X = torch.floor(5*X)
            choice_X[choice_X == 5] = 4
            return PestControl(negate=True)(choice_X)
        def cost_function(X):
            choice_X = torch.floor(5*X)
            choice_X[choice_X == 5] = 4
            res = torch.stack([pest_control_price(x) for x in choice_X]).to(choice_X).unsqueeze(0)
            # Add a small amount of noise to prevent training instabilities
            res += 1e-6 * torch.randn_like(res)
            return res
        
    if problem == "RobotPushing14D":
        cost_function_type = config["cost_function_type"]
        dim = 14
        target_location = torch.tensor([-4.4185, -4.3709])
        target_location2 = torch.tensor([-3.7641, -4.4742])
        def unnorm_X(X: torch.Tensor) -> torch.Tensor:
            X_unnorm = X.clone()
            # Check if the tensor is higher than 3-dimensional
            if X.dim() > 3:
                # Assuming the extra unwanted dimension is at position 1 (the second position)
                X_unnorm = X_unnorm.view(-1, X.size(-1))  # Remove the singleton dimension
            # Check the dimensionality of X and adjust accordingly
            if X.dim() == 3:
                # Remove the singleton dimension assuming it's the second one
                X_unnorm = X_unnorm.squeeze(1)
            elif X.dim() == 1:
                # If 1-dimensional, add a dimension to make it 2D (e.g., for batch size of 1)
                X_unnorm = X_unnorm.unsqueeze(0)
            X_unnorm[:, :2] = 10.0 * X_unnorm[:, :2] - 5.0
            X_unnorm[:, 2:4] = 5 * X_unnorm[:, 2:4]
            X_unnorm[:, 4] = 29.0 * X_unnorm[:, 4] + 1.0
            X_unnorm[:, 5] = 2 * np.pi * X_unnorm[:, 5]
            X_unnorm[:, 6:8] = 10.0 * X_unnorm[:, 6:8] - 5.0
            X_unnorm[:, 8:10] = 5 * X_unnorm[:, 8:10]
            X_unnorm[:, 10] = 29 * X_unnorm[:, 10] + 1.0
            X_unnorm[:, 11] = 2 * np.pi * X_unnorm[:, 11]
            X_unnorm[:, 12:] = 2 * np.pi * X_unnorm[:, 12:]

            return X_unnorm            

        if cost_function_type == "max":
            def objective_function(X: torch.Tensor) -> torch.Tensor:
                X_unnorm = unnorm_X(X)
                objective_X = []
                for x in X_unnorm:
                    # Set the seed based on X to ensure consistent randomness
                    np.random.seed(0)
                    object_location, object_location2, robot_location, robot_location2 = torch.tensor(robot_pushing_14d(x[0].item(), x[1].item(), x[2].item(), x[3].item(), x[4].item(), x[5].item(), x[6].item(), x[7].item(), x[8].item(), x[9].item(), x[10].item(), x[11].item(), x[12].item(), x[13].item()))
                    objective_X.append(-torch.dist(target_location, object_location)-torch.dist(target_location2, object_location2))
                np.random.seed()  # Reset the seed
                return torch.tensor(objective_X)
            def cost_function(X: torch.Tensor) -> torch.Tensor:
                X_unnorm = unnorm_X(X)
                return torch.max(X_unnorm[:, 4], X_unnorm[:, 10]).unsqueeze(0)
            
        if cost_function_type == 'unknown':
            def objective_cost_function(X: torch.Tensor) -> torch.Tensor:
                X_unnorm = unnorm_X(X)
                objective_X = []
                cost_X = []
                
                for x in X_unnorm:
                    np.random.seed(0)
                    object_location, object_location2, robot_location, robot_location2 = torch.tensor(robot_pushing_14d(x[0].item(), x[1].item(), x[2].item(), x[3].item(), x[4].item(), x[5].item(), x[6].item(), x[7].item(), x[8].item(), x[9].item(), x[10].item(), x[11].item(), x[12].item(), x[13].item()))
                    objective_X.append(-torch.dist(target_location, object_location)-torch.dist(target_location2, object_location2))
                    moving_distance = torch.dist(x[:2], robot_location)+torch.dist(x[6:8], robot_location2)+0.1
                    
                    cost_X.append(moving_distance)

                np.random.seed()  # Reset the seed
                
                objective_X = torch.tensor(objective_X)
                cost_X = torch.tensor(cost_X)
                return objective_X, cost_X

    seed = config['seed']
    torch.manual_seed(seed)
    draw_initial_method = config['draw_initial_method']
    if draw_initial_method == 'sobol':
        bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])
        init_x = draw_sobol_samples(bounds=bounds, n=1, q=2*(dim+1)).squeeze(0)
    output_standardize = config['output_standardize']
    maximize = True

    # Test performance of different policies
    policy = config['policy']
    include_prb = config.get('include_prb', False)  # Default to False since PRB is slow
    print("policy:", policy)
    print("include_prb:", include_prb)
    
    if problem == "LunarLander" or (problem == "RobotPushing14D" and cost_function_type == "unknown"):
        Optimizer = BayesianOptimizer(
            dim=dim, 
            maximize=maximize, 
            initial_points=init_x,
            objective_cost=objective_cost_function, 
            output_standardize=output_standardize
        )
    else:
        Optimizer = BayesianOptimizer(
            dim=dim, 
            maximize=maximize, 
            initial_points=init_x,
            objective=objective_function, 
            cost=cost_function,
            output_standardize=output_standardize
        )
    # Store include_prb in Optimizer
    Optimizer.include_prb = include_prb
    if policy == 'PBGI(1e-3)':
        Optimizer.run(
            num_iterations=num_iterations, 
            acquisition_function_class = StableGittinsIndex,
            lmbda = 0.001
        )
    elif policy == 'PBGI(1e-4)':
        Optimizer.run(
            num_iterations=num_iterations, 
            acquisition_function_class = StableGittinsIndex,
            lmbda = 0.0001,
        )
    elif policy == 'PBGI(1e-5)':
        Optimizer.run(
            num_iterations=num_iterations, 
            acquisition_function_class = StableGittinsIndex,
            lmbda = 0.00001
        )
    elif policy == 'LogEIPC':
        Optimizer.run(
            num_iterations=num_iterations, 
            acquisition_function_class = LogExpectedImprovementWithCost
        )
    elif policy == "UCB":
        Optimizer.run(
            num_iterations=num_iterations, 
            acquisition_function_class = UpperConfidenceBound,
            heuristic = True
        )
    elif policy == 'TS':
        Optimizer.run(
            num_iterations=num_iterations, 
            acquisition_function_class = "ThompsonSampling"
        )
    cost_history = Optimizer.get_cost_history()
    best_history = Optimizer.get_best_history()
    acq_history = Optimizer.get_acq_history()
    stopping_history = Optimizer.get_stopping_history()

    print("Cost history:", cost_history)
    print("Best history:", best_history)
    print("Acquisition history:", acq_history)
    print("Stopping history:", stopping_history)
    print()

    return cost_history, best_history, acq_history, stopping_history


if __name__ == "__main__":
    run = wandb.init()
    config = run.config
    print(config)

    (cost_history, best_history, acq_history, stopping_history) = run_bayesopt_experiment(wandb.config)

    # Log full info
    include_prb = config.get('include_prb', False)
    for idx in range(len(cost_history)):
        log_dict = {
            "cumulative cost": cost_history[idx],
            "current best observed": best_history[idx],
            "PBGI(1e-3) acq": stopping_history.get('StablePBGI(0.001)', [np.nan] * len(cost_history))[idx],
            "PBGI(1e-4) acq": stopping_history.get('StablePBGI(0.0001)', [np.nan] * len(cost_history))[idx],
            "PBGI(1e-5) acq": stopping_history.get('StablePBGI(0.00001)', [np.nan] * len(cost_history))[idx],
            "LogEIPC acq": stopping_history.get('LogEIPC', [np.nan] * len(cost_history))[idx],
            "exp min regret gap": stopping_history.get('Expected-Min-Regret-Gap', stopping_history.get('exp min regret gap', [np.nan] * len(cost_history)))[idx],
            "regret upper bound": stopping_history.get('UCB-LCB', stopping_history.get('regret upper bound', [np.nan] * len(cost_history)))[idx]
        }
        if include_prb and 'PRB_0.1' in stopping_history:
            log_dict["PRB"] = stopping_history['PRB_0.1'][idx]
        run.log(log_dict)
        time.sleep(0.5)  # Delay of 0.5s per entry

    run.finish()
