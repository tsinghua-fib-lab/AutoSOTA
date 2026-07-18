import torch
import numpy as np
import math
from scipy.stats import norm
from pandora_automl.utils import fit_gp_model, normalize_config
from pandora_automl.acquisition.stable_gittins import StableGittinsIndex
from botorch.acquisition import UpperConfidenceBound
from pandora_automl.acquisition.lcb import LowerConfidenceBound
from pandora_automl.acquisition.log_ei_puc import LogExpectedImprovementWithCost
from botorch.sampling.pathwise import draw_matheron_paths
import wandb
import time

# Set default tensor type to float64
torch.set_default_dtype(torch.float64)

import os
import importlib.util

# Go up to the project root (PandoraAutoML)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Path to api.py
api_path = os.path.join(project_root, "LCBench", "api.py")

# Load the api module dynamically
spec = importlib.util.spec_from_file_location("lcbench_api", api_path)
lcbench_api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lcbench_api)

# Use Benchmark from api.py
Benchmark = lcbench_api.Benchmark

# Load the benchmark file
bench_path = os.path.join(project_root, "LCBench", "cached", "six_datasets_lw.json")

bench = Benchmark(bench_path, cache=False)


def run_bayesopt_experiment(bayesopt_config):
    print(bayesopt_config)
    dataset_name = bayesopt_config['dataset_name']
    seed = bayesopt_config['seed']
    output_standardize = bayesopt_config['output_standardize']
    maximize = bayesopt_config['maximize']
    dim = bayesopt_config['dim']
    n_iter = bayesopt_config['num_iteration']
    num_configs = bayesopt_config['num_configs']
    acq = bayesopt_config['acquisition_function']

    # Gather all configurations and their corresponding values.
    all_x = []
    all_y = []
    all_c = []
    estimated_costs = []
    for config_id in bench.data[dataset_name].keys():
        config = bench.query(dataset_name, "config", config_id)
        x = normalize_config(config)
        all_x.append(x)
        val_acc = bench.query(dataset_name, "final_val_accuracy", config_id)
        all_y.append(100-val_acc)
        runtime = bench.query(dataset_name, "time", config_id)[-1]
        all_c.append(runtime)
        model_param = bench.query(dataset_name, "model_parameters", config_id)
        estimated_costs.append(0.001*model_param)

    all_x = torch.stack(all_x)
    all_y = torch.tensor(all_y).unsqueeze(1)
    all_c = torch.tensor(all_c).unsqueeze(1)
    estimated_costs = torch.tensor(estimated_costs).unsqueeze(1)

    # Sample initial configurations
    torch.manual_seed(seed)
    init_config_id = torch.randint(low=0, high=num_configs, size=(2*(dim+1),))
    config_id_history = init_config_id.tolist()
    print(f"  Initial config id: {config_id_history}")
    x = all_x[init_config_id]
    y = all_y[init_config_id]
    c = all_c[init_config_id]
    best_y_history = [y.min().item()]
    best_id_history = [config_id_history[y.argmin().item()]]
    cost_history = [0]
    estimated_cost_history = [0]

    acq_history = {
        'PBGI-D': [np.nan]
    }

    init_lmbda = 200  # lambda_0 = B / (C+U) = 10000 / 50 = 200
    cur_lmbda = init_lmbda

    for i in range(n_iter):
        # 1. Fit a GP model on the current data.
        model = fit_gp_model(X=x, objective_X=y, output_standardize=output_standardize)
        
        # 2. Determine the best observed objective value.
        best_f = y.min()
            
        # 3. Define the acquisition function.
        PBGI_D = StableGittinsIndex(model=model, maximize=maximize, lmbda=cur_lmbda)

        # 4. Evaluate the acquisition function on all candidate x's.
        PBGI_D_acq = PBGI_D.forward(all_x.unsqueeze(1), cost_X = estimated_costs)
        PBGI_D_acq[config_id_history] = y.squeeze(-1)

        # 5. Select the candidate with the optimal acquisition value.
        all_ids = torch.arange(num_configs)
        mask = torch.ones(num_configs, dtype=torch.bool)
        mask[config_id_history] = False
        candidate_ids = all_ids[mask]
        
        if acq == "PBGI-D":
            candidate_acqs = PBGI_D_acq[mask]
            new_config_id = candidate_ids[torch.argmin(candidate_acqs)]
            new_config_acq = torch.min(candidate_acqs)

        new_config_x = all_x[new_config_id]
        
        # 6. Query the objective for the new configuration.
        new_config_y = all_y[new_config_id]
        new_config_c = all_c[new_config_id]
        new_config_estimated_c = estimated_costs[new_config_id]

        # 7. Record the acquisition value.
        cur_acq = torch.min(PBGI_D_acq).item()
        acq_history['PBGI-D'].append(cur_acq)
        
        # 8. Append the new data to our training set.
        x = torch.cat([x, new_config_x.unsqueeze(0)], dim=0)
        y = torch.cat([y, new_config_y.unsqueeze(0)], dim=0)
        c = torch.cat([c, new_config_c.unsqueeze(0)], dim=0)
        config_id_history.append(new_config_id.item())
        best_y_history.append(best_f.item())
        best_id_history.append(config_id_history[y.argmin().item()])
        cost_history.append(new_config_c.item())
        estimated_cost_history.append(new_config_estimated_c.item())

        # 9. Check lmbda decay (stopping) condition.
        if cur_acq >= best_f.item():
            cur_lmbda = cur_lmbda / 2

        print(f"Iteration {i + 1}:")
        print(f"  Selected config_id: {new_config_id}")
        print(f"  Acquisition value: {new_config_acq.item():.4f}")
        print(f"  Objective (validation error): {new_config_y.item():.4f}")
        print(f"  Cost (runtime): {new_config_c.item():.4f}")
        print(f"  Estimated cost (0.001*model_parameters): {new_config_estimated_c.item():.4f}")
        print(f"  Current best observed: {best_f.item():.4f}")
        print()

    best_y_history.append(y.min().item())

    # Return the history including the acq_history dictionary.
    return (cost_history,
            estimated_cost_history,
            [best_id_history[0]] + config_id_history[-n_iter:], 
            best_id_history,
            best_y_history,
            acq_history)


run = wandb.init(reinit=True, sync_tensorboard=False, settings=wandb.Settings(_disable_stats=True))

result = run_bayesopt_experiment(run.config)

(cost_history, estimated_cost_history, config_id_history, best_id_history, best_y_history, acq_history) = result

cumulative_costs = np.cumsum(cost_history)
estimated_cumulative_costs = np.cumsum(estimated_cost_history)

# Log full info
for idx in range(len(cost_history)):
    log_dict = {
        "config id": config_id_history[idx],
        "cumulative cost": cumulative_costs[idx],
        'estimated cumulative cost': estimated_cumulative_costs[idx],
        "current best id": best_id_history[idx],
        "current best observed": best_y_history[idx],
        "PBGI-D acq": acq_history['PBGI-D'][idx]
    }
    run.log(log_dict)
    time.sleep(0.5)  # Delay of 0.5s per entry

run.finish()