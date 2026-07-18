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
import gc
import wandb

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
    for config_id in bench.data[dataset_name].keys():
        config = bench.query(dataset_name, "config", config_id)
        all_x.append(normalize_config(config))
        val_acc = bench.query(dataset_name, "final_val_accuracy", config_id)
        all_y.append(100-val_acc)
        runtime = bench.query(dataset_name, "time", config_id)[-1]
        all_c.append(runtime)

    all_x = torch.stack(all_x)
    all_y = torch.tensor(all_y).unsqueeze(1)
    all_c = torch.tensor(all_c).unsqueeze(1)

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

    # Initialization for expected minimum simple regret gap stopping rule
    old_model = fit_gp_model(X=x[:-1], objective_X=y[:-1], output_standardize=output_standardize)
    old_config_x = x[-1]

    # Initialization for probabilistic regret bound (PRB) stopping rule
    if dataset_name == "Fashion-MNIST":
        epsilon = 0.05
    if dataset_name == "higgs":
        epsilon = 0.14
    if dataset_name == "adult":
        epsilon = 0.08
    if dataset_name == "volkert":
        epsilon = 0.19
    num_samples = 64

    # Independent seed for Thompson sampling
    ts_seed  = seed + 1

    acq_history = {
        'StablePBGI(1e-3)': [np.nan],
        'StablePBGI(1e-4)': [np.nan],
        'StablePBGI(1e-5)': [np.nan],
        'LogEIC-inv': [np.nan],
        'LogEIC-exp': [np.nan],
        'regret upper bound': [np.nan],
        'exp min regret gap': [np.nan],
        'PRB': [np.nan]
    }

    for i in range(n_iter):
        # 1. Fit a GP model on the current data.
        model = fit_gp_model(X=x, objective_X=y, cost_X=c, unknown_cost=True, output_standardize=output_standardize)
        
        # 2. Determine the best observed objective value.
        best_f = y.min()
            
        # 3. Define the acquisition function.
        StablePBGI_1e_3 = StableGittinsIndex(model=model, maximize=maximize, lmbda=1e-3, unknown_cost=True)
        StablePBGI_1e_4 = StableGittinsIndex(model=model, maximize=maximize, lmbda=1e-4, unknown_cost=True)
        StablePBGI_1e_5 = StableGittinsIndex(model=model, maximize=maximize, lmbda=1e-5, unknown_cost=True)
        LogEIC_inv = LogExpectedImprovementWithCost(model=model, best_f=best_f, maximize=maximize, unknown_cost=True, inverse_cost=True)
        LogEIC_exp = LogExpectedImprovementWithCost(model=model, best_f=best_f, maximize=maximize, unknown_cost=True, inverse_cost=False)
        single_outcome_model = fit_gp_model(X=x, objective_X=y, output_standardize=output_standardize)
        beta = 2 * np.log(dim * ((i + 1) ** 2) * (math.pi ** 2) / (6 * 0.1)) / 5
        UCB = UpperConfidenceBound(model=single_outcome_model, maximize=maximize, beta=beta)
        LCB = LowerConfidenceBound(model=single_outcome_model, maximize=maximize, beta=beta)

        # 4. Evaluate the acquisition function on all candidate x's.
        StablePBGI_1e_3_acq = StablePBGI_1e_3.forward(all_x.unsqueeze(1))
        StablePBGI_1e_3_acq[config_id_history] = y.squeeze(-1)
        StablePBGI_1e_4_acq = StablePBGI_1e_4.forward(all_x.unsqueeze(1))
        StablePBGI_1e_4_acq[config_id_history] = y.squeeze(-1)
        StablePBGI_1e_5_acq = StablePBGI_1e_5.forward(all_x.unsqueeze(1))
        StablePBGI_1e_5_acq[config_id_history] = y.squeeze(-1)
        LogEIC_inv_acq = LogEIC_inv.forward(all_x.unsqueeze(1))
        LogEIC_exp_acq = LogEIC_exp.forward(all_x.unsqueeze(1))
        UCB_acq = UCB.forward(all_x.unsqueeze(1))
        LCB_acq = LCB.forward(all_x.unsqueeze(1))

        # 5. Record information for stopping.
        all_ids = torch.arange(num_configs)
        mask = torch.ones(num_configs, dtype=torch.bool)
        mask[config_id_history] = False
        candidate_ids = all_ids[mask]
        
        if acq == "StablePBGI(1e-3)":
            candidate_acqs = StablePBGI_1e_3_acq[mask]
            new_config_id = candidate_ids[torch.argmin(candidate_acqs)]
            new_config_acq = torch.min(candidate_acqs)
        if acq == "StablePBGI(1e-4)":
            candidate_acqs = StablePBGI_1e_4_acq[mask]
            new_config_id = candidate_ids[torch.argmin(candidate_acqs)]
            new_config_acq = torch.min(candidate_acqs)
        if acq == "StablePBGI(1e-5)":
            candidate_acqs = StablePBGI_1e_5_acq[mask]
            new_config_id = candidate_ids[torch.argmin(candidate_acqs)]
            new_config_acq = torch.min(candidate_acqs)
        if acq == "LogEIC-inv":
            candidate_acqs = LogEIC_inv_acq[mask]
            new_config_id = candidate_ids[torch.argmax(candidate_acqs)]
            new_config_acq = torch.max(candidate_acqs)
        if acq == "LogEIC-exp":
            candidate_acqs = LogEIC_exp_acq[mask]
            new_config_id = candidate_ids[torch.argmax(candidate_acqs)]
            new_config_acq = torch.max(candidate_acqs)
        if acq == "LCB":
            candidate_acqs = LCB_acq[mask]
            new_config_id = candidate_ids[torch.argmin(candidate_acqs)]
            new_config_acq = torch.min(candidate_acqs)
        if acq == "TS":
            prev_state = torch.get_rng_state()
            torch.manual_seed(ts_seed)
            sample_path = draw_matheron_paths(single_outcome_model, sample_shape=torch.Size([1]))
            torch.set_rng_state(prev_state)
            TS_acq = sample_path(all_x).squeeze()
            candidate_acqs = TS_acq[mask]
            new_config_id = candidate_ids[torch.argmin(candidate_acqs)]
            new_config_acq = torch.min(candidate_acqs)

        new_config_x = all_x[new_config_id]

        # 6. Query the objective for the new configuration.
        new_config_y = all_y[new_config_id]
        new_config_c = all_c[new_config_id]

        # 7.1. Get the posterior mean for old and new GPs at the new and old best points.
        # new_config_x and old_config_x should be the configurations corresponding to the current
        # and previous best indices, respectively.
        x_pair = torch.stack([new_config_x, old_config_x])

        # 7.2. Get posterior mean and covariance from the new model.
        new_posterior = single_outcome_model.posterior(x_pair)
        new_mean = new_posterior.mean         # Shape: [2]
        new_covar = new_posterior.mvn.covariance_matrix     # Shape: [2, 2]

        # 7.3. Get posterior mean and covariance from the old model.
        old_posterior = old_model.posterior(x_pair)
        old_mean = old_posterior.mean           # Shape: [2]
        old_covar = old_posterior.mvn.covariance_matrix       # Shape: [2, 2]

        # 7.4. Compute delta_mu (the absolute change in best posterior mean)
        # Here, we assume that new_config_x corresponds to the current best (new point)
        # and old_config_x corresponds to the previous best.
        delta_mu = abs(old_mean[1].item() - new_mean[0].item())

        # 7.5. Compute κ_{t−1} = UCB - LCB gap.
        kappa = torch.min(UCB_acq[~mask]) - torch.min(LCB_acq)

        # 7.6. Compute KL divergence between old and new posteriors at the new point.
        old_var = old_covar[0, 0].clamp(min=1e-12)
        new_var = new_covar[0, 0].clamp(min=1e-12)
        old_mu_val = old_mean[0]
        new_mu_val = new_mean[0]
        kl = 0.5 * (torch.log(new_var / old_var) +
                    (old_var + (old_mu_val - new_mu_val).pow(2)) / new_var - 1).item()

        # 7.7. Compute ei_diff, the expected-improvement gap difference.
        # If new_config_x and old_config_x are (approximately) equal, we set ei_diff to zero.
        if not torch.allclose(new_config_x, old_config_x, atol=1e-4):
            # We use the new model's posterior for these two points.
            # new_mean and new_covar already contain the predictions.
            # Compute the difference in means:
            g = (new_mean[0] - new_mean[1]).item()
            # Compute the effective variance difference
            diff_var = (new_covar[0, 0] - 2 * new_covar[0, 1] + new_covar[1, 1]).item()
            if diff_var < 0:
                beta_val = 0.0
                pdf_val = np.sqrt(1.0 / (2 * np.pi))
                cdf_val = 1.0
            else:
                beta_val = np.sqrt(diff_var)
                u = g / beta_val if beta_val > 0 else 0.0
                pdf_val = norm.pdf(u)
                cdf_val = norm.cdf(u)
            ei_diff = beta_val * pdf_val + g * cdf_val
        else:
            ei_diff = 0.0

        # 7.8. Final expression for ΔR̃_t (the expected minimal regret gap).
        exp_min_regret_gap = delta_mu + ei_diff + kappa.item() * np.sqrt(0.5 * kl)
        acq_history['exp min regret gap'].append(exp_min_regret_gap)
        acq_history['regret upper bound'].append(kappa.item())

        # 7.9. Reassign old_model and old_config_x for the next iteration.
        old_model = single_outcome_model
        old_config_x = new_config_x

        # Probabilistic regret bound
        paths = draw_matheron_paths(single_outcome_model, sample_shape=torch.Size([num_samples]))
        best_x = all_x[config_id_history[y.argmin().item()]]
        regrets = paths(best_x.unsqueeze(0)).squeeze(-1) - paths(all_x).min(dim=1).values
        prb_estimate = (regrets <= epsilon).float().mean().item()
        acq_history['PRB'].append(prb_estimate)
        num_samples = min(math.ceil(num_samples * 1.5), 1000)

        # Other stopping rules
        acq_history['StablePBGI(1e-3)'].append(torch.min(StablePBGI_1e_3_acq).item())
        acq_history['StablePBGI(1e-4)'].append(torch.min(StablePBGI_1e_4_acq).item())
        acq_history['StablePBGI(1e-5)'].append(torch.min(StablePBGI_1e_5_acq).item())
        acq_history['LogEIC-inv'].append(torch.max(LogEIC_inv_acq[mask]).item())
        acq_history['LogEIC-exp'].append(torch.max(LogEIC_exp_acq[mask]).item())
        
        # 8. Append the new data to our training set.
        x = torch.cat([x, new_config_x.unsqueeze(0)], dim=0)
        y = torch.cat([y, new_config_y.unsqueeze(0)], dim=0)
        c = torch.cat([c, new_config_c.unsqueeze(0)], dim=0)
        config_id_history.append(new_config_id.item())
        best_y_history.append(best_f.item())
        best_id_history.append(config_id_history[y.argmin().item()])
        cost_history.append(new_config_c.item())

        print(f"Iteration {i + 1}:")
        print(f"  Selected config_id: {new_config_id}")
        print(f"  Acquisition value: {new_config_acq.item():.4f}")
        print(f"  Objective (validation error): {new_config_y.item():.4f}")
        print(f"  Cost (runtime): {new_config_c.item():.4f}")
        print(f"  Current best observed: {best_f.item():.4f}")
        print()

        del StablePBGI_1e_3, StablePBGI_1e_4, StablePBGI_1e_5
        del LogEIC_inv, LogEIC_exp, UCB, LCB
        gc.collect()

    best_y_history.append(y.min().item())

    # Return the history including the acq_history dictionary.
    return (cost_history,
            [best_id_history[0]] + config_id_history[-n_iter:], 
            best_id_history,
            best_y_history,
            acq_history)

wandb.init(reinit=True, sync_tensorboard=False, settings=wandb.Settings(_disable_stats=True))

result = run_bayesopt_experiment(wandb.config)

(cost_history, config_id_history, best_id_history, best_y_history, acq_history) = result

cumulative_costs = np.cumsum(cost_history)

# Log full info
for idx in range(len(cost_history)):
    log_dict = {
        "config id": config_id_history[idx],
        "cumulative cost": cumulative_costs[idx],
        "current best id": best_id_history[idx],
        "current best observed": best_y_history[idx],
        "StablePBGI(1e-3) acq": acq_history['StablePBGI(1e-3)'][idx],
        "StablePBGI(1e-4) acq": acq_history['StablePBGI(1e-4)'][idx],
        "StablePBGI(1e-5) acq": acq_history['StablePBGI(1e-5)'][idx],
        "LogEIC-inv acq": acq_history['LogEIC-inv'][idx],
        "LogEIC-exp acq": acq_history['LogEIC-exp'][idx],
        "exp min regret gap": acq_history['exp min regret gap'][idx],
        "regret upper bound": acq_history['regret upper bound'][idx],
        "PRB": acq_history['PRB'][idx]
    }
    wandb.log(log_dict)

wandb.finish()