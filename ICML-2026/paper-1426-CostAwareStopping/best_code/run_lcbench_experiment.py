#!/usr/bin/env python3
"""Run LCBench BO experiments without wandb, saving results locally.

Usage: python3 run_lcbench_experiment.py --dataset Fashion-MNIST --seed 0
       python3 run_lcbench_experiment.py --all --n_workers 8
"""

import torch
import numpy as np
import math
import os
import sys
import json
import importlib.util
import argparse
import pickle
from scipy.stats import norm
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Set default tensor type to float64
torch.set_default_dtype(torch.float64)

# Import pandora_automl modules
from pandora_automl.utils import fit_gp_model, normalize_config
from pandora_automl.acquisition.stable_gittins import StableGittinsIndex
from botorch.acquisition import UpperConfidenceBound
from pandora_automl.acquisition.lcb import LowerConfidenceBound
from pandora_automl.acquisition.log_ei_puc import LogExpectedImprovementWithCost
from botorch.sampling.pathwise import draw_matheron_paths


def setup_benchmark():
    """Set up LCBench benchmark."""
    project_root = "/repo"
    api_path = os.path.join(project_root, "LCBench", "api.py")
    spec = importlib.util.spec_from_file_location("lcbench_api", api_path)
    lcbench_api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lcbench_api)
    Benchmark = lcbench_api.Benchmark
    bench_path = os.path.join(project_root, "LCBench", "cached", "six_datasets_lw.json")
    bench = Benchmark(bench_path, cache=False)
    return bench


def precompute_dataset_data(bench, dataset_name):
    """Pre-compute all_x, all_y, all_c, estimated_costs for a dataset."""
    all_x = []
    all_y = []
    all_c = []
    estimated_costs = []

    for config_id in bench.data[dataset_name].keys():
        config = bench.query(dataset_name, "config", config_id)
        x = normalize_config(config)
        all_x.append(x)
        val_acc = bench.query(dataset_name, "final_val_accuracy", config_id)
        all_y.append(100 - val_acc)
        runtime = bench.query(dataset_name, "time", config_id)[-1]
        all_c.append(runtime)
        model_param = bench.query(dataset_name, "model_parameters", config_id)
        if dataset_name in ["Fashion-MNIST", "adult", "higgs", "volkert"]:
            estimated_costs.append(0.001 * model_param)
        else:
            estimated_costs.append(0.0006 * model_param)

    all_x = torch.stack(all_x)
    all_y = torch.tensor(all_y).unsqueeze(1)
    all_c = torch.tensor(all_c).unsqueeze(1)
    estimated_costs = torch.tensor(estimated_costs).unsqueeze(1)
    num_configs = len(all_x)

    return all_x, all_y, all_c, estimated_costs, num_configs


def run_bayesopt_experiment(bayesopt_config):
    """Run a single BO experiment. Returns results dict."""
    dataset_name = bayesopt_config["dataset_name"]
    seed = bayesopt_config["seed"]
    output_standardize = bayesopt_config["output_standardize"]
    maximize = bayesopt_config["maximize"]
    dim = bayesopt_config["dim"]
    n_iter = bayesopt_config["num_iteration"]
    num_configs = bayesopt_config["num_configs"]
    acq = bayesopt_config["acquisition_function"]

    # Pre-computed data
    all_x = bayesopt_config["all_x"]
    all_y = bayesopt_config["all_y"]
    all_c = bayesopt_config["all_c"]
    estimated_costs = bayesopt_config["estimated_costs"]

    # Sample initial configurations
    torch.manual_seed(seed)
    init_config_id = torch.randint(low=0, high=num_configs, size=(2 * (dim + 1),))
    config_id_history = init_config_id.tolist()
    x = all_x[init_config_id]
    y = all_y[init_config_id]
    c = all_c[init_config_id]

    best_y_history = [y.min().item()]
    best_id_history = [config_id_history[y.argmin().item()]]
    cost_history = [0]
    estimated_cost_history = [0]

    # Initialization for expected minimum simple regret gap
    old_model = fit_gp_model(X=x[:-1], objective_X=y[:-1], output_standardize=output_standardize)
    old_config_x = x[-1]

    # PRB initialization
    best_acc = bayesopt_config["best_acc"]
    epsilon = 0.5 * 0.01 * (100 - best_acc)
    num_samples = 64

    # TS seed
    ts_seed = seed + 1

    acq_history = {
        "PBGI(1e-3)": [np.nan],
        "PBGI(1e-4)": [np.nan],
        "PBGI(1e-5)": [np.nan],
        "LogEIPC": [np.nan],
        "regret upper bound": [np.nan],
        "exp min regret gap": [np.nan],
        "PRB": [np.nan],
    }

    for i in range(n_iter):
        # 1. Fit GP model
        model = fit_gp_model(X=x, objective_X=y, output_standardize=output_standardize)

        # 2. Best observed
        best_f = y.min()

        # 3. Define acquisition functions
        PBGI_1e_3 = StableGittinsIndex(model=model, maximize=maximize, lmbda=1e-3)
        PBGI_1e_4 = StableGittinsIndex(model=model, maximize=maximize, lmbda=1e-4)
        PBGI_1e_5 = StableGittinsIndex(model=model, maximize=maximize, lmbda=1e-5)
        LogEIPC = LogExpectedImprovementWithCost(model=model, best_f=best_f, maximize=maximize)
        beta = 2 * np.log(dim * ((i + 1) ** 2) * (math.pi ** 2) / (6 * 0.1)) / 5
        UCB = UpperConfidenceBound(model=model, maximize=maximize, beta=beta)
        LCB = LowerConfidenceBound(model=model, maximize=maximize, beta=beta)

        # 4. Evaluate all acquisition functions
        PBGI_1e_3_acq = PBGI_1e_3.forward(all_x.unsqueeze(1), cost_X=estimated_costs)
        PBGI_1e_3_acq[config_id_history] = y.squeeze(-1)
        PBGI_1e_4_acq = PBGI_1e_4.forward(all_x.unsqueeze(1), cost_X=estimated_costs)
        PBGI_1e_4_acq[config_id_history] = y.squeeze(-1)
        PBGI_1e_5_acq = PBGI_1e_5.forward(all_x.unsqueeze(1), cost_X=estimated_costs)
        PBGI_1e_5_acq[config_id_history] = y.squeeze(-1)
        LogEIPC_acq = LogEIPC.forward(all_x.unsqueeze(1), cost_X=estimated_costs)
        UCB_acq = UCB.forward(all_x.unsqueeze(1))
        LCB_acq = LCB.forward(all_x.unsqueeze(1))

        # 5. Select candidate based on acquisition function
        all_ids = torch.arange(num_configs)
        mask = torch.ones(num_configs, dtype=torch.bool)
        mask[config_id_history] = False
        candidate_ids = all_ids[mask]

        if acq == "PBGI(1e-3)":
            candidate_acqs = PBGI_1e_3_acq[mask]
            new_config_id = candidate_ids[torch.argmin(candidate_acqs)]
        elif acq == "PBGI(1e-4)":
            candidate_acqs = PBGI_1e_4_acq[mask]
            new_config_id = candidate_ids[torch.argmin(candidate_acqs)]
        elif acq == "PBGI(1e-5)":
            candidate_acqs = PBGI_1e_5_acq[mask]
            new_config_id = candidate_ids[torch.argmin(candidate_acqs)]
        elif acq == "LogEIPC":
            candidate_acqs = LogEIPC_acq[mask]
            new_config_id = candidate_ids[torch.argmax(candidate_acqs)]
        elif acq == "LCB":
            candidate_acqs = LCB_acq[mask]
            new_config_id = candidate_ids[torch.argmin(candidate_acqs)]
        elif acq == "TS":
            prev_state = torch.get_rng_state()
            torch.manual_seed(ts_seed)
            sample_path = draw_matheron_paths(model, sample_shape=torch.Size([1]))
            torch.set_rng_state(prev_state)
            TS_acq = sample_path(all_x).squeeze()
            candidate_acqs = TS_acq[mask]
            new_config_id = candidate_ids[torch.argmin(candidate_acqs)]

        new_config_x = all_x[new_config_id]

        # 6. Query objective
        new_config_y = all_y[new_config_id]
        new_config_c = all_c[new_config_id]
        new_config_estimated_c = estimated_costs[new_config_id]

        # 7. Record stopping information
        x_pair = torch.stack([new_config_x, old_config_x])
        new_posterior = model.posterior(x_pair)
        new_mean = new_posterior.mean
        new_covar = new_posterior.mvn.covariance_matrix
        old_posterior = old_model.posterior(x_pair)
        old_mean = old_posterior.mean
        old_covar = old_posterior.mvn.covariance_matrix
        delta_mu = abs(old_mean[1].item() - new_mean[0].item())
        kappa = torch.min(UCB_acq[~mask]) - torch.min(LCB_acq)

        old_var = old_covar[0, 0].clamp(min=1e-12)
        new_var = new_covar[0, 0].clamp(min=1e-12)
        old_mu_val = old_mean[0]
        new_mu_val = new_mean[0]
        kl = 0.5 * (torch.log(new_var / old_var) +
                    (old_var + (old_mu_val - new_mu_val).pow(2)) / new_var - 1).item()

        if not torch.allclose(new_config_x, old_config_x, atol=1e-4):
            g = (new_mean[0] - new_mean[1]).item()
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

        exp_min_regret_gap = delta_mu + ei_diff + kappa.item() * np.sqrt(0.5 * kl)
        acq_history["exp min regret gap"].append(exp_min_regret_gap)
        acq_history["regret upper bound"].append(kappa.item())

        old_model = model
        old_config_x = new_config_x

        # PRB
        paths = draw_matheron_paths(model, sample_shape=torch.Size([num_samples]))
        best_x = all_x[config_id_history[y.argmin().item()]]
        regrets = paths(best_x.unsqueeze(0)).squeeze(-1) - paths(all_x).min(dim=1).values
        prb_estimate = (regrets <= epsilon).float().mean().item()
        acq_history["PRB"].append(prb_estimate)
        num_samples = min(math.ceil(num_samples * 1.5), 1000)

        # Other stopping rules
        acq_history["PBGI(1e-3)"].append(torch.min(PBGI_1e_3_acq).item())
        acq_history["PBGI(1e-4)"].append(torch.min(PBGI_1e_4_acq).item())
        acq_history["PBGI(1e-5)"].append(torch.min(PBGI_1e_5_acq).item())
        acq_history["LogEIPC"].append(torch.max(LogEIPC_acq[mask]).item())

        # 8. Append data
        x = torch.cat([x, new_config_x.unsqueeze(0)], dim=0)
        y = torch.cat([y, new_config_y.unsqueeze(0)], dim=0)
        c = torch.cat([c, new_config_c.unsqueeze(0)], dim=0)
        config_id_history.append(new_config_id.item())
        best_y_history.append(best_f.item())
        best_id_history.append(config_id_history[y.argmin().item()])
        cost_history.append(new_config_c.item())
        estimated_cost_history.append(new_config_estimated_c.item())

    best_y_history.append(y.min().item())

    return {
        "cost_history": cost_history,
        "estimated_cost_history": estimated_cost_history,
        "config_id_history": [best_id_history[0]] + config_id_history[-n_iter:],
        "best_id_history": best_id_history,
        "best_y_history": best_y_history,
        "acq_history": acq_history,
    }


def run_single(args_tuple):
    """Run a single experiment, handling torch multiprocessing."""
    dataset_name, seed, acq, bayesopt_config_template, bench = args_tuple

    # Re-setup in worker process
    config = bayesopt_config_template.copy()
    config["dataset_name"] = dataset_name
    config["seed"] = seed
    config["acquisition_function"] = acq

    # Precompute dataset data
    all_x, all_y, all_c, estimated_costs, num_configs = precompute_dataset_data(bench, dataset_name)
    config["all_x"] = all_x
    config["all_y"] = all_y
    config["all_c"] = all_c
    config["estimated_costs"] = estimated_costs
    config["num_configs"] = num_configs

    # Get best accuracy
    best_acc = -1
    for cid in bench.data[dataset_name].keys():
        acc = bench.query(dataset_name, "final_test_accuracy", cid)
        if acc > best_acc:
            best_acc = acc
    config["best_acc"] = best_acc

    result = run_bayesopt_experiment(config)
    return dataset_name, seed, acq, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--acq", type=str, default="PBGI(1e-4)")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--output", type=str, default="/repo/lcbench_results.pkl")
    args = parser.parse_args()

    bench = setup_benchmark()
    all_datasets = bench.get_dataset_names()

    base_config = {
        "output_standardize": True,
        "maximize": False,
        "dim": 7,
        "num_iteration": 200,
    }

    if not args.all:
        # Single run
        dataset_name = args.dataset or "Fashion-MNIST"
        seed = args.seed or 0

        print(f"Running single experiment: dataset={dataset_name}, seed={seed}, acq={args.acq}")

        config = base_config.copy()
        config["dataset_name"] = dataset_name
        config["seed"] = seed
        config["acquisition_function"] = args.acq

        all_x, all_y, all_c, estimated_costs, num_configs = precompute_dataset_data(bench, dataset_name)
        config["all_x"] = all_x
        config["all_y"] = all_y
        config["all_c"] = all_c
        config["estimated_costs"] = estimated_costs
        config["num_configs"] = num_configs

        best_acc = -1
        for cid in bench.data[dataset_name].keys():
            acc = bench.query(dataset_name, "final_test_accuracy", cid)
            if acc > best_acc:
                best_acc = acc
        config["best_acc"] = best_acc

        result = run_bayesopt_experiment(config)
        print(f"Completed. Best observed: {result['best_y_history'][-1]:.4f}")
        return

    # Full run
    seeds = list(range(50))
    acq = args.acq
    print(f"Running full experiment: {len(all_datasets)} datasets x {len(seeds)} seeds x 1 acq = {len(all_datasets)*len(seeds)} runs")
    print(f"Acquisition: {acq}, Workers: {args.n_workers}")

    # Prepare results dict
    results = {}
    for d in all_datasets:
        results[d] = {}

    # Create task list
    tasks = [(d, s, acq, base_config, bench) for d in all_datasets for s in seeds]

    if args.n_workers > 1:
        with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
            futures = [executor.submit(run_single, t) for t in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Running"):
                d, s, a, result = future.result()
                results[d][s] = result
                # Save incrementally
                with open(args.output, "wb") as f:
                    pickle.dump(results, f)
    else:
        for task in tqdm(tasks, desc="Running"):
            d, s, a, result = run_single(task)
            results[d][s] = result
            # Save incrementally every 10 runs
            if (len(results[d]) * len(all_datasets)) % 10 == 0:
                with open(args.output, "wb") as f:
                    pickle.dump(results, f)

    # Final save
    with open(args.output, "wb") as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
