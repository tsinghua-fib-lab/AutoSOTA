#!/usr/bin/env python3
"""Reproduction runner for paper 3099: MNIST->USPS MLP W2R2.
Runs the exact experiment matching rubric conditions:
MLP, Adam, lr=1e-3, batch_size=64, epochs=10, pretrain_epochs=5, n_trials=5,
margin multiclass loss, mm-UOT solver, lambda=1.

Usage: python run_reproduction.py
"""
import builtins
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from lightning import Fabric

import utils
import adapt
import loss
from shifts.init_scenario import init as init_scenario
from debug.debug import Debugger
from reproduction_config import setup_reproduction_config
from load_model import load_model, init_model, pretrain_model


def reset_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_algorithm(config, name, model, loss_fun, opt, scenario, fabric):
    if name == "wrr":
        alg = adapt.wrr.WRR(config["wrr"], fabric, model, loss_fun, opt)
    elif name == "weighted_wrr":
        alg = adapt.weighted_wrr.WeightedWRR(
            config["weighted_wrr"], fabric, model, loss_fun, opt
        )
    elif name == "cons_wrr":
        alg = adapt.constrained_wrr.ConstrainedWRR(
            config["cons_wrr"], fabric, model, loss_fun, opt
        )
    elif name == "lje":
        alg = adapt.oracle.OracleLJE(fabric, model, loss_fun, opt)
    elif name == "cc":
        alg = adapt.oracle.OracleCC(config["cc"], fabric, model, loss_fun, opt)
    elif name == "erm":
        alg = adapt.erm.ERM(model, fabric, loss_fun, opt)
    else:
        raise Exception("UDA method not found!")
    return alg, model


def report_init_performance(config, model, scenario, loss_fun, fabric):
    methods = config["algs"]
    num_methods = len(methods)
    num_epochs = config["num_epochs"]
    num_runs = config["num_runs"]
    results = torch.zeros(num_methods, num_runs, num_epochs + 1, 4)
    res_pretrain = utils.report_metrics(
        scenario, model, loss_fun,
        config["report_source_train_risk"],
        config["report_target_train_risk"],
        fabric,
    )
    for i in range(num_methods):
        for j in range(num_runs):
            results[i, j, 0, :] = res_pretrain
    return results


def setup_uda(config, fabric):
    scenario = init_scenario(config["scenario_options"], fabric)
    model = init_model(config, scenario)
    loss_fun = init_loss(config)
    if config["pretrain"]:
        opt = init_opt(config, model)
        model = pretrain_model(model, config, fabric, scenario, loss_fun, opt)
    else:
        model = fabric.setup(model)
    results = report_init_performance(config, model, scenario, loss_fun, fabric)
    return scenario, model, loss_fun, results


def run_uda(config, fabric):
    methods = config["algs"]
    num_methods = len(methods)
    num_epochs = config["num_epochs"]
    num_runs = config["num_runs"]
    scenario, model, loss_fun, results = setup_uda(config, fabric)

    for i in range(num_methods):
        for j in range(num_runs):
            reset_all(seed=j)
            debugger = Debugger(scenario)
            model = load_model(config, fabric, scenario)
            loss_fun = init_loss(config)
            opt = init_opt(config, model)
            alg, model = init_algorithm(
                config, methods[i], model, loss_fun, opt, scenario, fabric
            )
            print("===============================")
            print(f"Algorithm {alg.name}, run number: {j}")
            for epoch in range(num_epochs):
                batch_idx = 0
                print(f"Epoch {epoch+1}")
                for (X_train, y_train), (X_shift, y_shift) in zip(
                    scenario.source_dataloader, scenario.target_dataloader
                ):
                    y_train = utils.one_hot(y_train, scenario.num_classes)
                    y_shift = utils.one_hot(y_shift, scenario.num_classes)
                    alg.adapt(model, fabric, X_train, y_train, X_shift, y_shift)

                    if batch_idx % 10 == 0:
                        print(f"Batch id: {batch_idx}")
                    batch_idx += 1
                    if (config["n_batches_per_epoch"] != -1
                            and batch_idx % config["n_batches_per_epoch"] == 0):
                        print("Terminating epoch early for debugging purposes...")
                        break

                print("===============================")
                print(f"Algorithm {alg.name}")
                results[i, j, epoch + 1, :] = utils.report_metrics(
                    scenario, model, loss_fun,
                    config["report_source_train_risk"],
                    config["report_target_train_risk"],
                    fabric,
                )
    return results


def init_loss(config):
    if config["loss"] == "margin":
        loss_fun = loss.MarginLoss()
    elif config["loss"] == "euclidean":
        loss_fun = loss.EuclideanLoss()
    elif config["loss"] == "cross-entropy":
        loss_fun = loss.CELoss()
    else:
        raise Exception("Loss function not implemented!")
    return loss_fun


def init_opt(config, model):
    if config["optimizer"] == "adam":
        opt = torch.optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
            betas=(0.9, config["adam_beta2"]),
            eps=1e-8,
        )
    elif config["optimizer"] == "sgd":
        opt = torch.optim.SGD(
            model.parameters(),
            lr=config["learning_rate"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
        )
    else:
        raise Exception("Unknown optimizer!")
    return opt


def save_results(results, config):
    scenario_name = config["scenario_options"]["scenario"]
    model_name = config["model"]
    folder_name = os.path.join("results", scenario_name, model_name)
    os.makedirs(folder_name, exist_ok=True)
    methods = config["algs"]
    num_methods, num_runs, num_epochs, _ = results.shape
    metrics = ["loss_source", "acc_source", "loss_target", "acc_target"]

    data = {"config": config}
    for m, metric in enumerate(metrics):
        data[metric] = {}
        fig, ax = plt.subplots()
        for i, method in enumerate(methods):
            data[metric][method] = results[i, :, :, m]
            stds, means = torch.std_mean(results[i, :, :, m], dim=0)
            ax.errorbar(x=np.arange(num_epochs), y=means.cpu(), yerr=stds.cpu(), label=method)
        ax.legend(loc="lower right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.set_title(f"{scenario_name} {model_name} - {metric}")
        plt.savefig(os.path.join(folder_name, metric + ".pdf"), format="pdf")
        plt.close()
    print(f"Saving metric plots and data in folder {folder_name}")
    torch.save(data, os.path.join(folder_name, "metrics.pth"))

    # Also save a JSON summary for downstream parsing
    import json
    summary = {}
    for m, metric in enumerate(metrics):
        summary[metric] = {}
        for i, method in enumerate(methods):
            vals = results[i, :, -1, m].cpu().tolist()  # final epoch
            mean_val = float(torch.mean(results[i, :, -1, m]).cpu())
            std_val = float(torch.std(results[i, :, -1, m]).cpu())
            summary[metric][method] = {
                "mean": mean_val,
                "std": std_val,
                "per_run": vals,
            }
    with open(os.path.join(folder_name, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {folder_name}/summary.json")
    return summary


def main():
    torch.set_default_dtype(torch.float32)
    config = setup_reproduction_config()
    fabric = Fabric(accelerator=config["device"], devices=1, strategy="auto")
    fabric.launch()

    if fabric.global_rank != 0:
        builtins.print = lambda *args, **kwargs: None

    print(f"Fabric device: {fabric.device}")
    print(f"Config: model={config['model']}, algs={config['algs']}, "
          f"epochs={config['num_epochs']}, pretrain_epochs={config['num_pretrain_epochs']}, "
          f"runs={config['num_runs']}, lr={config['learning_rate']}, loss={config['loss']}")
    print(f"Weighted WRR config: {config['weighted_wrr']}")

    reset_all(seed=0)
    res = run_uda(config, fabric)
    summary = save_results(res, config)

    # Print final metrics for parsing
    metrics = ["loss_source", "acc_source", "loss_target", "acc_target"]
    for m_idx, metric in enumerate(metrics):
        for method in config["algs"]:
            final_vals = res[0, :, -1, m_idx].cpu()
            mean_v = float(torch.mean(final_vals))
            std_v = float(torch.std(final_vals))
            print(f"FINAL_METRIC {metric} {method} mean={mean_v:.6f} std={std_v:.6f} "
                  f"per_run={final_vals.tolist()}")


if __name__ == "__main__":
    main()
