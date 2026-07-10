"""Script for testing unsupervised domain adaptation algorithms in several different distribution shift scenarios."""
import builtins
import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from lightning import Fabric
from lightning.fabric.strategies import DDPStrategy

# Code from this repo
import utils
import adapt
import loss
from shifts.init_scenario import init as init_scenario
from debug.debug import Debugger
from config.local_config import setup_local_config
from config.cluster_config import setup_cluster_config
from load_model import load_model, init_model, pretrain_model
from models.prob_model import ProbModel

def reset_all(seed):
    # Python & NumPy
    np.random.seed(seed)

    # PyTorch CPU/CUDA
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Determinism (optional but recommended for fair comps)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_algorithm(config, name, model, loss_fun, opt, scenario, fabric):
    # Prepare adaptation methods
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
    elif name == "pseudolabel":
        alg = adapt.Pseudolabel(config["pseudolabel"], fabric, model, loss_fun, opt)
    elif name == "jdot":
        alg = adapt.jdot.JDOT(config["jdot"], fabric, model, loss_fun, opt)
    elif name == "dann":
        alg = adapt.dann.DANN(config["dann"], fabric, model, loss_fun, scenario)
    elif name == "fdal":
        alg = adapt.fdal.FDAL(config["fdal"], fabric, model, loss_fun, scenario)
    elif name == "mmd":
        alg = adapt.mmd.MMD(config["mmd"], fabric, model, loss_fun, opt)
    elif name == "reverse_kl":
        model = ProbModel(model)
        alg = adapt.reverse_kl.ReverseKL(
            config["reverse_kl"], fabric, model, loss_fun, opt
        )
    else:
        raise Exception("UDA method not found!")
    return alg, model


def report_init_performance(config, model, scenario, loss_fun, fabric):
    methods = config["algs"]
    num_methods = len(methods)
    num_epochs = config["num_epochs"]
    num_runs = config["num_runs"]
    results = torch.zeros(num_methods, num_runs, num_epochs + 1, 4)
    # reporting loss_source, acc_source, loss_target, acc_target
    res_pretrain = utils.report_metrics(
        scenario,
        model,
        loss_fun,
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

    # Run adaptation
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
                # print(f"Number of batches: {len(scenario.source_dataloader)}")
                for (X_train, y_train), (X_shift, y_shift) in zip(
                    scenario.source_dataloader, scenario.target_dataloader
                ):
                    y_train = utils.one_hot(y_train, scenario.num_classes)
                    y_shift = utils.one_hot(y_shift, scenario.num_classes)
                    alg.adapt(model, fabric, X_train, y_train, X_shift, y_shift)

                    if batch_idx % 10 == 0:
                        print(f"Batch id: {batch_idx}")
                    if config["debug"] is True and (
                        batch_idx % config["debug_every_n"] == 0
                    ):
                        debugger.calc_metrics(config, model, loss_fun, scenario, fabric)
                    batch_idx += 1
                    if (
                        config["n_batches_per_epoch"] != -1
                        and batch_idx % config["n_batches_per_epoch"] == 0
                    ):
                        print("Terminating epoch early for debugging purposes...")
                        break

                print("===============================")
                print(f"Algorithm {alg.name}")
                results[i, j, epoch + 1, :] = utils.report_metrics(
                    scenario,
                    model,
                    loss_fun,
                    config["report_source_train_risk"],
                    config["report_target_train_risk"],
                    fabric,
                )
            debugger.save_metrics_plot(config["debug_options"])
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


# For now we assume that all algorithms share the optimizer, but we can change that later
def init_opt(config, model):
    if config["optimizer"] == "adam":
        opt = torch.optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
            betas=(0.9, config["adam_beta2"]),
            eps=1e-8,
        )
    elif config["optimizer"] == "adamw":
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
            betas=(0.9, 0.999),
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
            ax.errorbar(x=np.arange(num_epochs), y=means, yerr=stds, label=method)
        ax.legend(loc="lower right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.savefig(os.path.join(folder_name, metric + ".pdf"), format="pdf")
    print(f"Saving metric plots and data in folder {folder_name}")
    torch.save(data, os.path.join(folder_name, "metrics.pth"))


def run_on_cluster(fabric, config):
    _original_print = builtins.print
    def rank_print(*args, **kwargs):
        kwargs.setdefault("flush", True)
        if config["debug_distributed"]:
            _original_print(f"[rank {fabric.global_rank}'", *args, **kwargs)
        elif fabric.global_rank == 0:
            _original_print(*args, **kwargs)
    builtins.print = rank_print

    print(f"Fabric device: {fabric.device}")
    reset_all(seed=0)

    # Run all experiments
    for i, model in enumerate(config["models"]):
        if model == "ResNet":
            if config["optimizer"] == "adam":
                print("YOU ARE TRAINING RESNETS WITH ADAM, BE CAREFUL!")
        for scenario in config["scenario_options"]["scenarios"]:
            config["model"] = model
            config["scenario_options"]["scenario"] = scenario
            res = run_uda(config, fabric)
            save_results(res, config)


def run_on_local():
    config = setup_local_config()
    fabric = Fabric(accelerator=config["device"], devices="auto", strategy="auto")
    fabric.launch()

    if fabric.global_rank != 0:
        builtins.print = lambda *args, **kwargs: None

    print(f"Fabric device: {fabric.device}")
    reset_all(seed=0)

    res = run_uda(config, fabric)
    save_results(res, config)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    #torch.set_printoptions(precision=3, sci_mode=False)
    #torch.autograd.set_detect_anomaly(True)
    #os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()

    if args.config == "local":
        run_on_local()
    elif args.config == "cluster":
        config = setup_cluster_config()
        fabric = Fabric(accelerator=config["device"], devices=config["num_devices"],
                        strategy=DDPStrategy(process_group_backend="nccl",
                                             start_method="spawn"))
        fabric.launch(run_on_cluster, config)
    else:
        raise Exception("Unknown config, choose local or cluster")
