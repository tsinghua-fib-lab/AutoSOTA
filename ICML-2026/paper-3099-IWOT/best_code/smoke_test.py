#!/usr/bin/env python3
"""Quick smoke test: 1 run, 1 epoch for paper 3099."""
import builtins, os, numpy as np, torch, matplotlib, json
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightning import Fabric
import utils, adapt, loss
from shifts.init_scenario import init as init_scenario
from reproduction_config import setup_reproduction_config
from load_model import load_model, init_model, pretrain_model

def reset_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def init_alg(cfg, name, model, lf, opt, sc, fab):
    if name == "weighted_wrr":
        alg = adapt.weighted_wrr.WeightedWRR(cfg["weighted_wrr"], fab, model, lf, opt)
    else:
        raise Exception("Unknown")
    return alg, model

def init_loss(config):
    if config["loss"] == "margin":
        return loss.MarginLoss()
    raise Exception("Unknown loss")

def init_opt(config, model):
    return torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
        weight_decay=config["weight_decay"], betas=(0.9, config["adam_beta2"]), eps=1e-8)

def main():
    torch.set_default_dtype(torch.float32)
    config = setup_reproduction_config()
    config["num_epochs"] = 1
    config["num_runs"] = 1
    fab = Fabric(accelerator="cuda", devices=1, strategy="auto")
    fab.launch()
    if fab.global_rank != 0:
        builtins.print = lambda *a, **kw: None
    print(f"Smoke test: {config['model']}, {config['algs']}, 1 epoch, 1 run")
    reset_all(seed=0)
    sc = init_scenario(config["scenario_options"], fab)
    model = init_model(config, sc)
    lf = init_loss(config)
    if config["pretrain"]:
        opt = init_opt(config, model)
        model = pretrain_model(model, config, fab, sc, lf, opt)
    else:
        model = fab.setup(model)
    # initial eval
    init_res = utils.report_metrics(sc, model, lf, False, False, fab)
    print(f"Initial: loss_s={init_res[0]:.4f} acc_s={init_res[1]:.4f} loss_t={init_res[2]:.4f} acc_t={init_res[3]:.4f}")
    # reload for adaptation
    model = load_model(config, fab, sc)
    lf = init_loss(config)
    opt = init_opt(config, model)
    alg, model = init_alg(config, "weighted_wrr", model, lf, opt, sc, fab)
    for epoch in range(config["num_epochs"]):
        bi = 0
        for (Xs, ys), (Xt, yt) in zip(sc.source_dataloader, sc.target_dataloader):
            ys_oh = utils.one_hot(ys, sc.num_classes)
            yt_oh = utils.one_hot(yt, sc.num_classes)
            alg.adapt(model, fab, Xs, ys_oh, Xt, yt_oh)
            if bi % 10 == 0:
                print(f"  Batch {bi}")
            bi += 1
        final_res = utils.report_metrics(sc, model, lf, False, False, fab)
        print(f"Epoch {epoch+1}: loss_s={final_res[0]:.4f} acc_s={final_res[1]:.4f} loss_t={final_res[2]:.4f} acc_t={final_res[3]:.4f}")
    print("SMOKE TEST PASSED")

if __name__ == "__main__":
    main()
