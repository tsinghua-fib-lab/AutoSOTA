import builtins, os, numpy as np, torch
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

torch.set_default_dtype(torch.float32)
config = setup_reproduction_config()
config["weighted_wrr"]["separate_optim"] = False
config["num_epochs"] = 5
config["num_runs"] = 1

fab = Fabric(accelerator="cuda", devices=1, strategy="auto")
fab.launch()
if fab.global_rank != 0:
    builtins.print = lambda *a, **kw: None

print("Test: separate_optim=False, 1 run, 5 epochs")
print("weighted_wrr config:", config["weighted_wrr"])
reset_all(seed=0)
sc = init_scenario(config["scenario_options"], fab)
model = init_model(config, sc)
lf = loss.MarginLoss()
opt = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
    weight_decay=config["weight_decay"], betas=(0.9, config["adam_beta2"]), eps=1e-8)
model = pretrain_model(model, config, fab, sc, lf, opt)

init_res = utils.report_metrics(sc, model, lf, False, False, fab)
print("Pretrain: acc_s={:.4f} acc_t={:.4f}".format(init_res[1], init_res[3]))

model = load_model(config, fab, sc)
lf = loss.MarginLoss()
opt = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
    weight_decay=config["weight_decay"], betas=(0.9, config["adam_beta2"]), eps=1e-8)
alg = adapt.weighted_wrr.WeightedWRR(config["weighted_wrr"], fab, model, lf, opt)

for epoch in range(config["num_epochs"]):
    bi = 0
    for (Xs, ys), (Xt, yt) in zip(sc.source_dataloader, sc.target_dataloader):
        ys_oh = utils.one_hot(ys, sc.num_classes)
        yt_oh = utils.one_hot(yt, sc.num_classes)
        alg.adapt(model, fab, Xs, ys_oh, Xt, yt_oh)
        bi += 1
    res = utils.report_metrics(sc, model, lf, False, False, fab)
    print("Epoch {}: acc_s={:.4f} acc_t={:.4f}".format(epoch+1, res[1], res[3]))
print("TEST COMPLETE")
