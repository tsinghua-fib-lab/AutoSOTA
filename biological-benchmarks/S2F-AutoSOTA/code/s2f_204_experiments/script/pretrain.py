import os
import sys
import math
import random
import pprint

import numpy as np
import torch
import torch_geometric.data

from torchvision import datasets
from torchdrug import core, models
from torchdrug.utils import comm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import util
from s3f import dataset, model, task, gvp


def train_and_validate(cfg, solver):
    if cfg.train.num_epoch == 0:
        return

    step = math.ceil(cfg.train.num_epoch / 50)
    best_result = float("-inf")
    best_epoch = -1

    for i in range(0, cfg.train.num_epoch, step):
        kwargs = cfg.train.copy()
        kwargs["num_epoch"] = min(step, cfg.train.num_epoch - i)
        solver.train(**kwargs)
        metric = solver.evaluate("valid")
        result = metric[cfg.metric]
        if result > best_result:
            best_result = result
            best_epoch = solver.epoch
            # Save only the best checkpoint (overwrite) to bound disk usage
            solver.save("model_best.pth")
        # IDEA-301: additionally snapshot the weights every 10 epochs so the
        # epoch can later be selected by a downstream ProteinGym signal
        # (sel-24) instead of CATH validation accuracy, which is empirically
        # anti-correlated with downstream Spearman (failures.md F2/T3).
        if solver.epoch % 10 == 0:
            solver.save("model_ep%d.pth" % solver.epoch)

    solver.load("model_best.pth")
    return solver


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    # IDEA-402: allow launching different training seeds without touching the
    # harness command line. Each DDP rank still gets seed + rank, so distinct
    # S2F_SEED values give fully decorrelated training trajectories.
    if os.environ.get("S2F_SEED"):
        args.seed = int(os.environ["S2F_SEED"])

    seed = args.seed + comm.get_rank()
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger = util.get_root_logger()
    if comm.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))

    dataset = core.Configurable.load_config_dict(cfg.dataset)
    solver = util.build_solver(cfg, dataset)

    train_and_validate(cfg, solver)