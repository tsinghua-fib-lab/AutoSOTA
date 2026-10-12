import os
import time
import logging
import argparse

import yaml
import jinja2
from jinja2 import meta
import easydict

import torch
from torch.utils import data as torch_data
from torch import distributed as dist

from torchdrug import core, utils
from torchdrug.utils import comm


logger = logging.getLogger(__file__)


def get_root_logger(file=True):
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    format = logging.Formatter("%(asctime)-10s %(message)s", "%H:%M:%S")

    if file:
        handler = logging.FileHandler("log.txt")
        handler.setFormatter(format)
        logger.addHandler(handler)

    return logger


def create_working_directory(cfg):
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "null")
    file_name = "%s_working_dir.tmp" % slurm_job_id
    world_size = comm.get_world_size()
    if world_size > 1 and not dist.is_initialized():
        comm.init_process_group("nccl", init_method="env://")

    if isinstance(cfg.task.get("model"), dict):
        model_class = cfg.task.model["class"]
    else:
        model_class = "Placeholder"
    working_dir = os.path.join(os.path.expanduser(cfg.output_dir),
                               cfg.task["class"], cfg.dataset["class"], model_class, slurm_job_id,
                               time.strftime("%Y-%m-%d-%H-%M-%S"))

    # synchronize working directory
    if comm.get_rank() == 0:
        with open(file_name, "w") as fout:
            fout.write(working_dir)
        os.makedirs(working_dir)
    comm.synchronize()
    if comm.get_rank() != 0:
        with open(file_name, "r") as fin:
            working_dir = fin.read()
    comm.synchronize()
    if comm.get_rank() == 0:
        os.remove(file_name)

    os.chdir(working_dir)
    return working_dir


def detect_variables(cfg_file):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    env = jinja2.Environment()
    ast = env.parse(raw)
    vars = meta.find_undeclared_variables(ast)
    return vars


def load_config(cfg_file, context=None):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    template = jinja2.Template(raw)
    instance = template.render(context)
    cfg = yaml.safe_load(instance)
    cfg = easydict.EasyDict(cfg)
    return cfg


def build_solver(cfg, dataset):
    generator = torch.Generator().manual_seed(0)
    lengths = [int(len(dataset) * cfg.split[0]), int(len(dataset) * cfg.split[1])]
    lengths.append(len(dataset) - sum(lengths))
    train_set, valid_set, test_set = torch_data.random_split(dataset, lengths, generator=generator)
    if comm.get_rank() == 0:
        logger.warning("#train: %d, #valid: %d, #test: %d" % (len(train_set), len(valid_set), len(test_set)))

    task = core.Configurable.load_config_dict(cfg.task)

    if "fix_sequence_model" in cfg:
        model = task.model
        assert cfg.task.model ["class"] == "FusionNetwork"
        for p in model.sequence_model.parameters():
            p.requires_grad = False
        cfg.optimizer.params = [p for p in task.parameters() if p.requires_grad]
    else:
        cfg.optimizer.params = task.parameters()        
    optimizer = core.Configurable.load_config_dict(cfg.optimizer)

    solver = core.Engine(task, train_set, valid_set, test_set, optimizer, **cfg.engine)

    if cfg.get("checkpoint") is not None:
        if comm.get_rank() == 0:
            logger.warning("Load checkpoint from %s" % cfg.checkpoint)
        solver.load(cfg.checkpoint)
    
    return solver


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file", required=True)
    parser.add_argument("-s", "--seed", help="random seed for PyTorch", type=int, default=1024)

    args, unparsed = parser.parse_known_args()
    # get dynamic arguments defined in the config file
    vars = detect_variables(args.config)
    parser = argparse.ArgumentParser()
    for var in vars:
        parser.add_argument("--%s" % var, default="null")
    vars = parser.parse_known_args(unparsed)[0]
    vars = {k: utils.literal_eval(v) for k, v in vars._get_kwargs()}

    return args, vars