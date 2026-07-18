import logging
from configparser import ConfigParser
import numpy as np
import pandas as pd
import random
import os
import torch
# import wandb
logger = logging.getLogger(__name__)
msgs = []
def log_info(msg):
    logger.info(msg=msg)
    msgs.append(msg)

def parse_configs(configs_path: str) -> ConfigParser:
    parser = ConfigParser()
    parser.read(filenames=configs_path)
    return parser

def init_seeds(param):
    np.random.seed(seed=param["random_seed"])
    # pd.core.common.random_state(param["random_seed"])
    random.seed(a=param["random_seed"])
    torch.manual_seed(param["random_seed"])
    logger.info(msg=f"Seeds: Numpy {param['random_seed']}, Random {param['random_seed']}, Pytorch {param['random_seed']}")

def get_logger(version: int, log_dir: dict) -> logging.Logger:
    save_dir = log_dir
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    sh = logging.StreamHandler()
    sh.setLevel(level=logging.INFO)
    sh.setFormatter(fmt=formatter)

    filename = os.path.join(save_dir, f"logging_{version}.log")
    if os.path.exists(filename):
        os.remove(filename)
    fh = logging.FileHandler(filename=filename, mode="a", encoding="UTF-8")
    fh.setLevel(level=logging.INFO)
    fh.setFormatter(fmt=formatter)

    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    logger.addHandler(hdlr=sh)
    logger.addHandler(hdlr=fh)

    return logger

def init_folders(version: int, param: dict):
    logger.info(msg="Initializing folders ...")
    version = str(version)
    # tb_chkpt = os.path.join(param["TBada_checkpoints_dir"], version)
    lm_chkpt = os.path.join(param["checkpoints_dir"], version)
    # param["TBada_checkpoints_dir"] = tb_chkpt
    param["checkpoints_dir"] = lm_chkpt
    # param["predictions_dir"] = pred_dir
    if not os.path.exists(lm_chkpt):
        os.mkdir(lm_chkpt)
        logger.info(msg=f"{lm_chkpt} has been created!")
    else:
        for each in os.listdir(lm_chkpt):
            os.remove(os.path.join(lm_chkpt, each))
        logger.info(msg=f"{lm_chkpt} has been cleared!")
