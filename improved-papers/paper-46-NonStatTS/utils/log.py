import os
from datetime import datetime

import wandb
from yacs.config import CfgNode as CN

from utils.misc import mkdir
from config import get_norm_module_cfg


def init_wandb(cfg):
    wandb.init(
        project=cfg.WANDB.PROJECT,
        name=cfg.WANDB.NAME,
        job_type=cfg.WANDB.JOB_TYPE,
        notes=cfg.WANDB.NOTES,
        dir=cfg.WANDB.DIR,
        config=cfg
    )
    if cfg.WANDB.SET_LOG_DIR:
        # save checkpoints and results in the wandb log directory
        cfg.TRAIN.CHECKPOINT_DIR = str(mkdir(os.path.join(cfg.TRAIN.CHECKPOINT_DIR, 'wandb', wandb.run.id)))
        cfg.RESULT_DIR = str(mkdir(os.path.join(cfg.RESULT_DIR, 'wandb', wandb.run.id)))
        if cfg.NORM_MODULE.ENABLE:
            norm_module_cfg = get_norm_module_cfg(cfg)
            norm_module_cfg.TRAIN.CHECKPOINT_DIR = str(mkdir(os.path.join(norm_module_cfg.TRAIN.CHECKPOINT_DIR, 'wandb', wandb.run.id)))
            norm_module_cfg.RESULT_DIR = str(mkdir(os.path.join(norm_module_cfg.RESULT_DIR, 'wandb', wandb.run.id)))
