"""Small shared utilities."""

import logging
import os
import random
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(log_dir: str, name: str = "tap") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def save_checkpoint(
    checkpoint_dir: str,
    step: int,
    agent,
    env_state: Dict[str, Any],
    best_metric: Optional[float] = None,
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        "step": step,
        "policy_state_dict": agent.policy.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "z_0": agent.z_0,
        "env_state": env_state,
        "best_metric": best_metric,
    }
    torch.save(checkpoint, os.path.join(checkpoint_dir, "latest_checkpoint.pt"))
