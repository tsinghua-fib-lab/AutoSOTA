import numpy as np
import os
import yaml
from typing import Any, Dict, Union
from random_generator import sample_h_q_w


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    if yaml is None:
        raise ImportError("PyYAML is required. Please install with: pip install pyyaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    

def _ensure_list_of_pairs(x, expected_len: int, name: str):
    arr = list(x)
    if len(arr) != expected_len:
        raise ValueError(f"Config for {name} must have length {expected_len}, got {len(arr)}")
    for idx, pair in enumerate(arr):
        if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
            raise ValueError(f"Each entry of {name} must be a pair [low, high]; bad entry at index {idx}: {pair}")
    return arr


def sample_from_config(cfg_or_path: Union[str, Dict[str, Any]], train=True, rng=None) -> Dict[str, Any]:
    """
    Load parameters from YAML config (or dict) and call the sampler.

    Expected YAML structure (keys):
      random_generator:
        n: int
        J: int
        p: float
        lam_r: float
        lam_w: float
        h_int_r: [low, high]
        q_ints_r: [[low, high], ...]  # length J
        w_ints_r: [[low, high], ...]  # length J
        h_int_w: [low, high]
        q_ints_w: [[low, high], ...]  # length J
        w_ints_w: [[low, high], ...]  # length J
        T: [[...], [...], ...]        # 2D matrix
    """
    if isinstance(cfg_or_path, str):
        cfg = load_config(cfg_or_path)
    else:
        cfg = cfg_or_path

    g = cfg.get("random_generator", cfg)
    if train:
        n = int(g["train_n"])
    else:
        n = int(g["test_n"]) 
    J = int(g["J"]) 
    p = float(g["p"]) 
    lam_r = float(g["lam_r"]) 
    lam_w = float(g["lam_w"]) 

    h_int_r = list(g["h_int_r"])  # [low, high]
    h_int_w = list(g["h_int_w"])  # [low, high]

    q_ints_r = _ensure_list_of_pairs(g["q_ints_r"], J, "q_ints_r")
    w_ints_r = _ensure_list_of_pairs(g["w_ints_r"], J, "w_ints_r")
    q_ints_w = _ensure_list_of_pairs(g["q_ints_w"], J, "q_ints_w")
    w_ints_w = _ensure_list_of_pairs(g["w_ints_w"], J, "w_ints_w")

    out = sample_h_q_w(
        n=n, J=J, p=p,
        lam_r=lam_r, lam_w=lam_w,
        h_int_r=h_int_r, q_ints_r=q_ints_r, w_ints_r=w_ints_r,
        h_int_w=h_int_w, q_ints_w=q_ints_w, w_ints_w=w_ints_w,
        seed=None, rng=rng)
    
    T_cfg = np.array(g["T"], dtype=float)
    out["T"] = T_cfg
    return out
