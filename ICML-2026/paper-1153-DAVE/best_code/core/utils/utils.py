import yaml
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from timm import create_model
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform

from pathlib import Path
from typing import Dict, Tuple, Callable, Optional


def load_from_yaml(path: str | Path) -> Dict:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return raw


def load_pil_image(image_path: Path) -> np.ndarray:
    image = Image.open(image_path)
    image = image.convert("RGB")
    return image


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
        
    elif torch.cuda.is_available():
        return torch.device("cuda:0")
    
    return "cpu"


def load_timm_model(model_spec: str) -> Tuple[nn.Module, Callable]:
    model = create_model(model_spec, pretrained=True)
    model = model.eval()
    
    for p in model.parameters():
        p.requires_grad_(False)

    config = resolve_model_data_config(model)
    transform = create_transform(**config)
        
    return model, transform


def quantile_clamp_img(x: Tensor) -> Tensor:
    N, C, H, W = x.shape
    x = x.reshape(N, -1)
    q_low  = torch.quantile(x, 0.01, dim=-1, keepdim=True)
    q_high = torch.quantile(x, 0.99, dim=-1, keepdim=True)
    x = torch.clamp(x, q_low, q_high)
    return x.reshape(N, C, H, W)
    
