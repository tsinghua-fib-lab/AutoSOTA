import os
import torch
import numpy as np
import random

# Global configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def to_tensor(x):
    """Converts input to a float tensor on the configured device."""
    if isinstance(x, torch.Tensor):
        return x.to(DEVICE).float()
    return torch.tensor(x, dtype=torch.float32).to(DEVICE)

def to_numpy(x):
    """Converts tensor to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)