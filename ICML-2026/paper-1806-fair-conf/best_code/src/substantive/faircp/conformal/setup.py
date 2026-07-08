import random
import torch
import numpy as np


def set_seed(seed: int):
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed()
