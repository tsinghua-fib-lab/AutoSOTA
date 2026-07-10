import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

__all__ = ["set_seed"]


def set_seed(
        seed: int,
        cudnn_benchmark: bool = True,
        cudnn_deterministic: bool = False,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if cudnn_benchmark:
            cudnn.benchmark = True
        if cudnn_deterministic:
            cudnn.deterministic = True
