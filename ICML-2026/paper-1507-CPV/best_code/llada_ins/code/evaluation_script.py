import os
import torch
import random
import numpy as np
from dllm_eval.__main__ import cli_evaluate

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
if __name__ == "__main__":
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
    set_seed(42)
    cli_evaluate()
