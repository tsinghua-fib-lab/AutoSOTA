from typing import Optional

import torch
import torch.nn as nn

from models.Statistics_prediction import Statistics_prediction


def normalize_enc_window(enc_window: torch.Tensor, norm_module: Optional[nn.Module] = None) -> torch.Tensor:
    if norm_module is None:  #! NST style
        means = enc_window.mean(1, keepdim=True).detach()
        enc_window = enc_window - means
        stdev = torch.sqrt(torch.var(enc_window, dim=1, keepdim=True, unbiased=False) + 1e-5)
        enc_window /= stdev
    elif isinstance(norm_module, Statistics_prediction):
        enc_window, statistics_pred = norm_module.normalize(enc_window)
        
    return enc_window, means, stdev
