import os
import torch
import sys
import torch.nn as nn
sys.path.extend(['./', '../', '../../'])
from models import Nonstationary_Transformer, DLinear, \
    PatchTST, iTransformer, TimeMixer, TimeXer, S_Mamba, BiMamba4TS, Chronos


class _PointBaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_dict = {
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'PatchTST': PatchTST,
            'iTransformer': iTransformer,
            'TimeMixer': TimeMixer,
            'TimeXer': TimeXer,
        }
        self.model_dict['S_Mamba'] = S_Mamba
        self.model_dict['BiMamba4TS'] = BiMamba4TS
        self.model_dict['Chronos'] = Chronos
