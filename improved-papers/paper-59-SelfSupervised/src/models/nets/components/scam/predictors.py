import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.ensemble import EnsembleModule, TensorDictModule
# from tensordict.nn import TensorDictModule
# from tensordict.nn import TensorDictModule, EnsembleModule
from tensordict import TensorDict
from einops import rearrange
from dataclasses import dataclass


class MultiPredictor(nn.Module):
    def __init__(self,
                 pred_model: nn.Module,
                 num_models: int, 
                 input_size: int,
                 output_size: int,
                 num_channels: int,
                 **kwargs):
        super(MultiPredictor, self).__init__()
        self.pred_model = SinglePredictor(pred_model=pred_model, 
                                          input_size=input_size, 
                                          output_size=output_size, 
                                          num_channels=num_channels, 
                                          num_models=1)
        mod = TensorDictModule(self.pred_model, in_keys=['x', 'x_mark'], out_keys=['y_hat'])
        self.models = EnsembleModule(mod, num_copies=num_models, randomness='different')

    def forward(self, x, x_mark=None, *args, **kwargs):
        for name, param in self.models.named_parameters():
            assert param.device == x.device, f'{name} is not on the same device as x'
        for name, buffer in self.models.named_buffers():
            assert buffer.device == x.device, f'{name} is not on the same device as x'
        
        
        data = TensorDict({'x': x, 'x_mark': x_mark}, batch_size=x.shape[0])
        y_hats = self.models(data)['y_hat']
        y_hats = rearrange(y_hats, 'n b c l -> b c n l')
        return y_hats
    
class SinglePredictor(nn.Module):
    def __init__(self, 
                 pred_model: nn.Module,
                 num_models: int,
                 input_size: int,
                 output_size: int,
                 num_channels: int,
                 **kwargs):
        super(SinglePredictor, self).__init__()
        self.pred_model = pred_model(input_size=input_size, output_size=output_size, num_channels=num_channels)
        self.num_models = num_models
        for name, param in self.pred_model.named_buffers():
            assert not (param.dtype != torch.float32 and param.requires_grad), f'{name} is not torch.float32, is {param.dtype}'
                
        
    def forward(self, x, x_mark, *args, **kwargs):
        y_hat = self.pred_model.predict(x, x_mark)
        if self.num_models > 1:
            y_hat = y_hat[:, :, None, :].repeat(1, 1, self.num_models, 1)
        return y_hat
