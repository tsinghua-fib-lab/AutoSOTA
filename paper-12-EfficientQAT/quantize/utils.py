from collections import OrderedDict
from quantize.int_linear_fake import QuantLinear
import torch
from torch import nn
from typing  import Optional


class MultiBlock(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.block_list = nn.ModuleList([])
    
    def add_block(self, block):
        self.block_list.append(block)
        
    def forward(self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None):
        for block in self.block_list:
            hidden_states = block(hidden_states, attention_mask=attention_mask,position_ids=position_ids)[0]
        return (hidden_states, )


def set_weight_parameters(model, requires_grad):
    params = []
    for n, m in model.named_parameters():
        if n.find('weight') > -1 and not (n.find('scale') > -1 or n.find('zero_point') > -1):
            m.requires_grad = requires_grad
    return iter(params)

def weight_parameters(model):
    params = []
    for n, m in model.named_parameters():
        if n.find('weight') > -1 and not (n.find('scale') > -1 or n.find('zero_point') > -1):
            params.append(m)
    return iter(params)

def set_quant_parameters(model, requires_grad):
    params = []
    for n, m in model.named_parameters():
        if n.find('scale') > -1 or n.find('zero_point') > -1:
            m.requires_grad = requires_grad
    return iter(params)  

def quant_parameters(model):
    params = []
    for n, m in model.named_parameters():
        if n.find('scale') > -1 or n.find('zero_point') > -1:
            params.append(m)
    return iter(params)  


def trainable_parameters(model):
    params = []
    for n, m in model.named_parameters():
        if m.requires_grad:
            params.append(m)
    return iter(params)  

def trainable_parameters_num(model):
    params = []
    total = 0
    for n, m in model.named_parameters():
        if m.requires_grad:
            total += m.numel()
    return total

def set_quant_state(model, weight_quant: bool = False):
    for m in model.modules():
        if isinstance(m, QuantLinear):
            m.set_quant_state(weight_quant)
            
@torch.no_grad()   
def quant_inplace(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.weight.data = module.weight_quantizer(module.weight.data)


class TruncateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        truncated_tensor = input.clone()
        truncated_tensor[truncated_tensor.abs() < threshold] = truncated_tensor[truncated_tensor.abs() < threshold].sign() * threshold
        return truncated_tensor
        

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

     
def truncate_number(number, threshold=1e-2):
    # avoid overflow with AMP training
    return TruncateFunction.apply(number, threshold)     


def get_named_linears(module, type):
    # return {name: m for name, m in module.named_modules() if isinstance(m, torch.nn.Linear)}
    return {name: m for name, m in module.named_modules() if isinstance(m, type)}

def set_op_by_name(layer, name, new_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)
        
# def add_new_module(name, original_module, added_module):
#     levels = name.split('.')
#     if len(levels) > 1:
#         mod_ = original_module
#         for l_idx in range(len(levels)-1):
#             if levels[l_idx].isdigit():
#                 mod_ = mod_[int(levels[l_idx])]
#             else:
#                 mod_ = getattr(mod_, levels[l_idx])
#         setattr(mod_, levels[-1], added_module)
#     else:
#         setattr(original_module, name, added_module)   