import torch
import torch.nn as nn
import torch.nn.functional as F
from ..quant_utils.models import MatMul
import re


def _fold_bn(conv_module, bn_module):
    w = conv_module.weight.data
    y_mean = bn_module.running_mean
    y_var = bn_module.running_var
    safe_std = torch.sqrt(y_var + bn_module.eps)
    w_view = (conv_module.out_channels, 1, 1, 1)
    if bn_module.affine:
        weight = w * (bn_module.weight / safe_std).view(w_view)
        beta = bn_module.bias - bn_module.weight * y_mean / safe_std
        if conv_module.bias is not None:
            bias = bn_module.weight * conv_module.bias / safe_std + beta
        else:
            bias = beta
    else:
        weight = w / safe_std.view(w_view)
        beta = -y_mean / safe_std
        if conv_module.bias is not None:
            bias = conv_module.bias / safe_std + beta
        else:
            bias = beta
    return weight, bias

def fold_bn_into_conv(conv_module, bn_module):
    w, b = _fold_bn(conv_module, bn_module)
    if conv_module.bias is None:
        conv_module.bias = nn.Parameter(b.data)
    else:
        conv_module.bias.data = b.data
    conv_module.weight.data = w.data


def wrap_modules_in_net(net,cfg):
    wrapped_modules={}
    module_dict={}
    module_types = {"qkv":"qlinear_qkv", "proj":'qlinear_proj', 'fc1':'qlinear_MLP_1', 'fc2':"qlinear_MLP_2", 'head':'qlinear_classifier','matmul1':"qmatmul_qk", 'matmul2':"qmatmul_scorev", "reduction": "qlinear_reduction"}
    
    it=[(name,m) for name,m in net.named_modules()]
    for name,m in it:
        module_dict[name]=m
        idx=name.rfind('.')
        if idx==-1:
            idx=0
        father_name=name[:idx]
        if father_name in module_dict:
            father_module=module_dict[father_name]
        else:
            raise RuntimeError(f"father module {father_name} not found")
        if isinstance(m,nn.Conv2d):
            # Embedding Layer
            idx = idx+1 if idx != 0 else idx
            new_m=cfg.get_module("qconv",m.in_channels,m.out_channels,m.kernel_size,m.stride,m.padding,m.dilation,m.groups,m.bias is not None,m.padding_mode)
            new_m.weight.data=m.weight.data
            new_m.bias=m.bias
            replace_m=new_m
            wrapped_modules[name] = new_m
            setattr(father_module,name[idx:],replace_m)
        elif isinstance(m,nn.Linear):
            # Linear Layer
            idx = idx+1 if idx != 0 else idx
            new_m = cfg.get_module(module_types[name[idx:]],m.in_features,m.out_features)
            new_m.weight.data=m.weight.data
            new_m.bias=m.bias
            replace_m=new_m
            wrapped_modules[name] = new_m
            setattr(father_module,name[idx:],replace_m)
        elif isinstance(m,MatMul):
            # Matmul Layer
            idx = idx+1 if idx != 0 else idx
            new_m = cfg.get_module(module_types[name[idx:]])
            replace_m=new_m
            wrapped_modules[name] = new_m
            setattr(father_module,name[idx:],replace_m)
    print("Completed net wrap.")
    return wrapped_modules

def wrap_certain_modules_in_net(net,cfg,layers,modules_to_wrap,wrap_embedding=False):
    """
    wrap specific module inside transformer block of specific layer
    layers: list of integers, indicating layers to wrap
    modules_to_wrap: list of modules to wrap
    """
    wrapped_modules={}
    module_dict={}
    module_types = {"qkv":"qlinear_qkv", "proj":'qlinear_proj', 'fc1':'qlinear_MLP_1', 'fc2':"qlinear_MLP_2", 'head':'qlinear_classifier','matmul1':"qmatmul_qk", 'matmul2':"qmatmul_scorev"}
    
    it=[(name,m) for name,m in net.named_modules()]
    for name,m in it:
        module_dict[name]=m
        idx=name.rfind('.')
        if idx==-1:
            idx=0
        father_name=name[:idx]
        if father_name in module_dict:
            father_module=module_dict[father_name]
        else:
            raise RuntimeError(f"father module {father_name} not found")
        layer = re.search('\d+', name)
        if layer is not None: # inside a transformer block
            layer = int(name[layer.span()[0]:layer.span()[1]])
            if layer not in layers: continue
        if isinstance(m,nn.Conv2d):
            # Embedding Layer
            idx = idx+1 if idx != 0 else idx
            if not wrap_embedding:
                continue  # timm patch_embed use proj as well...
            # if name[idx:] not in modules_to_wrap: continue
            new_m=cfg.get_module("qconv",m.in_channels,m.out_channels,m.kernel_size,m.stride,m.padding,m.dilation,m.groups,m.bias is not None,m.padding_mode)
            new_m.weight.data=m.weight.data
            new_m.bias=m.bias
            replace_m=new_m
            wrapped_modules[name] = new_m
            setattr(father_module,name[idx:],replace_m)
        elif isinstance(m,nn.Linear):
            # Linear Layer
            idx = idx+1 if idx != 0 else idx
            if name[idx:] not in modules_to_wrap: continue
            new_m = cfg.get_module(module_types[name[idx:]],m.in_features,m.out_features)
            new_m.weight.data=m.weight.data
            new_m.bias=m.bias
            replace_m=new_m
            wrapped_modules[name] = new_m
            setattr(father_module,name[idx:],replace_m)
        elif isinstance(m,MatMul):
            # Matmul Layer
            idx = idx+1 if idx != 0 else idx
            if name[idx:] not in modules_to_wrap: continue
            new_m = cfg.get_module(module_types[name[idx:]])
            replace_m=new_m
            wrapped_modules[name] = new_m
            setattr(father_module,name[idx:],replace_m)
    print("Completed net wrap.")
    return wrapped_modules
