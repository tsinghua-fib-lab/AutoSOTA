import torch
import torch.nn as nn
from collections import OrderedDict
from torch.nn import functional as F
import numpy as np
import copy
import psutil
from typing import List, Optional, Tuple, Union

from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def get_model(arch, seed, pretrained=True, checkpoints_dir=None):
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(arch, torch_dtype='auto', cache_dir=checkpoints_dir, attn_implementation="eager")
    model.seqlen = 2048
    model.config.output_attentions = True

    criterion = torch.nn.functional.cross_entropy
    layers_to_prune = find_layers(model)

    for layer in list(layers_to_prune.keys()):
        if "layers" not in layer:
            del layers_to_prune[layer]

    l_blocks = get_blocks_llama(model)

    l_layers_name = np.array(list(layers_to_prune.keys()))
    l_layers_to_prune_per_block = []
    acc_layer = 0
    for block in l_blocks:
        layers_block = find_layers(block)
        layers_to_prune_block = {}
        for key in layers_block:
            if layers_block[key] == layers_to_prune[l_layers_name[acc_layer]]:
                layers_to_prune_block[l_layers_name[acc_layer]] = layers_to_prune[l_layers_name[acc_layer]]
                acc_layer += 1
                acc_layer = min(acc_layer, len(l_layers_name)-1)
        l_layers_to_prune_per_block.append(layers_to_prune_block)

    return model, criterion, l_layers_to_prune_per_block, l_blocks

def get_blocks_llama(model):
    l_blocks = list(model.model.layers)
    if model.model.norm is not None:
        l_blocks+=[model.model.norm]
    l_blocks+=[model.lm_head]
    return l_blocks
