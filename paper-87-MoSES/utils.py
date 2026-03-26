import sys

from torch.utils.data import DataLoader
from rl4co.data.dataset import TensorDictDataset
from typing import Dict, Any, Union
import os
import torch

def get_dataloader(td, batch_size=4):
    """Get a dataloader from a TensorDictDataset"""
    # Set up the dataloader
    dataloader = DataLoader(
        TensorDictDataset(td.clone()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=TensorDictDataset.collate_fn,
    )
    return dataloader


def collect_lora_state_dict(ckpt_path=None, policy_weights=None) -> Union[Dict, Any]:
    if ckpt_path != None:
        assert os.path.exists(ckpt_path)
        policy_weights = torch.load(ckpt_path, map_location=torch.device("cpu"), weights_only=False)['state_dict']
    else:
        assert policy_weights != None

    lora_state_dict = {}
    for key, value in policy_weights.items():
        if 'lora_layer' in key.split('.'):
            lora_state_dict[key.lstrip('policy.')] = value

    return lora_state_dict


def collect_multi_lora_state_dict(multi_lora_fixed_params=None, lora_n_experts=None) -> Union[Dict, Any]:
    assert multi_lora_fixed_params != None
    assert lora_n_experts != None

    multi_lora_state_dict = {i: dict() for i in range(lora_n_experts)}
    for i, (key, value) in enumerate(multi_lora_fixed_params.items()):
        key_ls = key.split('.')
        assert ('lora_layers' in key_ls) and ('lora_layer' in key_ls)
        assert key_ls.index('lora_layers') - 1 == key_ls.index('lora_layer')

        lora_num_idx = key_ls.index('lora_layers') + 1
        lora_idx = int(key_ls[lora_num_idx])

        sub_key = ".".join(key_ls[:lora_num_idx - 1] + key_ls[lora_num_idx + 1:])
        multi_lora_state_dict[lora_idx][sub_key] = value

    assert len(multi_lora_state_dict) == lora_n_experts
    return multi_lora_state_dict


def load_target_lora_module(source_lora_state_dict=None, target_lora_state_dict=None):
    assert target_lora_state_dict.keys() == source_lora_state_dict.keys()
    for key, value in target_lora_state_dict.items():
        value.copy_(source_lora_state_dict[key])

