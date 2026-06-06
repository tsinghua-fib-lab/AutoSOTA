import torch
import torch.nn as nn
from typing import Tuple, Union
from resnet import BinaryConv, TernaryConv
from modules import OnlineNeuron, LIFNeuron

def unpack_for_conv(x: Union[Tuple[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    if isinstance(x, tuple):
        assert x.__len__() == 1
        x = x[0]
    if len(x.shape) == 5:
        return x.flatten(0, 1)
    return x

class BaseMonitor:
    def __init__(self):
        self.hooks = []
        self.monitored_layers = []
        self.records = []
        self.synaptic_name_records_index = {}
        self.synaptic_fr_name_records_index = {}
        self.neural_name_records_index = {}
        self._enable = True

    def __getitem__(self, i):
        if isinstance(i, int):
            return self.records[i]
        elif isinstance(i, str):
            sops, nops, tot_ops = [], [], []
            if i in self.synaptic_name_records_index:
                for index in self.synaptic_name_records_index[i]:
                    sops.append(self.records[index])
            if i in self.synaptic_fr_name_records_index:
                for index in self.synaptic_fr_name_records_index[i]:
                    tot_ops.append(self.records[index])
            if i in self.neural_name_records_index:
                for index in self.neural_name_records_index[i]:
                    nops.append(self.records[index])
            return sops, nops, tot_ops
        else:
            raise ValueError(i)

    def clear_recorded_data(self):
        self.records.clear()
        for k, v in self.synaptic_name_records_index.items():
            v.clear()
        for k, v in self.synaptic_fr_name_records_index.items():
            v.clear()
        for k, v in self.neural_name_records_index.items():
            v.clear()

    def enable(self):
        self._enable = True

    def disable(self):
        self._enable = False

    def is_enable(self):
        return self._enable

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def __del__(self):
        self.remove_hooks()


class SOPMonitor(BaseMonitor):
    def __init__(self, net: nn.Module):
        super().__init__()
        for name, m in net.named_modules():
            if name in net.skip_name:
                continue
            if isinstance(m, (nn.Conv2d, BinaryConv, TernaryConv)):
                self.monitored_layers.append(name)
                self.synaptic_name_records_index[name] = []
                self.synaptic_fr_name_records_index[name] = []
                self.hooks.append(m.register_forward_hook(
                    self.create_hook_conv(name)))
            elif isinstance(m, (OnlineNeuron, LIFNeuron)):
                self.monitored_layers.append(name)
                self.neural_name_records_index[name] = []
                self.hooks.append(m.register_forward_hook(
                    self.create_hook_neuron(name)))

    def cal_sop_conv(self, x, m):
        with torch.no_grad():
            x_mask = (x.abs() > 0).float()
            weight_mask = (m.weight.abs() > 0).float()
            out = torch.nn.functional.conv2d(x_mask, weight_mask, None, m.stride, m.padding, 1, 1)
            tot_out = torch.nn.functional.conv2d(torch.ones_like(x), torch.ones_like(m.weight), None, m.stride, m.padding, 1, 1)
            return (out.sum().item() / (1000**2), tot_out.sum().item() / (1000**2)), (x_mask.sum().item() / (1000**2), x_mask.numel() / (1000**2))
    
    def cal_sop_neuron(self, x, m):
        if hasattr(m, 'parallel_mode') and m.parallel_mode is True:
            return x.numel() / (1000**2)
        else:
            return 3 * x.numel() / (1000**2)

    def create_hook_conv(self, name):
        def hook(m, x, y):
            if self.is_enable():
                (sops, tot_sops), (fire_spikes, tot_spikes) = self.cal_sop_conv(unpack_for_conv(x).detach(), m)
                self.synaptic_name_records_index[name].append(self.records.__len__())
                self.records.append((sops, tot_sops))
                self.synaptic_fr_name_records_index[name].append(self.records.__len__())
                self.records.append((fire_spikes, tot_spikes))
        return hook

    def create_hook_neuron(self, name):
        def hook(m, x, y):
            if self.is_enable():
                self.neural_name_records_index[name].append(self.records.__len__())
                self.records.append(self.cal_sop_neuron(unpack_for_conv(x).detach(), m))
        return hook

