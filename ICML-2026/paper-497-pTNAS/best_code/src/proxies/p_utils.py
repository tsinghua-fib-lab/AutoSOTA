

from collections.abc import Callable
import torch
import torch.nn as nn


def get_layer_metric_array(net, metric: Callable, mode: str):
    """
    1. Recursively traverse on each layer of the module,
    2. apply callback function metric on it.
    3. And then return a list of metric

    :param net: architecture
    :param metric: callback function
    :param mode: "channel" or "param"
    :return: A list of score, each corresponding to a weight.
    """
    metric_array = []
    for layer in net.modules():
        if mode == 'channel' and hasattr(layer, 'dont_ch_prune'):
            continue
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            metric_array.append(metric(layer))
    
    return metric_array


def reshape_elements(elements, shapes, device):
    def broadcast_val(elements, shapes):
        ret_grads = []
        for e, sh in zip(elements, shapes):
            ret_grads.append(torch.stack([torch.Tensor(sh).fill_(v) for v in e], dim=0).to(device))
        return ret_grads

    if type(elements[0]) == list:
        outer = []
        for e, sh in zip(elements, shapes):
            outer.append(broadcast_val(e, sh))
        return outer
    else:
        return broadcast_val(elements, shapes)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

