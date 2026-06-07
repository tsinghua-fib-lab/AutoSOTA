import torch
import random
import numpy as np
from collections.abc import Iterable
from fgl.flcore.FedSDR.server import FedSDRServer


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def load_client(args, client_id, data, data_dir, message_pool, device):
    from fgl.flcore.FedSDR.client import FedSDRClient
    return FedSDRClient(args, client_id, data, data_dir, message_pool, device)


def load_server(args, global_data, data_dir, message_pool, device):
    return FedSDRServer(args, global_data, data_dir, message_pool, device)


def load_optim(args):
    if args.optim == "adam":
        from torch.optim import Adam
        return Adam


def load_task(args, client_id, data, data_dir, device):
    from fgl.task.node_cls import NodeClsTask
    return NodeClsTask(args, client_id, data, data_dir, device)


def extract_floats(s):
    from decimal import Decimal
    parts = s.split('-')
    train = float(parts[0])
    val = float(parts[1])
    test = float(parts[2])
    assert Decimal(parts[0]) + Decimal(parts[1]) + \
        Decimal(parts[2]) == Decimal(1)
    return train, val, test


def idx_to_mask_tensor(idx_list, length):
    mask = torch.zeros(length)
    mask[idx_list] = 1
    return mask


def mask_tensor_to_idx(tensor):
    result = tensor.nonzero().squeeze().tolist()
    if type(result) is not list:
        result = [result]
    return result


def total_size(o):
    size = 0
    if isinstance(o, torch.Tensor):
        size += o.element_size() * o.numel()
    elif isinstance(o, dict):
        size += sum(total_size(v) for v in o.values())
    elif isinstance(o, Iterable):
        size += sum(total_size(i) for i in o)
    return size
