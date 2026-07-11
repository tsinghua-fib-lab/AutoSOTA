import torch
import random
import numpy as np
import os
import torch.nn.functional as F

def get_logging_function(output_dir, file_name="logs.txt"):
    log_path = output_dir / file_name
    f = open(log_path, "w")

    def print_to_both(*args):
        print(*args)
        print(*args, file=f, flush=True)

    def cleanup():
        f.close()

    print_to_both.cleanup = cleanup
    return print_to_both

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def restore_int_keys(obj):
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            # try to convert key back to int if possible
            try:
                new_key = int(k)
            except ValueError:
                new_key = k
            new_dict[new_key] = restore_int_keys(v)
        return new_dict
    elif isinstance(obj, list):
        return [restore_int_keys(item) for item in obj]
    else:
        return obj


def get_mlp_primitives(x, primitive):
    shape = x.size()[:-1]
    if x.dim() == 1:
        x = x.unsqueeze(0)
    elif x.dim() == 2:
        pass
    elif x.dim() == 3:
        x = x.flatten(end_dim=1)
    else:
        raise RuntimeError
    
    match primitive:
        case "no_op":
            pass
        case ("sharpen", n):
            x = x.pow(n)
            x /= x.sum(dim=-1, keepdim=True)
        case "erase":
            x = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
        case "harden":
            x = F.one_hot(x.argmax(dim=-1), num_classes=x.size(-1)).float()
        case ("exists", idx):   # can try different threshold
            x = (x[:, idx] > 0.1).float()
            # x = x[:, idx].pow(1/3)
            x = torch.stack([x, 1-x], dim=1)
        case ("forall", threshold):   # expressivity > individual ones, e.g. x[:, idx]> threshold
            x = (x > threshold).float()
            x = torch.cat([x, 1-x.sum(dim=1, keepdim=True)], dim=1)
        case ("equal", indices):
            x = x[:, indices]
            x = ((x.max(dim=1)[0] - x.min(dim=1)[0]) < 0.01).float()
            x = torch.stack([x, 1-x], dim=1)
        case ("01balance", pow, center):   # 1/2..1/4
            # print("mean", (x[:, 1] - x[:, 0]).mean(), x.size())
            x = x[:, 1] - x[:, 0] - center
            x = torch.stack([x, -x], dim=1).clamp(min=0).pow(pow)
            x = torch.cat([x, 1-x.sum(dim=1, keepdim=True)], dim=1)
        case ("ABbalance", pow, center):   # 1/2..1/4
            # print("mean", (x[:, 1] - x[:, 0]).mean(), x.size())
            x = x[:, 3] - x[:, 2] - center
            x = torch.stack([x, -x], dim=1).clamp(min=0).pow(pow)
            x = torch.cat([x, 1-x.sum(dim=1, keepdim=True)], dim=1)
        case ("diff", idx1, idx2):
            num_bins = 11
            edges = torch.linspace(-0.6, 0.6, num_bins-1).tolist()
            edges.insert(0, -1.1)
            edges.append(1.1)
            edges = torch.tensor(edges, device=x.device)
            diff = x[:, idx1] - x[:, idx2]
            bos = x[:, [idx1, idx2]].sum(dim=1) < 0.2

            diff[bos] = -100

            x = (diff.unsqueeze(1) < edges[1:].unsqueeze(0)) & (diff.unsqueeze(1) >= edges[:-1].unsqueeze(0))
            x = torch.cat([x, bos.unsqueeze(1)], dim=1)
            x = x.float()
            assert (x.sum(dim=1) == 1).all().item(), diff[(x.sum(dim=1) == 1)]

        case _:
            raise NotImplementedError(primitive, "is not implemented")
    
    return x.view(*shape, x.size(-1))

def get_mlp_primitives_multi_source(x: list[torch.FloatTensor], primitive):
    device, dtype = x[0].device, x[0].dtype
    shape = x[0].size()[:-1]
    if x[0].dim() == 1:
        x = [item.unsqueeze(0) for item in x]
    elif x[0].dim() == 2:
        pass
    elif x[0].dim() == 3:
        x = [item.flatten(end_dim=1) for item in x]
    else:
        raise RuntimeError
    
    match primitive:
        case "erase":
            x = torch.ones(x[0].size(0), 1, device=device, dtype=dtype)
        case ("keep_one", n):
            x = x[n]
        case "combine":
            combined = x[0]
            for item in x[1:]:
                if combined.size(1) * item.size(1) > 10000:
                    combined = torch.ones(combined.size(0), 10001, device=combined.device)
                    break   # will be stopped later
                combined = torch.min(combined.unsqueeze(-1), item.unsqueeze(1)).flatten(start_dim=1)

            x = combined / combined.sum(dim=-1, keepdim=True)  
            # x1, x2 = 00, 01, 02...
            # x1, x2, x3 = 000, 001, 002, ..., 010, 011, 012...

        case _:
            raise NotImplementedError(primitive, "is not implemented")
    
    return x.view(*shape, x.size(-1))
