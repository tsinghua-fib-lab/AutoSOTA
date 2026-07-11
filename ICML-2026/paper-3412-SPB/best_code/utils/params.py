import torch

def get_flat_params(model):
    return torch.cat([p.detach().view(-1) for p in model.parameters()])

def set_flat_params(model, flat_params):
    """
    Copy a flat parameter vector into the model's parameters (in-place).
    """
    pointer = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_params[pointer:pointer + numel].view_as(p))
        pointer += numel

    assert pointer == len(flat_params), "Mismatch in parameter vector size"