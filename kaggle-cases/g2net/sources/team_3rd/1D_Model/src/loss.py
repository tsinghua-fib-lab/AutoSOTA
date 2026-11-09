import torch
import torch.nn.functional as F


def rank_loss(input, target):
    input = input.view(-1).float()
    target = target.view(-1)
    p = input[target == 1]
    n = input[target == 0]
    if len(p) == 0: p = torch.Tensor([1]).to(input.device)
    if len(n) == 0: n = torch.Tensor([-1]).to(input.device)
    x = p[:, None] - n[None, :]
    return -F.logsigmoid(x).mean() + 1e-4 * (input ** 2).mean()
