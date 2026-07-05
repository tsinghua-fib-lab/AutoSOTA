import torch


def constraint(device, prompt):
    if isinstance(prompt, list):
        sum = 0
        for p in prompt:
            sum = sum + torch.norm(torch.mm(p, p.T) - torch.eye(p.shape[0]).to(device))
        return sum / len(prompt)
    else:
        return torch.norm(torch.mm(prompt, prompt.T) - torch.eye(prompt.shape[0]).to(device))
