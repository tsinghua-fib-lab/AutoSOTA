import torch
import torch.nn.functional as F


def act(x=None, act_type='leakyrelu'):
    if act_type == 'leakyrelu':
        return torch.nn.LeakyReLU() if x is None else F.leaky_relu(x)
    elif act_type == 'tanh':
        return torch.nn.Tanh() if x is None else torch.tanh(x)
    elif act_type == 'relu':
        return torch.nn.ReLU() if x is None else F.relu(x)
    elif act_type == 'sigmoid':
        return torch.nn.Sigmoid() if x is None else torch.sigmoid(x)
    elif act_type == 'softmax':
        # Note: softmax requires specifying dimensions; assume that softmax is applied to the last dimension
        return torch.nn.Softmax(dim=-1) if x is None else F.softmax(x, dim=-1)
    else:
        raise ValueError(f"Unsupported activation type: {act_type}")

# exmaple
# act_type = 'relu', 'sigmoid', 'softmax', 'tanh', 'leakyrelu'
# x = torch.tensor(...)
# result = act(x, act_type='relu')
