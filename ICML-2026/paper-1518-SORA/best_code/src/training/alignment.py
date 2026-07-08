import torch.nn.functional as F

def calc_alignment(input_grad, backprop_grad):
    """
    Calculate PertAlign between the input gradient and the delta gradient.
    
    Args:
        input_grad (torch.Tensor): Input gradient.
        delta (torch.Tensor): Delta gradient.
    """
    return F.cosine_similarity(backprop_grad.view(backprop_grad.shape[0], -1), input_grad.view(input_grad.shape[0], -1)).mean().item()
