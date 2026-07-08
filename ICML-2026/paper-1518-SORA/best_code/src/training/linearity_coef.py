import torch
import torch.nn.functional as F

def calc_linearity_coef(input_grad, backprop_grad, method: str):
    """
    Calculate the linearity coefficient for SORA
    
    Args:
        input_grad (torch.Tensor): Input gradient.
        delta (torch.Tensor): Delta gradient.
        method (String): Varient of SORA to calculate the linearity coefficient
    """
    match method:
        case "Second Order Theory":
            norm2_input_grad = torch.linalg.norm(input_grad.view(input_grad.shape[0], -1), dim=1, ord=2)
            linearity_coef = (input_grad.view(input_grad.size(0), -1) * backprop_grad.view(backprop_grad.size(0), -1)).sum(dim=1) / torch.pow(norm2_input_grad, 2)
            return linearity_coef.mean().item()
        case "Second Order Theory Sign":
            p = torch.sign(input_grad)
            numerator = (p.view(p.size(0), -1) * backprop_grad.view(backprop_grad.size(0), -1)).sum(dim=1)
            denominator = input_grad.view(input_grad.size(0), -1).abs().sum(dim=1).clamp_min(1e-12)
            linearity_coef = (numerator / denominator).mean().item()
            return linearity_coef
        case _:
            return None
