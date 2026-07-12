"""
Reference:
    Sundararajan et al., "Axiomatic Attribution for Deep Networks", ICML 2017
"""
import torch

from cleanig.explainer.path_utils import LinearPathGenerator


class IGExplainer:
    def __init__(self, model, baseline_method, num_steps, device, exp_obj='prob', preprocess_fn=None):
        self.model = model
        self.baseline_method = baseline_method
        self.num_steps = num_steps
        self.device = device
        self.exp_obj = exp_obj
        if preprocess_fn is not None:
            self.preprocess_fn = preprocess_fn
        else:
            self.preprocess_fn = lambda x: x

        self.path_generator = LinearPathGenerator(
            baseline_method=self.baseline_method,
            preprocess_fn=self.preprocess_fn,
            device=self.device,
            num_steps=self.num_steps,
        )

    def get_attributions(self, inputs, labels=None, return_paths=False):
        paths = self.path_generator.get_paths(inputs, labels)
        attributions = compute_ig(self.model, paths, labels, self.exp_obj)
        if return_paths:
            return attributions, paths
        else:
            return attributions    


def compute_ig(model, paths, labels=None, exp_obj='prob'):
    """
    Compute attributions along paths.
        
    Args:
        model: The model to compute gradients for
        paths: Tensor of shape [B, num_steps, C, H, W] representing the path
        labels: Target labels [B]
        exp_obj: Objective function ('prob' or 'logit')
    
    Returns:
        attributions: Attribution maps [B, C, H, W]
    """    
    # Get gradients at each point along the path
    # Also get model outputs for IDGI direction decomposition
    grads, outputs = get_grads(model, paths, labels, exp_obj, return_outputs=True)
    
    # IDGI (Important Direction Gradient Integration):
    # Instead of scalar projection deltas * grads, use direction-aware:
    # att = SUM(grads_left^2 * delta_f / ||grads_left||^2)
    # This eliminates gradient-orthogonal noise while maintaining completeness.
    
    # Compute delta_f = f(x_{t+1}) - f(x_t) for each step [B, num_steps-1]
    delta_f = outputs[:, 1:] - outputs[:, :-1]  # [B, num_steps-1]
    
    # Gradient at left point of each step [B, num_steps-1, C, H, W]
    grads_left = grads[:, :-1]  # [B, num_steps-1, C, H, W]
    
    # Gradient squared norm per step [B, num_steps-1]
    grad_sq_norm = (grads_left * grads_left).sum(dim=(2, 3, 4))  # [B, num_steps-1]
    
    # Avoid division by zero
    grad_sq_norm = torch.clamp(grad_sq_norm, min=1e-12)
    
    # IDGI attribution: grads * grads * delta_f / ||grads||^2
    weight = delta_f / grad_sq_norm  # [B, num_steps-1]
    weight = weight[:, :, None, None, None]  # [B, num_steps-1, 1, 1, 1]
    
    idgi_steps = grads_left * grads_left * weight  # [B, num_steps-1, C, H, W]
    attributions = idgi_steps.sum(dim=1)  # [B, C, H, W]
    
    return attributions


def get_grads(model, paths, labels=None, exp_obj='prob', return_outputs=False):
    """Original implementation - processes each step separately.
    When return_outputs=True, also returns model output values at each step
    for use with IDGI (Important Direction Gradient Integration)."""
    device = paths.device

    grads = torch.zeros(paths.shape).float().to(device)
    if return_outputs:
        outputs = torch.zeros(paths.shape[0], paths.shape[1]).float().to(device)

    for i in range(paths.shape[1]):
        particular_slice = paths[:, i]
        particular_slice.requires_grad = True

        output = model(particular_slice)
        if labels is None:
            labels = output.max(1, keepdim=False)[1]

        if exp_obj == 'logit':
            output = output[torch.arange(output.shape[0]), labels]   
        elif exp_obj == 'prob':
            output = torch.softmax(output, dim=-1)
            output = output[torch.arange(output.shape[0]), labels]
        else:
            raise ValueError(f'Invalid objective function: {exp_obj}')

        grad = torch.autograd.grad(output.sum(), particular_slice)[0].detach()

        grads[:, i, :] = grad
        if return_outputs:
            outputs[:, i] = output.detach()

    if return_outputs:
        return grads, outputs
    return grads
