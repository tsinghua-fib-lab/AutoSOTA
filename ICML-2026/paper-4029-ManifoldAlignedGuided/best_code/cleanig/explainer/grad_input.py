"""
Reference:
    "Not just a black box: Learning important features through propagating activation differences", arXiv 2016
"""

import torch


class GradInputExplainer:
    def __init__(
        self,
        model,
        device='cuda',
        exp_obj='prob',
        preprocess_fn=None,
        baseline_method='zero',
    ):
        self.model = model
        self.device = device
        self.exp_obj = exp_obj
        self.baseline_method = baseline_method
        
        if preprocess_fn is not None:
            self.preprocess_fn = preprocess_fn
        else:
            self.preprocess_fn = lambda x: x

    def _get_gradients(self, x, labels=None):
        x = x.clone().detach().requires_grad_(True)
        
        output = self.model(x)
        if labels is None:
            labels = output.max(1, keepdim=False)[1]
        
        if self.exp_obj == 'logit':
            output = output[torch.arange(output.shape[0], device=self.device), labels]
        elif self.exp_obj == 'prob':
            output = torch.softmax(output, dim=-1)
            output = output[torch.arange(output.shape[0], device=self.device), labels]
        else:
            raise ValueError(f'Invalid objective function: {self.exp_obj}')
        
        grad = torch.autograd.grad(output.sum(), x)[0].detach()
        return grad

    def _get_baseline(self, x):
        if self.baseline_method == 'zero':
            return self.preprocess_fn(torch.zeros_like(x).float().to(self.device))
        else:
            raise ValueError(f'Invalid baseline method: {self.baseline_method}')

    def get_attributions(self, inputs, labels=None):
        inputs = inputs.to(self.device)
        baselines = self._get_baseline(inputs)
        
        grads = self._get_gradients(inputs, labels)
        diff = inputs - baselines
        
        attributions = grads * diff
        return attributions
