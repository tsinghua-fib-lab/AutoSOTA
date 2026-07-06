  
 
from proxies.autograd_hacks import *
from proxies.p_utils import get_layer_metric_array
from proxies.task_adapter import proxy_forward, proxy_loss

import types
import numpy as np


class SnipEvaluator:

    def __init__(self):
        super().__init__()

    def evaluate(self, arch: nn.Module, device, batch_data: object, batch_labels: torch.Tensor, space_name: str) -> float:
        """
        This is implementation of paper
        "SNIP: SINGLE -SHOT NETWORK PRUNING BASED ON CONNECTION SENSITIVITY"
        """

        # update module's forward and backward function
        self._update_module_compute(arch)

        # Compute gradients (but don't apply them)
        # run a forward + backward on mini-batch， spit data if it cannot fit into one GPU

        outputs = proxy_forward(arch, batch_data)
        loss = proxy_loss(outputs, batch_labels, space_name)
        loss.backward()

        # select the gradients that we want to use for search/prune
        def snip(layer):
            if layer.weight_mask.grad is not None:
                return torch.abs(layer.weight_mask.grad)
            else:
                return torch.zeros_like(layer.weight)

        grads_abs = get_layer_metric_array(arch, snip, "param")

        # Sum over all parameter's results to get the final score.
        score = 0.
        for i in range(len(grads_abs)):
            score += grads_abs[i].detach().cpu().sum().item()
        return score

    def _update_module_compute(self, arch):
        # override the computation, where self is the layer,
        # assign a master vector to the weight,
        def snip_forward_conv2d(self, x):
            return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                            self.stride, self.padding, self.dilation, self.groups)

        def snip_forward_linear(self, x):
            return F.linear(x, self.weight * self.weight_mask, self.bias)

        # for each layer, do the replacement,
        for layer in arch.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
                layer.weight.requires_grad = False

            # Override the forward methods:
            if isinstance(layer, nn.Conv2d):
                layer.forward = types.MethodType(snip_forward_conv2d, layer)

            if isinstance(layer, nn.Linear):
                layer.forward = types.MethodType(snip_forward_linear, layer)
        return
