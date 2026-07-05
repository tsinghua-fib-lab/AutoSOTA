from proxies.autograd_hacks import *
from proxies.p_utils import get_layer_metric_array
from torch import nn


class GradPlainEvaluator:

    def __init__(self):
        super().__init__()

    def evaluate(self, arch: nn.Module, device, batch_data: object, batch_labels: torch.Tensor,
                 space_name: str) -> float:
        """
        The score takes 3 steps:
            1. Run a forward & backward pass to calculate gradient of loss on weight, grad_w = d_loss/d_w
            2. Then calculate gradient, grad
            3. Sum up all weights' grad and get the overall architecture score.
        """

        loss_fn = F.cross_entropy
        outputs = arch(batch_data)
        loss = loss_fn(outputs, batch_labels)
        loss.backward()

        # select the gradients that we want to use for search/prune
        def plain(layer):
            if layer.weight.grad is not None:
                return layer.weight.grad * layer.weight
            else:
                return torch.zeros_like(layer.weight)

        grads_abs = get_layer_metric_array(arch, plain, "param")

        # Sum over all parameter's results to get the final score.
        score = 0.
        for i in range(len(grads_abs)):
            score += grads_abs[i].detach().cpu().sum().item()
        return score
