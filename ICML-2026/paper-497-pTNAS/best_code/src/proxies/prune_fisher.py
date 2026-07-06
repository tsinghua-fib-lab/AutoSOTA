  
from proxies.autograd_hacks import *
 
from proxies.p_utils import get_layer_metric_array, reshape_elements
from proxies.task_adapter import proxy_forward, proxy_loss

import types


class FisherEvaluator:

    def __init__(self):
        super().__init__()

    def evaluate(self, arch: nn.Module, device, batch_data: object, batch_labels: torch.Tensor, space_name: str) -> float:
        """
        This is implementation of paper
        "Faster gaze prediction with dense archworks and fisher pruning"
        The score takes 5 steps:
            1. Run a forward & backward pass to calculate gradient of loss on weight, grad_w = d_loss/d_w
            2. Then calculate norm for each gradient, grad.norm(p), default p = 2
            3. Sum up all weights' norm and get the overall architecture score.
        """

        # alg cfgs
        # 'Fisher pruning does not support parameter pruning.'
        mode = "channel"

        # update model's forward and backward function
        self._update_module_compute(arch)

        # forward & backward
        outputs = proxy_forward(arch, batch_data)
        loss = proxy_loss(outputs, batch_labels, space_name)
        loss.backward()

        # retrieve fisher info
        def fisher(layer):
            if layer.fisher is not None:
                return torch.abs(layer.fisher.detach())
            else:
                return torch.zeros(layer.weight.shape[0])  # size=ch

        grads_abs_ch = get_layer_metric_array(arch, fisher, mode)

        # broadcast channel value here to all parameters in that channel
        # to be compatible with stuff downstream (which expects per-parameter metrics)
        # TODO cleanup on the selectors/apply_prune_mask side (?)
        shapes = get_layer_metric_array(arch, lambda l: l.weight.shape[1:], mode)

        grads_abs = reshape_elements(grads_abs_ch, shapes, device)

        # Sum over all parameter's results to get the final score.
        score = 0.
        for i in range(len(grads_abs)):
            score += grads_abs[i].detach().cpu().sum().item()
        return score

    def _update_module_compute(self, arch):
        """
        override the computation, where self is the layer,
        :param arch: architecture
        :return: None
        """

        def fisher_forward_conv2d(self, x):
            x = F.conv2d(x, self.weight, self.bias, self.stride,
                         self.padding, self.dilation, self.groups)
            # intercept and store the activations after passing through 'hooked' identity op
            self.act = self.dummy(x)
            return self.act

        def fisher_forward_linear(self, x):
            x = F.linear(x, self.weight, self.bias)
            self.act = self.dummy(x)
            return self.act

        for layer in arch.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                # variables/op needed for fisher computation
                layer.fisher = None
                layer.act = 0.
                layer.dummy = nn.Identity()

                # replace forward method of conv/linear
                if isinstance(layer, nn.Conv2d):
                    layer.forward = types.MethodType(fisher_forward_conv2d, layer)
                if isinstance(layer, nn.Linear):
                    layer.forward = types.MethodType(fisher_forward_linear, layer)

                # function to call during backward pass (hooked on identity op at output of layer)
                def hook_factory(layer):
                    def hook(module, grad_input, grad_output):
                        act = layer.act.detach()
                        grad = grad_output[0].detach()
                        if len(act.shape) > 2:
                            g_nk = torch.sum((act * grad), list(range(2, len(act.shape))))
                        else:
                            g_nk = act * grad
                        del_k = g_nk.pow(2).mean(0).mul(0.5)
                        if layer.fisher is None:
                            layer.fisher = del_k
                        else:
                            layer.fisher += del_k
                        del layer.act  # without deleting this, a nasty memory leak occurs! related: https://discuss.pytorch.org/t/memory-leak-when-using-forward-hook-and-backward-hook-simultaneously/27555

                    return hook

                # register backward hook on identity fcn to compute fisher info
                layer.dummy.register_full_backward_hook(hook_factory(layer))
                # layer.dummy.register_backward_hook(hook_factory(layer))
