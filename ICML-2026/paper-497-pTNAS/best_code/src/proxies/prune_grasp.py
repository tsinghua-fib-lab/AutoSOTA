from torch import autograd

from proxies.autograd_hacks import *
from proxies.p_utils import get_layer_metric_array
from proxies.task_adapter import proxy_forward, proxy_loss


class GraspEvaluator:

    def __init__(self):
        super().__init__()

    def evaluate(self, arch: nn.Module, device, batch_data: object, batch_labels: torch.Tensor, space_name: str) -> float:
        """
        This is implementation of paper
        "PICKING WINNING TICKETS BEFORE TRAINING BY PRESERVING GRADIENT FLOW"
        The score takes 5 steps:
            1. Run a forward & backward pass to calculate gradient of loss on weight, grad_w = d_loss/d_w
            2. Run  forward & backward again, this is to calculate  H*g, it uses hessian vector product to calculate it.
                eg, (d_loss / d_w)^2 * g = d_( (d_loss / d_w) * g) / d_w
                the code first calculate z = (d_loss / d_w) * g, and then calculate gradient on z with z.gradient()
            3. Then calculate Hg.0
            4. Sum up all weights' score and get the overall architecture score.
        """

        # alg cfgs
        T = 1
        num_iters = 1
        split_data = 1
        # get all applicable weights
        weights = []
        for layer in arch.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                weights.append(layer.weight)
                # TODO isn't this already true?
                layer.weight.requires_grad_(True)

        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            # forward/grad pass #1
            grad_w = None
            for _ in range(num_iters):
                outputs = proxy_forward(arch, batch_data) / T
                loss = proxy_loss(outputs, batch_labels, space_name)
                grad_w_p = autograd.grad(loss, weights, allow_unused=True)
                if grad_w is None:
                    grad_w = list(grad_w_p)
                else:
                    for idx in range(len(grad_w)):
                        grad_w[idx] += grad_w_p[idx]

            # forward/grad pass #2
            outputs = proxy_forward(arch, batch_data) / T
            loss = proxy_loss(outputs, batch_labels, space_name)
            grad_f = autograd.grad(loss, weights, create_graph=True, allow_unused=True)

        # accumulate gradients computed in previous step and call backwards
        z, count = 0, 0
        for layer in arch.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                if grad_w[count] is not None:
                    z += (grad_w[count].data * grad_f[count]).sum()
                count += 1
        z.backward()

        # compute final sensitivity metric and put in grads
        def grasp(layer):
            if layer.weight.grad is not None:
                return -layer.weight.data * layer.weight.grad  # -theta_q Hg
                # NOTE in the grasp code they take the *bottom* (1-p)% of values
                # but we take the *top* (1-p)%, therefore we remove the -ve sign
                # EDIT accuracy seems to be negatively correlated with this metric, so we add -ve sign here!
            else:
                return torch.zeros_like(layer.weight)

        grads = get_layer_metric_array(arch, grasp, "param")

        # Sum over all parameter's results to get the final score.
        score = 0.
        for i in range(len(grads)):
            score += grads[i].detach().cpu().sum().item()
        return score
