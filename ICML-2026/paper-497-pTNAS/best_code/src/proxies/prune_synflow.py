import torch
from torch import nn

from proxies.p_utils import get_layer_metric_array
from proxies.task_adapter import proxy_forward, synflow_input
from common.constant import Config


class SynFlowEvaluator:

    def __init__(self):
        super().__init__()

    def evaluate(self, arch: nn.Module, device, batch_data: object, batch_labels: torch.Tensor,
                 space_name: str) -> float:
        """
        This is implementation of paper
        "Pruning neural networks without any data by iteratively conserving synaptic flow"
        The score takes 5 steps:
            1. For each layer, for each parameter, calculate the absolute value |0|
            2. Use a single all-one-vector with dim = [1, c, h, w] to run a forward,
               Since only consider linear and Con2d operation, the forward output is multiple( [ |0l| for l in L] )
            3. New loss function R = sum(output), and then run backward
            4. for each layer, calculate Sl = Hadamard product( df/dw, w), where Sij=aij×bij
            5. score = sum( [ Sl for l in layers ] )
        Comments:
            1. this is data-Agnostic
            2. only compute on a single example
        """

        # 1. Convert params to their abs. Record sign for converting it back.
        @torch.no_grad()
        def linearize(arch):
            signs = {}
            for name, param in arch.state_dict().items():
                signs[name] = torch.sign(param)
                param.abs_()
            return signs

        # convert to orig values with sign
        @torch.no_grad()
        def nonlinearize(arch, signs):
            for name, param in arch.state_dict().items():
                if 'weight_mask' not in name:
                    param.mul_(signs[name])

        was_training = arch.training
        arch.eval()

        try:
            # Step 1: Linearize
            if space_name == Config.MLPSP:
                signs = linearize(arch.mlp)
                arch.mlp.double()
            else:
                signs = linearize(arch)
                arch.double()

            flow_input = synflow_input(batch_data).to(next(arch.parameters()).device).double()

            if space_name == Config.MLPSP:
                output = arch.forward_wo_embedding(flow_input)
            else:
                output = proxy_forward(arch, flow_input)

            # 3.R = sum(output)
            torch.sum(output).backward()

            # 4. Select the gradients that we want to use for search/prune
            def synflow(layer):
                if layer.weight.grad is not None:
                    return torch.abs(layer.weight * layer.weight.grad)
                else:
                    return torch.zeros_like(layer.weight)

            grads_abs = get_layer_metric_array(arch, synflow, "param")

            # 5. Sum over all parameter's results to get the final score.
            score = sum([grad.sum() for grad in grads_abs])
            return float(score)
        finally:
            if space_name == Config.MLPSP:
                nonlinearize(arch.mlp, signs)
                arch.mlp = arch.mlp.float()
            else:
                nonlinearize(arch, signs)
                arch.float()
            arch.train(was_training)
