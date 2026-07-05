import numpy as np
import torch
from torch import nn

from common.constant import Config
from proxies.task_adapter import activation_module_types, proxy_batch_size, proxy_forward


class NWTEvaluator:

    def __init__(self):
        super().__init__()

    def evaluate(self, arch: nn.Module, device, batch_data: object, batch_labels: torch.Tensor,
                 space_name: str) -> float:
        """
        This is implementation of paper "Neural Architecture Search without Training"
        The score takes 5 steps:
            1. for ech example, get the binary vector for each relu layer, where 1 means x > 0, 0 otherwise,
            2. calculate K = [Na - hamming_distance (ci, cj) for each ci, cj]
        """

        handles = []

        def counting_forward_hook(module, inp, out):
            if isinstance(inp, tuple):
                inp = inp[0]
            inp = inp.reshape(inp.size(0), -1)
            x = (inp > 0).float()
            K = x @ x.t()
            K2 = (1. - x) @ (1. - x.t())
            arch.K = arch.K + K.cpu().numpy() + K2.cpu().numpy()

        batch_size = proxy_batch_size(batch_data)
        arch.K = np.zeros((batch_size, batch_size), dtype=np.float64)

        try:
            for module in arch.modules():
                if isinstance(module, activation_module_types()):
                    handles.append(module.register_forward_hook(counting_forward_hook))

            proxy_forward(arch, batch_data)
            arch.K = arch.K + np.eye(batch_size, dtype=np.float64) * 1e-6
            _, ld = np.linalg.slogdet(arch.K)
            return float(ld)
        finally:
            for handle in handles:
                handle.remove()

    def get_batch_jacobian(self, arch, x, target):
        arch.zero_grad()
        x.requires_grad_(True)
        y = arch(x)
        y.backward(torch.ones_like(y))
        jacob = x.grad.detach()
        return jacob, target.detach(), y.detach()
