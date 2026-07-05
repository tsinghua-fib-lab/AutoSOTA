import time

import numpy as np

from proxies.autograd_hacks import *
from proxies.task_adapter import proxy_batch_size, proxy_forward, proxy_loss

from common.constant import Config


class NTKTraceEvaluator:

    def __init__(self):
        super().__init__()

    def evaluate(self, arch: nn.Module, device, batch_data: object, batch_labels: torch.Tensor,
                 space_name: str) -> float:
        """
        This is implementation of paper
        "NASI: Label- and Data-agnostic Neural Architecture Search at Initialization"
        The score takes 5 steps:
            1. run forward on a mini-batch
            2. output = sum( [ yi for i in mini-batch N ] ) and then run backward
            3. explicitly calculate gradient of f on each example, df/dxi,
                grads = [ df/ dxi for xi in [1, ..., N] ], dim = [N, number of parameters]
            4. calculate NTK = grads * grads_t
            5. calculate M_trace = traceNorm(NTK), score = np.sqrt(trace_norm / batch_size)
        """

        batch_size = proxy_batch_size(batch_data)

        add_hooks(arch)

        # 1. forward on mini-batch
        outputs = proxy_forward(arch, batch_data)

        # 2. run backward
        loss = proxy_loss(outputs, batch_labels, space_name, reduction='sum')
        loss.backward()

        # 3. calculate gradient for each sample in the batch
        # grads = ∇0 f(X), it is N*P , N is number of sample, P is number of parameters,
        compute_grad1(arch, loss_type='sum')

        grads = [param.grad1.flatten(start_dim=1) for param in arch.parameters() if hasattr(param, 'grad1')]

        # remove those in GPU
        clear_backprops(arch)

        # print("gradient calculated done, delete arch, begin to compute NTK")

        # 4. ntk = ∇0 f(X) * Transpose( ∇0 f(X) ) [ batch_size * batch_size ]
        begin = time.time()
        grads_final = torch.zeros(batch_size, batch_size).to(device)
        for ele in grads:
            grads_final += torch.matmul(ele, ele.t())
        end = time.time()

        ntk = grads_final.detach()
        del grads
        del grads_final
        torch.cuda.empty_cache()

        # 5. calculate M_trace = sqrt ( |ntk|_tr * 1/m )

        # For a Hermitian matrix, like a density matrix,
        # the absolute value of the eigenvalues are exactly the singular values,
        # so the trace norm is the sum of the absolute value of the eigenvalues of the density matrix.
        # eigenvalues, _ = torch.symeig(ntk)  # ascending
        eigenvalues, _ = torch.linalg.eigh(ntk)

        trace_norm = eigenvalues.cpu().sum().item()
        score = trace_norm / batch_size
        score = score ** 0.5 if score >= 0 else -((-score) ** 0.5)

        del eigenvalues
        del ntk
        torch.cuda.empty_cache()
        remove_hooks(arch)
        return float(score)
