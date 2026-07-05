import numpy as np

from proxies.autograd_hacks import *
from proxies.task_adapter import proxy_batch_size, proxy_forward, proxy_loss
from torch import nn

from common.constant import Config


class NTKCondNumEvaluator:

    def __init__(self):
        super().__init__()

    def evaluate(self, arch: nn.Module, device, batch_data: object, batch_labels: torch.Tensor,
                 space_name: str) -> float:
        """
        This is implementation of paper TE-NAS,
        "NEURAL ARCHITECTURE SEARCH ON IMAGENET IN FOUR GPU HOURS: A THEORETICALLY INSPIRED PERSPECTIVE"
        The implementation only consider K in paper.
        The score takes 5 steps:
            1. run forward on a mini-batch
            2. output = sum( [ yi for i in mini-batch N ] ) and then run backward
            3. explicitly calculate gradient of f on each example, df/dxi,
                grads = [ df/ dxi for xi in [1, ..., N] ], dim = [N, number of parameters]
            4. calculate NTK = grads * grads_t
            5. calculate score = 1/K = eigenvalues_max / eigenvalues_min
        """

        import time

        batch_size = proxy_batch_size(batch_data)

        # print("\n3. ----------------- Begin to evaluate NTK condNum -----------------\n")
        # print(showUtilization()[0])

        begin = time.time()
        add_hooks(arch)
        end = time.time()
        # print("==== add_hooks: " + str(end-begin))

        # 1. forward on mini-batch
        begin = time.time()
        outputs = proxy_forward(arch, batch_data)
        end = time.time()
        # print("4. forward done")
        # print("==== forward: " + str(end-begin))
        # print(showUtilization()[0])

        # 2. run backward
        begin = time.time()
        loss = proxy_loss(outputs, batch_labels, space_name, reduction='sum')
        loss.backward()
        end = time.time()
        # print("5. backward done")
        # print("==== backward: " + str(end-begin))
        # print(showUtilization()[0])

        # 3. calculate gradient for each sample in the batch
        begin = time.time()
        compute_grad1(arch, loss_type='sum')
        origin_grads = [param.grad1.flatten(start_dim=1) for param in arch.parameters() if hasattr(param, 'grad1')]
        end = time.time()
        # print("6. compute_grad1 done")
        # print("==== compute_grad1: " + str(end-begin))
        # print(showUtilization()[0])

        clear_backprops(arch)
        # print("gradient calculated done, delete arch, begin to compute NTK")

        # 4. ntk = ∇0 f(X) * Transpose( ∇0 f(X) ) [ batch_size * batch_size ]
        begin = time.time()
        grads_final = torch.zeros(batch_size, batch_size).to(device)
        for ele in origin_grads:
            grads_final += torch.matmul(ele, ele.t())
        end = time.time()

        ntk = grads_final.detach()
        del origin_grads
        del grads_final
        torch.cuda.empty_cache()

        # print("9. calculate ntk done")
        # print("==== calculate ntk: " + str(end-begin))
        # print(showUtilization()[0])

        # 5. sort eigenvalues and then calculate k = lambda_0 / lambda_m
        # since k is negatively correlated with the architecture’s test accuracy. So, it uses k = lambda_m / lambda_0
        # eigenvalues, _ = torch.symeig(ntk)  # ascending
        begin = time.time()
        eigenvalues, _ = torch.linalg.eigh(ntk)
        end = time.time()

        # print("10. compute eigenvalues done")
        # print("==== eigenvalues: " + str(end - begin))
        # print(showUtilization()[0])

        # convert nan and inf into 0 and 10000
        score = np.nan_to_num((eigenvalues[-1] / eigenvalues[0]).item(), copy=True, nan=100000.0).item()

        # clear un-used instance, this will cause GPU out of memory
        del eigenvalues
        del ntk
        torch.cuda.empty_cache()
        remove_hooks(arch)
        return float(score)
