from proxies.autograd_hacks import *
from proxies.task_adapter import proxy_forward, proxy_loss
from torch.autograd import grad
import numpy as np


class NTKTraceApproxEvaluator:

    def __init__(self):
        super().__init__()

    def evaluate(self, arch: nn.Module, device, batch_data: object, batch_labels: torch.Tensor,
                 space_name: str) -> float:
        """
        This is implementation of paper
        "NASI: Label- and Data-agnostic Neural Architecture Search at Initialization"
        The implementation uses loss to approximate the summation over mini-batches,
            ∥Θ(A)∥tr ≈ ∥0_loss(A)∥tr
            ∥0_loss(A)∥tr = sum of [ ∥d_loss_x / dθ(A)∥2 for x∈Xj ]
        The score takes 5 steps:
            1. run forward on a mini-batch
            2. calculate cross_entropy loss and then run backward
            3. calculate NTK with loss
        """

        mu = 0
        gap = 0

        # run in train mode
        # get all weights from architecture
        # todo: don't include the embedding
        model_params = [p for n, p in arch.named_parameters() if p is not None and "embedding" not in str(n)]

        # 1. forward on mini-batch & calculate loss
        output_f = proxy_forward(arch, batch_data)
        task_loss = proxy_loss(output_f, batch_labels, space_name)

        # 2. run backward on parameters
        # Evaluate gradient Gt = ∇Theta L(x) with data Dt
        grads = grad(task_loss, model_params, create_graph=False, allow_unused=True)

        # 3. calculate final score with ∥0(A)∥tr − mu * F.relu(∥0(A)∥tr − gap) .
        score, avg_eigen = self.trace_loss(grads, gap, mu)
        return float(np.nan_to_num(score, nan=0.0, posinf=1e8, neginf=-1e8))

    def trace_loss(self, grads, reg_weight, gap) -> (float, float):
        """
        Evaluate score of a given architecture with only one batch
        :param grads: y grads on weights
        :param gap: v in paper, v = r * n* 1/learning rate
        :param reg_weight: mu in paper.
        :return: score and NTK
        """

        grad_list = []
        for g in grads:
            if g is not None:
                ele = (g ** 2).sum()
                grad_list.append(ele)

        avg_eigen = sum(grad_list) / len(grad_list)

        # avg_eigen = sum([ (g ** 2).sum() for g in grads])
        score = avg_eigen - reg_weight * F.relu(avg_eigen - gap)
        return score.item(), avg_eigen
