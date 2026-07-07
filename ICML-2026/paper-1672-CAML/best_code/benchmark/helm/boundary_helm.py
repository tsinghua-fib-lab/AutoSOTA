import math
import torch
from boundary import BoundaryCondition


class BoundaryHelm(BoundaryCondition):
    def __init__(self, n: int, normal_vector: torch.Tensor, U0: float = 100.0):
        alpha = torch.ones([n, 1])
        beta = torch.zeros([n, 1])
        super().__init__(alpha=alpha, beta=beta, normal_vector=normal_vector)

        self._pi = math.pi
        self.U0 = float(U0)

    def u_star(self, x: torch.Tensor) -> torch.Tensor:
        xx = x[:, 0:1]
        yy = x[:, 1:2]
        pi = torch.tensor(self._pi, device=x.device, dtype=x.dtype)

        U0 = torch.tensor(self.U0, device=x.device, dtype=x.dtype)
        return U0 + torch.sin(pi * xx) * torch.cos(2.0 * pi * yy) + 0.2 * torch.exp(xx + yy)

    def source(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        u_all = self.u_star(x)
        return self._split_points(u_all, mask)
