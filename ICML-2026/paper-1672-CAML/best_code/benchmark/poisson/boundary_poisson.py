import math

import torch

from boundary import BoundaryCondition


class BoundaryPoisson(BoundaryCondition):
    def __init__(self, n: int, normal_vector: torch.Tensor):
        alpha = torch.ones([n, 1])
        beta = torch.zeros([n, 1])
        super().__init__(alpha=alpha, beta=beta, normal_vector=normal_vector)

        self.A = 20.0 + 10
        self.B = 15.0 + 10
        self.C = 8.0 + 10
        self.D = 6.0 + 10
        self._pi = math.pi

    def source(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        xx = x[:, 0:1]
        yy = x[:, 1:2]

        pi = torch.tensor(self._pi, device=x.device, dtype=x.dtype)

        g_all = (
                self.A * torch.sin(2.0 * pi * xx)
                + self.B * torch.cos(3.0 * pi * yy)
                + self.C * xx
                + self.D * yy
        )

        return self._split_points(g_all, mask) + 10
