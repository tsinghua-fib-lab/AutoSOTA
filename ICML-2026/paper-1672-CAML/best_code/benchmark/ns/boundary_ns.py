import math
import torch
from boundary import BoundaryCondition


class BoundaryNS(BoundaryCondition):
    def __init__(self, n: int, normal_vector: torch.Tensor, U: tuple[float, float] = (1.0, 1.0)):
        alpha = torch.ones(n, 2, dtype=normal_vector.dtype, device=normal_vector.device)
        beta = torch.zeros(n, 2, dtype=normal_vector.dtype, device=normal_vector.device)
        super().__init__(alpha=alpha, beta=beta, normal_vector=normal_vector)

        self.Ux = float(U[0])
        self.Uy = float(U[1])

    def exact_uv(self, x: torch.Tensor) -> torch.Tensor:
        X = x[:, 0:1]
        Y = x[:, 1:2]
        pi = math.pi

        u0 = pi * torch.sin(pi * X) * torch.cos(pi * Y)
        v0 = -pi * torch.cos(pi * X) * torch.sin(pi * Y)

        if (self.Ux != 0.0) or (self.Uy != 0.0):
            u0 = u0 + self.Ux
            v0 = v0 + self.Uy

        return torch.cat([u0, v0], dim=1)

    def source(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        g_all = self.exact_uv(x)
        g_bc = self._split_points(g_all, mask)
        return g_bc

    def forward(self, x: torch.Tensor, u: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        uv = u[:, :2]
        bc = super().forward(x=x, u=uv, mask=mask)
        pad = torch.zeros_like(bc[:, 0:1])
        bc = torch.cat([bc, pad], dim=1)
        return bc
