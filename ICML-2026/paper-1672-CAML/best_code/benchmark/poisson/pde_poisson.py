import math
import torch

from pde import PDE


class Poisson(PDE):
    def __init__(self, n: int):
        super(Poisson, self).__init__(linear=True, gamma=torch.zeros([n, 1]))

        self.A = 20.0 + 10
        self.B = 15.0 + 10
        self.C = 8.0 + 10
        self.D = 6.0 + 10

        self._pi = math.pi

    def zeroth(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(u, device=u.device)

    def derivative(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        du = torch.autograd.grad(
            outputs=u,
            inputs=x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]

        u_x = du[:, 0:1]
        u_y = du[:, 1:2]

        u_xx = torch.autograd.grad(
            outputs=u_x,
            inputs=x,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True
        )[0][:, 0:1]

        u_yy = torch.autograd.grad(
            outputs=u_y,
            inputs=x,
            grad_outputs=torch.ones_like(u_y),
            create_graph=True,
            retain_graph=True
        )[0][:, 1:2]

        return u_xx + u_yy

    def source(self, x: torch.Tensor) -> torch.Tensor:
        xx = x[:, 0:1]
        yy = x[:, 1:2]

        pi = torch.tensor(self._pi, device=x.device, dtype=x.dtype)

        term1 = -((2.0 * pi) ** 2) * self.A * torch.sin(2.0 * pi * xx)
        term2 = -((3.0 * pi) ** 2) * self.B * torch.cos(3.0 * pi * yy)
        term3 = -2.0 * (pi ** 2) * torch.sin(pi * xx) * torch.sin(pi * yy)
        return term1 + term2 + term3
