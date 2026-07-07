import math
import torch
from pde import PDE


class Helm(PDE):
    def __init__(self, n: int, U0: float = 100.0):
        super().__init__(linear=True, gamma=torch.zeros([n, 1]))
        self._pi = math.pi
        self.U0 = float(U0)

    def _coeff_A_q(self, x: torch.Tensor):
        xx = x[:, 0:1]
        yy = x[:, 1:2]
        pi = torch.tensor(self._pi, device=x.device, dtype=x.dtype)

        a11 = 1.0 + 0.3 * xx
        a22 = 1.0 + 0.3 * yy
        a12 = 0.15 * torch.sin(pi * xx) * torch.sin(pi * yy)

        q = 2.0 + torch.cos(pi * xx) * torch.cos(pi * yy)
        return a11, a12, a22, q

    def u_star(self, x: torch.Tensor) -> torch.Tensor:
        xx = x[:, 0:1]
        yy = x[:, 1:2]
        pi = torch.tensor(self._pi, device=x.device, dtype=x.dtype)

        U0 = torch.tensor(self.U0, device=x.device, dtype=x.dtype)
        return U0 + torch.sin(pi * xx) * torch.cos(2.0 * pi * yy) + 0.2 * torch.exp(xx + yy)

    def zeroth(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(u)

    def derivative(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        a11, a12, a22, q = self._coeff_A_q(x)
        self.gamma = q.detach()

        du = torch.autograd.grad(
            outputs=u,
            inputs=x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        u_x = du[:, 0:1]
        u_y = du[:, 1:2]

        fx = a11 * u_x + a12 * u_y
        fy = a12 * u_x + a22 * u_y

        dfx = torch.autograd.grad(
            outputs=fx,
            inputs=x,
            grad_outputs=torch.ones_like(fx),
            create_graph=True,
            retain_graph=True
        )[0]
        dfy = torch.autograd.grad(
            outputs=fy,
            inputs=x,
            grad_outputs=torch.ones_like(fy),
            create_graph=True,
            retain_graph=True
        )[0]

        div_flux = dfx[:, 0:1] + dfy[:, 1:2]
        return -div_flux + q * u

    def source(self, x: torch.Tensor) -> torch.Tensor:
        if not x.requires_grad:
            x = x.requires_grad_(True)

        u_s = self.u_star(x)
        f = self.derivative(x, u_s)
        return f
