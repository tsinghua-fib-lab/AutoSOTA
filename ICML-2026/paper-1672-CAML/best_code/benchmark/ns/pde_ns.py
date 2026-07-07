import math
import torch
from pde import PDE


class NS(PDE):

    def __init__(self, Re: float, U: tuple[float, float] = (1.0, 1.0)):
        super().__init__(linear=False, gamma=None)
        if Re <= 0:
            raise ValueError("Re must be positive.")
        self.Re = float(Re)

        self.Ux = float(U[0])
        self.Uy = float(U[1])

        self.grad_u = None
        self.grad_v = None
        self.grad_p = None
        self.lap_u = None
        self.lap_v = None

    @staticmethod
    def _grad_scalar(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if y.ndim == 2 and y.shape[1] == 1:
            y = y[:, 0]
        return torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=torch.ones_like(y),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

    def _laplacian_scalar(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        dy = self._grad_scalar(y, x)
        d2x = self._grad_scalar(dy[:, 0], x)
        d2y = self._grad_scalar(dy[:, 1], x)
        return d2x[:, 0] + d2y[:, 1]

    def forcing(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        X = x[:, 0:1]
        Y = x[:, 1:2]
        pi = math.pi
        Re = self.Re

        sx = torch.sin(pi * X)
        cx = torch.cos(pi * X)
        sy = torch.sin(pi * Y)
        cy = torch.cos(pi * Y)

        u0_x = (pi ** 2) * cx * cy
        u0_y = -(pi ** 2) * sx * sy
        v0_x = (pi ** 2) * sx * sy
        v0_y = -(pi ** 2) * cx * cy

        conv_u = (pi ** 3) * sx * cx
        conv_v = (pi ** 3) * sy * cy

        if (self.Ux != 0.0) or (self.Uy != 0.0):
            conv_u = conv_u + self.Ux * u0_x + self.Uy * u0_y
            conv_v = conv_v + self.Ux * v0_x + self.Uy * v0_y

        p_x = 2.0 * pi * torch.cos(2.0 * pi * X) * torch.sin(2.0 * pi * Y)
        p_y = 2.0 * pi * torch.sin(2.0 * pi * X) * torch.cos(2.0 * pi * Y)

        lap_u = -2.0 * (pi ** 3) * sx * cy
        lap_v = 2.0 * (pi ** 3) * cx * sy

        f_u = conv_u + p_x - (1.0 / Re) * lap_u
        f_v = conv_v + p_y - (1.0 / Re) * lap_v
        return f_u, f_v

    def zeroth(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return torch.zeros((u.shape[0], 3), device=u.device, dtype=u.dtype)

    def derivative(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        if u.shape[1] < 3:
            raise ValueError("NS expects backbone output with 3 channels [u,v,p].")

        uu = u[:, 0:1]
        vv = u[:, 1:2]
        pp = u[:, 2:3]

        if self.first:
            grad_u = self._grad_scalar(uu, x)
            grad_v = self._grad_scalar(vv, x)
            grad_p = self._grad_scalar(pp, x)
            self.grad_u = grad_u
            self.grad_v = grad_v
            self.grad_p = grad_p
        else:
            grad_u = self.grad_u
            grad_v = self.grad_v
            grad_p = self.grad_p

        u_x, u_y = grad_u[:, 0], grad_u[:, 1]
        v_x, v_y = grad_v[:, 0], grad_v[:, 1]
        p_x, p_y = grad_p[:, 0], grad_p[:, 1]

        if self.first:
            lap_u = self._laplacian_scalar(uu, x)
            lap_v = self._laplacian_scalar(vv, x)
            self.lap_u = lap_u
            self.lap_v = lap_v
        else:
            lap_u = self.lap_u
            lap_v = self.lap_v

        conv_u = uu[:, 0] * u_x + vv[:, 0] * u_y
        conv_v = uu[:, 0] * v_x + vv[:, 0] * v_y

        r_u = conv_u + p_x - (1.0 / self.Re) * lap_u
        r_v = conv_v + p_y - (1.0 / self.Re) * lap_v
        r_div = u_x + v_y

        return torch.stack([r_u, r_v, r_div], dim=1)

    def source(self, x: torch.Tensor) -> torch.Tensor:
        f_u, f_v = self.forcing(x)
        zeros = torch.zeros_like(f_u)
        return torch.cat([f_u, f_v, zeros], dim=1)
