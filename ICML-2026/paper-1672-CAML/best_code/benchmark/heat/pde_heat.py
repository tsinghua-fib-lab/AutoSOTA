import torch

from pde import PDE


class Heat(PDE):
    def __init__(
            self,
            n: int
    ):
        super(Heat, self).__init__(linear=True, gamma=torch.zeros([n, 1]))
        self.n = n

    def zeroth(self, x, u):
        return torch.zeros_like(u, device=u.device)

    def derivative(self, x, u):
        du = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]

        u_x = du[:, 0:1]
        u_y = du[:, 1:2]

        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True, retain_graph=True
        )[0][:, 0:1]

        u_yy = torch.autograd.grad(
            u_y, x,
            grad_outputs=torch.ones_like(u_y),
            create_graph=True, retain_graph=True
        )[0][:, 1:2]

        return u_xx + u_yy

    def source(self, x):
        return torch.zeros([x.shape[0], 1], device=x.device)
