from abc import abstractmethod
import torch
from torch import nn


class BoundaryCondition(nn.Module):
    def __init__(
            self,
            alpha: torch.Tensor,
            beta: torch.Tensor,
            normal_vector: torch.Tensor
    ):
        super().__init__()

        self.register_buffer("alpha", alpha)
        self.register_buffer("beta", beta)
        self.register_buffer("normal_vector", normal_vector)

        self.first = True
        self.derivative_values = None
        self.source_values = None

    def _split_points(
            self,
            tensor: torch.Tensor,
            mask: torch.Tensor
    ):
        bc_idx = mask.bool().view(-1)
        tensor_boundary = tensor[bc_idx]
        return tensor_boundary

    def _mask_points(
            self,
            tensor: torch.Tensor,
            mask: torch.Tensor
    ):
        mask = mask.reshape(mask.shape[0], *([1] * (tensor.ndim - 1))).expand_as(tensor).float()
        tensor_boundary = tensor * mask
        return tensor_boundary

    def zeroth(self, u: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        u = self._split_points(u, mask)
        return self.alpha * u

    def derivative(self, x: torch.Tensor, u: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        u_ = self._mask_points(u, mask)

        grads = []
        for c in range(u_.shape[1]):
            du_c_dx = torch.autograd.grad(
                outputs=u_[:, c],
                inputs=x,
                grad_outputs=torch.ones_like(u_[:, c]),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            grads.append(du_c_dx)

        du_dx = torch.stack(grads, dim=1)
        du_dx = self._split_points(du_dx, mask)

        normal_derivative = torch.einsum("ni,nci->nc", self.normal_vector, du_dx)

        return self.beta * normal_derivative

    @abstractmethod
    def source(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pass

    def reset(self):
        self.first = True
        self.derivative_values = None
        self.source_values = None

    def forward(self, x: torch.Tensor, u: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.first:
            self.derivative_values = self.derivative(x, u, mask)
            self.source_values = self.source(x, mask)
        res = self.zeroth(u, mask) + self.derivative_values - self.source_values
        self.first = False
        return res
