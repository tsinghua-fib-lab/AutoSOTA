import torch
from overrides import overrides

from boundary import BoundaryCondition


class BoundaryHeat(BoundaryCondition):
    def __init__(
            self,
            alpha: torch.Tensor,
            beta: torch.Tensor,
            normal_vector: torch.Tensor
    ):
        super().__init__(alpha, beta, normal_vector)

        self.register_buffer("T0", torch.tensor(100.0, dtype=self.alpha.dtype, device=self.alpha.device))
        self.register_buffer("grad_const", torch.tensor(-15.0, dtype=self.alpha.dtype, device=self.alpha.device))

    @overrides
    def source(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.T0 + self.beta * self.grad_const
