import torch
from torch import nn

from boundary import BoundaryCondition
from pde import PDE


class PINNLoss(nn.Module):
    def __init__(
            self,
            mask: torch.Tensor,
            pde: PDE,
            boundary_condition: BoundaryCondition,
            w_res: float = 1.0,
            w_bc: float = 1.0
    ):
        super().__init__()

        self.mask = mask
        self.pde = pde
        self.bc = boundary_condition

        self.w_res = float(w_res)
        self.w_bc = float(w_bc)

    def forward(
            self,
            mesh: torch.Tensor,
            pred: torch.Tensor,
            mask: torch.Tensor = None,
            t: int = 0
    ):
        r_i_ = self.pde(mesh, pred)
        pde_loss = nn.MSELoss()(r_i_, torch.zeros_like(r_i_))

        s_b_ = self.bc(mesh, pred, mask)
        bc_loss = nn.MSELoss()(s_b_, torch.zeros_like(s_b_))
        total = self.w_res * pde_loss + self.w_bc * bc_loss

        return {
            "total": total,
            "pde": pde_loss,
            "boundary_condition": bc_loss
        }
