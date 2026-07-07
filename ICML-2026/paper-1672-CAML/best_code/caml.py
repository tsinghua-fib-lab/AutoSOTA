import torch
from torch import nn

from boundary import BoundaryCondition
from pde import PDE


class CAML(nn.Module):
    """
    CAML - Constraint-Aligned loss with Manifold Lifting.

    Args:
        mask: [N,] tensor boolean mask for boundary points
        pde: PDE object representing the partial differential equation
        boundary_condition: BoundaryCondition object representing the BCs
        w_res: weight for the PDE residual loss
        w_bc: weight for the boundary condition loss
        td: training delay for dynamic lambda (default: 0)
        tr: training ramp for dynamic lambda (default: 0)
        linear: whether to use linear closed-form solution for c (default: True)
        K_init: number of Newton iterations for initial c estimation (default: 10)
        K_few: number of Newton iterations for subsequent c updates (default: 2)
        tc: training iteration to stop updating c (default: 1000)
    """

    def __init__(
            self,
            mask: torch.Tensor,
            pde: PDE,
            boundary_condition: BoundaryCondition,
            w_res: float = 1.0,
            w_bc: float = 1.0,
            td: int = 0,
            tr: int = 0,
            linear: bool = True,
            K_init: int = 10,
            K_few: int = 2,
            tc: int = 1000
    ):
        super().__init__()

        self.mask = mask
        self.pde = pde
        self.bc = boundary_condition

        self.w_res = float(w_res)
        self.w_bc = float(w_bc)
        self.td = td
        self.tr = tr

        self.linear = linear
        self.K_init = K_init
        self.K_few = K_few
        self.tc = tc

        self.c = None

    def _calculate_c_linear(
            self,
            N_res: int,
            N_bc: int,
            gamma: torch.Tensor,
            alpha: torch.Tensor,
            r_i: torch.Tensor,
            s_b: torch.Tensor
    ):
        with torch.no_grad():
            c_res = self.w_res / N_res
            c_bc = self.w_bc / N_bc

            sum_res = (gamma * r_i).sum(dim=0)
            sum_bc = (alpha * s_b).sum(dim=0)
            sum_res_gamma = (gamma * gamma).sum(dim=0)
            sum_bc_alpha = (alpha * alpha).sum(dim=0)

            c = -(c_res * sum_res + c_bc * sum_bc) / (c_res * sum_res_gamma + c_bc * sum_bc_alpha)
        return c.detach()

    def _calculate_c_nonlinear(
            self,
            pde: PDE,
            boundary_condition: BoundaryCondition,
            mesh: torch.Tensor,
            pred: torch.Tensor,
            mask: torch.Tensor,
            iter: int = 2
    ):
        if self.c is None:
            self.c = torch.zeros_like(pred[0])

        c = self.c.clone()
        for _step in range(iter):
            c_ = c.clone().requires_grad_(True)

            # ============================================================
            # compute shifted residuals
            r_i = pde(mesh, pred + c_)
            s_b = boundary_condition(mesh, pred + c_, mask)

            # ============================================================
            # compute losses
            pde_loss = (r_i ** 2).mean(dim=0)
            bc_loss = (s_b ** 2).mean(dim=0)

            total_loss = self.w_res * pde_loss + self.w_bc * bc_loss

            # ============================================================
            # compute gradients
            df_dc = torch.autograd.grad(
                outputs=total_loss,
                inputs=c_,
                grad_outputs=torch.ones_like(total_loss),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            d2f_dc2 = torch.autograd.grad(
                outputs=df_dc,
                inputs=c_,
                grad_outputs=torch.ones_like(df_dc),
                create_graph=False,
                retain_graph=False,
                only_inputs=True
            )[0]

            # ============================================================
            # Newton's update
            c = c - df_dc / (d2f_dc2 + 1e-8)

        return c.detach()

    def _get_lambda(
            self,
            t: int
    ):
        td = self.td
        tr = self.tr

        # no delay / no ramp
        if td <= 0 and tr <= 0:
            return 1.0

        # step schedule if tr == 0
        if tr <= 0:
            return 0.0 if t < td else 1.0

        if t < td:
            return 0.0
        elif t < td + tr:
            return float(t - td) / float(tr)
        else:
            return 1.0

    def get_c(self):
        """
        Get the current estimated coefficient c.

        Returns:
            tensor of shape [D]
        """
        return self.c

    def forward(
            self,
            mesh: torch.Tensor,
            pred: torch.Tensor,
            mask: torch.Tensor = None,
            t: int = 0
    ):
        """
        Compute CALM loss for linear case.

        Args:
            mesh: [N, D] tensor of input points
            pred: [N, D] tensor of backbone predictions at mesh points
            mask: [N,] tensor boolean mask for boundary points
            t: current training iteration (for dynamic lambda)

        Returns:
            total: weighted total loss (tensor)
            pde:   float scalar (python) for logging
            boundary_condition: float scalar for logging
            c:     float scalar for logging
        """

        if mask is None:
            mask = self.mask

        # ============================================================
        # compute coefficient c
        if self.linear:
            # Linear case: closed-form solution for c
            r_i = self.pde(mesh, pred)
            s_b = self.bc(mesh, pred, mask)

            N_res = r_i.shape[0]
            N_bc = s_b.shape[0]
            gamma = self.pde.gamma
            alpha = self.bc.alpha

            c = self._calculate_c_linear(N_res, N_bc, gamma, alpha, r_i, s_b)

        else:
            # Non-linear case: iterative Newton's method for c
            if t == 0:
                c = self._calculate_c_nonlinear(self.pde, self.bc, mesh, pred, mask, self.K_init)
            elif t < self.tc:
                c = self._calculate_c_nonlinear(self.pde, self.bc, mesh, pred, mask, self.K_few)
            else:
                c = self.c.detach()

        self.c = c.detach()

        self.pde.reset()
        self.bc.reset()

        # ============================================================
        # compute shifted residuals
        r_i_ = self.pde(mesh, pred + c)
        pde_loss = nn.MSELoss()(r_i_, torch.zeros_like(r_i_))

        s_b_ = self.bc(mesh, pred + c, mask)
        bc_loss = nn.MSELoss()(s_b_, torch.zeros_like(s_b_))

        # ============================================================
        # total loss with dynamic lambda
        lambda_ = self._get_lambda(t)
        total = self.w_res * lambda_ * pde_loss + self.w_bc * bc_loss

        return {
            "total": total,
            "pde": pde_loss,
            "boundary_condition": bc_loss,
            "c": c.detach().cpu()
        }
