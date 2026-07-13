"""LBFGS-only infimal convolution solver used in mechanism training."""

import torch
from torch.autograd import Function
from torch.nn.utils import stateless
from tools.feedback import logger


class InfConvolution(Function):
    """
    Solver-only LBFGS for:

        x*(y) = argmin_x [ K(x,y) - f(x) + 0.5*lam ||x||^2 ].

    Forward:
        returns (x_star, converged), both detached.

    Backward:
        returns no gradients (None for all inputs).
        The envelope theorem gradient wrt parameters comes
        from computing g(y) = K(x*,y) - f(x*) OUTSIDE this op
        and backpropagating through f(x*).
    """

    @staticmethod
    def forward(
        ctx,
        y,
        f_net,
        K,
        x_init,
        solver_steps: int = 40,
        lr: float = 1.0,
        optimizer: str = "lbfgs",
        lam: float = 1e-5,
        tol: float = 1e-6,
        patience: int = 5,
        projection=None,
    ):
        """Run LBFGS solve for x* minimizing K(x,y) - f(x) + 0.5 * lam ||x||^2."""

        if optimizer.lower() != "lbfgs":
            raise ValueError("InfConvolution only supports LBFGS in solver-only mode.")

        if projection is None:
            projection = lambda x: x

        # detach y so solver is not differentiable wrt y
        y_det = y.detach()

        # initialize x
        x_var = x_init.detach().clone()
        with torch.no_grad():
            x_var = projection(x_var)
        x_var.requires_grad_(True)

        # freeze parameters (no grad wrt θ inside solver)
        params = {name: p.detach() for name, p in f_net.named_parameters()}

        def f_solve(x):
            # x: (dim,) or (1,dim)
            if x.dim() == 1:
                x_in = x.unsqueeze(0)
            else:
                x_in = x
            # functional_call with detached params: no θ-grad, but x-grad OK
            out = stateless.functional_call(f_net, params, (x_in,))
            return out.squeeze()   # <- NO detach here

        def has_bad(t):
            return bool(torch.isnan(t).any() or torch.isinf(t).any())

        proj_hits = 0

        def closure():
            nonlocal proj_hits
            if x_var.grad is not None:
                x_var.grad.zero_()

            # projection
            with torch.no_grad():
                new_x = projection(x_var)
                if not torch.allclose(new_x, x_var):
                    proj_hits += 1
                x_var.copy_(new_x)

            k_val = K(x_var, y_det).squeeze()
            f_val = f_solve(x_var)
            obj = k_val - f_val + 0.5 * lam * x_var.pow(2).sum()

            if has_bad(obj):
                raise RuntimeError("LBFGS produced NaN/Inf objective")

            obj.backward()
            return obj

        optim = torch.optim.LBFGS(
            [x_var],
            lr=lr,
            max_iter=solver_steps,
            tolerance_grad=tol,
            tolerance_change=tol,
            line_search_fn="strong_wolfe",
        )

        try:
            optim.step(closure)
        except RuntimeError as e:
            logger.warning(f"[InfConvolution] LBFGS step failed: {e}")

        # final projection
        with torch.no_grad():
            x_var.copy_(projection(x_var))

        # convergence heuristic
        x_chk = x_var.detach().clone().requires_grad_(True)
        obj_chk = K(x_chk, y_det).squeeze() - f_solve(x_chk) + 0.5 * lam * x_chk.pow(2).sum()
        try:
            obj_chk.backward()
            gn = x_chk.grad.norm().item()
        except RuntimeError:
            gn = float("inf")

        try:
            state = optim.state.get(optim.param_groups[0]["params"][0], {})
            n_iter = state.get("n_iter", solver_steps)
        except Exception:
            n_iter = solver_steps

        converged = (n_iter < solver_steps) and (gn < tol * (1 + abs(obj_chk.item())))

        if proj_hits > 5:
            logger.debug(f"[InfConvolution] Projection hit {proj_hits} times")
        if not converged:
            logger.debug(
                f"[InfConvolution] DID NOT CONVERGE: iters={n_iter}, grad_norm={gn:.2e}"
            )

        x_star = x_var.detach()
        # we don’t need to save anything for backward since it always returns None
        return x_star, converged

    @staticmethod
    def backward(ctx, *grad_outputs):
        """Backward hook returns None because solver does not propagate gradients."""
        return (None,) * 11
