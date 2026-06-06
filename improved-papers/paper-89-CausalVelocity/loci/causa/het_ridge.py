import logging
import torch
from torch.distributions import Normal
from torch.optim import LBFGS, Adam


EPS = 1e-7


def log_natural_normal(y, eta_1, eta_2):
    """log pdf of the normal with natural parameters

    Parameters
    ----------
    eta_1 : (N,) first natural parameter
    eta_2 : (N,) second natural parameter
    """
    scale = torch.sqrt(- 0.5 / eta_2)
    loc = - 0.5 * eta_1 / eta_2
    dist = Normal(loc=loc, scale=scale)
    assert loc.shape == y.shape == scale.shape
    return dist.log_prob(y).sum()


def convex_fgls(Phi, Psi, y, delta_Phi, delta_Psi, n_steps=1000):
    """Implements feasible generalised least squares (FGLS) with natural parameterization
    and guaranteed convexity.

    We model the natural parameters of the Gaussian likelihood with
    eta_1(x) = Phi(x)^T w_1
    eta_2(x) = -Psi(x)^T w_2
    s.t. constraints
    Psi(x) >= 0 forall x
    w_2 >= 0

    To satisfy the first constraint, the user can, for example, pass abs(Psi(x)) as feature map.
    The second constraint is satisfied by simple projection.

    Parameters
    ----------
    Phi : N x D feature map for mean/var output (eta1)
    Psi : N x D feature map for 1/var output (eta2)
    y : N observations
    delta_Phi : scalar prior precision on weights of Phi
    delta_Psi : scalar prior precision on weights of Psi
    n_steps : max number of optimization steps
    """
    dtype = Phi.dtype
    device = Phi.device
    N, D = Phi.shape
    losses = list()
    if not torch.all(Psi >= 0):
        raise ValueError('Feature map Psi needs to be non-negative.')

    w_1 = torch.zeros((D,), dtype=dtype, device=device)
    w_2 = torch.ones((D,), dtype=dtype, device=device)
    y = y.flatten()

    def loss_closure():
        eta_1 = Phi @ w_1
        eta_2 = - (Psi @ w_2).clamp(min=EPS)
        loss = (- log_natural_normal(y, eta_1, eta_2)
                + delta_Phi/2 * w_1.norm().square()
                + delta_Psi/2 * w_2.norm().square()) / N
        return loss

    def log_loss():
        eta_1 = Phi @ w_1
        eta_2 = - (Psi @ w_2).clamp(min=EPS)
        loss = - log_natural_normal(y, eta_1, eta_2) / N
        return loss

    losses.append(loss_closure().item())
    optim = LBFGS([w_2], line_search_fn='strong_wolfe')
    logging.info(f'Initial loss {losses[-1]}')

    for n in range(n_steps):

        if n % 2 == 0:
            # update w_1 with closed-form update
            eta_2 = - (Psi @ w_2).clamp(min=EPS)
            alpha = 0.5 / (- eta_2)
            A = Phi.T @ (Phi * alpha.reshape(-1, 1)) + delta_Phi * torch.eye(D, dtype=dtype, device=device)
            A = torch.linalg.cholesky(A)
            w_1 = torch.cholesky_solve((Phi.T @ y).unsqueeze(-1), A).squeeze()
            losses.append(loss_closure().item())
            logging.info(f'Loss after closed-form update on w_1: {losses[-1]}')

        else:
            # update w_2 with L-BFGS (no closed-form available)
            # INFO: tried GD and Newton with line search but L-BFGS works best and fastest
            w_2.requires_grad = True
            for i in range(10):
                optim.zero_grad()
                loss = loss_closure()
                loss.backward()
                loss = optim.step(loss_closure)

                # project back to constraint
                w_2.data = w_2.data.clamp(min=0.0)
            losses.append(loss.item())
            w_2.requires_grad = False
            logging.info(f'Loss after L-BFGS on w_2: {losses[-1]}')

        if abs(losses[-1] - losses[-2]) < 1e-6:
            logging.info('Finish optimization, absolute difference below tolerance')
            break

    return w_1, w_2, losses, log_loss().item()
