import math

import torch
import torch.nn.functional as f


__all__ = ["POGO"]


# see https://en.wikipedia.org/wiki/Quartic_equation#Summary_of_Ferrari's_method
def _solve_quartic_equation(coefs: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-3
    assert coefs.shape[1] == 5

    coefs = coefs.cfloat()
    coefs = coefs[:, 1:] / coefs[:, :1]
    B, C, D, E = coefs.T

    # Step 1
    a = -3./8 * B**2 + C
    b = B**3 / 8. - B * C / 2. + D
    c = - 3. / 256 * B**4 + C * B**2 / 16 - B * D / 4 + E

    is_b_zero = b.real.abs() < epsilon
    
    # Sol b == 0
    opt1 = torch.sqrt(a** 2 - 4 * c)
    sol_b_zero = [torch.sqrt((-a + (-1)**i * opt1) / 2) for i in [0,1]]
    sol_b_zero.extend([-x for x in sol_b_zero])
    sol_b_zero = torch.stack(sol_b_zero, dim=-1)
    sol_b_zero -= B.unsqueeze(-1) / 4.

    # Otherwise
    P = -a**2 / 12 - c
    Q = -a**3 / 108 + a*c / 3 - b**2 / 8
    R = -Q / 2 + torch.sqrt(Q**2 / 4 + P**3 / 27)
    U = torch.pow(R, 1/3.)

    y = -5/6 * a + torch.where(
        U == 0, # TODO close to zero?
        - torch.pow(Q, 1/3.),
        U - P / (3*U)
    )
    W = torch.sqrt(a + 2 * y)

    opt1 = 2 * b / W
    sol_b_nonzero = [torch.sqrt(
            -(3*a + 2*y + (-1)**i * opt1)
        ) for i in [0,1]]
    sol_b_nonzero.extend([-x for x in sol_b_nonzero])
    sol_b_nonzero = torch.stack(sol_b_nonzero, dim=-1)
    sol_b_nonzero[:, [0, 2]] += W.unsqueeze(-1)
    sol_b_nonzero[:, [1, 3]] -= W.unsqueeze(-1)
    sol_b_nonzero = sol_b_nonzero / 2. - B.unsqueeze(-1) / 4.
    
    # Choose based on b
    solution = torch.where(
        is_b_zero.unsqueeze(-1),
        sol_b_zero,
        sol_b_nonzero
    )
    return solution


# see https://en.wikipedia.org/wiki/Cubic_equation
def _solve_cubic_equation(coefs: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-3
    assert coefs.shape[1] == 4

    coefs = coefs.cfloat()
    a, b, c, d = coefs.T

    d0 = b**2 - 3*a*c
    d1 = 2*b**3 - 9*a*b*c + 27*a**2*d
    
    C = torch.pow((d1 + torch.sqrt(d1**2 - 4*d0**3)) / 2, 1/3.)
    psi = (-1 + math.sqrt(3)*1j) / 2.
    
    solution = [
        - (b + psi**k * C + d0 / (psi**k * C)) / (3*a)
        for k in [0, 1, 2]
    ]
    solution = torch.stack(solution, dim=-1)
    
    return solution


# see https://en.wikipedia.org/wiki/Quadratic_formula
def _solve_quadratic_equation(coefs: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-3
    assert coefs.shape[1] == 3

    coefs = coefs.cfloat()
    a, b, c = coefs.T

    disc = torch.sqrt(b**2 - 4*a*c)
    
    solution = torch.stack([
        disc, -disc
    ], dim=-1)
    solution = (solution - b.unsqueeze(-1)) / (2 * a.unsqueeze(-1))

    return solution


# a*x + b = 0
def _solve_monic_equation(coefs: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-3
    assert coefs.shape[1] == 2

    coefs = coefs.cfloat()
    a, b = coefs.T

    solution = -b/a 
    solution = solution.unsqueeze(-1)

    return solution


def _solve_poly_equation(coefs: torch.Tensor) -> torch.Tensor:
    eps = 1e-10
    eps = 1e-4
    assert coefs.shape[1] == 5

    quartic = coefs[:, 0].abs() > eps
    cubic = ~quartic & (coefs[:, 1].abs() > eps)
    quadratic = ~quartic & ~cubic & (coefs[:, 2].abs() > eps)
    monic = ~quartic & ~cubic & ~quadratic

    sol = torch.empty((coefs.shape[0], 4), device=coefs.device, dtype=torch.cfloat)
    if quartic.any():
        sol_quartic = _solve_quartic_equation(coefs[quartic])
        sol[quartic] = sol_quartic
    if cubic.any():
        #sol_cubic = solve_cubic(*coefs[cubic][..., 1:].unsqueeze(-1).unbind(dim=1))[..., 0] + 0j
        sol_cubic = _solve_cubic_equation(coefs[cubic][..., 1:])
        sol_cubic = torch.cat((sol_cubic, sol_cubic[:, :1]), dim=-1)
        sol[cubic] = sol_cubic
    if quadratic.any():
        sol_quadratic = _solve_quadratic_equation(coefs[quadratic][..., 2:])
        sol_quadratic = torch.cat((sol_quadratic, sol_quadratic), dim=-1)
        sol[quadratic] = sol_quadratic
    if monic.any():
        sol_monic = _solve_monic_equation(coefs[monic][..., 3:])
        sol_monic = torch.cat([sol_monic for _ in range(4)], dim=-1)
        sol[monic] = sol_monic

    return sol


def compute_lambda(A, B):
    p = A.shape[-2]
    I = torch.eye(p, device=A.device, dtype=A.dtype).view(*[1 for _ in A.shape[:-2]], p, p)
    
    # A @ A.T - I  # l^0
    AAT = torch.bmm(A, A.conj().transpose(-1, -2)) - I
    # B @ B.T      # l^2
    BBT = torch.bmm(B, B.conj().transpose(-1, -2)) 
    # A @ B.T + B @ A.T  # l^1
    ABT = torch.bmm(A, B.conj().transpose(-1, -2)) + torch.bmm(B, A.conj().transpose(-1, -2))
    # foo(X, Y): trace(Y.T @ X)
    foo = lambda x, y: (x*y).sum(dim=(-1, -2))


    distance = foo(AAT, AAT)
    coefs = torch.stack([
        foo(BBT, BBT),  # 2+2
        2*foo(ABT, BBT),  # 2+1
        2*foo(BBT, AAT) + foo(ABT, ABT),  # 1+1 and 2+0
        2*foo(AAT, ABT),  # 1+0
        distance  # 0+0
    ], dim=-1)

    # Mask out matrices where B is zero
    mask = coefs[:, 0] == 0
    coefs[mask, :-1] = 1.

    # Solve the polynomial
    coefs = coefs / coefs.abs().max(dim=-1, keepdim=True)[0]
    roots = _solve_poly_equation(coefs)
    
    # Take the root that moves us the least
    if torch.is_complex(A):                                                                                         
        indices = roots.abs().argmin(dim=1)                                                                         
        lambda_regul = roots.flatten()[torch.arange(0, roots.shape[0] * 4, 4, device=indices.device) + indices]     
    else:                                                                                                           
        indices = roots.imag.abs().argmin(dim=1)                                                                    
        lambda_regul = roots.real.flatten()[torch.arange(0, roots.shape[0] * 4, 4, device=indices.device) + indices]
    return lambda_regul


def _check_size(param):
    *b, p, q = param.shape
    assert len(b) > 0, f"The parameter must be shaped as [*batch_dims, p, q] with p < q. Instead got {param.shape}"
    if p > q:
        raise ValueError(
            "The second to last dimension of the parameters should be greater or equal than its last dimension. "
            "Only tall matrices are supported so far"
        )


def _relative_gradient(point, grad):
    XXtG = torch.bmm(torch.bmm(point, point.conj().transpose(-1, -2)), grad)
    XGtX = torch.bmm(torch.bmm(point, grad.conj().transpose(-1, -2)), point)
    rel_grad = .5*(XXtG - XGtX)  

    return rel_grad


# Implemented on the StiefelT manifold
class POGO(torch.optim.Optimizer):
    r"""
    POGO algorithm on the Stiefel manifold.

    Parameters
    ----------
    params : iterable
        iterable of parameters to optimize or dicts defining
        parameter groups. Must contain square orthogonal matrices.
    base_optimizer: 
        Base Euclidean optimizer from `.base`. 
    lr : float
        learning rate
    weight_decay : float (optional)
        weight decay (L2 penalty) (default: 0)
    check_size : bool (optional)
        whether to check that the parameters are of the correct size.
    flatten_fn: function (optional)
        a function to reshape the tensor (without other changes, e.g., transposes)
        to make the point and parameters be of shape [batch, p, n] or [batch, n, p] 
        (see `rows` parameter).
    lambda_every : int (optional)
        frequency for which to compute the value of lambda solving the 
        landing polynomial. A negative value disables it. (default: -1)
    rows: bool (optional)
        whether to optimize row-orthogonal matrices (p x n) or column-orthogonal
        matrices (n x p), with p <= n.
    """

    def __init__(
        self,
        params,
        base_optimizer, 
        lr,
        weight_decay=0,
        check_size=True,
        flatten_fn=lambda x: x,
        lambda_every: int = -1,
        rows: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay)
            )
        self.base_optimizer = base_optimizer
        
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            check_size=check_size,
            flatten_fn=flatten_fn,
            lambda_every=lambda_every,
            rows=rows,
            **base_optimizer.get_defaults()
        )
        
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        for group in self.param_groups:
            if "step" not in group:
                group["step"] = 0
            weight_decay = group["weight_decay"]
            lr = group["lr"]
            check_size = group["check_size"]
            flatten_fn = group["flatten_fn"]
            lambda_every = group["lambda_every"]
            rows = group["rows"]
            group["step"] += 1
            for point_ind, point_ in enumerate(group["params"]):
                if point_.grad is None:
                    continue

                if torch.allclose(point_.grad, torch.zeros_like(point_.grad)):
                    continue

                o_sh = point_.shape
                point = flatten_fn(point_)
                grad = flatten_fn(point_.grad)

                if not rows:
                    point = point.transpose(-1, -2)
                    grad = grad.transpose(-1, -2)

                if check_size:
                    _check_size(point)

                if grad.is_sparse:
                    raise RuntimeError(
                        "NoLanding does not support sparse gradients."
                    )
                state = self.state[point_]

                grad.add_(point, alpha=weight_decay)

                # Use the BaseOptimizer
                grad = self.base_optimizer(point, grad, state, group)

                # Compute the relative gradient
                rel_grad = _relative_gradient(point, grad)

                # New point after following rel_grad
                midpoint = point - lr * rel_grad

                # Compute polynomial coefficients
                # endpoint = A + lambda * B
                A = midpoint
                B = midpoint - torch.bmm(torch.bmm(midpoint, midpoint.conj().transpose(-1, -2)), midpoint)
                
                if (lambda_every > 0) and (group["step"] % lambda_every == 0):
                    lambda_regul = compute_lambda(A, B)
                    lambda_regul = lambda_regul.unsqueeze(-1).unsqueeze(-1)
                else:
                    lambda_regul = 0.5

                # Compute the final point and copy it
                endpoint = A + lambda_regul * B
                if not rows:
                    endpoint = endpoint.transpose(-1, -2)

                point_.copy_(endpoint.view(o_sh))
        return loss
    
