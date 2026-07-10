from scipy.optimize import minimize
from scipy.optimize import SR1
from scipy.optimize import LinearConstraint, Bounds
import numpy as np
import torch


def test_qp():
    torch.manual_seed(1)
    # Generate samples from p and q
    m = 20
    n = 20
    dim = 2
    x = torch.randn(m, dim)
    y = torch.randn(n, dim)
    costs = torch.cdist(x, y, p=2)

    # Create the problem matrices
    lamb = 0.1
    one_n = torch.ones(n)
    one_m = torch.ones(m)
    I_m = torch.eye(m)
    I_n = torch.eye(n)
    M = torch.kron(one_n, I_m)
    N = torch.kron(I_n, one_m)

    # Shape them into input that scipy requires
    c = torch.flatten(costs).numpy()
    mat = (M.T @ M).numpy()
    C = torch.cat((N, -N)).numpy()
    b = torch.cat((one_n / n, -one_n / n)).numpy()

    def fun(x):
        return c @ x - lamb * (x.T @ mat @ x)

    def fun_jac(x):
        return c - 2 * lamb * mat @ x

    def fun_hess(x):
        return -2 * lamb * mat

    linear_constraint = LinearConstraint(C, lb=b)
    bounds = Bounds(lb=np.zeros(m * n))

    reg = 1e-12
    w_target = one_n / n
    ot_mat = torch.softmax(-costs / reg, dim=0) * w_target[None, :]
    x0 = torch.flatten(ot_mat)
    # x0 = torch.ones(m*n) / (m*n)
    res = minimize(
        fun,
        x0,
        method="trust-constr",
        jac=fun_jac,
        hess=fun_hess,
        constraints=[linear_constraint],
        tol=1e-4,
        options={"verbose": 1},
        bounds=bounds,
    )

    x = torch.tensor(res.x)
    # Shape x into ot_mat
    ot_mat = torch.unflatten(x, dim=-1, sizes=(m, n))
    print(f"Function value: {res.fun}")
    # print(f"Optimized mat: {ot_mat}")

    # Shape them into input that quadprog requires
    # G = -2 * lamb * M.T @ M
    # a = -torch.flatten(costs)
    # C = torch.cat((N, -N)).T
    # b = torch.cat((one_n / n, -one_n / n))
    # G = G.double().numpy()
    # a = a.double().numpy()
    # C = C.double().numpy()
    # b = b.double().numpy()
    # x, f, xu, iterations, lagrangian, iact = quadprog.solve_qp(G, a, C, b)[0]


def test_scipy_minimize():
    x0 = np.array([0.5, 0])

    def rosen(x):
        """The Rosenbrock function"""
        return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)

    linear_constraint = LinearConstraint([[1, 2], [2, 1]], [-np.inf, 1], [1, 1])
    res = minimize(
        rosen,
        x0,
        method="trust-constr",
        jac="2-point",
        hess=SR1(),
        constraints=[linear_constraint],
        options={"verbose": 1},
        bounds=None,
    )
    print(res.x)


if __name__ == "__main__":
    # test_scipy_minimize()
    test_qp()
