# Class that represents a conic problem

# The conic optimization problem is
# ========================================
# minimize c^T x
# subject to Ax + s = b
#            s in K subset R^m
#            x in R^n
# ========================================
# where K is a closed convex proper cone

# The dual problem is
# ========================================
# minimize b^T y
# subject to - A^T y + r = c
#            y in K^* subset R^m
#            r in {0}^n
# ========================================
# where K^* is the dual cone of K

# The problem data is given by A, b, c
# The problem "structure" is given by the cone K

import numpy as np
import clarabel
from scipy import sparse
from trainable_ot_dro.cones import *

class ConicProblem:
    def __init__(self):
        self.settings = clarabel.DefaultSettings()
        self.settings.verbose = True
        self.settings.time_limit = 10

    def solve(self, A, b, c, cone, verbose=True):
        # in our conic problems, P = 0, q = c, A = A, b = b, cones = cone
        # where cone is usually a cartesian product of smaller convex cones

        # check that A is in R^(m x n), b is in R^m, c is in R^n
        assert A.shape[0] == b.shape[0]
        assert A.shape[1] == c.shape[0]
        # check also that A is indeed a matrix
        assert len(A.shape) == 2
        # check that b and c are vectors
        assert (len(b.shape) == 1 or b.shape[1] == 1)
        assert (len(c.shape) == 1 or c.shape[1] == 1)

        n = A.shape[1]
        m = A.shape[0]

        # create P as zero matrix
        P = np.zeros((n, n))
        P = sparse.csc_matrix(P)
        P = sparse.triu(P).tocsc()
        q = np.array(c).reshape(-1, 1)
        b = np.array(b).reshape(-1, 1)
        A = sparse.csc_matrix(A)
        cones = cone.clarabel_cone

        tmp_settings = self.settings
        tmp_settings.verbose = verbose

        solver = clarabel.DefaultSolver(P, q, A, b, cones, tmp_settings)

        solution = solver.solve()

        return solution

    # computes the derivative of the solution map, using the approach in
    # Bolte et al. (2021) "Nonsmooth Implicit Differentiation for Machine Learning and Optimization"
    def solve_and_derivative(self, A, b, c, cone, verbose=True):
        # first we have to solve the conic problem
        solution = self.solve(A, b, c, cone, verbose=verbose)
        x = np.array(solution.x).reshape(-1, 1)
        y = np.array(solution.z).reshape(-1, 1)
        s = np.array(solution.s).reshape(-1, 1)

        m, n = A.shape
        N = m + n

        xCone = FreeCone(dim=n)
        yCone = cone.dual_cone()
        bigK = CartesianProductCone([xCone, yCone])
        # bigK = R^n x K*

        z = np.concatenate([x, y - s])
        alpha, beta = np.split(z, [n])
        # alpha = x
        # beta = y - s

        pi_z = bigK.project(z)

        I_N = np.eye(N)

        DPi_z = bigK.derivative(z)

        Q = np.zeros((N, N))
        Q[0:n, 0:n]         = 0.0
        Q[0:n, n:n+m]       = A.T
        Q[n:n+m, 0:n]       = -A
        Q[n:n+m, n:n+m]     = 0.0

        M = ( (Q - I_N) @ DPi_z + I_N )
        MT = M.T
        # convert to numpy array
        MT_np = np.array(MT)
        #convert to numpy array
        M_np = np.array(M)

        def dS_alt(dA, db, dc):
            # (a) Build dQ from dA
            dQ = np.zeros((N, N))
            dQ[0:n, n:n+m]   = dA.T
            dQ[n:n+m, 0:n]   = - dA

            dV = np.concatenate([dc, db], axis=0).reshape(N, 1)

            g = dQ @ pi_z + dV
            g_np = np.array(g)

            rhs = - g_np
            # solve the system M * dz = rhs for dz
            dz = np.linalg.solve(M_np, rhs)
            # sol = sparse.linalg.lsqr(M_np, -g_np)
            # dz = sol[0].reshape(-1, 1)

            dalpha, dbeta = np.split(dz, [n])

            dalpha = dalpha.reshape(-1, 1)
            dbeta = dbeta.reshape(-1, 1)

            dx = dalpha
            dy = cone.dual_cone().derivative(beta) @ dbeta
            ds = dy - dbeta

            return dx, dy, ds

        def dST_alt(dx, dy, ds):
            dx = dx.reshape(-1, 1)
            dy = dy.reshape(-1, 1)
            ds = ds.reshape(-1, 1)

            dBeta = cone.dual_cone().derivative(beta).T @ (dy + ds)  - ds
            dAlpha = dx

            dz = np.concatenate([dAlpha, dBeta], axis=0)

            # sol = sparse.linalg.lsqr(MT_np, -dz)
            # g = sol[0].reshape(-1, 1)

            rhs = - dz
            # solve the system MT * dg = rhs for dg
            g = np.linalg.solve(MT_np, rhs)

            dQ = g @ pi_z.T
            dV = g

            dQ_12 = dQ[:n, n:n+m]
            dQ_21 = dQ[n:n+m, :n]
            dA = (dQ_12.T - dQ_21)
            # dA = dQ_12.T

            # V = [c, b], so
            dc = dV[:n, :].squeeze()
            db = dV[n:, :].squeeze()

            return (dA, db, dc)

        result = {}
        result['derivative'] = dS_alt
        result['derivative_adjoint'] = dST_alt
        result['solution'] = solution

        return result

    # normalized residual map
    def normalized_residual_map(obj, x, y, s, A, b, c, cone):
        n = A.shape[1]
        m = A.shape[0]
        u = np.concatenate([x, y, np.ones((1, 1))])
        v = np.concatenate([np.zeros((n, 1)), s, np.zeros((1, 1))])
        z = u - v
        xCone = FreeCone(dim=n)
        yCone = cone.dual_cone()
        tauCone = NonnegativeCone(dim=1)
        bigK = CartesianProductCone([xCone, yCone, tauCone])
        pi_zw = bigK.project(z) / np.abs(z[m+n])
        N = m+n+1
        I_N = np.eye(N)

        Q = np.zeros((N, N))
        # Q_11 is zero in R^(n x n)
        Q[0:n, 0:n] = np.zeros((n, n))
        # Q_12 is dA^T in R^(n x m)
        Q[0:n, n:n+m] = A.T
        # Q_13 is dc in R^(n x 1)
        Q[0:n, n+m] = c.reshape(-1)
        # Q_21 is -dA in R^(m x n)
        Q[n:n+m, 0:n] = -A
        # Q_22 is zero in R^(m x m)
        Q[n:n+m, n:n+m] = np.zeros((m, m))
        # Q_23 is db in R^(m x 1)
        Q[n:n+m, n+m] = b.reshape(-1)
        # Q_31 is -dc^T in R^(1 x n)
        Q[n+m, 0:n] = -c.T
        # Q_32 is -db^T in R^(1 x m)
        Q[n+m, n:n+m] = -b.T
        # Q_33 is zero in R^(1 x 1)
        Q[n+m, n+m] = 0

        # print the shape of Q and pi_zw as well as z
        NR = (Q - I_N) @ pi_zw + z / np.abs(z[m+n])
        return NR
