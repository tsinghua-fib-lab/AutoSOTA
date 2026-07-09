# Files that contains the class ConvexCone
# as well as classes that inherit from ConvexCone
# - FreeCone
# - ZeroCone
# - NonnegativeCone
# - SecondOrderCone

import numpy as np
import scipy as sp
import clarabel

class ConvexCone:
    # A cone is a closed convex proper cone
    # The cone is defined by the following properties:
    # 1. Contains the origin
    # 2. Is closed under nonnegative scaling
    # 3. Is closed under addition
    # 4. Is closed under taking convex combinations
    # The cone is represented by the set K
    # The dual cone is represented by the set K^*

    # we add the corresponding clarabel cone as members of the class
    def __init__(self, dim=None):
        self.dim = dim
        self.clarabel_cone = None

    def project(self, x):
        raise NotImplementedError

    def contains(self, x):
        raise NotImplementedError

    def dual_cone(self):
        raise NotImplementedError

    def derivative(self, x):
        raise NotImplementedError


class FreeCone(ConvexCone):
    # The free cone is the set R^n
    def __init__(self, dim):
        super().__init__(dim=dim)

        # clarabel doesn't have a free cone
        self.clarabel_cone = None

    def project(self, x):
        # The projection is
        # Pi(x) = x
        return x.reshape(-1, 1)

    def contains(self, x):
        return True

    def dual_cone(self):
        return ZeroCone(dim=self.dim)

    def derivative(self, x):
        return np.eye(self.dim)


class ZeroCone(ConvexCone):
    # The zero cone is the set {0}
    def __init__(self, dim):
        super().__init__(dim=dim)

        self.clarabel_cone = clarabel.ZeroConeT(self.dim)

    def project(self, x):
        # The projection is
        # Pi(x) = 0
        return np.zeros(self.dim).reshape(-1, 1)

    def contains(self, x):
        return np.linalg.norm(x) < 1e-15

    def dual_cone(self):
        return FreeCone(dim=self.dim)

    def derivative(self, x):
        deriv = np.zeros((self.dim, self.dim))
        assert self.dim == deriv.shape[0]
        assert x.shape[0] == self.dim
        return deriv


class NonnegativeCone(ConvexCone):
    # The nonnegative cone is the set R^n_+
    def __init__(self, dim):
        super().__init__(dim=dim)

        self.clarabel_cone = clarabel.NonnegativeConeT(self.dim)

    def project(self, x):
        # The projection is
        # Pi(x)_i = max(0, x_i)
        return np.maximum(0, x).reshape(-1, 1)

    def contains(self, x):
        return np.all(x >= 0)

    def dual_cone(self):
        # Self-dual cone
        return NonnegativeCone(dim=self.dim)

    def derivative(self, x):
        # for x==0, the derivative is not defined
        # we throw an error in this case
        if np.any(x == 0):
            raise ValueError("The derivative is not defined at x=0")
        # deriv is a diagonal matrix
        # (i, i)-th entry is one if x_i >= 0
        # (i, i)-th entry is zero if x_i < 0
        zeroonevec = np.where(x >= 0, 1, 0)
        deriv = np.diag(zeroonevec.reshape(-1))

        assert self.dim == deriv.shape[0], "self.dim: {}, deriv.shape: {}".format(self.dim, deriv.shape)
        assert x.shape[0] == self.dim, "x.shape: {}, self.dim: {}".format(x.shape[0], self.dim)
        return deriv


class SecondOrderCone(ConvexCone):
    # The second order cone is the set
    # K = {(t, x) | ||x|| <= t}
    # t is in R and x is in R^n
    def __init__(self, dim):
        # dim should be n+1 (it is the dimension of the vector (t, x))
        super().__init__(dim=dim)

        self.clarabel_cone = clarabel.SecondOrderConeT(self.dim)

    def project(self, z):
        # The projection is defined as
        # Pi(t, x) = (t, x) if ||x|| <= t
        #            (0, 0) if ||x|| <= -t
        #            0.5(t + ||x||) (1, x/||x||) otherwise

        # check that z is given as a vector in R^{n+1}
        assert z.shape[0] == self.dim

        t = z[0].reshape(1, 1)
        x = z[1:].reshape(-1, 1)

        norm_x = np.linalg.norm(x)
        if norm_x <= float(t):
            return np.concatenate([t, x])
        elif norm_x <= -float(t):
            return np.zeros((x.shape[0] + 1, 1))
        else:
            one_arr = np.ones((1,1))
            return 0.5 * (t + norm_x) * np.concatenate([one_arr, x / norm_x])

    def contains(self, z):
        t = z[0]
        x = z[1:]
        return np.linalg.norm(x) <= t

    def dual_cone(self):
        # The Lorentz cone is self-dual
        return SecondOrderCone(dim=self.dim)

    def derivative(self, z):
        t = z[0].reshape(1, 1)
        x = z[1:].reshape(-1, 1)
        # The derivative is undefined at ||x|| = t
        if np.linalg.norm(x) == float(t):
            raise ValueError("The derivative is not defined at ||x|| = t")
        # DPi(t, x) = I if ||x|| < t
        #           = 0 if ||x|| < -t
        #           = 1 / (2 * ||x||) * [||x||, x^T;
        #                                 x, (t + ||x||)I - t*(x*x^T)/||x||^2] otherwise
        n = x.shape[0]
        if np.linalg.norm(x) < float(t):
            deriv = np.eye(n + 1)

        elif np.linalg.norm(x) < -float(t):
            deriv = np.zeros((n + 1, n + 1))
        else:
            a = 1 / (2 * np.linalg.norm(x))
            M_11 = np.linalg.norm(x)
            M_11 = M_11.reshape(1, 1)
            M_12 = x.reshape(1, -1)
            M_21 = x.reshape(-1, 1)
            M_22 = (t + np.linalg.norm(x)) * np.eye(n) - t * np.outer(x, x) / np.linalg.norm(x)**2
            M = np.block([[M_11, M_12], [M_21, M_22]])
            deriv = a * M

        assert self.dim == deriv.shape[0]
        assert z.shape[0] == self.dim
        return deriv


# Cartesian product of cones
class CartesianProductCone(ConvexCone):
    # The Cartesian product of cones is the set
    # K = K_1 x K_2 x ... x K_n
    # where each K_i is a cone and has a dimension of dim_i
    # the dimension of the Cartesian product is dim = dim_1 + dim_2 + ... + dim_n
    def __init__(self, cones):
        self.cones = cones
        # add up the dimensions of the cones
        dim = sum([cone.dim for cone in cones])

        super().__init__(dim=dim)

        # there is no clarabel cone for the Cartesian product of smaller
        # cones, but for solving clarabel expects a list of clarabel cones
        # so here we just store the list of clarabel cones
        self.clarabel_cone = [cone.clarabel_cone for cone in cones]

    def project(self, x):
        # The projection is
        # Pi(x) = (Pi_1(x_1), Pi_2(x_2), ..., Pi_n(x_n))
        # where each projection has the dimension of the corresponding cone
        # x is given as a vector in R^{dim_1 + dim_2 + ... + dim_n}
        proj = []
        start = 0
        for cone in self.cones:
            proj.append(cone.project(x[start:start+cone.dim]))
            start += cone.dim
        return np.concatenate(proj, axis=0)

    def contains(self, x):
        start = 0
        for cone in self.cones:
            if not cone.contains(x[start:start+cone.dim]):
                return False
            start += cone.dim
        return True

    def dual_cone(self):
        # The dual cone is the Cartesian product of the dual cones
        return CartesianProductCone([cone.dual_cone() for cone in self.cones])

    def derivative(self, x):
        # The derivative is the block diagonal matrix of the derivatives of the cones
        D = []
        start = 0
        for cone in self.cones:
            D.append(cone.derivative(x[start:start+cone.dim]))
            start += cone.dim
        deriv = sp.linalg.block_diag(*D)
        assert self.dim == deriv.shape[0]
        assert x.shape[0] == self.dim
        return deriv
