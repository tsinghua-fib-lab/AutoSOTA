import numpy as np
import cvxpy as cp
from optimization.base import BaseOptimization

class LinearProgramming(BaseOptimization):
    '''
    Assuming nZ = nY
    '''
    def __init__(self, A, b, y_center):
        '''
        Args:
        - c: [ nZ ], objective function
        - A: [ nC, nZ ], constraint matrix
        - b: [ nC ], constraint vector
        - y_center: [ nY ], center of the uncertainty set
        '''
        super().__init__(y_center)
        self.A = A
        self.b = b

    def solve(self, y_arr):
        '''
        Args:
        - y_arr: [ nbatch, nY ]
        '''
        n       = y_arr.shape[0]
        y_dim   = y_arr.shape[1]
        z_var       = cp.Variable(y_dim)
        y_param     = cp.Parameter(y_dim)
        objective   = cp.Minimize(z_var @ y_param) # [ nbatch ]
        constraints = [ self.A @ z_var <= self.b ]
        prob = cp.Problem(objective, constraints)

        z_arr = np.zeros_like(y_arr)
        for i in range(n):
            y_param.value = y_arr[i]
            prob.solve()
            z_arr[i] = z_var.value # [ nZ ]
        return z_arr

    def robust_solve(self, lam):
        z_var = cp.Variable(self.A.shape[1])              # decision z in R^{nZ}
        objective = cp.Minimize(self.y_center @ z_var + lam * cp.norm1(z_var))
        constraints = [self.A @ z_var <= self.b]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        return z_var.value
    
    def obj(self, z_arr, y_arr):
        '''
        Args:
        - z_arr: [ nbatch, nZ ]
        - y_arr: [ nbatch, nY ]
        '''
        return np.sum(z_arr * y_arr, axis=1) # [ nbatch ]

# helper function
def circle_as_polytope(R=1.0, m=32):
    '''
    Approximate a circle with m-sided polygon via Az <= b
    NOTE: example: A, b = circle_as_polytope(R=1.0, m=16)
    '''
    thetas = np.linspace(0, 2*np.pi, m, endpoint=False)
    A = np.c_[np.cos(thetas), np.sin(thetas)]  # [m, 2]
    b = np.ones(m) * R
    return A, b