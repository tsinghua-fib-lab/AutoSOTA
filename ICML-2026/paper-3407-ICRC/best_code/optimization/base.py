import numpy as np
import cvxpy as cp

class BaseOptimization:

    def __init__(self, y_center):
        self.y_center = y_center

    def solve(self, y_arr):
        '''
        Args:
        - y_arr: [ nbatch, nY ]
        '''
        raise NotImplementedError

    def robust_solve(self, lam):
        '''
        Args:
        - lam: scalar, regularization parameter
        '''
        raise NotImplementedError

    def obj(self, z_arr, y_arr):
        '''
        Args:
        - z_arr: [ nbatch, nZ ]
        - y_arr: [ nbatch, nY ]
        '''
        raise NotImplementedError