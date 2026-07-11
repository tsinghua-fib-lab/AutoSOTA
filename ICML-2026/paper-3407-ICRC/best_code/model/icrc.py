import numpy as np

class InverseConformalRiskControl:
    def __init__(self, opt, y_center):
        '''
        Args:
        - opt:      optimization problem
        - y_center:   [ nY ], center of the uncertainty set
        '''
        self.opt = opt
        self.y_center = y_center

    def compute(self, y_arr, lam):
        '''
        Compute and store the regret and miscoverage
        Args:
        - y_arr:    [ nbatch, nY ]
        - lam:      scalar, robustness parameter
        '''
        # compute regret
        z_arr    = self.opt.solve(y_arr)                                                # [ nbatch, nZ ]
        z_robust = self.opt.robust_solve(lam)[None, :] * np.ones((y_arr.shape[0], 1))   # [ nbatch, nZ ]
        regret   = self.opt.obj(z_robust, y_arr) - self.opt.obj(z_arr, y_arr)           # [ nbatch ]
        
        # compute miscoverage
        miscoverage = np.linalg.norm(self.y_center[None, :] - y_arr, axis=1, ord=np.inf) > lam # [ nbatch ]

        # store results
        self.regret = regret
        self.miscoverage = miscoverage
        self.z_arr = z_arr
        self.y_arr = y_arr
        return None

    def estimate(self, B=None, output_mc=False, offset=0.0):
        '''
        Args:
        - B:        upper bound of the regret 
        - output_mc: bool, whether to output the Monte Carlo estimate
        - offset:    scalar, offset to B
        '''
        # initialize parameters
        if B is None:
            B = self.regret.max() + offset
        n = self.y_arr.shape[0]

        # conformal correction
        if output_mc:
            regret_hat = self.regret.mean()               # scalar
            miscoverage_hat = self.miscoverage.mean()     # scalar 
        else:
             regret_hat = self.regret.mean() * n / (n + 1) + (B + offset) / (n + 1)               # scalar
             miscoverage_hat = self.miscoverage.mean() * n / (n + 1) + (1 + offset) / (n + 1)     # scalar
        return regret_hat, miscoverage_hat