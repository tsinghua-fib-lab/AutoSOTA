######################################################################################################################
# This file contains code from https://github.com/msesia/arc.git
#                   by "Classification with Valid and Adaptive Coverage" (Romano et. al 2020)
#
######################################################################################################################
import numpy as np
from utils import init_dict_of_lists


class SplitConformalRegression:
    def __init__(self, method_name, X_calib, y_calib, alpha, random_state=2020, epsilon=None, **kwargs):
        self.alpha = alpha
        self.is_cqr_score = False
        # generate nonconformity scores - residual score / cqr
        if len(X_calib.shape) == 1 or X_calib.shape[1] == 1:    # residual score
            self.s_calib = np.abs(X_calib.reshape((-1,)) - y_calib.reshape((-1,)))
        elif X_calib.shape[1] == 2:   # cqr
            self.is_cqr_score = True
            low_err = X_calib[:, 0].reshape((-1,)) - y_calib.reshape((-1,))
            upp_err = y_calib.reshape((-1,)) - X_calib[:, 1].reshape((-1,))
            assert low_err.shape == upp_err.shape
            self.s_calib = np.maximum(low_err, upp_err)
        else:
            raise ValueError(f'Calibration for regression task for the following pred shape is not supported - X_calib.shape = {X_calib.shape}.')

        # sort scores
        s_calib_ind = np.argsort(self.s_calib)
        self.s_calib = self.s_calib[s_calib_ind]
        self.y_calib = y_calib[s_calib_ind]

        # support list of alphas
        self.quantiles = {}
        for alpha_ in alpha:
            level_adjusted = (1.0-alpha_)*(1.0+1.0/float(len(X_calib)))
            if level_adjusted > 1:
                quantile = np.inf
            else:
                quantile = self.s_calib[int(np.ceil(level_adjusted*len(X_calib))) - 1]
            self.quantiles[alpha_] = quantile

    def predict(self, X, random_state=2020, **kwargs):
        intervals_hat = init_dict_of_lists(keys=self.quantiles.keys())
        n = X.shape[0]
        for i in range(len(X)):
            for alpha_ in self.quantiles.keys():
                if self.is_cqr_score:
                    curr_interval = (X[i,0] - self.quantiles[alpha_], X[i,1] + self.quantiles[alpha_])
                else:
                    curr_interval = (X[i] - self.quantiles[alpha_], X[i] + self.quantiles[alpha_])
                intervals_hat[alpha_].append(curr_interval)
        return intervals_hat


class MajSplitConformalRegression(SplitConformalRegression):
    def __init__(self, method_name, X_calib, y_calib, X_maj_calib, y_maj_calib, alpha, random_state=2020,
                 epsilon=None, epsilon_maj=None, **kwargs):
        super().__init__(method_name=method_name, X_calib=X_maj_calib, y_calib=y_maj_calib,
                         alpha=alpha, random_state=random_state, epsilon=epsilon_maj, **kwargs)

