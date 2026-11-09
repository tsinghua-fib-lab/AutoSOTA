import numpy as np
from sklearn.metrics import mean_squared_error
from .base import MetricTemplate


class RMSE(MetricTemplate):
    '''
    Root mean square error
    '''
    def __init__(self):
        super().__init__(maximize=False)

    def _test(self, target, approx):
        return np.sqrt(mean_squared_error(target, approx))
