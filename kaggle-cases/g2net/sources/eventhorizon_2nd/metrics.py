import numpy as np
from sklearn.metrics import roc_auc_score
from kuma_utils.metrics import MetricTemplate


class AUC(MetricTemplate):
    '''
    Area under ROC curve
    '''
    def __init__(self):
        super().__init__(maximize=True)

    def _test(self, target, approx):
        if len(approx.shape) == 1:
            pass
        elif approx.shape[1] == 1:
            approx = np.squeeze(approx)
        elif approx.shape[1] == 2:
            approx = approx[:, 1]
        else:
            raise ValueError(f'Invalid approx shape: {approx.shape}')
        target = np.round(target)
        return roc_auc_score(target, approx)
