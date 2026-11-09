import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
from .base import MetricTemplate


class AUC(MetricTemplate):
    '''
    Area under ROC curve
    '''
    def __init__(self):
        super().__init__(maximize=True)

    def _test(self, target, approx):
        if len(approx.shape) == 1:
            approx = approx
        elif approx.shape[1] == 1:
            approx = np.squeeze(approx)
        elif approx.shape[1] == 2:
            approx = approx[:, 1]
        else:
            raise ValueError(f'Invalid approx shape: {approx.shape}')
        return roc_auc_score(target, approx)


class Accuracy(MetricTemplate):
    '''
    Accuracy
    '''
    def __init__(self):
        super().__init__(maximize=True)

    def _test(self, target, approx):
        assert(len(target) == len(approx))
        target = np.asarray(target, dtype=int)
        approx = np.asarray(approx, dtype=float)
        if len(approx.shape) == 1:
            approx = approx
        elif approx.shape[1] == 1:
            approx = np.squeeze(approx)
        elif approx.shape[1] >= 2:
            approx = np.argmax(approx, axis=1)
        approx = approx.round().astype(int)
        return np.mean((target == approx).astype(int))


class SeWithFixedSp(MetricTemplate):
    '''
    Maximize sensitivity with fixed specificity
    '''
    def __init__(self, sp=0.9):
        super().__init__(maximize=True)
        self.sp = 0.9

    def _get_threshold(self, target, approx):
        tn_idx = (target == 0)
        p_tn = np.sort(approx[tn_idx])

        return p_tn[int(len(p_tn) * self.sp)]

    def _test(self, target, approx):
        if not isinstance(target, np.ndarray):
            target = np.array(target)
        if not isinstance(approx, np.ndarray):
            approx = np.array(approx)

        if len(approx.shape) == 1:
            pass
        elif approx.shape[1] == 1:
            approx = np.squeeze(approx)
        elif approx.shape[1] == 2:
            approx = approx[:, 1]
        else:
            raise ValueError(f'Invalid approx shape: {approx.shape}')

        if min(approx) < 0:
            approx -= min(approx)  # make all values positive
        target = target.astype(int)
        thres = self._get_threshold(target, approx)
        pred = (approx > thres).astype(int)
        tn, fp, fn, tp = confusion_matrix(target, pred).ravel()
        se = tp / (tp + fn)
        sp = tn / (tn + fp)

        return se


class QWK(MetricTemplate):
    '''
    Quandric Weight Kappa
    '''
    def __init__(self, max_rat):
        super().__init__(maximize=True)
        self.max_rat = max_rat

    def _test(self, target, approx):
        assert(len(target) == len(approx))
        target = np.asarray(target, dtype=int)
        approx = np.asarray(approx, dtype=float)
        if len(approx.shape) == 1:
            approx = approx
        elif approx.shape[1] == 1:
            approx = np.squeeze(approx)
        elif approx.shape[1] >= 2:
            approx = np.argmax(approx, axis=1)
        approx = np.clip(approx.round(), 0, self.max_rat-1).astype(int)

        hist1 = np.zeros((self.max_rat+1, ))
        hist2 = np.zeros((self.max_rat+1, ))

        o = 0
        for k in range(target.shape[0]):
            i, j = target[k], approx[k]
            hist1[i] += 1
            hist2[j] += 1
            o += (i - j) * (i - j)

        e = 0
        for i in range(self.max_rat + 1):
            for j in range(self.max_rat + 1):
                e += hist1[i] * hist2[j] * (i - j) * (i - j)

        e = e / target.shape[0]

        return 1 - o / e
