######################################################################################################################
# This file contains code from https://github.com/msesia/arc.git
#                   by "Classification with Valid and Adaptive Coverage" (Romano et. al 2020)
#
######################################################################################################################
import numpy as np
from utils import construct_prediction_sets, get_classes_by_dataset_name


class SplitConformal:
    def __init__(self, method_name, dataset, X_calib, y_calib, alpha, random_state=2020, epsilon=None, **kwargs):
        self.alpha = alpha
        # generate nonconformity scores
        from transporter.utils import get_aps_score
        if len(X_calib.shape) == 1 or X_calib.shape[1] == 1:
            self.s_calib = X_calib.reshape((-1,))
            self.is_aps_score = False
        else:
            self.s_calib = get_aps_score(X_calib, y_calib, epsilon=epsilon)
            self.is_aps_score = True
        # sort scores
        s_calib_ind = np.argsort(self.s_calib)
        self.s_calib = self.s_calib[s_calib_ind]
        self.y_calib = y_calib[s_calib_ind]
        self.alpha = alpha

        # support list of alphas
        self.class_conditional = kwargs.get('class_conditional', False)
        if self.class_conditional:
            self.classes = np.unique(y_calib)
        else:
            self.classes, _ = get_classes_by_dataset_name(dataset, X_calib.shape)
        self.quantiles = self.compute_quantiles(alpha_list=alpha, s_sorted=self.s_calib, y=self.y_calib, class_conditional=self.class_conditional, is_aps_score=self.is_aps_score)

    def predict(self, X, random_state=2020, epsilon=None):
        S_hat = construct_prediction_sets(X, self.alpha, self.quantiles, classes=self.classes, class_conditional=self.class_conditional,
                                          is_aps_score=self.is_aps_score, epsilon=epsilon, random_state=random_state)
        return S_hat

    def compute_quantiles(self, alpha_list, s_sorted, y, class_conditional=False, is_aps_score=False):
        quantiles = {}
        N = len(s_sorted)
        if class_conditional:
            classes = np.unique(y)
            all_quantiles = {}
            for class_ in classes:
                s_ = s_sorted[y == class_].copy()
                y_ = y[y == class_].copy()
                all_quantiles[class_] = self.compute_quantiles(alpha_list, s_, y_, class_conditional=False, is_aps_score=is_aps_score)
            return all_quantiles
        for alpha_ in alpha_list:
            level_adjusted = (1.0-alpha_)*(1.0+1.0/float(N))
            if level_adjusted > 1:
                quantile = 1 if is_aps_score else np.inf
            else:
                quantile = s_sorted[int(np.ceil(level_adjusted*N)) - 1]
            quantiles[alpha_] = quantile
        return quantiles


class MajSplitConformal(SplitConformal):
    def __init__(self, method_name, dataset, X_calib, y_calib, X_maj_calib, y_maj_calib, alpha, random_state=2020,
                 epsilon=None, epsilon_maj=None, **kwargs):
        super().__init__(method_name=method_name, dataset=dataset, X_calib=X_maj_calib, y_calib=y_maj_calib,
                         alpha=alpha, random_state=random_state, epsilon=epsilon_maj, **kwargs)
