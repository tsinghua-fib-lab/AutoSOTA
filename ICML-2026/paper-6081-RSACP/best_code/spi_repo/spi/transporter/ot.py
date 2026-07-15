######################################################################################################################
# This file contains code from https://github.com/msesia/arc.git
#                   by "Classification with Valid and Adaptive Coverage" (Romano et. al 2020)
#
######################################################################################################################
import numpy as np
from .utils import get_aps_score, get_nearest_neighbor, find_k_nearest_neighbors_cdf
from utils import init_dict_of_lists, construct_prediction_sets, get_classes_by_dataset_name


class SPPredictiveInference:
    def __init__(self, method_name, dataset, X_calib, y_calib, alpha, random_state=2020,
                 X_maj_calib=None, y_maj_calib=None, epsilon=None, epsilon_maj=None,
                 beta=0.1, **kwargs):
        self.epsilon_maj = epsilon_maj
        self.epsilon_calib = epsilon
        self.method_name = method_name
        self.alpha = alpha
        self.random_state = random_state
        self.n_classes = 1 if len(X_calib.shape) == 1 else X_calib.shape[1]
        self.beta = beta

        # generate nonconformity scores
        if len(X_calib.shape) == 1 or X_calib.shape[1] == 1:
            self.s_calib = np.abs(X_calib.reshape((-1,)) - y_calib.reshape((-1,)))
            self.s_maj = np.abs(X_maj_calib.reshape((-1,)) - y_maj_calib.reshape((-1,)))
            self.is_aps_score = False
        else:
            self.s_calib = get_aps_score(X_calib, y_calib, epsilon=epsilon)
            self.s_maj = get_aps_score(X_maj_calib, y_maj_calib, epsilon=epsilon_maj)
            self.is_aps_score = True
        # sort majority scores
        s_maj_ind = np.argsort(self.s_maj)
        self.s_maj_sorted = self.s_maj[s_maj_ind]
        del self.s_maj
        self.y_maj = y_maj_calib[s_maj_ind]
        self.epsilon_maj = self.epsilon_maj[s_maj_ind]
        # sort minority scores
        s_calib_ind = np.argsort(self.s_calib)
        self.s_calib = self.s_calib[s_calib_ind]
        self.y_calib = y_calib[s_calib_ind]
        self.epsilon_calib = self.epsilon_calib[s_calib_ind]

        self.class_conditional = kwargs.get('class_conditional', False)
        if self.class_conditional:
            self.class_conditional_maj = True if dataset.endswith('_gen') else False
            self.classes = np.unique(y_calib)
        else:
            self.classes, _ = get_classes_by_dataset_name(dataset, X_calib.shape)
        self.update_transformed_calib_scores()

        self.quantiles = self.compute_ot_quantiles(alpha_list=alpha, s_calib=self.s_calib, y_calib=self.y_calib, s_maj_sorted=self.s_maj_sorted, y_maj=self.y_maj,
                                                   class_conditional=self.class_conditional, is_aps_score=self.is_aps_score)

    def predict(self, X, random_state=2020, epsilon=None):
        S_hat_plus = construct_prediction_sets(X, self.alpha, self.quantiles, classes=self.classes, class_conditional=self.class_conditional,
                                               is_aps_score=self.is_aps_score, epsilon=epsilon, random_state=random_state)
        return S_hat_plus

    def compute_ot_quantiles(self, alpha_list, s_calib, y_calib, s_maj_sorted, y_maj, class_conditional=False, is_aps_score=False):
        quantiles = {}
        N = len(s_maj_sorted)
        if class_conditional:
            classes = np.unique(y_calib)
            all_quantiles = {}
            for class_ in classes:
                s_ = s_calib[y_calib == class_].copy()
                y_ = y_calib[y_calib == class_].copy()
                if self.class_conditional_maj:
                    s_maj_ = s_maj_sorted[y_maj == class_].copy()
                    y_maj_ = y_maj[y_maj == class_].copy()
                else:
                    s_maj_ = s_maj_sorted.copy()
                    y_maj_ = y_maj.copy()
                all_quantiles[class_] = self.compute_ot_quantiles(alpha_list, s_, y_, s_maj_, y_maj_, class_conditional=False, is_aps_score=is_aps_score)
            return all_quantiles
        sorted_min_T = self.get_transportation(s_min=np.array(range(len(s_calib) + 1)), s_maj_sorted=s_maj_sorted)
        for alpha_ in alpha_list:
            level_adjusted = (1.0 - alpha_) * (1.0 + 1.0 / float(N))
            # plus
            level_adjusted_plus = (1.0 - alpha_) * (
                        1.0 + 1.0 / float(N) + 1.0 / float(N * (1 - alpha_)))
            if level_adjusted_plus > 1:
                quantile_maj_plus = 1 if is_aps_score else np.inf
            else:
                quantile_maj_plus = s_maj_sorted[int(np.ceil(level_adjusted_plus * N)) - 1]
            r_plus_indices = [np.max(np.where(row > 0)[0]) + 1 if np.any(row > 0) else -1 for row in sorted_min_T]
            r_plus = np.max(np.where(np.array(r_plus_indices) <= np.ceil(N * level_adjusted))[0]) + 1
            r_minus_indices = [np.min(np.where(row > 0)[0]) + 1 if np.any(row > 0) else -1 for row in sorted_min_T]
            r_minus = np.max(np.where(np.array(r_minus_indices) <= np.ceil(N * level_adjusted))[0]) + 1
            if r_minus > len(s_calib):
                s_r_minus = 1 if is_aps_score else np.inf
            else:
                s_r_minus = s_calib[r_minus - 1]
            if r_plus > len(s_calib):
                s_r_plus = 1 if is_aps_score else np.inf
            else:
                s_r_plus = s_calib[r_plus - 1]
            quantile_maj_plus = np.min([quantile_maj_plus, s_r_minus])
            quantile_maj_plus = np.max([quantile_maj_plus, s_r_plus])
            quantiles[alpha_] = quantile_maj_plus
        return quantiles

    def update_transformed_calib_scores(self):
        pass

    def get_transportation(self, s_min, s_maj_sorted):
        from .transportations import buckets_by_hg_dist
        T = buckets_by_hg_dist(s_min, s_maj_sorted, weights=True, by_quantiles=True, beta=self.beta)
        return T

    def get_transported_score(self, scores, mapping, test_score, sorted_majority_scores, minority_scores):
        assert scores.shape == mapping.shape
        t_scores = scores[mapping > 0]
        # T(s(X_m+1,y) <= s_m+1
        filtered_t_scores = t_scores[t_scores <= test_score]
        if len(filtered_t_scores) == 0:
            t_score = t_scores[0]
        else:
            t_score = get_nearest_neighbor(filtered_t_scores, test_score, sorted_majority_scores, minority_scores)
        return t_score


class SPPredictiveInferenceClustered(SPPredictiveInference):
    def __init__(self, method_name, dataset, X_calib, y_calib, alpha, random_state=2020,
                 X_maj_calib=None, y_maj_calib=None, epsilon=None, epsilon_maj=None,
                 beta=0.1, **kwargs):
        self.k = kwargs['k']
        self.criterion = kwargs['dist_criterion']
        super().__init__(method_name, dataset, X_calib, y_calib, alpha, random_state, X_maj_calib, y_maj_calib, epsilon, epsilon_maj, beta, **kwargs)

    def update_transformed_calib_scores(self):
        # find the k nearest neighbors using only the train data
        if self.class_conditional:
            clusters_per_class = {}
            dist_per_class = {}
            classes = np.unique(self.y_calib)
            for class_ in classes:
                s_ = self.s_calib[self.y_calib == class_]
                k_labels, _, _ = find_k_nearest_neighbors_cdf(k=self.k, s=s_, s_maj=self.s_maj_sorted, y_maj=self.y_maj, criterion=self.criterion)
                clusters_per_class[class_] = k_labels
            self.clusters_per_class = clusters_per_class
            self.dist_per_class = dist_per_class
        else:
            k_labels, _, _ = find_k_nearest_neighbors_cdf(k=self.k, s=self.s_calib, s_maj=self.s_maj_sorted, y_maj=self.y_maj, criterion=self.criterion)
            # update s_maj to include only labels from k_labels
            mask = np.isin(self.y_maj, k_labels)
            self.s_maj_sorted = self.s_maj_sorted[mask]
            self.y_maj = self.y_maj[mask]
            self.epsilon_maj = self.epsilon_maj[mask]

    def compute_ot_quantiles(self, alpha_list, s_calib, y_calib, s_maj_sorted, y_maj, class_conditional=False, is_aps_score=False):
        quantiles = {}
        N = len(s_maj_sorted)
        if class_conditional:
            classes = np.unique(y_calib)
            all_quantiles = {}
            for class_ in classes:
                s_ = s_calib[y_calib == class_].copy()
                y_ = y_calib[y_calib == class_].copy()
                # update s_maj to include only labels from the cluster of class_
                k_labels = self.clusters_per_class[class_]
                mask = np.isin(self.y_maj, k_labels)
                s_maj_ = self.s_maj_sorted[mask]
                y_maj_ = self.y_maj[mask]
                all_quantiles[class_] = self.compute_ot_quantiles(alpha_list, s_, y_, s_maj_, y_maj_, class_conditional=False, is_aps_score=is_aps_score)
            return all_quantiles
        sorted_min_T = self.get_transportation(s_min=np.array(range(len(s_calib) + 1)), s_maj_sorted=s_maj_sorted)
        for alpha_ in alpha_list:
            level_adjusted = (1.0 - alpha_) * (1.0 + 1.0 / float(N))
            # plus
            level_adjusted_plus = (1.0 - alpha_) * (
                        1.0 + 1.0 / float(N) + 1.0 / float(N * (1 - alpha_)))
            if level_adjusted_plus > 1:
                quantile_maj_plus = 1 if is_aps_score else np.inf
            else:
                quantile_maj_plus = s_maj_sorted[int(np.ceil(level_adjusted_plus * N)) - 1]
            r_plus_indices = [np.max(np.where(row > 0)[0]) + 1 if np.any(row > 0) else -1 for row in sorted_min_T]
            r_plus = np.max(np.where(np.array(r_plus_indices) <= np.ceil(N * level_adjusted))[0]) + 1
            r_minus_indices = [np.min(np.where(row > 0)[0]) + 1 if np.any(row > 0) else -1 for row in sorted_min_T]
            r_minus = np.max(np.where(np.array(r_minus_indices) <= np.ceil(N * level_adjusted))[0]) + 1
            if r_minus > len(s_calib):
                s_r_minus = 1 if is_aps_score else np.inf
            else:
                s_r_minus = s_calib[r_minus - 1]
            if r_plus > len(s_calib):
                s_r_plus = 1 if is_aps_score else np.inf
            else:
                s_r_plus = s_calib[r_plus - 1]
            quantile_maj_plus = np.min([quantile_maj_plus, s_r_minus])
            quantile_maj_plus = np.max([quantile_maj_plus, s_r_plus])
            quantiles[alpha_] = quantile_maj_plus
        return quantiles


class SPPredictiveInferenceRegression(SPPredictiveInference):
    def __init__(self, method_name, dataset, X_calib, y_calib, alpha, random_state=2020,
                 X_maj_calib=None, y_maj_calib=None, epsilon=None, epsilon_maj=None,
                 beta=0.1, **kwargs):
        self.method_name = method_name
        self.alpha = alpha
        self.random_state = random_state
        self.beta = beta

        # generate nonconformity scores
        self.is_cqr_score = False
        if len(X_calib.shape) == 1 or X_calib.shape[1] == 1:    # residual score
            self.s_calib = np.abs(X_calib.reshape((-1,)) - y_calib.reshape((-1,)))
            self.s_maj = np.abs(X_maj_calib.reshape((-1,)) - y_maj_calib.reshape((-1,)))
        elif X_calib.shape[1] == 2:   # cqr
            self.is_cqr_score = True
            low_err = X_calib[:, 0].reshape((-1,)) - y_calib.reshape((-1,))
            upp_err = y_calib.reshape((-1,)) - X_calib[:, 1].reshape((-1,))
            assert low_err.shape == upp_err.shape
            self.s_calib = np.maximum(low_err, upp_err)
            low_err_maj = X_maj_calib[:, 0].reshape((-1,)) - y_maj_calib.reshape((-1,))
            upp_err_maj = y_maj_calib.reshape((-1,)) - X_maj_calib[:, 1].reshape((-1,))
            assert low_err_maj.shape == upp_err_maj.shape
            self.s_maj = np.maximum(low_err_maj, upp_err_maj)
        else:
            raise ValueError(f'Calibration for regression task for the following pred shape is not supported - X_calib.shape = {X_calib.shape}.')
        # sort majority scores
        s_maj_ind = np.argsort(self.s_maj)
        self.s_maj_sorted = self.s_maj[s_maj_ind]
        del self.s_maj
        self.y_maj = y_maj_calib[s_maj_ind]
        # sort minority scores
        s_calib_ind = np.argsort(self.s_calib)
        self.s_calib = self.s_calib[s_calib_ind]
        self.y_calib = y_calib[s_calib_ind]

        self.update_transformed_calib_scores()

        self.quantiles = self.compute_ot_quantiles(alpha_list=alpha, s_calib=self.s_calib, y_calib=self.y_calib, s_maj_sorted=self.s_maj_sorted, y_maj=self.y_maj,
                                                   class_conditional=False, is_aps_score=False)

    def predict(self, X, random_state=2020, **kwargs):
        intervals_hat = init_dict_of_lists(keys=self.alpha)
        for i in range(len(X)):
            for alpha_ in self.alpha:
                if self.is_cqr_score:
                    curr_interval = (X[i,0] - self.quantiles[alpha_], X[i,1] + self.quantiles[alpha_])
                else:
                    curr_interval = (X[i] - self.quantiles[alpha_], X[i] + self.quantiles[alpha_])
                intervals_hat[alpha_].append(curr_interval)
        return intervals_hat

