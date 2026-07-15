import numpy as np
from scipy.stats import wasserstein_distance, cramervonmises_2samp
from calibration import ProbAccum
from transporter.transportations import get_hg_quantiles_window_matrix


def get_aps_score(X, y, epsilon=None):
    ## Note: The APS score must be computed as CDF - PROB * U, not as 1 - CDF + PROB * U.
    ## The latter does not ensure that if label y has a higher probability than label y_2,
    ## the score of y is necessarily smaller than or equal to the score of y_2.
    ## This can lead to incorrect prediction sets.
    grey_box = ProbAccum(X)
    ranks = np.array([grey_box.ranks[i, y[i]] for i in range(len(y))])
    prob_cum = np.array([grey_box.Z[i, ranks[i]] for i in range(len(y))])
    prob = np.array([grey_box.prob_sort[i, ranks[i]] for i in range(len(y))])
    scores = prob_cum
    if epsilon is not None:
        scores -= np.multiply(prob, epsilon)
    else:
        scores -= prob
    scores = np.maximum(scores, 0)
    return scores


def get_nearest_neighbor(t_scores, test_score, majority_scores, minority_scores):
    # find the largest NN (t_scores is sorted)
    t_scores = t_scores[::-1]  # it doesn't matter as there are no ties in t_scores
    distance = np.abs(t_scores - test_score)
    nn_idx = np.argmin(distance)   # Note - this already takes the nearest neighbor from the bottom because t_scores is sorted
    return t_scores[nn_idx]


def compute_distance(A, B, criterion='wasserstein'):
    if criterion == 'wasserstein':
        return wasserstein_distance(A, B)
    elif criterion == 'cramer-von-mises':
        result = cramervonmises_2samp(A, B)
        return result.statistic
    else:
        raise ValueError(f'The following distance metric is not supported: {criterion}')


def find_k_nearest_neighbors_cdf(k, s, s_maj, y_maj, criterion='wasserstein'):
    distances = {}
    classes = np.unique(y_maj)
    if len(classes) <= k:
        print(f'Warning: Number of classes in the majority is {len(classes)} while k={k}... using all classes.')
        # return classes
    for class_ in classes:
        s_ = s_maj[y_maj == class_]
        dist = compute_distance(s, s_, criterion=criterion)
        distances[class_] = dist
    nearest_classes = sorted(distances.items(), key=lambda x: x[1])[:k]
    nearest_class_labels = [cls_ for cls_, dist in nearest_classes]
    nearest_dist = [dist for cls_, dist in nearest_classes]
    avg_dist = np.mean(nearest_dist)
    # compute the distance of all chosen groups to the real set
    mask = np.isin(y_maj, nearest_class_labels)
    s_maj_selected = s_maj[mask]
    dist = compute_distance(s, s_maj_selected, criterion=criterion)
    return nearest_class_labels, avg_dist, dist


def get_wc_bounds(n_cal_maj, n_cal_min, beta, alpha_list=[0.02, 0.03, 0.05, 0.1, 0.15, 0.2]):
    wc_lower, wc_upper = {}, {}
    T_ = get_hg_quantiles_window_matrix(n_cal_maj, n_cal_min+1, beta=beta, weights=True)
    r_plus_indices = [np.max(np.where(row > 0)[0]) + 1 if np.any(row > 0) else -1 for row in T_]
    r_minus_indices = [np.min(np.where(row > 0)[0]) + 1 if np.any(row > 0) else -1 for row in T_]
    for alpha in alpha_list:
        threshold = int(np.ceil((1 - alpha) * (n_cal_maj + 1)))
        lower = np.sum(np.array(r_plus_indices) <= threshold)
        upper = np.sum(np.array(r_minus_indices) <= threshold)
        wc_lower[alpha] = round(lower/(n_cal_min+1), 3)
        wc_upper[alpha] = round(upper/(n_cal_min+1), 3)
    return wc_lower, wc_upper
