import numpy as np
from utils.functions import EPE
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from scipy.special import logsumexp, softmax, log_softmax


def bounding_box(X):
    if len(X.shape) == 1:
        B = np.zeros(2)
        B[0] = np.min(X)
        B[1] = np.max(X)
    else:
        B = np.vstack([np.min(X, axis=0), np.max(X, axis=0)])
    return B


def in_region(X, R):
    if len(X.shape) < 2:
        X = X.reshape(1, len(X))
    return np.array([np.all(x >= R[0, :]) and np.all(x <= R[1, :]) for x in X])


def fit_cox(X, Y):
    if len(X.shape) == 1:
        beta = CoxPHSurvivalAnalysis().fit(X.reshape(-1, 1), Y).coef_
    else:
        beta = CoxPHSurvivalAnalysis().fit(X, Y).coef_
    return beta


def epe_metric(X, Y):
    try:
        beta = fit_cox(X, Y)
        return EPE(X, Y, beta)
    except:
        return np.inf


def c_ind_metric(X, Y):
    """
    C-index metric for selecting core group.
    We select the core group with the lowest score, so return negative C-index.
    """
    try:
        beta = fit_cox(X, Y)
        event_indicator = Y['failure']
        event_time = Y['time']
        risk_estimate = X @ beta
        return -concordance_index_censored(event_indicator, event_time, risk_estimate)[0]
    except:
        return 0.


def pl_metric(X, Y):
    """
    Partial log-likelihood metric for selecting core group.
    We select the core group with the lowest score, so return negative log-likelihood.
    Event times must be sorted.
    """
    try:
        beta = fit_cox(X, Y)
        risk = X @ beta
        failure_indices = np.where(Y['failure'])[0]
        return -sum([log_softmax(risk[i:])[0] for i in failure_indices])
    except:
        return np.inf


def fast_log_rank_probs(x, X, Y, beta):
    """
    All notation used in this function corresponds to Section 3.3 in the paper.
    THIS ASSUMES EVENT TIMES ARE SORTED IN ASCENDING ORDER!!!
    """
    n = len(X)

    logits = X @ beta
    alphas = np.exp(logits)
    sum_alphas = np.sum(alphas)

    logit0 = np.dot(x, beta)
    alpha0 = np.exp(logit0)

    log_rs = np.zeros(n + 1)

    # Compute log r_1.
    S = sum_alphas
    log_prod = logit0 - np.log(S + alpha0)
    for i in range(n):
        log_prod += Y['failure'][i] * (logits[i] - np.log(S))
        S -= alphas[i]
    log_rs[0] = log_prod

    # Recursively compute log r_{k+1} from log r_k.
    S = sum_alphas
    for k in range(n):
        log_rs[k + 1] = log_rs[k] + np.log(S + (1 - Y['failure'][k]) * alpha0) - np.log(S + alpha0 - alphas[k])
        S -= alphas[k]
    
    return log_rs


def fast_cond_rank_probs(x, X, Y, beta):
    return softmax(fast_log_rank_probs(x, X, Y, beta))


def log_tail_rej_score(x, y, X, Y, beta):
    ind = np.searchsorted(Y['time'], y['time'])
    logits = fast_log_rank_probs(x, X, Y, beta) # IAIN - called a lot
    
    if y['failure']: # If the point failed, look at two-sided tails. Multiply by 2 to look at alpha/2 two-sided tails for failed points with same threshold as alpha one-sided tail for censored points.
        log_tail = min(logsumexp(logits[:ind+1]) - logsumexp(logits), logsumexp(logits[ind:]) - logsumexp(logits)) + np.log(2.)
    else: # If the point was censored, look only at right tail P(survived at least this long | group).
        log_tail = logsumexp(logits[ind:]) - logsumexp(logits)
    return log_tail


def c_ind_rej_score(x, y, X, Y, beta):
    """
    Compute a C-index based rejection score for a single test point (x, y) given training data (X, Y) and Cox model coefficients beta.
    REQUIRES Y TO BE SORTED IN ASCENDING ORDER OF TIME!!!
    """
    core_gp_risk = X @ beta
    test_risk = np.dot(x, beta)

    ind = np.searchsorted(Y['time'], y['time'])
    
    comparable_pairs = 0
    concordant_pairs = 0
    
    for i in range(ind):
        if Y['failure'][i]:
            comparable_pairs += 1
            if core_gp_risk[i] > test_risk:
                concordant_pairs += 1
    
    if y['failure']:
        for i in range(ind, len(Y)):
            comparable_pairs += 1
            if core_gp_risk[i] <= test_risk:
                concordant_pairs += 1
    
    return concordant_pairs / comparable_pairs if comparable_pairs > 0 else 0
        

def pl_rej_score(x, y, X, Y, beta):
    """
    Compute a partial likelihood-based rejection score for a single test point (x, y) given training data (X, Y) and Cox model coefficients beta.
    REQUIRES Y TO BE SORTED IN ASCENDING ORDER OF TIME!!!
    """
    ind = np.searchsorted(Y['time'], y['time'])
    
    exp_risk = np.exp((X - x) @ beta)
    sum_exp_risk = 1. + np.sum(exp_risk[ind:])

    if y['failure']:
        return -np.log(sum_exp_risk)
    else:
        partial_likelihood = 0.
        for i in range(ind):
            if Y['failure'][i]:
                partial_likelihood += 1. / sum_exp_risk
            sum_exp_risk -= exp_risk[i]
        return np.log(partial_likelihood) if partial_likelihood > 0 else -np.inf


def get_best_boxes(X_adjust, X_subgp, Y, boxes, num_subgroups, metric):
    """
    Utility for tree-based methods.
    """
    metric_vals = []
    betas = []
    for R in boxes:
        ind = in_region(X_subgp, R)
        try:
            metric_vals.append(metric(X_adjust[ind], Y[ind]))
            betas.append(fit_cox(X_adjust[ind], Y[ind]))
        except:
            metric_vals.append(np.inf)
            betas.append(np.full(X_adjust.shape[1], np.nan) if len(X_adjust.shape) > 1 else np.full(1, np.nan))

    ranking_by_metric = sorted(range(len(metric_vals)), key=lambda k: metric_vals[k]) # Sort the leaf bounding boxes by the metric
    results = [(boxes[i], betas[i]) for i in ranking_by_metric[:num_subgroups]]
    return results


def retrieve_branches(number_nodes, children_left_list, children_right_list):
    """Retrieve decision tree branches"""
    
    # Calculate if a node is a leaf
    is_leaves_list = [(False if cl != cr else True) for cl, cr in zip(children_left_list, children_right_list)]
    
    # Store the branches paths
    paths = []
    
    for i in range(number_nodes):
        if is_leaves_list[i]:
            # Search leaf node in previous paths
            end_node = [path[-1] for path in paths]

            # If it is a leave node yield the path
            if i in end_node:
                output = paths.pop(np.argwhere(i == np.array(end_node))[0][0])
                yield output

        else:
            
            # Origin and end nodes
            origin, end_l, end_r = i, children_left_list[i], children_right_list[i]

            # Iterate over previous paths to add nodes
            for index, path in enumerate(paths):
                if origin == path[-1]:
                    paths[index] = path + [end_l]
                    paths.append(path + [end_r])

            # Initialize path in first iteration
            if i == 0:
                paths.append([i, children_left_list[i]])
                paths.append([i, children_right_list[i]])


def tree_to_bounding_boxes(tree, max_depth=np.inf, B_subgp=None):
    all_branches = list(retrieve_branches(tree.node_count, tree.children_left, tree.children_right))
    
    if B_subgp is None:
        B_subgp = np.zeros((2, tree.n_features))
        B_subgp[0, :] = -np.inf
        B_subgp[1, :] = np.inf
    

    boxes = []
    for branch in all_branches:
        box = B_subgp.copy()
        for j in range(min(len(branch) - 1, max_depth)):
            current_node = branch[j]
            next_node = branch[j+1]
            left_or_right = 1 if next_node == tree.children_left[current_node] else 0 # If we go left, the split threshold is an upper bound (row 1 in bounding box); if we go right, the split threshold is a lower bound (row 0 in bounding box)
            feature_id = tree.feature[current_node]
            split_threshold = tree.threshold[current_node]
            box[left_or_right, feature_id] = split_threshold
        boxes.append(box)

    return boxes