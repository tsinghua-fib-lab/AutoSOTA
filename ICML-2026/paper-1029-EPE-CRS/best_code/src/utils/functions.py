import numpy as np

def sigmoid(z):
    return 1. / (1. + np.exp(-z))


def EPE(X, Y, beta):
    X = np.asarray(X)
    beta = np.asarray(beta)
    times = Y['time']
    events = Y['failure']
    n = len(X)

    # Create pairwise indices
    i_idx, j_idx = np.triu_indices(n, k=1)
    ti, tj = times[i_idx], times[j_idx]
    ei, ej = events[i_idx], events[j_idx]

    mask1 = (ti <= tj) & ei
    mask2 = (tj < ti) & ej

    diffs1 = X[i_idx[mask1]] - X[j_idx[mask1]]
    diffs2 = X[j_idx[mask2]] - X[i_idx[mask2]]
    diffs = np.vstack((diffs1, diffs2))

    if diffs.shape[0] == 0:
        return np.inf

    logits = diffs @ beta
    log_probs = -np.logaddexp(0, -logits)

    r = -np.mean(log_probs)
    
    return r