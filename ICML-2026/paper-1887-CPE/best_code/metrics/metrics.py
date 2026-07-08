import numpy as np

def expected_true_class_prob(posterior, beta_edge, beta_dir, lam, A_true):
    """
    Average probability the posterior assigns to the true label
    (i->j, j->i, or none).
    """
    D = A_true.shape[0]
    tot, n = 0.0, 0
    for i in range(D):
        for j in range(D):
            if i == j:
                continue
            # determine true label
            if A_true[i,j] == 1 and A_true[j,i] == 0:
                ystar = 0   # i->j
            elif A_true[j,i] == 1 and A_true[i,j] == 0:
                ystar = 1   # j->i
            else:
                ystar = 2   # none
            p = posterior.predictive_answer_dist(i, j, beta_edge, beta_dir, lam)
            tot += float(p[ystar])
            n += 1
    return tot / max(n, 1)


def mean_brier_score(posterior, beta, beta0, lam, A_true):
    """
    Multiclass Brier score averaged over node pairs.
    Lower is better (0 is perfect).
    """
    D = A_true.shape[0]
    tot, n = 0.0, 0
    e = np.eye(3)
    for i in range(D):
        for j in range(D):
            if i == j:
                continue
            if A_true[i,j] == 1 and A_true[j,i] == 0:
                ystar = 0
            elif A_true[j,i] == 1 and A_true[i,j] == 0:
                ystar = 1
            else:
                ystar = 2
            p = posterior.predictive_answer_dist(i, j, beta, beta0, lam)
            tot += float(np.sum((p - e[ystar])**2))
            n += 1
    return tot / max(n, 1)

