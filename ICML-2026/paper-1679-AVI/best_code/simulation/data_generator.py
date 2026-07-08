
import numpy as np


def generate_true_probs(m, sd=4):

    probs = np.full((m, m), 0.5)
    
    strengths = np.sort(np.random.normal(0, sd, m))[::-1]
    
    for j in range(m):
        for i in range(m):
            if j != i:
                strength_diff = strengths[j] - strengths[i]
                prob = 1 / (1 + np.exp(-strength_diff))
                probs[j, i] = prob
    
    true_ranking = np.argsort(strengths)[::-1] + 1
    
    return {
        'probs': probs,
        'true_ranking': true_ranking,
        'strengths': strengths
    }


def generate_true_probs_with_covariates(m, sd_x=1, sd_beta=0.1, sd_alpha=0.1):

    probs = np.full((m, m), 0.5)
    
    X = np.random.normal(0, sd_x, m)
    beta = np.random.normal(0, sd_beta, m)
    alpha = np.random.normal(0, sd_alpha, m)
    
    Z = X * beta + alpha
    
    strengths = np.sort(Z)[::-1]
    
    for j in range(m):
        for i in range(m):
            if j != i:
                strength_diff = strengths[j] - strengths[i]
                prob = 1 / (1 + np.exp(-strength_diff))
                probs[j, i] = prob
    
    true_ranking = np.argsort(strengths)[::-1] + 1
    
    return {
        'probs': probs,
        'true_ranking': true_ranking,
        'strengths': strengths,
        'X': X,
        'beta': beta,
        'alpha': alpha,
        'Z': Z
    }

