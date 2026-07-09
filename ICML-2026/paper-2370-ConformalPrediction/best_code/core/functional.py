import numpy as np
from scipy.optimize import bisect

def pinball_loss(u, alpha):
    """
    Computes pinball loss between the true coverage level beta and predicted alpha.
    l(beta, alpha) = alpha(beta - alpha) - min(0, beta - alpha)
    """
    # Matches the definition in DtACI.R: alpha*u - vecZeroMin(u)
    # Note: 'self.alpha' here is the TARGET alpha (e.g. 0.1), not the prediction.
    return max(-(1 - alpha) * u, alpha * u)
    
def compute_empirical_cdf(past_scores, current_score):
    """
    Computes beta_t: The probability that a score from the underlying distribution
    is >= S_new. This is effectively 1 - CDF(S_new).
    
    Matches logic of findBeta in VolatilityExperiment.ipynb.
    """
    if (not past_scores) or (current_score >= max(past_scores)):
        return 1.0 # Default fallback if no history
    elif current_score <= min(past_scores):
        return 0.0
    else:
        return bisect(lambda x: np.quantile(past_scores, x) - current_score, 0.0, 1.0)

# def find_beta(past_scores, current_score):
#     """
#     Computes beta_t: The probability that a score from the underlying distribution
#     is >= S_new. This is effectively 1 - CDF(S_new).
    
#     Matches logic of findBeta in VolatilityExperiment.ipynb.
#     """
#     if not past_scores:
#         return 0.5 # Default fallback if no history

#     # beta = bisect(lambda beta: np.quantile(self.experts[0].scores, 1 - beta) - S_new, 0.0, 1.0)
#     top = 1.0 
#     bot = 0.0
#     mid = top + bot / 2.0
#     epsilon = 1e-3
#     while top - bot > epsilon:
#         mid = (top + bot) / 2.0
#         q = np.quantile(past_scores, 1 - mid)
#         if q < current_score:
#             top = mid
#         else:
#             bot = mid
#     return mid