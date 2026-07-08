
import numpy as np
from scipy import special
from scipy.stats import beta as beta_dist
from scipy.integrate import quad


def compute_e_value(S_t, t):

    if t == 0:
        return 1.0
    

    if S_t == 0:
        log_denominator = 0
        p_hat = 0
    elif S_t == t:
        p_hat = 0.5
        log_denominator = t * np.log(0.5)
    else:
        p_hat = min(S_t / t, 0.5)
        log_denominator = S_t * np.log(p_hat) + (t - S_t) * np.log(1 - p_hat)
    
    log_beta = (special.gammaln(S_t + 1) + 
                special.gammaln(t - S_t + 1) - 
                special.gammaln(t + 2))
    
    prob_ge_05 = beta_dist.sf(0.5, S_t + 1, t - S_t + 1)
    
    if prob_ge_05 == 0 and S_t/t < 0.5:
        return 0.0
    
    log_numerator = np.log(2) + log_beta + np.log(prob_ge_05)
    log_e_value = log_numerator - log_denominator
    
    if log_e_value < -700:
        return 0.0
    
    e_value = np.exp(log_e_value)
    
    if np.isnan(e_value) or np.isinf(e_value):
        if S_t/t <= 0.5:
            return 0.0
        else:
            return np.inf
    
    return e_value


def compute_covariate_integral(z, alpha_t, beta_t):
    """

    """
    def integrand(p):
        if p <= 0.5 or p >= 1.0:
            return 0.0
        
        likelihood = (p ** z) * ((1 - p) ** (1 - z))
        

        u = 2 * p - 1
        if u <= 0 or u >= 1:
            return 0.0
        
        beta_density = beta_dist.pdf(u, alpha_t, beta_t)
        g_p = 2 * beta_density
        
        return likelihood * g_p
    
    result, _ = quad(integrand, 0.5, 1.0, limit=100)
    return max(result, 1e-300)


def compute_e_value_covariate(observations, bt_predictions, S_t, t):
    """

    """
    if t == 0:
        return 1.0
    
    
    log_numerator = 0.0
    
    for s in range(t):
        z_s = observations[s]
        p_hat_s = bt_predictions[s]
        
        alpha_s = 1.0 + max(p_hat_s - 0.5, 0.0)
        beta_s = 1.0
        
        integral = compute_covariate_integral(z_s, alpha_s, beta_s)
        log_numerator += np.log(integral)
    
    if S_t == 0:
        log_denominator = 0.0
    elif S_t == t:
        p_hat = 0.5
        log_denominator = t * np.log(0.5)
    else:
        p_hat = min(S_t / t, 0.5)
        if p_hat > 0 and (1 - p_hat) > 0:
            log_denominator = S_t * np.log(p_hat) + (t - S_t) * np.log(1 - p_hat)
        else:
            log_denominator = -np.inf
    
    log_e_value = log_numerator - log_denominator
    
    if log_e_value < -700:
        return 0.0
    if log_e_value > 700:
        return np.inf
    
    e_value = np.exp(log_e_value)
    
    if np.isnan(e_value) or np.isinf(e_value):
        if S_t / t <= 0.5:
            return 0.0
        else:
            return np.inf
    
    return e_value
