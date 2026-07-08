import numpy as np


def calc_residuals(y_hat, y):
    if isinstance(y, float):
        return y - y_hat
    else:
        assert y_hat.shape == y.shape
        return y - y_hat


def calc_default_conformal_Q(eps, alpha):
    conformal_with = np.quantile(np.abs(eps), float((1 - alpha)))
    return conformal_with


def calc_cqr_score(quantile_low_hat, quantile_high_hat, y):
    """ Calculate Conformal Score for CQR - see: Conformal quantile regression (CQR) Romano et al 2018 - (Eq 6)"
    :param quantile_low_hat: forecasted lower quantile
    :param quantile_high_hat: forecasted higher quantile
    :param y: real y value in this priod
    :return: score (for each step) of the prediction
    """
    if isinstance(y, float):
        return max(quantile_low_hat - y, y - quantile_high_hat)
    else:
        assert quantile_low_hat.shape == quantile_high_hat.shape == y.shape
        return np.maximum(quantile_low_hat - y, y - quantile_high_hat)


def calc_cqr_Q(cqr_calib_scores, alpha):
    """ Calc Quantile for CQR -  see: Conformal quantile regression (CQR) Romano et al 2018 - (Eq 8)
    :param cqr_calib_scores: CQR scores (see calc_cqr_score)
    :param alpha: alpha
    """
    return np.quantile(cqr_calib_scores, min(1.0, float((1 - alpha)*(1 + 1/len(cqr_calib_scores)))))
