from statsmodels.regression.quantile_regression import QuantReg
# from sklearn.linear_model import QuantileRegressor
import numpy as np
import math


def generate_bootstrap_samples(n, m, B):
    '''
      Return: B-by-m matrix, where row b gives the indices for b-th bootstrap sample
    '''
    samples_idx = np.zeros((B, m), dtype=int)
    for b in range(B):
        sample_idx = np.random.choice(n, m)
        samples_idx[b, :] = sample_idx
    return(samples_idx)


def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n))


def binning(past_resid, alpha):
    '''
    Input:
        past residuals: evident
        alpha: signifance level
    Output:
        beta_hat_bin as argmin of the difference
    Description:
        Compute the beta^hat_bin from past_resid, by breaking [0,alpha] into bins (like 20). It is enough for small alpha
        number of bins are determined rather automatic, relative the size of whole domain
    '''
    bins = 5  # For computation, can just reduce it to like 10 or 5 in real data
    beta_ls = np.linspace(start=0, stop=alpha, num=bins)
    width = np.zeros(bins)
    for i in range(bins):
        width[i] = np.percentile(past_resid, math.ceil(100 * (1 - alpha + beta_ls[i]))) - \
            np.percentile(past_resid, math.ceil(100 * beta_ls[i]))
    i_star = np.argmin(width)
    return beta_ls[i_star]


def binning_use_RF_quantile_regr(quantile_regr, feature, alpha):
    bins = 5
    beta_ls = np.linspace(start=0, stop=alpha, num=bins)
    width = np.zeros(bins)
    for i in range(bins):
        width[i] = quantile_regr.predict(feature.reshape(1, -1), math.ceil(
            100 * (1 - alpha + beta_ls[i]))) - quantile_regr.predict(feature.reshape(1, -1), math.ceil(100 * beta_ls[i]))
    i_star = np.argmin(width)
    return beta_ls[i_star]


# def binning_use_linear_quantile_regr(residX, residY, alpha):
#     # bins = 5
#     # beta_ls = np.linspace(start=1e-5, stop=alpha-1e-5, num=bins)
#     bins = 1
#     beta_ls = [alpha/2]  # No search, as this is too slow.
#     width = np.zeros(bins)
#     width_left = np.zeros(bins)
#     width_right = np.zeros(bins)
#     for i in range(bins):
#         feature = residX[-1]
#         '''
#             Sklearn class
#             See scipy: https://docs.scipy.org/doc/scipy/reference/optimize.linprog-interior-point.html#optimize-linprog-interior-point
#             for a list of option. "solver_options" are given as "options" therein
#
#             NOTE, we CANNOT afford many iterations, as this is VERY COSTLY (about 4 sec per point for this loop below even for 10 iterations...)
#             Even just 1 iter, stll like 2 sec
#
#             See sklearn for which solver to use:
#             https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.QuantileRegressor.html#sklearn.linear_model.QuantileRegressor
#
#             BUT solver = 'highs' is claimed to be fast but actually does not work
#         '''
#         solver = 'interior-point'
#         sol_options = {'maxiter': 10}
#         reg_low = QuantileRegressor(
#             quantile=beta_ls[i], solver=solver, solver_options=sol_options)
#         reg_high = QuantileRegressor(
#             quantile=1 - alpha + beta_ls[i], solver=solver, solver_options=sol_options)
#         reg_low.fit(residX[:-1], residY)
#         reg_high.fit(residX[:-1], residY)
#         width_left[i] = reg_low.predict(feature.reshape(1, -1))
#         width_right[i] = reg_high.predict(feature.reshape(1, -1))
#         # ############################
#         # # Statsmodel class
#         # '''
#         #     https://www.statsmodels.org/dev/generated/statsmodels.regression.quantile_regression.QuantReg.html?highlight=quantreg
#         #     Actually, still not fast....
#         #     Hence, removed this "Optimizer width", but width can then be wider than necessary
#         # '''
#         # mod = QuantReg(residY, residX[:-1], max_iter=1)
#         # reg_low = mod.fit(q=beta_ls[i])
#         # reg_high = mod.fit(q=1-alpha+beta_ls[i])
#         # width_left[i] = mod.predict(reg_low.params, feature)
#         # width_right[i] = mod.predict(reg_high.params, feature)
#         width[i] = width_right[i] - width_left[i]
#     i_star = np.argmin(width)
#     return width_left[i_star], width_right[i_star]


#######
