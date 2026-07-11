import numpy as np

# NOTE: Non-contextual version (without X input)

def CREME(y_arr, lam_list, icrc,
        output_posthoc=False, w=np.array([1., 1.])):
    '''
    Args:
    - y_arr: [ n, nY ], response matrix calibration data
    - lam_list: list of scalar, robustness parameters
    - output_posthoc: bool, whether to output the post-hoc correction results
    - w: [ 2 ], preference weight
    Returns:
    - regret_hat: scalar, estimated regret
    - miscoverage_hat: scalar, estimated miscoverage
    - lam: scalar, chosen robustness parameter
    - F1: [ nlam, 2 ] first split frontier
    - F2: [ nlam, 2 ] second split frontier
    '''

    n = y_arr.shape[0]
    if output_posthoc:
        index_perm = np.random.permutation(np.arange(n))
        index_list = [index_perm[:n//2], index_perm[n//2:]] # random splitting
    else:
        index_list = [np.arange(n), np.arange(n)]

    # compute first split frontier
    index = index_list[0]
    F1 = []
    for lam in lam_list:
        icrc.compute(y_arr[index], lam)
        regret_hat, miscoverage_hat = icrc.estimate()
        F1.append(np.array([regret_hat, miscoverage_hat]))

    # compute second split frontier
    index = index_list[1]
    F2 = []
    for lam in lam_list:
        icrc.compute(y_arr[index], lam)
        regret_hat, miscoverage_hat = icrc.estimate()
        F2.append(np.array([regret_hat, miscoverage_hat]))

    # choose lam
    score_list = [w[0] * F1[i][0] + w[1] * F1[i][1] for i in range(len(lam_list))]
    lam_index = np.argmin(score_list)
    lam = lam_list[lam_index]

    # compute tradeoff
    index = index_list[1]
    icrc.compute(y_arr[index], lam)
    regret_hat, miscoverage_hat = icrc.estimate()
    
    return regret_hat, miscoverage_hat, lam, F1, F2