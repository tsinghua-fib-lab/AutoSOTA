import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests

def granger(data, maxlag=None, p_val_threshold=0.05, test=None):
    assert test in [None, 'ssr_ftest', 'ssr_chi2test', 'lrtest', 'params_ftest']

	# data: pandas dataframe
    if maxlag is None:
        maxlag = 3
    node_names = data.columns.to_list()
    adj = np.zeros((len(node_names), len(node_names)))

    for i in range(len(node_names)):
        for j in range(len(node_names)):
            if i == j:
                continue
            # test j -> i
            output = grangercausalitytests(
                data[[node_names[i], node_names[j]]], maxlag, verbose=False
            )
            caused = False
            for time_lag, out in output.items():
                out = out[0]

                if test is None:
                    avg_p_val = sum([v[1] for k, v in out.items()]) / len(out)
                else:
                    avg_p_val = out[test][1]

                if avg_p_val < p_val_threshold:  # average p-value
                    caused = True
                    break
            if caused:
                adj[i, j] = 1
    return adj
