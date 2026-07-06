import numpy as np
import pandas as pd
from scipy.stats import norm

import scipy as scp
from sklearn.linear_model import LinearRegression as lr
from sklearn.feature_selection import f_regression as fr


from sklearn.linear_model import LinearRegression as lr

def grubb_test(quatification_list, confidence_level=0.05):
    n = len(quatification_list)

    stat_value = max(abs(np.array(quatification_list) - np.mean(quatification_list)))/ np.std(quatification_list)

    t_dist = scp.stats.t.ppf(1 - confidence_level / (2 * n), n - 2)
    numerator = (n - 1) * np.sqrt(np.square(t_dist))
    denominator = np.sqrt(n) * np.sqrt(n - 2 + np.square(t_dist))
    critical_value = numerator / denominator

    quatification_list_sorted_idx = sorted(range(len(quatification_list)), key=quatification_list.__getitem__)
    quatification_list_sorted = sorted(quatification_list)
    # locate potential outlier
    l1 = quatification_list_sorted[1] - quatification_list_sorted[0]
    l2 = quatification_list_sorted[-1] - quatification_list_sorted[-2]

    anomaly_position = None
    if stat_value > critical_value:
        if l1 > l2:
            anomaly_position = quatification_list_sorted_idx[0]
        else:
            anomaly_position = quatification_list_sorted_idx[-1]

    return {"anomaly_position": anomaly_position, "stat_value": stat_value, "critical_value": critical_value}
