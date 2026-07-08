
import numpy as np


def safe_mean(x):

    if len(x) == 0 or np.all(np.isnan(x)):
        return np.nan
    return np.nanmean(x)


def safe_sd(x):

    if len(x) <= 1 or np.all(np.isnan(x)):
        return np.nan
    return np.nanstd(x, ddof=1)

