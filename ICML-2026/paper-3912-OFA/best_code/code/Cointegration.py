from statsmodels.tsa.vector_ar.vecm import coint_johansen
import numpy as np
import statsmodels.tsa.stattools as ts
import pandas as pd

# from PredictTest import pvalue  # unused import, pvalue() not defined in PredictTest


def cointegration(ES, S):
    r = False
    p = 0.05
    length = S.shape[0]
    ES = np.tile(ES, (length, 1))

    for i in range(len(S)):
        test = ts.coint(ES[i, :], S[i, :])
        if test == None:
            test = 0
            r = True
            break
        # print(test[1])

        if test[1] > p:
            r = True
            # print(r)
            break

    return r




def weight_order(data, target, condition, n_order=5):
    worder=[]
    for i in range(n_order):
        x = np.max(data[:, target])
        y = data.loc(condition,x)
        data = np.delete[y,:]
        worder.append(y)
    return worder



