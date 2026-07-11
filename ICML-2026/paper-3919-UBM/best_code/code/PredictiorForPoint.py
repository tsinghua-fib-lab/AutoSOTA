# the predictor for ESE, as time series
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA


def ESE_predictor_part(part, state, equilibrium):

    diff = (state-equilibrium) / state
    p = part * (1 + diff)

    return p

def ESE_predictor_system(w, equilibrium):
    ps = []
    for i in range(len(equilibrium)):
        p = w * equilibrium[i]
        ps.append(p)
    return ps

def ESE_predictor_system_ar(w, equilibrium,h):
    ps = []
    # mod = AutoReg(w,lags=1).fit()
    p=select_order(w)
    mod = ARIMA(endog=w, order=(p, 0, 0)).fit()
    fcast = mod.forecast(h)
    #print(fcast)
    for i in range(len(equilibrium)):
        p = fcast * equilibrium[i]
        ps.append(p)
    return ps
#def ESE_predictor_system_ar(w, equilibrium,h):
#    model = sm.tsa.AR(w)
 #   result = model.fit(maxlag=1)
 #   forecast = result.predict(start=len(w), end=len(w) + h)
  #  for i in range(len(equilibrium)):
   #     p = mod.params[1] * w_traning[:-1] * equilibrium[i]
   #     ps.append(p)
 #   return forecast
def select_order(y, max_p=10, criterion="aic"):
    best_ic = np.inf
    best_p = None
    best_res = None
    for p in range(1, max_p+1):
        try:
            res = ARIMA(y, order=(p,0,0)).fit()
            ic = getattr(res, criterion)
            if ic < best_ic:
                best_ic = ic
                best_p = p
        except Exception as e:
            continue
    return best_p