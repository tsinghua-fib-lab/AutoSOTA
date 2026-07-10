# benchmark/tasks/utils.py
import time
import numpy as np
from typing import Callable, Dict, Any
from sklearn.metrics import r2_score, mean_squared_error

def scorer_r2(y_true, y_pred):  return r2_score(y_true, y_pred)
def scorer_mse(y_true, y_pred): return mean_squared_error(y_true, y_pred)

def timed_call(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return out, (t1 - t0)
