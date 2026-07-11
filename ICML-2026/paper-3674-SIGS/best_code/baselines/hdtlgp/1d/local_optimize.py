# coding:UTF-8
# @Time: 2023/8/14 11:55
# @Author: Lulu Cao (patched for 1D)
# @File: local_optimize.py
from scipy.optimize import minimize
import numpy as np
from deap import gp

# ---------- NEW: 1D (x,t) loss ----------
def loss1d(params, ind, X, y, pset):
    i = 0
    for node in ind:
        if isinstance(node, gp.Terminal) and not isinstance(node.value, str):
            node.value = params[i]
            i += 1
    ind.expr = gp.compile(ind, pset)
    xs, ts = X
    y_pred = [ind.expr(xs[k], ts[k]) for k in range(len(xs))]
    # Clip predictions before computing loss
    y_pred = np.clip(y_pred, -1e6, 1e6)
    return sum((a - b)**2 for a, b in zip(y, y_pred)) / max(1, len(y))
def loss3d(params, ind, X, y, pset):
    i = 0
    for node in ind:
        if isinstance(node, gp.Terminal) and not isinstance(node.value, str):
            node.value = params[i]; i += 1
    ind.expr = gp.compile(ind, pset)
    y_pred = [ind.expr(X[0][k], X[1][k], X[2][k], X[3][k]) for k in range(len(X[0]))]
    return sum((a - b)**2 for a, b in zip(y, y_pred)) / max(1, len(y))

def local_optimize(individual, X, y, pset):
    """
    Adjust numeric terminals of `individual` by minimizing supervised MSE on (X,y).
    Supports:
      - len(X)==2 : 1D (x,t)
      - len(X)==3 : 2D (x1,x2,t)
      - len(X)==4 : 3D (x1,x2,x3,t)
    """
    params = [node.value for node in individual if (isinstance(node, gp.Terminal) and not isinstance(node.value, str))]
    if len(params) == 0:
        # still compile so the individual stays usable
        individual.expr = gp.compile(individual, pset)
        return

    bounds = [(-100, 100)] * len(params)
    tol = 1e-6

    try:
        if len(X) == 2:
            if loss1d(params, individual, X, y, pset) < 1e-12:
                return
            minimize(loss1d, params, args=(individual, X, y, pset),
                     method='SLSQP', bounds=tuple(bounds), tol=tol)
        # elif len(X) == 3:
        #     if loss2d(params, individual, X, y, pset) < 1e-6:
        #         return
        #     minimize(loss2d, params, args=(individual, X, y, pset),
        #              method='SLSQP', bounds=tuple(bounds), tol=tol)
        elif len(X) == 4:
            if loss3d(params, individual, X, y, pset) < 1e-6:
                return
            minimize(loss3d, params, args=(individual, X, y, pset),
                     method='SLSQP', bounds=tuple(bounds), tol=tol)
        else:
            # unsupported arity: just compile to keep usable
            individual.expr = gp.compile(individual, pset)
            return
    except Exception:
        # on failure, at least compile
        try:
            if len(X) == 2: loss1d(params, individual, X, y, pset)
            # elif len(X) == 3: loss2d(params, individual, X, y, pset)
            elif len(X) == 4: loss3d(params, individual, X, y, pset)
        except Exception:
            individual.expr = gp.compile(individual, pset)
        return
