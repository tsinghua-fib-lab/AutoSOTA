# local_optimize.py
# Constant tuning for DEAP individuals using SciPy.
from scipy.optimize import minimize
from deap import gp
import random, math

# ---------- utilities ----------
def _numeric_param_nodes(ind):
    """
    Collect references (indices) to numeric terminals we want to tune.
    We SKIP math.pi so spectral structures aren't destroyed.
    """
    idxs = []
    for i, node in enumerate(ind):
        if isinstance(node, gp.Terminal) and not isinstance(node.value, str):
            v = node.value
            # skip exact pi (or extremely close)
            if isinstance(v, (int, float)) and abs(float(v) - math.pi) < 1e-12:
                continue
            idxs.append(i)
    return idxs

def _assign_params(ind, param_idcs, params):
    assert len(param_idcs) == len(params)
    for i, p in zip(param_idcs, params):
        ind[i].value = float(p)

def _maybe_mask_poisson(X0, X1, X2):
    """
    Heuristic to decide whether we're in the Poisson (steady) case:
    - 3 inputs (x,y,t), but all t == 0 (or ~0)
    - x,y appear to be in [0,1] (tolerant)
    """
    if len(X2) == 0:  # shouldn't happen
        return False
    max_abs_t = max(abs(ti) for ti in X2)
    if max_abs_t > 1e-12:
        return False
    # check x,y roughly in [0,1]
    xs = [float(v) for v in X0]; ys = [float(v) for v in X1]
    x_ok = (min(xs) >= -1e-6) and (max(xs) <= 1.0 + 1e-6)
    y_ok = (min(ys) >= -1e-6) and (max(ys) <= 1.0 + 1e-6)
    return x_ok and y_ok

def _subsample_ids(n, max_pts=2000):
    if n <= max_pts:
        return list(range(n))
    return random.sample(range(n), max_pts)

# ---------- losses ----------
def loss1d(params, ind, X, y, pset, param_idcs, max_pts=2000):
    _assign_params(ind, param_idcs, params)
    ind.expr = gp.compile(ind, pset)
    xs, ts = X
    n = len(xs)
    ids = _subsample_ids(n, max_pts)
    err = 0.0
    for k in ids:
        yk = ind.expr(xs[k], ts[k])
        diff = (yk - y[k])
        err += diff*diff
    return err / max(1, len(ids))

def loss2d(params, ind, X, y, pset, param_idcs, max_pts=2000):
    _assign_params(ind, param_idcs, params)
    ind.expr = gp.compile(ind, pset)
    xs, ys, ts = X
    n = len(xs)
    ids = _subsample_ids(n, max_pts)
    # Poisson? then mask with sin(pi x) sin(pi y) to match evaluator
    use_mask = _maybe_mask_poisson(xs, ys, ts)
    err = 0.0
    if use_mask:
        for k in ids:
            mask = math.sin(math.pi*xs[k]) * math.sin(math.pi*ys[k])
            yk = mask * ind.expr(xs[k], ys[k], ts[k])  # t is 0
            diff = (yk - y[k])
            err += diff*diff
    else:
        for k in ids:
            yk = ind.expr(xs[k], ys[k], ts[k])  # damped wave path
            diff = (yk - y[k])
            err += diff*diff
    return err / max(1, len(ids))

def loss3d(params, ind, X, y, pset, param_idcs, max_pts=2000):
    _assign_params(ind, param_idcs, params)
    ind.expr = gp.compile(ind, pset)
    n = len(X[0])
    ids = _subsample_ids(n, max_pts)
    err = 0.0
    for k in ids:
        yk = ind.expr(X[0][k], X[1][k], X[2][k], X[3][k])
        diff = (yk - y[k])
        err += diff*diff
    return err / max(1, len(ids))

# ---------- main entry ----------
def local_optimize(individual, X, y, pset, max_pts=2000):
    """
    Tune numeric terminals (excluding pi) by minimizing supervised MSE on (X,y).
    Supports:
      - len(X)==2 : 1D (x,t)
      - len(X)==3 : 2D (x,y,t)  -> damped-wave or Poisson (auto-masked)
      - len(X)==4 : 3D (x1,x2,x3,t)
    """
    param_idcs = _numeric_param_nodes(individual)
    if len(param_idcs) == 0:
        # still compile so the individual stays usable
        individual.expr = gp.compile(individual, pset)
        return

    init = [individual[i].value for i in param_idcs]
    bounds = [(-20, 20)] * len(param_idcs)   # tighter than (-50,50) for stability
    tol = 1e-6

    try:
        if len(X) == 2:
            # quick short-circuit if already good
            if loss1d(init, individual, X, y, pset, param_idcs, max_pts) < 1e-2:
                return
            minimize(loss1d, init, args=(individual, X, y, pset, param_idcs, max_pts),
                     method='SLSQP', bounds=tuple(bounds), tol=tol)
        elif len(X) == 3:
            if loss2d(init, individual, X, y, pset, param_idcs, max_pts) < 1e-2:
                return
            minimize(loss2d, init, args=(individual, X, y, pset, param_idcs, max_pts),
                     method='SLSQP', bounds=tuple(bounds), tol=tol)
        elif len(X) == 4:
            if loss3d(init, individual, X, y, pset, param_idcs, max_pts) < 0.02:
                return
            minimize(loss3d, init, args=(individual, X, y, pset, param_idcs, max_pts),
                     method='SLSQP', bounds=tuple(bounds), tol=tol)
        else:
            individual.expr = gp.compile(individual, pset)
            return
    except Exception:
        # on failure, at least compile
        try:
            if len(X) == 2: loss1d(init, individual, X, y, pset, param_idcs, max_pts)
            elif len(X) == 3: loss2d(init, individual, X, y, pset, param_idcs, max_pts)
            elif len(X) == 4: loss3d(init, individual, X, y, pset, param_idcs, max_pts)
        except Exception:
            pass
        individual.expr = gp.compile(individual, pset)
        return
