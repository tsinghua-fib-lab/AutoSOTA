import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from utils import to_numpy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# Helper function for WSC
def sample_sphere(n, p, random_state):
    rng = np.random.default_rng(random_state)
    v = rng.normal(0, 1, size=(p, n))
    v /= np.linalg.norm(v, axis=0)
    return v.T

# WSC metric, M: num of projection vectors
def wsc_unbiased(X, covered, eta=0.2, M=1000, test_size=0.75, random_state=42):
    """Computes Worst-Slice Coverage."""
    if len(X) < 100: return 0.0 
    X_train, X_test, cov_train, cov_test = train_test_split(X, covered, test_size=test_size, random_state=random_state)
    n = len(X_train)
    V = sample_sphere(M, X_train.shape[1], random_state)
    z = np.dot(X_train, V.T)
    z_order = np.argsort(z, axis=0)
    cover_ordered = np.take_along_axis(cov_train[:, None], z_order, axis=0)
    
    k_min = int(math.ceil(eta * n))
    cum_cov = np.vstack([np.zeros((1, M)), np.cumsum(cover_ordered, axis=0)])
    roll_cov = (cum_cov[k_min:] - cum_cov[:-k_min]) / k_min
    
    min_cov_idx = np.argmin(roll_cov, axis=0)
    idx_star = np.argmin(roll_cov[min_cov_idx, np.arange(M)])
    
    v_star = V[idx_star]
    start_idx = min_cov_idx[idx_star]
    z_sorted = np.take_along_axis(z, z_order, axis=0)
    a_star = z_sorted[start_idx, idx_star]
    b_star = z_sorted[start_idx + k_min - 1, idx_star]
    
    z_test = np.dot(X_test, v_star)
    valid = (z_test >= a_star) & (z_test <= b_star)
    if np.sum(valid) == 0: return 1.0
    return np.mean(cov_test[valid])


class ConditionalCoverageComputer:
    """Computes MSCE via partition-based approximation."""
    def __init__(self, X, nb_partitions=10, random_state=42):
        self.kmeans = KMeans(n_clusters=nb_partitions, n_init='auto', random_state=random_state).fit(X)
        self.labels = self.kmeans.labels_
        self.nb = nb_partitions

    def compute_error(self, coverages, alpha):
        err = 0
        total_len = len(coverages)
        for i in range(self.nb):
            mask = self.labels == i
            c = np.sum(mask)
            if c > 0: 
                w = c / total_len
                err += w * (np.mean(coverages[mask]) - (1-alpha))**2
        return err

def _loss_l1_ert(p, y, c):
    """Proper score associated with L1-ERT (see Braun et al., 2025)."""
    return np.sign(p - c) * (c - y)


def _loss_brier(p, y, c=None):
    return (y - p) ** 2


def _loss_logloss(p, y, c=None, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))

# self-contained ERT implementation
def ert_metric(
    x,
    cover,
    alpha=0.1,
    n_splits=5,
    random_state=42,
    model=None,
    loss="l1",
):
    """Estimate Excess Risk of the Target coverage (ERT)."""
    x = to_numpy(x)
    y = to_numpy(cover).astype(int).reshape(-1)
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    c = 1.0 - float(alpha)

    if model is None:
        model = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            LogisticRegression(max_iter=2000, solver="lbfgs"),
        )
       
    loss = loss.lower()
    if loss in {"l1", "l1_ert"}:
        loss_fn = lambda p, yy: _loss_l1_ert(p, yy, c)
        const_loss_fn = lambda yy: _loss_l1_ert(np.full_like(yy, c, dtype=float), yy, c)
    elif loss in {"l2", "brier"}:
        loss_fn = lambda p, yy: _loss_brier(p, yy)
        const_loss_fn = lambda yy: _loss_brier(np.full_like(yy, c, dtype=float), yy)
    elif loss in {"kl", "logloss"}:
        loss_fn = lambda p, yy: _loss_logloss(p, yy)
        const_loss_fn = lambda yy: _loss_logloss(np.full_like(yy, c, dtype=float), yy)
    else:
        raise ValueError(f"Unknown ERT loss: {loss}")

    n = len(y)
    if n < 20:
        return 0.0

    if n_splits and n_splits > 1:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        p_hat = np.empty(n, dtype=float)
        global_mean = float(np.mean(y))

        for tr_idx, te_idx in skf.split(x, y):
            y_tr = y[tr_idx]
            if len(np.unique(y_tr)) < 2:
                p_hat[te_idx] = global_mean
                continue
            m = model
            m.fit(x[tr_idx], y_tr)
            p_hat[te_idx] = m.predict_proba(x[te_idx])[:, 1]
    else:
        if len(np.unique(y)) < 2:
            return 0.0
        model.fit(x, y)
        p_hat = model.predict_proba(x)[:, 1]

    risk_const = float(np.mean(const_loss_fn(y)))
    risk_model = float(np.mean(loss_fn(p_hat, y)))
    return risk_const - risk_model


def get_metrics_nd(y_test, y_lo, y_hi, x_test, alpha=0.1, covered=None, size_metric=None):
    """
    Calculates all metrics for N-dimensional regression.
    """
    y_test = to_numpy(y_test)
    x_test = to_numpy(x_test)
    
    # 1. Marginal Coverage & Size
    if covered is None:
        y_lo, y_hi = to_numpy(y_lo), to_numpy(y_hi)
        in_bounds = (y_test >= y_lo) & (y_test <= y_hi)
        covered = np.all(in_bounds, axis=1)
    else:
        covered = to_numpy(covered)

    if size_metric is None:
        if y_lo is None or y_hi is None:
            raise ValueError("Either (y_lo, y_hi) or size_metric must be provided.")
        D = y_test.shape[1]
        y_lo, y_hi = to_numpy(y_lo), to_numpy(y_hi)
        widths = np.maximum(y_hi - y_lo, 1e-6)
        if D > 1: 
            size_metric = np.mean(np.mean(np.log(widths), axis=1))
        else: 
            size_metric = np.mean(widths)
    
    cov = np.mean(covered)
    
    # 2. WSC
    wsc_val = wsc_unbiased(x_test, covered)
    
    # 3. MSCE (K=30) 
    msce_computer_30 = ConditionalCoverageComputer(x_test, nb_partitions=30)
    msce_val_30 = msce_computer_30.compute_error(covered, alpha)

    # 4. MSCE (K=10) 
    msce_computer_10 = ConditionalCoverageComputer(x_test, nb_partitions=10)
    msce_val_10 = msce_computer_10.compute_error(covered, alpha)

    # 5. ERT
    try:
        ert_val_l1 = ert_metric(x_test, covered.astype(int), alpha=alpha, n_splits=5, loss="l1")
        ert_val_l2 = ert_metric(x_test, covered.astype(int), alpha=alpha, n_splits=5, loss="l2")
        
    except Exception:
        ert_val_l1, ert_val_l2 = 0.0, 0.0
    
    return {
        "Cov": cov, 
        "Size": size_metric, 
        "WSC": wsc_val, 
        "MSCE_30": msce_val_30, 
        "MSCE_10": msce_val_10,
        "L1-ERT": ert_val_l1, 
        "L2-ERT": ert_val_l2
    }