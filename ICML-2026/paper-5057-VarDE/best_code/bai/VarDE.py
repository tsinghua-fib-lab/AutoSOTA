from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
import random
from env import Bandit
from utils import Welford
 
# Abstract class for VarDEBAI algorithms
class VarDEBAI:
    def __init__(self, bandit: Bandit, 
                 T: int, 
                 warm_start: int = 1,
                 var_floor: float = 0.01,
                 initial_var: float = float('inf'),
                 use_empirical_variance: bool = True,
                 use_influence_weights: bool = True,
                 fixed_variance: float = 1.0,
                 variance_exponent: float = 1.0,
                 ema_alpha: float = 0.0):
        self.env = bandit
        self.T = T
        self.warm_start = warm_start
        self.K = bandit.K
        self.var_floor = float(var_floor)
        self.initial_var = float(initial_var)
        self.use_empirical_variance = bool(use_empirical_variance)
        self.use_influence_weights = bool(use_influence_weights)
        self.fixed_variance = float(fixed_variance)
        if self.fixed_variance <= 0:
            raise ValueError("fixed_variance must be positive.")
        self.variance_exponent = float(variance_exponent)
        if self.variance_exponent < 0:
            raise ValueError("variance_exponent must be non-negative.")
        self.ema_alpha = float(ema_alpha)
        self.trackers = [
            Welford(var_floor=self.var_floor, initial_var=self.initial_var, ema_alpha=self.ema_alpha)
            for _ in range(self.K)
        ]
        self.N = np.zeros(self.K, dtype=int)

        # Store each step data
        self.means_history = []
        self.vars_history = []
        self.decision_n_history = []
        self.n_history = []
        self.w_history = []
        self.scores_history = []
        self.rec_history = []

        # Warm-start phase
        for _ in range(self.warm_start):
            arm_indices = list(range(self.K))
            random.shuffle(arm_indices)
            for i in arm_indices:
                r = self.env.pull(i)
                self.trackers[i].update(r)
                self.N[i] += 1
                self.n_history.append(self.N.copy())
                self.rec_history.append(np.argmax([tr.mean for tr in self.trackers]))

    def compute_w(self, t: int, means: np.array) -> np.array:
        pass

    def effective_w(self, w: np.ndarray) -> np.ndarray:
        if self.use_influence_weights:
            return np.asarray(w, dtype=float)
        return np.full(self.K, 1.0 / self.K, dtype=float)

    def effective_vars(self) -> np.ndarray:
        if self.use_empirical_variance:
            return np.array([tr.var for tr in self.trackers], dtype=float)
        return np.full(self.K, self.fixed_variance, dtype=float)

    def select_arm(self, w: np.array) -> int:
        vars = self.effective_vars()
        denom = self.N * (self.N + 1)
        denom = np.maximum(denom, 1)
        score = (w ** 2) * ((vars ** self.variance_exponent) / denom)

        # Record scores
        self.scores_history.append(score)

        return int(np.argmax(score))

    def run(self) -> Dict[str, np.ndarray]:
        T = self.T

        for t in range(T-self.warm_start*self.K):
            # Record histories
            self.decision_n_history.append(self.N.copy())
            self.means_history.append([tr.mean for tr in self.trackers])
            self.vars_history.append([tr.var for tr in self.trackers])
            
            raw_w = self.compute_w(t, np.array([tr.mean for tr in self.trackers]))
            w = self.effective_w(raw_w)
            self.w_history.append(w)

            i = self.select_arm(w)
            r = self.env.pull(i)
            self.N[i] += 1
            self.n_history.append(self.N.copy())

            self.trackers[i].update(r)
            self.rec_history.append(np.argmax([tr.mean for tr in self.trackers]))

        return int(np.argmax([tr.mean for tr in self.trackers]))
    
    def compute_Y(self, t: int, means: np.array) -> float:
        pass

    def plot_diagnostics(self):
        K = self.env.K

        def to_array(hist):
            if len(hist) == 0:
                return np.zeros((0, K))
            arr = np.array(hist)
            # if each entry is a 1D vector of length K -> shape (T, K)
            if arr.ndim == 2 and arr.shape[1] == K:
                return arr
            # try to coerce each element to length-K row
            try:
                rows = [np.array(x).reshape(K,) for x in hist]
                return np.vstack(rows)
            except Exception:
                # fallback: return as-is (may be empty or malformed)
                return arr

        means_arr = to_array(self.means_history)
        vars_arr = to_array(self.vars_history)
        w_arr = to_array(self.w_history)
        scores_arr = to_array(self.scores_history)
        n_arr = to_array(self.n_history)

        fig, axs = plt.subplots(3, 2, figsize=(12, 10))

        # Empirical means
        ax = axs[0, 0]
        for k in range(min(K, means_arr.shape[1] if means_arr.ndim > 1 else 1)):
            ax.plot(means_arr[:, k], label=f'Arm {k}')
        ax.set_title('Empirical Means')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.4)

        # Empirical variances
        ax = axs[0, 1]
        for k in range(min(K, vars_arr.shape[1] if vars_arr.ndim > 1 else 1)):
            ax.plot(vars_arr[:, k], label=f'Arm {k}')
        ax.set_title('Empirical Variances')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.4)

        # w history
        ax = axs[1, 0]
        for k in range(min(K, w_arr.shape[1] if w_arr.ndim > 1 else 1)):
            ax.plot(w_arr[:, k], label=f'Arm {k}')
        ax.set_title('w over Time')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.4)

        # scores history
        ax = axs[1, 1]
        for k in range(min(K, scores_arr.shape[1] if scores_arr.ndim > 1 else 1)):
            ax.plot(scores_arr[:, k], label=f'Arm {k}')
        ax.set_title('Scores over Time')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.4)

        # number of pulls per arm
        ax = axs[2, 0]
        for k in range(min(K, n_arr.shape[1] if n_arr.ndim > 1 else 1)):
            ax.plot(n_arr[:, k], label=f'Arm {k}')
        ax.set_title('Number of Pulls per Arm')
        ax.set_xlabel('Time step')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.4)

        # hide unused subplot if any
        axs[2, 1].axis('off')

        plt.tight_layout()
        plt.show()

# Concrete implementation: VarDE-const
class VarDE_const(VarDEBAI):
    def __init__(self, bandit, T, warm_start = 1,
                 c: float = 4.0, **kwargs):
        super().__init__(bandit, T, warm_start, **kwargs)
        self.c = c

    def compute_w(self, t: int, means: np.array) -> np.array:
        c = self.c
        K = self.K
        w = np.ones(means.shape)
        w /= K + c - 1
        i_max = int(np.argmax(means))
        w[i_max] = c / (K + c - 1)
        return w
    
# Concrete implementation: VarDE-lse
class VarDE_lse(VarDEBAI):
    def __init__(self, bandit, T, warm_start = 1,
                 tau: float = 0.1, **kwargs):
        super().__init__(bandit, T, warm_start, **kwargs)
        self.tau = float(tau)
        if self.tau <= 0:
            raise ValueError("tau must be positive.")

    def compute_w(self, t: int, means: np.array) -> np.array:
        return lse_softmax_weights(means, tau=self.tau)

    def compute_Y(self, t: int, means: np.array) -> float:
        return lse_objective(means, tau=self.tau)
    
# Concrete implementation: VarDE-nesterov
def _project_simplex_1d(y: np.ndarray) -> np.ndarray:
    u = np.sort(y)[::-1]
    cssv = np.cumsum(u)
    j = np.arange(1, y.size + 1)
    cond = u - (cssv - 1) / j > 0
    if not np.any(cond):
        theta = (cssv[-1] - 1) / y.size
    else:
        rho = j[cond][-1]
        theta = (cssv[rho - 1] - 1) / rho
    p = np.maximum(y - theta, 0.0)
    s = p.sum()
    return p / s if s > 0 else np.full_like(y, 1.0 / y.size)

def _ensure_2d(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    return (x[None, :], True) if x.ndim == 1 else (x, False)

def _validate_positive_temperature(tau: float) -> float:
    tau = float(tau)
    if tau <= 0:
        raise ValueError("tau must be positive.")
    return tau

def lse_softmax_weights(means: np.ndarray, tau: float) -> np.ndarray:
    tau = _validate_positive_temperature(tau)
    X, squeeze = _ensure_2d(means)
    shifted = X - np.max(X, axis=1, keepdims=True)
    W = np.exp(shifted / tau)
    W /= np.sum(W, axis=1, keepdims=True)
    return W[0] if squeeze else W

def lse_objective(means: np.ndarray, tau: float):
    tau = _validate_positive_temperature(tau)
    X, squeeze = _ensure_2d(means)
    maxes = np.max(X, axis=1, keepdims=True)
    values = maxes[:, 0] + tau * np.log(np.sum(np.exp((X - maxes) / tau), axis=1))
    return float(values[0]) if squeeze else values

def empirical_mean_sampling_variances(variances: np.ndarray, counts: np.ndarray) -> np.ndarray:
    vars_arr = np.asarray(variances, dtype=float)
    counts_arr = np.asarray(counts, dtype=float)
    if vars_arr.shape != counts_arr.shape:
        raise ValueError("variances and counts must have the same shape.")
    if np.any(counts_arr <= 0):
        raise ValueError("counts must be strictly positive.")
    return np.maximum(vars_arr, 0.0) / counts_arr

def lse_first_order_decision_variance(
    means: np.ndarray,
    variances: np.ndarray,
    counts: np.ndarray,
    tau: float,
) -> float:
    weights = lse_softmax_weights(means, tau=tau)
    sampling_vars = empirical_mean_sampling_variances(variances, counts)
    return float(np.sum((weights ** 2) * sampling_vars))

def sample_empirical_mean_distribution(
    means: np.ndarray,
    variances: np.ndarray,
    counts: np.ndarray,
    num_samples: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if num_samples < 2:
        raise ValueError("num_samples must be at least 2.")
    means_arr = np.asarray(means, dtype=float)
    # Empirical sampling law of the arm means: mu_tilde_i ~ N(mu_hat_i, sigma_hat_i^2 / N_i).
    sampling_vars = empirical_mean_sampling_variances(variances, counts)
    if not np.all(np.isfinite(means_arr)):
        raise ValueError("means must be finite.")
    if not np.all(np.isfinite(sampling_vars)):
        raise ValueError("sampling variances must be finite.")
    rng = np.random.default_rng() if rng is None else rng
    return rng.normal(
        loc=means_arr,
        scale=np.sqrt(sampling_vars),
        size=(int(num_samples), means_arr.size),
    )

def lse_full_decision_variance(
    means: np.ndarray,
    variances: np.ndarray,
    counts: np.ndarray,
    tau: float,
    num_samples: int = 4096,
    rng: np.random.Generator | None = None,
) -> float:
    draws = sample_empirical_mean_distribution(
        means=means,
        variances=variances,
        counts=counts,
        num_samples=num_samples,
        rng=rng,
    )
    values = lse_objective(draws, tau=tau)
    return float(np.var(values, ddof=1))

def lse_decision_variance_gap(
    means: np.ndarray,
    variances: np.ndarray,
    counts: np.ndarray,
    tau: float,
    num_samples: int = 4096,
    rng: np.random.Generator | None = None,
    eps: float = 1e-12,
) -> Dict[str, float]:
    if eps <= 0:
        raise ValueError("eps must be positive.")
    V_1st = lse_first_order_decision_variance(
        means=means,
        variances=variances,
        counts=counts,
        tau=tau,
    )
    V_full = lse_full_decision_variance(
        means=means,
        variances=variances,
        counts=counts,
        tau=tau,
        num_samples=num_samples,
        rng=rng,
    )
    abs_gap = abs(V_full - V_1st)
    rel_gap = abs_gap / max(V_full, eps)
    return {
        "V_1st": V_1st,
        "V_full": V_full,
        "abs_gap": abs_gap,
        "rel_gap": rel_gap,
    }

def lse_decision_variance_gap_history(
    means_history: np.ndarray,
    variances_history: np.ndarray,
    counts_history: np.ndarray,
    tau: float,
    num_samples: int = 4096,
    seed: int | None = None,
    eps: float = 1e-12,
) -> Dict[str, np.ndarray]:
    means_arr = np.asarray(means_history, dtype=float)
    vars_arr = np.asarray(variances_history, dtype=float)
    counts_arr = np.asarray(counts_history, dtype=float)
    if means_arr.shape != vars_arr.shape or means_arr.shape != counts_arr.shape:
        raise ValueError("means, variances, and counts histories must share the same shape.")

    T = means_arr.shape[0]
    metrics = {
        "V_1st": np.full(T, np.nan, dtype=float),
        "V_full": np.full(T, np.nan, dtype=float),
        "abs_gap": np.full(T, np.nan, dtype=float),
        "rel_gap": np.full(T, np.nan, dtype=float),
    }
    rng = np.random.default_rng(seed)

    for t in range(T):
        means_t = means_arr[t]
        vars_t = vars_arr[t]
        counts_t = counts_arr[t]
        if not np.all(np.isfinite(means_t)) or not np.all(np.isfinite(vars_t)):
            continue
        if np.any(counts_t <= 0):
            continue
        try:
            step_metrics = lse_decision_variance_gap(
                means=means_t,
                variances=vars_t,
                counts=counts_t,
                tau=tau,
                num_samples=num_samples,
                rng=rng,
                eps=eps,
            )
        except ValueError:
            continue
        for key, value in step_metrics.items():
            metrics[key][t] = value
    return metrics

def varde_w_nesterov(means: np.ndarray, mu: float = 1.0) -> np.ndarray:
    if mu <= 0:
        raise ValueError("mu must be positive.")
    X, squeeze = _ensure_2d(means)
    W = np.vstack([_project_simplex_1d(row / mu) for row in X])
    return W[0] if squeeze else W

class VarDE_nesterov(VarDEBAI):
    def __init__(self, bandit, T, warm_start = 1, mu: float = 1.0, **kwargs):
        super().__init__(bandit, T, warm_start, **kwargs)
        self.mu = mu

    def compute_w(self, t: int, means: np.array) -> np.array:
        return varde_w_nesterov(means, mu=self.mu)
    
# Concrete implementation: VarDE-entmax
def _entmax_map_1d(x: np.ndarray, alpha: float, mu: float) -> np.ndarray:
    if not (1.0 < alpha <= 2.0):
        raise ValueError("alpha must be in (1, 2].")
    K = x.size
    c = ((alpha - 1.0) / (mu * alpha)) ** (1.0 / (alpha - 1.0))
    lo, hi = x.min() - 1e3, x.max()
    for _ in range(60):
        tau = 0.5 * (lo + hi)
        z = np.maximum(c * (x - tau), 0.0)
        p = z if alpha == 2.0 else z ** (1.0 / (alpha - 1.0))
        if p.sum() > 1.0: lo = tau
        else:             hi = tau
    tau = 0.5 * (lo + hi)
    z = np.maximum(c * (x - tau), 0.0)
    p = z if alpha == 2.0 else z ** (1.0 / (alpha - 1.0))
    s = p.sum()
    return p / s if s > 0 else np.full_like(x, 1.0 / K)

def varde_w_entmax(means: np.ndarray, mu: float = 1.0, alpha: float = 1.5) -> np.ndarray:
    X, squeeze = _ensure_2d(means)
    W = np.vstack([_entmax_map_1d(row, alpha=alpha, mu=mu) for row in X])
    return W[0] if squeeze else W

class VarDE_entmax(VarDEBAI):
	def __init__(self, bandit, T, warm_start = 1, mu: float = 1.0, alpha: float = 1.5, **kwargs):
		super().__init__(bandit, T, warm_start, **kwargs)
		self.mu = mu
		self.alpha = alpha

	def compute_w(self, t: int, means: np.array) -> np.array:
		return varde_w_entmax(means, mu=self.mu, alpha=self.alpha)
    
# Concrete implementation: VarDE-pairwise-softplus
def _pair_softplus_merge(a: float, b: float, delta: float):
    # s = (a+b)/2 + sqrt(((a-b)/2)^2 + delta^2)
    d = 0.5 * (a - b)
    r = np.sqrt(d * d + delta * delta)
    s = 0.5 * (a + b) + r
    if r == 0.0:
        da = db = 0.5
    else:
        da = 0.5 + (a - b) / (2.0 * r)
    db = 1.0 - da
    return s, da, db

def varde_w_pairwise_softplus(means: np.ndarray, delta: float = 0.1, order: str = "balanced") -> np.ndarray:
    X, squeeze = _ensure_2d(means)
    B, K = X.shape
    W = np.zeros_like(X)
    for b in range(B):
        x = X[b]
        if order == "balanced" and K > 1:
            nodes = [(x[i], np.eye(K)[i]) for i in range(K)]
            while len(nodes) > 1:
                nxt = []
                for i in range(0, len(nodes), 2):
                    if i + 1 < len(nodes):
                        (va, ga), (vb, gb) = nodes[i], nodes[i+1]
                        s, da, db = _pair_softplus_merge(va, vb, delta)
                        g = da * ga + db * gb
                        nxt.append((s, g))
                    else:
                        nxt.append(nodes[i])
                nodes = nxt
            w = nodes[0][1]
        else:
            s = x[0]; g = np.zeros(K); g[0] = 1.0
            for j in range(1, K):
                s, da, db = _pair_softplus_merge(s, x[j], delta)
                g = da * g; g[j] += db
            w = g
        ssum = w.sum()
        W[b] = w / ssum if ssum > 0 else np.full(K, 1.0 / K)
    return W[0] if squeeze else W

class VarDE_pairwise_softplus(VarDEBAI):
	def __init__(self, bandit, T, warm_start = 1, delta: float = 0.1, order: str = "balanced", **kwargs):
		super().__init__(bandit, T, warm_start, **kwargs)
		self.delta = delta
		self.order = order

	def compute_w(self, t: int, means: np.array) -> np.array:
		return varde_w_pairwise_softplus(means, delta=self.delta, order=self.order)
    
# Concrete implementation: VarDE-power-mean
def varde_w_power_mean(means: np.ndarray, p: float = 4.0, eps: float = 1e-8) -> np.ndarray:
    if p <= 1.0:
        raise ValueError("p must be > 1 for max-like behavior.")
    X, squeeze = _ensure_2d(means)
    B, K = X.shape
    W = np.zeros_like(X)
    for b in range(B):
        x = X[b]
        m = x.min()
        if m <= 0: x = x - m + eps
        s_p = np.mean(x ** p)
        base = (s_p) ** (1.0 / p - 1.0)
        w = (x ** (p - 1.0)) * base / K
        ssum = w.sum()
        W[b] = w / ssum if ssum > 0 else np.full(K, 1.0 / K)
    return W[0] if squeeze else W

class VarDE_power_mean(VarDEBAI):
	def __init__(self, bandit, T, warm_start = 1, p: float = 4.0, eps: float = 1e-8, **kwargs):
		super().__init__(bandit, T, warm_start, **kwargs)
		self.p = p
		self.eps = eps

	def compute_w(self, t: int, means: np.array) -> np.array:
		return varde_w_power_mean(means, p=self.p, eps=self.eps)
