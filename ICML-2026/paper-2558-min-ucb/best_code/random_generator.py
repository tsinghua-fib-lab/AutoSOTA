import numpy as np

# --- 1) sample positive stable distribution S(α)（Chambers–Mallows–Stuck, α∈[0,1]）---
def positive_stable(alpha, size=None, seed=2349, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    U = rng.random(size) * np.pi/2  # Uniform(0, pi/2)
    W = rng.exponential(scale=1.0, size=size)  # Exp(1)
    # Chambers-Mallows-Stuck formula for positive stable
    # V = (sin(alpha*U) / (cos(U))**(1/alpha)) * ( cos(U - alpha*U) / W )**((1-alpha)/alpha)
    s1 = np.sin(alpha * U) / (np.cos(U))**(1/alpha)
    s2 = np.cos(U - alpha * U) / W
    V = (s1 * (s2)**((1-alpha)/alpha))
    return V

# --- 2) sample from Gumbel copula ---
def sample_gumbel_copula(n, d, lam, seed=2349, rng=None):
    """
    返回形状 (n, d) 的 U∈(0,1)^d,服从 Gumbel(λ) copula,λ>=1
    """
    if rng is None:
        rng = np.random.default_rng()
    alpha = 1.0 / lam
    V = positive_stable(alpha, size=n, rng=rng)  # (n,)
    E = rng.exponential(scale=1.0, size=(n, d))  # (n,d)
    U = np.exp(- (E / V[:, None])**alpha)
    return U

def map_to_uniform_intervals(U, lows, highs):
    """
    U: (n,d) in (0,1); lows, highs: (d,) 对应每一维的区间端点
    """
    lows = np.asarray(lows); highs = np.asarray(highs)
    return lows + (highs - lows) * U

def sample_h_q_w(n, J, p, lam_r, lam_w,
                 h_int_r, q_ints_r, w_ints_r,
                 h_int_w, q_ints_w, w_ints_w,
                 seed=2349, rng=None):
    """
    n: 样本数; J: 部门个数; p: 常态期概率
    lam_r/lam_w: Gumbel 参数 λ^r / λ^w (>=1)
    h_int_*: (low, high)
    q_ints_*, w_ints_*: List of J, each element is (low, high)
    return: dict 含 h, q (n,J), w (n,J)
    """
    if rng is None:
        rng = np.random.default_rng(seed=seed)

    d = 1 + J + J
    regime = rng.random(n) < p  

    n_r = int(np.sum(regime))
    n_w = n - n_r
    U_r = sample_gumbel_copula(n_r, d, lam_r, rng=rng) if n_r>0 else np.empty((0,d))
    U_w = sample_gumbel_copula(n_w, d, lam_w, rng=rng) if n_w>0 else np.empty((0,d))

    U = np.empty((n, d))
    U[regime] = U_r
    U[~regime] = U_w

    def build_intervals(h_int, q_ints, w_ints):
        lows = [h_int[0]] + [a for a,b in q_ints] + [a for a,b in w_ints]
        highs= [h_int[1]] + [b for a,b in q_ints] + [b for a,b in w_ints]
        return np.array(lows), np.array(highs)
    lows_r, highs_r = build_intervals(h_int_r, q_ints_r, w_ints_r)
    lows_w, highs_w = build_intervals(h_int_w, q_ints_w, w_ints_w)

    X = np.empty((n, d))
    if n_r>0:
        X[regime] = map_to_uniform_intervals(U[regime], lows_r, highs_r)
    if n_w>0:
        X[~regime] = map_to_uniform_intervals(U[~regime], lows_w, highs_w)

    h = X[:, 0]
    q = X[:, 1:1+J]
    w = X[:, 1+J:1+2*J]

    return dict(q=q,W=w,h=h,T=None)
