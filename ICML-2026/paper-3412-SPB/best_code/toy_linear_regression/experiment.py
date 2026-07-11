import numpy as np
from dataclasses import dataclass

# =========================
# Configuration
# =========================

@dataclass
class Config:
    n_samples: int = 2000
    test_ratio: float = 0.2
    n_runs: int = 10
    laplace_scale: float = 5.0  # b parameter
    noise_sigma: float = 0.1
    epsilon: float = 1e-2
    alpha_star: float = 3.0
    beta_star: float = 1.0
    seed: int = 42

    # PAC-Bayes parameters
    sigma_p2: float = 0.2
    sigma_q2: float = 0.2
    delta: float = 0.05


# =========================
# Data Generation
# =========================

def sample_data(cfg: Config, rng: np.random.Generator):
    # Sample R ~ Laplace(0, b)
    R = rng.laplace(loc=0.0, scale=cfg.laplace_scale, size=cfg.n_samples)

    # Sample theta ~ Uniform(0, 2pi)
    theta = rng.uniform(0.0, 2 * np.pi, size=cfg.n_samples)

    # Convert to Cartesian coordinates
    X1 = R * np.cos(theta)
    X2 = R * np.sin(theta)

    # Safeguard: remove near-zero coordinates
    mask = (np.abs(X1) > cfg.epsilon) & (np.abs(X2) > cfg.epsilon)
    X1, X2 = X1[mask], X2[mask]

    # True target function
    Y = cfg.alpha_star * np.sin(X1 / X2) + cfg.beta_star

    # Add Gaussian noise
    Y += rng.normal(0.0, cfg.noise_sigma, size=Y.shape)

    X = np.stack([X1, X2], axis=1)
    return X, Y


# =========================
# Feature Map
# =========================

def feature_map(X):
    x1 = X[:, 0]
    x2 = X[:, 1]

    phi1 = np.sin(x1 / x2)
    phi2 = np.sin(x2 / x1)
    phi3 = np.sin(x1)
    phi4 = np.sin(x2)
    phi5 = np.ones_like(x1)

    return np.stack([phi1, phi2, phi3, phi4, phi5], axis=1)


# =========================
# Linear Regression (Closed Form)
# =========================

def fit_linear_regression(Phi, Y):
    # w = (Phi^T Phi)^(-1) Phi^T Y
    return np.linalg.pinv(Phi.T @ Phi) @ Phi.T @ Y


def predict(Phi, w):
    return Phi @ w


# =========================
# Train/Test Split
# =========================

def train_test_split(X, Y, cfg: Config, rng: np.random.Generator):
    n = X.shape[0]
    indices = rng.permutation(n)

    split = int(n * (1 - cfg.test_ratio))
    train_idx, test_idx = indices[:split], indices[split:]

    return X[train_idx], X[test_idx], Y[train_idx], Y[test_idx]


# =========================
# Metrics
# =========================

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# =========================
# PAC-Bayes Components
# =========================

def kl_gaussian(w_p, w_q , sigma_p2, sigma_q2):
    d = len(w_q)

    return 0.5 * (
        np.sum((w_p - w_q)**2) / sigma_q2
        + d * (sigma_p2 / sigma_q2)
        - d
        + d * np.log(sigma_q2 / sigma_p2)
    )


def mcallester_bound(emp_risk, kl, n, delta):
    return emp_risk + np.sqrt((kl + np.log(1 / delta)) / (2 * n))

def estimate_risk(w_mean, sigma2, phi, y_true, n_post=50, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    risks = []

    for _ in range(n_post):
        # sample weight from posterior
        w_sample = rng.normal(loc=w_mean, scale=np.sqrt(sigma2), size=len(w_mean))

        y_pred = phi @ w_sample
        risks.append(mse(y_true, y_pred))

    return np.mean(risks)

# =========================
# Experiment Loop
# =========================

def run_experiment(cfg: Config):
    results = []
    rng = np.random.default_rng(cfg.seed)

    for run in range(cfg.n_runs):
        # Independent RNG per run for reproducibility
        run_rng = np.random.default_rng(rng.integers(1e9))

        X, Y = sample_data(cfg, run_rng)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, cfg, run_rng)

        Phi_train = feature_map(X_train)
        Phi_test = feature_map(X_test)

        w_post = fit_linear_regression(Phi_train, Y_train)

        train_pred = predict(Phi_train, w_post)
        test_pred = predict(Phi_test, w_post)

        train_mse = mse(Y_train, train_pred)
        test_mse = mse(Y_test, test_pred)

        n = len(Y_train)

        # --- Prior ---
        w_prior = run_rng.standard_normal(size=w_post.shape)

        # --- Standard PAC-Bayes ---
        kl_full = kl_gaussian(w_prior, w_post, cfg.sigma_p2, cfg.sigma_q2)
        mse_post_full = estimate_risk(w_post, cfg.sigma_q2, Phi_train, Y_train, n_post=100, rng=run_rng)

        bound_full = mcallester_bound(mse_post_full, kl_full, n, cfg.delta)

        risk_post_full = estimate_risk(w_post, cfg.sigma_q2, Phi_test, Y_test, n_post=100, rng=run_rng)

        # --- Symmetry-aware (projection to a,b,e) ---
        w_prior_sym = w_prior[[0, 1, 4]]
        w_post_sym = w_post[[0, 1, 4]]
        Phi_train_sym = Phi_train[:, [0, 1, 4]]
        kl_sym = kl_gaussian(w_prior_sym, w_post_sym, cfg.sigma_p2, cfg.sigma_q2)
        mse_post_sym = estimate_risk(w_post_sym, cfg.sigma_q2, Phi_train_sym, Y_train, n_post=100, rng=run_rng)

        bound_sym = mcallester_bound(mse_post_sym, kl_sym, n, cfg.delta)

        Phi_test_sym = Phi_test[:, [0, 1, 4]]
        risk_post_sym = estimate_risk(w_post_sym, cfg.sigma_q2, Phi_test_sym, Y_test, n_post=100, rng=run_rng)

        results.append({
            "run": run,
            "expected_emp_risk": mse_post_full,
            "expected_true_risk": risk_post_full,
            "expected_emp_risk_sym": mse_post_sym,
            "expected_true_risk_sym": risk_post_sym,
            "bound_full": bound_full,
            "bound_sym": bound_sym,
            "kl_full": kl_full,
            "kl_sym": kl_sym,
            "weights": w_post
        })

    return results


# =========================
# Summary
# =========================

def summarize(results):
    exp_emp_risk = np.array([r["expected_emp_risk"] for r in results])
    exp_true_risk = np.array([r["expected_true_risk"] for r in results])
    exp_emp_risk_sym = np.array([r["expected_emp_risk_sym"] for r in results])
    exp_true_risk_sym = np.array([r["expected_true_risk_sym"] for r in results])
    weights = np.array([r["weights"] for r in results])
    bound_full = np.array([r["bound_full"] for r in results])
    bound_sym = np.array([r["bound_sym"] for r in results])
    kl_full = np.array([r["kl_full"] for r in results])
    kl_sym = np.array([r["kl_sym"] for r in results])

    print("==== Performance ====")
    print(f"Expected empirical risk: mean={exp_emp_risk.mean():.4f}, std={exp_emp_risk.std():.4f}")
    print(f"Expected true risk: mean={exp_true_risk.mean():.4f}, std={exp_true_risk.std():.4f}")
    print("==== Performance with symmetry ====")
    print(f"Expected empirical risk: mean={exp_emp_risk_sym.mean():.4f}, std={exp_emp_risk_sym.std():.4f}")
    print(f"Expected true risk: mean={exp_true_risk_sym.mean():.4f}, std={exp_true_risk_sym.std():.4f}")

    print("\n==== PAC-Bayes Bounds ====")
    print(f"Standard KL: mean={kl_full.mean():.4f}")
    print(f"Symmetric KL: mean={kl_sym.mean():.4f}")
    print(f"Standard bound: mean={bound_full.mean():.4f}")
    print(f"Symmetry bound: mean={bound_sym.mean():.4f}")

    improvement = bound_full - bound_sym
    print("\n==== Comparison ====")
    print(f"Mean improvement: {improvement.mean():.4f}")
    print(f"Fraction sym < std: {(improvement > 0).mean():.2f}")

    print("\n==== Learned Weights ====")
    print(f"weights={weights.mean(axis=0)}")


# =========================
# Main
# =========================

if __name__ == "__main__":
    cfg = Config()
    results = run_experiment(cfg)
    summarize(results)
