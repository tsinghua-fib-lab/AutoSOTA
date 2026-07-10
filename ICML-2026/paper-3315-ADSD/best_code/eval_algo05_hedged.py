"""
ALGO-05: Hedged Betting with Predictable lambda_t Sequence
Adapts lambda at each round based on recent evidence strength.
lambda_t = lambda_base * (1 + clip(avg_log_growth, -0.5, 1.0))
Predictable (F_{t-1}-measurable), preserving supermartingale property.
"""
import numpy as np
import time
import sys

U1 = np.array([[0.9, 0.2], [0.3, 0.7]])
U2 = np.array([[0.5, 0.3], [0.2, 0.7]])
pi1_ne = np.array([5/7, 2/7])
pi2_ne = np.array([5/11, 6/11])

pi_alts = {
    0.05: (np.array([0.9, 0.1]), np.array([10/11, 1/11])),
    0.10: (np.array([0.8, 0.2]), np.array([10/11, 1/11])),
    0.15: (np.array([0.7, 0.3]), np.array([10/11, 1/11])),
}

A_sizes = [2, 2]
m = sum(A_sizes)
T = 4000
R = 300

# Parse CLI
mode = sys.argv[1] if len(sys.argv) > 1 else 'both'
lambda_base = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
eta = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 42

# Hedging parameters
WINDOW = 50  # window for computing log-growth
CLIP_MIN = -0.5
CLIP_MAX = 2.0  # allow lambda up to 3x base
LAMBDA_MIN = 0.01
LAMBDA_MAX = 0.5

threshold = m / alpha


def compute_lambda_t(log_growth_history, t):
    """Compute adaptive lambda_t based on recent log-wealth growth."""
    if t <= WINDOW + 1:
        return lambda_base
    recent = log_growth_history[t - WINDOW:t]
    avg_growth = np.mean(recent)
    # Clip and scale
    factor = 1.0 + np.clip(avg_growth * 10, CLIP_MIN, CLIP_MAX)
    lam = lambda_base * factor
    return np.clip(lam, LAMBDA_MIN, LAMBDA_MAX)


def run_fwer_hedged(seed):
    """FWER under H0 using hedged lambda_t."""
    np.random.seed(seed)
    M1_all = np.ones((R, 2, T + 1))
    M2_all = np.ones((R, 2, T + 1))
    lambda_history = np.zeros((R, T + 1))

    for r in range(R):
        M1 = np.ones((2, T + 1))
        M2 = np.ones((2, T + 1))
        log_growth = np.zeros(T + 1)  # track log-growth for adaptation
        for t in range(1, T + 1):
            lam_t = compute_lambda_t(log_growth, t)
            lambda_history[r, t] = lam_t
            a1 = np.random.choice([0, 1], p=pi1_ne)
            a2 = np.random.choice([0, 1], p=pi2_ne)
            for a1_prime in [0, 1]:
                X1 = U1[a1, a2] - U1[a1_prime, a2]
                M1[a1_prime, t] = M1[a1_prime, t - 1] * (1 - lam_t * X1)
            for a2_prime in [0, 1]:
                X2 = U2[a1, a2] - U2[a1, a2_prime]
                M2[a2_prime, t] = M2[a2_prime, t - 1] * (1 - lam_t * X2)
            # Track log-growth: use log of max martingale growth
            max_M_prev = max(np.max(M1[:, t - 1]), np.max(M2[:, t - 1]))
            max_M_curr = max(np.max(M1[:, t]), np.max(M2[:, t]))
            if max_M_prev > 0:
                log_growth[t] = np.log(max_M_curr / max_M_prev)
        M1_all[r] = M1
        M2_all[r] = M2

    max_per_run = np.maximum(np.max(M1_all, axis=(1, 2)), np.max(M2_all, axis=(1, 2)))
    rejections = np.sum(max_per_run >= threshold)
    avg_lambda = np.mean(lambda_history[:, 1:])
    return rejections / R, int(rejections), avg_lambda


def run_detection_hedged(eta, seed):
    """Detection time under H1 using hedged lambda_t."""
    np.random.seed(seed)
    pi1_alt, pi2_alt = pi_alts[eta]
    tau_ubs = np.full(R, T)

    for r in range(R):
        M1 = np.ones((2, T + 1))
        M2 = np.ones((2, T + 1))
        log_growth = np.zeros(T + 1)
        for t in range(1, T + 1):
            if tau_ubs[r] < T:
                break
            lam_t = compute_lambda_t(log_growth, t)
            a1 = np.random.choice([0, 1], p=pi1_alt)
            a2 = np.random.choice([0, 1], p=pi2_alt)
            for a1_prime in [0, 1]:
                X1 = U1[a1, a2] - U1[a1_prime, a2]
                M1[a1_prime, t] = M1[a1_prime, t - 1] * (1 - lam_t * X1)
            for a2_prime in [0, 1]:
                X2 = U2[a1, a2] - U2[a1, a2_prime]
                M2[a2_prime, t] = M2[a2_prime, t - 1] * (1 - lam_t * X2)

            max_M_prev = max(np.max(M1[:, t - 1]), np.max(M2[:, t - 1]))
            max_M_curr = max(np.max(M1[:, t]), np.max(M2[:, t]))
            if max_M_prev > 0:
                log_growth[t] = np.log(max_M_curr / max_M_prev)

            if np.any(M1[:, t] >= threshold) or np.any(M2[:, t] >= threshold):
                tau_ubs[r] = t

    avg_tau = np.mean(tau_ubs)
    q1 = np.percentile(tau_ubs, 25)
    q3 = np.percentile(tau_ubs, 75)
    med = np.median(tau_ubs)
    detected = int(np.sum(tau_ubs < T))
    return avg_tau, q1, q3, med, detected


print("=" * 60)
print("ALGO-05: Hedged Betting with Predictable lambda_t")
print("Base lambda: {:.2f}, Window: {}, Clip: [{:.1f}, {:.1f}]".format(
    lambda_base, WINDOW, CLIP_MIN, CLIP_MAX))
print("alpha={:.2f} threshold={:.1f}".format(alpha, threshold))
print("=" * 60)

if mode in ('fwer', 'both'):
    print()
    print("--- FWER (H0) ---")
    start = time.time()
    fwer, rej, avg_lam = run_fwer_hedged(seed)
    elapsed = time.time() - start
    print("Empirical FWER: {:.3f}".format(fwer))
    print("Rejections: {} / {}".format(rej, R))
    print("FWER <= alpha ({:.2f}): {}".format(alpha, fwer <= alpha))
    print("Avg lambda used: {:.4f}".format(avg_lam))
    print("Time: {:.1f}s".format(elapsed))

if mode in ('detection', 'both'):
    print()
    print("--- Detection Time (H1, eta={:.2f}) ---".format(eta))
    start = time.time()
    avg_tau, q1, q3, med, detected = run_detection_hedged(eta, seed + 1000)
    elapsed = time.time() - start
    print("Avg Detection Time: {:.1f}".format(avg_tau))
    print("Median: {:.1f}".format(med))
    print("Q1: {:.1f}".format(q1))
    print("Q3: {:.1f}".format(q3))
    print("Detected: {} / {}".format(detected, R))
    print("Detection Rate: {:.3f}".format(detected / R))
    print("Time: {:.1f}s".format(elapsed))
