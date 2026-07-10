"""
ALGO-06: Cross-Hypothesis Evidence Pooling with Mixture-of-lambda
Requires multiple supermartingales to agree before rejecting.
Tests: (a) second-largest also exceeds threshold/2, (b) at least 2 of 4 exceed threshold.
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

MIXTURE_LAMBDAS = np.array([0.05, 0.10, 0.15, 0.40])
K = len(MIXTURE_LAMBDAS)
WEIGHTS = np.ones(K) / K

# Parse CLI
mode = sys.argv[1] if len(sys.argv) > 1 else 'both'
pooling = sys.argv[2] if len(sys.argv) > 2 else 'second_half'  # 'none', 'second_half', 'at_least_2'
alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
eta = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 42

threshold = m / alpha
second_threshold = threshold / 2.0  # for second-largest check


def check_rejection(M1_t, M2_t):
    """Check if any hypothesis triggers rejection under the pooling rule."""
    all_vals = np.array([
        M1_t[0], M1_t[1], M2_t[0], M2_t[1]
    ])
    max_val = np.max(all_vals)

    if pooling == 'none':
        return max_val >= threshold
    elif pooling == 'second_half':
        if max_val < threshold:
            return False
        # Also require second-largest >= threshold/2
        sorted_vals = np.sort(all_vals)[::-1]
        return sorted_vals[1] >= second_threshold
    elif pooling == 'at_least_2':
        return np.sum(all_vals >= threshold) >= 2
    return max_val >= threshold


def run_fwer_mixture_pooled(seed):
    np.random.seed(seed)
    M1_all = np.ones((R, K, 2, T + 1))
    M2_all = np.ones((R, K, 2, T + 1))
    for r in range(R):
        M1 = np.ones((K, 2, T + 1))
        M2 = np.ones((K, 2, T + 1))
        for t in range(1, T + 1):
            a1 = np.random.choice([0, 1], p=pi1_ne)
            a2 = np.random.choice([0, 1], p=pi2_ne)
            for k in range(K):
                lam = MIXTURE_LAMBDAS[k]
                for a1_prime in [0, 1]:
                    X1 = U1[a1, a2] - U1[a1_prime, a2]
                    M1[k, a1_prime, t] = M1[k, a1_prime, t - 1] * (1 - lam * X1)
                for a2_prime in [0, 1]:
                    X2 = U2[a1, a2] - U2[a1, a2_prime]
                    M2[k, a2_prime, t] = M2[k, a2_prime, t - 1] * (1 - lam * X2)
        M1_all[r] = M1
        M2_all[r] = M2

    M1_mixture = np.average(M1_all, axis=1, weights=WEIGHTS)
    M2_mixture = np.average(M2_all, axis=1, weights=WEIGHTS)

    # Apply pooling check
    rejections = 0
    for r in range(R):
        for t in range(1, T + 1):
            if check_rejection(M1_mixture[r, :, t], M2_mixture[r, :, t]):
                rejections += 1
                break

    return rejections / R, rejections


def run_detection_mixture_pooled(eta, seed):
    np.random.seed(seed)
    pi1_alt, pi2_alt = pi_alts[eta]
    tau_ubs = np.full(R, T)
    for r in range(R):
        M1 = np.ones((K, 2, T + 1))
        M2 = np.ones((K, 2, T + 1))
        for t in range(1, T + 1):
            if tau_ubs[r] < T:
                break
            a1 = np.random.choice([0, 1], p=pi1_alt)
            a2 = np.random.choice([0, 1], p=pi2_alt)
            for k in range(K):
                lam = MIXTURE_LAMBDAS[k]
                for a1_prime in [0, 1]:
                    X1 = U1[a1, a2] - U1[a1_prime, a2]
                    M1[k, a1_prime, t] = M1[k, a1_prime, t - 1] * (1 - lam * X1)
                for a2_prime in [0, 1]:
                    X2 = U2[a1, a2] - U2[a1, a2_prime]
                    M2[k, a2_prime, t] = M2[k, a2_prime, t - 1] * (1 - lam * X2)

            M1_mix_t = np.average(M1[:, :, t], axis=0, weights=WEIGHTS)
            M2_mix_t = np.average(M2[:, :, t], axis=0, weights=WEIGHTS)

            if check_rejection(M1_mix_t, M2_mix_t):
                tau_ubs[r] = t

    avg_tau = np.mean(tau_ubs)
    q1 = np.percentile(tau_ubs, 25)
    q3 = np.percentile(tau_ubs, 75)
    med = np.median(tau_ubs)
    detected = int(np.sum(tau_ubs < T))
    return avg_tau, q1, q3, med, detected


print("=" * 60)
print("ALGO-06: Cross-Hypothesis Pooling + Mixture")
print("Pooling rule: {}".format(pooling))
print("Mixture lambdas: {}".format(MIXTURE_LAMBDAS.tolist()))
print("alpha={:.2f} threshold={:.1f} second_threshold={:.1f}".format(alpha, threshold, second_threshold))
print("=" * 60)

if mode in ('fwer', 'both'):
    print()
    print("--- FWER (H0) ---")
    start = time.time()
    fwer, rej = run_fwer_mixture_pooled(seed)
    elapsed = time.time() - start
    print("Empirical FWER: {:.3f}".format(fwer))
    print("Rejections: {} / {}".format(rej, R))
    print("FWER <= alpha ({:.2f}): {}".format(alpha, fwer <= alpha))
    print("Time: {:.1f}s".format(elapsed))

if mode in ('detection', 'both'):
    print()
    print("--- Detection Time (H1, eta={:.2f}) ---".format(eta))
    start = time.time()
    avg_tau, q1, q3, med, detected = run_detection_mixture_pooled(eta, seed + 1000)
    elapsed = time.time() - start
    print("Avg Detection Time: {:.1f}".format(avg_tau))
    print("Median: {:.1f}".format(med))
    print("Q1: {:.1f}".format(q1))
    print("Q3: {:.1f}".format(q3))
    print("Detected: {} / {}".format(detected, R))
    print("Detection Rate: {:.3f}".format(detected / R))
    print("Time: {:.1f}s".format(elapsed))
