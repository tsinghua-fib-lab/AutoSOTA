"""
ALGO-01: Mixture-of-lambda Supermartingales (Paper Corollary 2.5)
Uses discrete mixture over K lambda values instead of single fixed lambda.
M_t = (1/K) * sum_k M_t^{(lambda_k)} is a valid supermartingale.
Evaluates both FWER (H0) and detection time (H1).
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

# Mixture of lambda values (paper's grid)
MIXTURE_LAMBDAS = np.array([0.05, 0.10, 0.15, 0.40])
K = len(MIXTURE_LAMBDAS)

# Parse CLI args
mode = sys.argv[1] if len(sys.argv) > 1 else 'both'
alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
eta = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42

threshold = m / alpha


def run_fwer_mixture(seed):
    """FWER under H0 using mixture-of-lambda supermartingales."""
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

    # Mixture: average across K components
    M1_mixture = np.mean(M1_all, axis=1)  # (R, 2, T+1)
    M2_mixture = np.mean(M2_all, axis=1)  # (R, 2, T+1)

    max_per_run = np.maximum(
        np.max(M1_mixture, axis=(1, 2)),
        np.max(M2_mixture, axis=(1, 2))
    )
    rejections = np.sum(max_per_run >= threshold)
    return rejections / R, int(rejections)


def run_detection_mixture(eta, seed):
    """Detection time under H1 using mixture-of-lambda supermartingales."""
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

            # Mixture: average across K components
            M1_mix_t = np.mean(M1[:, :, t], axis=0)
            M2_mix_t = np.mean(M2[:, :, t], axis=0)

            if np.any(M1_mix_t >= threshold) or np.any(M2_mix_t >= threshold):
                tau_ubs[r] = t

    avg_tau = np.mean(tau_ubs)
    q1 = np.percentile(tau_ubs, 25)
    q3 = np.percentile(tau_ubs, 75)
    med = np.median(tau_ubs)
    detected = int(np.sum(tau_ubs < T))
    return avg_tau, q1, q3, med, detected


print("=" * 60)
print("ALGO-01: Mixture-of-lambda Supermartingales")
print("Mixture lambdas: {}".format(MIXTURE_LAMBDAS.tolist()))
print("alpha={:.2f} threshold={:.1f}".format(alpha, threshold))
print("=" * 60)

if mode in ('fwer', 'both'):
    print()
    print("--- FWER (H0) ---")
    start = time.time()
    fwer, rej = run_fwer_mixture(seed)
    elapsed = time.time() - start
    print("Empirical FWER: {:.3f}".format(fwer))
    print("Rejections: {} / {}".format(rej, R))
    print("FWER <= alpha ({:.2f}): {}".format(alpha, fwer <= alpha))
    print("Time: {:.1f}s".format(elapsed))

if mode in ('detection', 'both'):
    print()
    print("--- Detection Time (H1, eta={:.2f}) ---".format(eta))
    start = time.time()
    avg_tau, q1, q3, med, detected = run_detection_mixture(eta, seed + 1000)
    elapsed = time.time() - start
    print("Avg Detection Time: {:.1f}".format(avg_tau))
    print("Median: {:.1f}".format(med))
    print("Q1: {:.1f}".format(q1))
    print("Q3: {:.1f}".format(q3))
    print("Detected: {} / {}".format(detected, R))
    print("Detection Rate: {:.3f}".format(detected / R))
    print("Time: {:.1f}s".format(elapsed))
