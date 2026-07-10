"""
ALGO-07: Block-Aggregated E-Values
Aggregates e-values over blocks of B rounds; all hypotheses share same actions.
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

mode = sys.argv[1] if len(sys.argv) > 1 else 'both'
lambda_val = float(sys.argv[2]) if len(sys.argv) > 2 else 0.10
block_size = int(sys.argv[3]) if len(sys.argv) > 3 else 5
alpha = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2
eta = float(sys.argv[5]) if len(sys.argv) > 5 else 0.05
seed = int(sys.argv[6]) if len(sys.argv) > 6 else 42

threshold = m / alpha
num_blocks = T // block_size


def run_fwer_blocks(seed, use_ne=True):
    np.random.seed(seed)
    pi1, pi2 = (pi1_ne, pi2_ne) if use_ne else pi_alts[eta]
    M1_all = np.ones((R, 2, num_blocks + 1))
    M2_all = np.ones((R, 2, num_blocks + 1))
    for r in range(R):
        M1 = np.ones((2, num_blocks + 1))
        M2 = np.ones((2, num_blocks + 1))
        for b in range(1, num_blocks + 1):
            # Pre-generate all actions for this block
            block_a1 = np.random.choice([0, 1], size=block_size, p=pi1)
            block_a2 = np.random.choice([0, 1], size=block_size, p=pi2)
            for a1_prime in [0, 1]:
                block_e = 1.0
                for s in range(block_size):
                    X1 = U1[block_a1[s], block_a2[s]] - U1[a1_prime, block_a2[s]]
                    block_e *= (1 - lambda_val * X1)
                M1[a1_prime, b] = M1[a1_prime, b - 1] * block_e
            for a2_prime in [0, 1]:
                block_e = 1.0
                for s in range(block_size):
                    X2 = U2[block_a1[s], block_a2[s]] - U2[block_a1[s], a2_prime]
                    block_e *= (1 - lambda_val * X2)
                M2[a2_prime, b] = M2[a2_prime, b - 1] * block_e
        M1_all[r] = M1
        M2_all[r] = M2
    max_per_run = np.maximum(np.max(M1_all, axis=(1, 2)), np.max(M2_all, axis=(1, 2)))
    rejections = np.sum(max_per_run >= threshold)
    return rejections / R, int(rejections)


def run_detection_blocks(eta, seed):
    np.random.seed(seed)
    pi1_alt, pi2_alt = pi_alts[eta]
    tau_ubs = np.full(R, T)
    for r in range(R):
        M1 = np.ones((2, num_blocks + 1))
        M2 = np.ones((2, num_blocks + 1))
        for b in range(1, num_blocks + 1):
            if tau_ubs[r] < T:
                break
            block_a1 = np.random.choice([0, 1], size=block_size, p=pi1_alt)
            block_a2 = np.random.choice([0, 1], size=block_size, p=pi2_alt)
            for a1_prime in [0, 1]:
                block_e = 1.0
                for s in range(block_size):
                    X1 = U1[block_a1[s], block_a2[s]] - U1[a1_prime, block_a2[s]]
                    block_e *= (1 - lambda_val * X1)
                M1[a1_prime, b] = M1[a1_prime, b - 1] * block_e
            for a2_prime in [0, 1]:
                block_e = 1.0
                for s in range(block_size):
                    X2 = U2[block_a1[s], block_a2[s]] - U2[block_a1[s], a2_prime]
                    block_e *= (1 - lambda_val * X2)
                M2[a2_prime, b] = M2[a2_prime, b - 1] * block_e
            if np.any(M1[:, b] >= threshold) or np.any(M2[:, b] >= threshold):
                tau_ubs[r] = b * block_size
    avg_tau = np.mean(tau_ubs)
    q1 = np.percentile(tau_ubs, 25)
    q3 = np.percentile(tau_ubs, 75)
    med = np.median(tau_ubs)
    detected = int(np.sum(tau_ubs < T))
    return avg_tau, q1, q3, med, detected


print("=" * 60)
print("ALGO-07: Block-Aggregated E-Values")
print("lambda={:.2f} block={} alpha={:.2f} thr={:.1f}".format(lambda_val, block_size, alpha, threshold))
print("=" * 60)

if mode in ('fwer', 'both'):
    print()
    print("--- FWER (H0) ---")
    start = time.time()
    fwer, rej = run_fwer_blocks(seed, use_ne=True)
    elapsed = time.time() - start
    print("Empirical FWER: {:.3f}".format(fwer))
    print("Rejections: {} / {}".format(rej, R))
    print("FWER <= alpha ({:.2f}): {}".format(alpha, fwer <= alpha))
    print("Time: {:.1f}s".format(elapsed))

if mode in ('detection', 'both'):
    print()
    print("--- Detection Time (H1, eta={:.2f}) ---".format(eta))
    start = time.time()
    avg_tau, q1, q3, med, detected = run_detection_blocks(eta, seed + 1000)
    elapsed = time.time() - start
    print("Avg Detection Time: {:.1f}".format(avg_tau))
    print("Median: {:.1f}".format(med))
    print("Q1: {:.1f}".format(q1))
    print("Q3: {:.1f}".format(q3))
    print("Detected: {} / {}".format(detected, R))
    print("Detection Rate: {:.3f}".format(detected / R))
    print("Time: {:.1f}s".format(elapsed))
