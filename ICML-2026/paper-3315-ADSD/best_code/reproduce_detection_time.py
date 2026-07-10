"""
Fast detection time evaluation for paper 3315.
Measures avg stopping time under H1 at specified parameter settings.
"""
import numpy as np
import time
import sys

U1 = np.array([[0.9, 0.2], [0.3, 0.7]])
U2 = np.array([[0.5, 0.3], [0.2, 0.7]])

pi_alts = {
    0.05: (np.array([0.9, 0.1]), np.array([10/11, 1/11])),
    0.10: (np.array([0.8, 0.2]), np.array([10/11, 1/11])),
    0.15: (np.array([0.7, 0.3]), np.array([10/11, 1/11])),
}

A_sizes = [2, 2]
m = sum(A_sizes)
T = 4000
R = 300

# Parse CLI: lambda alpha eta [seed]
lambda_val = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
eta = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42

threshold = m / alpha
pi1_alt, pi2_alt = pi_alts[eta]

np.random.seed(seed)

start = time.time()
tau_ubs = np.full(R, T)

for r in range(R):
    M1 = np.ones((2, T + 1))
    M2 = np.ones((2, T + 1))
    for t in range(1, T + 1):
        if tau_ubs[r] < T:
            break
        a1 = np.random.choice([0, 1], p=pi1_alt)
        a2 = np.random.choice([0, 1], p=pi2_alt)
        for a1_prime in [0, 1]:
            if M1[a1_prime, t - 1] < threshold:
                X1 = U1[a1, a2] - U1[a1_prime, a2]
                M1[a1_prime, t] = M1[a1_prime, t - 1] * (1 - lambda_val * X1)
            else:
                M1[a1_prime, t] = M1[a1_prime, t - 1]
        for a2_prime in [0, 1]:
            if M2[a2_prime, t - 1] < threshold:
                X2 = U2[a1, a2] - U2[a1, a2_prime]
                M2[a2_prime, t] = M2[a2_prime, t - 1] * (1 - lambda_val * X2)
            else:
                M2[a2_prime, t] = M2[a2_prime, t - 1]
        if np.any(M1[:, t] >= threshold) or np.any(M2[:, t] >= threshold):
            tau_ubs[r] = t

elapsed = time.time() - start

avg_tau = np.mean(tau_ubs)
q1 = np.percentile(tau_ubs, 25)
q3 = np.percentile(tau_ubs, 75)
med = np.median(tau_ubs)
detected = int(np.sum(tau_ubs < T))

print("=" * 60)
print("DETECTION TIME (H1)")
print("lambda={:.2f} alpha={:.2f} eta={:.2f} threshold={:.1f}".format(
    lambda_val, alpha, eta, threshold))
print("=" * 60)
print("Avg Detection Time: {:.1f}".format(avg_tau))
print("Median: {:.1f}".format(med))
print("Q1: {:.1f}".format(q1))
print("Q3: {:.1f}".format(q3))
print("Detected: {} / {}".format(detected, R))
print("Detection Rate: {:.3f}".format(detected / R))
print("Elapsed: {:.1f}s".format(elapsed))
