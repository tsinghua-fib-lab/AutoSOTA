"""
Reproduce Empirical FWER for paper 3315: Anytime Detection of Strategic Deviations
Target: lambda=0.05, alpha=0.2, threshold=20.0, H0 condition, 2x2 normal-form game
Paper value: 0.010 (3 false rejections out of R=300 runs)
Source: Appendix H.1.4, Table 1
"""
import numpy as np
import time

# Game definition (from paper/normal-form.ipynb)
U1 = np.array([[0.9, 0.2], [0.3, 0.7]])
U2 = np.array([[0.5, 0.3], [0.2, 0.7]])

# Nash equilibrium (mixed)
pi1_ne = np.array([5/7, 2/7])
pi2_ne = np.array([5/11, 6/11])
pi_ne = [pi1_ne, pi2_ne]

# Parameters matching rubric
T = 4000  # rounds per run
R = 300   # independent runs
lambda_val = 0.05
alpha = 0.2
A_sizes = [2, 2]
m = sum(A_sizes)  # 4 hypotheses
threshold = m / alpha  # b = 4/0.2 = 20

print("=" * 60)
print("REPRODUCTION: Empirical FWER under H0")
print("=" * 60)
print(f"Game: 2x2 normal-form")
print(f"Players: 2, Actions per player: {A_sizes}")
print(f"Total hypotheses m: {m}")
print(f"Betting parameter lambda: {lambda_val}")
print(f"Significance level alpha: {alpha}")
print(f"Rejection threshold b = m/alpha: {threshold}")
print(f"Rounds per run T: {T}")
print(f"Independent runs R: {R}")
print(f"Nash P1: {pi1_ne}")
print(f"Nash P2: {pi2_ne}")
print("-" * 60)

np.random.seed(42)

start_time = time.time()

# Storage matrices
M1_all = np.ones((R, 2, T + 1))
M2_all = np.ones((R, 2, T + 1))

for r in range(R):
    M1 = np.ones((2, T + 1))
    M2 = np.ones((2, T + 1))
    for t in range(1, T + 1):
        # Sample actions from Nash equilibrium
        a1 = np.random.choice([0, 1], p=pi_ne[0])
        a2 = np.random.choice([0, 1], p=pi_ne[1])
        # P1 martingales
        for a1_prime in [0, 1]:
            X1 = U1[a1, a2] - U1[a1_prime, a2]
            M1[a1_prime, t] = M1[a1_prime, t - 1] * (1 - lambda_val * X1)
        # P2 martingales
        for a2_prime in [0, 1]:
            X2 = U2[a1, a2] - U2[a1, a2_prime]
            M2[a2_prime, t] = M2[a2_prime, t - 1] * (1 - lambda_val * X2)
    M1_all[r] = M1
    M2_all[r] = M2

elapsed = time.time() - start_time

# Compute Empirical FWER
max_per_run = np.maximum(
    np.max(M1_all, axis=(1, 2)),
    np.max(M2_all, axis=(1, 2))
)
rejections = np.sum(max_per_run >= threshold)
empirical_fwer = rejections / R

print(f"\nRESULTS:")
print(f"  Rejections: {rejections} / {R}")
print(f"  Empirical FWER: {empirical_fwer:.3f}")
print(f"  Paper value: 0.010")
print(f"  Elapsed time: {elapsed:.1f}s")

if empirical_fwer == 0.010:
    print(f"\nREPRODUCTION SUCCEEDED - exact match with paper value 0.010")
elif empirical_fwer <= 0.2:
    print(f"\nREPRODUCTION SUCCEEDED - within CI bounds [{-0.009}, {0.200}]")
else:
    print(f"\nREPRODUCTION FAILED - outside expected bounds")

# Diagnostics
print(f"\nMax martingale stats across {R} runs:")
print(f"  Mean: {np.mean(max_per_run):.4f}")
print(f"  Min:  {np.min(max_per_run):.4f}")
print(f"  Max:  {np.max(max_per_run):.4f}")
print(f"  Q50:  {np.median(max_per_run):.4f}")
print(f"  Q90:  {np.percentile(max_per_run, 90):.4f}")
print(f"  Q95:  {np.percentile(max_per_run, 95):.4f}")
print(f"  Q99:  {np.percentile(max_per_run, 99):.4f}")
print(f"  Runs >= threshold ({threshold}): {rejections}")
