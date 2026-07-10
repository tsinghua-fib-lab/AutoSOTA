"""
e-BH Procedure: Anytime-valid FDR control with stopped e-BH (Wang et al. 2025)
Evaluates both FWER detection time and FDR detection time.
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

m = 4
T = 4000
R = 300

# Parse CLI
lambda_val = float(sys.argv[1]) if len(sys.argv) > 1 else 1.50
alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
eta = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42

threshold_fwer = m / alpha  # FWER threshold (standard)
gamma = 1.0 / m  # uniform weights for e-BH


def ebh_rejection_set(e_values):
    """Compute e-BH rejection set from e-values.
    Returns (k_t, rejected_indices) where k_t is number of rejections."""
    sorted_idx = np.argsort(e_values)[::-1]
    k_t = 0
    for k in range(1, m + 1):
        idx = sorted_idx[k - 1]
        threshold_k = m / (k * alpha * gamma)
        if e_values[idx] >= threshold_k:
            k_t = k
        else:
            break
    if k_t > 0:
        return k_t, sorted_idx[:k_t]
    return 0, np.array([], dtype=int)


print("=" * 60)
print("e-BH FDR Control")
print("lambda={:.2f} alpha={:.2f} FWER_thr={:.1f}".format(lambda_val, alpha, threshold_fwer))
print("=" * 60)

# --- FWER (H0) ---
print()
print("--- FWER under H0 ---")
np.random.seed(seed)
fwer_rejections = 0
fdr_rejections_per_run = np.zeros(R)
fwer_taus = np.full(R, T)

start = time.time()
for r in range(R):
    M1 = np.ones((2, T + 1))
    M2 = np.ones((2, T + 1))
    for t in range(1, T + 1):
        if fwer_taus[r] < T:
            break
        a1 = np.random.choice([0, 1], p=pi1_ne)
        a2 = np.random.choice([0, 1], p=pi2_ne)
        for ap in [0, 1]:
            X1 = U1[a1, a2] - U1[ap, a2]
            M1[ap, t] = M1[ap, t - 1] * (1 - lambda_val * X1)
        for ap in [0, 1]:
            X2 = U2[a1, a2] - U2[a1, ap]
            M2[ap, t] = M2[ap, t - 1] * (1 - lambda_val * X2)

        # FWER check
        e_vals = np.array([M1[0, t], M1[1, t], M2[0, t], M2[1, t]])
        if np.any(e_vals >= threshold_fwer):
            fwer_rejections += 1
            fwer_taus[r] = t
            # Count FDR rejections at stopping time
            k_t, _ = ebh_rejection_set(e_vals)
            fdr_rejections_per_run[r] = k_t

fwer_elapsed = time.time() - start

# FDR computation under H0 (false discovery proportion)
# Under H0, all rejections are false discoveries
total_fdr_rejections = np.sum(fdr_rejections_per_run)
total_fwer_rejections = fwer_rejections
total_runs_with_rejections = np.sum(fdr_rejections_per_run > 0)
# FDR = E[FDP] where FDP = false discoveries / max(1, total discoveries)
fdp_values = np.where(fdr_rejections_per_run > 0,
                      1.0,  # under H0, all discoveries are false
                      0.0)
empirical_fdr = np.mean(fdp_values)

print("Empirical FWER: {:.3f} ({}/{})".format(fwer_rejections/R, fwer_rejections, R))
print("Empirical FDR: {:.3f}".format(empirical_fdr))
print("Runs with FDR rejections: {}".format(int(total_runs_with_rejections)))
print("Avg FDR rejections per run: {:.2f}".format(np.mean(fdr_rejections_per_run)))
print("FWER valid (<= {:.2f}): {}".format(alpha, fwer_rejections/R <= alpha))
print("Time: {:.1f}s".format(fwer_elapsed))

# --- Detection Time (H1) ---
print()
print("--- Detection under H1 (eta={:.2f}) ---".format(eta))
pi1_alt, pi2_alt = pi_alts[eta]
np.random.seed(seed + 1000)

fwer_h1_taus = np.full(R, T)
fdr_h1_taus = np.full(R, T)

start = time.time()
for r in range(R):
    M1 = np.ones((2, T + 1))
    M2 = np.ones((2, T + 1))
    for t in range(1, T + 1):
        fwer_done = fwer_h1_taus[r] < T
        fdr_done = fdr_h1_taus[r] < T
        if fwer_done and fdr_done:
            break
        a1 = np.random.choice([0, 1], p=pi1_alt)
        a2 = np.random.choice([0, 1], p=pi2_alt)
        for ap in [0, 1]:
            X1 = U1[a1, a2] - U1[ap, a2]
            M1[ap, t] = M1[ap, t - 1] * (1 - lambda_val * X1)
        for ap in [0, 1]:
            X2 = U2[a1, a2] - U2[a1, ap]
            M2[ap, t] = M2[ap, t - 1] * (1 - lambda_val * X2)

        e_vals = np.array([M1[0, t], M1[1, t], M2[0, t], M2[1, t]])

        if not fwer_done and np.any(e_vals >= threshold_fwer):
            fwer_h1_taus[r] = t

        if not fdr_done:
            k_t, _ = ebh_rejection_set(e_vals)
            if k_t > 0:
                fdr_h1_taus[r] = t

h1_elapsed = time.time() - start

fwer_avg = np.mean(fwer_h1_taus)
fwer_det = np.sum(fwer_h1_taus < T)
fdr_avg = np.mean(fdr_h1_taus)
fdr_det = np.sum(fdr_h1_taus < T)

print("FWER Detection: avg={:.1f} det={}/{}".format(fwer_avg, fwer_det, R))
print("FDR Detection:  avg={:.1f} det={}/{}".format(fdr_avg, fdr_det, R))
print("Time: {:.1f}s".format(h1_elapsed))

print()
print("Empirical FWER: {:.3f}".format(fwer_rejections / R))
print("Avg Detection Time: {:.1f}".format(fwer_avg))
print("FDR Detection Time: {:.1f}".format(fdr_avg))
