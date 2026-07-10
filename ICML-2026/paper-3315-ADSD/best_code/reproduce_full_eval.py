"""
Full evaluation script for paper 3315 optimization.
Measures both Empirical FWER (H0) and Avg Detection Time (H1)
across the paper's parameter grid.
"""
import numpy as np
import time

U1 = np.array([[0.9, 0.2], [0.3, 0.7]])
U2 = np.array([[0.5, 0.3], [0.2, 0.7]])
pi1_ne = np.array([5/7, 2/7])
pi2_ne = np.array([5/11, 6/11])
pi_ne = [pi1_ne, pi2_ne]

pi_alts = {
    0.05: (np.array([0.9, 0.1]), np.array([10/11, 1/11])),
    0.10: (np.array([0.8, 0.2]), np.array([10/11, 1/11])),
    0.15: (np.array([0.7, 0.3]), np.array([10/11, 1/11])),
}

A_sizes = [2, 2]
m = sum(A_sizes)
T = 4000
R = 300
alphas = [0.2, 0.1, 0.05]
lambdas = [0.05, 0.1, 0.15, 0.4]
target_etas = [0.05, 0.10, 0.15]
SEED_BASE = 42


def run_fwer_simulation(lambda_val, alpha, seed, verbose=False):
    np.random.seed(seed)
    threshold = m / alpha
    M1_all = np.ones((R, 2, T + 1))
    M2_all = np.ones((R, 2, T + 1))
    for r in range(R):
        M1 = np.ones((2, T + 1))
        M2 = np.ones((2, T + 1))
        for t in range(1, T + 1):
            a1 = np.random.choice([0, 1], p=pi_ne[0])
            a2 = np.random.choice([0, 1], p=pi_ne[1])
            for a1_prime in [0, 1]:
                X1 = U1[a1, a2] - U1[a1_prime, a2]
                M1[a1_prime, t] = M1[a1_prime, t - 1] * (1 - lambda_val * X1)
            for a2_prime in [0, 1]:
                X2 = U2[a1, a2] - U2[a1, a2_prime]
                M2[a2_prime, t] = M2[a2_prime, t - 1] * (1 - lambda_val * X2)
        M1_all[r] = M1
        M2_all[r] = M2
    max_per_run = np.maximum(np.max(M1_all, axis=(1, 2)), np.max(M2_all, axis=(1, 2)))
    rejections = np.sum(max_per_run >= threshold)
    return rejections / R


def run_detection_simulation(lambda_val, alpha, eta, seed):
    np.random.seed(seed)
    threshold = m / alpha
    pi1_alt, pi2_alt = pi_alts[eta]
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
    avg_tau = np.mean(tau_ubs)
    q1_tau = np.percentile(tau_ubs, 25)
    q3_tau = np.percentile(tau_ubs, 75)
    detected = np.sum(tau_ubs < T)
    return avg_tau, q1_tau, q3_tau, detected / R


def full_evaluation(verbose=True):
    total_start = time.time()

    if verbose:
        print("=" * 70)
        print("FWER CONTROL STUDY (H0)")
        print("=" * 70)
        header = "{:>8} {:>8} {:>10} {:>10} {:>12}".format("Lambda", "Alpha", "Threshold", "FWER", "Rejections")
        print(header)
        print("-" * 70)

    fwer_results = {}
    for lambda_val in lambdas:
        for alpha in alphas:
            seed = SEED_BASE + int(lambda_val * 1000) + int(alpha * 1000)
            fwer = run_fwer_simulation(lambda_val, alpha, seed)
            fwer_results[(lambda_val, alpha)] = fwer
            threshold = m / alpha
            rejections = int(fwer * R)
            print("{:8.2f} {:8.2f} {:10.1f} {:10.3f} {:>8} / {}".format(
                lambda_val, alpha, threshold, fwer, rejections, R))

    if verbose:
        print()
        print("=" * 70)
        print("DETECTION TIME STUDY (H1)")
        print("=" * 70)
        header = "{:>8} {:>8} {:>8} {:>10} {:>8} {:>8} {:>10}".format(
            "Eta", "Alpha", "Lambda", "Avg Tau", "Q1", "Q3", "Det Rate")
        print(header)
        print("-" * 70)

    detection_results = {}
    for eta in target_etas:
        for alpha in alphas:
            for lambda_val in lambdas:
                seed = SEED_BASE + int(lambda_val * 1000) + int(alpha * 1000) + int(eta * 10000)
                avg_tau, q1, q3, det_rate = run_detection_simulation(lambda_val, alpha, eta, seed)
                detection_results[(eta, alpha, lambda_val)] = (avg_tau, q1, q3, det_rate)
                print("{:8.2f} {:8.2f} {:8.2f} {:10.1f} {:8.1f} {:8.1f} {:10.3f}".format(
                    eta, alpha, lambda_val, avg_tau, q1, q3, det_rate))

    elapsed = time.time() - total_start

    worst_fwer = max(fwer_results.values())
    worst_fwer_pair = max(fwer_results, key=fwer_results.get)
    primary_setting = (0.05, 0.2, 0.05)
    primary_avg_tau = detection_results[primary_setting][0]
    all_taus = [v[0] for v in detection_results.values()]
    overall_avg_tau = np.mean(all_taus)
    fwer_valid = all(fwer_results[(l, a)] <= a for l in lambdas for a in alphas)

    if verbose:
        print()
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print("Worst-case Empirical FWER: {:.3f} at lam={:.2f}, alp={:.2f}".format(
            worst_fwer, worst_fwer_pair[0], worst_fwer_pair[1]))
        print("Primary detection time (eta=0.05, alp=0.2, lam=0.05): {:.1f} rounds".format(primary_avg_tau))
        print("Overall avg detection time: {:.1f} rounds".format(overall_avg_tau))
        print("FWER valid (all <= respective alpha): {}".format(fwer_valid))
        print("Elapsed: {:.1f}s".format(elapsed))

    print("Empirical FWER: {:.3f}".format(worst_fwer))
    print("Avg Detection Time: {:.1f}".format(overall_avg_tau))
    print("Primary Detection Time (eta=0.05,alpha=0.2,lambda=0.05): {:.1f}".format(primary_avg_tau))

    return {
        'fwer_results': fwer_results,
        'detection_results': detection_results,
        'worst_fwer': worst_fwer,
        'overall_avg_tau': overall_avg_tau,
        'primary_avg_tau': primary_avg_tau,
        'fwer_valid': fwer_valid,
        'elapsed': elapsed
    }


if __name__ == "__main__":
    full_evaluation()
