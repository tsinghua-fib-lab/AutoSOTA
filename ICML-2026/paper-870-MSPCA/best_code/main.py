from itertools import product
import numpy as np
import numpy.linalg as LA
import pandas as pd
from tqdm import tqdm
from rpca import RobustPCA

rng = np.random.default_rng(seed=233)
fname = "result/gaussian/pc1compare2_ns15_t25"

def estimate_theta_square(lambda_hat, c):
    """
    Reverse function of l |-> 1 + l + c * (1 + l)/l.
    """
    nabla = np.square(c + 1 - lambda_hat) - 4*c
    nabla_sqrt = np.sqrt(np.abs(nabla))
    linear_w = (c - lambda_hat + 1)
    solution1 = (-linear_w + nabla_sqrt)/2
    return solution1


def _ms_pca(X_tilde, max_k_r = 10, C = None):
    d, n = X_tilde.shape
    c = d / n
    if C is None:
        C = 1/c
    max_k_r = min(d, max_k_r)

    U_tilde, S_tilde, Vh_tilde = LA.svd(X_tilde/np.sqrt(n))
    theta_square_prime = estimate_theta_square(S_tilde[0], c)

    # Generate new noisy data
    noise_proportion_prime = 1
    gamma_prime = np.random.binomial(1, noise_proportion_prime, n)
    noise_norm_prime = 2 * (theta_square_prime / noise_proportion_prime)**0.5

    m_prime, _ = LA.qr(rng.normal(size=(d, 1)))
    m_prime = noise_norm_prime * m_prime
    A_prime = m_prime * gamma_prime
    X_prime = X_tilde + A_prime 
    U_prime, S_prime, Vh_prime = LA.svd(X_prime/np.sqrt(n))

    radius = C * n**(-1/2)

    # Step 5: Invariance check
    stable_indices = []

    for i in range(max_k_r):
        for j in range(max_k_r):
            if abs(S_tilde[i] - S_prime[j]) < radius:
                stable_indices.append(i)
                break
            if S_prime[j] < S_tilde[i] - 10* radius:
                break

    stable_eigenvalues = S_tilde[stable_indices]**2
    components = U_tilde[:, stable_indices]
    return stable_eigenvalues, components

    
def ms_pca(X_tilde, max_k_r = 10, C = None):
    d, n = X_tilde.shape
    c = d / n
    if C is None:
        C = 1/c
    max_k_r = min(d, max_k_r)

    svd_tilde = TruncatedSVD(n_components=max_k_r, n_iter=7, random_state=42)
    svd_tilde.fit(X_tilde.T/np.sqrt(n))
    S_tilde = svd_tilde.singular_values_
    U_tilde = svd_tilde.components_.T


    # U_tilde, S_tilde, Vh_tilde = LA.svd(X_tilde/np.sqrt(n))
    theta_square_prime = estimate_theta_square(S_tilde[0], c)

    # Generate new noisy data
    noise_proportion_prime = 1
    gamma_prime = np.random.binomial(1, noise_proportion_prime, n)
    noise_norm_prime = 2 * (theta_square_prime / noise_proportion_prime)**0.5

    m_prime, _ = LA.qr(rng.normal(size=(d, 1)))
    m_prime = noise_norm_prime * m_prime
    A_prime = m_prime * gamma_prime
    X_prime = X_tilde + A_prime 
    svd_prime = TruncatedSVD(n_components=max_k_r, n_iter=7, random_state=42)
    svd_prime.fit(X_prime.T/np.sqrt(n))
    S_prime = svd_prime.singular_values_

    # U_prime, S_prime, Vh_prime = LA.svd(X_prime/np.sqrt(n))

    radius = C * n**(-1/2)

    # Step 5: Invariance check
    stable_indices = []

    for i in range(max_k_r):
        for j in range(max_k_r):
            if abs(S_tilde[i] - S_prime[j]) < radius:
                stable_indices.append(i)
                break
            # if S_prime[j] < S_tilde[i] - 10* radius:
            #     break

    stable_eigenvalues = S_tilde[stable_indices]**2
    components = U_tilde[:, stable_indices]
    return stable_eigenvalues, components

cs = [0.1, 1/2, 1, 2]
noise_proportions = [0.05, 0.1, 0.15, 0.25]
c_np_paris = product(cs, noise_proportions)

ns = np.logspace(2, 4, num=15, dtype=int)
n_trial = 25

k = 1
magnitude_a = 2
results = []

for c, noise_proportion in tqdm(c_np_paris):
    spike_base = np.sqrt(c)
    spike = [2 * spike_base]
    for trial in range(n_trial):
        for n in ns:
            d = int(n * c)
            spiked_vector, _ = LA.qr(rng.normal(size=(d, k)))
            sigma = np.identity(d) # d x d
            for i in range(k):
                sigma += spike[i] * np.outer(spiked_vector[:, i], spiked_vector[:, i])
            X = rng.multivariate_normal(np.zeros(d), sigma, n).T
            U, S, Vh = LA.svd(X/np.sqrt(n), full_matrices=False)

            theta_bar_square = (np.sqrt(c)) 
            # noise_proportion = 0.2
            noise_norm_base = np.sqrt(theta_bar_square / noise_proportion)
            noise_norm = magnitude_a * noise_norm_base
            theta_square = noise_proportion * np.square(noise_norm)

            m1, _ = LA.qr(rng.normal(size=(d, 1))) 
            m1 = noise_norm * m1
            gamma = np.random.binomial(1, noise_proportion, n)
            A = m1 * gamma
            X_tilde = A + X

            U_tilde, S_tilde, Vh_tilde = LA.svd(X_tilde/np.sqrt(n))

            stable_eigenvalues, components = ms_pca(X_tilde, C = 1/c)
            rpca = RobustPCA(n_components=1, verbose=False)
            rpca.fit(X_tilde)
            L = rpca.low_rank_
            ms_alignment = abs(U[:, 0] @ components[:, 0]) 
            rpca_alignment = abs(U[:, 0] @ L[:, 0] / LA.norm(L[:, 0]))
            pc_alignment = abs(U[:, 0] @ U_tilde[:, 0])


            # Store results
            result = {
                'c': c,
                'noise_proportion': noise_proportion,
                'trial': trial,
                'n': n,
                'd': d,
                'true_spike_magnitude': spike[0],
                'contamination_magnitude': noise_norm,
                'ms_alignment': ms_alignment,
                'rpca_alignment': rpca_alignment,
                'pc_alignment': pc_alignment
            }
            results.append(result)

df_results = pd.DataFrame(results)
df_results.to_csv(f'{fname}.csv', index=False)
