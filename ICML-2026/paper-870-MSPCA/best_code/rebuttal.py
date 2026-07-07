from itertools import product
import numpy as np
import numpy.linalg as LA
import pandas as pd
from tqdm import tqdm
from rpca import RobustPCA
from pyriemann.utils.covariance import covariance_mest
from scipy.stats.mstats import winsorize
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import time
from tqdm import tqdm



rng = np.random.default_rng(seed=233)
# fname = "result/gaussian/pc1_rebuttal"
fname = "result/rebuttal/pc1_rebuttal"


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

# Reference:
# This script approximates the optimal L1-principal components of real-valued data,
# as presented in the article:
# P. P. Markopoulos, S. Kundu, S. Chamadia, and D. A. Pados, 
# ``Efficient L1-norm Principal-Component Analysis via Bit Flipping" 
# in IEEE Transactions on Signal Processing, vol. 65, no. 16, pp. 4252-4264, 15 Aug.15, 2017.
#
# ---
# Function Description:
# Inputs: X => Fat data matrix,
#		  K => subspace dimensionality,
#		  num_init => number initializations,
#	      print_flag => print statistics option.
# Outputs: Q => L1-PCs, 
#		   B => Binary nuc-norm solution,
#		   vmax => L1-norm PCA value.
# 
# ---
# Dependencies: 
#	1) scipy (publicly available from: https://www.scipy.org/install.html)
#
# ---
# Note:
# Inquiries regarding the script provided below are cordially welcome.
# In case you spot a bug, please let me know.
# If you use some piece of code for your own work, please cite the
# corresponding article above.

def l1pca_sbfk(X, K, num_init, print_flag):
    # Parameters
    toler =10e-8;

    # Get the dimentions of the matrix.
    dataset_matrix_size = X.shape	
    D = dataset_matrix_size[0]	# Row dimension.
    N = dataset_matrix_size[1]	# Column dimension.

    # Initialize the matrix with the SVD.
    dummy, S_x, V_x = LA.svd(X , full_matrices = False)	# Hint: The singular values are in vector form.
    if D < N:
        V_x = V_x.transpose()
        
    X_t = np.matmul(np.diag(S_x),V_x.transpose())

    # Initialize the required matrices and vectors.
    Bprop = np.ones((N,K),dtype=float)
    nucnormmax = 0
    iterations = np.zeros((1,num_init),dtype=float)

    # For each initialization do.
    for ll in range(0, num_init):
        
        start_time = time.time()	# Start measuring execution time.

        v = np.random.randn(N,K)	# Random initialized vector.
        if ll<2:	# In the first initialization, initialize the B matrix to sign of the product of the first 
                    # right singular vector of the input matrix with an all-ones matrix.
            z = np.zeros((N,1),dtype=float)
            z = V_x[:,0]
            z_x = z.reshape(N,1)
            v = np.matmul(z_x,np.ones((1,K), dtype=float))
        B = np.sign(v)	# Get a binary vector containing the signs of the elements of v.

        # Calculate the nuclear norm of X*B.
        X_temp = np.matmul(X_t,B)
        dummy1, S, dummy2 = LA.svd(X_temp , full_matrices = False)
        nucnorm = np.sum(np.sum(np.diag(S)))
        nuckprev = nucnorm*np.ones((K,1), dtype=float)

        # While not converged bit flip.
        iter_ = 0
        while True:
            iter_ = iter_ + 1

            flag = False

            # Calculate all the possible binary vectors and all posible bit flips.
            for k in range(0, K):

                a = np.zeros((N,1), dtype=float)

                for n in range(0, N):
                    B_t = B
                    B_t[n,k] = -B[n,k]
                    dummy1, S, dummy2 = LA.svd(np.matmul(X_t,B), full_matrices=False)
                    a[n] = sum(sum(np.diag(S)))
                
                ma = np.max(a)	# Find which binary vector and bit flips maximize the quadratic.
                if ma > nucnorm:
                    nc = np.where(a == ma)
                    B_t[nc[0],k] = -B_t[nc[0],k]
                    nucnorm = ma

                # If the maximum quadratic is attained, stop iterating.
                if iter_ > 1 and nucnorm<nuckprev[k] + toler:
                    flag = True
                    break

                nuckprev[k] = nucnorm # Save the calculated nuclear norm of the current initialization.

            if flag == True:
                break

        # Find the maximum nuclear norm across all initializations.
        iterations[0,ll] = iter_
        if nucnorm > nucnormmax:
            nucnormmax = nucnorm
            Bprop = B

    # Calculate the final subspace.
    U, dummy, V = LA.svd(np.matmul(X,Bprop), full_matrices=False)
    Uprop = U[:,0:K]
    Vprop = V[:,0:K]
    Qprop = np.matmul(Uprop,Vprop.transpose())

    end_time = time.time()	# End of execution timestamp.
    timelapse = (end_time - start_time)	# Calculate the time elapsed.

    convergence_iter = np.mean(iterations, dtype=float) # Calculate the mean iterations per initialization.
    vmax = sum(sum(abs(np.matmul(Qprop.transpose(),X))))
    
    # If print true, print execution statistics.
    if print_flag:
        print("------------------------------------------")
        print("Avg. iterations/initialization: ", (convergence_iter))
        print("Time elapsed (sec): ", (timelapse))
        print("Metric value:", vmax)
        print("------------------------------------------")

    return Qprop, Bprop, vmax


k = 1
magnitude_a = 2

n = 1000
c = 0.9
spike_base = np.sqrt(c)
spike = [2 * spike_base]

noise_proportions = [0.05, 0.1, 0.15, 0.2]
table_trials = []
n_trials = 200

for _ in tqdm(range(n_trials)):
    table = []
    for noise_proportion in noise_proportions:

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


        stable_eigenvalues, components = ms_pca(X_tilde)
        ms_alignment = abs(U[:, 0] @ components[:, 0]) 

        rpca = RobustPCA(n_components=1, verbose=False)
        rpca.fit(X_tilde)
        L = rpca.low_rank_
        rpca_alignment = abs(U[:, 0] @ L[:, 0] / LA.norm(L[:, 0]))


        tyl_cov = covariance_mest(X_tilde, "tyl")
        U_tyl, S_tyl, Vt_tyl = LA.svd(tyl_cov)
        tyl_align = abs(U[:, 0] @ U_tyl[:, 0]) 

        hub_cov = covariance_mest(X_tilde, "hub")
        U_hub, S_hub, Vt_hub = LA.svd(hub_cov)
        hub_align = abs(U[:, 0] @ U_hub[:, 0]) 

        l1_pc, b, l1_pc_v = l1pca_sbfk(X_tilde, 1, 100, True)
        l1_align = abs(U[:, 0] @ l1_pc.reshape(d)) 

        X_win = winsorize(X_tilde, axis = 0)
        U_win, S_win, Vt_win = LA.svd(X_win/np.sqrt(n))
        win_align = abs(U[:, 0] @ U_win[:, 0]).data.item()

        pca = PCA(n_components=1)
        pca.fit(X_tilde.T)
        pca_align = abs(U[:, 0] @ pca.components_.reshape(d))

        table.append((ms_alignment, rpca_alignment, tyl_align, hub_align, l1_align, win_align, pca_align))
    
    table_trials.append(table)

stats_array = np.array(table_trials)
df_mean = pd.DataFrame(stats_array.mean(axis=0), columns=["MS-PCA", "RPCA-AAP", "Tyler", "Huber", "l1-PCA", "winsorized-PCA", "center-PCA"],
index=noise_proportions)
df_std = pd.DataFrame(stats_array.std(axis=0), columns=["MS-PCA", "RPCA-AAP", "Tyler", "Huber", "l1-PCA", "winsorized-PCA", "center-PCA"],
index=noise_proportions)

print(df_mean)
print(df_std)

fname = "result/rebuttal/pc1_rebuttal"
df_mean.to_csv(f"{fname}_n1000_d900_t{n_trials}_mean")
df_std.to_csv(f"{fname}_n1000_d900_t{n_trials}_std")

