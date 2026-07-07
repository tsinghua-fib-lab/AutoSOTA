import numpy as np; import numpy.linalg as LA
from sklearn.decomposition import TruncatedSVD

rng = np.random.default_rng(seed=233)

def estimate_theta_square(lambda_hat, c):
    nabla = np.square(c + 1 - lambda_hat) - 4*c
    nabla_sqrt = np.sqrt(np.abs(nabla))
    linear_w = (c - lambda_hat + 1)
    return (-linear_w + nabla_sqrt)/2

def ms_pca(X_tilde, max_k_r=10, C=None):
    d, n = X_tilde.shape
    c = d / n
    if C is None:
        C = 1/c
    max_k_r = min(d, max_k_r)
    svd_tilde = TruncatedSVD(n_components=max_k_r, n_iter=7, random_state=42)
    svd_tilde.fit(X_tilde.T/np.sqrt(n))
    S_tilde = svd_tilde.singular_values_
    U_tilde = svd_tilde.components_.T
    theta_square_prime = estimate_theta_square(S_tilde[0], c)
    noise_proportion_prime = 1
    gamma_prime = rng.binomial(1, noise_proportion_prime, n)
    noise_norm_prime = 2 * (theta_square_prime / noise_proportion_prime)**0.5
    m_prime, _ = LA.qr(rng.normal(size=(d, 1)))
    m_prime = noise_norm_prime * m_prime
    A_prime = m_prime * gamma_prime
    X_prime = X_tilde + A_prime
    svd_prime = TruncatedSVD(n_components=max_k_r, n_iter=7, random_state=42)
    svd_prime.fit(X_prime.T/np.sqrt(n))
    S_prime = svd_prime.singular_values_
    radius = C * n**(-1/2)
    stable_indices = []
    for i in range(max_k_r):
        for j in range(max_k_r):
            if abs(S_tilde[i] - S_prime[j]) < radius:
                stable_indices.append(i)
                break
    stable_eigenvalues = S_tilde[stable_indices]**2
    components = U_tilde[:, stable_indices]
    return stable_eigenvalues, components

k=1; magnitude_a=2; n=1000; c=0.9; d=int(n*c)
spike_base=np.sqrt(c); spike=[2*spike_base]
noise_proportion=0.15
alns=[]
for trial in range(5):
    spiked_vector,_=LA.qr(rng.normal(size=(d,k)))
    sigma=np.identity(d)
    for i in range(k):
        sigma+=spike[i]*np.outer(spiked_vector[:,i],spiked_vector[:,i])
    X=rng.multivariate_normal(np.zeros(d),sigma,n).T
    U,S,Vh=LA.svd(X/np.sqrt(n),full_matrices=False)
    theta_bar_square=np.sqrt(c)
    noise_norm_base=np.sqrt(theta_bar_square/noise_proportion)
    noise_norm=magnitude_a*noise_norm_base
    m1,_=LA.qr(rng.normal(size=(d,1))); m1=noise_norm*m1
    gamma=rng.binomial(1,noise_proportion,n)
    X_tilde=m1*gamma+X
    sev,comps=ms_pca(X_tilde,C=1/c)
    aln=abs(U[:,0]@comps[:,0])
    alns.append(aln)
    print("Trial", trial, "alignment=", round(aln*100,2), "%")
print("Mean alignment:", round(np.mean(alns)*100, 2), "%")
