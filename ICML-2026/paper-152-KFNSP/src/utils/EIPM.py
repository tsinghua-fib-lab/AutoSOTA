import torch

def Kernel_matrix(Xi, Xj, a):
    # Xi, Xj: a pair of representation vectors of size [n, m]
    # a: bandwidth for kernel
    m = - torch.cdist(Xi, Xj, p=2)**2
    m = m / (2 * a**2)
    m = torch.exp(m)
    return m

# Function that outputs EIPM value between z and s using bandwidths σ and γ.
def compute_EIPM(z, s, sigma, gamma):
    # z: representation vectors of size [n, m]
    # s: sensitive attributes of size [n, 1]
    # σ: bandwidth in MMD
    # γ: bandwidth of kernel estimator
    #
    # Kernel method for s
    n = s.shape[0]

    A = Kernel_matrix(s.unsqueeze(dim=1), s.unsqueeze(dim=1), gamma) - torch.eye(n) # was B?
    A = A / (A.sum(dim=0)+1e-12)
    A = A - 1 / (n - 1)
    A = A.fill_diagonal_(0)
    # MMD
    K = Kernel_matrix(z, z, sigma)
    # Compute EIPM
    EIPM = torch.einsum('ij,ik->ijk', A.T, A.T)
    EIPM = torch.sum(EIPM * K.unsqueeze(dim=0), dim=(1, 2)).sum()
    EIPM = EIPM / n
    #
    return EIPM

"""
x = torch.rand(100,1)
print(compute_EIPM(x,x,sigma=1,gamma=1))
"""