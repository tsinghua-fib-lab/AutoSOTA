import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import numpy as np


def helmert_matrix(n):
    H = np.zeros((n, n - 1))
    H[0, :] = -1
    for i in range(1, n):
        H[i, i - 1] = i
        H[i, i:] = -1
    return H

def normalize_columns(m):
    for j in range(m.shape[1]):
        m[:, j] = m[:, j] / np.sqrt(np.sum(m[:, j]**2))
    return m

def create_C_matrix_cs(num_cat, rho):
    return (1 - rho) * np.eye(num_cat) + rho * np.ones((num_cat, num_cat))


def decompose_cs(n_cat, rho):
    # evals = np.zeros(n_cat, dtype=np.float32)
    # evecs = np.zeros((n_cat, n_cat), dtype=np.float32)
    
    # # First eigenvalue is 1 + (n_cat - 1) * rho, others are 1 - rho
    # evals[0] = 1 + (n_cat - 1) * rho
    # evals[1:] = 1 - rho
    
    # # First eigenvector is all ones
    # evecs[:, 0] = np.ones(n_cat)
    
    # # Remaining eigenvectors are given by the Helmert matrix
    # H = helmert_matrix(n_cat)
    # evecs[:, 1:] = H
    
    # # Normalize eigenvectors
    # evecs = normalize_columns(evecs)

    K = create_C_matrix_cs(n_cat, rho)
    evals, evecs = np.linalg.eigh(K)
    return evals.astype(np.float32), evecs.astype(np.float32)