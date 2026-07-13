import numpy as np


"""
    generate a matrix A m x n which satisfy:
    1. sum of each column is 1
    2. columns are mutually exclusive (rows where column i is non-zero must be zero in all other columns)

    parameters:
    m (int): rows of matrix 
    n (int): columns of matrix
    m must be larger than n

    return:
    numpy.ndarray: A
"""

def generate_markov_kernel(n, k):
    
    # non-zero counts of each column >= 1
    max_non_zeros = n - k  
    
    # generate a vector(len=n), denotes the number of extra non-zero elements in each column   
    extra_non_zeros = np.random.multinomial(max_non_zeros, np.ones(k)/k)
    
    # Ni = 1 + extra_non_zeros
    non_zero_counts = 1 + extra_non_zeros
    total_non_zeros = np.sum(non_zero_counts)

    # 2. distribute the position of each non-zero elements (no overlap in each column)
    all_row_indices = np.random.choice(n, size=total_non_zeros, replace=False)
    column_row_indices_lis = []

    # 3. generate matrix A
    A = np.zeros((n, k), dtype=float)
    current_idx = 0

    for i in range(k):
        Ni = non_zero_counts[i]
        
        # Extract the Ni mutually exclusive row indices corresponding to the non-zero elements in column i.
        column_row_indices = all_row_indices[current_idx : current_idx + Ni]
        column_row_indices_lis.append(column_row_indices.tolist())

        random_values = np.random.rand(Ni) 
        probabilities = random_values / np.sum(random_values)
        
        A[column_row_indices, i] = probabilities
        
        current_idx += Ni

    return A, column_row_indices_lis

def dimension_rising(Q, n):
    k = Q.shape[0]
    A, column_row_indices = generate_markov_kernel(n,k)
    column_row_indices_flattened = [idx for sublist in column_row_indices for idx in sublist]
    P = A@Q
    P_gd = P[np.array(column_row_indices_flattened), :]
    A_gd = A[np.array(column_row_indices_flattened), :]

    return P_gd, A_gd