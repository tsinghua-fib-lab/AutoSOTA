import os
import itertools
import time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import norm
from scipy.linalg import eigh
from scipy.linalg import lstsq
from findiff import FinDiff, Coef

def uniform_points_pixelwise(n, domain_length, boundary = False, dim=2):
        xi = []
        for i in range(dim):
            pixel_size = domain_length / n
            if boundary:
                start = 0
                end = domain_length
            else:
                start = pixel_size / 2
                end = domain_length - pixel_size / 2
            xi.append(np.linspace(start, end, num=n))
        x = np.array(list(itertools.product(*xi)))
        total_points = n ** dim
        if total_points != len(x):
            print(
                "Warning: {} points required, but {} points sampled.".format(total_points, len(x))
            )
        return x

def create_f_s(x, y, w = 0.125, r = 10.):
    condition1 = np.abs(x - 0.5 * w) <= 0.5 * w
    condition2 = np.abs(x - 1 + 0.5 * w) <= 0.5 * w
    condition3 = np.abs(y - 0.5 * w) <= 0.5 * w
    condition4 = np.abs(y - 1 + 0.5 * w) <= 0.5 * w
    result = np.zeros_like(x)
    result[np.logical_and(condition1, condition3)] = r
    result[np.logical_and(condition2, condition4)] = -r
    return result

def complete_covariance_matrix(grid, l):
    """Create the complete covariance matrix for a grid of shape (a, a) using vectorized operations."""
    # calculate differences in each dimension between points
    dx = grid[:, None, 0] - grid[None, :, 0]
    dy = grid[:, None, 1] - grid[None, :, 1]
    # compute squared Euclidean distances
    distances_squared = dx**2 + dy**2
    # apply the covariance formula
    covariance_matrix = np.exp(-np.sqrt(distances_squared) / l)
    return covariance_matrix

def compute_eigenpairs(cov_matrix, q):
    """
    Compute the first 'q' eigenvalues and eigenvectors of the covariance matrix.
    """
    eigenvalues, eigenvectors = eigh(cov_matrix)
    # sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return eigenvalues[:q], eigenvectors[:, :q]

def KLE_expansion(eigenvalues, eigenvectors, q, grid_points, seed=None):
    """
    Compute the KLE expansion for the log-permeability field.
    """
    # set the random seed if provided
    if seed is not None:
        np.random.seed(seed)
    # sample from standard normal
    z = norm.rvs(size=q)

    # compute the KLE sum for each grid point
    G_s = np.zeros(grid_points)
    for k in range(q):
        G_s += np.sqrt(eigenvalues[k]) * z[k] * eigenvectors[:, k]

    return G_s, z

def create_boundary_idcs(shape):    
    # xmin boundary idcs
    xmin_bd = np.zeros(shape, dtype=np.bool_)
    xmin_bd[0,:] = 1
    xmin_bd = xmin_bd.reshape(-1)
    # xmax boundary idcs
    xmax_bd = np.zeros(shape, dtype=np.bool_)
    xmax_bd[-1,:] = 1
    xmax_bd = xmax_bd.reshape(-1)
    # ymin boundary idcs
    ymin_bd = np.zeros(shape, dtype=np.bool_)
    ymin_bd[:,0] = 1
    ymin_bd = ymin_bd.reshape(-1)
    # ymax boundary idcs
    ymax_bd = np.zeros(shape, dtype=np.bool_)
    ymax_bd[:,-1] = 1
    ymax_bd = ymax_bd.reshape(-1)
    return xmin_bd, xmax_bd, ymin_bd, ymax_bd

def create_int_cond(use_trapezoid, shape, d0):
    if use_trapezoid:
        # identify corner nodes
        int_cond = np.zeros(shape)
        int_cond[0,0] = 1.
        int_cond[0,-1] = 1.
        int_cond[-1,0] = 1.
        int_cond[-1,-1] = 1.
        # identify boundary nodes
        int_cond[1:-1,0] = 2.
        int_cond[1:-1,-1] = 2.
        int_cond[0,1:-1] = 2.
        int_cond[-1,1:-1] = 2.
        # identify interior nodes
        int_cond[1:-1,1:-1] = 4.
        # assert that no node is 0
        assert np.all(int_cond != 0)
        int_cond *= d0**2 / 4.
    else:
        # simple mean
        pixels_per_dim= shape[0]
        int_cond = np.ones(shape).reshape(-1,1) / (pixels_per_dim**2)
    return int_cond

def generate_sample(args):

    pid = os.getpid()
    current_time = int(time.time() * 1000)  # current time in milliseconds
    unique_seed = pid * current_time % (2**32)  # XOR for uniqueness and modulo to fit the range

    i, eigenvalues, eigenvectors, q, pixels_per_dim, shape, acc, d0, d1, f_s, int_cond, xmin_bd, xmax_bd, ymin_bd, ymax_bd, reverse_dy = args
    
    log_permeability_field, z = KLE_expansion(eigenvalues, eigenvectors, q, pixels_per_dim**2, seed=unique_seed)
    # exponentiate to get the permeability field K
    K = np.exp(log_permeability_field.reshape(shape))

    K_d0 = FinDiff(0, d0, 1, acc=acc)(K)
    K_d1 = FinDiff(1, d1, 1, acc=acc)(K)

    darcy_fd = Coef(-K) * FinDiff(0, d0, 2, acc=acc) - Coef(K_d0) * FinDiff(0, d0, 1, acc=acc) - Coef(K) * FinDiff(1, d1, 2, acc=acc) - Coef(K_d1) * FinDiff(1, d1, 1, acc=acc)
    grad_p_d0 = FinDiff(0, d0, 1, acc=acc)
    grad_p_d1 = FinDiff(1, d1, 1, acc=acc)

    A = darcy_fd.matrix(shape).toarray()
    b = f_s.reshape(-1,1)

    # add boundary conditions
    grad_p_d0_np = grad_p_d0.matrix(shape).toarray()
    grad_p_d1_np = grad_p_d1.matrix(shape).toarray()

    if reverse_dy:
        A_bc = np.concatenate((A, -grad_p_d0_np[xmin_bd, :], grad_p_d0_np[xmax_bd, :], grad_p_d1_np[ymin_bd, :], -grad_p_d1_np[ymax_bd, :]), axis=0)
    else:
        A_bc = np.concatenate((A, -grad_p_d0_np[xmin_bd, :], grad_p_d0_np[xmax_bd, :], -grad_p_d1_np[ymin_bd, :], grad_p_d1_np[ymax_bd, :]), axis=0)
    b_bc = np.concatenate((b, np.zeros(len(A_bc) - len(A)).reshape(-1,1)), axis=0)
    assert len(A_bc) - len(A) == 4 * pixels_per_dim

    # add this to linear system
    A_bc_int = np.concatenate((A_bc, int_cond.reshape(1,-1)), axis=0)
    b_bc_int = np.concatenate((b_bc, np.array([0.]).reshape(1,1)), axis=0)

    # solve linear system
    p, residuals, _, _ = lstsq(A_bc_int, b_bc_int)

    residual_test = A_bc_int @ p.reshape(-1) - b_bc_int.reshape(-1)
    
    return K.flatten(), p.flatten(), np.abs(residual_test).mean(), unique_seed

def main():
    
    start_time = time.time()

    # num_processes = int(sys.argv[1])
    num_processes = 1
    n_samples = 10
    
    pixels_per_dim = 64
    pixels_at_boundary = True
    domain_length = 1.
    length_scale = 0.1
    q = 64  # number of KLE coefficients
    acc = 2
    reverse_dy = True # reverse y axis for consistency in plots (y goes from bottom to top)

    shape = (pixels_per_dim, pixels_per_dim)
    evaluation_points = uniform_points_pixelwise(pixels_per_dim, domain_length, pixels_at_boundary)

    if pixels_at_boundary:
        d0 = domain_length / (pixels_per_dim - 1)
        d1 = domain_length / (pixels_per_dim - 1)
    else:
        d0 = domain_length / pixels_per_dim
        d1 = domain_length / pixels_per_dim

    if reverse_dy:
        d1 *= -1.

    cov_matrix = complete_covariance_matrix(evaluation_points, length_scale)
    eigenvalues, eigenvectors = compute_eigenpairs(cov_matrix, q)    
    f_s = create_f_s(evaluation_points[:,0], evaluation_points[:,1])

    xmin_bd, xmax_bd, ymin_bd, ymax_bd = create_boundary_idcs(shape)

    # add integral constraint - trapezoidal rule if outer pixels are at boundary, otherwise simple mean
    if pixels_at_boundary:
        use_trapezoid = True
    else:
        use_trapezoid = False
    int_cond = create_int_cond(use_trapezoid, shape, d0)

    args = [(i, eigenvalues, eigenvectors, q, pixels_per_dim, shape, acc, d0, d1, f_s, int_cond, xmin_bd, xmax_bd, ymin_bd, ymax_bd, reverse_dy) for i in range(n_samples)]

    # create a pool of processes
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(executor.map(generate_sample, args))

    print(f"Time elapsed for data generation: {time.time() - start_time}")
    mid_time = time.time()

    print('Finished generating samples. Saving...')

    data_K, data_p, data_res, data_seed = zip(*results)

    save_dir = './data/darcy/'
    
    os.makedirs(save_dir, exist_ok=True)
    pd.DataFrame(data_seed).to_csv(save_dir + 'seeds.csv', index=False, header=False)
    pd.DataFrame(data_K).to_csv(save_dir + 'K_data.csv', index=False, header=False)
    pd.DataFrame(data_p).to_csv(save_dir + 'p_data.csv', index=False, header=False)
    pd.DataFrame(data_res).to_csv(save_dir + 'res_data.csv', index=False, header=False)

    # check that we truly have unique seeds
    if len(np.unique(data_seed)) != n_samples:
        raise ValueError('Seeds are not unique!')

    print(f'Time elapsed for data storing: {time.time() - mid_time}')

    print('Data generation finished.')

if __name__ == "__main__":
    main()