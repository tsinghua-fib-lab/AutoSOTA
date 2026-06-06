# RBF_update.py

from scipy.linalg import cho_factor, cho_solve
import numpy as np

def phi(r, epsilon=1):
    return np.exp(-(epsilon * r)**2)

def precompute_rbf_data(band_points, Tree_Band_points, final_surface_points, k=16, epsilon=1.0):
    '''
    Precompute RBF interpolation data for local parts. Approximately 3 seconds for 17000 
    final_surface_points.

    Parameters:
        band_points (np.ndarray): Array of shape (N, 3), local bands.
        Tree_Band_points (KDTree): KDTree built on band_points.
        final_surface_points (np.ndarray): Array of shape (N, 3), predicted surface points.
        k (int): Number of nearest neighbors.
        epsilon (float): RBF kernel parameter.

    Returns:
        neighbors_indices (np.ndarray): Array (N, k) with neighbor indices.
        factors (List[Tuple]): List of Cholesky factors per point.
        phi_vecs (List[np.ndarray]): List of RBF kernel vectors per point.
    '''
    N = final_surface_points.shape[0]
    idxs = []
    factors = []
    phi_vecs = []

    for j in range(N):
        _, idx = Tree_Band_points.query(final_surface_points[j], k=k)
        idxs.append(idx)
        
        
        X = band_points[idx]                                           # (k, 3)
        dists = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)  # (k, k)
        A = phi(dists, epsilon)
        try:
            factor = cho_factor(A)
        except np.linalg.LinAlgError:
            print(f"[!] Cholesky failed at point {j}")
            factor = (None, A)

        r = np.linalg.norm(X - final_surface_points[j], axis=1)  # (k,)
        phi_vec = phi(r, epsilon)

        factors.append(factor)
        phi_vecs.append(phi_vec)

    neighbors_indices = np.array(idxs)  # (N, k)

    return neighbors_indices, factors, phi_vecs

def interpolate_from_precomputed(u_band, neighbors_indices, factors, phi_vecs, clipping=False):
    '''
    Interpolate values at surface points using precomputed RBF data.

    Parameters:
        u_band (np.ndarray): Array of shape (M,) containing function values at local bands.
        neighbors_indices (List[np.ndarray]): List of arrays (k,) of neighbor indices per point.
        phi_A_factors (List[Tuple]): List of Cholesky factors or (None, A) per point.
        rbf_weights_vectors (List[np.ndarray]): List of arrays (k,) with φ(r) vectors per surface point.
        clipping (bool): True if you want to clip the value of the interpolation with respect to the neighbors 
                         values

    Returns:
        interpolated_values (np.ndarray): Array of shape (N,), interpolated values at surface points.
    '''
    N = len(factors)
    interpolated_values = np.zeros(N)

    for j in range(N):
        idx = neighbors_indices[j]        # (k,)
        f_vals = u_band[idx]              # (k,)

        factor = factors[j]
        phi_vec = phi_vecs[j]                        # (k,)

        if factor[0] is None:
            w = np.linalg.solve(factor[1], f_vals)
        else:
            w = cho_solve(factor, f_vals)

        val = np.dot(w, phi_vec)
        
        if clipping:
            interpolated_values[j] = np.clip(val, np.min(f_vals), np.max(f_vals))
        else:
            interpolated_values[j] = val
    
    return interpolated_values