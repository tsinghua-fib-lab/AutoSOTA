# Laplacian_matrix.py 

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

def test_dist_Laplacian(distance, tol, index, i):
    '''
    Return the value of index or i depending on the proximity of a neighbor if the neighbor 
    exist; here if the distance is too high, this mean that we do not consider the neighbor 
    given by index.

            Parameters:
                    distance (float): Euclidean distance to the closest neighbor.
                    tol (float): Tolerance threshold to accept the neighbor's value.
                    index (int): Index of the closest neighbor in the band.
                    i (int): Index of the current point in the band.

            Returns:
                    float: Value of u_band at neighbor index if within tolerance, else at current 
                           index.
    '''
    if distance<tol:
        return index
    else:
        return i
    
def Laplacian_matrix(band_points, Tree_Band_points, delta_x):
    '''
    Compute the Laplacian matrix for a given domain without the division by delta_x**2 !!!!!
            
            Parameters:
                band_points (np.ndarray): Array of shape (N_points, 3) representing the coordinates 
                                          of the narrow band points.
                Tree_Band_points (scipy.spatial.KDTree): KDTree built from band_points to query 
                                                         nearest neighbors efficiently.
                delta_x (float): Spatial resolution used to compute finite differences.
                
            Returns:
                Lap (csr_matrix): Laplacian matrix, shape (N_points, N_points).
                delta_x**2 (float): Divide by this to obtain the true Laplacian.
    '''
    N_pts = band_points.shape[0]
    Lap = lil_matrix((N_pts,N_pts))
    
    tol = delta_x/10
    
    for i in range(N_pts):
        x, y, z = band_points[i,0], band_points[i,1], band_points[i,2]
                        
        Lap[i, i] += -6
        
        distance, index = Tree_Band_points.query(np.array([x + delta_x, y, z]), k=1)
        Lap[i, test_dist_Laplacian(distance, tol, index, i)] += 1
                    
        distance, index = Tree_Band_points.query(np.array([x - delta_x, y, z]), k=1)
        Lap[i, test_dist_Laplacian(distance, tol, index, i)] += 1

        distance, index = Tree_Band_points.query(np.array([x, y + delta_x, z]), k=1)
        Lap[i, test_dist_Laplacian(distance, tol, index, i)] += 1

        distance, index = Tree_Band_points.query(np.array([x, y - delta_x, z]), k=1)
        Lap[i, test_dist_Laplacian(distance, tol, index, i)] += 1

        distance, index = Tree_Band_points.query(np.array([x, y, z + delta_x]), k=1)
        Lap[i, test_dist_Laplacian(distance, tol, index, i)] += 1

        distance, index = Tree_Band_points.query(np.array([x, y, z - delta_x]), k=1)
        Lap[i, test_dist_Laplacian(distance, tol, index, i)] += 1
        
    return csr_matrix(Lap), delta_x**2

# NOTE:
# The Laplacian is assembled on the full band.
# Near band boundaries, missing neighbors fallback to self.
# Since only interior band points are used in the final surface solution, 
# this does not affect the reported results.