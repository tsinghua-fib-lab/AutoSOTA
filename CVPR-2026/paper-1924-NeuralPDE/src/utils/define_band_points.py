# define_band_points.py 

import numpy as np 

def bounding_box(points_cloud):
    '''
    Return the smallest box in which the points are.

            Parameters:
                    points_cloud (np.ndarray): Array of size (N,d) where N is the number of points of the mesh 
                                            and d the dimension.

            Returns:
                    bound_box (np.ndarray): Array of size (d,2) where d is the dimension of the points and 
                                          for each dimension we have the min and max values
    '''
    mins = np.min(points_cloud, axis=0)
    maxs = np.max(points_cloud, axis=0)
    
    bound_box = np.stack((mins, maxs), axis=1)
    
    return bound_box

def create_grid_points(bound_box, delta_x, lam):
    '''
    Create 1D grids for each axis

            Parameters:
                    bound_box (np.ndarray): Array of size (d,2) where d is the dimension of the points and 
                                          for each dimension we have the min and max values
                    delta_x (float): space between each grid points.
                    lam (float): define the distance between the original surface and the embedded volume.

            Returns:
                    grid_points (list): List of len d, where each elements correspond to a discretisation 
                    of a given dimension. 
    '''
    d = bound_box.shape[0]
    grid_points = [np.arange(bound_box[i,0]-lam, bound_box[i,1]+lam+delta_x, delta_x) for i in range(d)]
    
    return grid_points

def keep_closest_points_to_S(Volumetric_domain_points, Tree, lam, threshold):
    '''
    Return the closest points (with a distance of lam) of the volumetric domain to the surface.

            Parameters:
                    Volumetric_domain_points (np.ndarray): Array of points (N,d), N number of points, d their 
                                                         dimensions which represents the discretisation of 
                                                         the bounding box around the surface (the mesh).
                    Tree (KDTree): Tree to quickly access the closest points of mesh_points.
                    lam (float): define the distance between the original surface and the embedded volume.
                    threshold (float): enable to know which points are in the band but far away from the surface.

            Returns:
                    band_points (np.ndarray): Array of size (N_points, d) where N_points is the number of points
                                            in the band volume and d their dimensions. 
    '''
    N = Volumetric_domain_points.shape[0]
    band = np.zeros(N, dtype=bool)
    mask_threshold = np.zeros(N, dtype=bool)

    for j in range(N):
        distance, _ = Tree.query(Volumetric_domain_points[j], k=1)
        if distance <= lam:
            band[j] = True
            if distance > threshold:
                mask_threshold[j] = True
        else:
            band[j] = False
          
    band_points = Volumetric_domain_points[band]
    
    return band_points, mask_threshold[band]

def define_band_points(delta_x, surface_points, Tree_surface_points, dist_to_surface, threshold):
    '''
    Return the band arround the surface.

            Parameters:
                    delta_x (float): space between each band points.
                    surface_points (np.ndarray): Array of size (N,d) where N is the number of points on the surface 
                                                 and d the dimension.
                    Tree_surface_points (KDTree): Tree to quickly access the closest points of surface_points.
                    define_distance (bool): True if you want to define your own lam
                    dist_to_mesh (float or NoneType): The value of lam which is the distance from the mesh to the 
                                                      border of the band. Otherwise None.

            Returns:
                    band_points (np.ndarray): Array of size (N_points, d) where N_points is the number of points
                                              in the band volume and d their dimensions. 
    '''
    print("Warning: dist_to_surface, delta_x and the number k should verify the equation to ensure that when you defined " \
    "local_parts after you are able to catch every band points.")
    
    bound_box = bounding_box(surface_points)
    grid_pts = create_grid_points(bound_box, delta_x, dist_to_surface)
    
    Volumetric_domain = np.meshgrid(*grid_pts, indexing='ij')
    Volumetric_domain_points = np.stack([m.ravel() for m in Volumetric_domain], axis=-1)

    band_points, mask_threshold = keep_closest_points_to_S(
        Volumetric_domain_points, Tree_surface_points, dist_to_surface, threshold)
    
    return band_points, mask_threshold