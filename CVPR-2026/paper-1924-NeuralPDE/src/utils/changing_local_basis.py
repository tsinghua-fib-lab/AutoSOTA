# changing_local_basis.py

import numpy as np

def random_unit_vector():
    v = np.random.normal(size=3)  
    return v / np.linalg.norm(v)

def ensure_BON(normal, tangent1, tangent2, eps=1e-8):
    '''
    Ensure that the triplet (normal, tangent1, tangent2) forms an orthonormal basis. Because those
    vectors can come with noise.
    In practice we should not enter in the if statements, but just in case.
    '''
    # Step 1: normalize the normal
    n = normal / np.linalg.norm(normal)

    # Step 2: make tangent1 orthogonal to normal
    if np.linalg.norm(np.cross(n, tangent1)) < eps:
        random_vector = random_unit_vector()
        while np.linalg.norm(np.cross(n, random_vector)) < eps:
            random_vector = random_unit_vector()
        t1 = random_vector - np.dot(random_vector, n) * n
        t1 /= np.linalg.norm(t1)
    else:
        comp1 = np.dot(tangent1, n)
        t1 = tangent1 - comp1 * n   
        t1 /= np.linalg.norm(t1)

    # Step 3: make tangent2 orthogonal to both normal and tangent1
    if np.abs(np.dot(np.cross(n, t1), tangent2)) < eps:
        random_vector = random_unit_vector()
        while np.abs(np.dot(np.cross(n, t1), random_vector)) < eps:
            random_vector = random_unit_vector()
        t2 = random_vector - np.dot(random_vector, n) * n - np.dot(random_vector, t1) * t1
        t2 /= np.linalg.norm(t2)
    else:
        t2 = tangent2 - np.dot(tangent2, n) * n - np.dot(tangent2, t1) * t1
        t2 /= np.linalg.norm(t2)

    return n, t1, t2

def change_local_basis(central_pt, normal, tangent1, tangent2, points_cloud1, points_cloud2=None):
    '''
    Args :
        central_pt: (3,) array, the point on the surface where the basis is defined
        normal, tangent1, tangent2: (3,) arrays defining the local basis at central_pt
        points_cloud: (N, 3) array of points to change basis

    Returns :
        new_origin: (N, 3) array of points in the new basis, centered at central_pt
    '''
    # Basis matrix 
    n, t1, t2 = ensure_BON(normal, tangent1, tangent2)    
    R = np.stack([n, t1, t2], axis=1)

    # Change of basis
    new_origin1 = points_cloud1 - central_pt
    new_basis1 = new_origin1 @ R  
    
    if points_cloud2 is not None:
        new_origin2 = points_cloud2 - central_pt
        new_basis2 = new_origin2 @ R
        return new_basis1, new_basis2
    
    return new_basis1
