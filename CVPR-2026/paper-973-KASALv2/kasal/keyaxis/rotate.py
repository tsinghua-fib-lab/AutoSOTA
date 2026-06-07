# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

import numpy as np

def rotate_translate(axis, theta, t):
    """ Compute the transformation matrix based on the direction  
        of the symmetry/key axis, rotation angle, and translation component.  
    Parameters:  
        axis: Direction vector of the symmetry/key axis.  
        theta: Rotation angle.  
        t: Translation vector.  
    Returns:  
        M: Transformation matrix containing rotation and translation.  
    """

    theta = np.deg2rad(theta)
    axis = np.asarray(axis)
    axis /= np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    M = np.eye(4)
    M[:3, :3] = R
    t_ = np.dot(R, t.reshape((3)), )
    dt_ = t - t_ 
    M[:3, 3] = dt_
    return M

