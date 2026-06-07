# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

import numpy as np 
from scipy import spatial

def rot_axis_color_error(pts, colors, mat,):
    ''' Use KDTree to find the nearest point pairs before and after  
        rotation around the axis, then compute the average distance  
        and color errors.  
    Parameters:  
        pts: Vertex coordinates of the object.  
        colors: Vertex colors of the object.  
        mat: Transformation matrix containing rotation and translation.  
    Returns:  
        distance_error: Distance error.  
        color_error: Color error.  
    '''
    ply_pts_m = np.dot(pts, mat[:3,:3])+mat[:3,3]
    nn_index = spatial.cKDTree(pts)
    d1, t1_index = nn_index.query(ply_pts_m, k=1)
    dc_ = colors - colors[t1_index, :]
    ndc_ = np.linalg.norm(dc_, axis=1)
    color_error = np.mean(ndc_)
    distance_error = np.mean(d1)
    return distance_error, color_error