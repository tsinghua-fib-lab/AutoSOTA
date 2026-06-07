# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

import math
import numpy as np

def circular_sampling(axis, ang_b12, radius=1, num_samples=3600):
    ''' Uniformly sample 3D points on a circle.  
    Parameters:  
        axis: Direction of the first key axis.  
        ang_b12: Angle between the first and second key axes.  
        radius: Sampling radius.  
        num_samples: Number of samples.  
    Returns:  
        samples: Uniformly sampled 3D points.  
    '''
    
    axis_1 = np.array([0, 0, 1])
    axis_2 = np.array(axis)
    axis_3 = np.cross(axis_1, axis_2)
    if np.linalg.norm(axis_3) == 0:
        R = np.eye(3)
    else:
        axis_3 /= np.linalg.norm(axis_3)
        axis_10 = np.cross(axis_1, axis_3)
        axis_20 = np.cross(axis_2, axis_3)
        axis_10 /= np.linalg.norm(axis_10)
        axis_20 /= np.linalg.norm(axis_20)
        points_original = np.array([axis_1, axis_3, axis_10])
        points_translated = np.array([axis_2, axis_3, axis_20])
    R = np.dot(points_translated.T, np.linalg.inv(points_original.T))
    samples = []
    ang_b12_ = ang_b12/180*np.pi
    for i in range(num_samples):
        theta = 2 * math.pi * i / num_samples
        p = np.dot(R, np.array([radius*math.cos(theta)*np.sin(ang_b12_), radius*math.sin(theta)*np.sin(ang_b12_), radius*np.cos(ang_b12_)]))
        samples.append(p)
    return np.array(samples)
