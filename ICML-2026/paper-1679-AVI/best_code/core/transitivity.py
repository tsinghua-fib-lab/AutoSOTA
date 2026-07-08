"""

"""
import numpy as np


def propagate_transitivity(R, R_t, m):
    """

    """
    
    current_rejections = R | R_t
    
    T_matrix = np.zeros((m, m), dtype=bool)
    
    for i in range(m):
        for j in range(m):
            if j == i:
                continue
            for k in range(m):
                if k == i or k == j:
                    continue
                
                if current_rejections[j, i] and current_rejections[i, k]:
                    T_matrix[j, k] = True
    
    return T_matrix

