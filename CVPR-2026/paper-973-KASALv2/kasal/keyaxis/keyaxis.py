# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

import numpy as np
from tqdm import tqdm
from kasal.keyaxis.rotate import rotate_translate
from kasal.geometry.circular_sampling import circular_sampling
from kasal.keyaxis.color_error import rot_axis_color_error
from kasal.bop_toolkit_lib.view_sampler import fibonacci_sampling
from kasal.bop_toolkit_lib.transform import euler_matrix


def cal_KA1(pts, colors, diameter, div = 2, center_ch=None, sample_num=20001, half_sphere = True, op = 'pts', xyz_op=None):
    ''' Locate the first key axis.  
    Parameters:  
        pts: Vertex coordinates of the object.  
        colors: Vertex colors of the object.  
        diameter: Diameter of the object.  
        div: Order of the first key axis.  
        center_ch: Initial rotation center.  
        sample_num: Number of Fibonacci sphere sampling points.  
        half_sphere: Whether to sample only half of the sphere.  
        op: If set to 'pts', the object has geometric rotational symmetry;  
            if set to 'colors', the object has texture rotational symmetry.  
        xyz_op: Set the key axis to be close to the x, y, or z coordinate axis,  
                used only for objects where the symmetry axis cannot be  
                correctly located.  
    Returns:  
        axis: Direction of the first key axis.  
        axis_mat: Set of transformation matrices around the first key axis.  
    '''
    
    if center_ch is None:
        center_ch = np.array([0,0,0], dtype=np.float64)
    xyz_axis = fibonacci_sampling(sample_num, 1)
    xyz_axis = np.array(xyz_axis)
    ai, aj, ak = (4*np.pi) * (np.random.random(3) - 0.5)
    M_ = euler_matrix(ai, aj, ak)[:3, :3]
    xyz_axis = np.dot(M_, xyz_axis.T).T
    if half_sphere:
        xyz_axis = xyz_axis[ xyz_axis[:, 2]>=0, :]
    if xyz_op:
        if 'X' in xyz_op:
            xyz_axis_c = xyz_axis.copy()
            xyz_axis_c[:, 0] = np.abs(xyz_axis_c[:, 0])
            xyz_axis_c = np.linalg.norm(xyz_axis_c - np.array([1, 0, 0]), axis=1)
            xyz_axis = xyz_axis[xyz_axis_c < 0.5, :]
        if 'Y' in xyz_op:
            xyz_axis_c = xyz_axis.copy()
            xyz_axis_c[:, 1] = np.abs(xyz_axis_c[:, 1])
            xyz_axis_c = np.linalg.norm(xyz_axis_c - np.array([0, 1, 0]), axis=1)
            xyz_axis = xyz_axis[xyz_axis_c < 0.5, :]
        if 'Z' in xyz_op:
            xyz_axis_c = xyz_axis.copy()
            xyz_axis_c[:, 2] = np.abs(xyz_axis_c[:, 2])
            xyz_axis_c = np.linalg.norm(xyz_axis_c - np.array([0, 0, 1]), axis=1)
            xyz_axis = xyz_axis[xyz_axis_c < 0.5, :]
    dis_list = []
    color_list = []
    mat_list = []
    vec_list = []
    dis_ang = 360 / div  # IDEA-005: exact fundamental rotation for all orders
    for ch_p in tqdm(xyz_axis):
        r_m = rotate_translate(ch_p, dis_ang, center_ch)
        dis_, mnc_ = rot_axis_color_error(pts, colors, r_m)
        dis_list.append(dis_)
        mat_list.append(r_m)
        vec_list.append(ch_p)
        color_list.append(mnc_)
    dis_list = np.array(dis_list)
    color_list = np.array(color_list)
    
    if op == 'pts':
        dis_list = dis_list
    if op == 'colors':
        # dis_list = (dis_list + color_list) / 2
        dis_list = color_list
    sort_id = np.argsort(dis_list)
    dis_list_s = np.array(dis_list)[sort_id[:10]]
    mat_list_s = np.array(mat_list)[sort_id[:10]]
    vec_list_s = np.array(vec_list)[sort_id[:10]]
    print('cal_KA1  %s / %s'%(str(np.min(dis_list_s[0])) , str(diameter)) )
    
    axis_mat = []
    for div_i in range(div - 1):
        c_ang = (div_i+1) * dis_ang
        r_m = rotate_translate(vec_list_s[0], c_ang, center_ch)
        axis_mat.append(r_m)
        
    return {
        'axis' : vec_list_s[0],
        'axis_mat' : axis_mat,
        'xyz_axis' : xyz_axis,
        'dis_list' : dis_list,
    }

def cal_KA2(ang_b12, pts, colors, axis_1, diameter, div = 4, center_ch=None, sample_num=360, op = 'pts'):
    ''' Locate the second key axis.  
    Parameters:  
        ang_b12: Angle between the first and second key axes.  
        pts: Vertex coordinates of the object.  
        colors: Vertex colors of the object.  
        axis_1: Direction of the first key axis.  
        diameter: Diameter of the object.  
        div: Order of the second key axis.  
        center_ch: Initial rotation center.  
        sample_num: Number of circular sampling points.  
        op: If set to 'pts', the object has geometric rotational symmetry;  
            if set to 'colors', the object has texture rotational symmetry.  
    Returns:  
        axis: Direction of the second key axis.  
        axis_mat: Set of transformation matrices around the second key axis.  
    '''
    
    if center_ch is None:
        center_ch = np.array([0,0,0], dtype=np.float64)
    xyz_axis = circular_sampling(axis_1, ang_b12, num_samples = sample_num)
    dis_list = []
    mat_list = []
    vec_list = []
    color_list = []
    dis_ang = 360 / div
    for ch_p in xyz_axis:
        r_m = rotate_translate(ch_p, dis_ang, center_ch)
        dis_, mnc_ = rot_axis_color_error(pts, colors, r_m)
        dis_list.append(dis_)
        mat_list.append(r_m)
        vec_list.append(ch_p)
        color_list.append(mnc_)
    
    dis_list = np.array(dis_list)
    color_list = np.array(color_list)
    if op == 'pts':
        dis_list = dis_list
    if op == 'colors':
        dis_list = color_list
    sort_id = np.argsort(dis_list)
    dis_list_s = np.array(dis_list)[sort_id[:10]]
    vec_list_s = np.array(vec_list)[sort_id[:10]]
    print('cal_KA2  %s / %s'%(str(np.min(dis_list_s[0])) , str(diameter)) )
    
    axis_mat = []
    for div_i in range(div - 1):
        c_ang = (div_i+1) * dis_ang
        r_m = rotate_translate(vec_list_s[0], c_ang, center_ch)
        axis_mat.append(r_m)
        
    return {
        'axis' : vec_list_s[0],
        'axis_mat' : axis_mat,
        'xyz_axis' : xyz_axis,
        'dis_list' : dis_list,
    }
