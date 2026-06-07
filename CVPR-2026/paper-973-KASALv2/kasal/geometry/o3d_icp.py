# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

import open3d as o3d
import numpy as np


def refine_registration(source, target, voxel_size):
    ''' ICP using Open3D. '''
    
    distance_threshold = voxel_size*10
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),  # IDEA-008: point-to-plane for better convergence
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-06, 
                                                          relative_rmse=1e-06,
                                                          max_iteration=100))
    return result

def refine_center_direction(axis_info, model_info, center_ch):
    ''' Use ICP to refine the direction of key axes and the rotation center.  
    Parameters:  
        axis_info: Information of the key axes.  
        model_info: Information of the object model.  
        center_ch: Initial rotation center.  
    Returns:  
        axis_refined: Refined direction of the key axes.  
        center_ch_refined: Refined rotation center.  
    '''

    voxel_size = model_info['diameter'] / 50
    pcd_1 = o3d.geometry.PointCloud()
    pcd_1.points = o3d.utility.Vector3dVector(model_info['vertices'].astype(np.float32))
    pcd_1 = pcd_1.voxel_down_sample(voxel_size / 4)
    pcd_1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size, max_nn=50))
    mat_list = []
    mat_list.append(np.eye(4))
    for id_ in range(len(axis_info['axis_mat'])):
        pcd_2 = o3d.geometry.PointCloud()
        pcd_2.points = o3d.utility.Vector3dVector(np.dot(axis_info['axis_mat'][id_][:3,:3], model_info['vertices'].T).T.astype(np.float32) + axis_info['axis_mat'][id_][:3,3])
        pcd_2 = pcd_2.voxel_down_sample(voxel_size / 4)
        pcd_2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size, max_nn=50))
        results_ = refine_registration(pcd_2, pcd_1, voxel_size, )
        mat_ = results_.transformation
        mat_list.append(mat_)
    center_ch_list = []
    axis_list = []
    for mat_ in mat_list:
        center_ch_list.append(np.dot(mat_[:3,:3], center_ch) + mat_[:3, 3])
        axis_list.append(np.dot(mat_[:3,:3], center_ch + axis_info['axis']) - np.dot(mat_[:3,:3], center_ch))
    center_ch_list = np.array(center_ch_list) 
    axis_list = np.array(axis_list) 
    center_ch_refined = np.mean(center_ch_list, axis = 0)
    axis_refined = np.mean(axis_list, axis = 0)
    axis_refined /= np.linalg.norm(axis_refined)
    return  center_ch_refined, axis_refined
