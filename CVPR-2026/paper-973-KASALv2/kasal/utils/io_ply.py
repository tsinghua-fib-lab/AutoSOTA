# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

import copy
import numpy as np
import pymeshlab as ml
from kasal.bop_toolkit_lib import inout, misc
import kasal.config.config as config
from kasal.datasets.datasets_path import arrow_path


def simplify_3DModel_v2(input_file = '', targetfacenum = 20000, color_op = True):
    """ Simplify the object model.  
    Parameters:  
        input_file: Path to the object model.  
        targetfacenum: Number of faces after simplification.  
        color_op: Whether to retain color.  
    Returns:  
        vertex_matrix: Vertex coordinates.  
        vertex_color_matrix: Vertex colors.  
        face_matrix: Face indices.  
        vertex_normal_matrix: Vertex normals.  
    """

    mesh = ml.MeshSet()
    mesh.load_new_mesh(input_file)
    mesh_c = mesh.current_mesh()
    print(mesh.current_mesh_id())
    n1_ = mesh_c.face_number()
    print('Number of Vertices：', mesh_c.vertex_number())
    print('Number of Faces：', mesh_c.face_number())
    k0_ = int(np.log(targetfacenum/n1_) / np.log(4))
    if k0_ < 0:
        k0_ = 0
        
    key_ = list(mesh_c.textures().keys())
    if mesh_c.has_vertex_tex_coord() and len(key_) == 1:
        mesh.compute_texcoord_transfer_vertex_to_wedge()
        
    for _ in range(1):
        mesh.meshing_remove_duplicate_vertices()
        mesh.meshing_remove_duplicate_faces()
        mesh.meshing_repair_non_manifold_edges(method = 'Remove Faces')
        mesh.meshing_surface_subdivision_midpoint(iterations = 4)
        mesh_c = mesh.current_mesh()
        print('Number of Vertices：', mesh_c.vertex_number())
        print('Number of Faces：', mesh_c.face_number())
        if mesh_c.has_wedge_tex_coord() and len(key_) == 1 and color_op:
            mesh.meshing_decimation_quadric_edge_collapse_with_texture(targetfacenum=targetfacenum)
        else:
            mesh.meshing_decimation_quadric_edge_collapse(targetfacenum=targetfacenum, 
                                                                preservenormal = True, 
                                                                preserveboundary = True,
                                                                preservetopology = True,
                                                                autoclean = True,
                                                                planarquadric = True,
                                                                )
        mesh_c = mesh.current_mesh()
        print('Number of Vertices：', mesh_c.vertex_number())
        print('Number of Faces：', mesh_c.face_number())
    
    mesh.compute_normal_for_point_clouds()
    mesh_c = mesh.current_mesh()
    vertex_matrix = mesh_c.vertex_matrix().copy()
    if color_op:
        if mesh_c.has_wedge_tex_coord():
            mesh.transfer_texture_to_color_per_vertex()
        vertex_color_matrix = mesh_c.vertex_color_matrix().copy()
    else:
        vertex_color_matrix = np.ones((vertex_matrix.shape[0], 4), dtype=np.uint8) * 255
    face_matrix = mesh_c.face_matrix().copy()
    vertex_normal_matrix = mesh_c.vertex_normal_matrix().copy()
    mesh.clear()
    return vertex_matrix.astype(np.float32), vertex_color_matrix.astype(np.float32), face_matrix.astype(np.uint32), vertex_normal_matrix.astype(np.uint32)


def load_ply_model(input_ply, color_op=True):
    """ Load and simplify the object model. """
    
    model_i_ = {}
    vertices, colors, faces, normals = simplify_3DModel_v2(
                        input_file = input_ply,
                        targetfacenum = 40000,
                        color_op=color_op,
                        )
    
    model_i_['vertices'] = vertices
    model_i_['colors'] = colors
    model_i_['faces'] = faces
    model_i_['normals'] = normals
    model_i_['diameter'] = misc.calc_pts_diameter(vertices)
    return model_i_

def save_ply_model(model_i_, output_ply):
    """ Save the object model, the model representing rotational symmetry, and rotational symmetry information. """
    
    def add_arrow(model_save, model_arrow, rot_i, center_ch, k_ = 0):
        model_arrow_ = copy.deepcopy(model_arrow)
        model_arrow_['pts'] = np.dot(rot_i, model_arrow_['pts'].T).T + center_ch
        model_save['faces'] = np.concatenate((model_save['faces'], model_arrow_['faces'] + model_save['pts'].shape[0]), axis=0)
        model_save['pts'] = np.concatenate((model_save['pts'], model_arrow_['pts']), axis=0)
        model_save['normals'] = np.concatenate((model_save['normals'], model_arrow_['normals']), axis=0)
        if k_ == 0:
            model_save['colors'] = np.concatenate((model_save['colors'], model_arrow_['colors']), axis=0)
        elif k_ == 1:
            model_arrow_c = model_arrow_['colors'].copy()
            model_arrow_c[:,0] = model_arrow_['colors'][:,2]
            model_arrow_c[:,2] = model_arrow_['colors'][:,0]
            model_save['colors'] = np.concatenate((model_save['colors'], model_arrow_c), axis=0)
        elif k_ == 2:
            model_arrow_c = model_arrow_['colors'].copy()
            model_arrow_c[:,0] = model_arrow_['colors'][:,1]
            model_arrow_c[:,1] = model_arrow_['colors'][:,0]
            model_save['colors'] = np.concatenate((model_save['colors'], model_arrow_c), axis=0)
        return

    def transform_from_vec2_2_vec1(axis_2, axis_1):
        axis_3 = np.cross(axis_1, axis_2)
        if np.linalg.norm(axis_3) == 0:
            rot_ = np.eye(3)
        else:
            axis_3 /= np.linalg.norm(axis_3)
            axis_10 = np.cross(axis_1, axis_3)
            axis_20 = np.cross(axis_2, axis_3)
            axis_10 /= np.linalg.norm(axis_10)
            axis_20 /= np.linalg.norm(axis_20)
            points_original = np.array([axis_1, axis_3, axis_10])
            points_translated = np.array([axis_2, axis_3, axis_20])
        rot_ = np.dot(points_original.T, np.linalg.inv(points_translated.T))
        return rot_

    info_ = model_i_
    model_save = {}
    model_save['pts'] = np.array(info_['vertices'], dtype='float')
    model_save['normals'] = np.array(info_['normals'], dtype='float')
    model_save['faces'] = np.array(info_['faces'], dtype='int')
    while True:
        if np.max(np.array(info_['colors']).reshape((-1))) > 1:
            info_['colors'] /= 255
        else:
            break
    model_save['colors'] = np.array(info_['colors'] * 255, dtype='int')
    if 'sym_colors' in info_:
        model_save['colors'] = np.array(info_['sym_colors'] * 255, dtype='int')
        arrow_ply = inout.load_ply(arrow_path)
        colors_4 = np.zeros((arrow_ply['colors'].shape[0], 4), dtype=np.uint8)
        colors_4[:, 3] = 255
        colors_4[:, :3] = arrow_ply['colors']
        arrow_ply['colors'] = colors_4
        arrow_pts = arrow_ply['pts']
        arrow_x, arrow_y, arrow_z = arrow_pts[:,0], arrow_pts[:,1], arrow_pts[:,2]
        arrow_ply['pts'][:,0] -= (np.min(arrow_x)+np.max(arrow_x))/2
        arrow_ply['pts'][:,1] -= (np.min(arrow_y)+np.max(arrow_y))/2
        arrow_ply['pts'] = arrow_ply['pts'] / 1000 * info_['diameter'] * config.arrow_ratio
        arrow_x, arrow_y, arrow_z = arrow_pts[:,0], arrow_pts[:,1], arrow_pts[:,2]
        for axis_i in info_['axis']:
            rot_i = transform_from_vec2_2_vec1(np.array([0,0,1]), axis_i)
            add_arrow(model_save, arrow_ply, rot_i, info_['center_ch'])
        inout.save_ply(output_ply, model_save)
    else:
        arrow_ply = inout.load_ply(arrow_path)
        colors_4 = np.zeros((arrow_ply['colors'].shape[0], 4), dtype=np.uint8)
        colors_4[:, 3] = 255
        colors_4[:, :3] = arrow_ply['colors']
        arrow_ply['colors'] = colors_4
        arrow_pts = arrow_ply['pts']
        arrow_x, arrow_y, arrow_z = arrow_pts[:,0], arrow_pts[:,1], arrow_pts[:,2]
        arrow_ply['pts'][:,0] -= (np.min(arrow_x)+np.max(arrow_x))/2
        arrow_ply['pts'][:,1] -= (np.min(arrow_y)+np.max(arrow_y))/2
        arrow_ply['pts'] = arrow_ply['pts'] / 1000 * info_['diameter'] * config.arrow_ratio
        arrow_x, arrow_y, arrow_z = arrow_pts[:,0], arrow_pts[:,1], arrow_pts[:,2]
        if 'symmetries_continuous' in info_:
            if len(info_['symmetries_continuous'][0]['axis']):
                rot_i = transform_from_vec2_2_vec1(np.array([0,0,1]), np.array(info_['symmetries_continuous'][0]['axis']))
                add_arrow(model_save, arrow_ply, rot_i, np.array(info_['symmetries_continuous'][0]['offset']))
            else:
                rot_i = rot_i = transform_from_vec2_2_vec1(np.array([0,0,1]).astype(np.float32), np.array([1,0,0]).astype(np.float32))
                add_arrow(model_save, arrow_ply, rot_i, np.array(info_['symmetries_continuous'][0]['offset']), k_ = 0)
                rot_i = transform_from_vec2_2_vec1(np.array([0,0,1]).astype(np.float32), np.array([0,1,0]).astype(np.float32))
                add_arrow(model_save, arrow_ply, rot_i, np.array(info_['symmetries_continuous'][0]['offset']), k_ = 2)
                rot_i = np.eye(3)
                add_arrow(model_save, arrow_ply, rot_i, np.array(info_['symmetries_continuous'][0]['offset']), k_ = 1)
            inout.save_ply(output_ply, model_save)
        if 'symmetries_discrete' in info_ and config.save_2_fold_a is True:
            rot_i = transform_from_vec2_2_vec1(np.array([1,0,0]), np.array(info_['symmetries_continuous'][0]['axis']))
            add_arrow(model_save, arrow_ply, rot_i, np.array(info_['symmetries_continuous'][0]['offset']), k_=1)
            inout.save_ply(output_ply, model_save)
    model_info_i = {}
    info_list_key_1 = [
        'diameter', 'min_x', 'min_y', 'min_z', 'size_x', 'size_y', 'size_z'
    ]
    info_list_key_2 = [
        'symmetries_discrete', 'symmetries_continuous', 
        'symmetries_continuous_2', 'symmetries_continuous_3'
    ]
    for key_i in info_list_key_1:
        if key_i in info_:
            model_info_i[key_i] = float(info_[key_i])
    for key_i in info_list_key_2:
        if key_i in info_:
            model_info_i[key_i] = np.array(info_[key_i]).tolist()
    return model_info_i
