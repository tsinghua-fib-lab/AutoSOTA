# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

import fpsample, trimesh
import numpy as np
from scipy import spatial
from kasal.symmetry_lab.symmetry_axis_template import clear_sym
from kasal.utils.load_stp import stp2info, axis_num
from kasal.geometry.o3d_icp import refine_center_direction
from kasal.keyaxis.keyaxis import cal_KA1, cal_KA2
from kasal.keyaxis.rotate import rotate_translate

def cal_model_sym(model_i_, step_path = None, sym_op = None, sym_aware = False, op = 'pts', sample_num = 10001, fpsample_num = 1500, icp_op = True, xyz_op = None, half_sphere = True):
    """ Compute the rotational symmetry of an object.  
    Parameters:  
        model_i_: Object model information, including vertex coordinates, colors, normals, face indices, etc.  
        step_path: You can design object models using software like AutoCAD or SolidWorks and export them as STP files.  
                This function analyzes the rotational symmetry type based on the STP file and locates the symmetry axis  
                within the object model.  
                Note: The STP file must not contain any curved surfaces, only polygonal planes.  
                Additionally, the object model in the STP file must be strictly closed.  
        sym_op: Type of rotational symmetry.  
        op: If set to 'pts', the object has geometric rotational symmetry;  
            if set to 'colors', the object has texture rotational symmetry.  
        sample_num: Number of Fibonacci sphere sampling points.  
        fpsample_num: Number of FPS algorithm sampling points.  
        icp_op: Whether to enable ICP to refine the key axis direction and rotation center.  
        xyz_op: Sets the key axis to be close to the x, y, or z axis,  
                used only for objects where the symmetry axis cannot be correctly located.  

    Returns:  
        model_i_: Updated object model information, including the direction of the rotational symmetry axis and rotation center.  
    """
    
    axis_list = []
    if isinstance(step_path, str):
        if step_path is not None:
            face_line_point_list = stp2info(step_path)
            norm_axis_f_all, num_f_all = axis_num(face_line_point_list)
            for (a_, n_) in zip(norm_axis_f_all, num_f_all):
                axis_list.append({'axis_l' : [a_.tolist()], 'num' : int(n_)})
    elif isinstance(step_path, list):
        axis_list = step_path
    elif step_path is None:
        step_path
    else:
        raise ValueError('The type of "step_path" is unsupported ! ')
    np.random.seed(0)
    kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(model_i_['vertices'], min([fpsample_num, model_i_['vertices'].shape[0]]), h=3)
    ply_pts = model_i_['vertices'][kdline_fps_samples_idx,:]
    colors_ = model_i_['colors'][kdline_fps_samples_idx,:]
    diameter = model_i_['diameter']
    clear_sym(model_i_)
    if sym_op == 'symmetries_continuous' or sym_op == 'symmetries_continuous_2' or sym_op == 'symmetries_continuous_3':
        mesh_mass = trimesh.Trimesh(vertices=model_i_['vertices'].astype(np.float32), faces=model_i_['faces'].astype(np.uint32))
        mesh_mass_c = mesh_mass.convex_hull
        center_ch = mesh_mass_c.mass_properties['center_mass']
        print('center_ch: ', center_ch)
        axis_1_info = cal_KA1(ply_pts, colors_, diameter, div=3, sample_num = sample_num, center_ch=center_ch, op=op, xyz_op=xyz_op, half_sphere=half_sphere)
        loc_sym_axis_1 = axis_1_info['axis']
        if icp_op:
            center_ch, loc_sym_axis_1 = refine_center_direction(axis_1_info, model_i_, center_ch)
        if sym_op == 'symmetries_continuous_2' or sym_op == 'symmetries_continuous_3':
            axis_2_info = cal_KA2(90, ply_pts, colors_, loc_sym_axis_1, diameter, div=2, sample_num = 360, center_ch=center_ch, op=op)
            loc_sym_axis_2 = axis_2_info['axis']
            if icp_op:
                center_ch, loc_sym_axis_2 = refine_center_direction(axis_2_info, model_i_, center_ch)
        rot_model_ = False
        if rot_model_:
            target_vec_ = np.array([0, 0, 1]) 
            axis_1 = target_vec_
            axis_2 = np.array(loc_sym_axis_1)
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
            model_i_['vertices'][:, 0] -= center_ch[0]
            model_i_['vertices'][:, 1] -= center_ch[1]
            model_i_['vertices'][:, 2] -= center_ch[2]
            model_i_['vertices'] = np.dot(rot_, model_i_['vertices'].T).T
            model_i_["min_x"] = np.min(model_i_['vertices'][:, 0])
            model_i_["min_y"] = np.min(model_i_['vertices'][:, 1])
            model_i_["min_z"] = np.min(model_i_['vertices'][:, 2])
            model_i_["size_x"] = np.max(model_i_['vertices'][:, 0]) - np.min(model_i_['vertices'][:, 0])
            model_i_["size_y"] = np.max(model_i_['vertices'][:, 1]) - np.min(model_i_['vertices'][:, 1])
            model_i_["size_z"] = np.max(model_i_['vertices'][:, 2]) - np.min(model_i_['vertices'][:, 2])
            if sym_op == 'symmetries_continuous_2':
                model_i_['symmetries_continuous'] = [{"axis": target_vec_.reshape((3)).tolist(), "offset": [0, 0, 0]}]
                model_i_['symmetries_discrete'] = [[1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1]]
            else:
                model_i_['symmetries_continuous'] = [{"axis": target_vec_.reshape((3)).tolist(), "offset": [0, 0, 0]}]
        else:
            model_i_["min_x"] = np.min(model_i_['vertices'][:, 0])
            model_i_["min_y"] = np.min(model_i_['vertices'][:, 1])
            model_i_["min_z"] = np.min(model_i_['vertices'][:, 2])
            model_i_["size_x"] = np.max(model_i_['vertices'][:, 0]) - np.min(model_i_['vertices'][:, 0])
            model_i_["size_y"] = np.max(model_i_['vertices'][:, 1]) - np.min(model_i_['vertices'][:, 1])
            model_i_["size_z"] = np.max(model_i_['vertices'][:, 2]) - np.min(model_i_['vertices'][:, 2])
            if sym_op == 'symmetries_continuous_2':
                axis_2 = loc_sym_axis_2
                r_m = rotate_translate(axis_2, 180, center_ch)
                model_i_['symmetries_continuous'] = [{"axis": loc_sym_axis_1.reshape((3)).tolist(), "offset": center_ch.reshape((3)).tolist()}]
                model_i_['symmetries_discrete'] = [r_m.reshape((16)).tolist()]
            elif sym_op == 'symmetries_continuous':
                model_i_['symmetries_continuous'] = [{"axis": loc_sym_axis_1.reshape((3)).tolist(), "offset": center_ch.reshape((3)).tolist()}]
            elif sym_op == 'symmetries_continuous_3':
                model_i_['symmetries_continuous'] = [{"axis" : [], "offset": center_ch.reshape((3)).tolist()}]
                
    elif sym_op == 'symmetries_discrete':
        mesh_mass = trimesh.Trimesh(vertices=model_i_['vertices'].astype(np.float32), faces=model_i_['faces'].astype(np.uint32))
        mesh_mass_c = mesh_mass.convex_hull
        center_ch = mesh_mass_c.mass_properties['center_mass']
        print('center_ch: ', center_ch)
        num_list = []
        for axis_i in axis_list:
            num_list.append(axis_i['num'])
        num_arg_sort = np.argsort( - np.array(num_list))
        axis_info_in_obj_pt_model = {}
        if len(num_list) == 1:
            idx_sym_axis_1 = num_arg_sort[0] 
            div_1 = axis_list[idx_sym_axis_1]['num'] 
            axis_1_info = cal_KA1(ply_pts, colors_, diameter, div=div_1+1, sample_num = sample_num, center_ch=center_ch, op=op, xyz_op=xyz_op, half_sphere=half_sphere)
            loc_sym_axis_1 = axis_1_info['axis']
            if icp_op:
                center_ch, loc_sym_axis_1 = refine_center_direction(axis_1_info, model_i_, center_ch)
            axis_info_in_obj_pt_model['axis'] = [loc_sym_axis_1]
            model_i_['axis'] = axis_info_in_obj_pt_model['axis']
            model_i_['center_ch'] = center_ch
            axis_info_in_obj_pt_model['axis_mat'] = []
            for div_i in range(div_1):
                r_m = rotate_translate(loc_sym_axis_1, (div_i+1) * 360 / (div_1+1), center_ch)
                axis_info_in_obj_pt_model['axis_mat'].append(r_m)
            axis_info_in_obj_pt_model['details_sym_axis_1'] = axis_1_info
        
        if len(num_list) >= 2:
            idx_sym_axis_1 = num_arg_sort[0] 
            div_1 = axis_list[idx_sym_axis_1]['num'] 
            axis_1_info = cal_KA1(ply_pts, colors_, diameter, div=div_1+1, sample_num = sample_num, center_ch=center_ch, op=op, half_sphere=half_sphere)
            loc_sym_axis_1 = axis_1_info['axis']
            if icp_op:
                center_ch, loc_sym_axis_1 = refine_center_direction(axis_1_info, model_i_, center_ch)
            idx_sym_axis_2 = num_arg_sort[1] 
            div_2 = axis_list[idx_sym_axis_2]['num'] 
            axis_1 = axis_list[idx_sym_axis_1]['axis_l'][0]
            axis_1 = np.array(axis_1) / np.linalg.norm(axis_1)
            axis_2 = axis_list[idx_sym_axis_2]['axis_l'][0]
            axis_2 = np.array(axis_2) / np.linalg.norm(axis_2)
            dot_product = np.dot(axis_1, axis_2)
            angle_b12 = np.arccos(dot_product)
            angle_degrees = np.degrees(angle_b12)
            ang_b12 = angle_degrees
            axis_2_info = cal_KA2(ang_b12, ply_pts, colors_, loc_sym_axis_1, diameter, div=div_2+1, sample_num = 360, center_ch=center_ch, op=op)
            loc_sym_axis_2 = axis_2_info['axis']
            if icp_op:
                center_ch, loc_sym_axis_2 = refine_center_direction(axis_2_info, model_i_, center_ch)
            
            points_original = np.array([axis_1, axis_2, np.cross(axis_1, axis_2)/np.linalg.norm(np.cross(axis_1, axis_2))])
            points_translated = np.array([loc_sym_axis_1, loc_sym_axis_2, np.cross(loc_sym_axis_1, loc_sym_axis_2)/np.linalg.norm(np.cross(loc_sym_axis_1, loc_sym_axis_2))])
            rotation = np.dot(points_translated.T, np.linalg.inv(points_original.T))
            axis_mat = []
            axises = []
            for axis_i in axis_list:
                for axis_ii in axis_i['axis_l']:
                    div_ = axis_i['num']+1
                    dis_ang = 360 / div_
                    for div_i in range(div_ - 1):
                        c_ang = (div_i+1) * dis_ang
                        r_m = rotate_translate(np.dot(rotation, np.array(axis_ii)), c_ang, center_ch)
                        axis_mat.append(r_m)
                    axises.append(np.dot(rotation, np.array(axis_ii)))
            axis_info_in_obj_pt_model['axis'] = axises
            model_i_['axis'] = axis_info_in_obj_pt_model['axis']
            model_i_['center_ch'] = center_ch
            axis_info_in_obj_pt_model['axis_mat'] = axis_mat
            axis_info_in_obj_pt_model['details_sym_axis_1'] = axis_1_info
            axis_info_in_obj_pt_model['details_sym_axis_2'] = axis_2_info
        symmetries_discrete = []
        sym_draw_list = []
        sym_draw_list.append(np.eye(4))
        for mat_1 in axis_info_in_obj_pt_model['axis_mat']:
            symmetries_discrete.append(mat_1.reshape((16)))
            sym_draw_list.append(mat_1)
        symmetries_discrete = np.array(symmetries_discrete)
        status_aware = 0
        for i in range(1000):
            r_axis = np.random.uniform(0,1,3)
            axis_s_ = []
            for mat_1 in sym_draw_list:
                vr_axis = np.dot(mat_1[:3, :3], r_axis) + mat_1[:3, 3]
                axis_s_.append(vr_axis)
            axis_s_ = np.array(axis_s_)
            distances = spatial.distance.cdist(axis_s_, axis_s_, metric='euclidean').reshape((-1))
            distances_0 = distances[distances == 0]
            if distances_0.shape[0] == axis_s_.shape[0]:
                status_aware = 1
                break
        if status_aware == 0: raise ValueError("status aware")
        if sym_aware:
            rand_Color = np.random.randint(0, 255, (len(sym_draw_list), 3)).astype('int')
            vertices_ = model_i_['vertices']
            vertices_sym = []
            for mat_1 in sym_draw_list:
                vertices_sym.append(np.dot(mat_1[:3,:3], vertices_.T).T + mat_1[:3,3] - r_axis)
            vertices_sym = np.array(vertices_sym)
            vertices_sym = np.linalg.norm(vertices_sym, axis = 2)
            vertices_sym = np.argmin(vertices_sym, axis=0)
            colors_ = np.zeros((model_i_['colors'].shape[0], 4))
            for i, rand_Color_i in enumerate(rand_Color):
                colors_[vertices_sym == i, :3] = rand_Color_i
            colors_[:,3] = 255
            model_i_['sym_colors'] = colors_.astype(np.float32)/255
        model_i_["min_x"] = np.min(model_i_['vertices'][:, 0])
        model_i_["min_y"] = np.min(model_i_['vertices'][:, 1])
        model_i_["min_z"] = np.min(model_i_['vertices'][:, 2])
        model_i_["size_x"] = np.max(model_i_['vertices'][:, 0]) - np.min(model_i_['vertices'][:, 0])
        model_i_["size_y"] = np.max(model_i_['vertices'][:, 1]) - np.min(model_i_['vertices'][:, 1])
        model_i_["size_z"] = np.max(model_i_['vertices'][:, 2]) - np.min(model_i_['vertices'][:, 2])
        model_i_[sym_op] = symmetries_discrete.tolist()
        
    return model_i_
