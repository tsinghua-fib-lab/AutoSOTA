# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
try:
    import win32gui, win32con, win32api
    ICON_SHOW = True
except ImportError:
    ICON_SHOW = False
import os, cv2
import numpy as np
import kasal.config.config as config
# from config.config import is_true2, is_true3, ui_int, ui_options, ui_options_selected, \
#     ui_xyz_options, ui_xyz_options_selected, files_name_list, current_file_id, \
#         psm_list, uv_texture_size, ui_int_upper, targetfacenum, sample_num, current_obj_info, \
#             arrow_ratio, save_2_fold_a, close_ADI_c, start_id_json_file
from kasal.utils.io_json import load_json2dict, save_symmetry_type, load_symmetry_type, get_all_ply_obj
from kasal.utils.load_obj import OBJ
from kasal.utils.io_ply import save_ply_model, load_ply_model
from kasal.symmetry_lab.symmetry_axis_template import get_sym_axis_temp
from kasal.symmetry_lab.symmetry_axis_localization import cal_model_sym
from kasal.datasets.datasets_path import arrow_xyz_path, icon_path

import pymeshlab as ml
import polyscope
import polyscope.imgui as psim

polyscope.set_verbosity(0)
polyscope.set_max_fps(33)
polyscope.set_program_name("Key-Axis-based Symmetry Axis Localization")
mesh = ml.MeshSet()


def set_window_icon():
    ''' Load the application's icon. '''
    
    hwnd = win32gui.FindWindow(None, "Key-Axis-based Symmetry Axis Localization")
    if hwnd == 0:
        print(f"'{'Key-Axis-based Symmetry Axis Localization'}' was not found !")
        return
    ico_x = win32api.GetSystemMetrics(win32con.SM_CXSMICON) * 50
    ico_y = win32api.GetSystemMetrics(win32con.SM_CYSMICON) * 50
    icon = win32gui.LoadImage(0, icon_path, win32con.IMAGE_ICON, ico_x, ico_y, win32con.LR_LOADFROMFILE)
    win32gui.SendMessage(hwnd, win32con.WM_SETICON, win32con.ICON_BIG, icon)

def build_psm(m, mesh_id, input_file, transparency):
    ''' Convert a pymeshlab variable to a polyscope variable.  
    Parameters:  
        m: The object model variable in pymeshlab.  
        mesh_id: The object ID in polyscope (each object in polyscope must have a unique ID).  
        input_file: The path to the object model.  
        transparency: The transparency of the object in polyscope.  
    Returns:  
        psm: The object model variable in polyscope.  
    '''
    
    is_enabled = m.is_visible()
    if m.is_point_cloud():
        psm = polyscope.register_point_cloud(str(mesh_id), m.transformed_vertex_matrix(), enabled=is_enabled, transparency=transparency)
    else:
        psm = polyscope.register_surface_mesh(str(mesh_id), m.transformed_vertex_matrix(), m.face_matrix(), enabled=is_enabled, transparency=transparency)
    if m.has_wedge_tex_coord():
        key_ = list(m.textures().keys())
        polyscope.remove_all_structures()
        obj_ = OBJ(input_file)
        info_ = obj_.faces_material
        ver_uv_face_list = []
        image_list = []
        transparency_list = []
        for info_key_ in info_:
            ver_uv_face = info_[info_key_]
            uv_path_ = os.path.join(os.path.dirname(input_file), info_key_)
            if os.path.exists(uv_path_):
                try:
                    image_ = cv2.cvtColor(cv2.imread(uv_path_), cv2.COLOR_RGB2BGR)
                    transparency_list.append(1)
                except:
                    image_ = np.ones((16,16,3),dtype=np.uint8) * 255
                    image_[:, :, :] = 255
                    transparency_list.append(1.0)
            else:
                image_ = np.ones((16,16,3),dtype=np.uint8) * 255
                image_[:, :, :] = 255 
                transparency_list.append(1.0)
            ver_uv_face_list.append(ver_uv_face)
            image_list.append(image_)
        psm = polyscope.create_group(str(mesh_id))
        psm_i_list = []
        for i_ in range(len(image_list)):
            new_image = image_list[i_]
            ver_uv_face = ver_uv_face_list[i_]
            uv_i_ = np.array(ver_uv_face['uv'])
            uv_i_min = np.min(uv_i_, axis=0)
            uv_i_max = np.max(uv_i_, axis=0)
            uv_i_min_0 = np.array(uv_i_min).astype(np.int64) - 1
            uv_i_max_0 = np.array(uv_i_max).astype(np.int64) + 1
            d_x_ = uv_i_max_0[0] - uv_i_min_0[0]
            d_y_ = uv_i_max_0[1] - uv_i_min_0[1]
            uv_i_[:, 0] = (uv_i_[:, 0] - uv_i_min_0[0]) / d_x_
            uv_i_[:, 1] = (uv_i_[:, 1] - uv_i_min_0[1]) / d_y_
            static_size = 256 * 256
            h_new_image, w_new_image,  = new_image.shape[:2]
            new_image_size = w_new_image * h_new_image
            resize_ratio_ = np.sqrt(static_size / (new_image_size * d_x_ * d_y_))
            if resize_ratio_ > 1: resize_ratio_ = 1
            w_new_image = int(w_new_image*resize_ratio_)
            if w_new_image == 0: 
                w_new_image = 1
                h_new_image = 1
            h_new_image = int(h_new_image*resize_ratio_)
            if h_new_image == 0: 
                w_new_image = 1
                h_new_image = 1
            new_image = cv2.resize(new_image, (w_new_image, h_new_image))
            hh_, ww_ = new_image.shape[0], new_image.shape[1]
            if np.max([hh_, ww_]) == 1:
                uv_i_min = np.min(uv_i_, axis=0)
                uv_i_max = np.max(uv_i_, axis=0)
                uv_i_min_0 = np.array(uv_i_min).astype(np.int64) - 1
                uv_i_max_0 = np.array(uv_i_max).astype(np.int64) + 1
                d_x_ = uv_i_max_0[0] - uv_i_min_0[0]
                d_y_ = uv_i_max_0[1] - uv_i_min_0[1]
                uv_i_[:, 0] = (uv_i_[:, 0] - uv_i_min_0[0]) / d_x_
                uv_i_[:, 1] = (uv_i_[:, 1] - uv_i_min_0[1]) / d_y_
                new_image_big = new_image
            else:
                x_img_l = []
                for jj_ in range(d_x_):
                    x_img_l.append(new_image)
                x_img_l = np.hstack(x_img_l)
                y_img_l = []
                for ii_ in range(d_y_):
                    y_img_l.append(x_img_l)
                new_image_big = np.vstack(y_img_l)
            psm_i = polyscope.register_surface_mesh(str(mesh_id)+'_'+str(i_), 
                                                    m.transformed_vertex_matrix(),
                                                    np.array(ver_uv_face['faces'], dtype=np.int64), 
                                                    enabled=is_enabled, transparency=transparency)
            psm_i.add_parameterization_quantity("vertex_uv_coords", np.array(uv_i_), coords_type='unit',
                                    defined_on='corners', enabled=True)
            new_image_big = np.array(new_image_big, dtype=np.float32) / 255
            psm_i.add_color_quantity("vertex_texture", new_image_big, 
                        defined_on='texture', param_name="vertex_uv_coords", enabled=True)
            psm_i.add_to_group(psm)
            psm_i_list.append(psm_i)
        psm.set_enabled(True)
        psm.set_hide_descendants_from_structure_lists(True)
        psm.set_show_child_details(False)
    elif m.has_vertex_tex_coord():
        v_uv = m.vertex_tex_coord_matrix()
        psm.add_parameterization_quantity("vertex_uv_coords", v_uv, defined_on='vertices', enabled=True)
        key_ = list(m.textures().keys())
        if len(key_) > 1:
            raise ValueError(' The number of textures is not equal to 1! ')
        v_tex = cv2.cvtColor(cv2.imread(os.path.join(os.path.dirname(input_file), key_[0])), cv2.COLOR_RGB2BGR)
        v_tex = cv2.resize(v_tex, (config.uv_texture_size, config.uv_texture_size))
        v_tex = np.array(v_tex, dtype=np.float32) / 255
        psm.add_color_quantity("vertex_texture", v_tex, defined_on='texture', param_name="vertex_uv_coords", enabled=True)
    elif m.has_vertex_scalar():
        psm.add_scalar_quantity('vertex_scalar', m.vertex_scalar_array(),enabled=True)
    elif m.has_vertex_color():
        vc = m.vertex_color_matrix()
        vc = np.delete(vc, 3, 1)
        psm.add_color_quantity('vertex_color', vc, enabled=True)
    elif not m.is_point_cloud() and m.has_face_color() and not m.has_wedge_tex_coord():
        fc = m.face_color_matrix()
        fc = np.delete(fc, 3, 1)
        psm.add_color_quantity('face_color', fc, defined_on='faces',enabled=True)
    elif not m.is_point_cloud() and m.has_face_scalar():
        psm.add_scalar_quantity('face_scalar', m.face_scalar_array(), defined_on='faces',enabled=True)
    return psm   

def last_object_func():
    ''' Load the previous object. '''
    
    mesh.clear()
    save_symmetry_type()
    if config.current_file_id - 1 >= 0:
        config.current_file_id = config.current_file_id - 1
        input_file = config.files_name_list[config.current_file_id]
        mesh.load_new_mesh(input_file)
        mesh_id = mesh.current_mesh_id()
        mesh_id_list = [mesh_id]
        transparency_list = [1.0]
        if config.is_true3:
            mesh_c = mesh.current_mesh()
            bbox = mesh_c.bounding_box()
            center_ = bbox.center()
            dim_0 = np.max([bbox.dim_x(), bbox.dim_y(), bbox.dim_z()]) * 1.5
            mesh.load_new_mesh(arrow_xyz_path)
            mesh_c = mesh.current_mesh()
            bbox = mesh_c.bounding_box()
            dim_1 = np.max([bbox.dim_x(), bbox.dim_y(), bbox.dim_z()])
            mesh.compute_matrix_from_translation_rotation_scale(scalex = dim_0 / dim_1,
                                                                scaley = dim_0 / dim_1,
                                                                scalez = dim_0 / dim_1,
                                                                )
            mesh.compute_matrix_from_translation_rotation_scale(translationx = center_[0],
                                                                translationy = center_[1],
                                                                translationz = center_[2],
                                                                )
            mesh_id = mesh.current_mesh_id()
            mesh_id_list.append(mesh_id)
            transparency_list.append(0.3)
        input_sym_file = os.path.join(os.path.dirname(input_file), os.path.basename(input_file).split('.')[0]+'_sym.ply')
        if os.path.exists(input_sym_file):
            mesh.load_new_mesh(input_sym_file)
            mesh_id = mesh.current_mesh_id()
            mesh_id_list.append(mesh_id)
            transparency_list.append(1.0)
        polyscope.remove_all_groups()
        polyscope.remove_all_structures()
        config.psm_list = []
        for (m, mesh_id, transparency) in zip(mesh, mesh_id_list, transparency_list):
            psm = build_psm(m, mesh_id, input_file, transparency)
            config.psm_list.append(psm)
    config.ui_options_selected = config.ui_options[0]    
    config.ui_xyz_options_selected = config.ui_xyz_options[0]   
    config.is_true2 = False
    load_symmetry_type()

def next_object_func():
    ''' Load the next object. '''
    
    mesh.clear()
    save_symmetry_type()
    obj_number = len(config.files_name_list)
    if config.current_file_id + 1 < obj_number:
        config.current_file_id = config.current_file_id + 1
        input_file = config.files_name_list[config.current_file_id]
        mesh.load_new_mesh(input_file)
        mesh_id = mesh.current_mesh_id()
        mesh_id_list = [mesh_id]
        transparency_list = [1.0]
        if config.is_true3:
            mesh_c = mesh.current_mesh()
            bbox = mesh_c.bounding_box()
            center_ = bbox.center()
            dim_0 = np.max([bbox.dim_x(), bbox.dim_y(), bbox.dim_z()]) * 1.5
            mesh.load_new_mesh(arrow_xyz_path)
            mesh_c = mesh.current_mesh()
            bbox = mesh_c.bounding_box()
            dim_1 = np.max([bbox.dim_x(), bbox.dim_y(), bbox.dim_z()])
            mesh.compute_matrix_from_translation_rotation_scale(scalex = dim_0 / dim_1,
                                                                scaley = dim_0 / dim_1,
                                                                scalez = dim_0 / dim_1,
                                                                )
            mesh.compute_matrix_from_translation_rotation_scale(translationx = center_[0],
                                                                translationy = center_[1],
                                                                translationz = center_[2],
                                                                )
            mesh_id = mesh.current_mesh_id()
            mesh_id_list.append(mesh_id)
            transparency_list.append(0.3)
        input_sym_file = os.path.join(os.path.dirname(input_file), os.path.basename(input_file).split('.')[0]+'_sym.ply')
        if os.path.exists(input_sym_file):
            mesh.load_new_mesh(input_sym_file)
            mesh_id = mesh.current_mesh_id()
            mesh_id_list.append(mesh_id)
            transparency_list.append(1.0)
        polyscope.remove_all_groups()
        polyscope.remove_all_structures()
        config.psm_list = []
        for (m, mesh_id, transparency) in zip(mesh, mesh_id_list, transparency_list):
            psm = build_psm(m, mesh_id, input_file, transparency)
            config.psm_list.append(psm)
    config.ui_options_selected = config.ui_options[0]
    config.ui_xyz_options_selected = config.ui_xyz_options[0]   
    config.is_true2 = False
    load_symmetry_type()

def reload_object_func():
    ''' Reload the object. When the object's symmetry axis information changes  
        or the XYZ axis display is enabled, the object displayed in polyscope  
        will be updated accordingly.  
    '''
    
    mesh.clear()
    load_symmetry_type()
    input_file = config.files_name_list[config.current_file_id]
    mesh.load_new_mesh(input_file)
    mesh_id = mesh.current_mesh_id()
    mesh_id_list = [mesh_id]
    transparency_list = [1.0]
    if config.is_true3:
        mesh_c = mesh.current_mesh()
        bbox = mesh_c.bounding_box()
        center_ = bbox.center()
        dim_0 = np.max([bbox.dim_x(), bbox.dim_y(), bbox.dim_z()]) * 1.5
        mesh.load_new_mesh(arrow_xyz_path)
        mesh_c = mesh.current_mesh()
        bbox = mesh_c.bounding_box()
        dim_1 = np.max([bbox.dim_x(), bbox.dim_y(), bbox.dim_z()])
        mesh.compute_matrix_from_translation_rotation_scale(scalex = dim_0 / dim_1,
                                                            scaley = dim_0 / dim_1,
                                                            scalez = dim_0 / dim_1,
                                                            )
        mesh.compute_matrix_from_translation_rotation_scale(translationx = center_[0],
                                                            translationy = center_[1],
                                                            translationz = center_[2],
                                                            )
        mesh_id = mesh.current_mesh_id()
        mesh_id_list.append(mesh_id)
        transparency_list.append(0.3)
    input_sym_file = os.path.join(os.path.dirname(input_file), os.path.basename(input_file).split('.')[0]+'_sym.ply')
    if os.path.exists(input_sym_file):
        mesh.load_new_mesh(input_sym_file)
        mesh_id = mesh.current_mesh_id()
        mesh_id_list.append(mesh_id)
        transparency_list.append(1.0)
    polyscope.remove_all_groups()
    polyscope.remove_all_structures()
    config.psm_list = []
    for (m, mesh_id, transparency) in zip(mesh, mesh_id_list, transparency_list):
        psm = build_psm(m, mesh_id, input_file, transparency)
        config.psm_list.append(psm)

def cal_current_sym():
    ''' Locate the rotational symmetry axis of the current object. '''
    
    sym_info_ = get_sym_axis_temp(config.ui_options_selected, config.ui_int)
    input_f_ = config.files_name_list[config.current_file_id]
    sym_op = None
    if config.ui_options_selected == "C(>1): Cylindrical Item":
        sym_op = "symmetries_continuous_2"
    elif config.ui_options_selected == "C(=1): Circular Item":
        sym_op = 'symmetries_continuous'
    elif config.ui_options_selected in ["D(>1): n-fold Prismatic Item", 
                                "D(=1): n-fold Pyramidal Item", 
                                "P(4): Tetrahedral Item",
                                "P(8): Octahedral Item",
                                "P(20): Icosahedral Item",]:
        sym_op = 'symmetries_discrete'
    elif config.ui_options_selected == "C(>>1): Spherical Item":
        sym_op = 'symmetries_continuous_3'
    if config.close_ADI_c is True:
        adi_op = 'pts'
    else:
        if config.is_true2:
            adi_op = 'colors'
        else:
            adi_op = 'pts'
    color_op = False
    if adi_op == 'colors':
        color_op = True
    model_i_ = load_ply_model(input_f_, color_op = color_op)
    model_i_ = cal_model_sym(model_i_, step_path = sym_info_, sym_op=sym_op, sym_aware=True, op = adi_op, sample_num=config.sample_num, icp_op=True, xyz_op = config.ui_xyz_options_selected)
    output_f_ = os.path.join(os.path.dirname(input_f_), os.path.basename(input_f_).split('.')[0]+'_sym.ply')
    model_info_i = save_ply_model(model_i_, output_f_)
    config.current_obj_info = model_info_i
    save_symmetry_type()
    reload_object_func()

def cal_all_obj_sym():
    ''' Locate the rotational symmetry axes of all objects. '''
    
    current_file_id_copy = int(config.current_file_id)
    for start_id, f_ in enumerate(config.files_name_list):
        sym_type_file = os.path.join(os.path.dirname(f_), os.path.basename(f_).split('.')[0]+'_sym_type.json')
        if os.path.exists(sym_type_file):
            run_polyscope(config.files_name_list, start_id=start_id)
            cal_current_sym()
    run_polyscope(config.files_name_list, start_id=current_file_id_copy)

def callback():
    ''' GUI interface of polyscope. '''
    
    psim.PushItemWidth(200)
    psim.TextUnformatted("Toolkit for Symmetry Axis Localization")
    psim.Separator()
    psim.PushItemWidth(200)
    psim.TextUnformatted("Symmetry Type:")
    psim.SameLine() 
    changed = psim.BeginCombo(" "*1, config.ui_options_selected)
    if changed:
        for val in config.ui_options:
            _, selected = psim.Selectable(val, config.ui_options_selected==val)
            if selected:
                config.ui_options_selected = val
        psim.EndCombo()
    psim.PopItemWidth()
    psim.Separator()
    psim.TextUnformatted("Setting of D(>1) and D(=1)")
    psim.TextUnformatted("Note: if n < 2, n = 2; if n > 90, n = 90 !")
    psim.Separator()
    psim.TextUnformatted("N (n-fold):")
    psim.SameLine() 
    changed, config.ui_int = psim.InputInt(" "*2, config.ui_int, step=1, step_fast=10) 
    psim.SameLine() 
    psim.TextUnformatted("ADI-C:")
    psim.SameLine() 
    changed, config.is_true2 = psim.Checkbox(" "*3, config.is_true2)
    if(changed):
        pass 
    psim.Separator()
    psim.TextUnformatted("axis xyz: ")
    psim.SameLine() 
    changed = psim.BeginCombo(" "*4, config.ui_xyz_options_selected)
    if changed:
        for val in config.ui_xyz_options:
            _, selected_2 = psim.Selectable(val, config.ui_xyz_options_selected==val)
            if selected_2:
                config.ui_xyz_options_selected = val
        psim.EndCombo()
    psim.PopItemWidth()
    psim.SameLine() 
    psim.TextUnformatted("show xyz: ")
    psim.SameLine() 
    changed, config.is_true3 = psim.Checkbox(" "*5, config.is_true3)
    if(changed): 
        reload_object_func()
        pass 
    psim.Separator()
    if(psim.Button("Last Object")):
        last_object_func()
    psim.SameLine() 
    if(psim.Button("Next Object")):
        next_object_func()
    psim.Separator()
    psim.TextUnformatted('|')
    psim.TextUnformatted('|')
    psim.TextUnformatted('|')
    psim.TextUnformatted('|')
    psim.Separator()
    if(psim.Button("Cal Current Obj")):
        cal_current_sym()
        print('Cal Current Obj')
    psim.SameLine() 
    psim.Separator()
    psim.TextUnformatted('|')
    psim.TextUnformatted('|')
    psim.TextUnformatted('|')
    psim.TextUnformatted('|')
    psim.Separator()
    if(psim.Button("Cal All Objs")):
        cal_all_obj_sym()
        print('Cal All Objs')
            
def run_polyscope(files_name_list, start_id = 0):
    ''' Use polyscope to display objects from pymeshlab.  
    Parameters:  
        files_name_list: A list of paths to all object models in the folder.  
        start_id: The index of the object to start loading.  
    '''
    
    mesh.clear()
    polyscope.remove_all_groups()
    polyscope.remove_all_structures()
    input_file = files_name_list[start_id]
    config.current_file_id = start_id
    load_symmetry_type()
    mesh.load_new_mesh(input_file)
    mesh_id = mesh.current_mesh_id()
    mesh_id_list = [mesh_id]
    transparency_list = [1.0]
    if config.is_true3:
        mesh_c = mesh.current_mesh()
        bbox = mesh_c.bounding_box()
        center_ = bbox.center()
        dim_0 = np.max([bbox.dim_x(), bbox.dim_y(), bbox.dim_z()]) * 1.5
        mesh.load_new_mesh(arrow_xyz_path)
        mesh_c = mesh.current_mesh()
        bbox = mesh_c.bounding_box()
        dim_1 = np.max([bbox.dim_x(), bbox.dim_y(), bbox.dim_z()])
        mesh.compute_matrix_from_translation_rotation_scale(scalex = dim_0 / dim_1,
                                                            scaley = dim_0 / dim_1,
                                                            scalez = dim_0 / dim_1,
                                                            )
        mesh.compute_matrix_from_translation_rotation_scale(translationx = center_[0],
                                                            translationy = center_[1],
                                                            translationz = center_[2],
                                                            )
        mesh_id = mesh.current_mesh_id()
        mesh_id_list.append(mesh_id)
        transparency_list.append(0.3)
    
    input_sym_file = os.path.join(os.path.dirname(input_file), os.path.basename(input_file).split('.')[0]+'_sym.ply')
    if os.path.exists(input_sym_file):
        mesh.load_new_mesh(input_sym_file)
        mesh_id = mesh.current_mesh_id()
        mesh_id_list.append(mesh_id)
        transparency_list.append(1.0)
    
    for (m, mesh_id, transparency_) in zip(mesh, mesh_id_list, transparency_list):
        psm = build_psm(m, mesh_id, input_file, transparency_)
        config.psm_list.append(psm)

def app(models_dir, start_id = -1):
    ''' Application of symmetry axis localization.  
    Parameters:  
        models_dir: The folder containing object models. The application scans for  
                    PLY and OBJ files in all subdirectories; other file types are  
                    currently not supported.  
        start_id: The index of the object to start loading (default: -1).  
                -1 means the initial loading follows the start_id recorded in KASAL.json.  
                If KASAL.json does not exist, start_id is set to 0.  
    '''
    
    config.start_id_json_file = os.path.join(models_dir, 'KASAL.json')
    if start_id == -1:
        if os.path.exists(config.start_id_json_file):
            try:
                start_id = load_json2dict(config.start_id_json_file)['start_id']
            except:
                start_id = 0
    config.files_name_list = get_all_ply_obj(models_dir)
    if start_id >= len(config.files_name_list):
        start_id = 0
    print(config.files_name_list[start_id])
    print('id:', start_id)
    polyscope.init()
    polyscope.set_user_callback(callback)
    if ICON_SHOW:
        set_window_icon()
    run_polyscope(config.files_name_list, start_id=start_id)
    polyscope.show()
