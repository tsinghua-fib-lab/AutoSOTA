# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

import os, json
import kasal.config.config as config

def load_json2dict(path):
    
    with open(path, 'r') as f:
        dict_ = json.load(f)
    f.close()
    return dict_

def write_dict2json(path, dict):
    
    with open(path, 'w') as f:
        json.dump(dict, f,indent=4)
    f.close()
    return

def get_all_ply_obj(dir):
    """ Load all files with the extensions .ply and .obj, excluding files with the suffix _sym.ply."""
    
    Filelist = []
    for home, dirs, files in os.walk(dir):
        for filename in files:
            if filename.endswith('.ply') or filename.endswith('.obj'):
                if not filename.endswith('_sym.ply'):
                    Filelist.append(os.path.join(home, filename))
    return Filelist

def save_symmetry_type():
    """ Save the rotational symmetry information of the current object."""
    
    if config.ui_int > config.ui_int_upper:
        ui_int_c = config.ui_int_upper
    else:
        ui_int_c = config.ui_int
    if config.ui_int <  2:
        ui_int_c = 2
    else:
        ui_int_c = config.ui_int
    if config.ui_options_selected != 'None':
        symmetry_type_dict = {
            'sym_type' : config.ui_options_selected,
            'n-fold' : ui_int_c,
            'ADI-C' : config.is_true2,
            'current_obj_info' : config.current_obj_info,
        }
        if config.ui_xyz_options_selected != 'None':
            symmetry_type_dict['axis_xyz'] = config.ui_xyz_options_selected
        f_ = config.files_name_list[config.current_file_id]
        sym_type_file = os.path.join(os.path.dirname(f_), os.path.basename(f_).split('.')[0]+'_sym_type.json')
        write_dict2json(sym_type_file, symmetry_type_dict)
    else:
        try:
            f_ = config.files_name_list[config.current_file_id]
            sym_type_file = os.path.join(os.path.dirname(f_), os.path.basename(f_).split('.')[0]+'_sym_type.json')
            os.remove(sym_type_file)
        except:
            1
        try:
            f_ = config.files_name_list[config.current_file_id]
            sym_type_file = os.path.join(os.path.dirname(f_), os.path.basename(f_).split('.')[0]+'_sym.ply')
            os.remove(sym_type_file)
        except:
            1
    return

def load_symmetry_type():
    """ Load the rotational symmetry information of the current object."""
    
    print('id / N : %s / %s'%(str(config.current_file_id), str(len(config.files_name_list))))
    
    write_dict2json(config.start_id_json_file, 
                    {
                        'start_id' : config.current_file_id,
                    }
                    )
    f_ = config.files_name_list[config.current_file_id]
    print(f_)
    sym_type_file = os.path.join(os.path.dirname(f_), os.path.basename(f_).split('.')[0]+'_sym_type.json')
    if os.path.exists(sym_type_file):
        symmetry_type_dict = load_json2dict(sym_type_file)
        config.ui_options_selected = symmetry_type_dict['sym_type']
        config.ui_int = symmetry_type_dict['n-fold']
        config.is_true2 = symmetry_type_dict['ADI-C']
        config.current_obj_info = symmetry_type_dict['current_obj_info']
        if 'axis_xyz' in symmetry_type_dict:
            config.ui_xyz_options_selected = symmetry_type_dict['axis_xyz']
        else:
            config.ui_xyz_options_selected = 'None'
    return
