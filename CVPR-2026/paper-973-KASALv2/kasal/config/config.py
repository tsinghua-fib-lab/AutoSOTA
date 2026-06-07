# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

start_id_json_file = ''
# Path to KASAL.json  

is_true2 = False
# ADI-C activation status  

is_true3 = False
# Display status of the XYZ axes  

ui_int = 2
# Value of n in n-fold symmetry  

ui_options = [
    "None",
    "C(>>1): Spherical Item",
    "C(>1): Cylindrical Item", 
    "C(=1): Circular Item", 
    "D(>1): n-fold Prismatic Item", 
    "D(=1): n-fold Pyramidal Item", 
    "P(4): Tetrahedral Item",
    "P(8): Octahedral Item",
    "P(20): Icosahedral Item",
]
# Options for the 8 types of rotational symmetry.  
# 'C' represents continuous rotational symmetry,  
# while 'D' represents discrete rotational symmetry.  

ui_options_selected = ui_options[0]
# The rotational symmetry type of the object is set to None by default.  

ui_xyz_options = [
    "None",
    "axis X (red)",
    "axis Y (green)",
    "axis Z (blue)",
]
# Options for x, y, and z axes.  
# For a few objects, KASAL may fail to accurately locate the symmetry axis.  
# For these objects, you can simply set one of the x, y, or z axes,  
# and KASAL will locate a symmetry axis close to the specified axis.  

ui_xyz_options_selected = ui_xyz_options[0]

files_name_list = None
# Paths to all object models in the folder.  

current_file_id = 0
# ID of the current object.  

psm_list = []
# List of objects displayed in polyscope.  

uv_texture_size = 500
# Texture image size of the object.  

ui_int_upper = 90
# Maximum value of n in the set of n-fold symmetry axes.  

sample_num = 5250 * 1 + 1
# Number of Fibonacci sphere sampling points.  

current_obj_info = {}
# Information of the current object.  

arrow_ratio = 1.0
# Scaling factor of the saved symmetry axes.  

save_2_fold_a = True

close_ADI_c = False
# Disable ADI-C.  