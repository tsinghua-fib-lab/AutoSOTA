# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

import chardet
import numpy as np
from scipy import spatial
from kasal.keyaxis.rotate import rotate_translate


def stp2info(step_path):
    """ Load an STP object model. """
    
    def get_info(start, lines):
        res = {}
        tmp = lines[start]
        cut_position = tmp.find("(")
        k, v = tmp[:cut_position], tmp[cut_position+1:-2].split(",")
        for i, vv in enumerate(v):
            v[i] = vv.replace('(','').replace(')','').strip()
            if v[i].startswith("#"):
                v[i] = get_info(v[i], lines)
        res[k] = v
        return res
    with open(step_path,'rb') as f:
        text = f.read()
    file = open(step_path, 'rb')
    file_encoding = chardet.detect(text)['encoding']
    file = open(step_path, encoding=file_encoding)
    s = file.read()
    file.close()
    print('stp: ', file_encoding)
    s = s.replace('\r','').replace('ISO-10303-21\n','').replace('"END-ISO-10303-21\n"','')
    header = s[s.index('HEADER') + len('HEADER') + 2:s.index('ENDSEC')]
    data = s[s.index('DATA') + len('DATA') + 2:]
    step_root = {}
    header_root = {}
    data_root = {}
    data_lines = {}
    header_list = header.split('\n')
    for line in header_list:
        cut_position1 = line.find("(")
        cut_position2 = line.find(";")
        key = line[0:cut_position1]
        value = line[cut_position1:cut_position2]
        header_root[key] = value
    step_root["header"] = header_root
    data_list = data.split('\n')
    for line in data_list:
        cut_position = line.find("=")
        if cut_position != -1:
            key = line[0:cut_position].strip()
            value = line[cut_position + 2:].strip()
            data_lines[key] = value.strip()
    for key in data_lines.keys():
        line = data_lines[key]
        cut_position1 = line.find("(")
        cut_position2 = line.find(";") - 1
        list = []
        if line.find('#') == -1:continue
        key1 = line[0:cut_position1].strip()
        if key1 != 'EDGE_LOOP' : continue
        key = key + ' ' + key1
        temp = line[cut_position1 + 1:cut_position2]
        list = temp.split(',')
        list1 = temp.split(',')
        for i in range(len(list)):
            cur = list[i].replace('(','').replace(')','').strip()
            if cur[0] == "#":
                res = get_info(cur.strip(), data_lines)
                list1[i] = res
        data_root[key] = list1
    step_root["data"] = data_root
    
    face_line_point_list = []
    for face_key in data_root:
        line_point_list = []
        face_ = data_root[face_key]
        for lines_ in face_:
            if isinstance(lines_, dict):
                for lines_key_ in lines_:
                    lines_info = lines_[lines_key_]
                    for line_ in lines_info:
                        if isinstance(line_, dict):
                            point_list = []
                            for line_key in line_:
                                line_info = line_[line_key]
                                for e_key in line_info:
                                    if isinstance(e_key, dict):
                                        for e_key_key in e_key:
                                            if 'VERTEX_POINT' in e_key_key:
                                                e_info_ = e_key[e_key_key]
                                                for e_e_ in e_info_:
                                                    if isinstance(e_e_, dict):
                                                        for e_e_key in e_e_:
                                                            e_e_info = e_e_[e_e_key]
                                                            x_ = float(e_e_info[1])
                                                            y_ = float(e_e_info[2])
                                                            z_ = float(e_e_info[3])
                                                            point_list.append([x_,y_,z_])
                            line_point_list.append(point_list)
        face_line_point_list.append(line_point_list)
    return face_line_point_list

def axis_num(face_line_point_list):
    """ Compute rotational symmetry information from an STP file. """
    
    def center_of_line(lines):
        l1, l2 = lines
        return [(l1[0] + l2[0])/2, (l1[1] + l2[1])/2, (l1[2] + l2[2])/2, ]
    def distance_of_line(lines):
        return np.linalg.norm(np.array(lines[0])-np.array(lines[1]))
    def exists_center_of_line(cen_list, cen, thre):
        for cen_i in cen_list:
            if distance_of_line([cen, cen_i]) < thre:
                return 1
        return 0
    def rot_axis_max_error(ply_pts_, mat):
        ply_pts_m = np.dot(ply_pts_, mat[:3,:3])+mat[:3,3]
        nn_index = spatial.cKDTree(ply_pts_)
        d1_, t1_index = nn_index.query(ply_pts_m, k=1)
        return np.max(d1_)

    thre_ = 1e+3
    
    lines_all = []
    axis_lines = []
    for face_ in face_line_point_list:
        for lines in face_:
            c_lines = center_of_line(lines)
            dis_ = distance_of_line(lines)
            if exists_center_of_line(axis_lines, c_lines, dis_/thre_) == 0:
                lines_all.append(lines)
                axis_lines.append(c_lines)
    axis_lines = np.array(axis_lines)
    num_lines = np.ones((axis_lines.shape[0]))
    
    dis_mean_of_lines = []
    for lines in lines_all:
        dis_mean_of_lines.append(distance_of_line(lines))
    dis_mean_of_lines = np.mean(dis_mean_of_lines)
    
    points_all = []
    for face_ in face_line_point_list:
        for lines in face_:
            for points in lines:
                if exists_center_of_line(points_all, points, dis_mean_of_lines/thre_) == 0:
                    points_all.append(points)
    points_all = np.array(points_all)
    axis_points = points_all
    num_points = []
    for point in points_all:
        num_ = 0
        for line in lines_all:
            if exists_center_of_line(line, point, dis_mean_of_lines/thre_) == 1:
                num_ += 1
        num_points.append(num_ - 1)
    num_points = np.array(num_points)
    
    axis_faces = []
    num_faces = []
    for face_ in face_line_point_list:
        num_faces.append(len(face_) - 1)
        points_l = []
        for lines in face_:
            for points in lines:
                points_l.append(points)
        axis_faces.append(np.mean(np.array(points_l), axis = 0))
    axis_faces = np.array(axis_faces)
    num_faces = np.array(num_faces)
    
    center = np.mean(points_all, axis = 0)
    
    axis_points = axis_points - center
    norm_axis_points = axis_points / np.linalg.norm(axis_points, axis=1, keepdims=True)
    num_points = np.array(num_points)
      
    axis_lines = axis_lines - center
    norm_axis_lines = axis_lines / np.linalg.norm(axis_lines, axis=1, keepdims=True)
    num_lines = np.array(num_lines)
    
    axis_faces = axis_faces - center
    norm_axis_faces = axis_faces / np.linalg.norm(axis_faces, axis=1, keepdims=True)
    num_faces = np.array(num_faces)
    
    axis_all = np.concatenate([axis_points, axis_lines, axis_faces], axis = 0)
    norm_axis_all = np.concatenate([norm_axis_points, norm_axis_lines, norm_axis_faces], axis = 0)
    num_all = np.concatenate([num_points, num_lines, num_faces], axis = 0).astype('int')
    
    norm_axis_f_all = []
    num_f_all = []
    for (norm_axis_i, num_i) in zip(norm_axis_all, num_all):
        num_ture = 0
        dis_ang = 360 / (num_i + 1)
        for i in range(num_i):
            r_m = rotate_translate(norm_axis_i, dis_ang*(i+1), np.zeros((3)))
            dis_max = rot_axis_max_error(axis_all, r_m)
            if dis_max < dis_mean_of_lines/thre_:
                num_ture += 1
        
        if exists_center_of_line(norm_axis_f_all, norm_axis_i, 1/1000) + exists_center_of_line(norm_axis_f_all, -norm_axis_i, 1/1000) == 0:
            if num_ture > 0:
                norm_axis_f_all.append(norm_axis_i)
                num_f_all.append(num_ture)  
        else:
            for j in range(len(norm_axis_f_all)):
                if distance_of_line([norm_axis_f_all[j], norm_axis_i]) < 1/1000 or distance_of_line([norm_axis_f_all[j], -norm_axis_i]) < 1/1000:
                    if num_f_all[j] < num_ture:
                        num_f_all[j] = num_ture
            
    return np.array(norm_axis_f_all), np.array(num_f_all)
