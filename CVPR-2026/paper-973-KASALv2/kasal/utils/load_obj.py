# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

import os
import numpy as np

class OBJ:
    """ Load an OBJ object model. """
    
    @classmethod
    def loadMaterial(cls, filename):
        contents = {}
        mtl = None
        dirname = os.path.dirname(filename)

        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'newmtl':
                mtl = contents[values[1]] = {}
            elif mtl is None:
                raise ValueError("mtl file doesn't start with newmtl stmt")
            elif values[0] == 'map_Kd':
                mtl[values[0]] = values[1]
                imagefile = os.path.join(dirname, mtl['map_Kd'])
            elif values[0] == 'map_d':
                mtl
            else:
                mtl[values[0]] = list(map(float, values[1:]))
        return contents

    def __init__(self, filename, swapyz=False):
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        self.faces_material = {}
        self.gl_list = 0
        dirname = os.path.dirname(filename)

        material = None
        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == 'vn':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                self.texcoords.append(list(map(float, values[1:3])))
            elif values[0] in ('usemtl', 'usemat'):
                material = values[1]
            elif values[0] == 'mtllib':
                self.mtl = self.loadMaterial(os.path.join(dirname, values[1]))
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                self.faces.append((face, norms, texcoords, material))
        
        self.vertices = np.array(self.vertices)
        self.texcoords = np.array(self.texcoords)
        
        for f_i in self.faces:
            if 'map_Kd' in self.mtl[f_i[3]]:
                material_i = self.mtl[f_i[3]]['map_Kd']
                if material_i not in self.faces_material:
                    self.faces_material[material_i] = {
                        'uv' : [],
                        'faces' : [],
                    }
                self.faces_material[material_i]['uv'].append(self.texcoords[f_i[2][0]-1])
                self.faces_material[material_i]['uv'].append(self.texcoords[f_i[2][1]-1])
                self.faces_material[material_i]['uv'].append(self.texcoords[f_i[2][2]-1])
                self.faces_material[material_i]['faces'].append(np.array([f_i[0][0]-1, f_i[0][1]-1, f_i[0][2]-1]))
            else:
                if f_i[3] not in self.faces_material:
                    mtl_0 = self.mtl[f_i[3]]
                    self.faces_material[f_i[3]] = {
                        'uv' : [],
                        'faces' : [],
                        'mtl_0' : mtl_0,
                    }
                self.faces_material[f_i[3]]['uv'].append(self.texcoords[f_i[2][0]-1])
                self.faces_material[f_i[3]]['uv'].append(self.texcoords[f_i[2][1]-1])
                self.faces_material[f_i[3]]['uv'].append(self.texcoords[f_i[2][2]-1])
                self.faces_material[f_i[3]]['faces'].append(np.array([f_i[0][0]-1, f_i[0][1]-1, f_i[0][2]-1]))
