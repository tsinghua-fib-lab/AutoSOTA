import math
import numpy as np
import numpy.typing as npt
import os
import PIL.Image
import torch
from torch.utils.data import Dataset

from . import utils

class CameraInfo:
    def __init__(self):
        self.id:int=0
        self.model:str=''
        self.width:int=0
        self.height:int=0
        return
    
    def __init__(self,id:int,model_name:str,width:int,height:int):
        self.id:int=id
        self.model:str=model_name
        self.width:int=width
        self.height:int=height
        return
    
    def get_project_matrix(self)->npt.NDArray:
        return None
    def get_focal(self):
        return None
    
class PinHoleCameraInfo(CameraInfo):
    def __init__(self,id:int,width:int,height:int,parameters:list[float],z_near=0.01,z_far=100.0):
        super(PinHoleCameraInfo,self).__init__(id,"PINHOLE",width,height)
        focal_length_x=parameters[0]
        focal_length_y=parameters[1]
        focal_x=focal_length_x/(width*0.5)
        focal_y=focal_length_y/(height*0.5)
        self.proj_matrix=np.array([[focal_x,0,0,0],
                  [0,focal_y,0,0],
                  [0,0,(z_far+z_near)/(z_far-z_near),-2*z_far*z_near/(z_far-z_near)],
                  [0,0,1,0]],dtype=np.float32).transpose()
        self.inv_z_proj_matrix=np.array([[focal_x,0,0,0],
                  [0,focal_y,0,0],
                  # Inverse-z style projection: after perspective divide, ndc.z is in [0,1]
                  # with near plane close to 1 and far plane close to 0.
                  [0,0,-z_near/(z_far-z_near),z_far*z_near/(z_far-z_near)],
                  [0,0,1,0]],dtype=np.float32).transpose()
        return
    
    def get_project_matrix(self):
        return self.proj_matrix
    
    def get_inv_z_project_matrix(self):
        return self.inv_z_proj_matrix
    
WARNED = False

class CameraFrame:
    def __init__(self):
        self.id:int=0
        self.viewtransform_rotation:npt.NDArray=np.array((0,0,0,0))
        self.viewtransform_position:npt.NDArray=np.array((0,0,0))
        self.camera_id:int=0
        self.name:str=None
        self.img_source:str=None
        self.xys=np.array((0,0,0,0))
        return
    
    def __init__(self,id:int,qvec:npt.ArrayLike,tvec:npt.ArrayLike,camera_id:int,name:str,img_source:str,xys:npt.ArrayLike):
        self.id:int=id
        viewtransform_rotation:npt.NDArray=utils.qvec2rotmat(np.array(qvec))
        viewtransform_position:npt.NDArray=np.array(tvec)
        self.view_matrix = utils.get_view_matrix(viewtransform_rotation,viewtransform_position).transpose()
        self.camera_center = -viewtransform_rotation.transpose()@viewtransform_position
        self.camera_id:int=camera_id
        self.name:str=name
        self.img_source:str=img_source
        self.xys:npt.NDArray=np.array(xys)
        self.image={}
        return
    
    def load_image(self,downsample:int=-1):
        if self.image.get(downsample,None) is None:
            image:PIL.Image.Image=PIL.Image.open(self.img_source)

            orig_w, orig_h = image.size
            if downsample in [1, 2, 4, 8]:
                resolution = round(orig_w/ downsample), round(orig_h/ downsample)
            else:  # should be a type that converts to float
                if downsample == -1:
                    if orig_w > 1600:
                        global WARNED
                        if not WARNED:
                            print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                                "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                            WARNED = True
                        global_down = orig_w / 1600
                    else:
                        global_down = 1
                else:
                    global_down = orig_w / downsample

                scale = float(global_down)
                resolution = (int(orig_w / scale), int(orig_h / scale))  
            self.image[downsample]=np.array(image.resize(resolution),dtype=np.uint8).transpose(2,0,1)
        return self.image[downsample]
    
    def get_viewmatrix(self)->npt.NDArray:
        return self.view_matrix
    
    def get_camera_center(self)->npt.NDArray:
        return self.camera_center
    
class CameraFrameDataset(Dataset):
    def __get_frustumplane(self,view_matrix:npt.NDArray,proj_matrix:npt.NDArray)->npt.NDArray:
        viewproj_matrix=view_matrix@proj_matrix
        frustumplane=np.zeros((6,4),dtype=np.float32)
        #left plane
        frustumplane[0,0]=viewproj_matrix[0,3]+viewproj_matrix[0,0]
        frustumplane[0,1]=viewproj_matrix[1,3]+viewproj_matrix[1,0]
        frustumplane[0,2]=viewproj_matrix[2,3]+viewproj_matrix[2,0]
        frustumplane[0,3]=viewproj_matrix[3,3]+viewproj_matrix[3,0]
        #right plane
        frustumplane[1,0]=viewproj_matrix[0,3]-viewproj_matrix[0,0]
        frustumplane[1,1]=viewproj_matrix[1,3]-viewproj_matrix[1,0]
        frustumplane[1,2]=viewproj_matrix[2,3]-viewproj_matrix[2,0]
        frustumplane[1,3]=viewproj_matrix[3,3]-viewproj_matrix[3,0]

        #bottom plane
        frustumplane[2,0]=viewproj_matrix[0,3]+viewproj_matrix[0,1]
        frustumplane[2,1]=viewproj_matrix[1,3]+viewproj_matrix[1,1]
        frustumplane[2,2]=viewproj_matrix[2,3]+viewproj_matrix[2,1]
        frustumplane[2,3]=viewproj_matrix[3,3]+viewproj_matrix[3,1]

        #top plane
        frustumplane[3,0]=viewproj_matrix[0,3]-viewproj_matrix[0,1]
        frustumplane[3,1]=viewproj_matrix[1,3]-viewproj_matrix[1,1]
        frustumplane[3,2]=viewproj_matrix[2,3]-viewproj_matrix[2,1]
        frustumplane[3,3]=viewproj_matrix[3,3]-viewproj_matrix[3,1]

        #near plane
        frustumplane[4,0]=viewproj_matrix[0,2]
        frustumplane[4,1]=viewproj_matrix[1,2]
        frustumplane[4,2]=viewproj_matrix[2,2]
        frustumplane[4,3]=viewproj_matrix[3,2]

        #far plane
        frustumplane[5,0]=viewproj_matrix[0,3]-viewproj_matrix[0,2]
        frustumplane[5,1]=viewproj_matrix[1,3]-viewproj_matrix[1,2]
        frustumplane[5,2]=viewproj_matrix[2,3]-viewproj_matrix[2,2]
        frustumplane[5,3]=viewproj_matrix[3,3]-viewproj_matrix[3,2]
        return frustumplane
    
    def __init__(self,cameras:dict[int,PinHoleCameraInfo],frames:list[CameraFrame],downsample:int=-1,bDevice=True):
        self.cameras=cameras
        self.frames=frames
        self.downsample=downsample
        self.frustumplanes=[]
        if bDevice:
            for camera in cameras.values():
                camera.proj_matrix=torch.Tensor(camera.proj_matrix).cuda()
            for frame in frames:
                frame.view_matrix=torch.Tensor(frame.view_matrix).cuda()
                for key in frame.image.keys():
                    frame.image[key]=torch.tensor(frame.image[key]).cuda()
        for frame in self.frames:
            frustumplane=self.__get_frustumplane(frame.get_viewmatrix(),self.cameras[frame.camera_id].get_project_matrix())
            if bDevice:
                self.frustumplanes.append(torch.Tensor(frustumplane).cuda())
            else:
                self.frustumplanes.append(frustumplane)
        return
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self,idx:int)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,str]:
        image=self.frames[idx].load_image(self.downsample)
        view_matrix=self.frames[idx].get_viewmatrix()
        proj_matrix=self.cameras[self.frames[idx].camera_id].get_inv_z_project_matrix()
        frustumplane=self.frustumplanes[idx]
        frame_name=self.frames[idx].name
        return torch.Tensor(view_matrix),torch.Tensor(proj_matrix),torch.Tensor(frustumplane),torch.Tensor(image),frame_name
    
    def get_norm(self)->tuple[float,float]:
        def get_center_and_diag(cam_centers):
            cam_centers = np.hstack(cam_centers)
            avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
            center = avg_cam_center
            dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
            diagonal = np.max(dist)
            return center.flatten(), diagonal

        cam_centers = []

        for frame in self.frames:
            cam_centers.append(frame.get_camera_center()[:,None])

        center, diagonal = get_center_and_diag(cam_centers)
        radius = diagonal * 1.1
        translate = -center
        return translate,radius
        
