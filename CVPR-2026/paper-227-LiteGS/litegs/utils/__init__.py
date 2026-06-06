import numpy as np
import math
import torch
from .spherical_harmonics import sh0_to_rgb,sh_to_rgb,rgb_to_sh0
from . import wrapper

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def get_view_matrix(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def get_project_matrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = np.zeros((4, 4))

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

@torch.no_grad()
def viewproj_to_frustumplane(viewproj_matrix:torch.Tensor)->torch.Tensor:
    '''
    Parameters:
        viewproj_matrix - the viewproj transform matrix. [N,4,4]
    Returns:
        frustumplane - the planes of view frustum. [N,6,4]
    '''
    N=viewproj_matrix.shape[0]
    frustumplane=torch.zeros((N,6,4),device=viewproj_matrix.device)
    #left plane
    frustumplane[:,0,0]=viewproj_matrix[:,0,3]+viewproj_matrix[:,0,0]
    frustumplane[:,0,1]=viewproj_matrix[:,1,3]+viewproj_matrix[:,1,0]
    frustumplane[:,0,2]=viewproj_matrix[:,2,3]+viewproj_matrix[:,2,0]
    frustumplane[:,0,3]=viewproj_matrix[:,3,3]+viewproj_matrix[:,3,0]
    #right plane
    frustumplane[:,1,0]=viewproj_matrix[:,0,3]-viewproj_matrix[:,0,0]
    frustumplane[:,1,1]=viewproj_matrix[:,1,3]-viewproj_matrix[:,1,0]
    frustumplane[:,1,2]=viewproj_matrix[:,2,3]-viewproj_matrix[:,2,0]
    frustumplane[:,1,3]=viewproj_matrix[:,3,3]-viewproj_matrix[:,3,0]

    #bottom plane
    frustumplane[:,2,0]=viewproj_matrix[:,0,3]+viewproj_matrix[:,0,1]
    frustumplane[:,2,1]=viewproj_matrix[:,1,3]+viewproj_matrix[:,1,1]
    frustumplane[:,2,2]=viewproj_matrix[:,2,3]+viewproj_matrix[:,2,1]
    frustumplane[:,2,3]=viewproj_matrix[:,3,3]+viewproj_matrix[:,3,1]

    #top plane
    frustumplane[:,3,0]=viewproj_matrix[:,0,3]-viewproj_matrix[:,0,1]
    frustumplane[:,3,1]=viewproj_matrix[:,1,3]-viewproj_matrix[:,1,1]
    frustumplane[:,3,2]=viewproj_matrix[:,2,3]-viewproj_matrix[:,2,1]
    frustumplane[:,3,3]=viewproj_matrix[:,3,3]-viewproj_matrix[:,3,1]

    #near plane
    frustumplane[:,4,0]=viewproj_matrix[:,0,2]
    frustumplane[:,4,1]=viewproj_matrix[:,1,2]
    frustumplane[:,4,2]=viewproj_matrix[:,2,2]
    frustumplane[:,4,3]=viewproj_matrix[:,3,2]

    #far plane
    frustumplane[:,5,0]=viewproj_matrix[:,0,3]-viewproj_matrix[:,0,2]
    frustumplane[:,5,1]=viewproj_matrix[:,1,3]-viewproj_matrix[:,1,2]
    frustumplane[:,5,2]=viewproj_matrix[:,2,3]-viewproj_matrix[:,2,2]
    frustumplane[:,5,3]=viewproj_matrix[:,3,3]-viewproj_matrix[:,3,2]

    return frustumplane

@torch.no_grad()
def frustum_culling_aabb(frustumplane,aabb_origin,aabb_ext)->torch.Tensor:
    '''
    Parameters:
        frustumplane - the planes of view frustum. [N,6,4]

        aabb_origin - the origin of Axis-Aligned Bounding Boxes. [3,M]

        aabb_ext - the extension of Axis-Aligned Bounding Boxes. [3,M]
    Returns:
        visibility - is visible. [N,M]
    '''
    assert(aabb_origin.shape[0]==aabb_ext.shape[0])
    N=frustumplane.shape[0]
    M=aabb_origin.shape[0]
    frustumplane=frustumplane
    aabb_origin=aabb_origin
    aabb_ext=aabb_ext
    #project origin to plane normal [M,N,6,1]
    dist_origin=(frustumplane[...,0:3,None]*aabb_origin).sum(-2).permute(2,0,1)+frustumplane[...,3]
    #project extension to plane normal
    dist_ext=(frustumplane[...,0:3,None]*aabb_ext).abs().sum(-2).permute(2,0,1)
    #push out the origin
    pushed_origin_dist=dist_origin+dist_ext #M,N,6
    #is completely outside
    culling=(pushed_origin_dist<0).sum(dim=-1).transpose(0,1)
    visibility=(culling==0)
    return visibility



def img2tiles_torch(img:torch.Tensor,tile_size)->torch.Tensor:
    N,C,H,W=img.shape
    H_tile=math.ceil(H/tile_size)
    W_tile=math.ceil(W/tile_size)
    H_pad=H_tile*tile_size-H
    W_pad=W_tile*tile_size-W
    pad_img=torch.nn.functional.pad(img,(0,W_pad,0,H_pad),'constant',0)
    out=pad_img.reshape(N,C,H_tile,tile_size,W_tile,tile_size).transpose(3,4).reshape(N,C,-1,tile_size,tile_size)
    return out

def tiles2img_torch(tile_img:torch.Tensor,tilesNumX,tilesNumY)->torch.Tensor:
    N=tile_img.shape[0]
    C=tile_img.shape[1]
    tile_H=tile_img.shape[3]
    tile_W=tile_img.shape[4]
    translated_tile_img=tile_img.reshape(N,C,tilesNumY,tilesNumX,tile_H,tile_W).transpose(-2,-3)
    img=translated_tile_img.reshape((N,C,tilesNumY*tile_H,tilesNumX*tile_W))
    return img

