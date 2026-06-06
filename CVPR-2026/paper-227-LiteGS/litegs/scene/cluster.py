import torch
import math
import torch.cuda.nvtx as nvtx

from .. import utils

def cluster_points(chunksize,*args:list[torch.Tensor])->list[torch.Tensor]:
    '''
    input:[...,N]

    output:[...,chunks_num,chunksize]
    '''
    output=[]
    for input in args:
        if input.shape[-1]%chunksize!=0:
            padding_num=input.shape[-1]%chunksize
            padding_num=chunksize-padding_num
            input=torch.concat([input,input[...,-padding_num:]],dim=-1).contiguous()
        chunks_num=int(input.shape[-1]/chunksize)
        output.append(input.view(*input.shape[:-1],chunks_num,chunksize))
    return *output,

def uncluster(*args:list[torch.Tensor])->list[torch.Tensor]:
    output=[]
    for input in args:
        output.append(input.view(*input.shape[:-2],input.shape[-1]*input.shape[-2]))
    return *output,

@torch.no_grad()
def get_cluster_AABB(clustered_xyz:torch.Tensor,clustered_scale:torch.Tensor,clustered_rot:torch.Tensor)->torch.Tensor:
    '''
    '''
    chunk_size=clustered_xyz.shape[-1]
    chunks_num=clustered_xyz.shape[-2]
    xyz,scale,rot=uncluster(clustered_xyz,clustered_scale,clustered_rot)
    transform_matrix=utils.wrapper.CreateTransformMatrix.call(scale,rot)
    coefficient=2*math.log(255)
    extend_axis=transform_matrix*math.sqrt(coefficient)# == (coefficient*eigen_val).sqrt()*eigen_vec
    point_extend=extend_axis.abs().sum(dim=0)
    point_extend,=cluster_points(chunk_size,point_extend)

    max_xyz=(clustered_xyz+point_extend).max(dim=-1).values
    min_xyz=(clustered_xyz-point_extend).min(dim=-1).values
    origin=(max_xyz+min_xyz)/2
    extend=(max_xyz-min_xyz)/2
    return origin,extend

def get_visible_cluster(cluster_origin:torch.Tensor,cluster_extend:torch.Tensor,frustumplane:torch.Tensor)->torch.Tensor:
    nvtx.range_push("frustum culling")
    chunk_visibility=utils.frustum_culling_aabb(frustumplane,cluster_origin,cluster_extend)#[N,M]
    chunk_visibility=chunk_visibility.any(dim=0)
    nvtx.range_pop()
    nvtx.range_push("nonzero")
    visible_chunkid=chunk_visibility.nonzero()[:,0]
    nvtx.range_pop()
    return visible_chunkid

def culling(visible_chunkid:torch.Tensor,*args)->list[torch.Tensor]:
    culled_tensors=[]
    for tensor in args:
        culled_tensors.append(tensor[...,visible_chunkid,:].contiguous())
    return *culled_tensors,