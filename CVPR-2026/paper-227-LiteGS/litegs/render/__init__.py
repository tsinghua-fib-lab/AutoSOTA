import torch
import math
import typing
import torch.cuda.nvtx as nvtx

from .. import utils
from ..utils.statistic_helper import StatisticsHelperInst,StatisticsHelper
from ..spreading.utils import compute_gaussian_count_per_tile
from .. import arguments
from .. import scene

def render_preprocess(cluster_origin:torch.Tensor,cluster_extend:torch.Tensor,frustumplane:torch.Tensor,
                      xyz:torch.Tensor,scale:torch.Tensor,rot:torch.Tensor,sh_0:torch.Tensor,sh_rest:torch.Tensor,opacity:torch.Tensor,
                      op:arguments.OptimizationParams,pp:arguments.PipelineParams):
    nvtx.range_push("Culling")
    if pp.cluster_size:
        if cluster_origin is None or cluster_extend is None:
            cluster_origin,cluster_extend=scene.cluster.get_cluster_AABB(xyz,scale.exp(),torch.nn.functional.normalize(rot,dim=0))
        visible_chunkid=scene.cluster.get_visible_cluster(cluster_origin,cluster_extend,frustumplane)
        nvtx.range_push("compact")
        if pp.cluster_size and pp.sparse_grad:#enable sparse gradient
            culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity=utils.wrapper.CompactVisibleWithSparseGrad.apply(visible_chunkid,xyz,scale,rot,sh_0,sh_rest,opacity)
        else:
            culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity=scene.cluster.culling(visible_chunkid,xyz,scale,rot,sh_0,sh_rest,opacity)
        nvtx.range_pop()
        culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity=scene.cluster.uncluster(culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity)
        if StatisticsHelperInst.bStart:
            StatisticsHelperInst.set_compact_mask(visible_chunkid)
    else:
        culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity=xyz,scale,rot,sh_0,sh_rest,opacity
        visible_chunkid=None
    nvtx.range_pop()
    return visible_chunkid,culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity

def render(view_matrix:torch.Tensor,proj_matrix:torch.Tensor,
           xyz:torch.Tensor,scale:torch.Tensor,rot:torch.Tensor,sh_0:torch.Tensor,sh_rest:torch.Tensor,opacity:torch.Tensor,
           actived_sh_degree:int,output_shape:tuple[int,int],pp:arguments.PipelineParams,
           entropy_enabled:bool=False)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,tuple[torch.Tensor,torch.Tensor],tuple[torch.Tensor,torch.Tensor]]:

    nvtx.range_push("Activate")
    pad_one=torch.ones((1,xyz.shape[-1]),dtype=xyz.dtype,device=xyz.device)
    xyz=torch.concat((xyz,pad_one),dim=0)
    scale=scale.exp()
    rot=torch.nn.functional.normalize(rot,dim=0)
    opacity=opacity.sigmoid()
    nvtx.range_pop()

    #gs projection
    nvtx.range_push("Proj")
    transform_matrix=utils.wrapper.CreateTransformMatrix.call_fused(scale,rot)
    J=utils.wrapper.CreateRaySpaceTransformMatrix.call_fused(xyz,view_matrix,proj_matrix,output_shape,False)#todo script
    cov2d=utils.wrapper.CreateCov2dDirectly.call_fused(J,view_matrix,transform_matrix)
    eigen_val,eigen_vec,inv_cov2d=utils.wrapper.EighAndInverse2x2Matrix.call_fused(cov2d)
    ndc_pos=utils.wrapper.World2NdcFunc.apply(xyz,view_matrix@proj_matrix)
    nvtx.range_pop()

    #color
    nvtx.range_push("sh")
    camera_center=(-view_matrix[...,3:4,:3]@(view_matrix[...,:3,:3].transpose(-1,-2))).squeeze(1)
    dirs=xyz[:3]-camera_center.unsqueeze(-1)
    dirs=torch.nn.functional.normalize(dirs,dim=-2)
    color=utils.wrapper.SphericalHarmonicToRGB.call_fused(actived_sh_degree,sh_0,sh_rest,dirs)
    nvtx.range_pop()
    
    #visibility table
    tile_start_index,sorted_pointId,b_visible=utils.wrapper.Binning.call_fused(ndc_pos,eigen_val,eigen_vec,opacity,output_shape,pp.tile_size)
    if StatisticsHelperInst.bStart:
        StatisticsHelperInst.update_visible_count(b_visible)
        def gradient_wrapper(tensor:torch.Tensor) -> torch.Tensor:
            return tensor[:,:2].norm(dim=1)
        StatisticsHelperInst.register_tensor_grad_callback('mean2d_grad',ndc_pos,StatisticsHelper.update_mean_std_compact,gradient_wrapper)

    #raster
    tiles_x=int(math.ceil(output_shape[1]/float(pp.tile_size)))
    tiles_y=int(math.ceil(output_shape[0]/float(pp.tile_size)))
    
    # Calculate gaussian count per tile using helper function
    gaussian_count_per_tile = compute_gaussian_count_per_tile(tile_start_index)

    img,transmitance,depth,entropy,normal=utils.wrapper.GaussiansRasterFunc.apply(
        sorted_pointId,tile_start_index,ndc_pos,inv_cov2d,color,opacity,None,
        pp.tile_size,output_shape[0],output_shape[1],pp.enable_transmitance,pp.enable_depth,entropy_enabled
    )

    img=utils.tiles2img_torch(img,tiles_x,tiles_y)[...,:output_shape[0],:output_shape[1]].contiguous()
    if transmitance is not None:
        transmitance=utils.tiles2img_torch(transmitance,tiles_x,tiles_y)[...,:output_shape[0],:output_shape[1]].contiguous()
    if depth is not None:
        depth=utils.tiles2img_torch(depth,tiles_x,tiles_y)[...,:output_shape[0],:output_shape[1]].contiguous()
    if entropy is not None:
        entropy=utils.tiles2img_torch(entropy,tiles_x,tiles_y)[...,:output_shape[0],:output_shape[1]].contiguous()
    if normal is not None:
        normal=utils.tiles2img_torch(normal,tiles_x,tiles_y)[...,:output_shape[0],:output_shape[1]].contiguous()
    return img,transmitance,depth,entropy,normal,gaussian_count_per_tile,(scale, opacity)
