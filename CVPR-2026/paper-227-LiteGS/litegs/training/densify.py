import torch
import math

from ..arguments import DensifyParams
from ..utils.statistic_helper import StatisticsHelperInst
from ..utils import qvec2rotmat
from ..scene import cluster
from ..utils import wrapper
from .. import utils


class DensityControllerBase:
    def __init__(self,densify_params:DensifyParams,bCluster:bool) -> None:
        self.densify_params=densify_params
        self.bCluster=bCluster
        return
    
    @torch.no_grad()
    def step(self,optimizer:torch.optim.Optimizer,epoch:int):
        return
    
    @torch.no_grad()
    def _get_params_from_optimizer(self,optimizer:torch.optim.Optimizer)->list[torch.Tensor]:
        param_dict:dict[str,torch.Tensor]={}
        for param_group in optimizer.param_groups:
            name=param_group['name']
            tensor=param_group['params'][0]
            param_dict[name]=tensor
        xyz=param_dict["xyz"]
        rot=param_dict["rot"]
        scale=param_dict["scale"]
        sh_0=param_dict["sh_0"]
        sh_rest=param_dict["sh_rest"]
        opacity=param_dict["opacity"]
        return xyz,scale,rot,sh_0,sh_rest,opacity

    @torch.no_grad()
    def _cat_tensors_to_optimizer(self, tensors_dict:dict,optimizer:torch.optim.Optimizer):
        for group in optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = optimizer.state.get(group['params'][0], None)
            assert stored_state["exp_avg"].shape == stored_state["exp_avg_sq"].shape and stored_state["exp_avg"].shape==group["params"][0].shape
            if stored_state is not None:
                stored_state["exp_avg"].data=torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=-2).contiguous()
                stored_state["exp_avg_sq"].data=torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=-2).contiguous()
            new_param=torch.cat((group["params"][0], extension_tensor), dim=-2).contiguous()
            optimizer.state.pop(group['params'][0])#pop param
            group["params"][0]=torch.nn.Parameter(new_param)
            optimizer.state[group["params"][0]]=stored_state#assign to new param
            assert stored_state["exp_avg"].shape == stored_state["exp_avg_sq"].shape and stored_state["exp_avg"].shape==group["params"][0].shape
        return
    
    @torch.no_grad()
    def _prune_optimizer_with_state_maintained(self,valid_mask:torch.Tensor,optimizer:torch.optim.Optimizer):
        for group in optimizer.param_groups:
            stored_state = optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                if self.bCluster:
                    chunk_size=stored_state["exp_avg"].shape[-1]
                    uncluster_avg,uncluster_avg_sq=cluster.uncluster(stored_state["exp_avg"],stored_state["exp_avg_sq"])
                    uncluster_avg=uncluster_avg[...,valid_mask]
                    uncluster_avg_sq=uncluster_avg_sq[...,valid_mask]
                    new_avg,new_avg_sq=cluster.cluster_points(chunk_size,uncluster_avg,uncluster_avg_sq)
                else:
                    new_avg=stored_state["exp_avg"][...,valid_mask]
                    new_avg_sq=stored_state["exp_avg_sq"][...,valid_mask]
                stored_state["exp_avg"].data=new_avg
                stored_state["exp_avg_sq"].data=new_avg_sq
            
            if self.bCluster:
                chunk_size=group["params"][0].shape[-1]
                uncluster_param,=cluster.uncluster(group["params"][0])
                uncluster_param=uncluster_param[...,valid_mask]
                new_param,=cluster.cluster_points(chunk_size,uncluster_param)
            else:
                new_param=group["params"][0][...,valid_mask]
            optimizer.state.pop(group['params'][0])#pop param
            group["params"][0]=torch.nn.Parameter(new_param)
            optimizer.state[group["params"][0]]=stored_state#assign to new param
        return

    @torch.no_grad()
    def _prune_optimizer(self,valid_mask:torch.Tensor,optimizer:torch.optim.Optimizer):
        for group in optimizer.param_groups:
            if self.bCluster:
                chunk_size=group["params"][0].shape[-1]
                uncluster_param,=cluster.uncluster(group["params"][0])
                uncluster_param=uncluster_param[...,valid_mask]
                new_param,=cluster.cluster_points(chunk_size,uncluster_param)
            else:
                new_param=group["params"][0][...,valid_mask]

            group["params"][0]=torch.nn.Parameter(new_param)
        return
    
    @torch.no_grad()
    def get_prune_mask(self,actived_opacity:torch.Tensor,actived_scale:torch.Tensor)->torch.Tensor:
        transparent = (actived_opacity < self.min_opacity).squeeze()
        # print(f"actived_opacity: {actived_opacity.shape}")
        # print(f"transparent: {transparent.sum()}")
        invisible = StatisticsHelperInst.get_global_culling()
        prune_mask=transparent
        prune_mask[:invisible.shape[0]]|=invisible
        big_points_vs = StatisticsHelperInst.get_max('radii') > self.max_screen_size
        prune_mask[:invisible.shape[0]]|=big_points_vs
        
        # Scale-based pruning: sum scale across dimensions, get 99% percentile, prune gaussians larger than 10x that
        scale_sum_per_gaussian = actived_scale.sum(dim=0)  # Sum across 3 dimensions, keep per-gaussian
        scale_99_percentile = torch.quantile(scale_sum_per_gaussian.float(), 0.99)
        large_scale_threshold = scale_99_percentile * 10.0
        large_scale_mask = scale_sum_per_gaussian > large_scale_threshold
        prune_mask |= large_scale_mask
        
        return prune_mask

    @torch.no_grad()
    def prune(self,optimizer:torch.optim.Optimizer):
        optimizer.state.clear()
        xyz,scale,rot,sh_0,sh_rest,opacity=self._get_params_from_optimizer(optimizer)
        if self.bCluster:
            chunk_size=xyz.shape[-1]
            xyz,scale,rot,sh_0,sh_rest,opacity=cluster.uncluster(xyz,scale,rot,sh_0,sh_rest,opacity)

        prune_mask=self.get_prune_mask(opacity.sigmoid(),scale.exp())
        if self.bCluster:
            N=prune_mask.sum()
            chunk_num=int(N/chunk_size)
            del_limit=chunk_num*chunk_size
            del_indices=prune_mask.nonzero()[:del_limit,0]
            prune_mask=torch.zeros_like(prune_mask)
            prune_mask[del_indices]=True

        self._prune_optimizer(~prune_mask,optimizer)
        return
    
    @torch.no_grad()
    def reset_opacity(self,optimizer):
        optimizer.state.clear()
        xyz,scale,rot,sh_0,sh_rest,opacity=self._get_params_from_optimizer(optimizer)
        def inverse_sigmoid(x):
            return torch.log(x/(1-x))
        actived_opacities=opacity.sigmoid()
        decay_rate=0.5
        decay_mask=(actived_opacities>1/(255*decay_rate-1))
        decay_rate=decay_mask*decay_rate+(~decay_mask)*1.0
        opacity.data=inverse_sigmoid(actived_opacities*decay_rate)#(actived_opacities.clamp_max(0.005))
        return

    @torch.no_grad()
    def is_densify_actived(self,epoch:int):
        return epoch<self.densify_params.densify_until and epoch>=self.densify_params.densify_from and (
            epoch%self.densify_params.densification_interval==0 or
            epoch%self.densify_params.prune_interval==0)


class DensityControllerOfficial(DensityControllerBase):
    @torch.no_grad()
    def __init__(self,screen_extent:int,densify_params:DensifyParams,bCluster:bool)->None:
        self.grad_threshold=densify_params.densify_grad_threshold
        print(f"[DensityControllerOfficial] densify_grad_threshold: {self.grad_threshold}")
        self.min_opacity=densify_params.opacity_threshold
        self.percent_dense=densify_params.percent_dense
        self.prune_large_point_from=densify_params.prune_large_point_from # not used
        self.screen_extent=screen_extent
        self.max_screen_size=densify_params.screen_size_threshold
        super(DensityControllerOfficial,self).__init__(densify_params,bCluster)
        return

    @torch.no_grad()
    def get_clone_mask(self,actived_scale:torch.Tensor)->torch.Tensor:
        mean2d_grads=StatisticsHelperInst.get_mean('mean2d_grad')
        abnormal_mask = mean2d_grads >= self.grad_threshold
        tiny_pts_mask = actived_scale.max(dim=0).values <= self.percent_dense*self.screen_extent
        selected_pts_mask = abnormal_mask&tiny_pts_mask
        return selected_pts_mask
    
    @torch.no_grad()
    def get_split_mask(self,actived_scale:torch.Tensor,N=2)->torch.Tensor:
        mean2d_grads=StatisticsHelperInst.get_mean('mean2d_grad')
        abnormal_mask = mean2d_grads >= self.grad_threshold
        large_pts_mask = actived_scale.max(dim=0).values > self.percent_dense*self.screen_extent
        selected_pts_mask=abnormal_mask&large_pts_mask
        return selected_pts_mask

    @torch.no_grad()
    def split_and_clone(self,optimizer:torch.optim.Optimizer):
        
        xyz,scale,rot,sh_0,sh_rest,opacity=self._get_params_from_optimizer(optimizer)
        if self.bCluster:
            chunk_size=xyz.shape[-1]
            xyz,scale,rot,sh_0,sh_rest,opacity=cluster.uncluster(xyz,scale,rot,sh_0,sh_rest,opacity)

        clone_mask=self.get_clone_mask(scale.exp())
        split_mask=self.get_split_mask(scale.exp())

        #split
        stds=scale[...,split_mask].exp()
        means=torch.zeros((3,stds.size(-1)),device="cuda")
        samples = torch.normal(mean=means, std=stds).unsqueeze(0)
        transform_matrix=wrapper.CreateTransformMatrix.call_fused(scale[...,split_mask].exp(),torch.nn.functional.normalize(rot[...,split_mask],dim=0))
        rotation_matrix=transform_matrix[:3,:3]
        shift=(samples.permute(2,0,1))@rotation_matrix.permute(2,0,1)
        shift=shift.permute(1,2,0).squeeze(0)
        
        split_xyz=xyz[...,split_mask]+shift
        clone_xyz=xyz[...,clone_mask]
        append_xyz=torch.cat((split_xyz,clone_xyz),dim=-1)
        xyz.data[...,split_mask]-=shift
        
        split_scale = (scale[...,split_mask].exp() / (0.8*2)).log()
        clone_scale = scale[...,clone_mask]
        append_scale = torch.cat((split_scale,clone_scale),dim=-1)
        scale.data[...,split_mask]=split_scale

        split_rot=rot[...,split_mask]
        clone_rot=rot[...,clone_mask]
        append_rot = torch.cat((split_rot,clone_rot),dim=-1)

        split_sh_0=sh_0[...,split_mask]
        clone_sh_0=sh_0[...,clone_mask]
        append_sh_0 = torch.cat((split_sh_0,clone_sh_0),dim=-1)

        split_sh_rest=sh_rest[...,split_mask]
        clone_sh_rest=sh_rest[...,clone_mask]
        append_sh_rest = torch.cat((split_sh_rest,clone_sh_rest),dim=-1)

        split_opacity=opacity[...,split_mask]
        clone_opacity=opacity[...,clone_mask]
        append_opacity = torch.cat((split_opacity,clone_opacity),dim=-1)

        if self.bCluster:
            N=append_xyz.shape[-1]
            chunk_num=int(N/chunk_size)
            append_limit=chunk_num*chunk_size
            append_xyz,append_scale,append_rot,append_sh_0,append_sh_rest,append_opacity=cluster.cluster_points(
                chunk_size,append_xyz[...,:append_limit],append_scale[...,:append_limit],
                append_rot[...,:append_limit],append_sh_0[...,:append_limit],
                append_sh_rest[...,:append_limit],append_opacity[...,:append_limit])

        dict_clone = {"xyz": append_xyz,
                      "scale": append_scale,
                      "rot" : append_rot,
                      "sh_0": append_sh_0,
                      "sh_rest": append_sh_rest,
                      "opacity" : append_opacity}
        
        #print("\n#clone:{0} #split:{1} #points:{2}".format(clone_mask.sum().cpu(),split_mask.sum().cpu(),xyz.shape[-1]+append_xyz.shape[-1]*append_xyz.shape[-2]))
        self._cat_tensors_to_optimizer(dict_clone,optimizer)
        return

    @torch.no_grad()
    def step(self,optimizer:torch.optim.Optimizer,epoch:int):
        if epoch<self.densify_params.densify_until and epoch>=self.densify_params.densify_from:
            bUpdate=False
            if epoch%self.densify_params.densification_interval==0:
                self.split_and_clone(optimizer)
                bUpdate=True
            if epoch%self.densify_params.prune_interval==0:
                self.prune(optimizer)
                bUpdate=True
            if epoch%self.densify_params.opacity_reset_interval==0:
                self.reset_opacity(optimizer)
                bUpdate=True
            if bUpdate:
                xyz,scale,rot,sh_0,sh_rest,opacity=self._get_params_from_optimizer(optimizer)
                StatisticsHelperInst.reset(xyz.shape[-2],xyz.shape[-1],self.is_densify_actived)
                torch.cuda.empty_cache()
        return self._get_params_from_optimizer(optimizer)


class DensityControllerDashGaussian(DensityControllerBase):
    @torch.no_grad()
    def __init__(self,screen_extent:int,densify_params:DensifyParams,bCluster:bool)->None:
        self.grad_threshold=densify_params.densify_grad_threshold
        print(f"[DensityControllerDashGaussian] densify_grad_threshold: {self.grad_threshold}")
        self.min_opacity=densify_params.opacity_threshold
        self.percent_dense=densify_params.percent_dense
        self.prune_large_point_from=densify_params.prune_large_point_from # not used
        self.screen_extent=screen_extent
        self.max_screen_size=densify_params.screen_size_threshold
        super(DensityControllerDashGaussian,self).__init__(densify_params,bCluster)
        return

    @torch.no_grad()
    def split_and_clone(self,optimizer:torch.optim.Optimizer,n_densify_dashgaussian:int,mean2d_grads:torch.Tensor):
        xyz,scale,rot,sh_0,sh_rest,opacity=self._get_params_from_optimizer(optimizer)
        if self.bCluster:
            chunk_size=xyz.shape[-1]
            xyz,scale,rot,sh_0,sh_rest,opacity=cluster.uncluster(xyz,scale,rot,sh_0,sh_rest,opacity)

        # Pre-compute activated scale to avoid duplicate exp() calls
        actived_scale = scale.exp()

        # Compute scale-based masks once
        scale_threshold = self.percent_dense * self.screen_extent
        scale_max_values = actived_scale.max(dim=0).values
        tiny_pts_mask = scale_max_values <= scale_threshold
        large_pts_mask = scale_max_values > scale_threshold

        # Select topk n_densify_dashgaussian from mean2d_grads to build final_abnormal_mask
        _, top_indices = torch.topk(mean2d_grads, n_densify_dashgaussian)
        # Create boolean mask directly from indices
        final_abnormal_mask = torch.zeros(mean2d_grads.shape[0], dtype=torch.bool, device=mean2d_grads.device)
        final_abnormal_mask[top_indices] = True

        # Apply final masks
        clone_mask = final_abnormal_mask & tiny_pts_mask
        split_mask = final_abnormal_mask & large_pts_mask

        #split
        stds=actived_scale[...,split_mask]
        means=torch.zeros((3,stds.size(-1)),device="cuda")
        samples = torch.normal(mean=means, std=stds).unsqueeze(0)
        transform_matrix=wrapper.CreateTransformMatrix.call_fused(stds,torch.nn.functional.normalize(rot[...,split_mask],dim=0))
        rotation_matrix=transform_matrix[:3,:3]
        shift=(samples.permute(2,0,1))@rotation_matrix.permute(2,0,1)
        shift=shift.permute(1,2,0).squeeze(0)
        
        split_xyz=xyz[...,split_mask]+shift
        clone_xyz=xyz[...,clone_mask]
        append_xyz=torch.cat((split_xyz,clone_xyz),dim=-1)
        xyz.data[...,split_mask]-=shift
        
        split_scale = (scale[...,split_mask].exp() / (0.8*2)).log()
        clone_scale = scale[...,clone_mask]
        append_scale = torch.cat((split_scale,clone_scale),dim=-1)
        scale.data[...,split_mask]=split_scale

        split_rot=rot[...,split_mask]
        clone_rot=rot[...,clone_mask]
        append_rot = torch.cat((split_rot,clone_rot),dim=-1)

        split_sh_0=sh_0[...,split_mask]
        clone_sh_0=sh_0[...,clone_mask]
        append_sh_0 = torch.cat((split_sh_0,clone_sh_0),dim=-1)

        split_sh_rest=sh_rest[...,split_mask]
        clone_sh_rest=sh_rest[...,clone_mask]
        append_sh_rest = torch.cat((split_sh_rest,clone_sh_rest),dim=-1)

        split_opacity=opacity[...,split_mask]
        clone_opacity=opacity[...,clone_mask]
        append_opacity = torch.cat((split_opacity,clone_opacity),dim=-1)

        if self.bCluster:
            N=append_xyz.shape[-1]
            chunk_num=int(N/chunk_size)
            append_limit=chunk_num*chunk_size
            append_xyz,append_scale,append_rot,append_sh_0,append_sh_rest,append_opacity=cluster.cluster_points(
                chunk_size,append_xyz[...,:append_limit],append_scale[...,:append_limit],
                append_rot[...,:append_limit],append_sh_0[...,:append_limit],
                append_sh_rest[...,:append_limit],append_opacity[...,:append_limit])

        dict_clone = {"xyz": append_xyz,
                      "scale": append_scale,
                      "rot" : append_rot,
                      "sh_0": append_sh_0,
                      "sh_rest": append_sh_rest,
                      "opacity" : append_opacity}

        #print("\n#clone:{0} #split:{1} #points:{2}".format(clone_mask.sum().cpu(),split_mask.sum().cpu(),xyz.shape[-1]+append_xyz.shape[-1]*append_xyz.shape[-2]))
        self._cat_tensors_to_optimizer(dict_clone,optimizer)

    @torch.no_grad()
    def step(self,optimizer:torch.optim.Optimizer,epoch:int,densify_rate:float,cur_n_gaussian:int):
        if epoch<self.densify_params.densify_until and epoch>=self.densify_params.densify_from:
            bUpdate=False
            if epoch%self.densify_params.densification_interval==0:
                n_densify_dashgaussian = int(cur_n_gaussian * (1 + densify_rate) - cur_n_gaussian)
                mean2d_grads=StatisticsHelperInst.get_mean('mean2d_grad')
                self.split_and_clone(optimizer, n_densify_dashgaussian, mean2d_grads)
                bUpdate=True
            if epoch%self.densify_params.prune_interval==0:
                self.prune(optimizer)
                bUpdate=True
            if epoch%self.densify_params.opacity_reset_interval==0:
                # print(f"[DensityControllerDashGaussian] Reset opacity at epoch {epoch}")
                self.reset_opacity(optimizer)
                bUpdate=True
            if bUpdate:
                xyz,scale,rot,sh_0,sh_rest,opacity=self._get_params_from_optimizer(optimizer)
                StatisticsHelperInst.reset(xyz.shape[-2],xyz.shape[-1],self.is_densify_actived)
                torch.cuda.empty_cache()
        return self._get_params_from_optimizer(optimizer)


class DensityControllerWithFinalCount(DensityControllerBase):
    @torch.no_grad()
    def __init__(self,screen_extent:int,densify_params:DensifyParams,bCluster:bool)->None:
        self.grad_threshold=densify_params.densify_grad_threshold
        print(f"[DensityControllerWithFinalCount] densify_grad_threshold: {self.grad_threshold}")
        self.min_opacity=densify_params.opacity_threshold
        self.percent_dense=densify_params.percent_dense
        self.prune_large_point_from=densify_params.prune_large_point_from # not used
        self.screen_extent=screen_extent
        self.max_screen_size=densify_params.screen_size_threshold
        super(DensityControllerWithFinalCount,self).__init__(densify_params,bCluster)
        return

    @torch.no_grad()
    def calculate_densify_rate_linear(self, current_epoch:int, current_count:int, target_count:int) -> float:
        """
        Calculate densify_rate for linear growth from current count to target count.

        Args:
            current_epoch: Current training epoch
            current_count: Current number of gaussians
            target_count: Target number of gaussians (max_n_gaussian)

        Returns:
            densify_rate: The rate at which to densify (0.0 if outside densification window)
        """
        densify_from_epoch = self.densify_params.densify_from
        densify_until_epoch = self.densify_params.densify_until

        # Outside densification window
        if current_epoch < densify_from_epoch or current_epoch >= densify_until_epoch:
            return 0.0

        # Calculate total number of densification chances
        # densify_until_epoch is exclusive, so last epoch is densify_until_epoch - 1
        final_densify_epoch = densify_until_epoch - 1
        total_chances = (final_densify_epoch - densify_from_epoch) // self.densify_params.densification_interval + 1

        # Calculate remaining chances (including current epoch)
        remaining_chances = (final_densify_epoch - current_epoch) // self.densify_params.densification_interval + 1

        # Total gaussians to add
        total_to_add = target_count - current_count

        # Gaussians to add per chance for linear growth
        add_per_chance = total_to_add / remaining_chances

        # Densify rate = (gaussians to add this chance) / (current count)
        densify_rate = add_per_chance / current_count

        return min(max(densify_rate, 0.0), 1.0)

    @torch.no_grad()
    def split_and_clone(self,optimizer:torch.optim.Optimizer,n_densify_dashgaussian:int,mean2d_grads:torch.Tensor):
        xyz,scale,rot,sh_0,sh_rest,opacity=self._get_params_from_optimizer(optimizer)
        if self.bCluster:
            chunk_size=xyz.shape[-1]
            xyz,scale,rot,sh_0,sh_rest,opacity=cluster.uncluster(xyz,scale,rot,sh_0,sh_rest,opacity)

        # Pre-compute activated scale to avoid duplicate exp() calls
        actived_scale = scale.exp()

        # Compute scale-based masks once
        scale_threshold = self.percent_dense * self.screen_extent
        scale_max_values = actived_scale.max(dim=0).values
        tiny_pts_mask = scale_max_values <= scale_threshold
        large_pts_mask = scale_max_values > scale_threshold

        # Select topk n_densify_dashgaussian from mean2d_grads to build final_abnormal_mask
        _, top_indices = torch.topk(mean2d_grads, n_densify_dashgaussian)
        # Create boolean mask directly from indices
        final_abnormal_mask = torch.zeros(mean2d_grads.shape[0], dtype=torch.bool, device=mean2d_grads.device)
        final_abnormal_mask[top_indices] = True

        # Apply final masks
        clone_mask = final_abnormal_mask & tiny_pts_mask
        split_mask = final_abnormal_mask & large_pts_mask

        #split
        stds=actived_scale[...,split_mask]
        means=torch.zeros((3,stds.size(-1)),device="cuda")
        samples = torch.normal(mean=means, std=stds).unsqueeze(0)
        transform_matrix=wrapper.CreateTransformMatrix.call_fused(stds,torch.nn.functional.normalize(rot[...,split_mask],dim=0))
        rotation_matrix=transform_matrix[:3,:3]
        shift=(samples.permute(2,0,1))@rotation_matrix.permute(2,0,1)
        shift=shift.permute(1,2,0).squeeze(0)

        split_xyz=xyz[...,split_mask]+shift
        clone_xyz=xyz[...,clone_mask]
        append_xyz=torch.cat((split_xyz,clone_xyz),dim=-1)
        xyz.data[...,split_mask]-=shift

        split_scale = (scale[...,split_mask].exp() / (0.8*2)).log()
        clone_scale = scale[...,clone_mask]
        append_scale = torch.cat((split_scale,clone_scale),dim=-1)
        scale.data[...,split_mask]=split_scale

        split_rot=rot[...,split_mask]
        clone_rot=rot[...,clone_mask]
        append_rot = torch.cat((split_rot,clone_rot),dim=-1)

        split_sh_0=sh_0[...,split_mask]
        clone_sh_0=sh_0[...,clone_mask]
        append_sh_0 = torch.cat((split_sh_0,clone_sh_0),dim=-1)

        split_sh_rest=sh_rest[...,split_mask]
        clone_sh_rest=sh_rest[...,clone_mask]
        append_sh_rest = torch.cat((split_sh_rest,clone_sh_rest),dim=-1)

        split_opacity=opacity[...,split_mask]
        clone_opacity=opacity[...,clone_mask]
        append_opacity = torch.cat((split_opacity,clone_opacity),dim=-1)

        if self.bCluster:
            N=append_xyz.shape[-1]
            chunk_num=int(N/chunk_size)
            append_limit=chunk_num*chunk_size
            append_xyz,append_scale,append_rot,append_sh_0,append_sh_rest,append_opacity=cluster.cluster_points(
                chunk_size,append_xyz[...,:append_limit],append_scale[...,:append_limit],
                append_rot[...,:append_limit],append_sh_0[...,:append_limit],
                append_sh_rest[...,:append_limit],append_opacity[...,:append_limit])

        dict_clone = {"xyz": append_xyz,
                      "scale": append_scale,
                      "rot" : append_rot,
                      "sh_0": append_sh_0,
                      "sh_rest": append_sh_rest,
                      "opacity" : append_opacity}

        #print("\n#clone:{0} #split:{1} #points:{2}".format(clone_mask.sum().cpu(),split_mask.sum().cpu(),xyz.shape[-1]+append_xyz.shape[-1]*append_xyz.shape[-2]))
        self._cat_tensors_to_optimizer(dict_clone,optimizer)

    @torch.no_grad()
    def step(self,optimizer:torch.optim.Optimizer,epoch:int,densify_rate:float,cur_n_gaussian:int):
        if epoch<self.densify_params.densify_until and epoch>=self.densify_params.densify_from:
            bUpdate=False
            if epoch%self.densify_params.densification_interval==0:
                n_densify_dashgaussian = int(cur_n_gaussian * (1 + densify_rate) - cur_n_gaussian)
                mean2d_grads=StatisticsHelperInst.get_mean('mean2d_grad')
                self.split_and_clone(optimizer, n_densify_dashgaussian, mean2d_grads)
                bUpdate=True
            if epoch%self.densify_params.prune_interval==0:
                self.prune(optimizer)
                bUpdate=True
            if epoch%self.densify_params.opacity_reset_interval==0:
                self.reset_opacity(optimizer)
                bUpdate=True
            if bUpdate:
                xyz,scale,rot,sh_0,sh_rest,opacity=self._get_params_from_optimizer(optimizer)
                StatisticsHelperInst.reset(xyz.shape[-2],xyz.shape[-1],self.is_densify_actived)
                torch.cuda.empty_cache()
        return self._get_params_from_optimizer(optimizer)
