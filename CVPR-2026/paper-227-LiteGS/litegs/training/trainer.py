import torch
from torch.utils.data import DataLoader
import fused_ssim
from torchmetrics.image import psnr, ssim, lpip
from tqdm import tqdm
import os
import time
import itertools

from .. import arguments
from .. import data
from .. import io_manager
from .. import scene
from . import optimizer
from ..data import CameraFrameDataset
from .. import render
from ..utils.statistic_helper import StatisticsHelperInst
from ..spreading.scale_scheduler import ScaleScheduler
from ..spreading.entropy_scheduler import get_entropy_weight
from . import densify
from FastLanczos import lanczos_resample
from ..spreading.dashgaussian_scheduler import DashGaussianScheduler
from ..spreading.utils import save_gaussian_count, save_training_time, preprocess_gt_images

def __l1_loss(network_output:torch.Tensor, gt:torch.Tensor)->torch.Tensor:
    return torch.abs((network_output - gt)).mean()

def start(lp:arguments.ModelParams,op:arguments.OptimizationParams,pp:arguments.PipelineParams,dp:arguments.DensifyParams,
          test_epochs=[],save_ply=[],save_checkpoint=[],start_checkpoint=None):

    cameras_info:dict[int,data.CameraInfo]=None
    camera_frames:list[data.CameraFrame]=None
    cameras_info,camera_frames,init_xyz,init_color=io_manager.load_colmap_result(lp.source_path,lp.images)#lp.sh_degree,lp.resolution

    #preload
    for camera_frame in camera_frames:
        camera_frame.load_image(lp.resolution)

    # Print model path and image size
    print(f"Model path: {lp.model_path}")
    first_image = camera_frames[0].image[lp.resolution]
    h, w = first_image.shape[1], first_image.shape[2]
    print(f"GT image size: {w} x {h}")

    #Dataset
    if lp.eval:
        training_frames=[c for idx, c in enumerate(camera_frames) if idx % 8 != 0]
        test_frames=[c for idx, c in enumerate(camera_frames) if idx % 8 == 0]
    else:
        training_frames=camera_frames
        test_frames=None
    trainingset=CameraFrameDataset(cameras_info,training_frames,lp.resolution,pp.device_preload)
    print(f"training set length: {len(training_frames)}")
    train_loader = DataLoader(trainingset, batch_size=1,shuffle=True,pin_memory=not pp.device_preload)
    test_loader=None
    if lp.eval:
        testset=CameraFrameDataset(cameras_info,test_frames,lp.resolution,pp.device_preload)
        test_loader = DataLoader(testset, batch_size=1,shuffle=True,pin_memory=not pp.device_preload)
    norm_trans,norm_radius=trainingset.get_norm()

    #torch parameter
    cluster_origin=None
    cluster_extend=None
    if start_checkpoint is None:
        init_xyz=torch.tensor(init_xyz,dtype=torch.float32,device='cuda')
        init_color=torch.tensor(init_color,dtype=torch.float32,device='cuda')
        xyz,scale,rot,sh_0,sh_rest,opacity=scene.create_gaussians(init_xyz,init_color,lp.sh_degree)
        if pp.cluster_size:
            xyz,scale,rot,sh_0,sh_rest,opacity=scene.cluster.cluster_points(pp.cluster_size,xyz,scale,rot,sh_0,sh_rest,opacity)
        xyz=torch.nn.Parameter(xyz)
        scale=torch.nn.Parameter(scale)
        rot=torch.nn.Parameter(rot)
        sh_0=torch.nn.Parameter(sh_0)
        sh_rest=torch.nn.Parameter(sh_rest)
        opacity=torch.nn.Parameter(opacity)
        opt,schedular=optimizer.get_optimizer(xyz,scale,rot,sh_0,sh_rest,opacity,norm_radius,op,pp)
        start_epoch=0
    else:
        xyz,scale,rot,sh_0,sh_rest,opacity,start_epoch,opt,schedular=io_manager.load_checkpoint(start_checkpoint)
        if pp.cluster_size:
            cluster_origin,cluster_extend=scene.cluster.get_cluster_AABB(xyz,scale.exp(),torch.nn.functional.normalize(rot,dim=0))
    actived_sh_degree=0

    dashgaussian_enabled = pp.max_n_gaussian > 0
    print(f"dashgaussian_enabled: {dashgaussian_enabled}")
    print(f"scale_reset_factor: {op.scale_reset_factor}")
    print(f"lambda_entropy: {op.lambda_entropy}")

    #init
    # Fixed iterations per epoch instead of dataset-dependent iterations
    total_epoch=int(op.iterations/op.iterations_per_epoch)
    if dp.densify_until<0:
        if not dashgaussian_enabled:
            dp.densify_until=int(int(total_epoch/2)/dp.opacity_reset_interval)*dp.opacity_reset_interval
        else:
            dp.densify_until=int(int(total_epoch*0.9)/dp.opacity_reset_interval)*dp.opacity_reset_interval
    if not dashgaussian_enabled:
        density_controller=densify.DensityControllerWithFinalCount(norm_radius,dp,pp.cluster_size>0)
    else:
        density_controller=densify.DensityControllerDashGaussian(norm_radius,dp,pp.cluster_size>0)
    StatisticsHelperInst.reset(xyz.shape[-2],xyz.shape[-1],density_controller.is_densify_actived)

    if not dashgaussian_enabled:
        allowed_render_scales = [1]
        render_scale = 1
    else:
        dashgaussian_scheduler = DashGaussianScheduler(pp, dp, xyz.shape[1] * xyz.shape[2], [cam.image[lp.resolution] for cam in training_frames], iterations_per_epoch=op.iterations_per_epoch)
        allowed_render_scales = dashgaussian_scheduler.allowed_render_scales
        render_scale = dashgaussian_scheduler.get_res_scale(0)
    print(f'allowed_render_scales: {allowed_render_scales}')

    # Initialize scale scheduler
    scale_scheduler = ScaleScheduler(
        opacity_reset_interval=dp.opacity_reset_interval,
        total_epochs=total_epoch,
        until_epoch=int(int(total_epoch*0.9)/dp.opacity_reset_interval)*dp.opacity_reset_interval,
        iterations_per_epoch=op.iterations_per_epoch,
        scale_reset_factor=op.scale_reset_factor
    )
    
    # Cache for resized images: (frame_name, scale) -> resized_image
    resized_image_cache = {}
    preprocess_gt_images(train_loader, resized_image_cache, pp, allowed_render_scales)
    preprocess_gt_images(test_loader, resized_image_cache, pp, allowed_render_scales)

    total_testing_time = 0.0
    progress_bar = tqdm(range(start_epoch, total_epoch), desc="Training progress")
    progress_bar.update(0)

    for epoch in range(start_epoch,total_epoch):
        period_info = scale_scheduler.calculate_period_info(epoch)
        entropy_weight = get_entropy_weight(op.lambda_entropy, period_info, op.scale_reset_factor)
        with torch.no_grad():
            spatial_refine = (epoch==0) or (epoch>pp.spatial_refine_interval and (epoch-1)%pp.spatial_refine_interval==0)
            if spatial_refine:#spatial refine
                scene.spatial_refine(pp.cluster_size>0,opt,xyz)
            if pp.cluster_size>0 and (spatial_refine or density_controller.is_densify_actived(epoch-1)):
                cluster_origin,cluster_extend=scene.cluster.get_cluster_AABB(xyz,scale.exp(),torch.nn.functional.normalize(rot,dim=0))
            if actived_sh_degree<lp.sh_degree:
                actived_sh_degree=min(int(epoch/5),lp.sh_degree)

        with StatisticsHelperInst.try_start(epoch):
            cycled_loader = itertools.cycle(train_loader)
            for _ in range(op.iterations_per_epoch):
                view_matrix,proj_matrix,frustumplane,gt_image,frame_name = next(cycled_loader)

                view_matrix=view_matrix.cuda()
                proj_matrix=proj_matrix.cuda()
                frustumplane=frustumplane.cuda()

                # Load GT image from cache for any render scale
                cached_images = []
                for i in range(gt_image.shape[0]):
                    cache_key = (frame_name[i], render_scale, "gt")
                    if cache_key in resized_image_cache:
                        resized_img = resized_image_cache[cache_key]
                    else:
                        print(f"WARNING: cache missed at {cache_key}")
                        resized_img = lanczos_resample((gt_image[i].cuda() / 255.0).permute(1, 2, 0), scale_factor=render_scale).permute(2, 0, 1)
                        resized_image_cache[cache_key] = resized_img
                    cached_images.append(resized_img)
                gt_image = torch.stack(cached_images, dim=0)

                #cluster culling
                visible_chunkid,culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity=render.render_preprocess(cluster_origin,cluster_extend,frustumplane,
                                                                                                               xyz,scale,rot,sh_0,sh_rest,opacity,op,pp)
                img,_,_,entropy,_,_,_=render.render(view_matrix,proj_matrix,culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity,
                                                            actived_sh_degree,gt_image.shape[2:],pp,entropy_weight > 0.0)

                l1_loss = __l1_loss(img, gt_image)
                l2_loss = ((img - gt_image) ** 2).mean()
                ssim_loss = fused_ssim.fused_ssim(img, gt_image)

                ssim_loss = 1 - ssim_loss
                quality_loss = 0.8*l1_loss + 0.2*l2_loss + op.lambda_dssim*ssim_loss
                
                loss = 0.0
                loss += quality_loss

                # if pp.enable_transmitance:#example for transimitance grad
                #     trans_loss=transmitance.square().mean()*0.01
                #     loss+=trans_loss
                # if pp.enable_depth:#example for depth grad
                #     depth_loss=(1.0-depth).square().mean()*0.01
                #     loss+=depth_loss

                # Add entropy loss if entropy is enabled and entropy data is available
                if entropy_weight > 0.0:
                    entropy_loss = entropy.mean() * entropy_weight
                    loss += entropy_loss

                loss.backward()
                if StatisticsHelperInst.bStart:
                    StatisticsHelperInst.backward_callback()
                if pp.cluster_size and pp.sparse_grad:
                    opt.step(visible_chunkid)
                else:
                    opt.step()

                opt.zero_grad(set_to_none = True)
                # Before step(), the last_epoch is 0. After step(), the last_epoch is 1.
                schedular.step()

        current_iteration = schedular.last_epoch - 1

        #####################################################################################################################

        if epoch in test_epochs:
            # Start timing the testing phase
            test_start_time = time.time()
            # Calculate current training time (excluding testing time)
            current_total_elapsed = progress_bar.format_dict.get('elapsed', 0.0)
            current_training_time_sec = current_total_elapsed - total_testing_time
            print("\n\n")
            with torch.no_grad():
                # Use render_scale for testing when dash is enabled, otherwise use 1.0
                if dashgaussian_enabled:
                    test_render_scale = 1.0
                else:
                    test_render_scale = 1.0
                psnr_metrics=psnr.PeakSignalNoiseRatio(data_range=(0.0,1.0)).cuda()
                ssim_metrics=ssim.StructuralSimilarityIndexMeasure(data_range=(0.0,1.0)).cuda()
                lpip_metrics=lpip.LearnedPerceptualImagePatchSimilarity(net_type='vgg').cuda()
                loaders={"Trainingset":train_loader}
                if lp.eval:
                    loaders["Testset"]=test_loader
                for name,loader in loaders.items():
                    psnr_list=[]
                    ssim_list=[]
                    lpips_list=[]
                    for view_matrix,proj_matrix,frustumplane,gt_image,frame_name in loader:
                        view_matrix=view_matrix.cuda()
                        proj_matrix=proj_matrix.cuda()
                        frustumplane=frustumplane.cuda()
                        # Load GT image from cache with test_render_scale (always 1.0)
                        cached_images = []
                        for i in range(gt_image.shape[0]):
                            cache_key = (frame_name[i], test_render_scale, "gt")
                            cached_images.append(resized_image_cache[cache_key])
                        gt_image = torch.stack(cached_images, dim=0)
                        _,culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity=render.render_preprocess(cluster_origin,cluster_extend,frustumplane,
                                                                                                                xyz,scale,rot,sh_0,sh_rest,opacity,op,pp)
                        img,transmitance,depth,entropy,normal,gaussian_count_per_tile,_=render.render(view_matrix,proj_matrix,culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity,
                                                                    actived_sh_degree,gt_image.shape[2:],pp,False)
                        # Calculate PSNR for this single view
                        view_psnr = psnr_metrics(img, gt_image)
                        psnr_list.append(view_psnr.unsqueeze(0))
                        ssim_list.append(ssim_metrics(img,gt_image).unsqueeze(0))
                        lpips_list.append(lpip_metrics(img,gt_image).unsqueeze(0))
                    psnr_mean = torch.concat(psnr_list,dim=0).mean()
                    ssim_mean = torch.concat(ssim_list,dim=0).mean()
                    lpips_mean = torch.concat(lpips_list,dim=0).mean()
                    print("[EPOCH {}] {} Evaluating: PSNR {} SSIM {} LPIPS {}".format(epoch,name,psnr_mean,ssim_mean,lpips_mean))
            print("\n\n")
            # Record testing time
            test_end_time = time.time()
            epoch_test_time = test_end_time - test_start_time
            total_testing_time += epoch_test_time
            print(f"[EPOCH {epoch}] Testing took {epoch_test_time:.2f}s (Total testing time: {total_testing_time:.2f}s)")

        current_count = xyz.shape[1] * xyz.shape[2]
        if not dashgaussian_enabled:
            densify_rate = density_controller.calculate_densify_rate_linear(epoch, current_count, dp.final_gaussian_count)
            xyz,scale,rot,sh_0,sh_rest,opacity=density_controller.step(opt,epoch,densify_rate,current_count)
        else:
            densify_rate = dashgaussian_scheduler.get_densify_rate(current_iteration, current_count, render_scale)
            xyz,scale,rot,sh_0,sh_rest,opacity=density_controller.step(opt,epoch,densify_rate,current_count)
            render_scale = dashgaussian_scheduler.get_res_scale(current_iteration)

        # Perform scale reset if needed
        scale_scheduler.step(opt, epoch, op.lambda_entropy)
        progress_bar.update()  

        if epoch in save_ply or epoch==total_epoch-1:
            if epoch == total_epoch-1:
                progress_bar.close()
                total_elapsed_time = progress_bar.format_dict['elapsed']
                training_time = total_elapsed_time - total_testing_time
                print("{} takes: {} s (total elapsed: {} s, testing: {} s)".format(
                    lp.model_path, training_time, total_elapsed_time, total_testing_time))
                save_gaussian_count(xyz.shape[1] * xyz.shape[2], total_epoch, schedular.last_epoch, lp.model_path)
                save_training_time(training_time, lp.model_path)

            if pp.cluster_size:
                tensors=scene.cluster.uncluster(xyz,scale,rot,sh_0,sh_rest,opacity)
            else:
                tensors=xyz,scale,rot,sh_0,sh_rest,opacity
            param_nyp=[]
            for tensor in tensors:
                param_nyp.append(tensor.detach().cpu().numpy())
            if epoch==total_epoch-1:
                ply_path=os.path.join(lp.model_path,"point_cloud","finish","point_cloud.ply")
            else:
                ply_path=os.path.join(lp.model_path,"point_cloud","iteration_{}".format(epoch),"point_cloud.ply")
            io_manager.save_ply(ply_path,*param_nyp)
            pass

        if epoch in save_checkpoint:
            io_manager.save_checkpoint(lp.model_path,epoch,opt,schedular)
        
    return
