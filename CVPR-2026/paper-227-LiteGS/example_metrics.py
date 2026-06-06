from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
from torchmetrics.image import psnr,ssim,lpip
import sys
import os
from tqdm import tqdm
import torchvision.utils as vutils
import litegs
import litegs.config


# ── Output group flags ──────────────────────────────────────────────────────
SAVE_BASIC      = True  # gt, render
# ────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp_cdo,op_cdo,pp_cdo,dp_cdo=litegs.config.get_default_arg()
    litegs.arguments.ModelParams.add_cmdline_arg(lp_cdo,parser)
    litegs.arguments.OptimizationParams.add_cmdline_arg(op_cdo,parser)
    litegs.arguments.PipelineParams.add_cmdline_arg(pp_cdo,parser)
    litegs.arguments.DensifyParams.add_cmdline_arg(dp_cdo,parser)
    
    parser.add_argument("--save_images", action="store_true", help="Save rendered and ground truth images")
    args = parser.parse_args(sys.argv[1:])
    
    lp=litegs.arguments.ModelParams.extract(args)
    op=litegs.arguments.OptimizationParams.extract(args)
    pp=litegs.arguments.PipelineParams.extract(args)
    dp=litegs.arguments.DensifyParams.extract(args)

    cameras_info:dict[int,litegs.data.CameraInfo]=None
    camera_frames:list[litegs.data.CameraFrame]=None
    cameras_info,camera_frames,init_xyz,init_color=litegs.io_manager.load_colmap_result(lp.source_path,lp.images)#lp.sh_degree,lp.resolution

    #preload
    for camera_frame in camera_frames:
        camera_frame.load_image()

    #Dataset
    training_frames=[c for idx, c in enumerate(camera_frames) if idx % 8 != 0]
    test_frames=[c for idx, c in enumerate(camera_frames) if idx % 8 == 0]
    trainingset=litegs.data.CameraFrameDataset(cameras_info,training_frames,lp.resolution,pp.device_preload)
    train_loader = DataLoader(trainingset, batch_size=1,shuffle=False,pin_memory=not pp.device_preload)
    testset=litegs.data.CameraFrameDataset(cameras_info,test_frames,lp.resolution,pp.device_preload)
    test_loader = DataLoader(testset, batch_size=1,shuffle=False,pin_memory=not pp.device_preload)
    norm_trans,norm_radius=trainingset.get_norm()

    #model
    xyz,scale,rot,sh_0,sh_rest,opacity=litegs.io_manager.load_ply(os.path.join(lp.model_path,"point_cloud","finish","point_cloud.ply"),lp.sh_degree)
    xyz=torch.Tensor(xyz).cuda()
    scale=torch.Tensor(scale).cuda()
    rot=torch.Tensor(rot).cuda()
    sh_0=torch.Tensor(sh_0).cuda()
    sh_rest=torch.Tensor(sh_rest).cuda()
    opacity=torch.Tensor(opacity).cuda()
    cluster_origin=None
    cluster_extend=None
    if pp.cluster_size>0:
        xyz,scale,rot,sh_0,sh_rest,opacity=litegs.scene.point.spatial_refine(False,None,xyz,scale,rot,sh_0,sh_rest,opacity)
        xyz,scale,rot,sh_0,sh_rest,opacity=litegs.scene.cluster.cluster_points(pp.cluster_size,xyz,scale,rot,sh_0,sh_rest,opacity)
        cluster_origin,cluster_extend=litegs.scene.cluster.get_cluster_AABB(xyz,scale.exp(),torch.nn.functional.normalize(rot,dim=0))

    #metrics
    ssim_metrics=ssim.StructuralSimilarityIndexMeasure(data_range=(0.0,1.0)).cuda()
    psnr_metrics=psnr.PeakSignalNoiseRatio(data_range=(0.0,1.0)).cuda()
    lpip_metrics=lpip.LearnedPerceptualImagePatchSimilarity(net_type='vgg').cuda()

    # Create output directories if saving images
    if args.save_images:
        output_dir = os.path.join(lp.model_path, "evaluation")
        os.makedirs(output_dir, exist_ok=True)

    #iter
    loaders={"Trainingset":train_loader,"Testset":test_loader}
    for loader_name,loader in loaders.items():
        ssim_list=[]
        psnr_list=[]
        lpips_list=[]

        fps_total_ms = 0.0
        fps_frame_count = 0

        # Create dataset specific output directory
        if args.save_images:
            dataset_dir = os.path.join(output_dir, loader_name.lower())
            if SAVE_BASIC:
                gt_dir = os.path.join(dataset_dir, "gt")
                render_dir = os.path.join(dataset_dir, "render")
                os.makedirs(gt_dir, exist_ok=True)
                os.makedirs(render_dir, exist_ok=True)
        
        print(f"Evaluating {loader_name}...")
        for idx, (view_matrix,proj_matrix,frustumplane,gt_image,frame_name) in enumerate(tqdm(loader, desc=f"{loader_name}")):
            view_matrix=view_matrix.cuda()
            proj_matrix=proj_matrix.cuda()
            frustumplane=frustumplane.cuda()
            gt_image=gt_image.cuda()/255.0

            # Render-only timing: includes preprocess + render, excludes metric computation and IO.
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            _,culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity=litegs.render.render_preprocess(cluster_origin,cluster_extend,frustumplane,
                                                                                                    xyz,scale,rot,sh_0,sh_rest,opacity,op,pp)
            img,_,_,_,_,_,_=litegs.render.render(
                view_matrix,proj_matrix,culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity,
                lp.sh_degree,gt_image.shape[2:],pp,False
            )
            end_event.record()
            torch.cuda.synchronize()
            fps_total_ms += start_event.elapsed_time(end_event)
            fps_frame_count += 1

            frame_name_str = frame_name[0] if isinstance(frame_name, (tuple, list)) else frame_name

            ssim_list.append(ssim_metrics(img,gt_image).unsqueeze(0))
            psnr_list.append(psnr_metrics(img,gt_image).unsqueeze(0))
            lpips_list.append(lpip_metrics(img,gt_image).unsqueeze(0))
            
            # Save images if requested
            if args.save_images:
                base_name = os.path.splitext(frame_name_str)[0]

                if SAVE_BASIC:
                    current_ssim = ssim_metrics(img, gt_image).item()
                    current_psnr = psnr_metrics(img, gt_image).item()
                    current_lpips = lpip_metrics(img, gt_image).item()
                    # gt: base filename only (GT is fixed, no metrics needed)
                    vutils.save_image(gt_image.squeeze(0), os.path.join(gt_dir, f"{base_name}.png"), normalize=False)
                    # render: psnr + ssim + lpips
                    render_suffix = f"_psnr{current_psnr:.2f}_ssim{current_ssim:.3f}_lpips{current_lpips:.3f}"
                    vutils.save_image(img.squeeze(0), os.path.join(render_dir, f"{base_name}{render_suffix}.png"), normalize=False)

        ssim_mean=torch.concat(ssim_list,dim=0).mean()
        psnr_mean=torch.concat(psnr_list,dim=0).mean()
        lpips_mean=torch.concat(lpips_list,dim=0).mean()
        avg_render_fps = (fps_frame_count * 1000.0 / fps_total_ms) if fps_total_ms > 0 else 0.0

        print("  Scene:{0}".format(lp.model_path+" "+loader_name))
        print("  SSIM : {:>12.7f}".format(float(ssim_mean)))
        print("  PSNR : {:>12.7f}".format(float(psnr_mean)))
        print("  LPIPS: {:>12.7f}".format(float(lpips_mean)))
        print("  FPS  : {:>12.4f} (render-only, no warmup)".format(float(avg_render_fps)))
        print("")
        
        # Save metrics to file
        metrics_file = os.path.join(lp.model_path, f"metrics_{loader_name.lower()}.txt")
        with open(metrics_file, 'w') as f:
            f.write(f"Scene: {lp.model_path} {loader_name}\n")
            f.write(f"SSIM:  {float(ssim_mean):.7f}\n")
            f.write(f"PSNR:  {float(psnr_mean):.7f}\n")
            f.write(f"LPIPS: {float(lpips_mean):.7f}\n")
            f.write(f"FPS:   {float(avg_render_fps):.4f} (render-only, no warmup)\n")

        # Save FPS to a dedicated file for easier post-processing.
        fps_file = os.path.join(lp.model_path, f"fps_{loader_name.lower()}.txt")
        with open(fps_file, 'w') as f:
            f.write(f"Scene: {lp.model_path} {loader_name}\n")
            f.write(f"FPS: {float(avg_render_fps):.4f}\n")
            f.write("Timing: render-only, no warmup\n")
