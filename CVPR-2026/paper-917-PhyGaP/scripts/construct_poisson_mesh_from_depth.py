import os
import sys
import torch
from tqdm import tqdm
# Load model and scene (refer to viz_results for details)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene import Scene, GaussianModel
from gaussian_renderer import render_surfel, render_initial, render_volume, render_surfel_nodefer

from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
from utils.graphics_utils import linear_to_srgb

from utils.image_utils import depth2wpos
from utils.gridmap_envmap_utils import render_cubemap, build_cubemap,build_surf_cubemap
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pytorch3d.ops import knn_points
import pymeshlab

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def poisson_mesh(path, vtx, normal, color, depth, thrsh):

    pbar = tqdm(total=4)
    pbar.update(1)
    pbar.set_description('Poisson meshing')

    # create pcl with normal from sampled points
    ms = pymeshlab.MeshSet()
    pts = pymeshlab.Mesh(vtx.cpu().numpy(), [], normal.cpu().numpy())
    ms.add_mesh(pts)


    # poisson reconstruction
    ms.generate_surface_reconstruction_screened_poisson(depth=depth, preclean=True, samplespernode=1.5)
    vert = ms.current_mesh().vertex_matrix()
    face = ms.current_mesh().face_matrix()
    ms.save_current_mesh(path + '_plain.ply')


    pbar.update(1)
    pbar.set_description('Mesh refining')
    # knn to compute distance and color of poisson-meshed points to sampled points
    nn_dist, nn_idx, _ = knn_points(torch.from_numpy(vert).to(torch.float32).cuda()[None], vtx.cuda()[None], K=4)
    nn_dist = nn_dist[0]
    nn_idx = nn_idx[0]
    nn_color = torch.mean(color[nn_idx], axis=1)

    # create mesh with color and quality (distance to the closest sampled points)
    vert_color = nn_color.clip(0, 1).cpu().numpy()
    vert_color = np.concatenate([vert_color, np.ones_like(vert_color[:, :1])], 1)
    ms.add_mesh(pymeshlab.Mesh(vert, face, v_color_matrix=vert_color, v_scalar_array=nn_dist[:, 0].cpu().numpy()))

    pbar.update(1)
    pbar.set_description('Mesh cleaning')
    # prune outlying vertices and faces in poisson mesh
    ms.compute_selection_by_condition_per_vertex(condselect=f"q>{thrsh}")
    ms.meshing_remove_selected_vertices()

    # fill holes
    ms.meshing_close_holes(maxholesize=300)
    ms.save_current_mesh(path + '_pruned.ply')

    # smoothing, correct boundary aliasing due to pruning
    ms.load_new_mesh(path + '_pruned.ply')
    ms.apply_coord_laplacian_smoothing(stepsmoothnum=3, boundary=True)
    ms.save_current_mesh(path + '_pruned.ply')
    
    pbar.update(1)
    pbar.close()

def resample_points(camera, depth, normal, color, mask):
    camWPos = depth2wpos(depth, mask, camera).permute([1, 2, 0])
    camN = normal.permute([1, 2, 0])
    depth_mask = (depth>0).permute([1, 2, 0]) 
    mask = mask.permute([1, 2, 0]).to(torch.bool)
    
    real_mask = depth_mask & mask
    real_mask = real_mask.detach()[...,0]
    # import pdb; pdb.set_trace()
    camN = camN.detach()[real_mask]
    camWPos = camWPos.detach()[real_mask]
    camRGB = color.permute([1, 2, 0])[real_mask]
    points = torch.cat([camWPos, camN, camRGB], -1)
    return points

def viz_targets(targets):
    if len(targets) > 0:
        all_targets = targets.cpu().numpy()
        kmeans = KMeans(n_clusters=4, random_state=0).fit(all_targets)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        # # 使用KMedoids替代KMeans进行聚类
        # kmedoids = KMedoids(n_clusters=4, random_state=0, method='pam').fit(all_targets)
        # labels = kmedoids.labels_
        # centers = kmedoids.cluster_centers_
        fig = plt.figure(figsize=(16, 8))
        # 标记kmeans中心
    
        # 原视角
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        scatter1 = ax1.scatter(
        all_targets[:, 0], all_targets[:, 2], all_targets[:, 1],
        s=1, c=labels, cmap='tab10'
        )
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('Original View')

        # 沿z轴旋转180度
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        scatter2 = ax2.scatter(
        -all_targets[:, 0], -all_targets[:, 2], all_targets[:, 1],
        s=1, c=labels, cmap='tab10'
        )
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title('Rotated 180° around Z')

        # 调整center为同label target中距该center最近的点
        new_centers = []
        for i in range(centers.shape[0]):
            cluster_points = all_targets[labels == i]
            if len(cluster_points) == 0:
                new_centers.append(centers[i])
                continue
            dists = np.linalg.norm(cluster_points - centers[i], axis=1)
            min_idx = np.argmin(dists)
            new_centers.append(cluster_points[min_idx])
        new_centers = np.stack(new_centers, axis=0)
        centers = new_centers

        # 标记kmeans中心（黑色圆点）
        ax1.scatter(
            centers[:, 0], centers[:, 2], centers[:, 1],
            s=50, c='black', marker='o', edgecolors='k', linewidths=1, label='Centers'
        )
        ax2.scatter(
            -centers[:, 0], -centers[:, 2], centers[:, 1],
            s=50, c='black', marker='o', edgecolors='k', linewidths=1, label='Centers'
        )

        plt.tight_layout()
        plt.savefig(os.path.join("./", "target_points_kmeans_dualview.png"))
        plt.close(fig)
    else:
        print("No target points to visualize.")

def set_gaussian_para(gaussians, opt, vol=False):
    gaussians.enlarge_scale = opt.enlarge_scale
    gaussians.rough_msk_thr = opt.rough_msk_thr 
    gaussians.init_roughness_value = opt.init_roughness_value
    gaussians.init_refl_value = opt.init_refl_value
    gaussians.refl_msk_thr = opt.refl_msk_thr

def reset_gaussian_para(gaussians, opt):
    gaussians.reset_ori_color()
    gaussians.reset_refl_strength(opt.init_refl_value)
    gaussians.reset_roughness(opt.init_roughness_value)
    gaussians.refl_msk_thr = opt.refl_msk_thr
    gaussians.rough_msk_thr = opt.rough_msk_thr
def render_and_save_eval(dataset, opt, pipe, checkpoint, mask_folder, output_root="eval", final_itr=30000, relight=False, relight_env_path=None, subset=None):
    poisson_depth = 8
    # Setup output dirs
    # If subset is specified, write into subfolder (train/test)
    if subset in ("train", "test"):
        out_root = os.path.join(output_root, subset)
    else:
        out_root = output_root

    with torch.no_grad():
        gaussians = GaussianModel(dataset.albedo_sh_degree, dataset.sh_degree)
        set_gaussian_para(gaussians, opt, vol=(opt.volume_render_until_iter > opt.init_until_iter))
        scene = Scene(dataset, gaussians, load_iteration=final_itr, shuffle=False)
        if final_itr >= opt.indirect_from_iter:
            opt.indirect = 1
            bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
            gaussians.load_mesh_from_ply(os.path.dirname(checkpoint), final_itr)
        

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        train_cameras = scene.getTrainCameras()
        test_cameras = scene.getTestCameras()
        if subset == "train":
            cameras = train_cameras
        elif subset == "test":
            cameras = test_cameras
        else:
            cameras = train_cameras + test_cameras

        if pipe.indirect_type in ['obj_env','surf_env']:
            gaussians.reset_gridmap_envmap()
            if pipe.indirect_type == 'obj_env':
                gridmap_cubemaps = build_cubemap(cameras, gaussians, pipe, background, dataset, opt)
            elif pipe.indirect_type == 'surf_env':
                gridmap_cubemaps = build_surf_cubemap(cameras, gaussians, pipe, background, dataset, opt)
            gaussians.update_gridmap_envmap(gridmap_cubemaps,dataset)

        import re
        resampled = []
        for _, camera in enumerate(tqdm(cameras, desc="Rendering for eval")):
            _name = camera.image_name
            stem = os.path.splitext(_name)[0]
            m = re.search(r"(\d+)", stem)
            if m:
                cam_idx = int(m.group(1))
            else:
                raise ValueError(f"Could not extract integer index from image_name '{_name}'")
            # Render final rgb
            render = render_surfel
            render_pkg = render(camera, gaussians, pipe, background, opt=opt)
            depth = render_pkg["surf_depth"]
            normal = render_pkg["rend_normal"]
            rgb = render_pkg["render"]
            rgb = linear_to_srgb(rgb)
            mask = camera.gt_alpha_mask
            sampled_points = resample_points(camera, depth, normal, rgb, mask)
            resampled.append(sampled_points.cpu())
            # import pdb;pdb.set_trace()
        resampled = torch.cat(resampled, 0).cuda()
        # import pdb;pdb.set_trace()
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        mesh_path = f'{out_root}/poisson_mesh_{poisson_depth}'
        ptx = resampled[:, :3].cpu().numpy()
        normals = resampled[:, 3:6].cpu().numpy()
        colors = resampled[:, 6:].cpu().numpy()
        pcd.points = o3d.utility.Vector3dVector(ptx)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(f'{mesh_path}_points_orig.ply', pcd)   
        # poisson_mesh(mesh_path, resampled[:, :3], resampled[:, 3:6], resampled[:, 6:], poisson_depth, 3 * 1e-5)

# 用法示例（需根据实际参数替换）
# render_and_save(dataset, opt, pipe, checkpoint, mask_folder="/path/to/mask", output_root="eval")
if __name__ == "__main__":
    parser = ArgumentParser(description="Render for eval with optional cfg_args bootstrap from checkpoint")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10000,20000,30000,40000,50000,60000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[10000,20000,30000,40000,50000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint .pth (used to locate cfg_args)")
    parser.add_argument("--output_dir", type=str, default="debug/test", help="Directory to save the output images")
    parser.add_argument("--final_iterations",type = int, default = 50000, help="Final iterations to run the evaluation")
    parser.add_argument("--mask_folder", type=str, default=None, help="Folder containing the masks for the images")
    parser.add_argument("--render_relight",  action ="store_true",default=False, help="if relight")
    parser.add_argument("--relight_env_path", type=str, default=None, help="the path to relight env map")
    parser.add_argument("--subset", type=str, choices=["train", "test", "both"], default="both", help="Which camera subset to render")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.test_iterations = args.test_iterations + [i for i in range(10000, args.iterations+1, 5000)]
    args.test_iterations.append(args.volume_render_until_iter)

    # Bootstrap dataset/opt/pipe from cfg_args next to checkpoint (like render_envmap_rotation)
    ckpt_path = args.ckpt if args.ckpt is not None else args.start_checkpoint
    if ckpt_path is None:
        raise FileNotFoundError("Please provide --ckpt or --start_checkpoint to locate cfg_args")
    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))
    cfg_path = os.path.join(ckpt_dir, 'cfg_args')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"cfg_args not found in {ckpt_dir}")
    with open(cfg_path, 'r') as f:
        cfg_text = f.read()
    # Safe eval: only allow argparse.Namespace
    cfg_ns = eval(cfg_text, {"__builtins__": {}}, {"Namespace": Namespace})

    # Instantiate parameter builders with fresh parsers and extract from cfg_ns
    lp_boot = ModelParams(ArgumentParser())
    op_boot = OptimizationParams(ArgumentParser())
    pp_boot = PipelineParams(ArgumentParser())
    dataset = lp_boot.extract(cfg_ns)
    opt = op_boot.extract(cfg_ns)
    pipe = pp_boot.extract(cfg_ns)
    opt.voxel_size = 1
    if dataset.double_view:
        dataset.double_view = False  # Disable double view loading for eval rendering
        orig_source_path = dataset.source_path
        for cam in ["camera_0", "camera_1"]:
            # import pdb; pdb.set_trace()
            dataset.source_path = os.path.join(orig_source_path, cam)
            # import pdb; pdb.set_trace()
            output_subdir = os.path.join(args.output_dir, cam)
            os.makedirs(output_subdir, exist_ok=True)
            if args.subset == "both":
                # Render train
                render_and_save_eval(dataset, opt, pipe, ckpt_path, mask_folder=args.mask_folder, output_root=output_subdir,
                                    final_itr=args.final_iterations, relight=args.render_relight, relight_env_path=args.relight_env_path, subset="train")
                # Render test
                render_and_save_eval(dataset, opt, pipe, ckpt_path, mask_folder=args.mask_folder, output_root=output_subdir,
                                    final_itr=args.final_iterations, relight=args.render_relight, relight_env_path=args.relight_env_path, subset="test")
            else:
                render_and_save_eval(dataset, opt, pipe, ckpt_path, mask_folder=args.mask_folder, output_root=output_subdir,
                                    final_itr=args.final_iterations, relight=args.render_relight, relight_env_path=args.relight_env_path, subset=args.subset)
            # import pdb; pdb.set_trace()
    else:
        # import pdb; pdb.set_trace()
        if args.subset == "both":
            # Render train
            render_and_save_eval(dataset, opt, pipe, ckpt_path, mask_folder=args.mask_folder, output_root=args.output_dir,
                                final_itr=args.final_iterations, relight=args.render_relight, relight_env_path=args.relight_env_path, subset="train")
            # Render test
            render_and_save_eval(dataset, opt, pipe, ckpt_path, mask_folder=args.mask_folder, output_root=args.output_dir,
                                final_itr=args.final_iterations, relight=args.render_relight, relight_env_path=args.relight_env_path, subset="test")
        else:
            render_and_save_eval(dataset, opt, pipe, ckpt_path, mask_folder=args.mask_folder, output_root=args.output_dir,
                                final_itr=args.final_iterations, relight=args.render_relight, relight_env_path=args.relight_env_path, subset=args.subset)