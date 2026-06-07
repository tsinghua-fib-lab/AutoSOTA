import os
import re
import argparse
import numpy as np
import torch

from typing import List, Tuple, Dict

import torch.nn.functional as F
from torchvision.utils import save_image

# Extend path to import project modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.graphics_utils import linear_to_srgb, getWorld2View2, getProjectionMatrix, getProjectionMatrixCorrect
from gaussian_renderer import render_surfel
from utils.gridmap_envmap_utils import build_cubemap, build_surf_cubemap

# ----------------- Helpers -----------------

def parse_idx_list(s: str):
    parts = re.split(r'[; ,\t\n]+', s.strip())
    idxs = [int(p) for p in parts if p]
    if len(idxs) < 3:
        raise ValueError('需要至少 3 个相机索引用于拟合圆。')
    return sorted(idxs)

def set_gaussian_para(gaussians, opt, vol=False):
    gaussians.enlarge_scale = opt.enlarge_scale
    gaussians.rough_msk_thr = opt.rough_msk_thr
    gaussians.init_roughness_value = opt.init_roughness_value
    gaussians.init_refl_value = opt.init_refl_value
    gaussians.refl_msk_thr = opt.refl_msk_thr

def select_cameras_by_indices(cameras, indices):
    sel = []
    for idx in indices:
        found = None
        for cam_idx, cam in enumerate(cameras):
            if cam_idx == idx:
                found = cam; break
        if found is None:
            raise ValueError(f'找不到相机 idx={idx}')
        sel.append(found)
    return sel

def fit_circle_3d(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    mean = points.mean(axis=0)
    P = points - mean
    cov = P.T @ P / P.shape[0]
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    u_axis = eigvecs[:, 0]
    v_axis = eigvecs[:, 1]
    normal = eigvecs[:, 2]
    xy = np.stack([P @ u_axis, P @ v_axis], axis=1)
    x = xy[:, 0]
    y = xy[:, 1]
    A = np.stack([x, y, np.ones_like(x)], axis=1)
    b = -(x**2 + y**2)
    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    D, E, F = sol
    center2D = np.array([-D / 2.0, -E / 2.0])
    radius = np.sqrt(center2D[0] ** 2 + center2D[1] ** 2 - F)
    center3D = mean + center2D[0] * u_axis + center2D[1] * v_axis
    v_axis = np.cross(normal, u_axis)
    v_axis /= np.linalg.norm(v_axis) + 1e-8
    return center3D, normal, radius, u_axis, v_axis


def camera_forward_dir(cam) -> np.ndarray:
    wvt = cam.world_view_transform.T.cpu().numpy()  # [4,4] W2C
    R_w2c = wvt[:3, :3]
    z_cam_world = R_w2c.T[:, 2]
    forward = z_cam_world  # OpenGL: camera looks along -Z
    return forward / (np.linalg.norm(forward) + 1e-9)


def intersect_rays_least_squares(cams: List) -> Tuple[np.ndarray, bool]:
    I = np.eye(3)
    A = np.zeros((3, 3))
    b = np.zeros(3)
    ok = True
    for cam in cams:
        p = cam.camera_center.cpu().numpy()
        d = camera_forward_dir(cam)
        Q = I - np.outer(d, d)
        A += Q
        b += Q @ p
    try:
        x = np.linalg.lstsq(A, b, rcond=None)[0]
    except Exception:
        ok = False
        x = None
    if ok:
        try:
            cond = np.linalg.cond(A)
            if not np.isfinite(cond) or cond > 1e12:
                ok = False
        except Exception:
            ok = False
    return (x, ok)


def closest_points_two_lines(p0: np.ndarray, v0: np.ndarray, c: np.ndarray, n: np.ndarray) -> np.ndarray:
    v0 = v0 / (np.linalg.norm(v0) + 1e-9)
    n = n / (np.linalg.norm(n) + 1e-9)
    r = p0 - c
    a = v0 @ v0
    b = v0 @ n
    d = n @ n
    rhs1 = -v0 @ r
    rhs2 = n @ r
    M = np.array([[a, -b], [b, d]], dtype=np.float64)
    rhs = np.array([rhs1, rhs2], dtype=np.float64)
    try:
        t, s = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        s = (r @ n)
        t = 0.0
    p_on_0 = p0 + t * v0
    p_on_1 = c + s * n
    return 0.5 * (p_on_0 + p_on_1)

# ----------------- Main Script -----------------

# python scripts/render_orbit_rotation.py \
#   --ckpt output_2x/squirrel/squirrel_stokes_1_obj_env/point_cloud/iteration_30000/point_cloud.ply \
#   --idx_list "0,12,24,36" \
#   --final_itr 30000 \
#   --angle_step_deg 5 \
#   --output_dir orbit_out

def main():
    parser = argparse.ArgumentParser(description='根据若干相机索引拟合圆并每隔固定角度进行轨道旋转渲染')
    parser.add_argument('--ckpt', type=str, required=True, help='训练检查点路径 (包含 point_cloud 等)')
    parser.add_argument('--idx_list', type=str, required=True, help='用于拟合圆的相机索引列表，例如 "0,5,10,15"')
    parser.add_argument('--final_itr', type=int, default=20000, help='加载点云的迭代号')
    parser.add_argument('--angle_step_deg', type=float, default=5.0, help='采样角度步长（度）')
    parser.add_argument('--output_dir', type=str, default='orbit_out', help='输出目录')
    parser.add_argument('--save_mask', action='store_true', help='保存 mask/opacity 图像')
    parser.add_argument('--transparent_bg', action='store_true', help='输出带 alpha 通道的透明背景图像 (而不是白/黑背景)')
    parser.add_argument('--target_height_ratio', type=float, default=1.0, help='目标点相对圆半径在法线方向的高度比例，target = center + ratio*radius*normal')
    parser.add_argument('--reverse_rotation', action='store_true', help='反转旋转方向')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 读取 cfg_args
    ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt))
    cfg_path = os.path.join(ckpt_dir, 'cfg_args')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f'cfg_args 不存在: {cfg_path}')
    with open(cfg_path, 'r') as f:
        cfg_ns = eval(f.read(), {'__builtins__': {}}, {'Namespace': argparse.Namespace})

    # 提取参数
    lp = ModelParams(argparse.ArgumentParser())
    op = OptimizationParams(argparse.ArgumentParser())
    pp = PipelineParams(argparse.ArgumentParser())
    dataset = lp.extract(cfg_ns)
    opt = op.extract(cfg_ns)
    pipe = pp.extract(cfg_ns)
    dataset.envmap_max_res = 64

    # 构建场景与高斯
    gaussians = GaussianModel(dataset.albedo_sh_degree, dataset.sh_degree)
    set_gaussian_para(gaussians, opt, vol=(opt.volume_render_until_iter > opt.init_until_iter))
    scene = Scene(dataset, gaussians, shuffle = False)
    gaussians.load_ply(os.path.join(ckpt_dir, f'point_cloud/iteration_{args.final_itr}/point_cloud.ply'), relight=False, args=dataset)
    if args.final_itr >= opt.indirect_from_iter:
        opt.indirect = 1
        gaussians.load_mesh_from_ply(ckpt_dir, args.final_itr)

    cameras_all = scene.getTrainCameras() + scene.getTestCameras()

    # 挑选用于拟合圆的相机
    fit_indices = parse_idx_list(args.idx_list)
    circle_cams = select_cameras_by_indices(cameras_all, fit_indices)
    centers = np.stack([cam.camera_center.cpu().numpy() for cam in circle_cams], axis=0)
    center3D, plane_n, radius, u_axis, v_axis = fit_circle_3d(centers)
    first_cam = circle_cams[0]
    up_vec = plane_n
    # 目标点：由视线与圆心法线的最近点估计（若不相交则取最近点的中点，聚合多相机取均值）
    def point_angle(p):
        rel = p - center3D
        x = rel @ u_axis
        y = rel @ v_axis
        return np.arctan2(y, x)
    start_angle = point_angle(centers[0])

    # Common target from viewing rays with fallback
    target_pt, ok = intersect_rays_least_squares(circle_cams)
    if not ok or target_pt is None:
        p0 = first_cam.camera_center.cpu().numpy()
        v0 = camera_forward_dir(first_cam)
        target_pt = closest_points_two_lines(p0, v0, center3D, plane_n)

    # 采样角度列表
    angle_step = np.deg2rad(args.angle_step_deg)
    num_samples = int(np.ceil(2 * np.pi / angle_step))
    angles = start_angle + np.arange(num_samples) * angle_step
    angles = np.mod(angles - start_angle, 2 * np.pi) + start_angle  # 归一化范围

    # Determine background color
    if args.transparent_bg:
        # For transparent background, always use black/zero background
        bg_color = [0, 0, 0]
    else:
        # Use dataset's background setting
        bg_color = [1,1,1] if dataset.white_background else [0,0,0]
    background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')

    if args.reverse_rotation:
        angles = angles[::-1]

    # 构建 gridmap envmap（与 render_for_eval 保持一致）
    if pipe.indirect_type in ['obj_env', 'surf_env']:
        gaussians.reset_gridmap_envmap()
        if pipe.indirect_type == 'obj_env':
            gridmap_cubemaps = build_cubemap(cameras_all, gaussians, pipe, background, dataset, opt)
        else:
            gridmap_cubemaps = build_surf_cubemap(cameras_all, gaussians, pipe, background, dataset, opt)
        gaussians.update_gridmap_envmap(gridmap_cubemaps, dataset)

    # 渲染循环
    for i, ang in enumerate(angles):
        cam_center = center3D + radius * np.cos(ang) * u_axis + radius * np.sin(ang) * v_axis
        # Look-at basis
        f = (target_pt - cam_center); f /= (np.linalg.norm(f) + 1e-9)
        up_n = up_vec / (np.linalg.norm(up_vec) + 1e-9)
        up_n = up_n if (up_n@f)<0 else -up_n
        s = np.cross(f, up_n); s /= (np.linalg.norm(s) + 1e-9)
        u = -np.cross(s, f)
        # world->camera rotation rows (OpenGL -Z forward)
        R_wc = np.stack([s, u, f], axis=0)
        t_wc = -R_wc @ cam_center
        R_wc = R_wc.T
        Rt = getWorld2View2(R_wc, t_wc, translate=first_cam.trans, scale=first_cam.scale)
        # 使用第一个拟合相机的投影参数
        proto = circle_cams[0]
        Rt = getWorld2View2(R_wc, t_wc, translate=proto.trans, scale=proto.scale)
        world_view_transform = torch.tensor(Rt).transpose(0,1).float().cuda()
        zfar = 100.0
        znear = 0.01
        HWK = proto.HWK
        projection_matrix = getProjectionMatrixCorrect(znear, zfar, HWK[0], HWK[1], HWK[2]).transpose(0,1).cuda()
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center_tensor = world_view_transform.inverse()[3,:3]
        # 克隆并覆盖参数
        cam_new = proto
        cam_new.R = torch.tensor(R_wc).float().cuda()
        cam_new.T = torch.tensor(t_wc).float().cuda()
        cam_new.world_view_transform = world_view_transform
        cam_new.projection_matrix = projection_matrix
        cam_new.full_proj_transform = full_proj_transform
        cam_new.camera_center = camera_center_tensor
        cam_new.image_name = f'orbit_{i:04d}'
        cam_new.uid = i

        render_pkg = render_surfel(cam_new, gaussians, pipe, background, opt=opt)
        img = render_pkg['render']
        albedo = render_pkg['base_color_map'].clamp(0,1)
        specular = render_pkg['specular_map']
        diffuse = render_pkg['diffuse_map']
        mask = render_pkg['rend_alpha']

        # Create RGBA images (RGB + alpha channel for opacity)
        img_rgba = torch.cat([linear_to_srgb(img), mask], dim=0)
        albedo_rgba = torch.cat([linear_to_srgb(albedo), mask], dim=0)
        specular_rgba = torch.cat([linear_to_srgb(specular), mask], dim=0)
        diffuse_rgba = torch.cat([linear_to_srgb(diffuse), mask], dim=0)

        base_name = f'orbit_idx_start{fit_indices[0]}_{i:04d}'
        render_path = os.path.join(args.output_dir, "render")
        albedo_path = os.path.join(args.output_dir, "albedo")
        diffuse_path = os.path.join(args.output_dir, "diffuse")
        specular_path = os.path.join(args.output_dir, "specular")
        normal_path = os.path.join(args.output_dir, "normal")
        depth_normal_path = os.path.join(args.output_dir, "depth_normal")
        mask_path = os.path.join(args.output_dir, "mask")
        os.makedirs(render_path, exist_ok=True)
        os.makedirs(albedo_path, exist_ok=True)
        os.makedirs(diffuse_path, exist_ok=True)
        os.makedirs(specular_path, exist_ok=True)
        os.makedirs(normal_path, exist_ok=True)
        os.makedirs(depth_normal_path, exist_ok=True)
        if args.save_mask:
            os.makedirs(mask_path, exist_ok=True)
        save_image(img_rgba, os.path.join(render_path, base_name + '.png'))
        save_image(albedo_rgba, os.path.join(albedo_path, base_name + '.png'))
        save_image(diffuse_rgba, os.path.join(diffuse_path, base_name + '.png'))
        save_image(specular_rgba, os.path.join(specular_path, base_name + '.png'))

        if args.save_mask:
            # Save mask as grayscale (H x W -> 1 x H x W)
            mask_vis = mask.clamp(0, 1)
            save_image(mask_vis, os.path.join(mask_path, base_name + '.png'))

        if 'rend_normal' in render_pkg:
            normal_map = render_pkg['rend_normal']
            normal_map_vis = (normal_map * 0.5 + 0.5).clamp(0, 1)
            normal_map_vis = torch.cat([normal_map_vis, mask], dim=0)
            save_image(normal_map_vis, os.path.join(normal_path, base_name + '.png'))
        if 'surf_normal' in render_pkg:
            dep_normal_map = render_pkg['surf_normal']
            dep_normal_map_vis = (dep_normal_map * 0.5 + 0.5).clamp(0, 1)
            dep_normal_map_vis = torch.cat([dep_normal_map_vis, mask], dim=0)
            save_image(dep_normal_map_vis, os.path.join(depth_normal_path, base_name + '.png'))

        print(f'Saved orbit sample {i}/{len(angles)} -> {base_name}')

if __name__ == '__main__':
    with torch.no_grad():
        main()
