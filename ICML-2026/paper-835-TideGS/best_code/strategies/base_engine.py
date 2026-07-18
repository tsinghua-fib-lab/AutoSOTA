import os
import torch
import math
import plotly.graph_objects as go
import numpy as np
import utils.general_utils as utils
from utils.loss_utils import l1_loss
from clm_kernels import fused_ssim
from gsplat import (
    fully_fused_projection,
    spherical_harmonics,
    isect_tiles,
    isect_offset_encode,
    rasterize_to_pixels,
)

LAMBDA_DSSIM = 0.2  # Loss weight for SSIM
TILE_SIZE = 16

def debug_visualize_scene(xyz_gpu, batched_cameras, output_file="debug_scene.html"):
    """Write an optional projection debug scene.

    This is only called when ``--debug_frustum`` is enabled.  Release runs do
    not generate HTML artifacts or print projection diagnostics by default.
    """

    print(f"[DEBUG] Generating 3D visualization to {output_file}...")
    
    # 1. 提取高斯球点云。先在 GPU 上做轻量采样，再搬到 CPU，避免把整个 working set 拖回主机。
    sampled_xyz = xyz_gpu
    if sampled_xyz.shape[0] > 10000:
        stride = max(1, sampled_xyz.shape[0] // 10000)
        sampled_xyz = sampled_xyz[::stride][:10000]
    points = sampled_xyz.detach().cpu().numpy()
    
    # 2. 提取相机位置和朝向
    cam_centers = []
    cam_directions = [] # 视线方向 (Look-at)
    cam_ids = []
    
    for i, cam in enumerate(batched_cameras):
        if hasattr(cam, 'cam_center'):
            center = cam.cam_center
            if isinstance(center, torch.Tensor):
                center = center.cpu().numpy()
        else:
            w2c = cam.world_view_transform.transpose(0, 1).cpu().numpy()
            R = w2c[:3, :3]
            t = w2c[:3, 3]
            center = -R.T @ t
            
        cam_centers.append(center)
        cam_ids.append(f"Cam {i}")
        
        w2c = cam.world_view_transform.transpose(0, 1).cpu().numpy()
        c2w_rot = w2c[:3, :3].T
        direction = c2w_rot[:, 2]
        direction = direction / np.linalg.norm(direction) * 5.0
        cam_directions.append(direction)

    cam_centers = np.array(cam_centers)
    cam_directions = np.array(cam_directions)

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode='markers',
        marker=dict(size=2, color='blue', opacity=0.3),
        name='Gaussians (Sampled)'
    ))

    fig.add_trace(go.Scatter3d(
        x=cam_centers[:, 0], y=cam_centers[:, 1], z=cam_centers[:, 2],
        mode='markers+text',
        marker=dict(size=5, color='red'),
        text=cam_ids,
        name='Cameras'
    ))

    lines_x = []
    lines_y = []
    lines_z = []
    for i in range(len(cam_centers)):
        start = cam_centers[i]
        end = start + cam_directions[i]
        lines_x.extend([start[0], end[0], None])
        lines_y.extend([start[1], end[1], None])
        lines_z.extend([start[2], end[2], None])
    
    fig.add_trace(go.Scatter3d(
        x=lines_x, y=lines_y, z=lines_z,
        mode='lines',
        line=dict(color='red', width=2),
        name='View Direction'
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
            aspectmode='data'
        ),
        title="Debug: Cameras vs Gaussians"
    )

    fig.write_html(output_file)
    print(f"[DEBUG] Visualization saved to {output_file}. Open it in browser!")


def _unpack_fully_fused_projection_results(proj_results):
    """Handle packed fully_fused_projection outputs across gsplat versions."""
    if len(proj_results) == 8:
        return proj_results
    if len(proj_results) == 7:
        camera_ids, gaussian_ids, radii, means2d, depths, conics, compensations = proj_results
        return None, camera_ids, gaussian_ids, radii, means2d, depths, conics, compensations
    raise RuntimeError(
        f"Unexpected fully_fused_projection(packed=True) return length: {len(proj_results)}"
    )


def _choose_projection_camera_chunk_size(args, num_cameras, num_gaussians):
    forced_chunk = getattr(args, "projection_max_cameras_per_chunk", -1)
    if forced_chunk is not None and forced_chunk > 0:
        return max(1, min(num_cameras, int(forced_chunk)))

    if num_cameras <= 1:
        return num_cameras

    # Keep each packed projection call under a rough camera-Gaussian pair budget
    # so billion-scale working sets do not explode memory or temporary allocations.
    target_pairs_per_chunk = 96_000_000
    auto_chunk = max(1, target_pairs_per_chunk // max(int(num_gaussians), 1))
    return max(1, min(num_cameras, auto_chunk))


def _projection_verbose_enabled(args):
    return bool(
        getattr(args, "debug_frustum", False)
        or getattr(args, "paper_debug_logging", False)
    )


def _maybe_log_projection_contract_debug(
    batch_ids,
    camera_ids,
    gaussian_ids,
    depths,
    batched_cameras,
    xyz_gpu,
    batched_viewmats,
    batched_Ks,
    visible_camera_ids,
):
    args = utils.get_args()
    if not _projection_verbose_enabled(args):
        return

    num_cameras = len(batched_cameras)
    visible_set = set(visible_camera_ids)
    severe = len(visible_set) == 0 or len(visible_set) <= 1 or (num_cameras - len(visible_set)) >= max(1, num_cameras - 1)
    if not severe:
        return

    debug_count = getattr(calculate_filters, '_projection_debug_count', 0)
    if debug_count >= 3:
        return
    calculate_filters._projection_debug_count = debug_count + 1

    unique_batches = []
    if batch_ids is not None and batch_ids.numel() > 0:
        unique_batches = batch_ids.unique().cpu().tolist()
    unique_cameras = []
    if camera_ids.numel() > 0:
        unique_cameras = camera_ids.unique().cpu().tolist()

    if not hasattr(calculate_filters, '_packed_projection_format_logged'):
        calculate_filters._packed_projection_format_logged = True
        print('[PROJECTION DEBUG] packed fully_fused_projection format: '
              'batch_ids, camera_ids, gaussian_ids, radii, means2d, depths, conics, compensations')

    print('[PROJECTION DEBUG] packed fully_fused_projection summary:')
    print(f'  unique batch_ids: {unique_batches[:8]}')
    print(f'  unique camera_ids: {unique_cameras[:16]}' + ('...' if len(unique_cameras) > 16 else ''))
    print(f'  visible cameras: {len(unique_cameras)}/{num_cameras}')

    if gaussian_ids.numel() > 0:
        print(f'  gaussian id range: [{gaussian_ids.min().item()}, {gaussian_ids.max().item()}] / working set {xyz_gpu.shape[0]}')
    else:
        print(f'  gaussian id range: empty / working set {xyz_gpu.shape[0]}')

    if depths is not None and depths.numel() > 0:
        print(f'  depth range: [{depths.min().item():.6f}, {depths.max().item():.6f}]')
    else:
        print('  depth range: empty')

    candidates = []
    if num_cameras > 0:
        candidates.append(('first_camera', 0))
    if visible_camera_ids:
        candidates.append(('first_visible', visible_camera_ids[0]))
    missing_idx = next((i for i in range(num_cameras) if i not in visible_set), None)
    if missing_idx is not None:
        candidates.append(('first_missing', missing_idx))

    seen = set()
    for label, idx in candidates:
        if idx in seen or idx is None:
            continue
        seen.add(idx)
        cam = batched_cameras[idx]
        center = cam.camera_center.detach().cpu().numpy() if hasattr(cam, 'camera_center') else 'N/A'
        viewmat_row_major = batched_viewmats[idx].transpose(0, 1).detach().cpu().numpy()
        fx = batched_Ks[idx, 0, 0].item()
        fy = batched_Ks[idx, 1, 1].item()
        print(f'  {label}: local_idx={idx}, center={center}, fx={fx:.3f}, fy={fy:.3f}')
        print(f'    row-major W2C t={viewmat_row_major[:3, 3]}, forward_col={viewmat_row_major[:3, 2]}')


def calculate_filters(batched_cameras, xyz_gpu, opacity_gpu, scaling_gpu, rotation_gpu):
    """
    对于给定的 batch 内的相机，一次性计算出它们分别能看到哪些高斯球
    得到的filter的信息所有被用来决定：
    1. 哪些数据需要从 CPU 加载到 GPU；
    2. 哪些数据需要Retention；
    3. 哪些梯度需要回传。
    """
    del opacity_gpu  # Opacity is not used by the packed projection kernel but kept for API compatibility.

    args = utils.get_args()
    device = xyz_gpu.device
    num_cameras = len(batched_cameras)

    with torch.no_grad():
        batched_Ks = torch.stack([camera.create_k_on_gpu() for camera in batched_cameras])
        batched_viewmats = torch.stack([camera.world_view_transform for camera in batched_cameras])

        frustum_scene_debug_enabled = bool(getattr(args, "debug_frustum", False))
        projection_verbose_enabled = _projection_verbose_enabled(args)
        if frustum_scene_debug_enabled and not hasattr(calculate_filters, "debug_frustum_culling"):
            print("\n[DEBUG] 🚀 首次运行，正在生成 3D 可视化 scene.html ...")
            debug_output_dir = args.model_path or args.log_folder or "."
            debug_scene_path = os.path.join(debug_output_dir, "debug_scene.html")
            try:
                os.makedirs(debug_output_dir, exist_ok=True)
                debug_visualize_scene(xyz_gpu, batched_cameras, output_file=debug_scene_path)
            except Exception as e:
                print(f"[DEBUG] 生成 3D 可视化 scene.html 失败: {e}")

            viewmat = batched_viewmats[0]
            p_world_centroid = xyz_gpu.mean(dim=0)
            p_h = torch.cat([p_world_centroid, torch.ones(1, device=p_world_centroid.device)])
            p_cam = p_h @ viewmat

            right_axis = viewmat[:3, 0].detach().cpu().numpy()
            down_axis = viewmat[:3, 1].detach().cpu().numpy()
            forward_axis = viewmat[:3, 2].detach().cpu().numpy()

            R_mat = viewmat[:3, :3]
            T_vec = viewmat[3, :3]
            cam_center = -(R_mat.t() @ T_vec).detach().cpu().numpy()

            p_world_np = p_world_centroid.detach().cpu().numpy()
            vec_cam_to_point = p_world_np - cam_center
            dist = np.linalg.norm(vec_cam_to_point)
            if dist > 0:
                vec_norm = vec_cam_to_point / dist
                dot_x = np.dot(vec_norm, right_axis)
                dot_y = np.dot(vec_norm, down_axis)
                dot_z = np.dot(vec_norm, forward_axis)
            else:
                dot_x = dot_y = dot_z = 0.0

            print(f"  Camera Center: {cam_center}")
            print(f"  Working-set Centroid: {p_world_np}")
            print(f"  Vector (Cam->Centroid): {vec_cam_to_point} (Dist: {dist:.2f})")
            print("-" * 30)
            print(f"  Projection on Right Axis (X):   {dot_x:.4f}")
            print(f"  Projection on Down Axis  (Y):   {dot_y:.4f}")
            print(f"  Projection on Forward Axis (Z): {dot_z:.4f}")
            print("-" * 30)
            print("  [DEBUG] 说明: 这里检查的是整个 working set 的质心，而不是当前相机确认可见的点。")
            print("  [DEBUG] 如果质心落在画面外，或更靠近 X/Y 轴，这并不等价于相机姿态错误。")

            z_val = p_cam[2].item()
            if z_val < 0:
                print(f"  [DEBUG] Centroid camera-space Z = {z_val:.4f} < 0，说明 working-set 质心在当前相机后方。")
            elif z_val < 0.2:
                print(f"  [DEBUG] Centroid camera-space Z = {z_val:.4f}，质心离当前相机很近。")
            else:
                print(f"  [DEBUG] Centroid camera-space Z = {z_val:.4f}。")

            if z_val != 0:
                fx = batched_Ks[0, 0, 0].item()
                fy = batched_Ks[0, 1, 1].item()
                cx = batched_Ks[0, 0, 2].item()
                cy = batched_Ks[0, 1, 2].item()
                u = (p_cam[0] / z_val) * fx + cx
                v = (p_cam[1] / z_val) * fy + cy
                print(
                    f"  Projected centroid pixel: ({u:.2f}, {v:.2f}) / "
                    f"({int(utils.get_img_width())}, {int(utils.get_img_height())})"
                )
                print("  [DEBUG] 如果这个投影越界，只能说明 working-set 质心不在该相机画面中，不能单独用于判断相机朝向。")

            calculate_filters.debug_frustum_culling = True

        camera_chunk_size = _choose_projection_camera_chunk_size(args, num_cameras, xyz_gpu.shape[0])
        if camera_chunk_size < num_cameras:
            last_signature = getattr(calculate_filters, "_projection_chunk_signature", None)
            new_signature = (num_cameras, camera_chunk_size)
            if projection_verbose_enabled and last_signature != new_signature:
                print(
                    f"[PROJECTION] Camera chunking enabled: {num_cameras} cameras -> chunks of {camera_chunk_size} "
                    f"for working set {xyz_gpu.shape[0]:,}"
                )
                calculate_filters._projection_chunk_signature = new_signature

        packed_batch_ids = []
        packed_camera_ids = []
        packed_gaussian_ids = []
        packed_depths = []
        returns_batch_ids = None

        for chunk_start in range(0, num_cameras, camera_chunk_size):
            chunk_end = min(num_cameras, chunk_start + camera_chunk_size)
            proj_results = fully_fused_projection(
                means=xyz_gpu,
                covars=None,
                quats=rotation_gpu,
                scales=scaling_gpu,
                viewmats=batched_viewmats[chunk_start:chunk_end].transpose(1, 2).contiguous(),
                Ks=batched_Ks[chunk_start:chunk_end],
                radius_clip=args.radius_clip,
                width=int(utils.get_img_width()),
                height=int(utils.get_img_height()),
                packed=True,
            )

            (
                batch_ids_chunk,
                camera_ids_chunk,
                gaussian_ids_chunk,
                _,
                _,
                depths_chunk,
                _,
                _,
            ) = _unpack_fully_fused_projection_results(proj_results)

            if returns_batch_ids is None:
                returns_batch_ids = batch_ids_chunk is not None

            if camera_ids_chunk.numel() == 0:
                continue

            packed_camera_ids.append(camera_ids_chunk + chunk_start)
            packed_gaussian_ids.append(gaussian_ids_chunk)
            packed_depths.append(depths_chunk)
            if returns_batch_ids:
                packed_batch_ids.append(batch_ids_chunk + chunk_start)

        if packed_camera_ids:
            camera_ids = torch.cat(packed_camera_ids, dim=0)
            gaussian_ids = torch.cat(packed_gaussian_ids, dim=0)
            depths_packed = torch.cat(packed_depths, dim=0)
            batch_ids = torch.cat(packed_batch_ids, dim=0) if returns_batch_ids else None
        else:
            empty_long = torch.empty(0, dtype=torch.int64, device=device)
            camera_ids = empty_long
            gaussian_ids = empty_long
            depths_packed = torch.empty(0, dtype=torch.float32, device=device)
            batch_ids = empty_long if returns_batch_ids else None

        if camera_ids.numel() > 0:
            assert camera_ids.min() >= 0 and camera_ids.max() < num_cameras, (
                f"camera_ids out of range: min={camera_ids.min().item()}, max={camera_ids.max().item()}, "
                f"num_cameras={num_cameras}"
            )
        if gaussian_ids.numel() > 0:
            assert gaussian_ids.min() >= 0 and gaussian_ids.max() < xyz_gpu.shape[0], (
                f"gaussian_ids out of range: min={gaussian_ids.min().item()}, max={gaussian_ids.max().item()}, "
                f"working_set={xyz_gpu.shape[0]}"
            )

        output, counts = torch.unique_consecutive(camera_ids, return_counts=True)

        if output.numel() == 0:
            gaussian_ids_per_camera = tuple(
                torch.empty(0, dtype=torch.int64, device=device)
                for _ in range(num_cameras)
            )
            _maybe_log_projection_contract_debug(
                batch_ids=batch_ids,
                camera_ids=camera_ids,
                gaussian_ids=gaussian_ids,
                depths=depths_packed,
                batched_cameras=batched_cameras,
                xyz_gpu=xyz_gpu,
                batched_viewmats=batched_viewmats,
                batched_Ks=batched_Ks,
                visible_camera_ids=[],
            )
        elif output.numel() == num_cameras and torch.all(output == torch.arange(num_cameras, device=device)):
            counts_cpu = counts.cpu().numpy().tolist()
            assert sum(counts_cpu) == gaussian_ids.shape[0], (
                "sum(counts_cpu) is supposed to be equal to gaussian_ids.shape[0]"
            )
            gaussian_ids_per_camera = torch.split(gaussian_ids, counts_cpu)
        else:
            counts_cpu = counts.cpu().numpy().tolist()
            output_cpu = output.cpu().numpy().tolist()
            gaussian_ids_split = torch.split(gaussian_ids, counts_cpu)

            gaussian_ids_per_camera = []
            split_idx = 0
            for cam_id in range(num_cameras):
                if split_idx < len(output_cpu) and output_cpu[split_idx] == cam_id:
                    gaussian_ids_per_camera.append(gaussian_ids_split[split_idx])
                    split_idx += 1
                else:
                    gaussian_ids_per_camera.append(torch.empty(0, dtype=torch.int64, device=device))
            gaussian_ids_per_camera = tuple(gaussian_ids_per_camera)

            missing_cameras = set(range(num_cameras)) - set(output_cpu)
            if len(missing_cameras) <= 10:
                warning_msg = f"[WARNING] {len(missing_cameras)}/{num_cameras} cameras see no Gaussians: {sorted(missing_cameras)}"
            else:
                warning_msg = (
                    f"[WARNING] {len(missing_cameras)}/{num_cameras} cameras see no Gaussians: "
                    f"{sorted(missing_cameras)[:10]}... (truncated)"
                )
            print(warning_msg)
            log_file = utils.get_log_file()
            if log_file is not None:
                log_file.write(warning_msg + "\n")
                log_file.flush()

            if projection_verbose_enabled and len(missing_cameras) > 0:
                first_missing = sorted(missing_cameras)[0]
                cam = batched_cameras[first_missing]
                cam_pos = cam.camera_center.cpu().numpy() if hasattr(cam, 'camera_center') else 'N/A'
                print(f"  [DEBUG] First missing camera {first_missing}: position={cam_pos}")
                print(f"  [DEBUG] Total Gaussians in working set: {xyz_gpu.shape[0]}")
                print(f"  [DEBUG] Cameras with visible Gaussians: {len(output_cpu)}")

            _maybe_log_projection_contract_debug(
                batch_ids=batch_ids,
                camera_ids=camera_ids,
                gaussian_ids=gaussian_ids,
                depths=depths_packed,
                batched_cameras=batched_cameras,
                xyz_gpu=xyz_gpu,
                batched_viewmats=batched_viewmats,
                batched_Ks=batched_Ks,
                visible_camera_ids=output_cpu,
            )

    return gaussian_ids_per_camera, camera_ids, gaussian_ids


@torch.compile
def loss_combined(image, image_gt, ssim_loss):
    LAMBDA_DSSIM = 0.2  # TODO: allow this to be set by the user
    Ll1 = l1_loss(image, image_gt)
    loss = (1.0 - LAMBDA_DSSIM) * Ll1 + LAMBDA_DSSIM * (1.0 - ssim_loss)
    return loss


class FusedCompiledLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image, image_gt_original):
        image_gt = torch.clamp(image_gt_original / 255.0, 0.0, 1.0)
        ssim_loss = fused_ssim(image.unsqueeze(0), image_gt.unsqueeze(0))
        return loss_combined(image, image_gt, ssim_loss)


FUSED_COMPILED_LOSS_MODULE = FusedCompiledLoss()


def torch_compiled_loss(image, image_gt_original):
    global FUSED_COMPILED_LOSS_MODULE
    loss = FUSED_COMPILED_LOSS_MODULE(image, image_gt_original)
    return loss


def pipeline_forward_one_step(
    filtered_opacity_gpu,
    filtered_scaling_gpu,
    filtered_rotation_gpu,
    filtered_xyz_gpu,
    filtered_shs,
    camera,
    scene,
    gaussians,
    background,
    pipe_args,
    eval=False,
):
    MICRO_BATCH_SIZE = 1  # NOTE: microbatch here only contains one camera.
    image_width = int(utils.get_img_width())
    image_height = int(utils.get_img_height())
    tanfovx = math.tan(camera.FoVx * 0.5)
    tanfovy = math.tan(camera.FoVy * 0.5)
    focal_length_x = image_width / (2 * tanfovx)
    focal_length_y = image_height / (2 * tanfovy)
    K = torch.tensor(
        [
            [focal_length_x, 0, image_width / 2.0],
            [0, focal_length_y, image_height / 2.0],
            [0, 0, 1],
        ],
        device="cuda",
    )
    assert K.shape == (3, 3)

    viewmat = camera.world_view_transform.transpose(0, 1)  # why transpose
    n_selected = filtered_xyz_gpu.shape[0]

    batched_radiis, batched_means2D, batched_depths, batched_conics, _ = (
        fully_fused_projection(
            means=filtered_xyz_gpu,  # (N, 3)
            covars=None,
            quats=filtered_rotation_gpu,
            scales=filtered_scaling_gpu,
            viewmats=viewmat.unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=image_width,
            height=image_height,
            packed=False,
        )
    )  # (1, N), (1, N, 2), (1, N), (1, N, 3), (1, N)

    if not eval:
        batched_means2D.retain_grad()  # this is only for training.

    sh_degree = gaussians.active_sh_degree
    # camtoworlds = torch.inverse(viewmat.unsqueeze(0)) # (4, 4)
    camtoworlds = torch.inverse(viewmat.unsqueeze(0))
    dirs = filtered_xyz_gpu[None, :, :] - camtoworlds[:, None, :3, 3]
    filtered_shs = filtered_shs.reshape(1, n_selected, 16, 3)
    batched_colors = spherical_harmonics(
        degrees_to_use=sh_degree, dirs=dirs, coeffs=filtered_shs
    )
    batched_colors = torch.clamp_min(batched_colors + 0.5, 0.0)  # (1, N, 3)
    batched_opacities = filtered_opacity_gpu.squeeze(1).unsqueeze(0)  # (N, 1) -> (1, N)

    # NOTE: In the above code, we keep the first batch dimension, even if it is always 1.

    # render
    # Identify intersecting tiles.
    tile_width = math.ceil(image_width / float(TILE_SIZE))
    tile_height = math.ceil(image_height / float(TILE_SIZE))

    # flatten_ids: (C*N)
    _, isect_ids, flatten_ids = isect_tiles(
        means2d=batched_means2D,
        radii=batched_radiis,
        depths=batched_depths,
        tile_size=TILE_SIZE,
        tile_width=tile_width,
        tile_height=tile_height,
        packed=False,
    )
    isect_offsets = isect_offset_encode(
        isect_ids, MICRO_BATCH_SIZE, tile_width, tile_height
    )  # (MICRO_BATCH_SIZE, tile_height, tile_width)

    # Rasterize to pixels. batched_rendered_image: (MICRO_BATCH_SIZE, image_height, image_width, 3)
    backgrounds = (
        background.repeat(MICRO_BATCH_SIZE, 1) if background is not None else None
    )
    rendered_image, _ = rasterize_to_pixels(
        means2d=batched_means2D,
        conics=batched_conics,
        colors=batched_colors,
        opacities=batched_opacities,
        image_width=image_width,
        image_height=image_height,
        tile_size=TILE_SIZE,
        isect_offsets=isect_offsets,
        flatten_ids=flatten_ids,
        backgrounds=backgrounds,
    )

    rendered_image = rendered_image.squeeze(0).permute(2, 0, 1).contiguous()

    return rendered_image, batched_means2D, batched_radiis
