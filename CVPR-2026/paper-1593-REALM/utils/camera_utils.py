#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch, save_tensor_as_png
from utils.graphics_utils import fov2focal
import torch
from gaussian_renderer import render, render_target, render_dpt, render_mask
import os
from torchvision.transforms.functional import to_pil_image, to_tensor

WARNED = False
import random
# import numpy as np
from sklearn.cluster import KMeans
# import torch
from scene.cameras import Camera
from typing import List
from reasoneditor.id_utils import extract_selected_obj_ids
from tqdm import tqdm
from utils.reason_utils import tensor_to_pil
from utils.loss_utils import l1_loss
import pdb
# from lama_inpaint import inpaint_img_with_lama



def filter_views_by_diversity(
    views, gaussians, pipeline, background,
    classifier, min_classes=3, thr_ratio=0.005
):
    """Return list of view indices that pass the diversity test."""
    kept_idx = []
    for idx, cam in enumerate(views):
        out_pkg = render(cam, gaussians, pipeline, background)
        pred    = torch.argmax(classifier(out_pkg["render_object"]), dim=0)  # [H,W]

        # --- diversity metric ---------------------------------------------
        counts = torch.bincount(pred.flatten())
        ratio  = counts / counts.sum()
        num_classes = (ratio > thr_ratio).sum().item()
        # -------------------------------------------------------------------

        if num_classes >= min_classes:
            kept_idx.append(idx)
    return kept_idx


def sample_reason_cameras_diverse_clustered(
    views, gaussians, pipeline, background,
    classifier,
    n_clusters=24, min_classes=20, thr_ratio=0.005, seed=0
):
    # 1) pre-filter
    good_ids = filter_views_by_diversity(
        views, gaussians, pipeline, background,
        classifier, min_classes, thr_ratio
    )
    if len(good_ids) == 0:             # fallback: use all
        good_ids = list(range(len(views)))

    good_views = [views[i] for i in good_ids]

    # 2) run K-means on camera centers of filtered views
    n_clusters = min(n_clusters, len(good_views))
    centers = np.stack([v.camera_center.cpu().numpy() for v in good_views])
    kmeans  = KMeans(n_clusters=n_clusters, random_state=seed).fit(centers)
    labels  = kmeans.labels_
    cent    = kmeans.cluster_centers_

    rep_idx = []
    for k in range(n_clusters):
        inds   = np.where(labels == k)[0]
        if len(inds) == 0:
            continue
        dists  = np.linalg.norm(centers[inds] - cent[k], axis=1)
        rep_id = inds[np.argmin(dists)]
        rep_idx.append(int(rep_id))

    # map back to original list indices
    return [good_views[i] for i in sorted(rep_idx)]

def sample_reason_cameras(views, num_views):
    edit_cam_num = num_views
    total_view_num = len(views)
    # random.seed(0)
    
    view_index = random.sample(
        range(0, total_view_num),
        min(total_view_num, edit_cam_num),
    )
    print(view_index)
    # edit_cameras = [views[26]]
    
    edit_cameras = [views[idx] for idx in view_index]
    return edit_cameras

def sample_reason_cameras_top3(
    views: List[Camera],
    gaussians,
    pipeline,
    background,
    classifier,
    top_k: int = 5,
):
    """
    选取对象类别（pred_obj.unique()）数最多的 top_k 个视角。

    Args
    ----
    views       : List[Camera]         所有候选相机
    gaussians   : Gaussian scene data  传给 render
    pipeline    : Rendering pipeline   传给 render
    background  : Background tensor    传给 render
    classifier  : 语义分类器           接收 out_pkg["render_object"]
    top_k       : 需要保留的视角数量

    Returns
    -------
    rep_views   : List[Camera]         类别最丰富的 top_k 视角
    """
    diversity_list = []  # (num_classes, view_idx)

    for idx, cam in enumerate(views):
        # 渲染该视角
        out_pkg = render(cam, gaussians, pipeline, background)
        obj_feat = out_pkg["render_object"]          # [C, H, W]
        pred_obj = torch.argmax(classifier(obj_feat), dim=0)  # [H, W]

        # 统计不同类别数量
        num_classes = pred_obj.unique().numel()
        diversity_list.append((num_classes, idx))

    # 依据类别数降序排序
    diversity_list.sort(key=lambda x: x[0], reverse=True)

    # 取前 top_k
    top_indices = [idx for _, idx in diversity_list[:top_k]]
    rep_views   = [views[i] for i in top_indices]

    return rep_views

def _kmeans_representatives(
    views: List[Camera],
    n_cluster: int = 24,
    seed: int = 0,
) -> List[Camera]:
    n_cluster = min(n_cluster, len(views))
    centers   = np.stack([v.camera_center.cpu().numpy() for v in views])  # (N,3)

    # kmeans = KMeans(n_clusters=n_cluster, n_init=10)
    kmeans = KMeans(n_clusters=n_cluster, random_state=seed, n_init=10)
    kmeans.fit(centers)
    labels    = kmeans.labels_
    centroids = kmeans.cluster_centers_

    rep_idx = []
    for k in range(n_cluster):
        idxs   = np.where(labels == k)[0]
        if len(idxs) == 0:
            continue
        dists  = np.linalg.norm(centers[idxs] - centroids[k], axis=1)
        rep_id = idxs[np.argmin(dists)]
        rep_idx.append(int(rep_id))

    rep_idx.sort()
    return [views[i] for i in rep_idx]

# ------------------------------------------------------------
# Stage-2: among reps, pick views with most object classes
# ------------------------------------------------------------
def _select_topk_by_semantic_diversity(
    views: List[Camera],
    gaussians, pipeline, background, classifier,
    top_k: int = 3,
) -> List[Camera]:
    diversity = []
    for idx, cam in enumerate(views):
        out_pkg   = render(cam, gaussians, pipeline, background)
        obj_feat  = out_pkg["render_object"]
        pred_obj  = torch.argmax(classifier(obj_feat), dim=0)  # [H,W]
        num_cls   = pred_obj.unique().numel()
        diversity.append((num_cls, idx))

    diversity.sort(key=lambda x: x[0], reverse=True)
    chosen_idx = [idx for _, idx in diversity[:min(top_k, len(diversity))]]
    return [views[i] for i in chosen_idx]

def filter_views_by_obj_id(
    views: List[Camera],
    gaussians,
    pipeline,
    background,
    classifier,
    target_id: int,
    top_k: int = 8,
) -> list[Camera]:
    """
    From `views`, keep the top_k cameras whose rendered images
    contain the largest number of pixels with label == target_id.
    """
    stats = []          # list of (pixel_count, idx)

    for idx, cam in enumerate(views):
        out_pkg  = render(cam, gaussians, pipeline, background)
        pred_obj = torch.argmax(classifier(out_pkg["render_object"]), dim=0)

        # count target pixels
        pix_cnt = (pred_obj == target_id).sum().item()
        if pix_cnt > 0:                         # keep only views that contain the id
            stats.append((pix_cnt, idx))

    if not stats:                               # none contains target_id
        return []

    # sort by descending pixel count
    stats.sort(key=lambda x: x[0], reverse=True)

    # take top_k indices
    chosen_idx = [idx for _, idx in stats[:min(top_k, len(stats))]]
    return [views[i] for i in chosen_idx]



# ------------------------------------------------------------
# Combined sampler
# ------------------------------------------------------------
def sample_reason_cameras_cluster_then_topk(
    views: List[Camera],
    gaussians, pipeline, background, classifier,
    n_cluster: int = 24,
    top_k: int = 8,
    # seed: int = 20040813,
    # seed: int = 5731,
    seed: int = 0,
) -> List[Camera]:
    """1) cluster on pose → 24 reps; 2) keep the 3 richest in classes."""
    reps  = _kmeans_representatives(views, n_cluster=n_cluster, seed=seed)
    bests = _select_topk_by_semantic_diversity(
        reps, gaussians, pipeline, background, classifier, top_k=top_k
    )
    return bests
    
# def sample_reason_cameras_cluster_then_topk(
#     views: List[Camera],
#     gaussians, pipeline, background, classifier,
#     n_cluster: int = 24,
#     top_k: int = 4,
#     seed: int = 0,
# ) -> List[Camera]:
#     """1) cluster on pose → 24 reps; 2) randomly select 4 representatives."""
#     reps = _kmeans_representatives(views, n_cluster=n_cluster, seed=seed)
    
#     # 随机选取 top_k 个视角
#     random.seed(seed)
#     selected_reps = random.sample(reps, min(top_k, len(reps)))
    
#     return selected_reps

def sample_reason_cameras_cluster_then_find_id(
    views: List[Camera],
    gaussians, pipeline, background, classifier,
    target_id: int,
    n_cluster: int = 24,
    seed: int = 0,
) -> list[Camera]:
    """
    1) 先做 K-means 聚类得到代表视角 reps
    2) 再在 reps 中筛出包含 target_id 的视角
    """
    reps = _kmeans_representatives(views, n_cluster=n_cluster, seed=seed)
    # id_views = filter_views_by_obj_id(
    #     reps, gaussians, pipeline, background, classifier, target_id
    # )
    # 如果只想要一张，可取 id_views[0]
    return reps
def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
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
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  objects=torch.from_numpy(np.array(cam_info.objects)))

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry


def select_local(args, reason_model, sam_predictor, clip_model, clip_preprocess, views, gaussians, pipeline, background, classifier, best_id):
    refine_cameras = sample_reason_cameras_cluster_then_find_id(views, gaussians, pipeline, background, classifier, best_id)
    # refine_cameras = sample_reason_cameras_cluster_then_find_id(views, gaussians, pipeline, background, classifier, best_id)
    selected_views = []   # 存代表性 Camera
    selected_masks = []   # 存对应 mask (torch.bool [H,W])
    with torch.no_grad():
        for idx, view in enumerate(refine_cameras):
            out_pkg = render(view, gaussians, pipeline, background)
            out          = out_pkg["render"]
            obj_feature  = out_pkg["render_object"]
            pred_obj     = torch.argmax(classifier(obj_feature), dim=0)
            image_path = f"tmp_add/for_reason_{idx}.jpg"
            tensor_to_pil(out).save(image_path)
            reason_out = reason_model.reason(image_path, args.prompt)
            _, W, H    = out.shape

            try:
                sel_ids, boxes, mask, _ = extract_selected_obj_ids(
                    reason_out, out, pred_obj, sam_predictor,
                    input_width=W, input_height=H,
                    clip_model=clip_model, clip_preprocess=clip_preprocess
                )
                mask_path = f"tmp_add/reason_mask_{idx}.jpg"
                # pdb.set_trace()
            
                tensor_to_pil(mask.squeeze().unsqueeze(0).repeat(3, 1, 1)).save(mask_path)

                if (sel_ids == best_id).any():
                    selected_views.append(view)
                    selected_masks.append(mask.clone().cpu())  # 复制到 CPU，免受后续修改
            except Exception as e:
                print(f"[Warning] Failed to process view {view} due to error: {e}")
                continue
    return selected_views, selected_masks

def local_reason(args, gaussians, pipeline, background, classifier, selected_views, selected_masks):
    iteration_finetune = args.itr_finetune
    view_index_stack = list(range(len(selected_views)))
    for step in tqdm(range(iteration_finetune)):
        torch.cuda.empty_cache()
        if not view_index_stack:
            view_index_stack = list(range(len(selected_views)))
        view_index = random.choice(view_index_stack)
        view_index_stack.remove(view_index)

        out_target_pkg = render_mask(selected_views[view_index], gaussians, pipeline, background, gaussians._mask3d)
        
        loss = l1_loss(out_target_pkg['render'], selected_masks[view_index].squeeze().unsqueeze(0).repeat(3, 1, 1).float().cuda())
        
        if step % 50 ==0:
            tensor_to_pil(out_target_pkg['render']).save('tmp_add/for_reason.jpg')
        loss.backward()
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)


def remove_local(args, reason_model, sam_predictor, clip_model, clip_preprocess, views, gaussians, pipeline, background, classifier, best_id, mask3d):
    refine_cameras = sample_reason_cameras_cluster_then_find_id(views, gaussians, pipeline, background, classifier, best_id)
    # refine_cameras = sample_reason_cameras_cluster_then_find_id(views, gaussians, pipeline, background, classifier, best_id)
    selected_views = []   # 存代表性 Camera
    selected_images_ori = []   # 存代表性 Camera
    selected_images_inpaint = []   # 存代表性 Camera
    mask3d = ~mask3d

    selected_masks = []   # 存对应 mask (torch.bool [H,W])
    with torch.no_grad():
        for idx, view in enumerate(refine_cameras):

            mask_out_pkg = render_target(view, gaussians, pipeline, background, mask3d)
            target_image = mask_out_pkg["render"]
            
            out_pkg = render(view, gaussians, pipeline, background)

            out          = out_pkg["render"]
            obj_feature  = out_pkg["render_object"]
            pred_obj     = torch.argmax(classifier(obj_feature), dim=0)
            image_path = f"tmp_add/for_reason_remove_{idx}.jpg"
            to_pil_image(out).save(image_path)
            reason_out = reason_model.reason(image_path, args.prompt)
            _, W, H    = out.shape

            try:
                sel_ids, boxes, mask, _ = extract_selected_obj_ids(
                    reason_out, out, pred_obj, sam_predictor,
                    input_width=W, input_height=H,
                    clip_model=clip_model, clip_preprocess=clip_preprocess
                )
                mask_path = f"tmp_add/reason_mask_{idx}.jpg"
                inpaint_path = f"tmp_add/reason_inpaint_{idx}.jpg"
            
                tensor_to_pil(mask.squeeze().unsqueeze(0).repeat(3, 1, 1)).save(mask_path)

                if (sel_ids == best_id).any():
                    args.lama_config = 'inpaint_anything/lama/configs/prediction/default.yaml'
                    args.lama_ckpt = 'inpaint_anything/big-lama/big-lama'
                    # inpaint_img_with_lama(out, )
                    lama_img_input = (target_image*255).byte().cpu().numpy().transpose(1, 2, 0)
                    lama_mask_input = mask.squeeze().to('cpu').numpy().astype(np.uint8)
                    img_inpainted = inpaint_img_with_lama(lama_img_input, lama_mask_input, args.lama_config, args.lama_ckpt, device="cuda")
                    img_inpainted_tensor = torch.from_numpy(img_inpainted).permute(2, 0, 1).float().cuda()
                    img_inpainted_tensor = img_inpainted_tensor / 255.0
                    
                    # save_tensor_as_png(img_inpainted_tensor, inpaint_path)
                    tensor_to_pil(img_inpainted_tensor).save(inpaint_path)
                    tensor_to_pil(target_image).save(f"tmp_add/reason_remove_{idx}.jpg")
                    
                    selected_views.append(view)
                    selected_images_ori.append(out)
                    selected_images_inpaint.append(img_inpainted_tensor)
                    selected_masks.append(mask.clone().cpu())  # 复制到 CPU，免受后续修改
            except Exception as e:
                print(f"[Warning] Failed to process view {view} due to error: {e}")
                continue
    return selected_views, selected_masks


def select_local_style(args, reason_model, sam_predictor, clip_model, clip_preprocess, views, gaussians, pipeline, background, classifier, best_id):
    refine_cameras = sample_reason_cameras_cluster_then_find_id(views, gaussians, pipeline, background, classifier, best_id, n_cluster = 64)
    # refine_cameras = sample_reason_cameras_cluster_then_find_id(views, gaussians, pipeline, background, classifier, best_id)
    selected_views = []   # 存代表性 Camera
    selected_masks = []   # 存对应 mask (torch.bool [H,W])
    selected_reason_out = []
    with torch.no_grad():
        for idx, view in enumerate(refine_cameras):
            out_pkg = render(view, gaussians, pipeline, background)
            out          = out_pkg["render"]
            obj_feature  = out_pkg["render_object"]
            pred_obj     = torch.argmax(classifier(obj_feature), dim=0)
            image_path = f"tmp_add/for_reason_{idx}.jpg"
            tensor_to_pil(out).save(image_path)
            reason_out = reason_model.reason(image_path, args.prompt)
            _, W, H    = out.shape

            try:
                sel_ids, boxes, mask, _ = extract_selected_obj_ids(
                    reason_out, out, pred_obj, sam_predictor,
                    input_width=W, input_height=H,
                    clip_model=clip_model, clip_preprocess=clip_preprocess
                )
                mask_path = f"tmp_add/reason_mask_{idx}.jpg"
                # pdb.set_trace()
            
                tensor_to_pil(mask.squeeze().unsqueeze(0).repeat(3, 1, 1)).save(mask_path)

                if (sel_ids == best_id).any():
                    selected_views.append(view)
                    selected_masks.append(mask.clone().cpu())  # 复制到 CPU，免受后续修改
                    selected_reason_out.append(reason_out)
            except Exception as e:
                print(f"[Warning] Failed to process view {view} due to error: {e}")
                continue
    return selected_views, selected_masks, selected_reason_out
