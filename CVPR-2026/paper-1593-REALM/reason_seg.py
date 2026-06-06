import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render_target, render_dpt, render_mask
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from PIL import Image
import cv2
import ast
import rembg
from threestudio.utils.dpt import DPT
import sys
from pathlib import Path
from utils.graphics_utils import fov2focal
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from ext.grounded_sam import grouned_sam_output, load_model_hf, select_obj_ioa, segment_with_sam
from segment_anything import sam_model_registry, SamPredictor
import pdb
from utils.general_utils import save_tensor_as_png
from render import feature_to_rgb, visualize_obj
import imageio
from utils.reason_utils import ReasonModel, plot_bounding_boxes, tensor_to_pil, ReasonModelAPI, KIMIAPI
from argparse import ArgumentParser, Namespace
import sys
import os
from scipy.spatial import ConvexHull, Delaunay
import random
from omegaconf import OmegaConf
import json
import subprocess
import os
# from guidance.EditGuidance import EditGuidance
from threestudio.models.prompt_processors.stable_diffusion_prompt_processor import StableDiffusionPromptProcessor
import ssl
from torchvision.transforms.functional import to_pil_image, to_tensor

from threestudio.utils.misc import (
    get_device,
    step_check,
    dilate_mask,
    erode_mask,
    fill_closed_areas,
)

import ui_utils
import clip
import torch.nn.functional as F
from torchvision.ops import masks_to_boxes
from reasoneditor.id_utils import extract_selected_obj_ids
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
from PIL import ImageDraw
from collections import Counter
from reasoneditor.edit_utils import edit_remove, edit_replace, edit_style_transfer
from utils.reason_utils import parse_json  
import pdb
from reasoneditor.id_utils import points_inside_convex_hull
import time
import csv

from utils.camera_utils import sample_reason_cameras, sample_reason_cameras_cluster_then_topk, sample_reason_cameras_cluster_then_find_id, select_local, local_reason


def global_reason(args, gaussians, scene, classifier, sam_predictor, clip_model, clip_preprocess, reason_model, background):
    TEXT_PROMPT = args.prompt
    views = scene.getTrainCameras()
    # reason_cameras = sample_reason_cameras(views)
    # reason_cameras = sample_reason_cameras(views)
    # reason_cameras = sample_reason_cameras(views)
    # reason_cameras = sample_reason_cameras(views)
    # reason_cameras = sample_reason_cameras(views)
    # reason_cameras = sample_reason_cameras(views)
    # reason_cameras = sample_reason_cameras(views)
    reason_cameras = sample_reason_cameras_cluster_then_topk(views, gaussians, pipeline, background, classifier)
    # reason_cameras = sample_reason_cameras_cluster_then_topk(views, gaussians, pipeline, background, classifier)
    replacement_color = torch.tensor([1.0, 0.0, 0.0]).view(3, 1).cuda()
    from collections import Counter
    
    K = 10  
    all_sel_ids = []
    all_reason_outputs = []
    all_reason_cam_indices = []
    all_probs = [] 
    with torch.no_grad():
        for idx, view in enumerate(reason_cameras):
            out_pkg = render(view, gaussians, pipeline, background)
            out = out_pkg["render"]
            print(out.shape)
            out_obj_feature = out_pkg["render_object"]
            logits = classifier(out_obj_feature)
            pred_obj = torch.argmax(logits, dim=0)
            image_path = f'tmp_add/for_reason_{idx}.jpg'
            tensor_to_pil(out).save(image_path)

            reason_output = reason_model.reason(image_path, TEXT_PROMPT)
            _, input_width, input_height = out.shape

            try:
                sel_ids, boxes, mask, probs = extract_selected_obj_ids(
                    reason_output, out, pred_obj, sam_predictor, input_width, input_height,
                    clip_model, clip_preprocess
                )
                mask_path = f"tmp_add/reason_mask_{idx}.jpg"
                tensor_to_pil(mask.squeeze().unsqueeze(0).repeat(3, 1, 1)).save(mask_path)

            except Exception as e:
                # print(f"[Warning] Failed to process view {view} due to error: {e}")
                continue
            if probs is None:
                continue
            print(probs)

            print(sel_ids)
            
            replacement_color = torch.tensor([1.0, 0.0, 0.0]).view(3, 1).cuda()
            # print(mask.shape)
            pred_obj_mask = mask.squeeze().float()


            rendering_obj_prompt = out
            rendering_obj_prompt[:, pred_obj_mask.bool()] = replacement_color * 0.5 + rendering_obj_prompt[:, pred_obj_mask.bool()]*0.5
            rendering_obj_prompt[:, ~pred_obj_mask.bool()] = rendering_obj_prompt[:, ~pred_obj_mask.bool()] * 0.4
            tensor_to_pil(rendering_obj_prompt).save(f'tmp_add/obj_{idx}.png')
            

            all_probs.append((probs, idx, sel_ids, reason_output))  


    top_k = sorted(all_probs, key=lambda x: x[0], reverse=True)[:K]

    vote_counts = Counter()
    selected_cam_indices = []
    selected_reason_outputs = []
    all_sel_ids = []

    for prob, idx, sel_ids, reason_output in top_k:
        vote_counts.update(sel_ids.cpu().tolist())
        selected_cam_indices.append(idx)
        all_sel_ids.append(sel_ids)
        selected_reason_outputs.append(reason_output)

    if vote_counts:
        most_common_id, cnt = vote_counts.most_common(1)[0]
        best_id = torch.tensor([most_common_id], device='cuda')
        print(f"id = {most_common_id}, times = {cnt}")

        # Now filter your lists so they only contain views that voted for `most_common_id`
        filtered_cam_indices     = []
        filtered_sel_ids         = []
        filtered_reason_outputs  = []

        for idx, sel_ids, reason_output in zip(
            selected_cam_indices, all_sel_ids, selected_reason_outputs
        ):
            # sel_ids is a tensor of IDs predicted in that view
            if most_common_id in sel_ids.cpu().tolist():
                filtered_cam_indices.append(idx)
                filtered_sel_ids.append(sel_ids)
                filtered_reason_outputs.append(reason_output)

        # overwrite your lists with the filtered versions
        selected_cam_indices    = filtered_cam_indices  # [1, 11, 8, 5, 10, 2]
        all_sel_ids             = filtered_sel_ids
        selected_reason_outputs = filtered_reason_outputs

    else:
        best_id = torch.tensor([], device='cuda', dtype=torch.long)
        print("best_id is empty")
    # best_id = torch.tensor([], device='cuda', dtype=torch.long)

    return  best_id

def reason_and_edit(args, dataset, iteration, pipeline, opt):
        replacement_color = torch.tensor([1.0, 0.0, 0.0]).view(3, 1).cuda()

    # with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        num_classes = dataset.num_classes
        print("Num classes: ",num_classes)
        classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
        classifier.cuda()
        classifier.load_state_dict(torch.load(os.path.join(dataset.model_path,"point_cloud","iteration_"+str(30000),"classifier.pth")))

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        ckpt_repo_id = "ShilongLiu/GroundingDINO"
        ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
        ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
        sam_checkpoint = 'Tracking-Anything-with-DEVA/saves/sam_vit_l_0b3195.pth'
        sam_checkpoint = 'Tracking-Anything-with-DEVA/saves/sam_vit_h_4b8939.pth'
        sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        sam.to(device='cuda')
        sam_predictor = SamPredictor(sam)

        clip_model, clip_preprocess = clip.load("ViT-B/32", device="cuda")
        # reason_model = KIMIAPI()
        reason_model = ReasonModelAPI()
        print("model loaded")

        reason_model.sys_prompt = 'I am working on a task that involves finding an object within a scene. ' \
            'You are a helpful assistant. I will provide an implicit description of the editing intention, and your task is to identify and outline the most appropriate region containing the target object. ' \
            'Then, generate a label for the original object and provide a clear, explicit description of the new style that will replace it. ' \
            'Your output must strictly follow the JSON format (Strictly follow, very important!!!): '\
            '[{"bbox_2d": [x1, y1, x2, y2], "label": "...", "explanation":"..."}] ' \
            'The bbox_2d must be a list of FOUR numeric values..' \
            'You should only include one single dict in the list. The value in the dict should not be empty' \
            'Any reasoning process or explanation should only appear in the "explanation" field.' \
            'The "label" should specify the object. ' \
            'For example, given the prompt "Where is the fruit that is yellow", you should return the bounding box of the banana, set "label": "banana", and explain your reasoning process in "explanation".'

        # json_path = os.path.join('/root/autodl-tmp/aaai_workspace/autodl-fs/SAGA_data/LERF/figurines/video_path_figurines.json')
        # render_poses = scene.create_cameras_from_path(json_path)
        # render_poses = align_cameras_to_train(render_poses, scene.getTrainCameras()[0])
        # render_poses = scene.getRenderCameras(scale=1.0)
        # best_id = torch.tensor([1], device='cuda', dtype=torch.long)
        start_global_reason = time.time()
        best_id = global_reason(args, gaussians, scene, classifier, sam_predictor, clip_model, clip_preprocess, reason_model, background)
        end_global_reason = time.time()


        # selected_obj_ids = torch.tensor(most_common_id).unsqueeze(0).cuda()
        # pdb.set_trace()
        # assert 0==1
        
        if isinstance(best_id, torch.Tensor) and best_id.numel() == 0:
            print("No object found")
            print("No object found")
            print("No object found")
            print("No object found")
            print("No object found")
            print("No object found")
            print("No object found")
            print("No object found")
            print("No object found")
            print("No object found")
            print("No object found")
            N = gaussians._xyz.shape[0]
            mask3d = torch.zeros(N, dtype=torch.float, device=gaussians._xyz.device)
            # mask3d[0] = 1
        else:
            logits3d = classifier(gaussians._objects_dc.permute(2,0,1))
            prob_obj3d = torch.softmax(logits3d,dim=0)

            mask = prob_obj3d[best_id, :, :] > 0.5
            mask3d = mask.any(dim=0).squeeze()
            mask3d_convex = points_inside_convex_hull(gaussians._xyz.detach(),mask3d,outlier_factor=1.0)
            mask3d = torch.logical_or(mask3d,mask3d_convex).float()
            # mask3d = torch.logical_or(mask3d,mask3d_convex).float()
            # mask3d = torch.logical_or(mask3d,mask3d_convex).float()
            mask3d = mask3d.clone().detach().float().requires_grad_(True)
            
            # if not args.skip_local:
            #     start_select_local = time.time()
            #     selected_views, selected_masks = select_local(args, reason_model, sam_predictor, clip_model, clip_preprocess, scene.getTrainCameras(), gaussians, pipeline, background, classifier, best_id)
            #     end_select_local = time.time()
            #     if len(selected_views) > 0:
            #         gaussians.reasonseg_setup(opt, mask3d)
            #         start_local_reason = time.time()
            #         local_reason(args, gaussians, pipeline, background, classifier, selected_views, selected_masks)
            #         end_local_reason = time.time()
            #         mask3d = gaussians._mask3d
        

        ## compute running time and save to CSV

        # time_global_reason = end_global_reason - start_global_reason
        # time_select_local = end_select_local - start_select_local
        # time_local_reason = end_local_reason - start_local_reason

        # # 保存结果到CSV文件
        # with open('global_view.csv', mode='a', newline='') as file:
        #     writer = csv.writer(file)

        #     if file.tell() == 0:
        #         writer.writerow(['scene', 'N_global', 'time'])           
        #     writer.writerow([args.object, 16, time_global_reason+time_select_local+time_local_reason])



        # print(f"global_reason 执行时间: {time_global_reason:.4f} 秒")
        # print(f"select_local 执行时间: {time_select_local:.4f} 秒")
        # print(f"local_reason 执行时间: {time_local_reason:.4f} 秒")
        
        
        test_views = scene.getTrainCameras()
        mask_path_save = os.path.join(args.model_path, args.out_repo, 'reason_mask')
        ano_path_save = os.path.join(args.model_path, args.out_repo, 'target_obj')
        target_obj = f'{args.object}.png'

        views = scene.getTestCameras()
        rendered_frames = []  # 用于收集所有帧图像（PIL.Image 或 ndarray）
        # idx = torch.randint(0, len(views), (1,)).item()
        # view = views[idx]  # Choose a specific time for rendering

        # with torch.no_grad():
        #     for idx, pose in tqdm(enumerate(render_poses), total=len(render_poses), desc="Rendering frames"):
        #         # matrix = np.linalg.inv(np.array(pose))
        #         # R = -np.transpose(matrix[:3, :3])
        #         # R[:, 0] = -R[:, 0]
        #         # T = -matrix[:3, 3]
        #         # view.reset_extrinsic(R, T)

        #         render_pkg = render(view, gaussians, pipeline, background)
        #         mask_pkg = render_mask(view, gaussians, pipeline, background, mask3d)
                
        #         out = render_pkg['render']
        #         pred_obj_mask = (mask_pkg["render"][0] >= 0.5).float().squeeze()

        #         rendering_obj_prompt = out
        #         rendering_obj_prompt[:, pred_obj_mask.bool()] = replacement_color * 0.5 + rendering_obj_prompt[:, pred_obj_mask.bool()]*0.5
        #         rendering_obj_prompt[:, ~pred_obj_mask.bool()] = rendering_obj_prompt[:, ~pred_obj_mask.bool()] * 0.4

        #         # save_path = f'{mask_path_save}/{idx}/{target_obj}' 
        #         # save_path_img = f'{ano_path_save}/{idx}/{target_obj}' 
        #         # os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 确保目录存在
        #         # os.makedirs(os.path.dirname(save_path_img), exist_ok=True)  # 确保目录存在
        #         # mask_np = pred_obj_mask.detach().cpu().numpy().astype(np.uint8) * 255  # 转成 0/255
        #         # mask_pil = Image.fromarray(mask_np)
        #         # mask_pil.save(save_path)
        #         # tensor_to_pil(rendering_obj_prompt).save(save_path_img)
        #         # 转 PIL 并存入帧列表
        #         pil_img = tensor_to_pil(rendering_obj_prompt)
        #         rendered_frames.append(np.array(pil_img))  # imageio 要求是 numpy 格式

        # views = scene.getRenderCameras(scale=1.0, interp_per_pair=100)
        # views = scene.getRenderCameras(scale=1.0, interp_per_pair=20)
        with torch.no_grad():
            # for idx, view in enumerate(render_poses):
            for idx, view in tqdm(enumerate(views), total=len(views), desc="Rendering frames"):

                render_pkg = render(view, gaussians, pipeline, background)
                mask_pkg = render_mask(view, gaussians, pipeline, background, mask3d)
                
                out = render_pkg['render']
                pred_obj_mask = (mask_pkg["render"][0] >= 0.5).float().squeeze()
                # pdb.set_trace()
                # assert view.image_height == 728
                # assert view.image_width == 986
                
                rendering_obj_prompt = out
                rendering_obj_prompt[:, pred_obj_mask.bool()] = replacement_color * 0.5 + rendering_obj_prompt[:, pred_obj_mask.bool()]*0.5
                rendering_obj_prompt[:, ~pred_obj_mask.bool()] = rendering_obj_prompt[:, ~pred_obj_mask.bool()] * 0.4

                save_path = f'{mask_path_save}/{idx}/{target_obj}' 
                save_path_img = f'{ano_path_save}/{idx}/{target_obj}' 
                os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 确保目录存在
                os.makedirs(os.path.dirname(save_path_img), exist_ok=True)  # 确保目录存在
                mask_np = pred_obj_mask.detach().cpu().numpy().astype(np.uint8) * 255  # 转成 0/255
                mask_pil = Image.fromarray(mask_np)
                mask_pil.save(save_path)
                tensor_to_pil(rendering_obj_prompt).save(save_path_img)
                # # 转 PIL 并存入帧列表
                pil_img = tensor_to_pil(out)
                rendered_frames.append(np.array(pil_img))  # imageio 要求是 numpy 格式
                if idx == 3000:
                    break

        # video_save_path = f'{ano_path_save}/video_ori.mp4'
        # os.makedirs(os.path.dirname(video_save_path), exist_ok=True)
        # imageio.mimsave(video_save_path, rendered_frames, fps=30)

def align_cameras_to_train(external_cams, train_cam):
    """
    external_cams: list of LiteCamera（外部轨迹）
    train_cam: 一个参考相机（train_cameras 中的一帧）
    """

    # 选第一帧外部相机作为参考
    cam0_ext = external_cams[0]
    cam0_train = train_cam

    # 各自的世界坐标变换矩阵
    ext_c2w = np.eye(4)
    ext_c2w[:3, :3] = cam0_ext.R
    ext_c2w[:3, 3] = cam0_ext.T

    train_c2w = np.eye(4)
    train_c2w[:3, :3] = cam0_train.R
    train_c2w[:3, 3] = cam0_train.T

    # 计算从 external 到 train 的对齐矩阵
    align_matrix = train_c2w @ np.linalg.inv(ext_c2w)

    # 应用到所有 external 相机
    for cam in external_cams:
        c2w = np.eye(4)
        c2w[:3, :3] = cam.R
        c2w[:3, 3] = cam.T

        aligned_c2w = align_matrix @ c2w
        cam.R = aligned_c2w[:3, :3]
        cam.T = aligned_c2w[:3, 3]

    return external_cams
def render_wander_path(view):
    focal_length = fov2focal(view.FoVy, view.image_height)
    R = view.R
    R[:, 1] = -R[:, 1]
    R[:, 2] = -R[:, 2]
    T = -view.T.reshape(-1, 1)
    pose = np.concatenate([R, T], -1)

    num_frames = 60
    max_disp = 5000.0  # 64 , 48

    max_trans = max_disp / focal_length  # Maximum camera translation to satisfy max_disp parameter
    output_poses = []

    for i in tqdm(range(num_frames), desc="Generating wander path"):
        x_trans = max_trans * np.sin(2.0 * np.pi * float(i) / float(num_frames))
        y_trans = max_trans * np.cos(2.0 * np.pi * float(i) / float(num_frames)) / 3.0  # * 3.0 / 4.0
        z_trans = max_trans * np.cos(2.0 * np.pi * float(i) / float(num_frames)) / 3.0

        i_pose = np.concatenate([
            np.concatenate(
                [np.eye(3), np.array([x_trans, y_trans, z_trans])[:, np.newaxis]], axis=1),
            np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]
        ], axis=0)  # [np.newaxis, :, :]

        i_pose = np.linalg.inv(i_pose)  # torch.tensor(np.linalg.inv(i_pose)).float()

        ref_pose = np.concatenate([pose, np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]], axis=0)

        render_pose = np.dot(ref_pose, i_pose)
        output_poses.append(torch.Tensor(render_pose))

    return output_poses

def interpolate_camera_path(cameras, num_interp_per_pair=10):
    poses = []

    for i in range(len(cameras) - 1):
        cam0, cam1 = cameras[i], cameras[i + 1]
        R0, T0 = cam0.R, cam0.T
        R1, T1 = cam1.R, cam1.T

        # 将 R 转换为旋转矩阵（确保是 numpy array）
        R0 = np.array(R0)
        R1 = np.array(R1)
        T0 = np.array(T0)
        T1 = np.array(T1)

        for t in np.linspace(0, 1, num_interp_per_pair):
            # Rotation: LERP 每个元素（不是最优，但简单）
            Rt = (1 - t) * R0 + t * R1
            # 正交化（确保仍是旋转矩阵）
            U, _, Vt = np.linalg.svd(Rt)
            Rt = U @ Vt

            # Translation: 普通线性插值
            Tt = (1 - t) * T0 + t * T1

            # 构建 4x4 相机位姿矩阵
            pose = np.eye(4)
            pose[:3, :3] = Rt
            pose[:3, 3] = Tt
            poses.append(torch.tensor(pose, dtype=torch.float32))

    return poses

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--prompt", default='duck', type=str)
    parser.add_argument("--object", default='duck', type=str)
    parser.add_argument("--skip_local", action="store_true")
    parser.add_argument("--itr_finetune", default=50, type=int)
    parser.add_argument("--out_repo", default='realm', type=str)
    # parser.add_argument("--train_split", action="store_true")

    args = get_combined_args(parser)
    # MODEL_PATH = './output/rgb/lerf/teatime'
    # args = get_combined_args_reason(parser, MODEL_PATH)

    args.num_classes = 256
    # args.eval = False
    args.train_split = True
    print("Rendering " + args.model_path)
    dataset = model.extract(args)
    iteration = args.iteration
    pipeline = pipeline.extract(args)
    skip_train = args.skip_train
    skip_test = args.skip_test
    opt = op
    print(args.prompt)
    print(args.object)
    print(args.out_repo)
    reason_and_edit(args, dataset, iteration, pipeline, opt)