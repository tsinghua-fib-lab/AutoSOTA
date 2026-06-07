import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from CLIP import clip
from depth_anything.dpt import DepthAnything

from dataloaders.dataset_nyukiti import (
    change_to_kitti,
    change_to_nyu,
    change_to_void,
    MixedNYUKITTIVOIDC3VD,
)
from dataloaders.dataset_c3vd import change_to_c3vd

from options import MonodepthOptions
from scalemap_depth import ScaleMapModel
from utils import compute_errors, get_text, print_model_parameters, setup_tee_logging, visualize_eval

EPS = 1e-8

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

options = MonodepthOptions()
args = options.parse()


def export_scalemap_weights(src_ckpt_path: str, dst_ckpt_path: str, map_location="cpu"):
    """Strip the `scalemap.`/`module.scalemap.` prefix from a training checkpoint
    and save a bare `state_dict` that ScaleMapModel can load directly."""
    ckpt = torch.load(src_ckpt_path, map_location=map_location)

    if "model" not in ckpt:
        raise KeyError(f"`model` not found in checkpoint keys: {list(ckpt.keys())}")

    state = ckpt["model"]

    scalemap_state = {}
    for k, v in state.items():
        if k.startswith("scalemap."):
            scalemap_state[k[len("scalemap."):]] = v
        elif k.startswith("module.scalemap."):
            scalemap_state[k[len("module.scalemap."):]] = v

    if len(scalemap_state) == 0:
        sample_keys = list(state.keys())[:30]
        raise ValueError(
            "No scalemap weights found. "
            f"First keys in checkpoint['model']: {sample_keys}"
        )

    torch.save({"state_dict": scalemap_state}, dst_ckpt_path)
    print(f"[OK] exported {len(scalemap_state)} scalemap params -> {dst_ckpt_path}")


def recover_image(normalized_tensor):
    mean_tensor = torch.tensor(mean).view(3, 1, 1)
    std_tensor = torch.tensor(std).view(3, 1, 1)
    device = normalized_tensor.device
    mean_tensor = mean_tensor.to(device)
    std_tensor = std_tensor.to(device)

    original_tensor = normalized_tensor * std_tensor + mean_tensor
    original_tensor = original_tensor.mul(255).clamp(0, 255).byte()
    return original_tensor.cpu().numpy().transpose(1, 2, 0)


def eval(Scalemap_model, depth_model, CLIP_model, Image_f_model, dataloader_eval,
         vis_save_path=None, post_process=False, dataset=None):
    eval_measures = torch.zeros(10).cuda()

    for step, eval_sample_batched in tqdm(enumerate(dataloader_eval.data), file=sys.__stderr__):
        with torch.no_grad():
            image = torch.autograd.Variable(eval_sample_batched['image'].cuda())
            gt_depth = eval_sample_batched['depth']
            if dataset == 'nyu' or dataset == 'kitti':
                has_valid_depth = eval_sample_batched['has_valid_depth']
                if not has_valid_depth:
                    print('Invalid depth. continue.')
                    continue

            # Forward
            if dataset != 'c3vd':
                text = get_text(args.txt_path_eval, eval_sample_batched['sample_path'],
                                mode="eval", dataset=dataset,
                                combine_words_no_area=args.combine_words_no_area,
                                close_car_percent=args.close_car_percent,
                                far_car_percent=args.far_car_percent)
            else:
                text = eval_sample_batched['text']

            text_tokens = clip.tokenize(text, truncate=True).to("cuda")

            image_h, image_w = image.shape[2], image.shape[3]
            if "da" in args.depth_model:
                if image_h == 480:
                    a = 479 - 45
                    b = 603 - 43
                else:
                    a = 350
                    b = 1204
                image = torch.nn.functional.interpolate(
                    image, size=(a, b), mode="bilinear", align_corners=True,
                )

            text_features = CLIP_model.encode_text(text_tokens)
            text_features = text_features.unsqueeze(1)
            image_features = Image_f_model.get_intermediate_layers(
                image, n=1, return_class_token=False
            )[0]

            scale_pred, shift_pred, _, _ = Scalemap_model(
                img_features=image_features.float(),
                text_features=text_features.float(),
                patch_h=int(a / 14),
                patch_w=int(b / 14),
            )

            # --- TTA: original pass ---
            relative_depth = depth_model(image).unsqueeze(1)
            pred_depth_orig = 1 / (scale_pred * relative_depth + shift_pred)
            
            # --- TTA: horizontal flip ---
            image_flipped = torch.flip(image, [3])
            if "da" in args.depth_model:
                image_flipped_resized = image_flipped
                if image_h == 480:
                    a_flip = 479 - 45
                    b_flip = 603 - 43
                else:
                    a_flip = 350
                    b_flip = 1204
                image_flipped_resized = torch.nn.functional.interpolate(
                    image_flipped, size=(a_flip, b_flip), mode="bilinear", align_corners=True,
                )
            image_features_flip = Image_f_model.get_intermediate_layers(
                image_flipped_resized, n=1, return_class_token=False
            )[0]
            scale_pred_flip, shift_pred_flip, _, _ = Scalemap_model(
                img_features=image_features_flip.float(),
                text_features=text_features.float(),
                patch_h=int(a_flip / 14),
                patch_w=int(b_flip / 14),
            )
            relative_depth_flip = depth_model(image_flipped).unsqueeze(1)
            pred_depth_flip = 1 / (scale_pred_flip * relative_depth_flip + shift_pred_flip)
            # Unflip
            pred_depth_flip = torch.flip(pred_depth_flip, [3])
            
            # Ensemble: average original and flipped
            pred_depth = (pred_depth_orig + pred_depth_flip) / 2.0


            if "da" in args.depth_model or args.depth_model == "midas":
                pred_depth = torch.nn.functional.interpolate(
                    pred_depth, size=(image_h, image_w), mode="bilinear", align_corners=True,
                )
                relative_depth = torch.nn.functional.interpolate(
                    relative_depth, size=(image_h, image_w), mode="bilinear", align_corners=True,
                )

            pred_depth = pred_depth.cpu().numpy().squeeze()
            gt_depth = gt_depth.cpu().numpy().squeeze()
            relative_depth = relative_depth.cpu().numpy().squeeze()

            if vis_save_path is not None:
                image_vis = recover_image(image.squeeze())
                scale_pred_np = scale_pred.cpu().numpy().squeeze()
                shift_pred_np = shift_pred.cpu().numpy().squeeze()
                sample_path = eval_sample_batched['sample_path'][0]
                if dataset != 'c3vd':
                    sample_path = sample_path.split()[0]
                sample_path = sample_path.replace('/', '_')
                visualize_eval(image_vis, gt_depth, relative_depth, None, pred_depth,
                               scale_pred_np, shift_pred_np,
                               args.min_depth_eval, args.max_depth_eval, image_h, image_w,
                               sample_path, vis_save_path)

        if dataset == "kitti":
            height, width = gt_depth.shape
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
            pred_depth_uncropped[top_margin:top_margin + 352,
                                 left_margin:left_margin + 1216] = pred_depth
            pred_depth = pred_depth_uncropped

        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)

        if dataset == "kitti":
            gt_height, gt_width = gt_depth.shape
            eval_mask = np.zeros(valid_mask.shape)
            eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                      int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1
            valid_mask = np.logical_and(valid_mask, eval_mask)
        elif dataset == 'nyu' or dataset == 'void':
            gt_height, gt_width = gt_depth.shape
            eval_mask = np.zeros(valid_mask.shape)
            eval_mask[45:471, 41:601] = 1
            valid_mask = np.logical_and(valid_mask, eval_mask)

        measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])
        eval_measures[:9] += torch.tensor(measures).cuda()
        eval_measures[9] += 1

    eval_measures_cpu = eval_measures.cpu()
    cnt = eval_measures_cpu[9].item()
    eval_measures_cpu /= cnt
    print(f'Computing errors for {int(cnt)} eval samples, post_process: {post_process}')
    print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format(
        'silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3'))
    formatted_values = ", ".join([f"{value:7.3f}" for value in eval_measures_cpu[:-1]])
    print(f"[{formatted_values}]")

    sheet_layouts = {
        'nyu':   ("NYUD2", ['d1', 'd2', 'd3', 'abs_rel', 'log10', 'rms'], [6, 7, 8, 1, 2, 3]),
        'kitti': ("KITTI", ['d1', 'd2', 'd3', 'abs_rel', 'log_rms', 'rms'], [6, 7, 8, 1, 5, 3]),
        'void':  ("VOID",  ['d1', 'd2', 'd3', 'abs_rel', 'log10', 'rms'], [6, 7, 8, 1, 2, 3]),
        'c3vd':  ("C3VD",  ['d1', 'd2', 'd3', 'abs_rel', 'log10', 'rms'], [6, 7, 8, 1, 5, 3]),
    }
    if dataset in sheet_layouts:
        name, headers, idx = sheet_layouts[dataset]
        print(f"Results for sheets, {name}:")
        print("{:>6}, {:>6}, {:>6}, {:>6}, {:>6}, {:>6}".format(*headers))
        formatted_values = ", ".join([f"{eval_measures_cpu[i]:7.3f}" for i in idx])
        print(f"[{formatted_values}]")

    return eval_measures_cpu


def main(args):
    if args.log_file:
        from datetime import datetime
        log_path = args.log_file
        if log_path == 'auto':
            log_path = f"logs/eval_indomain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        setup_tee_logging(log_path)

    torch.cuda.empty_cache()
    cudnn.benchmark = True
    # Disable cuDNN/flash SDP backends to avoid compatibility issues with some driver/cuDNN versions
    # torch.backends.cuda.enable_flash_sdp(False)  # not available in PT 2.1
    # torch.backends.cuda.enable_mem_efficient_sdp(False)  # not available in PT 2.1
    # torch.backends.cuda.enable_cudnn_sdp(False)  # not available in PT 2.1

    args.distributed = False

    if getattr(args, 'nyu_root', None):
        change_to_nyu(args)
        dataloader_eval_nyu = MixedNYUKITTIVOIDC3VD(args, 'online_eval')
    else:
        dataloader_eval_nyu = None
        print("[skip] --nyu_root not set, skipping nyu.")

    if getattr(args, 'kitti_root', None) and getattr(args, 'kitti_gt_root', None):
        change_to_kitti(args)
        dataloader_eval_kitti = MixedNYUKITTIVOIDC3VD(args, 'online_eval')
    else:
        dataloader_eval_kitti = None
        print("[skip] --kitti_root and/or --kitti_gt_root not set, skipping kitti.")

    if getattr(args, 'void_root', None):
        change_to_void(args)
        dataloader_eval_void = MixedNYUKITTIVOIDC3VD(args, 'online_eval')
    else:
        dataloader_eval_void = None
        print("[skip] --void_root not set, skipping void.")

    if getattr(args, 'c3vd_root', None):
        change_to_c3vd(args)
        dataloader_eval_c3vd = MixedNYUKITTIVOIDC3VD(args, 'online_eval_all')
    else:
        dataloader_eval_c3vd = None
        print("[skip] --c3vd_root not set, skipping c3vd.")

    print("Depth Model:", args.depth_model)
    if "da" in args.depth_model:
        model_configs = {
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
        }
        encoder = 'vit' + args.depth_model[-1]
        depth_model = DepthAnything(model_configs[encoder])
        depth_model.load_state_dict(
            torch.load(f'/container_pip/da_backbone.pth')
        )
        depth_model.to("cuda").eval()
    else:
        raise ValueError(
            f"Unsupported depth_model: {args.depth_model}. "
            "This release supports DepthAnything (da_s/da_b/da_l)."
        )

    print(f"== Depth Model Initialized : {encoder}")

    Scalemap_model = ScaleMapModel(args.image_channels, args.out_channels, args.text_channels,
                                   args.features, args.cross_attn_nhead,
                                   args.coef_scale, args.coef_shift)
    Scalemap_model.eval()
    Scalemap_model.cuda()
    print("== ScaleMap Model Initialized")

    CLIP_model, _ = clip.load(args.text_encoder, device="cuda")
    Image_f_model = torch.hub.load('/container_pip/repo/torchhub/facebookresearch_dinov2_main', f'dinov2_{args.img_encoder}14', source='local').to('cuda')
    CLIP_model.eval()
    Image_f_model.eval()

    print_model_parameters(
        [CLIP_model.token_embedding, CLIP_model.positional_embedding,
         CLIP_model.transformer, CLIP_model.ln_final, CLIP_model.text_projection],
        "CLIP Text Model", print_or_log="print")
    print_model_parameters([Image_f_model], "Image Model", print_or_log="print")
    print_model_parameters([depth_model], "Relative Depth Model", print_or_log="print")
    print_model_parameters([Scalemap_model], "TR2M Model", print_or_log="print")

    print(f"== Text Model Initialized : {args.text_encoder}")
    print(f"== Image Model Initialized : {args.img_encoder}")

    if args.load_ckpt_path is not None:
        checkpoint = torch.load(args.load_ckpt_path)
        state_dict = (
            checkpoint["state_dict"]
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint
            else checkpoint
        )
        Scalemap_model.load_state_dict(state_dict)
        print(f"== Loaded ScaleMap weights from {args.load_ckpt_path}")

    if args.visualize_results:
        vis_root = args.load_ckpt_path + '_vis'
        vis_save_path_nyu   = os.path.join(vis_root, 'nyu')
        vis_save_path_kitti = os.path.join(vis_root, 'kitti')
        vis_save_path_void  = os.path.join(vis_root, 'void')
        vis_save_path_c3vd  = os.path.join(vis_root, 'c3vd')
        for p in (vis_save_path_nyu, vis_save_path_kitti, vis_save_path_void, vis_save_path_c3vd):
            os.makedirs(p, exist_ok=True)
    else:
        vis_save_path_nyu = vis_save_path_kitti = vis_save_path_void = vis_save_path_c3vd = None

    print("Evaluating")
    Scalemap_model.eval()

    with torch.no_grad():
        if dataloader_eval_nyu is not None:
            change_to_nyu(args)
            eval(Scalemap_model, depth_model, CLIP_model, Image_f_model,
                 dataloader_eval_nyu, vis_save_path_nyu, post_process=False, dataset="nyu")

        if dataloader_eval_kitti is not None:
            change_to_kitti(args)
            eval(Scalemap_model, depth_model, CLIP_model, Image_f_model,
                 dataloader_eval_kitti, vis_save_path_kitti, post_process=False, dataset="kitti")

        if dataloader_eval_void is not None:
            change_to_void(args)
            eval(Scalemap_model, depth_model, CLIP_model, Image_f_model,
                 dataloader_eval_void, vis_save_path_void, post_process=False, dataset="void")

        if dataloader_eval_c3vd is not None:
            change_to_c3vd(args)
            eval(Scalemap_model, depth_model, CLIP_model, Image_f_model,
                 dataloader_eval_c3vd, vis_save_path_c3vd, post_process=False, dataset="c3vd")


if __name__ == '__main__':
    main(args)
