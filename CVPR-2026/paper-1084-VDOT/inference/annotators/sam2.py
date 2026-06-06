# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import numpy as np
import torch
from scipy import ndimage

from .utils import convert_to_numpy, read_video_one_frame, single_mask_to_rle, single_rle_to_mask, single_mask_to_xyxy


class SAM2ImageAnnotator:
    def __init__(self, cfg, device=None):
        self.task_type = cfg.get('TASK_TYPE', 'input_box')
        self.return_mask = cfg.get('RETURN_MASK', False)
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except:
            import warnings
            warnings.warn("please pip install sam2 package, or you can refer to models/VACE-Annotators/sam2/SAM_2-1.0-cp310-cp310-linux_x86_64.whl")
        config_path = cfg['CONFIG_PATH']
        local_config_path = os.path.join(*config_path.rsplit('/')[-3:])
        if not os.path.exists(local_config_path):  # TODO
            os.makedirs(os.path.dirname(local_config_path), exist_ok=True)
            shutil.copy(config_path, local_config_path)
        pretrained_model = cfg['PRETRAINED_MODEL']
        sam2_model = build_sam2(local_config_path, pretrained_model)
        self.predictor = SAM2ImagePredictor(sam2_model)
        self.predictor.fill_hole_area = 0

    def forward(self,
                image,
                input_box=None,
                mask=None,
                task_type=None,
                return_mask=None):
        task_type = task_type if task_type is not None else self.task_type
        return_mask = return_mask if return_mask is not None else self.return_mask
        mask = convert_to_numpy(mask) if mask is not None else None

        if task_type == 'mask_point':
            if len(mask.shape) == 3:
                scribble = mask.transpose(2, 1, 0)[0]
            else:
                scribble = mask.transpose(1, 0)   # (H, W) -> (W, H)
            labeled_array, num_features = ndimage.label(scribble >= 255)
            centers = ndimage.center_of_mass(scribble, labeled_array,
                                             range(1, num_features + 1))
            point_coords = np.array(centers)
            point_labels = np.array([1] * len(centers))
            sample = {
                'point_coords': point_coords,
                'point_labels': point_labels
            }
        elif task_type == 'mask_box':
            if len(mask.shape) == 3:
                scribble = mask.transpose(2, 1, 0)[0]
            else:
                scribble = mask.transpose(1, 0)  # (H, W) -> (W, H)
            labeled_array, num_features = ndimage.label(scribble >= 255)
            centers = ndimage.center_of_mass(scribble, labeled_array,
                                             range(1, num_features + 1))
            centers = np.array(centers)
            # (x1, y1, x2, y2)
            x_min = centers[:, 0].min()
            x_max = centers[:, 0].max()
            y_min = centers[:, 1].min()
            y_max = centers[:, 1].max()
            bbox = np.array([x_min, y_min, x_max, y_max])
            sample = {'box': bbox}
        elif task_type == 'input_box':
            if isinstance(input_box, list):
                input_box = np.array(input_box)
            sample = {'box': input_box}
        elif task_type == 'mask':
            sample = {'mask_input': mask[None, :, :]}
        else:
            raise NotImplementedError

        self.predictor.set_image(image)
        masks, scores, logits = self.predictor.predict(
            multimask_output=False,
            **sample
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]

        if return_mask:
            return masks[0]
        else:
            ret_data = {
                "masks": masks,
                "scores": scores,
                "logits": logits
            }
            return ret_data


class SAM2VideoAnnotator:
    def __init__(self, cfg, device=None):
        self.task_type = cfg.get('TASK_TYPE', 'input_box')
        try:
            from sam2.build_sam import build_sam2_video_predictor
        except:
            import warnings
            warnings.warn("please pip install sam2 package, or you can refer to models/VACE-Annotators/sam2/SAM_2-1.0-cp310-cp310-linux_x86_64.whl")
        config_path = cfg['CONFIG_PATH']
        local_config_path = os.path.join(*config_path.rsplit('/')[-3:])
        if not os.path.exists(local_config_path):  # TODO
            os.makedirs(os.path.dirname(local_config_path), exist_ok=True)
            shutil.copy(config_path, local_config_path)
        pretrained_model = cfg['PRETRAINED_MODEL']
        self.video_predictor = build_sam2_video_predictor(local_config_path, pretrained_model)
        self.video_predictor.fill_hole_area = 0

    def forward(self,
                video,
                input_box=None,
                mask=None,
                task_type=None):
        task_type = task_type if task_type is not None else self.task_type

        mask = convert_to_numpy(mask) if mask is not None else None

        if task_type == 'mask_point':
            if len(mask.shape) == 3:
                scribble = mask.transpose(2, 1, 0)[0]
            else:
                scribble = mask.transpose(1, 0)   # (H, W) -> (W, H)
            labeled_array, num_features = ndimage.label(scribble >= 255)
            centers = ndimage.center_of_mass(scribble, labeled_array,
                                             range(1, num_features + 1))
            point_coords = np.array(centers)
            point_labels = np.array([1] * len(centers))
            sample = {
                'points': point_coords,
                'labels': point_labels
            }
        elif task_type == 'mask_box':
            if len(mask.shape) == 3:
                scribble = mask.transpose(2, 1, 0)[0]
            else:
                scribble = mask.transpose(1, 0)  # (H, W) -> (W, H)
            labeled_array, num_features = ndimage.label(scribble >= 255)
            centers = ndimage.center_of_mass(scribble, labeled_array,
                                             range(1, num_features + 1))
            centers = np.array(centers)
            # (x1, y1, x2, y2)
            x_min = centers[:, 0].min()
            x_max = centers[:, 0].max()
            y_min = centers[:, 1].min()
            y_max = centers[:, 1].max()
            bbox = np.array([x_min, y_min, x_max, y_max])
            sample = {'box': bbox}
        elif task_type == 'input_box':
            if isinstance(input_box, list):
                input_box = np.array(input_box)
            sample = {'box': input_box}
        elif task_type == 'mask':
            sample = {'mask': mask}
        else:
            raise NotImplementedError

        ann_frame_idx = 0
        object_id = 0
        with (torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16)):

            inference_state = self.video_predictor.init_state(video_path=video)
            if task_type in ['mask_point', 'mask_box', 'input_box']:
                _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=object_id,
                    **sample
                )
            elif task_type in ['mask']:
                _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=object_id,
                    **sample
                )
            else:
                raise NotImplementedError

            video_segments = {}  # video_segments contains the per-frame segmentation results
            for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(inference_state):
                frame_segments = {}
                for i, out_obj_id in enumerate(out_obj_ids):
                    mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze(0)
                    frame_segments[out_obj_id] = {
                        "mask": single_mask_to_rle(mask),
                        "mask_area": int(mask.sum()),
                        "mask_box": single_mask_to_xyxy(mask),
                    }
                video_segments[out_frame_idx] = frame_segments

        ret_data = {
            "annotations": video_segments
        }
        return ret_data


class SAM2SalientVideoAnnotator:
    def __init__(self, cfg, device=None):
        from .salient import SalientAnnotator
        from .sam2 import SAM2VideoAnnotator
        self.salient_model = SalientAnnotator(cfg['SALIENT'], device=device)
        self.sam2_model = SAM2VideoAnnotator(cfg['SAM2'], device=device)

    def forward(self, video, image=None):
        if image is None:
            image = read_video_one_frame(video)
        else:
            image = convert_to_numpy(image)
        salient_res = self.salient_model.forward(image)
        sam2_res = self.sam2_model.forward(video=video, mask=salient_res, task_type='mask')
        return sam2_res


class SAM2GDINOVideoAnnotator:
    def __init__(self, cfg, device=None):
        from .gdino import GDINOAnnotator
        from .sam2 import SAM2VideoAnnotator
        self.gdino_model = GDINOAnnotator(cfg['GDINO'], device=device)
        self.sam2_model = SAM2VideoAnnotator(cfg['SAM2'], device=device)

    def forward(self, video, image=None, classes=None, caption=None):
        if image is None:
            image = read_video_one_frame(video)
        else:
            image = convert_to_numpy(image)
        if classes is not None:
            gdino_res = self.gdino_model.forward(image, classes=classes)
        else:
            gdino_res = self.gdino_model.forward(image, caption=caption)
        if 'boxes' in gdino_res and len(gdino_res['boxes']) > 0:
            bboxes = gdino_res['boxes'][0]
        else:
            raise ValueError("Unable to find the corresponding boxes")
        sam2_res = self.sam2_model.forward(video=video, input_box=bboxes, task_type='input_box')
        return sam2_res