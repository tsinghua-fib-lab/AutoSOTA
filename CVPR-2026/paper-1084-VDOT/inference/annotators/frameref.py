# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import random
import numpy as np
from .utils import align_frames
import random

class FrameRefExtractAnnotator:
    para_dict = {}

    def __init__(self, cfg, device=None):
        # first / last / firstlast / random
        self.ref_cfg = cfg.get('REF_CFG', [{"mode": "first", "proba": 0.25},
                                           {"mode": "last", "proba": 0.25},
                                           {"mode": "firstlast", "proba": 0.25},
                                           {"mode": "random", "proba": 0.25}])
        self.ref_num = cfg.get('REF_NUM', 1)
        self.ref_color = cfg.get('REF_COLOR', 128)
        self.return_dict = cfg.get('RETURN_DICT', True)
        self.return_mask = cfg.get('RETURN_MASK', True)


    def forward(self, frames, ref_cfg=None, ref_num=None, return_mask=None, return_dict=None):
        return_mask = return_mask if return_mask is not None else self.return_mask
        return_dict = return_dict if return_dict is not None else self.return_dict
        ref_cfg = ref_cfg if ref_cfg is not None else self.ref_cfg
        ref_cfg = [ref_cfg] if not isinstance(ref_cfg, list) else ref_cfg
        probas = [item['proba'] if 'proba' in item else 1.0 / len(ref_cfg) for item in ref_cfg]
        sel_ref_cfg = random.choices(ref_cfg, weights=probas, k=1)[0]
        mode = sel_ref_cfg['mode'] if 'mode' in sel_ref_cfg else 'original'
        ref_num = int(ref_num) if ref_num is not None else self.ref_num
        ref_num = random.randint(1, 30)

        frame_num = len(frames)
        frame_num = min(frame_num, 81)

        frame_num_range = list(range(frame_num))
        if mode == "first":
            sel_idx = frame_num_range[:ref_num]
        elif mode == "last":
            sel_idx = frame_num_range[-ref_num:]
        elif mode == "firstlast":
            sel_idx = frame_num_range[:ref_num] + frame_num_range[-ref_num:]
        elif mode == "random":
            sel_idx = random.sample(frame_num_range, ref_num)
        else:
            raise NotImplementedError

        out_frames, out_masks = [], []
        for i in range(frame_num):
            if i in sel_idx:
                out_frame = frames[i]
                out_mask = np.zeros_like(frames[i][:, :, 0])
            else:
                out_frame = np.ones_like(frames[i]) * self.ref_color
                out_mask = np.ones_like(frames[i][:, :, 0]) * 255
            out_frames.append(out_frame)
            out_masks.append(out_mask)

        if return_dict:
            ret_data = {"frames": out_frames}
            if return_mask:
                ret_data['masks'] = out_masks
            return ret_data
        else:
            if return_mask:
                return out_frames, out_masks
            else:
                return out_frames



class FrameRefExpandAnnotator:
    para_dict = {}

    def __init__(self, cfg, device=None):
        # first / last / firstlast
        self.ref_color = cfg.get('REF_COLOR', 127.5)
        self.return_mask = cfg.get('RETURN_MASK', True)
        self.return_dict = cfg.get('RETURN_DICT', True)
        self.mode = cfg.get('MODE', "firstframe")
        assert self.mode in ["firstframe", "lastframe", "firstlastframe", "firstclip", "lastclip", "firstlastclip", "all"]

    def forward(self, image=None, image_2=None, frames=None, frames_2=None, mode=None, expand_num=None, return_mask=None, return_dict=None):
        mode = mode if mode is not None else self.mode
        return_mask = return_mask if return_mask is not None else self.return_mask
        return_dict = return_dict if return_dict is not None else self.return_dict

        if 'frame' in mode:
            frames = [image] if image is not None and not isinstance(frames, list) else image
            frames_2 = [image_2] if image_2 is not None and not isinstance(image_2, list) else image_2

        expand_frames = [np.ones_like(frames[0]) * self.ref_color] * expand_num
        expand_masks = [np.ones_like(frames[0][:, :, 0]) * 255] * expand_num
        source_frames = frames
        source_masks = [np.zeros_like(frames[0][:, :, 0])] * len(frames)

        if mode in ["firstframe", "firstclip"]:
            out_frames = source_frames + expand_frames
            out_masks = source_masks + expand_masks
        elif mode in ["lastframe", "lastclip"]:
            out_frames = expand_frames + source_frames
            out_masks = expand_masks + source_masks
        elif mode in ["firstlastframe", "firstlastclip"]:
            source_frames_2 = [align_frames(source_frames[0], f2) for f2 in frames_2]
            source_masks_2 = [np.zeros_like(source_frames_2[0][:, :, 0])] * len(frames_2)
            out_frames = source_frames + expand_frames + source_frames_2
            out_masks = source_masks + expand_masks + source_masks_2
        else:
            raise NotImplementedError

        if return_dict:
            ret_data = {"frames": out_frames}
            if return_mask:
                ret_data['masks'] = out_masks
            return ret_data
        else:
            if return_mask:
                return out_frames, out_masks
            else:
                return out_frames
