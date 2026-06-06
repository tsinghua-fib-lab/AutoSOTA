# ParticleSfM
# Copyright (C) 2022  ByteDance Inc.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import numpy as np
import cv2
import torch
# from cvbase.optflow.visualize import flow2rgb


def read_flow_png(name):
    # [x,y]
    flow_img = cv2.imread(name, -1)[:,:,:2]
    flow = (flow_img - 32000.0) / 100.0
    return flow

TAG_FLOAT = 202021.25

def read_flo(file):
    assert type(file) is str, "file is not str %r" % str(file)
    assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
    assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
    f = open(file,'rb')
    flo_number = np.fromfile(f, np.float32, count=1)[0]
    assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
    w = np.fromfile(f, np.int32, count=1)
    h = np.fromfile(f, np.int32, count=1)
    data = np.fromfile(f, np.float32, count=2*w[0]*h[0])
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))
    f.close()
    return flow

def resize_flow(flow, hw, to_rgb=True):
    h, w, _ = np.shape(flow)
    flow = cv2.resize(flow, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)
    flow[:, :, 0] = flow[:, :, 0] * hw[1] / w
    flow[:, :, 1] = flow[:, :, 1] * hw[0] / h
    if to_rgb: flow = np.clip((flow2rgb(flow) - 0.5) * 2, -1., 1.)
    return flow

def read_seg_gt_png(name):
    gt_img = cv2.imread(name)[:,:,0:1]
    gt_img[gt_img > 0] = 1
    return gt_img

def read_normalize_depth(name):
    depth = cv2.imread(name, -1) / 65535.0
    return depth

def read_point_traj(name, mask_name, label_name, max_num):
    pt = np.load(name)['arr_0']
    mask = np.load(mask_name)['arr_0']
    label = np.load(label_name)
    if pt.shape[0] > max_num:
        indices = np.random.choice(np.arange(pt.shape[0]), max_num, replace=False)
        pt = pt[indices]
        mask = mask[indices]
        label = label[indices]
    return pt, mask, label

def resize_point_traj(pt, raw_hw, target_hw):
    # resize the point trajectory coordinates if input images are resized
    ratio_h = float(raw_hw[0]) / float(target_hw[0])
    ratio_w = float(raw_hw[1]) / float(target_hw[1])
    pt_resize = np.copy(pt)
    pt_resize[:,:,0] /= ratio_w
    pt_resize[:,:,1] /= ratio_h
    return pt_resize

def normalize_point_traj(pt, hw):
    # resize the point trajectory coordinates if input images are resized
    pt_nor = np.copy(pt)
    pt_nor[:,:,0] /= hw[1]
    pt_nor[:,:,1] /= hw[0]
    pt_nor = np.clip(pt_nor, 0.0, 1.0)
    return pt_nor

def normalize_point_traj_torch(pt, hw):
    # Normalize the point trajectory coordinates for PyTorch tensor input
    pt_nor = pt.clone()
    # Assuming pt has shape [batch, 2, num_points, num_trajectories]
    # Normalize x coordinates
    pt_nor[:, 0, :, :] /= hw[1]
    # Normalize y coordinates
    pt_nor[:, 1, :, :] /= hw[0]
    # Clip values to be within the range [0.0, 1.0]
    pt_nor = pt_nor * 2 - 1
    # Optionally clip values to be within the range [-1.0, 1.0], though they should already be within this range
    pt_nor = torch.clamp(pt_nor, -1.0, 1.0)
    return pt_nor

def sample_point_traj_raw(pt_raw, pt, mask, idx, thres, max_num):
    # pt: [N, L, 2], mask: [N, L], label: [N], idx: [l]
    idx = np.array(idx)
    sample_pt_raw, sample_pt, sample_mask = pt_raw[:,idx,:], pt[:,idx,:], mask[:,idx]
    s_idx = np.arange(pt.shape[0])
    valid = (np.sum(1.0 - sample_mask[:,:,0], axis=1) >= thres)
    valid_pt_raw, valid_pt, valid_mask = sample_pt_raw[valid], sample_pt[valid], sample_mask[valid]
    valid_s_idx = s_idx[valid]
    if valid_pt.shape[0] > max_num:
        indices = np.random.choice(np.arange(valid_pt.shape[0]), max_num, replace=False)
        valid_pt_raw = valid_pt_raw[indices]
        valid_pt = valid_pt[indices]
        valid_mask = valid_mask[indices]
        valid_s_idx = valid_s_idx[indices]
    return valid_pt_raw, valid_pt, valid_mask, valid_s_idx


