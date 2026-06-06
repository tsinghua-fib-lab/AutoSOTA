import os
import numpy as np
import torch
from torch.utils.data import Dataset

import os.path as osp
from glob import glob
import imageio as iio
from tqdm import tqdm
import logging
# from utils import masked_median_blur, SceneNormDict
from PIL import Image
import random
import torch.nn.functional as F
import h5py
from torchvision.io import read_image

def parse_tapir_track_info(occlusions, expected_dist, thre = 0.5):
    """
    return:
        valid_visible: mask of visible & confident points
        valid_invisible: mask of invisible & confident points
        confidence: clamped confidence scores (all < 0.5 -> 0)
    """
    visiblility = 1 - F.sigmoid(occlusions)
    confidence = 1 - F.sigmoid(expected_dist)
    valid_visible = visiblility * confidence > thre
    valid_invisible = (1 - visiblility) * confidence > thre
    # set all confidence < 0.5 to 0
    confidence = confidence * (valid_visible | valid_invisible).float()
    return valid_visible, valid_invisible, confidence, visiblility, confidence

def load_target_tracks(
    q_name: str,  tracks_dir: str, num_frames, frame_names, dim: int = 1
):
    """
    tracks are 2d, occs and uncertainties
    :param dim (int), default 1: dimension to stack the time axis
    return (N, T, 4) if dim=1, (T, N, 4) if dim=0
    """
    # q_name = self.frame_names[query_index]
    target_indices = range(num_frames)
    all_tracks = []
    for ti in target_indices:
        t_name = frame_names[ti]
        path = f"{tracks_dir}/{q_name}_{t_name}.npy"
        tracks = np.load(path).astype(np.float32)
        all_tracks.append(tracks)
    return torch.from_numpy(np.stack(all_tracks, axis=dim))

def load_kubric_cameras(camera_path):
    """
    kubric returns cameras in Z back convention
    and also optionally applies a random crop to the images
    Converts the cameras to Z front and account for the possible crop
    :returns Ks_a (T, 3, 3) intrinsic matrices and Ts_aw (T, 4, 4) camera poses
    """
    camera_data = np.load(camera_path, allow_pickle=True).item()
    # unscaled intrinsics with z back
    Ks_u = camera_data["intrinsics"][0]  # (T, 3, 3)
    # extrinsics with z back
    Ts_cw = np.linalg.inv(camera_data["pose"][0])  # (T, 4, 4)

    # scale and account for crop in intrinsics
    input_size = camera_data["input_size"][0, ::-1].astype(float)  # (2,) (W, H)
    # (xmax, ymax, xmin, ymin)
    crop_window = np.flip(camera_data["crop_window"][0].astype(float), axis=-1)  # (4,)
    s = input_size / (crop_window[:2] - crop_window[2:])
    Ks_c = Ks_u.copy()  # (T, 3, 3)
    Ks_c[:, :2, :] = (s * input_size)[None, :, None] * Ks_u[:, :2, :]
    Ks_c[:, :2, 2] += s[None] * crop_window[None, 2:]

    # rotate 180 degrees around x axis
    T_ca = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    Ks_a = np.einsum("tij,jk->tik", Ks_c, T_ca[:3, :3])
    Ts_aw = np.einsum("ij,tjk->tik", np.linalg.inv(T_ca), Ts_cw)
    return {"Ks": Ks_a, "w2cs": Ts_aw}


def find_traj_label(traj, mask, gts, q_t):
    """
    Args:
        traj (tensor): [1,2,N,L]
        mask (tensor): [1,1,N,L]
        gts (tensor): [L,H,W]
        q_t (int): 1

    Returns:
        _type_: _description_
    """
    
    # traj: [N, L, 2], mask: [N, L, 1]
    # gts: [L, H, W]
    traj = traj.squeeze(0).permute(1,2,0)
    mask = mask.squeeze(0).permute(1,2,0)
    gts = gts.squeeze(0)
    
    N, L = traj.shape[:2]
    H, W = gts.shape[1:]
    label_cls = np.zeros(N)
    for i in range(N):
        if mask[i, q_t]:
            # If mask at time q_t for trajectory i is true, skip processing
            continue

        x, y = traj[i, q_t].squeeze()
        if x >= W or y >= H or x < 0 or y < 0 or not torch.isfinite(x) or not torch.isfinite(y):
            # If coordinates are out of bounds or not finite, skip this point
            continue

        # Convert coordinates to integers
        x = int(torch.round(x).item())
        y = int(torch.round(y).item())

        # Retrieve label at the specific ground truth position and time
        label = gts[q_t, y, x]
        if label > 0:
            label_cls[i] = 1
            
    return label_cls

class Kubric_dataset(Dataset):
    def __init__(self, data_dir, split, transform=None,
                 factor=1, depth_type="depths",
                 track_method="gt_point_tracks",load_dino=True,dino_later=False):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.data_dir = osp.join(data_dir,split)
        self.training = split == "train"
        self.factor = factor
        self.depth_type = depth_type
        self.track_method = track_method
        self.load_dino = load_dino
        self.dino_later = dino_later
        
        img_dir_root = osp.join(self.data_dir, "images")
        self.sample_list = sorted(os.listdir(img_dir_root))
        img_dirs_list, mask_dirs_list, label_dirs_list, track_dirs_list = [], [], [], []        

        for seq_name in tqdm(sorted(os.listdir(img_dir_root)), desc="loading data"):
            # Load image list.
            img_dir = osp.join(img_dir_root,seq_name)
            img_dirs_list.append(img_dir)
            mask_dirs_list.append(img_dir.replace("images", "masks"))
            label_dirs_list.append(img_dir.replace("images", "labels"))
            track_dirs_list.append(img_dir.replace("images", "bootstapir_dense"))
            
        self.img_dirs_list = img_dirs_list
        self.mask_dirs_list = mask_dirs_list
        self.label_dirs_list = label_dirs_list
        self.track_dirs_list = track_dirs_list
        
        # get image size, should under 1024x1024
        img_dir = self.img_dirs_list[0]
        img_paths = sorted(glob(osp.join(img_dir, "*.png"))) + sorted(
            glob(osp.join(img_dir, "*.jpg")) + sorted(glob(osp.join(img_dir, "*.jpeg")))
        )
        self.frame_names = [osp.splitext(osp.basename(p))[0] for p in img_paths]
        self.num_frames = len(img_paths)

        img = torch.from_numpy(np.array([iio.imread(img_paths[0])])).squeeze().permute(2, 0, 1) # [1, 512, 512, 3]
        self.image_size = img.shape[1]
        self.step = 8
        self.q_ts = list(range(0, self.num_frames, self.step))
        if track_method == "bootstapir":
            tracks_2d = load_target_tracks(self.frame_names[self.q_ts[0]], self.track_dirs_list[0], self.num_frames, self.frame_names)
            num = tracks_2d.shape[0]
        else:
            num = 256
        # max_num = num * len(self.q_ts)
        random_num = [512, 1024, 2048, 3000, 4096, 5000]
        self.random_num = [x for x in random_num if x <= num]
                
        print("data list loading done !")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        
        # Load images dir
        img_dir = self.img_dirs_list[idx]

        # Load dynamic masks. need normilize [0,1]
        mask_dir = self.mask_dirs_list[idx]
        mask_paths = sorted(glob(osp.join(mask_dir, "*.png"))) + sorted(
            glob(osp.join(mask_dir, "*.jpg")) + sorted(glob(osp.join(mask_dir, "*.jpeg")))
        )
        mask_list = []
        for path in mask_paths:
            mask_tensor = read_image(path) / 255.0
            mask_tensor = mask_tensor.squeeze(0)
            # mask = np.array(Image.open(path).convert('L')) / 255
            # mask_tensor = torch.from_numpy(mask)
            mask_list.append(mask_tensor)
        dynamic_masks = torch.stack(mask_list, dim=0)
        
        # Load depths. [24,256,256]
        depth_dir = img_dir.replace("images", self.depth_type)
        depth_paths = sorted(glob(osp.join(depth_dir, "*.png"))) + sorted(
            glob(osp.join(depth_dir, "*.jpg")) + sorted(glob(osp.join(depth_dir, "*.jpeg")))
        )
        depth_list = []
        for path in depth_paths:
            depth_img = np.array(Image.open(path).convert('L') )
            depth_tensor = torch.from_numpy(depth_img)
            depth_tensor = (depth_tensor - depth_tensor.min()) / (depth_tensor.max() - depth_tensor.min())
            depth_list.append(depth_tensor)
        depths = torch.stack(depth_list, dim=0)
                        
        # Load 2D tracks prompt. [2, N, L]
        if self.track_method == 'bootstapir':
            tracks_dir = self.track_dirs_list[idx]
            total_num = random.choice(self.random_num)
            q_t = random.choice(self.q_ts)
            # q_t = 0
            # for q_t in self.q_ts:
            tracks_2d = load_target_tracks(self.frame_names[q_t], tracks_dir, self.num_frames, self.frame_names)
            track_2d, occs, dists = (
                tracks_2d[..., :2],
                tracks_2d[..., 2],
                tracks_2d[..., 3],
            )
            visibles, invisibles, confidences, visib_value, confi_value = parse_tapir_track_info(occs, dists)

            pts_num = track_2d.shape[0]
            indices = torch.randperm(pts_num)
            selected_indices = indices[:total_num]
            
            track_2d = track_2d[selected_indices]
            visibles = visibles[selected_indices]
            confidences = confidences[selected_indices]
            visib_value = visib_value[selected_indices]
            confi_value = confi_value[selected_indices]

            rows_all_false = torch.all(~visibles, dim=1)
            valid_rows = ~rows_all_false
            track_2d = track_2d[valid_rows, :, :]
            visibles = visibles[valid_rows, :]
            confidences = confidences[valid_rows, :]
            visib_value = visib_value[valid_rows, :]
            confi_value = confi_value[valid_rows, :]

            cols_all_false = torch.all(~visibles.permute(1,0), dim=1)
            for t in range(cols_all_false.size(0)):
                if cols_all_false[t]:
                    max_v, max_conf_index = torch.max(confidences[:, t], dim=0)
                    print(f'get invisible time and confi: {t} and {max_v}')
                    
                    visibles[max_conf_index, t] = True

            track_2d = track_2d.permute(2,0,1)
            visible_mask = visibles        
        elif self.track_method == 'gt_point_tracks':
            track_dir = self.track_dirs_list[idx]
            track_path = osp.join(track_dir,"gt_tracks.npy")
            track_2d = torch.from_numpy(np.load(track_path))
            track_2d = track_2d.permute(2,0,1)
        elif self.track_method == 'cotracker':
            track_dir = self.track_dirs_list[idx]
            track_path = osp.join(track_dir,"pred_tracks.npy")
            track_2d = torch.from_numpy(np.load(track_path))
            track_2d = track_2d.permute(2,1,0)
            
            visible_path = osp.join(track_dir,"pred_visibility.npy")
            visible_mask = torch.from_numpy(np.load(visible_path)).permute(1,0)
            confi_value, visib_value, confidences = 1, 1, 1
            q_t = 0
            
        # Load DINO feature
        if self.load_dino:
            dino_dir = img_dir.replace("images", "dinos")
            dino_paths = sorted(glob(osp.join(dino_dir, "*.h5")))
            # with h5py.File(dino_paths[q_t], 'r') as hf:
            #     features = hf['dinos'][:]
            features = np.load(dino_paths[q_t], mmap_mode="r").squeeze()
            dino_qt = torch.from_numpy(features).unsqueeze(0)
            

        sample = {'depths': depths, 
                'track_2d': track_2d,  'dynamic_mask': dynamic_masks,
                'video_dir': img_dir, 'visible_mask': visible_mask,
                'confidences': confidences, 'q_t': q_t, 
                'visib_value': visib_value, 'confi_value': confi_value,
                }
        sample['case_name'] = f'{osp.basename(img_dir)}'
        if self.load_dino:
            sample['dinos'] = dino_qt

        return sample