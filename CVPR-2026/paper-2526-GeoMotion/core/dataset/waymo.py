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
from .kubric import parse_tapir_track_info, load_target_tracks

class Waymo_dataset(Dataset):
    def __init__(self, data_dir, split, transform=None,
                 factor=1, depth_type="depths",
                 track_method="gt_point_tracks", load_dino=True, dino_later=False):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.data_dir = data_dir
        self.training = split == "train"
        self.factor = factor
        self.depth_type = depth_type
        self.track_method = track_method
        self.load_dino = load_dino
        self.dino_later = dino_later
        
        self.sample_list = [
            item for item in sorted(os.listdir(self.data_dir))
            if os.path.isdir(os.path.join(self.data_dir, item)) and item.split('_')[1] in {'0', '1', '2'}
        ]

        img_dirs_list, mask_dirs_list, track_dirs_list = [], [], []    

        for seq_name in tqdm(self.sample_list, desc="loading data"):
            # Load image list.
            img_dir = osp.join(self.data_dir,seq_name, "images")
            img_dirs_list.append(img_dir)
            mask_dirs_list.append(img_dir.replace("images", "dynamic_masks"))
            track_dirs_list.append(img_dir.replace("images", f'{self.track_method}'))
            
        self.img_dirs_list = img_dirs_list
        self.mask_dirs_list = mask_dirs_list
        self.track_dirs_list = track_dirs_list
        
        # get image size, should under 1024x1024
        img_dir = self.img_dirs_list[0]
        img_paths = sorted(glob(osp.join(img_dir, "*.png"))) + sorted(
            glob(osp.join(img_dir, "*.jpg")) + sorted(glob(osp.join(img_dir, "*.jpeg")))
        )

        frame_names = [osp.splitext(osp.basename(p))[0] for p in img_paths]
        num_frames = len(img_paths)

        img = torch.from_numpy(np.array([iio.imread(img_paths[0])])).squeeze().permute(2, 0, 1) # [1, 512, 512, 3]
        self.image_size = [img.shape[1],img.shape[2]]
        self.step = 4

        self.q_ts = list(range(8, 17, self.step))
        # self.q_ts = [0]
        tracks_2d = load_target_tracks(frame_names[self.q_ts[0]], self.track_dirs_list[0], num_frames, frame_names)
        num = tracks_2d.shape[0]
        # max_num = num * len(self.q_ts)
        max_num = num # 2400
        random_num = [256, 512, 768, 1024, 1536, 2048, 4096, 5000, 6000]

        random_num = [x for x in random_num if x <= max_num and x >= 2048]

        self.random_num = sorted(random_num)
        
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
            depth_image_normalized = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min())
            depth_tensor = torch.from_numpy(depth_image_normalized)
            depth_list.append(depth_tensor)
        depths = torch.stack(depth_list, dim=0)
                        
        # Load 2D tracks prompt. [2, N, L]
        frame_names = [osp.splitext(osp.basename(p))[0] for p in depth_paths]
        num_frames = len(frame_names)
        
        if self.track_method == 'bootstapir':
            tracks_dir = self.track_dirs_list[idx]
            total_num = random.choice(self.random_num)
            q_ts = [x for x in self.q_ts if x <= num_frames]
            q_t = random.choice(q_ts)
            tracks_2d = load_target_tracks(frame_names[q_t], tracks_dir, num_frames, frame_names)
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
            visib_value, confi_value = visib_value[selected_indices], confi_value[selected_indices]

            rows_all_false = torch.all((~visibles), dim=1)
            track_2d = track_2d[~rows_all_false,:,:]
            visibles = visibles[~rows_all_false,:]
            confidences = confidences[~rows_all_false,:]
            visib_value, confi_value = visib_value[~rows_all_false,:], confi_value[~rows_all_false,:]
            
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
        
        # Load DINO feature
        dino_list = []
        if self.load_dino:
            dino_dir = img_dir.replace("images", "dinos")
            dino_paths = sorted(glob(osp.join(dino_dir, "*.npy")))
            features = np.load(dino_paths[q_t])
            dino_qt = torch.from_numpy(features).unsqueeze(0)
                
        sample = {'depths': depths, 
                'track_2d': track_2d,  'dynamic_mask': dynamic_masks,
                'video_dir': img_dir, 'visible_mask': visible_mask,
                'confidences': confidences, 'q_t': q_t, 
                'visib_value': visib_value, 'confi_value': confi_value,
                'dinos': dino_qt}
        sample['case_name'] = f'{osp.basename(osp.dirname(img_dir))}'
        return sample