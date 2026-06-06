import os
import numpy as np
import torch
from torch.utils.data import Dataset

import os.path as osp
from glob import glob
import imageio as iio
from tqdm import tqdm
import logging
import math
# from utils import masked_median_blur, SceneNormDict
from PIL import Image, ImageOps, ImageFilter, ImageStat
import random
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import h5py
from torchvision.io import read_image
# from .kubric import parse_tapir_track_info, load_target_tracks
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as transforms

class TriViewAugmentor:
    """
    Tri-view Data Augmentor for Motion Segmentation.
    """
    
    def __init__(self, enable_augmentation=True, crop_strategy='smart'):
        self.enable_augmentation = enable_augmentation
        self.crop_strategy = crop_strategy

    def _get_edge_mean(self, img):
        w, h = img.size
        if w < 10 or h < 10:
            m = ImageStat.Stat(img).mean[:3]
            return tuple(int(x) for x in m)
        top = img.crop((0, 0, w, 5))
        bottom = img.crop((0, h-5, w, h))
        left = img.crop((0, 0, 5, h))
        right = img.crop((w-5, 0, w, h))
        stats = [ImageStat.Stat(x) for x in [top, bottom, left, right]]
        r = sum(s.mean[0] for s in stats) / 4
        g = sum(s.mean[1] for s in stats) / 4
        b = sum(s.mean[2] for s in stats) / 4
        return (int(r), int(g), int(b))

    def _apply_exposure_drift(self, img, t, total_frames, magnitude=0.08):
        cycle = total_frames * 0.8
        factor = 1.0 + magnitude * math.sin(2 * math.pi * t / cycle)
        return TF.adjust_brightness(img, factor)

    # def _generate_smooth_trajectory(self, num_frames, max_translate=15, max_scale=0.03):
    #     t = np.linspace(0, np.pi, num_frames)
    #     dx_dir = random.choice([-1, 1])
    #     dy_dir = random.choice([-1, 1])
    #     shift_x = (np.sin(t - np.pi/2) + 1) / 2 * max_translate * dx_dir
    #     shift_y = (np.cos(t) - 1) / 2 * (max_translate * 0.5) * dy_dir
    #     scale_base = 1.0 + (np.sin(t * 2) * max_scale)
    #     return [(shift_x[i], shift_y[i], scale_base[i]) for i in range(num_frames)]
    
    def _generate_smooth_trajectory(self, num_frames, max_translate=15, max_scale=0.03, min_step=1):
        # 目标总位移用整数表示，后面不用再 round
        total_x = int(round(max_translate))
        total_y = int(round(max_translate * 0.5))

        # 如果总位移太小，无法保证每帧至少 1 像素，就把 min_step 自动降下来
        need = (num_frames - 1) * min_step
        if total_x < need:
            min_step_x = max(0, total_x // max(1, (num_frames - 1)))
        else:
            min_step_x = min_step

        need_y = (num_frames - 1) * min_step
        if total_y < need_y:
            min_step_y = max(0, total_y // max(1, (num_frames - 1)))
        else:
            min_step_y = min_step

        def make_steps(total, min_s):
            if num_frames <= 1:
                return np.array([0], dtype=np.int32)

            if total <= 0:
                return np.zeros((num_frames,), dtype=np.int32)

            w = np.hanning(num_frames - 1) + 1e-3
            w = w / w.sum()

            steps = np.maximum(np.round(w * total).astype(np.int32), min_s)

            # 把 steps 的和调成 total，同时不低于 min_s
            diff = int(steps.sum() - total)
            if diff > 0:
                # 减掉多出来的
                order = np.argsort(-steps)  # 从大到小
                k = 0
                while diff > 0 and k < len(order) * 10:
                    i = order[k % len(order)]
                    if steps[i] > min_s:
                        steps[i] -= 1
                        diff -= 1
                    k += 1
            elif diff < 0:
                # 补上缺少的
                diff = -diff
                order = np.argsort(-w)  # 给权重大的位置加
                k = 0
                while diff > 0:
                    i = order[k % len(order)]
                    steps[i] += 1
                    diff -= 1
                    k += 1

            pos = np.concatenate([[0], np.cumsum(steps)])
            return pos.astype(np.int32)

        dx = make_steps(total_x, min_step_x)
        dy = make_steps(total_y, min_step_y)

        dx_dir = random.choice([-1, 1])
        dy_dir = random.choice([-1, 1])
        dx = (dx * dx_dir).tolist()
        dy = (dy * dy_dir).tolist()

        t = np.linspace(0, np.pi, num_frames)
        scale = (1.0 + (np.sin(t * 2) * max_scale)).tolist()

        return [(dx[i], dy[i], scale[i]) for i in range(num_frames)]

    def _apply_augmentation(self, images, masks, mode='standard', spatial_params=None):
        if not self.enable_augmentation:
            return images, masks, {}

        w, h = images[0].size

        # ----------------------------------------------------------------
        # 1. 空间参数计算 (Union BBox Smart Crop + Flip)
        # ----------------------------------------------------------------
        if spatial_params is None:
            do_flip = random.random() < 0.5
            
            crop_x, crop_y = 0, 0
            crop_w, crop_h = w, h
            
            if self.crop_strategy != 'none':
                min_dim = min(w, h)
                scale = random.uniform(0.8, 1.0) if self.crop_strategy == 'random' else 1.0
                crop_size = int(min_dim * scale)
                max_x, max_y = max(0, w - crop_size), max(0, h - crop_size)

                if self.crop_strategy == 'smart':
                    # 先对 mask 做镜像，保证 bbox 坐标与后续 (mirror -> crop) 一致
                    masks_for_bbox = [ImageOps.mirror(m) for m in masks] if do_flip else masks
                    bboxes = [m.getbbox() for m in masks_for_bbox if m.getbbox() is not None]
                    if bboxes:
                        x1 = min(b[0] for b in bboxes); y1 = min(b[1] for b in bboxes)
                        x2 = max(b[2] for b in bboxes); y2 = max(b[3] for b in bboxes)
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        crop_x = max(0, min(max_x, center_x - crop_size // 2))
                        crop_y = max(0, min(max_y, center_y - crop_size // 2))
                    else:
                        crop_x, crop_y = random.randint(0, max_x), random.randint(0, max_y)
                elif self.crop_strategy == 'center':
                    crop_x, crop_y = (w - crop_size) // 2, (h - crop_size) // 2
                else: 
                    crop_x, crop_y = random.randint(0, max_x), random.randint(0, max_y)
                
                crop_w, crop_h = crop_size, crop_size

            spatial_params = {
                'do_flip': do_flip, 
                'crop': (crop_x, crop_y, crop_w, crop_h)
            }

        do_flip = spatial_params['do_flip']
        cx, cy, cw, ch = spatial_params['crop']

        # ----------------------------------------------------------------
        # 2. 光度参数 (Clip-level Shared)
        # ----------------------------------------------------------------
        # [修改点] Standard 模式也采用 Clip-level，防止时序闪烁
        photo_params = None
        
        if mode == 'standard':
            # 弱增强
            photo_params = {
                'brightness': random.uniform(0.9, 1.1),
                'contrast': random.uniform(0.9, 1.1),
                'do_aug': random.random() < 0.2  # 20% 概率触发整个序列的微调
            }
        elif mode == 'semantic':
            # 强增强
            photo_params = {
                'brightness': random.uniform(0.5, 1.5),
                'contrast': random.uniform(0.5, 1.5),
                'saturation': random.uniform(0.0, 2.0),
                'hue': random.uniform(-0.1, 0.1),
                'blur': random.random() < 0.4,
                'blur_radius': random.uniform(1, 3),
                'solarize': random.random() < 0.2,
                'do_aug': True
            }

        # ----------------------------------------------------------------
        # 3. 视图构造 (Stop Mode)
        # ----------------------------------------------------------------
        processing_images = images
        processing_masks = masks

        if mode == 'stop':
            num_frames = len(images)
            ref_idx = num_frames // 2
            ref_img = images[ref_idx].copy()
            ref_mask = masks[ref_idx].copy()
            
            fill_color = self._get_edge_mean(ref_img)
            
            # -------------------------------------------------------------------------
            # [修改 1] 基础幅度大幅提升
            # 不要写死 15，改为 Crop 宽度的 10% (例如 518 * 0.1 ≈ 52 pixels)
            # 这样动起来肉眼清晰可见
            # -------------------------------------------------------------------------
            base_translate = int(cw * 0.1)  # 之前是 15
            max_translate = base_translate
            
            # -------------------------------------------------------------------------
            # [修改 3] 增加缩放 (Zoom) 幅度
            # max_scale 从 0.03 (3%) 提升到 0.08 (8%)
            # 配合平移，视觉上会有明显的"手持运镜"感
            # -------------------------------------------------------------------------
            traj = self._generate_smooth_trajectory(
                num_frames, 
                max_translate=max_translate, 
                max_scale=0.08 
            )
            
            new_images = []
            new_masks = []
            
            for i in range(num_frames):
                dx, dy, scale = traj[i]
                dx_i, dy_i = int(round(dx)), int(round(dy))
                
                # A. 图像变换 (Ref Img 保持未 Flip 状态，由后续 Loop 统一 Flip)
                img_t = TF.affine(
                    ref_img, angle=0, translate=(dx_i, dy_i), scale=float(scale), 
                    shear=0, interpolation=TF.InterpolationMode.BILINEAR,
                    fill=fill_color 
                )
                
                # B. 曝光漂移
                img_t = self._apply_exposure_drift(img_t, i, num_frames, magnitude=0.08)
                
                # C. ROI Mask 同步变换
                mask_t = TF.affine(
                    ref_mask, angle=0, translate=(dx_i, dy_i), scale=float(scale),
                    shear=0, interpolation=TF.InterpolationMode.NEAREST,
                    fill=0 
                )
                
                new_images.append(img_t)
                new_masks.append(mask_t)
            
            processing_images = new_images
            processing_masks = new_masks

        # ----------------------------------------------------------------
        # 4. 最终应用 (Crop & Flip & Color)
        # ----------------------------------------------------------------
        aug_images = []
        aug_masks = []

        for img, mask in zip(processing_images, processing_masks):
            # A. Photometric (全帧一致)
            if photo_params and photo_params['do_aug']:
                img = TF.adjust_brightness(img, photo_params['brightness'])
                img = TF.adjust_contrast(img, photo_params['contrast'])
                
                if mode == 'semantic':
                    img = TF.adjust_saturation(img, photo_params['saturation'])
                    img = TF.adjust_hue(img, photo_params['hue'])
                    if photo_params['blur']:
                        img = img.filter(ImageFilter.GaussianBlur(photo_params['blur_radius']))
                    if photo_params['solarize']:
                        img = ImageOps.solarize(img, threshold=128)
            
            # B. Geometric (所有视图必须一致)
            if do_flip:
                img = ImageOps.mirror(img)
                mask = ImageOps.mirror(mask)
            
            if self.crop_strategy != 'none':
                img = img.crop((cx, cy, cx+cw, cy+ch))
                mask = mask.crop((cx, cy, cx+cw, cy+ch))

            aug_images.append(img)
            aug_masks.append(mask)

        return aug_images, aug_masks, spatial_params


class MotionSegmentationDatasetGotmoving(Dataset):
    
    def __init__(self, data_dir, split, transform=None, 
                 img_size=518, sequence_length=8, sample_stride=1,
                 enable_augmentation=True, crop_strategy='center',
                 enable_triview=False, motion_labels_file=None):
        self.split = split
        self.data_dir = data_dir
        self.training = split == "train"
        self.img_size = img_size
        self.sequence_length = sequence_length
        self.sample_stride = sample_stride
        self.enable_augmentation = enable_augmentation and self.training
        self.crop_strategy = crop_strategy
        self.enable_triview = enable_triview and self.training
        self.motion_labels = None
        
        # [修改点] 运动标签加载与 Sanity Check
        if motion_labels_file and os.path.exists(motion_labels_file):
            with open(motion_labels_file, 'r') as f:
                data = json.load(f)
                self.motion_labels = data.get('sequences', {})
            
            print(f"[Dataset] Loaded motion labels: {len(self.motion_labels)} entries.")
            # 打印前 3 个 key 供检查
            sample_keys = list(self.motion_labels.keys())[:3]
            print(f"[Dataset] Sample keys in JSON: {sample_keys}")
        
        if transform is None:
            self.normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        else:
            self.normalize = transform

        self.augmentor = TriViewAugmentor(
            enable_augmentation=self.enable_augmentation,
            crop_strategy=self.crop_strategy
        )

        if os.path.isfile(self.data_dir) and self.data_dir.endswith(".txt"):
            all_sequences = self._load_sequences_from_txt(self.data_dir)
        else:
            all_sequences = self._find_all_sequences(self.data_dir)

        self.img_dirs_list = []
        self.mask_dirs_list = []
        self.seq_names_list = []
        valid_samples = []
        
        # [修改点] 严格的数据校验
        print(f"[Dataset] Scanning sequences for split '{split}'...")
        for seq in tqdm(all_sequences):
            img_dir = seq["img_dir"]
            mask_dir = seq["mask_dir"]
            seq_name = seq["name"]

            img_paths = self._get_image_paths(img_dir)
            if not img_paths: continue

            # 检查实际存在的配对数量，防止 getitem 挂掉
            pair_count = 0
            for img_path in img_paths:
                name = os.path.splitext(os.path.basename(img_path))[0]
                if os.path.exists(os.path.join(mask_dir, name + ".npz")):
                    pair_count += 1
            
            if pair_count >= self.sequence_length:
                self.img_dirs_list.append(img_dir)
                self.mask_dirs_list.append(mask_dir)
                self.seq_names_list.append(seq_name)
                valid_samples.append(seq_name)

        self.sample_list = valid_samples
        print(f"[Dataset] Found {len(self.sample_list)} valid sequences.")
        if len(self.sample_list) > 0:
            print(f"[Dataset] Sample seq name from dataset: {self.sample_list[0]}")
            # 这里的打印可以让你对比 JSON keys 和 Dataset keys

    def _get_image_paths(self, img_dir):
        extensions = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
        img_paths = []
        for ext in extensions:
            img_paths.extend(glob(osp.join(img_dir, ext)))
        return sorted(img_paths)

    def _load_mask_npz(self, mask_path):
        """
        加载 .npz mask，生成 Instance ID map。
        Background = 0, Object 1 = 1, Object 2 = 2, ...
        """
        data = np.load(mask_path)
        keys = list(data.keys())
        
        # 提取整数键并排序
        int_keys = []
        for k in keys:
            try:
                int_keys.append(int(k))
            except (ValueError, TypeError):
                pass
        
        if len(int_keys) > 0:
            int_keys.sort()
            masks = []
            for k in int_keys:
                obj_mask = data[str(k)]
                if len(obj_mask.shape) == 3:
                    obj_mask = obj_mask[0]
                masks.append(obj_mask)
            
            # --- 修改开始 ---
            # 初始化全 0 (背景)
            instance_mask = np.zeros_like(masks[0], dtype=np.uint8)
            
            for i, m in enumerate(masks):
                # ID 从 1 开始 (i+1)
                # m > 0 的地方赋值为 i+1
                instance_mask[m > 0] = (i + 1)
            
            return instance_mask
        
    # def _load_mask_npz(self, mask_path):
    #     """
    #     加载 .npz 格式的 mask，并合并所有物体
        
    #     npz 文件可能包含多个物体，键为 0, 1, 2, ... 或 'mask', 'arr_0' 等
    #     将所有物体的 mask 合并为一个二值 mask
    #     """
    #     data = np.load(mask_path)
        
    #     # 处理多物体情况：键为数字 0, 1, 2, ...
    #     keys = list(data.keys())
        
    #     # 尝试将键转换为整数并排序
    #     int_keys = []
    #     for k in keys:
    #         try:
    #             int_keys.append(int(k))
    #         except (ValueError, TypeError):
    #             pass
        
    #     if len(int_keys) > 0:
    #         # 有多个物体，合并它们
    #         int_keys.sort()
    #         masks = []
    #         for k in int_keys:
    #             obj_mask = data[str(k)]
    #             if len(obj_mask.shape) == 3:
    #                 obj_mask = obj_mask[0]  # [1, H, W] -> [H, W]
    #             masks.append(obj_mask)
            
    #         # 合并所有物体：取并集
    #         union_mask = np.zeros_like(masks[0], dtype=np.uint8)
    #         for m in masks:
    #             union_mask = np.logical_or(union_mask, m > 0).astype(np.uint8)
            
    #         return union_mask
    
    def _find_all_sequences(self, root):
        seqs = []
        mask_dirs = glob(os.path.join(root, "**", "masks"), recursive=True)
        for mask_dir in mask_dirs:
            parent = os.path.dirname(mask_dir)
            seq_name = os.path.basename(parent)
            seqs.append({"img_dir": parent, "mask_dir": mask_dir, "name": seq_name})
        return seqs
        
    def _load_sequences_from_txt(self, list_path):
        seqs = []
        with open(list_path, "r") as f:
            roots = [line.strip() for line in f if line.strip()]
        for root in roots:
            seqs.append({
                "img_dir": root, 
                "mask_dir": os.path.join(root, "masks"), 
                "name": os.path.basename(root)
            })
        return seqs

    def _smart_resize(self, images, masks, target_size):
        resized_images = []
        resized_masks = []
        for img, mask in zip(images, masks):
            img_resized = img.resize((target_size, target_size), Image.LANCZOS)
            mask_resized = mask.resize((target_size, target_size), Image.NEAREST)
            resized_images.append(img_resized)
            resized_masks.append(mask_resized)
        return resized_images, resized_masks

    def _load_images_and_masks(self, pairs, start_idx, end_idx, sample_stride):
        # 1. Load Raw PIL
        pil_images = []
        pil_masks = []
        for i in range(start_idx, end_idx, sample_stride):
            curr_idx = min(i, len(pairs) - 1)
            img_path, mask_path = pairs[curr_idx]
            
            img = Image.open(img_path).convert('RGB')
            mask_np = self._load_mask_npz(mask_path) # 这里返回的是 0,1,2...
            
            # --- 修改：直接转 Image，不要 * 255 ---
            # mode='L' 存 8-bit (0-255)，足够存 255 个物体 ID
            mask = Image.fromarray(mask_np.astype(np.uint8), mode='L')
            
            pil_images.append(img)
            pil_masks.append(mask)


        # 2. Tri-view Logic
        triview_data = None
        seq_dir = os.path.dirname(os.path.dirname(pairs[0][1]))
        seq_name = os.path.basename(seq_dir)
        
        apply_triview = True
        # if self.enable_triview:
        #     if self.motion_labels:
        #         info = self.motion_labels.get(seq_name, {})
        #         apply_triview = info.get('is_high_motion', False)

        if apply_triview:
            # A. Base View (Geometry + Weak Clip-level Photo)
            aug_orig, aug_orig_masks, spatial_params = self.augmentor._apply_augmentation(
                pil_images, pil_masks, mode='standard', spatial_params=None
            )
            # B. Semantic View (Same Geometry + Strong Clip-level Photo)
            aug_sem, aug_sem_masks, _ = self.augmentor._apply_augmentation(
                pil_images, pil_masks, mode='semantic', spatial_params=spatial_params
            )
            # C. Stop View (Same Geometry + Static Physics)
            aug_stop, aug_stop_roi, _ = self.augmentor._apply_augmentation(
                pil_images, pil_masks, mode='stop', spatial_params=spatial_params
            )
            
            triview_data = {
                'flag': False,
                'stop': (aug_stop, aug_stop_roi)
            }
            x = random.random()

            # 与 0.5、0.75、1 进行比较判断
            if x < 0.5:
                pil_images = aug_orig
                pil_masks = aug_orig_masks
            elif x < 0.85:
                pil_images = aug_sem
                pil_masks = aug_sem_masks
            elif x < 1:
                pil_images = aug_stop
                pil_masks = aug_stop_roi
                triview_data['flag'] = True

            # pil_images = aug_orig
            # pil_masks = aug_orig_masks
            
        elif self.enable_augmentation:
            # Standard only
            pil_images, pil_masks, _ = self.augmentor._apply_augmentation(
                pil_images, pil_masks, mode='standard'
            )

        # 3. Resize & ToTensor
        pil_images, pil_masks = self._smart_resize(pil_images, pil_masks, self.img_size)
        
        images_tensor = []
        masks_tensor = []
        for img, mask in zip(pil_images, pil_masks):
            # Image: Normalize to [0, 1] then Gaussian Norm
            t_img = self.normalize(transforms.ToTensor()(img))
            
            # --- 修改：Mask 转 Tensor 逻辑 ---
            # 1. 转为 numpy 或者 PILToTensor (保留 0,1,2 数值)
            # 2. 不要用 transforms.ToTensor() (因为它会除以 255)
            # 3. 转为 torch.from_numpy().long()
            
            m_np = np.array(mask, dtype=np.int64) # (H, W)
            m_np_binary = (m_np > 0).astype(np.int64)  # 布尔值转int64，True→1，False→0
            # t_mask = torch.from_numpy(m_np_binary).long() # 必须是 Long 类型用于 ID
            t_mask = torch.from_numpy(m_np_binary) # 必须是 Long 类型用于 ID
            
            images_tensor.append(t_img)
            masks_tensor.append(t_mask)
            
        return torch.stack(images_tensor), torch.stack(masks_tensor), triview_data

    def _process_aux_view(self, pil_imgs, pil_masks):
        imgs_resized, masks_resized = self._smart_resize(pil_imgs, pil_masks, self.img_size)
        imgs_list = []
        masks_list = []
        for img, mask in zip(imgs_resized, masks_resized):
            t_img = self.normalize(transforms.ToTensor()(img))
            t_mask = (transforms.ToTensor()(mask).squeeze(0) > 0).float()
            imgs_list.append(t_img)
            masks_list.append(t_mask)
        return torch.stack(imgs_list, dim=0), torch.stack(masks_list, dim=0)

    def __getitem__(self, idx):
        # try:
        img_dir = self.img_dirs_list[idx]
        mask_dir = self.mask_dirs_list[idx]
        seq_name = self.seq_names_list[idx]
        
        img_paths = self._get_image_paths(img_dir)
        
        pairs = []
        for img_path in img_paths:
            name = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(mask_dir, name + ".npz")
            if os.path.exists(mask_path):
                pairs.append((img_path, mask_path))
        
        if len(pairs) < self.sequence_length:
            return None
        
        multiplier = random.choice([0.5, 1, 2])  # 随机选一个乘数
        sample_stride = int(self.sample_stride * multiplier)      # 计算最终步长
        if self.training:
            max_start = max(0, len(pairs) - self.sequence_length * sample_stride)
            start_idx = random.randint(0, max_start)
        else:
            max_start = max(0, len(pairs) - self.sequence_length * sample_stride)
            start_idx = max_start // 2
        end_idx = start_idx + self.sequence_length * sample_stride
        
        images, masks, triview_data = self._load_images_and_masks(pairs, start_idx, end_idx, sample_stride)

        if triview_data['flag'] == True:
            masks = torch.zeros_like(masks)
            # print('----')
        sample = {
            'images': images,
            'motion_masks': masks,
            'sequence_name': seq_name,
            'is_triview': False
        }
            
            # if triview_data is not None:
            #     sem_imgs, sem_masks = triview_data['semantic']
            #     sem_img_t, sem_mask_t = self._process_aux_view(sem_imgs, sem_masks)
                
            #     stop_imgs, stop_roi = triview_data['stop']
            #     stop_img_t, stop_roi_t = self._process_aux_view(stop_imgs, stop_roi)
                
            #     sample['semantic_images'] = sem_img_t
            #     sample['semantic_masks'] = sem_mask_t
                
            #     sample['stop_images'] = stop_img_t
            #     sample['stop_roi'] = stop_roi_t  # [重要] Ranking Loss 用这个作为 ROI
            #     sample['stop_label'] = torch.zeros_like(masks) # Debug 或 Metric 用
                
            #     sample['is_triview'] = True
                
        return sample
            
        # except Exception as e:
        #     print(f"Error loading sample {idx}: {e}")
        #     return None

    def __len__(self):
        return len(self.sample_list)

def collate_fn_filter_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None
    return torch.utils.data.dataloader.default_collate(batch)


def create_motion_seg_dataloader(cfg, split='train'):
    """
    Factory function to create motion segmentation dataloader with augmentation support
    """
    # Dataset parameters
    dataset_params = {
        'data_dir': cfg.data_dir,
        'split': split,
        'img_size': getattr(cfg, 'img_size', 518),
        'sequence_length': getattr(cfg, 'sequence_length', 8),
        'sample_stride': getattr(cfg, 'sample_stride', 1),
        'enable_augmentation': getattr(cfg, 'enable_augmentation', split == 'train'),
        'crop_strategy': getattr(cfg, 'crop_strategy', 'smart' if split == 'train' else 'center'),
    }
    
    # Create dataset
    dataset = MotionSegmentationDataset(**dataset_params)
    
    # Dataloader parameters
    dataloader_params = {
        'batch_size': getattr(cfg, 'batch_size', 2),
        'shuffle': split == 'train',
        'num_workers': getattr(cfg, 'num_workers', 4),
        'pin_memory': True,
        'drop_last': split == 'train',
    }
    
    # Create dataloader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, **dataloader_params)
    
    return dataloader


import os
import random
import json
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
# ImageNet normalize 的反归一化
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def unnormalize(img_chw: torch.Tensor) -> torch.Tensor:
    x = img_chw.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN
    return x.clamp(0, 1)

def to_hwc_uint8(img_chw_01: torch.Tensor) -> np.ndarray:
    x = (img_chw_01.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return x

def overlay_mask(rgb_uint8: np.ndarray, mask_hw: np.ndarray, alpha=0.45) -> np.ndarray:
    out = rgb_uint8.astype(np.float32).copy()
    m = mask_hw.astype(bool)
    if m.any():
        color = np.array([255, 0, 0], dtype=np.float32)  # 红色 overlay
        out[m] = out[m] * (1 - alpha) + color * alpha
    return out.clip(0, 255).astype(np.uint8)

def save_grid(views, save_path: Path, max_frames=8, dpi=150):
    """
    views: list of (view_name, imgs_tensor[S,3,H,W], masks_tensor[S,H,W])
    """
    S = views[0][1].shape[0]
    S_show = min(S, max_frames)
    nrows = len(views)
    ncols = S_show

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0*ncols, 3.2*nrows))
    if nrows == 1:
        axes = np.expand_dims(axes, 0)
    if ncols == 1:
        axes = np.expand_dims(axes, 1)

    for r, (name, imgs, masks) in enumerate(views):
        for c in range(S_show):
            img = unnormalize(imgs[c])
            rgb = to_hwc_uint8(img)
            m = masks[c].detach().cpu().numpy() > 0.5
            out = overlay_mask(rgb, m, alpha=0.45)

            ax = axes[r, c]
            ax.imshow(out)
            ax.axis("off")
            if c == 0:
                ax.set_title(name, fontsize=12)

    plt.tight_layout()
    fig.savefig(str(save_path), dpi=dpi)
    plt.close(fig)

def save_frames(view_name, imgs, masks, out_dir: Path, max_frames=8):
    """
    每帧保存:
      - raw.png
      - mask.png
      - overlay.png
    """
    S = imgs.shape[0]
    S_show = min(S, max_frames)
    vdir = out_dir / view_name
    (vdir / "raw").mkdir(parents=True, exist_ok=True)
    (vdir / "mask").mkdir(parents=True, exist_ok=True)
    (vdir / "overlay").mkdir(parents=True, exist_ok=True)

    for t in range(S_show):
        img = unnormalize(imgs[t])
        rgb = to_hwc_uint8(img)
        m = (masks[t].detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
        ov = overlay_mask(rgb, m > 0, alpha=0.45)

        plt.imsave(str(vdir / "raw" / f"{t:02d}.png"), rgb)
        plt.imsave(str(vdir / "mask" / f"{t:02d}.png"), m, cmap="gray")
        plt.imsave(str(vdir / "overlay" / f"{t:02d}.png"), ov)

def get_random_valid_sample(dataset, tries=100):
    for _ in range(tries):
        idx = random.randint(0, len(dataset) - 1)
        s = dataset[idx]
        if s is not None:
            return idx, s
    return None, None

def save_one_sample(dataset, idx=None, out_root="debug_aug", max_frames=8):
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if idx is None:
        idx, sample = get_random_valid_sample(dataset)
        if sample is None:
            print("没抽到有效样本，检查数据或 collate 过滤逻辑")
            return
    else:
        sample = dataset[idx]
        if sample is None:
            print(f"idx={idx} 是 None，换一个 idx")
            return

    seq = sample.get("sequence_name", "unknown_seq")
    is_triview = bool(sample.get("is_triview", False))

    sample_dir = out_root / f"{seq}_idx{idx}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # 准备需要保存的视图
    views = []
    views.append(("main", sample["images"], sample["motion_masks"]))

    if is_triview:
        views.append(("semantic", sample["semantic_images"], sample["semantic_masks"]))
        # stop 这里叠加的是 stop_roi，用于检查 ROI 是否跟轨迹走
        views.append(("stop_roi", sample["stop_images"], sample["stop_roi"]))

    # 1) 网格图
    grid_path = sample_dir / "grid_overlay.png"
    save_grid(views, grid_path, max_frames=max_frames)

    # 2) 每帧单独保存
    for (name, imgs, masks) in views:
        save_frames(name, imgs, masks, sample_dir, max_frames=max_frames)

    # 3) 额外把一些元信息存一下
    meta = {
        "idx": int(idx),
        "sequence_name": seq,
        "is_triview": is_triview,
        "num_frames": int(sample["images"].shape[0]),
        "saved_frames": int(min(sample["images"].shape[0], max_frames)),
        "files": {
            "grid": str(grid_path)
        }
    }
    with open(sample_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved to: {sample_dir}")
    print(f"Grid: {grid_path}")

def save_random_samples(dataset, n=5, out_root="debug_aug", max_frames=8):
    saved = 0
    tried = 0
    while saved < n and tried < n * 50:
        tried += 1
        idx, s = get_random_valid_sample(dataset)
        if s is None:
            continue
        save_one_sample(dataset, idx=idx, out_root=out_root, max_frames=max_frames)
        saved += 1

    print(f"Done. saved={saved}, tried={tried}")

# Example configuration class for motion segmentation
class MotionSegConfig:
    def __init__(self):
        # Data settings
        self.data_dir = "/data2"
        self.img_size = 518
        self.sequence_length = 8  # Number of frames per sample
        self.sample_stride = 1    # Frame sampling stride
        
        # Training settings
        self.batch_size = 2       # Small batch size due to video sequences
        self.num_workers = 4
        self.learning_rate = 1e-4
        self.num_epochs = 100
        
        # Model settings  
        self.vggt_model_path = "/data2/hxk/model.pt"
        
        # Logging
        self.log_dir = "logs/motion_segmentation"
        self.print_freq = 50


def test_motion_seg_dataset():
    """
    Test function to verify dataset loading
    """
    cfg = MotionSegConfig()
    cfg.data_dir = "/data1"  # Update this path
    
    # try:
    # Create dataset
    # dataset = MotionSegmentationDataset(
    #     data_dir=cfg.data_dir,
    #     split='train',
    #     img_size=518,
    #     sequence_length=4,
    #     sample_stride=1
    # )

    # 2. 启用 Tri-view（使用预计算的运动标签）
    dataset = MotionSegmentationDatasetGotmoving(
        data_dir=cfg.data_dir,
        split="train",
        img_size=518,
        sequence_length=12,
        sample_stride=2,
        enable_augmentation=True,
        crop_strategy='smart',
        enable_triview=True,
        motion_labels_file="motion_labels.json"
    )
    import pdb;pdb.set_trace()
    dataset[5]['motion_masks']
    save_random_samples(dataset, n=100, out_root="debug_aug", max_frames=8)  # 随机存 10 个
    
    print(f"Dataset created successfully with {len(dataset)} samples")
    
    # Test loading one sample
    # if len(dataset) > 0:
    #     sample = dataset[5]
    #     print(sample)
    #     print(f"Sample keys: {sample.keys()}")
    #     print(f"Images shape: {sample['images'].shape}")
    #     print(f"Motion masks shape: {sample['motion_masks'].shape}")
    #     print(f"Sequence name: {sample['sequence_name']}")
    #     print(f"Images range: [{sample['images'].min():.3f}, {sample['images'].max():.3f}]")
    #     print(f"Masks range: [{sample['motion_masks'].min():.3f}, {sample['motion_masks'].max():.3f}]")
    #     print("Dataset test passed!")
    # else:
    #     print("Warning: Dataset is empty")
            
    # except Exception as e:
    # print(f"Dataset test failed: {e}")


if __name__ == "__main__":
    test_motion_seg_dataset()